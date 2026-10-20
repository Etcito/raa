#!/usr/bin/env node
// Browser capture through nomorechat's GPU client (Conventions, "Browser captures").
//
//   node scripts/gpu-capture.mjs <name> [--input <gpu-input.json>] [--milestone mN]
//
// 1. requires NMC_GPU_CLIENT and dist/<name>/sizes.json (S7);
// 2. `node "$NMC_GPU_CLIENT" publish dist/<name>/` -> page_ref;
// 3. `node "$NMC_GPU_CLIENT" render <input> --page-ref REF -o target/gpu-out/<name>/`;
// 4. writes proofs/results/<name>.json (S6) and proofs/captures/<name>.png.
// Any client failure exits non-zero with the client's (redacted) stderr; no result is
// ever written that the client did not produce. Env values are never printed.
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync,
} from 'node:fs';
import { dirname, join, resolve } from 'node:path';

const HARDWARE_DENY = /swiftshader|llvmpipe|lavapipe|software|basic render/i;

function fail(message, code = 1) {
  process.stderr.write(`gpu-capture: ${redact(message)}\n`);
  process.exit(code);
}

// Child diagnostics can echo arbitrary environment values, including short tokens.
// Redact every nonempty value, longest first, even when that obscures ordinary text.
function secretValues() {
  const values = [];
  for (const value of Object.values(process.env)) {
    if (value) values.push(value);
  }
  return values.sort((a, b) => b.length - a.length);
}
const SECRETS = secretValues();

function redact(text) {
  let out = String(text ?? '');
  for (const value of SECRETS) out = out.split(value).join('[REDACTED]');
  return out;
}

function log(message) {
  process.stdout.write(redact(message) + '\n');
}

function parseArgs(argv) {
  const args = { name: null, input: null, milestone: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--input' || a === '--milestone') {
      const v = argv[++i];
      if (!v) fail(`${a} needs a value`);
      args[a.slice(2)] = v;
    } else if (a.startsWith('--')) {
      fail(`unknown option ${a}`);
    } else if (!args.name) {
      args.name = a.replace(/\/+$/, '');
    } else {
      fail(`unexpected argument ${a}`);
    }
  }
  if (!args.name || !/^[A-Za-z0-9_-]+$/.test(args.name)) {
    fail('usage: node scripts/gpu-capture.mjs <name> [--input <gpu-input.json>] [--milestone mN]');
  }
  return args;
}

// Runs the client with node, never through a shell. `timeoutMs` = 0 leaves the
// timing to the client itself.
function runClient(client, clientArgs, timeoutMs) {
  const label = clientArgs[0];
  const res = spawnSync(process.execPath, [client, ...clientArgs], {
    encoding: 'utf8',
    env: process.env,
    maxBuffer: 64 * 1024 * 1024,
    timeout: timeoutMs > 0 ? timeoutMs : undefined,
    killSignal: 'SIGKILL',
  });
  const stdout = res.stdout ?? '';
  const stderr = res.stderr ?? '';
  if (res.error || res.status !== 0) {
    const why = res.error?.code === 'ETIMEDOUT'
      ? `timed out after ${Math.round(timeoutMs / 1000)} s`
      : res.error ? res.error.message : `exited with ${res.status ?? res.signal}`;
    if (stdout.trim()) process.stderr.write(redact(stdout).trimEnd() + '\n');
    if (stderr.trim()) process.stderr.write(redact(stderr).trimEnd() + '\n');
    fail(`GPU client '${label}' ${redact(why)}; no browser result written`);
  }
  if (stderr.trim()) process.stderr.write(redact(stderr).trimEnd() + '\n');
  return stdout;
}

// page_ref from publish stdout: a JSON document (whole stdout or one line) with
// `page_ref`/`pageRef` (top level or under `result`/`data`), else a `page_ref: X` line.
function parsePageRef(stdout) {
  const pick = (obj) => {
    if (!obj || typeof obj !== 'object') return null;
    for (const o of [obj, obj.result, obj.data]) {
      if (!o || typeof o !== 'object') continue;
      const v = o.page_ref ?? o.pageRef;
      if (typeof v === 'string' && v.trim()) return v.trim();
    }
    return null;
  };
  const tryJson = (text) => { try { return pick(JSON.parse(text)); } catch { return null; } };
  const whole = tryJson(stdout.trim());
  if (whole) return whole;
  for (const line of stdout.split(/\r?\n/)) {
    const t = line.trim();
    if (t.startsWith('{')) {
      const v = tryJson(t);
      if (v) return v;
    }
    const m = /^\s*(?:page_ref|pageRef)\s*[:=]\s*["']?([^\s"']+)["']?\s*$/.exec(line);
    if (m) return m[1];
  }
  return null;
}

function listFiles(dir) {
  const out = [];
  const walk = (d) => {
    for (const entry of readdirSync(d, { withFileTypes: true }).sort((a, b) => (a.name < b.name ? -1 : 1))) {
      const p = join(d, entry.name);
      if (entry.isDirectory()) walk(p);
      else if (entry.isFile()) out.push(p);
    }
  };
  if (existsSync(dir)) walk(dir);
  return out;
}

// The op result may be the JSON document itself or wrapped by the client.
function unwrapOpResult(doc) {
  const queue = [doc];
  for (let i = 0; i < queue.length && i < 32; i++) {
    const o = queue[i];
    if (!o || typeof o !== 'object' || Array.isArray(o)) continue;
    if (o.adapter && typeof o.adapter === 'object' && 'nonEmpty' in o) return o;
    for (const k of ['result', 'value', 'data', 'output', 'opResult', 'op_result', 'run']) {
      if (o[k] && typeof o[k] === 'object') queue.push(o[k]);
    }
  }
  return null;
}

// Assumed output layout of `render -o <dir>` (the client's docs/gpu-client.md was not
// reachable when this was written; re-check it when it is): the directory (searched
// recursively, entries sorted by name) holds at least one *.json whose content is the
// value `window.nmcGpu.ops.run(args)` resolved to, either bare or wrapped under
// result/value/data/output, and at least one *.png screenshot taken after run resolved.
// The first JSON whose (unwrapped) object has `adapter` and `nonEmpty` is the op result;
// the first *.png is the capture.
function findOutputs(outDir) {
  const files = listFiles(outDir);
  let op = null;
  let opFile = null;
  for (const f of files.filter((p) => p.endsWith('.json'))) {
    let doc;
    try { doc = JSON.parse(readFileSync(f, 'utf8')); } catch { continue; }
    const found = unwrapOpResult(doc);
    if (found) { op = found; opFile = f; break; }
  }
  const png = files.find((p) => p.toLowerCase().endsWith('.png')) ?? null;
  return { op, opFile, png, files };
}

function isHardware(adapter) {
  if (adapter.isFallbackAdapter !== false) return false;
  return !['vendor', 'architecture', 'device', 'description']
    .some((k) => HARDWARE_DENY.test(String(adapter[k] ?? '')));
}

const sha256 = (path) => createHash('sha256').update(readFileSync(path)).digest('hex');
const numOrNull = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);

function validatePage(op) {
  const invalid = (field) => fail(`page result ${field} is invalid; no browser result written`);
  const dimensions = (v) => Array.isArray(v) && v.length === 2
    && v.every((n) => Number.isInteger(n) && n > 0);
  for (const key of ['vendor', 'architecture', 'device', 'description']) {
    if (typeof op.adapter[key] !== 'string') invalid(`adapter.${key}`);
  }
  if (!dimensions(op.canvas)) invalid('canvas');
  for (const key of ['firstFrameMs', 'cpuMsMedian', 'cpuMsP95']) {
    if (numOrNull(op[key]) === null || op[key] < 0) invalid(key);
  }
  for (const key of ['warmup', 'frames', 'instances', 'drawCalls', 'triangles']) {
    if (!Number.isInteger(op[key]) || op[key] < 0) invalid(key);
  }
  if (op.allocsPerFrameTotal !== null
    && (numOrNull(op.allocsPerFrameTotal) === null || op.allocsPerFrameTotal < 0)) {
    invalid('allocsPerFrameTotal');
  }
  if (op.resize !== null && (!op.resize || !dimensions(op.resize.to)
    || !dimensions(op.resize.back) || typeof op.resize.ok !== 'boolean')) invalid('resize');
}

function clientDocsHint(client) {
  const dir = dirname(resolve(client));
  for (const p of [join(dir, 'gpu-client.md'), join(dir, 'docs', 'gpu-client.md'), join(dir, '..', 'docs', 'gpu-client.md')]) {
    if (existsSync(p)) return p;
  }
  return null;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const { name } = args;

  const client = process.env.NMC_GPU_CLIENT;
  if (!client) {
    process.stderr.write('gpu-capture: NMC_GPU_CLIENT is unset; no browser result written\n');
    process.exit(2);
  }
  if (!existsSync(client)) fail('NMC_GPU_CLIENT does not point to an existing file; no browser result written');

  const distDir = join('dist', name);
  const sizesPath = join(distDir, 'sizes.json');
  if (!existsSync(sizesPath)) fail(`${sizesPath} is missing: run scripts/build-web.sh demos/${name} first`);
  let sizes;
  try { sizes = JSON.parse(readFileSync(sizesPath, 'utf8')); } catch (e) { fail(`${sizesPath} is not valid JSON (${e.message})`); }
  for (const k of ['raw', 'gzip', 'sha256']) {
    if (sizes[k] === undefined) fail(`${sizesPath} lacks "${k}": run scripts/build-web.sh demos/${name} first`);
  }
  if (!Number.isInteger(sizes.raw) || sizes.raw < 0
    || !Number.isInteger(sizes.gzip) || sizes.gzip < 0
    || typeof sizes.sha256 !== 'string' || !/^[a-f0-9]{64}$/i.test(sizes.sha256)) {
    fail(`${sizesPath} has invalid sizes or sha256: run scripts/build-web.sh demos/${name} first`);
  }

  const milestone = args.milestone ?? (/^(m\d+)(?:-|$)/.exec(name) || [])[1];
  if (!milestone || !/^m\d+$/.test(milestone)) fail(`cannot derive the milestone from '${name}'; pass --milestone mN`);

  const input = args.input ?? join('demos', name, 'gpu-input.json');
  if (!existsSync(input)) fail(`render input ${input} is missing`);

  const docs = clientDocsHint(client);
  if (docs) log(`gpu-capture: client docs at ${docs}`);

  const waitS = Number(process.env.NMC_GPU_WAIT_S ?? 60);
  const publishTimeoutMs = Number.isFinite(waitS) && waitS > 0 ? waitS * 1000 : 60000;

  log(`gpu-capture: publish ${distDir}/`);
  const publishOut = runClient(client, ['publish', `${distDir}/`], publishTimeoutMs);
  const pageRef = parsePageRef(publishOut);
  if (!pageRef) {
    process.stderr.write(redact(publishOut).trimEnd() + '\n');
    fail('could not parse page_ref from the publish output; no browser result written');
  }
  log(`gpu-capture: page_ref ${redact(pageRef)}`);

  // Start from an empty output dir so a stale file can never become the result.
  const outDir = join('target', 'gpu-out', name);
  rmSync(outDir, { recursive: true, force: true });
  mkdirSync(outDir, { recursive: true });

  log(`gpu-capture: render ${input} -> ${outDir}/`);
  const renderOut = runClient(client, ['render', input, '--page-ref', pageRef, '-o', `${outDir}/`], 0);
  if (renderOut.trim()) log(redact(renderOut).trimEnd());

  const { op, opFile, png, files } = findOutputs(outDir);
  if (!op) fail(`no op-result JSON (with adapter and nonEmpty) in ${outDir}/ (files: ${files.join(', ') || 'none'}); no browser result written`);
  if (!png) fail(`no screenshot PNG in ${outDir}/ (files: ${files.join(', ') || 'none'}); no browser result written`);
  log(`gpu-capture: op result ${opFile}, screenshot ${png}`);

  const a = op.adapter;
  const adapter = {
    vendor: String(a.vendor ?? ''),
    architecture: String(a.architecture ?? ''),
    device: String(a.device ?? ''),
    description: String(a.description ?? ''),
    isFallbackAdapter: a.isFallbackAdapter === true,
  };
  if (typeof a.isFallbackAdapter !== 'boolean') fail('page result adapter.isFallbackAdapter is not a boolean; no browser result written');
  const hardware = isHardware(adapter);
  if (op.hardware !== hardware) {
    fail(`page reports hardware=${op.hardware} but the Conventions rule gives ${hardware}; no browser result written`);
  }
  if (typeof op.nonEmpty !== 'boolean') fail('page result nonEmpty is not a boolean; no browser result written');
  validatePage(op);

  const capture = `proofs/captures/${name}.png`;
  const resultPath = join('proofs', 'results', `${name}.json`);
  mkdirSync(dirname(capture), { recursive: true });
  copyFileSync(png, capture);

  const result = {
    name,
    milestone,
    capturedAt: new Date().toISOString(),
    adapter,
    hardware,
    canvas: Array.isArray(op.canvas) ? op.canvas : null,
    nonEmpty: op.nonEmpty,
    firstFrameMs: op.firstFrameMs,
    warmup: op.warmup ?? null,
    frames: op.frames ?? null,
    cpuMsMedian: numOrNull(op.cpuMsMedian),
    cpuMsP95: numOrNull(op.cpuMsP95),
    instances: op.instances ?? null,
    drawCalls: op.drawCalls ?? null,
    triangles: op.triangles ?? null,
    allocsPerFrameTotal: numOrNull(op.allocsPerFrameTotal),
    resize: op.resize && typeof op.resize === 'object' ? op.resize : null,
    wasm: { raw: sizes.raw, gzip: sizes.gzip, sha256: sizes.sha256 },
    capture,
    captureSha256: sha256(capture),
  };
  mkdirSync(dirname(resultPath), { recursive: true });
  writeFileSync(resultPath, JSON.stringify(result, null, 2) + '\n');

  log(`gpu-capture: wrote ${resultPath} and ${capture}`);
  log(`gpu-capture: adapter "${adapter.description || adapter.device || adapter.vendor}" (${hardware ? 'hardware' : 'software'}), nonEmpty ${result.nonEmpty}, firstFrameMs ${result.firstFrameMs}`);
}

try { main(); } catch (error) { fail(error.message); }
