#!/usr/bin/env node
// Lookdev captures through nomorechat's GPU client (Conventions "Browser captures", S14).
//
//   node scripts/lookdev-capture.mjs [--page lookdev|sdfm-compare] [--milestone mNN]
//     [--only <regex on scene--variant>] [--judge] [--ref-prefix <p>]
//
// 1. requires NMC_GPU_CLIENT (unset: exit 2 at once; a path that does not exist yet is
//    waited for up to NMC_GPU_WAIT_S seconds, default 60), dist/<page>/scenes.json (S14),
//    dist/<page>/sizes.json (S7) and dist/<page>/nmc.gpu.json declaring the `render` op;
// 2. imports the scene modules named by scenes.json and selects their (scene, variant)
//    pairs, filtered by --milestone and by --only (a regex tested on `<scene>--<variant>`);
// 3. `node "$NMC_GPU_CLIENT" publish dist/<page>/` once -> page_ref (bounded by
//    NMC_GPU_WAIT_S, as in scripts/gpu-capture.mjs);
// 4. per pair, up to NMC_CPUS at a time (1 when unset): writes
//    target/lookdev/<pair>.input.json, runs `render <input> --page-ref REF -o
//    target/gpu-out/lookdev/<pair>/`, copies the PNG to proofs/captures/lookdev/<pair>.png,
//    with --judge on a `judge: true` variant runs the judge command (Conventions) on it,
//    and writes the S14 record proofs/results/lookdev/<pair>.json (judge null otherwise).
// <pair> is `<scene>--<variant>`. No record is written that the client did not produce;
// env values are never printed. A failing pair does not stop the others; the failures
// are listed at the end (exit 1). CPU output is never substituted for a GPU capture.
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync,
  writeFileSync,
} from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const USAGE = 'usage: node scripts/lookdev-capture.mjs [--page lookdev|sdfm-compare] '
  + '[--milestone mNN] [--only <regex on scene--variant>] [--judge] [--ref-prefix <p>]';
const PAGES = { lookdev: 'scripts/build-lookdev.sh', 'sdfm-compare': 'scripts/build-sdfm-compare.sh' };
const WIDTH = 1280; // LOOKDEV_CAPTURE
const HEIGHT = 720;
const JUDGE_MIN_SCORE = 6;
const JUDGE_OUTPUT_MAX = 4000;
const JUDGE_TIMEOUT_MS = 900000;
const HARDWARE_DENY = /swiftshader|llvmpipe|lavapipe|software|basic render/i;
const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
const SCENE_ID = /^m\d{2}-[A-Za-z0-9_-]+$/;
const VARIANT_ID = /^[a-z0-9-]+$/;
const REFERENCE = /^ref-[1-4]$/;

// Child output and page data can echo environment values. Every string that comes from
// outside this script (child output, page data, paths, arguments, error messages) goes
// through redact() before it is printed or stored; the fixed words and numbers this
// script adds (`hardware`, counts) are appended after. Every nonempty env value is
// redacted, longest first: values of 8+ characters wherever they occur, shorter ones
// (flags, counts, words such as `true`) where they stand as a whole word, so `ref-1`,
// `6.0` or `7/10` in a judge verdict stay readable.
const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const WORD = '[^\\s"\'`,;:()\\[\\]{}<>=]';
const SECRETS = [...new Set(Object.values(process.env).filter(Boolean))]
  .sort((a, b) => b.length - a.length);
const SECRET_RE = SECRETS.length === 0 ? null : new RegExp(SECRETS.map((v) => (v.length >= 8
  ? escapeRe(v) : `(?<!${WORD})${escapeRe(v)}(?!${WORD})`)).join('|'), 'g');

function redact(text) {
  const s = String(text ?? '');
  return SECRET_RE ? s.replace(SECRET_RE, '[REDACTED]') : s;
}

const log = (line) => process.stdout.write(`lookdev-capture: ${line}\n`);
const warn = (line) => process.stderr.write(`lookdev-capture: ${line}\n`);

// Callers pass external fragments through redact(); the rest of a message is fixed text.
function fail(message, code = 1) {
  warn(message);
  process.exit(code);
}

function parseArgs(argv) {
  const args = { page: 'lookdev', milestone: null, only: null, judge: false, refPrefix: null };
  const value = (i, flag) => {
    const v = argv[i];
    if (v === undefined || v === '') fail(`${flag} needs a value\n${USAGE}`, 2);
    return v;
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--page') args.page = value(++i, a);
    else if (a === '--milestone') args.milestone = value(++i, a);
    else if (a === '--only') args.only = value(++i, a);
    else if (a === '--ref-prefix') args.refPrefix = value(++i, a);
    else if (a === '--judge') args.judge = true;
    else if (a === '-h' || a === '--help') { process.stdout.write(`${USAGE}\n`); process.exit(0); }
    else fail(`unexpected argument ${redact(a)}\n${USAGE}`, 2);
  }
  if (!Object.hasOwn(PAGES, args.page)) fail(`--page must be one of ${Object.keys(PAGES).join(', ')}\n${USAGE}`, 2);
  if (args.milestone !== null && !/^m\d{2}$/.test(args.milestone)) {
    fail(`--milestone must look like m07 (two digits)\n${USAGE}`, 2);
  }
  if (args.only !== null) {
    try { args.onlyRe = new RegExp(args.only); } catch (e) { fail(`--only is not a valid regex (${redact(e.message)})`, 2); }
  }
  return args;
}

const sleep = (ms) => new Promise((done) => { setTimeout(done, ms); });

function cpus() {
  const n = Math.floor(Number(process.env.NMC_CPUS));
  return n >= 1 ? n : 1;
}

// Runs `worker(item, index)` with at most NMC_CPUS in flight; results in input order.
async function pool(items, worker) {
  const results = new Array(items.length);
  let next = 0;
  const lane = async () => {
    while (next < items.length) {
      const i = next++;
      results[i] = await worker(items[i], i);
    }
  };
  await Promise.all(Array.from({ length: Math.min(cpus(), items.length) }, lane));
  return results;
}

// Runs a command without a shell; `timeoutMs` = 0 leaves the timing to the command.
function runAsync(cmd, cmdArgs, timeoutMs) {
  return new Promise((done) => {
    const res = { code: null, signal: null, stdout: '', stderr: '', timedOut: false, error: null };
    let settled = false;
    let timer = null;
    const finish = () => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      done(res);
    };
    let child;
    try {
      child = spawn(cmd, cmdArgs, { env: process.env, stdio: ['ignore', 'pipe', 'pipe'], shell: false });
    } catch (e) {
      res.error = e;
      finish();
      return;
    }
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (d) => { res.stdout += d; });
    child.stderr.on('data', (d) => { res.stderr += d; });
    if (timeoutMs > 0) {
      timer = setTimeout(() => { res.timedOut = true; child.kill('SIGKILL'); }, timeoutMs);
    }
    child.on('error', (e) => { res.error = e; finish(); });
    child.on('close', (code, signal) => { res.code = code; res.signal = signal; finish(); });
  });
}

const succeeded = (res) => !res.error && !res.timedOut && res.code === 0;

// `limit` names the time limit, which may come from the environment.
function whyFailed(res, limit) {
  if (res.timedOut) return `timed out after ${limit}`;
  if (res.error) return redact(res.error.message);
  return res.code === null ? `was killed by ${res.signal}` : `exited with ${res.code}`;
}

// The redacted tail of a child's output, each line indented under its owner.
function tail(text, lines = 40) {
  const all = redact(text).trimEnd().split(/\r?\n/).filter((l) => l.trim());
  return all.slice(-lines).map((l) => `    ${l}`).join('\n');
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

function readJsonFile(path, hint) {
  if (!existsSync(path)) fail(`${path} is missing: run ${hint} first`);
  try { return JSON.parse(readFileSync(path, 'utf8')); } catch (e) {
    return fail(`${path} is not valid JSON (${redact(e.message)}): run ${hint} again`);
  }
}

// The S7 sizes of the page's wasm, copied into every record.
function readSizes(distDir, hint) {
  const path = join(distDir, 'sizes.json');
  const sizes = readJsonFile(path, hint);
  if (!sizes || !Number.isInteger(sizes.raw) || sizes.raw < 0
    || !Number.isInteger(sizes.gzip) || sizes.gzip < 0
    || typeof sizes.sha256 !== 'string' || !/^[a-f0-9]{64}$/i.test(sizes.sha256)) {
    fail(`${path} lacks valid raw, gzip and sha256 (S7): run ${hint} again`);
  }
  return { raw: sizes.raw, gzip: sizes.gzip, sha256: sizes.sha256 };
}

function checkManifest(distDir, hint) {
  const path = join(distDir, 'nmc.gpu.json');
  const manifest = readJsonFile(path, hint);
  if (!manifest || !Array.isArray(manifest.ops) || !manifest.ops.includes('render')) {
    fail(`${path} does not declare the op "render" (ops must be an array of names)`);
  }
}

// dist/<page>/scenes.json (S14): [{ id, milestone, file }].
function readSceneIndex(distDir, hint) {
  const path = join(distDir, 'scenes.json');
  const index = readJsonFile(path, hint);
  if (!Array.isArray(index)) fail(`${path} is not an array of { id, milestone, file } (S14)`);
  index.forEach((e, i) => {
    if (!e || typeof e.id !== 'string' || typeof e.milestone !== 'string' || typeof e.file !== 'string'
      || !SCENE_ID.test(e.id) || !/^m\d{2}$/.test(e.milestone)) {
      fail(`${path} entry ${i} is not { id: 'mNN-<name>', milestone: 'mNN', file } (S14)`);
    }
  });
  return index;
}

// The scene module of an index entry: `file` relative to dist/<page>/, else to the repo.
function sceneModulePath(distDir, file) {
  for (const p of [join(distDir, file), join(distDir, 'scenes', file), file]) {
    if (existsSync(p)) return resolve(p);
  }
  return null;
}

// Imports the selected scenes and lists their (scene, variant) pairs. A scene or variant
// that breaks S14 becomes a listed problem; the other scenes still run.
async function selectPairs(distDir, index, args) {
  const pairs = [];
  const problems = [];
  for (const entry of index) {
    if (args.milestone !== null && entry.milestone !== args.milestone) continue;
    const scene = entry.id;
    const path = sceneModulePath(distDir, entry.file);
    if (!path) { problems.push(`scene ${scene}: module ${redact(entry.file)} not found under ${distDir}/`); continue; }
    let def;
    try {
      def = (await import(pathToFileURL(path).href)).default;
    } catch (e) {
      problems.push(`scene ${scene}: import failed (${redact(e?.message ?? e)})`);
      continue;
    }
    if (!def || typeof def !== 'object') { problems.push(`scene ${scene}: no default export object`); continue; }
    if (def.id !== scene) { problems.push(`scene ${scene}: the module's id is ${redact(JSON.stringify(def.id))}`); continue; }
    if (def.milestone !== entry.milestone) {
      problems.push(`scene ${scene}: the module's milestone ${redact(JSON.stringify(def.milestone))} differs from scenes.json`);
      continue;
    }
    if (!Array.isArray(def.variants) || def.variants.length === 0) { problems.push(`scene ${scene}: no variants`); continue; }
    const seen = new Set();
    def.variants.forEach((v, i) => {
      const id = v?.id;
      if (typeof id !== 'string' || !VARIANT_ID.test(id)) {
        problems.push(`scene ${scene}: variants[${i}].id must match ${VARIANT_ID}`);
        return;
      }
      if (seen.has(id)) { problems.push(`scene ${scene}: duplicate variant ${id}`); return; }
      seen.add(id);
      const name = `${scene}--${id}`;
      if (args.onlyRe && !args.onlyRe.test(name)) return;
      if (typeof v.description !== 'string' || !v.description.trim()) problems.push(`${name}: no description`);
      else if (!Array.isArray(v.references) || !v.references.every((r) => typeof r === 'string' && REFERENCE.test(r))) {
        problems.push(`${name}: references must be names ref-1..ref-4`);
      } else if (typeof v.judge !== 'boolean') problems.push(`${name}: judge must be a boolean`);
      else pairs.push({ name, scene, milestone: entry.milestone, variant: v });
    });
  }
  return { pairs, problems };
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
    for (const k of ['json', 'result', 'value', 'data', 'output', 'opResult', 'op_result', 'run']) {
      if (o[k] && typeof o[k] === 'object') queue.push(o[k]);
    }
  }
  return null;
}

// `render -o <dir>` leaves result.json (the op's `json`) and the op's files (the page
// returns `capture.png`). The first JSON (result.json first) whose unwrapped object has
// `adapter` and `nonEmpty` is the op result; capture.png, else the first PNG, the capture.
function findOutputs(outDir) {
  const files = listFiles(outDir);
  const jsons = files.filter((p) => p.endsWith('.json'))
    .sort((a, b) => Number(!a.endsWith('result.json')) - Number(!b.endsWith('result.json')));
  let op = null;
  for (const f of jsons) {
    let doc;
    try { doc = JSON.parse(readFileSync(f, 'utf8')); } catch { continue; }
    op = unwrapOpResult(doc);
    if (op) break;
  }
  const pngs = files.filter((p) => p.toLowerCase().endsWith('.png'));
  const png = pngs.find((p) => p.endsWith('capture.png')) ?? pngs[0] ?? null;
  return { op, png, files };
}

// Conventions: hardware = !isFallbackAdapter && no software marker in the adapter names.
function isHardware(adapter) {
  if (adapter.isFallbackAdapter !== false) return false;
  return !['vendor', 'architecture', 'device', 'description']
    .some((k) => HARDWARE_DENY.test(String(adapter[k] ?? '')));
}

const sha256 = (path) => createHash('sha256').update(readFileSync(path)).digest('hex');

// Problems of the page result against S14 (empty = valid); `names` gets the passes.
function checkOpResult(op, pair) {
  const problems = [];
  if (op.scene !== pair.scene || op.variant !== pair.variant.id) {
    problems.push(`page answered for ${redact(`${op.scene}--${op.variant}`)}`);
  }
  if (Array.isArray(op.gpuErrors) && op.gpuErrors.length > 0) {
    problems.push(`page reported GPU errors: ${redact(op.gpuErrors.join(' | '))}`);
  }
  for (const k of ['vendor', 'architecture', 'device', 'description']) {
    if (typeof op.adapter[k] !== 'string') problems.push(`adapter.${k} is not a string`);
  }
  if (typeof op.adapter.isFallbackAdapter !== 'boolean') problems.push('adapter.isFallbackAdapter is not a boolean');
  if (!Array.isArray(op.canvas) || op.canvas[0] !== WIDTH || op.canvas[1] !== HEIGHT || op.canvas.length !== 2) {
    problems.push(`canvas ${redact(JSON.stringify(op.canvas))} is not [${WIDTH}, ${HEIGHT}]`);
  }
  if (typeof op.nonEmpty !== 'boolean') problems.push('nonEmpty is not a boolean');
  for (const k of ['instances', 'triangles']) {
    if (!Number.isInteger(op[k]) || op[k] < 0) problems.push(`${k} is not a count`);
  }
  if (op.firstFrameMs !== null && !(typeof op.firstFrameMs === 'number' && op.firstFrameMs >= 0)) {
    problems.push('firstFrameMs is not a duration');
  }
  if (!Array.isArray(op.passes)) problems.push('passes is not a PassInfo array');
  else if (!op.passes.every((p) => typeof p === 'string' || typeof p?.name === 'string')) {
    problems.push('passes has an entry without a name');
  }
  return problems;
}

// The names of the enabled passes, in the order the page reported them.
const enabledPasses = (passes) => passes
  .filter((p) => typeof p === 'string' || p.enabled !== false)
  .map((p) => (typeof p === 'string' ? p : p.name));

// Width and height of a PNG, or null when the file is no PNG.
function pngSize(path) {
  const b = readFileSync(path);
  if (b.length < 24 || !b.subarray(0, 8).equals(PNG_SIGNATURE) || b.toString('latin1', 12, 16) !== 'IHDR') return null;
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}

// The judge's score: a JSON `score`, else the last `score: N` in its output.
function parseScore(text) {
  for (const chunk of [text.trim(), ...text.split(/\r?\n/).reverse()]) {
    if (!chunk.trim().startsWith('{')) continue;
    try {
      const s = JSON.parse(chunk).score;
      if (typeof s === 'number' && Number.isFinite(s)) return s;
    } catch { /* not JSON */ }
  }
  const all = [...text.matchAll(/(?<![-\w])score\W{0,4}(\d+(?:\.\d+)?)/gi)];
  return all.length ? Number(all[all.length - 1][1]) : null;
}

// Head and tail of the text within JUDGE_OUTPUT_MAX characters.
function clip(text) {
  if (text.length <= JUDGE_OUTPUT_MAX) return text;
  const marker = `\n[... ${text.length} chars, middle omitted ...]\n`;
  const head = 1000;
  return text.slice(0, head) + marker + text.slice(text.length - (JUDGE_OUTPUT_MAX - head - marker.length));
}

// Conventions judge command on the saved capture. The stored output is redacted and
// names the reference prefix and the working directory only symbolically.
async function runJudge(capture, variant, refPrefix) {
  const cmdArgs = [process.env.NMC_JUDGE_CLIENT, capture, '--description', variant.description];
  for (const ref of variant.references) cmdArgs.push('--reference', `${refPrefix}${ref}.png`);
  cmdArgs.push('--min-score', String(JUDGE_MIN_SCORE));
  const res = await runAsync('python3', cmdArgs, JUDGE_TIMEOUT_MS);
  let text = [res.stdout, res.stderr].filter((s) => s.trim()).join('\n').trimEnd();
  if (!succeeded(res) && res.code === null) text += `\njudge ${whyFailed(res, `${JUDGE_TIMEOUT_MS / 1000} s`)}`;
  // Paths first (longest first, a prefix may lie inside the working directory), then the
  // env values, so a redaction marker is never rewritten.
  const paths = [[refPrefix, '<ref-prefix>'], [process.cwd(), '.']]
    .filter(([from]) => from).sort((a, b) => b[0].length - a[0].length);
  for (const [from, to] of paths) text = text.split(from).join(to);
  const output = redact(text);
  return {
    description: variant.description,
    references: [...variant.references],
    minScore: JUDGE_MIN_SCORE,
    exit: res.code === null ? -1 : res.code,
    score: parseScore(`${res.stdout}\n${res.stderr}`),
    output: clip(output),
    at: new Date().toISOString(),
  };
}

function writeJsonAtomic(path, value) {
  const tmp = `${path}.tmp`;
  writeFileSync(tmp, JSON.stringify(value, null, 2) + '\n');
  renameSync(tmp, path);
}

// One (scene, variant): render, check, capture, judge, record. Resolves to
// { name, problems, hardware?, nonEmpty?, judge? }; a record exists only after a render
// whose result and PNG passed the checks.
async function capturePair(pair, ctx) {
  const { name } = pair;
  const problems = [];
  const input = join('target', 'lookdev', `${name}.input.json`);
  const outDir = join('target', 'gpu-out', 'lookdev', name);
  writeFileSync(input, JSON.stringify({
    args: { scene: pair.scene, variant: pair.variant.id, width: WIDTH, height: HEIGHT },
  }, null, 2) + '\n');
  // An empty output dir, so a stale file can never become the result.
  rmSync(outDir, { recursive: true, force: true });
  mkdirSync(outDir, { recursive: true });

  log(`${name}: render ${input} -> ${outDir}/`);
  const res = await runAsync(process.execPath,
    [ctx.client, 'render', input, '--page-ref', ctx.pageRef, '-o', `${outDir}/`], 0);
  if (!succeeded(res)) {
    const out = tail(`${res.stdout}\n${res.stderr}`);
    warn(`${name}: GPU client 'render' ${whyFailed(res, '')}; no record written${out ? `\n${out}` : ''}`);
    return { name, problems: [`render ${whyFailed(res, '')}`] };
  }
  const { op, png, files } = findOutputs(outDir);
  const listed = files.map(redact).join(', ') || 'none';
  if (!op) return { name, problems: [`no op-result JSON with adapter and nonEmpty in ${outDir}/ (files: ${listed})`] };
  if (!png) return { name, problems: [`no PNG in ${outDir}/ (files: ${listed})`] };
  const invalid = checkOpResult(op, pair);
  const size = pngSize(png);
  if (!size) invalid.push(`${redact(png)} is not a PNG`);
  else if (size[0] !== WIDTH || size[1] !== HEIGHT) invalid.push(`capture is ${size[0]}x${size[1]}, not ${WIDTH}x${HEIGHT}`);
  const adapter = {
    vendor: op.adapter.vendor,
    architecture: op.adapter.architecture,
    device: op.adapter.device,
    description: op.adapter.description,
    isFallbackAdapter: op.adapter.isFallbackAdapter,
  };
  const hardware = isHardware(adapter);
  if (invalid.length === 0 && typeof op.hardware === 'boolean' && op.hardware !== hardware) {
    invalid.push(`page says hardware ${op.hardware} but the Conventions rule gives ${hardware}`);
  }
  if (invalid.length > 0) {
    warn(`${name}: page result rejected (${invalid.join('; ')}); no record written`);
    return { name, problems: invalid.map((p) => `page result: ${p}`) };
  }

  const capturedAt = new Date().toISOString();
  const capture = `proofs/captures/lookdev/${name}.png`;
  copyFileSync(png, capture);
  let judge = null;
  if (ctx.judge && pair.variant.judge) {
    log(`${name}: judge against ${pair.variant.references.join(', ') || 'no reference'}`);
    judge = await runJudge(capture, pair.variant, ctx.refPrefix);
  }
  const record = {
    scene: pair.scene,
    variant: pair.variant.id,
    milestone: pair.milestone,
    capturedAt,
    adapter,
    hardware,
    canvas: [WIDTH, HEIGHT],
    nonEmpty: op.nonEmpty,
    passes: enabledPasses(op.passes),
    instances: op.instances,
    triangles: op.triangles,
    firstFrameMs: op.firstFrameMs,
    wasm: ctx.wasm,
    capture,
    captureSha256: sha256(capture),
    judge,
  };
  writeJsonAtomic(join('proofs', 'results', 'lookdev', `${name}.json`), record);

  const judged = judge === null ? ''
    : `, judge exit ${judge.exit} score ${judge.score ?? 'unknown'} (min ${JUDGE_MIN_SCORE})`;
  log(`${name}: adapter "${redact(adapter.description || adapter.device || adapter.vendor)}" `
    + `${hardware ? 'hardware' : 'software'}, nonEmpty ${op.nonEmpty}, passes ${redact(record.passes.join(' > '))}${judged}`);
  if (judge !== null && judge.exit !== 0) {
    const out = tail(judge.output, 20);
    warn(`${name}: judge failed (exit ${judge.exit})${out ? `\n${out}` : ''}`);
    problems.push(`judge exit ${judge.exit}, score ${judge.score ?? 'unknown'}`);
  }
  if (!hardware) problems.push('captured on a software adapter');
  if (!op.nonEmpty) problems.push('capture is empty (nonEmpty false)');
  return { name, problems, hardware, nonEmpty: op.nonEmpty, judge };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const client = process.env.NMC_GPU_CLIENT;
  if (!client) {
    fail('NMC_GPU_CLIENT is unset: the GPU client is required for lookdev captures; '
      + 'no capture or record written (CPU output is never substituted). '
      + 'Stop and record it in docs/dogfood/.', 2);
  }
  const distDir = join('dist', args.page);
  const hint = PAGES[args.page];
  const wasm = readSizes(distDir, hint);
  checkManifest(distDir, hint);
  const index = readSceneIndex(distDir, hint);

  const { pairs, problems } = await selectPairs(distDir, index, args);
  const filters = [args.milestone && `--milestone ${args.milestone}`, args.only !== null && `--only ${redact(args.only)}`]
    .filter(Boolean).join(' ') || 'no filter';
  if (pairs.length === 0) {
    for (const p of problems) warn(`  - ${p}`);
    fail(`no (scene, variant) pair selected from ${distDir}/scenes.json (${filters}); nothing captured`);
  }
  log(`${pairs.length} pair(s) selected from ${distDir}/scenes.json (${filters}), up to NMC_CPUS in parallel`);

  const judging = args.judge && pairs.some((p) => p.variant.judge);
  if (judging) {
    if (args.refPrefix === null) fail(`--judge needs --ref-prefix <p> (references are <p>ref-N.png)\n${USAGE}`, 2);
    if (!process.env.NMC_JUDGE_CLIENT) fail('--judge needs NMC_JUDGE_CLIENT (the judge client); nothing captured', 2);
    if (!existsSync(process.env.NMC_JUDGE_CLIENT)) fail('NMC_JUDGE_CLIENT does not point to an existing file; nothing captured', 2);
    if (!args.refPrefix.includes('://')) {
      const refs = new Set(pairs.filter((p) => p.variant.judge).flatMap((p) => p.variant.references));
      const missing = [...refs].map((r) => `${args.refPrefix}${r}.png`).filter((f) => !existsSync(f));
      if (missing.length > 0) fail(`reference image(s) missing: ${redact(missing.join(', '))}; nothing captured`);
    }
  }

  // The client may be provided a moment after start; it is waited for, never replaced.
  const waitS = Number(process.env.NMC_GPU_WAIT_S) > 0 ? Number(process.env.NMC_GPU_WAIT_S) : 60;
  const waitLimit = 'NMC_GPU_WAIT_S (60 s when unset)';
  for (const deadline = Date.now() + waitS * 1000; !existsSync(client);) {
    if (Date.now() >= deadline) {
      fail(`NMC_GPU_CLIENT does not point to an existing file after waiting ${waitLimit}: `
        + 'the GPU client is missing; no capture or record written (CPU output is never substituted). '
        + 'Stop and record it in docs/dogfood/.', 2);
    }
    await sleep(1000);
  }

  log(`publish ${distDir}/`);
  const published = await runAsync(process.execPath, [client, 'publish', `${distDir}/`], waitS * 1000);
  if (!succeeded(published)) {
    const out = tail(`${published.stdout}\n${published.stderr}`);
    fail(`GPU client 'publish' ${whyFailed(published, waitLimit)}; no capture or record written${out ? `\n${out}` : ''}`);
  }
  const pageRef = parsePageRef(published.stdout);
  if (!pageRef) fail(`could not parse page_ref from the publish output; no capture or record written\n${tail(published.stdout)}`);
  log(`page_ref ${redact(pageRef)}`);

  for (const d of ['target/lookdev', 'proofs/results/lookdev', 'proofs/captures/lookdev']) mkdirSync(d, { recursive: true });
  const ctx = { client, pageRef, wasm, judge: args.judge, refPrefix: args.refPrefix ?? '' };
  const outcomes = await pool(pairs, async (pair) => {
    try {
      return await capturePair(pair, ctx);
    } catch (e) {
      warn(`${pair.name}: ${redact(e?.message ?? e)}`);
      return { name: pair.name, problems: [redact(e?.message ?? e)] };
    }
  });

  const recorded = outcomes.filter((o) => typeof o.hardware === 'boolean');
  const hw = recorded.filter((o) => o.hardware).length;
  log(`${recorded.length} of ${pairs.length} pair(s) recorded: ${hw} hardware, ${recorded.length - hw} software`);
  const failures = [...problems, ...outcomes.flatMap((o) => o.problems.map((p) => `${o.name}: ${p}`))];
  if (failures.length > 0) {
    warn(`${failures.length} failure(s):\n${failures.map((f) => `  - ${f}`).join('\n')}`);
    process.exit(1);
  }
  log('all selected pairs captured');
}

main().catch((error) => fail(redact(error?.stack ?? error)));
