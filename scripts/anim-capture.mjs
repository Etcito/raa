#!/usr/bin/env node
// Programmatic-card entry point; never substitutes native or Node hashes.
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { appendFileSync, cpSync, existsSync, mkdirSync, readFileSync, renameSync,
  rmSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';

process.chdir(resolve(dirname(fileURLToPath(import.meta.url)), '..'));
const DET_CASES = [[0, 0], [1, 60], [305419896, 600]];
const resultPath = 'proofs/results/anim/det-selftest.json';
const work = 'target/gpu-out/anim-brick';
const stage = 'dist/anim-brick';
const deny = /swiftshader|llvmpipe|lavapipe|software|basic render/i;
const json = path => JSON.parse(readFileSync(path, 'utf8'));
const writeJson = (path, value) => writeFileSync(path, JSON.stringify(value, null, 2) + '\n');
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
let phase = 'configuration';

class CaptureError extends Error {}
function requireValue(ok, message) {
  if (!ok) throw new CaptureError(message);
}

// Do not echo child output: it can contain credentials, URLs or arbitrary env values.
// Static phase labels and exit/timeout codes provide safe, actionable friction.
function execute(program, args, timeoutMs) {
  return new Promise((resolveRun, reject) => {
    const child = spawn(program, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let size = 0;
    let failure;
    const timer = setTimeout(() => {
      failure = 'deadline exceeded';
      child.kill('SIGKILL');
    }, timeoutMs);
    for (const stream of [child.stdout, child.stderr]) stream.on('data', chunk => {
      size += chunk.length;
      if (size > 16 * 1024 * 1024) {
        failure = 'output limit exceeded';
        child.kill('SIGKILL');
      } else if (stream === child.stdout) stdout += chunk;
    });
    child.on('error', () => { failure = 'could not start process'; });
    child.on('close', code => {
      clearTimeout(timer);
      if (failure || code !== 0) reject(new CaptureError(failure || `process exit ${code}`));
      else resolveRun(stdout);
    });
  });
}

function pageRef(output) {
  for (const text of [output, ...output.split(/\r?\n/)]) {
    let doc;
    try { doc = JSON.parse(text); } catch { continue; }
    for (const value of [doc, doc?.result, doc?.data]) {
      const ref = value?.page_ref ?? value?.pageRef;
      if (typeof ref === 'string' && ref.trim()) return ref;
    }
  }
  throw new CaptureError('publish returned no page_ref');
}

function unwrap(doc, predicate) {
  const queue = [doc];
  for (let i = 0; i < queue.length && i < 32; i++) {
    const value = queue[i];
    if (!value || typeof value !== 'object' || Array.isArray(value)) continue;
    if (predicate(value)) return value;
    for (const key of ['json', 'result', 'value', 'data', 'output', 'opResult', 'op_result', 'run']) {
      if (value[key]) queue.push(value[key]);
    }
  }
  throw new CaptureError('missing operation result');
}

function validate(op, seed, ticks) {
  requireValue(op.case === 'det-selftest' && op.ticks === ticks
    && Array.isArray(op.seeds) && op.seeds.length === 1 && op.seeds[0] === seed,
  'browser case inputs differ from request');
  requireValue(op.brick?.name === 'raa-anim' && op.brick.version === '0.5.0'
    && Object.keys(op.brick).length === 2, 'invalid browser brick metadata');
  const a = op.adapter;
  const fields = ['vendor', 'architecture', 'device', 'description'];
  requireValue(a && fields.every(key => typeof a[key] === 'string')
    && typeof a.isFallbackAdapter === 'boolean', 'invalid browser adapter');
  const hardware = a.isFallbackAdapter === false && !fields.some(key => deny.test(a[key]));
  requireValue(op.hardware === hardware, 'browser hardware classification disagrees with rule');
  requireValue(hardware, 'software or fallback adapter refused');
  requireValue(Array.isArray(op.results) && op.results.length === 1, 'expected one browser result');
  const r = op.results[0];
  requireValue(r && Object.keys(r).sort().join(',') === 'case,hash,seed,ticks'
    && r.case === 'det-selftest' && r.seed === seed && r.ticks === ticks
    && Number.isInteger(r.seed) && r.seed >= 0 && r.seed <= 0xffffffff
    && Number.isInteger(r.ticks) && r.ticks >= 0 && r.ticks <= 0xffffffff
    && typeof r.hash === 'string' && /^[0-9a-f]{8}$/.test(r.hash), 'invalid browser result record');
}

async function main() {
  requireValue(process.argv.slice(2).join(' ') === '--case det-selftest',
    'usage: node scripts/anim-capture.mjs --case det-selftest');
  const waitS = Number(process.env.NMC_GPU_WAIT_S ?? 60);
  requireValue(Number.isFinite(waitS) && waitS > 0, 'NMC_GPU_WAIT_S must be positive seconds');
  const client = process.env.NMC_GPU_CLIENT;
  requireValue(client && existsSync(client), 'NMC_GPU_CLIENT is unavailable');
  // Each client call respects both the configured wait and the 60-second step ceiling.
  const clientMs = Math.min(waitS, 60) * 1000;
  const call = args => execute(process.execPath, [client, ...args], clientMs);
  rmSync(work, { recursive: true, force: true });
  mkdirSync(work, { recursive: true });
  phase = 'blank-page probe';
  const probe = `${work}/probe`;
  mkdirSync(probe);
  writeFileSync(`${probe}/index.html`, '<!doctype html><script>window.nmcGpu={ops:{render:async()=>({json:{ok:1},files:[]})}};</script>');
  writeJson(`${probe}/nmc.gpu.json`, { page: 'index.html', ops: ['render'] });
  const probeRef = pageRef(await call(['publish', probe]));
  writeJson(`${work}/probe-input.json`, {});
  await call(['render', `${work}/probe-input.json`, '--page-ref', probeRef, '-o', `${work}/probe-out`]);
  unwrap(json(`${work}/probe-out/result.json`), value => value.ok === 1);

  phase = 'local brick build';
  console.log('anim-capture: building local bricks');
  await execute('bash', ['scripts/build-bricks.sh'], 1800000);
  phase = 'page staging';
  rmSync(stage, { recursive: true, force: true });
  cpSync('examples/anim-brick', stage, { recursive: true });
  cpSync('web/common', `${stage}/common`, { recursive: true });
  cpSync('dist/bricks/raa-anim', `${stage}/raa-anim`, { recursive: true });
  cpSync('dist/bricks/bricks.json', `${stage}/bricks.json`);
  const manifest = json(`${stage}/nmc.gpu.json`);
  requireValue(manifest.page === 'index.html' && manifest.ops.includes('render'), 'invalid page manifest');
  const bytes = readFileSync(`${stage}/raa-anim/pkg/raa_anim_js_bg.wasm`);
  const wasm = { raw: bytes.length, gzip: gzipSync(bytes, { level: 9 }).length, sha256: sha256(bytes) };
  const built = json(`${stage}/bricks.json`).bricks.find(brick => brick.name === 'raa-anim');
  requireValue(built && ['raw', 'gzip', 'sha256'].every(key => built.wasm[key] === wasm[key]),
    'published wasm differs from bricks manifest');
  requireValue(wasm.gzip <= 300000, 'animation wasm exceeds gzip budget');
  console.log(`anim-capture: wasm raw ${wasm.raw} B, gzip ${wasm.gzip} B, sha256 ${wasm.sha256}`);
  phase = 'animation page publish';
  const ref = pageRef(await call(['publish', stage]));
  const runs = [];
  for (const [seed, ticks] of DET_CASES) {
    phase = `Chrome case seed=${seed} ticks=${ticks}`;
    const input = `${work}/input-${ticks}.json`;
    writeJson(input, { args: { case: 'det-selftest', seeds: [seed], ticks } });
    const out = `${work}/case-${ticks}`;
    await call(['render', input, '--page-ref', ref, '-o', out]);
    const op = unwrap(json(`${out}/result.json`), value => 'adapter' in value && 'results' in value);
    validate(op, seed, ticks);
    if (runs.length) requireValue(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter']
      .every(key => op.adapter[key] === runs[0].adapter[key]), 'adapter changed between cases');
    runs.push(op);
    console.log(`anim-capture: seed=${seed} ticks=${ticks} hash=${op.results[0].hash} (hardware)`);
  }
  phase = 'validated record write';
  requireValue(sha256(readFileSync(`${stage}/raa-anim/pkg/raa_anim_js_bg.wasm`)) === wasm.sha256,
    'staged wasm changed during capture');
  const record = { name: 'det-selftest', milestone: 'm16', capturedAt: new Date().toISOString(),
    case: 'det-selftest', adapter: runs[0].adapter, hardware: runs[0].hardware,
    brick: runs[0].brick, wasm, results: runs.flatMap(run => run.results), runs };
  mkdirSync(dirname(resultPath), { recursive: true });
  writeJson(`${resultPath}.tmp`, record);
  renameSync(`${resultPath}.tmp`, resultPath);
  console.log(`anim-capture: wrote ${resultPath}`);
}

main().catch(error => {
  // Only locally generated messages are safe; parsing/fs errors may include input data.
  const detail = error instanceof CaptureError ? error.message : 'invalid JSON or filesystem setup failed';
  const message = `anim-capture: ${phase}: ${detail}; no new browser evidence written`;
  console.error(message);
  mkdirSync('docs/dogfood', { recursive: true });
  appendFileSync('docs/dogfood/m16.md', `\n- ${new Date().toISOString()} capture friction: ${message}. Inspect local inputs/tool availability; client output omitted to protect secrets.\n`);
  process.exitCode = 1;
});
