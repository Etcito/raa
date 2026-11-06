#!/usr/bin/env node
// M10 first-pixel capture driver: a box and a sphere GPU-stamped into the R8Snorm world
// volume and sphere-traced at screen resolution through nomorechat's managed GPU client.
//
//   node scripts/trace-first-pixel.mjs stage    dist/trace-first-pixel/ from packages/raa/pkg
//                                               (run scripts/build-package.sh first)
//   node scripts/trace-first-pixel.mjs clean    removes every earlier capture/result/record
//   node scripts/trace-first-pixel.mjs render --out proofs/captures/trace-first-pixel.png
//                                               scripts/gpu-capture.mjs on the staged page
//   node scripts/trace-first-pixel.mjs fresh    checks the capture is this run's GPU trace
//                                               output and writes the capture record
//
// The record (proofs/results/trace-first-pixel.trace.json) keeps the scene parameters, the
// code and wasm identity and the output dimensions next to the image the judge reads.
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  copyFileSync, cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync,
} from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';
import { SCENE, programBytes } from '../examples/trace-first-pixel/scene.js';

const NAME = 'trace-first-pixel';
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const EXAMPLE = join(ROOT, 'examples', NAME);
const DIST = join(ROOT, 'dist', NAME);
const PKG = join(ROOT, 'packages', 'raa');
const OUT = join(ROOT, 'target', 'gpu-out', NAME);
const CAPTURE = join(ROOT, 'proofs', 'captures', `${NAME}.png`);
const RESULT = join(ROOT, 'proofs', 'results', `${NAME}.json`);
const RECORD = join(ROOT, 'proofs', 'results', `${NAME}.trace.json`);
const CLEAN_MARK = join(ROOT, 'target', `${NAME}.clean`);
const INPUT = join(EXAMPLE, 'gpu-input.json');

function fail(message) {
  process.stderr.write(`trace-first-pixel: ${message}\n`);
  process.exit(1);
}

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

// dist/<name>/sizes.json in the S7 shape gpu-capture.mjs requires.
function writeSizes(file, rel, out) {
  const buf = readFileSync(file);
  const sizes = { wasm: rel, raw: buf.length, gzip: gzipSync(buf, { level: 9 }).length, sha256: sha256(buf) };
  writeFileSync(out, JSON.stringify(sizes, null, 2) + '\n');
}

function stage() {
  const wasm = join(PKG, 'pkg', 'raa_js_bg.wasm');
  if (!existsSync(wasm)) fail(`${wasm} is missing: run bash scripts/build-package.sh first`);
  rmSync(DIST, { recursive: true, force: true });
  mkdirSync(join(DIST, 'raa'), { recursive: true });
  for (const f of ['index.html', 'main.js', 'scene.js', 'nmc.gpu.json']) {
    copyFileSync(join(EXAMPLE, f), join(DIST, f));
  }
  for (const f of ['package.json', 'index.js', 'index.d.ts', 'README.md']) {
    copyFileSync(join(PKG, f), join(DIST, 'raa', f));
  }
  cpSync(join(PKG, 'pkg'), join(DIST, 'raa', 'pkg'), { recursive: true });
  cpSync(join(ROOT, 'web', 'common'), join(DIST, 'common'), { recursive: true });
  writeSizes(join(DIST, 'raa', 'pkg', 'raa_js_bg.wasm'), 'raa/pkg/raa_js_bg.wasm', join(DIST, 'sizes.json'));
  console.log(`trace-first-pixel: staged ${DIST}/`);
}

function clean() {
  for (const p of [CAPTURE, RESULT, RECORD, OUT]) rmSync(p, { recursive: true, force: true });
  mkdirSync(dirname(CLEAN_MARK), { recursive: true });
  writeFileSync(CLEAN_MARK, new Date().toISOString() + '\n');
  console.log('trace-first-pixel: earlier capture, result and record removed');
}

function render() {
  // `--out` names the capture so the commands file shows which entry writes it.
  const out = process.argv[process.argv.indexOf('--out') + 1];
  if (!process.argv.includes('--out') || resolve(ROOT, out) !== CAPTURE) {
    fail(`render needs --out proofs/captures/${NAME}.png (gpu-capture writes it there)`);
  }
  if (!existsSync(join(DIST, 'sizes.json'))) fail(`${DIST}/sizes.json is missing: run the stage step first`);
  const res = spawnSync(process.execPath,
    [join(ROOT, 'scripts', 'gpu-capture.mjs'), NAME, '--input', INPUT, '--milestone', 'm10'],
    { cwd: ROOT, stdio: 'inherit', env: process.env });
  if (res.error) fail(`gpu-capture did not start: ${res.error.message}`);
  if (res.status !== 0) process.exit(res.status ?? 1);
}

// Width and height from a PNG IHDR chunk.
export function pngSize(buf) {
  const sig = '89504e470d0a1a0a';
  if (buf.length < 24 || buf.subarray(0, 8).toString('hex') !== sig || buf.toString('ascii', 12, 16) !== 'IHDR') {
    return null;
  }
  return [buf.readUInt32BE(16), buf.readUInt32BE(20)];
}

// The raw op result the GPU client wrote (the page fields with `trace`).
function opResult() {
  if (!existsSync(OUT)) return null;
  for (const f of readdirSync(OUT, { recursive: true }).map(String).sort()) {
    if (!f.toLowerCase().endsWith('.json')) continue;
    let doc;
    try { doc = JSON.parse(readFileSync(join(OUT, f), 'utf8')); } catch { continue; }
    for (const o of [doc, doc?.result, doc?.value, doc?.data, doc?.output, doc?.json]) {
      if (o && typeof o === 'object' && o.trace && o.adapter) return o;
    }
  }
  return null;
}

function git(args) {
  const res = spawnSync('git', args, { cwd: ROOT, encoding: 'utf8' });
  return res.status === 0 ? res.stdout.trim() : null;
}

function fresh() {
  if (!existsSync(CLEAN_MARK)) fail('the clean step did not run before this capture');
  const since = statSync(CLEAN_MARK).mtimeMs;
  for (const p of [CAPTURE, RESULT]) {
    if (!existsSync(p)) fail(`${p} was not written by the render step`);
    if (statSync(p).mtimeMs < since) fail(`${p} predates the clean step (stale)`);
  }
  const result = JSON.parse(readFileSync(RESULT, 'utf8'));
  if (result.hardware !== true || result.adapter?.isFallbackAdapter) {
    fail(`the capture is not from a hardware GPU adapter (${JSON.stringify(result.adapter)})`);
  }
  if (result.nonEmpty !== true) fail('the page reports an empty canvas');
  const png = readFileSync(CAPTURE);
  const size = pngSize(png);
  const input = JSON.parse(readFileSync(INPUT, 'utf8')).args;
  if (!size || size[0] !== input.width || size[1] !== input.height) {
    fail(`capture is ${size ? size.join('x') : 'not a PNG'}, expected the ${input.width}x${input.height} canvas`);
  }
  const op = opResult();
  if (!op) fail(`no op result with trace stats in ${OUT}/`);
  const stats = op.trace.stats ?? {};
  if (stats.sources !== 2 || stats.instances !== 2) {
    fail(`the traced scene has ${stats.sources} sources and ${stats.instances} instances, expected the box and the sphere (2, 2)`);
  }
  if (stats.width !== size[0] || stats.height !== size[1]) {
    fail(`trace target ${stats.width}x${stats.height} is not the ${size.join('x')} screen output`);
  }
  if (JSON.stringify(op.trace.scene) !== JSON.stringify(SCENE)) fail('the page traced another scene than examples/trace-first-pixel/scene.js');
  const record = {
    name: NAME,
    milestone: 'm10',
    capture: 'proofs/captures/trace-first-pixel.png',
    output: { width: size[0], height: size[1] },
    scene: SCENE,
    volume: 'R8Snorm 3D world volume, GPU compute stamped (raa-sdf-trace TracePass)',
    trace: stats,
    adapter: result.adapter,
    code: { commit: git(['rev-parse', 'HEAD']), dirty: (git(['status', '--porcelain']) ?? '') !== '' },
    program: { bytes: programBytes(SCENE).length, sha256: sha256(Buffer.from(programBytes(SCENE))) },
    wasm: result.wasm,
    capturedAt: result.capturedAt,
  };
  writeFileSync(RECORD, JSON.stringify(record, null, 2) + '\n');
  console.log(`trace-first-pixel: fresh ${size.join('x')} hardware capture of 2 traced instances; record ${RECORD}`);
}

const steps = { stage, clean, render, fresh };
const step = steps[process.argv[2]];
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (!step) fail(`usage: node scripts/trace-first-pixel.mjs ${Object.keys(steps).join('|')}`);
  step();
}
