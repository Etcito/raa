#!/usr/bin/env node
// M10 first-pixel capture driver: a box and a sphere GPU-stamped into the R8Snorm world
// volume and sphere-traced at screen resolution through nomorechat's managed GPU client.
//
//   node scripts/trace-first-pixel.mjs stage    dist/trace-first-pixel/ from packages/raa/pkg
//                                               (run scripts/build-package.sh first)
//   node scripts/trace-first-pixel.mjs clean    removes every earlier capture/result/record
//   node scripts/trace-first-pixel.mjs render --out proofs/captures/trace-first-pixel.png
//                                               blank probe, small frame, full frame
//   node scripts/trace-first-pixel.mjs fresh    checks the capture is this run's GPU trace
//                                               output (dark background, both shapes, no
//                                               speckles) and writes the capture record
//   node scripts/trace-first-pixel.mjs retain --out out/trace-first-pixel
//                                               keeps the image and records for judgment
//
// Stage 7119 re-renders the same scene as its own fresh baseline (`--baseline`) and adds a
// mesh-occlusion case (`--occlusion`): a green mesh slab in front of the box proves that
// trace.composite's fragment depth lets meshes and traced surfaces occlude each other.
//
//   node scripts/trace-first-pixel.mjs render --baseline --out proofs/captures/trace-first-pixel-baseline.png
//   node scripts/trace-first-pixel.mjs fresh --baseline
//   node scripts/trace-first-pixel.mjs render --occlusion --out proofs/captures/trace-first-pixel-occlusion.png
//   node scripts/trace-first-pixel.mjs fresh --occlusion
//
// The record (proofs/results/trace-first-pixel[-baseline|-occlusion].trace.json) keeps the
// scene parameters, the code and wasm identity and the output dimensions next to the image
// the judge reads.
import { spawnSync } from 'node:child_process';
import assert from 'node:assert/strict';
import { inflateSync } from 'node:zlib';
import {
  copyFileSync, cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync,
} from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { OCCLUSION, SCENE, programBytes } from '../examples/trace-first-pixel/scene.js';

const NAME = 'trace-first-pixel';
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const EXAMPLE = join(ROOT, 'examples', NAME);
const DIST = join(ROOT, 'dist', NAME);
const PKG = join(ROOT, 'packages', 'raa');
const INPUT = join(EXAMPLE, 'gpu-input.json');

// `first` is the accepted 7129 gate; `baseline` is stage 7119's fresh box-and-sphere
// capture; `occlusion` adds the mesh slab. Each case owns its capture, result, record and
// output folders so the cases never overwrite one another.
const CASES = {
  first: { suffix: '', mesh: false },
  baseline: { suffix: '-baseline', mesh: false },
  occlusion: { suffix: '-occlusion', mesh: true },
};
const CASE = process.argv.includes('--occlusion') ? 'occlusion'
  : process.argv.includes('--baseline') ? 'baseline' : 'first';
const INFO = CASES[CASE];
const paths = (info) => ({
  capture: join(ROOT, 'proofs', 'captures', `${NAME}${info.suffix}.png`),
  result: join(ROOT, 'proofs', 'results', `${NAME}${info.suffix}.json`),
  record: join(ROOT, 'proofs', 'results', `${NAME}${info.suffix}.trace.json`),
  out: join(ROOT, 'target', 'gpu-out', `${NAME}${info.suffix}`),
  evidence: join(ROOT, 'out', `${NAME}${info.suffix}`),
});
const { capture: CAPTURE, result: RESULT, record: RECORD, out: OUT, evidence: EVIDENCE } = paths(INFO);
const CLEAN_MARK = join(ROOT, 'target', `${NAME}.clean`);
const CASE_FLAG = INFO.suffix ? ` --${CASE}` : '';

function fail(message) {
  process.stderr.write(`trace-first-pixel: ${message}\n`);
  process.exit(1);
}

// Artifact identity comes from the published page reference and source commit, without hashes.
function writeSizes(file, rel, out) {
  const buf = readFileSync(file);
  const sizes = { wasm: rel, raw: buf.length, commit: git(['rev-parse', 'HEAD']) };
  writeFileSync(out, JSON.stringify(sizes, null, 2) + '\n');
}

// Serialized into the staged page; reuse the delivered engine and managed-client protocol.
// No idle animation, timing run or compatibility adapter is needed for a still image.
function capturePage() {
  const canvas = document.getElementById('c');
  const phase = (name) => { window.__nmcGpuProgress = name; };
  window.nmcGpu = { ops: { render: async (input) => {
    const { width, height } = input.args;
    canvas.width = width;
    canvas.height = height;
    const commands = { volumes: [], passes: [], copies: [] };
    const errors = [];
    let device;
    phase('request core hardware adapter');
    const requestAdapter = navigator.gpu.requestAdapter.bind(navigator.gpu);
    navigator.gpu.requestAdapter = async (options) => {
      const adapter = await requestAdapter(options);
      if (!adapter || adapter.info.isFallbackAdapter) throw new Error('A core hardware adapter is required');
      console.info('first pixel adapter', JSON.stringify(adapter.info));
      const requestDevice = adapter.requestDevice.bind(adapter);
      adapter.requestDevice = async (descriptor) => {
        device = await requestDevice(descriptor);
        device.addEventListener('uncapturederror', (event) => errors.push(event.error.message));
        device.lost.then((info) => { if (info.reason !== 'destroyed') errors.push(info.message); });
        for (const kind of ['internal', 'out-of-memory', 'validation']) device.pushErrorScope(kind);
        const createTexture = device.createTexture.bind(device);
        device.createTexture = (desc) => {
          if (desc.label === 'raa-sdf-trace.volume') commands.volumes.push({
            format: desc.format, dimension: desc.dimension, size: desc.size, mips: desc.mipLevelCount,
          });
          return createTexture(desc);
        };
        const createEncoder = device.createCommandEncoder.bind(device);
        device.createCommandEncoder = (...args) => {
          const encoder = createEncoder(...args);
          const begin = encoder.beginComputePass.bind(encoder);
          encoder.beginComputePass = (desc) => {
            phase(desc?.label ?? 'compute pass');
            commands.passes.push(desc?.label);
            return begin(desc);
          };
          const copy = encoder.copyBufferToTexture.bind(encoder);
          encoder.copyBufferToTexture = (src, dst, size) => {
            if (dst.texture.label === 'raa-sdf-trace.volume') commands.copies.push({ mip: dst.mipLevel ?? 0, size });
            return copy(src, dst, size);
          };
          return encoder;
        };
        return device;
      };
      return adapter;
    };
    phase('load delivered RAA wasm and create engine');
    const raa = await loadRaa();
    const engine = await raa.createEngine(canvas, { msaa: 1 });
    // The first pixel is the traced shapes alone on a dark flat background: the default
    // sky image would fill every miss with a light gradient.
    engine.setPassEnabled('sky.background', false);
    engine.setBackground(SCENE.background);
    engine.setTraceScene(programBytes(SCENE));
    const { target, distance, yaw, pitch } = SCENE.camera;
    engine.setOrbit(target, distance, yaw, pitch);
    engine.setLight(SCENE.light);
    engine.setAmbient(SCENE.ambient);
    let mesh = null;
    if (input.args.mesh) {
      // A flat unlit slab drawn as a normal RAA mesh: trace.composite must lose the depth
      // test behind it, so the green face replaces the red box surface on that pixel.
      mesh = OCCLUSION.mesh;
      const handle = engine.primitiveMesh('box', { halfExtents: mesh.halfExtents });
      const material = engine.addMaterial({ color: mesh.color, model: 'flat' });
      engine.addInstance(engine.addNode(null, mesh.center), handle, material);
    }
    phase('stamp and trace one frame');
    engine.render(0);
    phase('read full-resolution canvas');
    // Read before awaiting: WebGPU canvas presentation may discard its contents at a task boundary.
    const png = canvas.toDataURL('image/png').split(',')[1];
    await device.queue.onSubmittedWorkDone();
    for (let i = 0; i < 3; i++) {
      const error = await device.popErrorScope();
      if (error) errors.push(error.message);
    }
    if (errors.length) throw new Error(errors.join('\n'));
    const stats = engine.stats();
    const trace = { scene: SCENE, stats: engine.traceStats(), commands, mesh };
    const artifact = await (await fetch('./sizes.json')).json();
    phase('return image and GPU command record');
    return { json: { adapter: stats.adapter, hardware: stats.hardware, canvas: [canvas.width, canvas.height], trace, artifact },
      files: [{ name: 'capture.png', base64: png }] };
  } } };
}

function stage() {
  preflight();
  const wasm = join(PKG, 'pkg', 'raa_js_bg.wasm');
  if (!existsSync(wasm)) fail(`${wasm} is missing: run bash scripts/build-package.sh first`);
  rmSync(DIST, { recursive: true, force: true });
  mkdirSync(join(DIST, 'raa'), { recursive: true });
  for (const f of ['index.html', 'scene.js', 'nmc.gpu.json']) {
    copyFileSync(join(EXAMPLE, f), join(DIST, f));
  }
  for (const f of ['package.json', 'index.js', 'index.d.ts', 'README.md']) {
    copyFileSync(join(PKG, f), join(DIST, 'raa', f));
  }
  cpSync(join(PKG, 'pkg'), join(DIST, 'raa', 'pkg'), { recursive: true });
  cpSync(join(ROOT, 'web', 'common'), join(DIST, 'common'), { recursive: true });
  writeFileSync(join(DIST, 'main.js'), `import { loadRaa } from './raa/index.js';\nimport { SCENE, OCCLUSION, programBytes } from './scene.js';\n(${capturePage.toString()})();\n`);
  writeSizes(join(DIST, 'raa', 'pkg', 'raa_js_bg.wasm'), 'raa/pkg/raa_js_bg.wasm', join(DIST, 'sizes.json'));
  mkdirSync(join(DIST, 'probe'), { recursive: true });
  writeFileSync(join(DIST, 'probe', 'nmc.gpu.json'), JSON.stringify({ page: 'index.html', ops: ['probe'] }));
  writeFileSync(join(DIST, 'probe', 'index.html'), '<script>window.nmcGpu={ops:{probe:async()=>({json:{ok:1}})}};</script>');
  console.log(`trace-first-pixel: staged ${DIST}/`);
}

function clean() {
  for (const info of Object.values(CASES)) {
    const p = paths(info);
    for (const path of [p.capture, p.result, p.record, p.out, p.evidence]) {
      rmSync(path, { recursive: true, force: true });
    }
  }
  mkdirSync(dirname(CLEAN_MARK), { recursive: true });
  writeFileSync(CLEAN_MARK, new Date().toISOString() + '\n');
  console.log('trace-first-pixel: earlier first, baseline and occlusion captures removed');
}

function render() {
  // `--out` names the capture so the commands file shows which entry writes it.
  const out = process.argv[process.argv.indexOf('--out') + 1];
  if (!process.argv.includes('--out') || resolve(ROOT, out) !== CAPTURE) {
    fail(`render needs --out proofs/captures/${NAME}${INFO.suffix}.png${CASE_FLAG}`);
  }
  if (!existsSync(join(DIST, 'sizes.json'))) fail(`${DIST}/sizes.json is missing: run the stage step first`);
  preflight();
  // The client owns queue waiting and job timeouts; never kill a client waiting for its slot.
  const publish = (dir) => {
    const output = client(['publish', dir]);
    for (const line of [output, ...output.split('\n')]) {
      try {
        const doc = JSON.parse(line);
        const ref = doc.page_ref ?? doc.pageRef ?? doc.result?.page_ref ?? doc.data?.page_ref;
        if (ref) return ref;
      } catch { /* publish may also print progress lines */ }
    }
    fail('publish returned no page_ref');
  };
  const run = (op, input, ref, folder) => {
    mkdirSync(folder, { recursive: true });
    const file = join(folder, 'input.json');
    writeFileSync(file, JSON.stringify(input));
    client([op, file, '--page-ref', ref, '-o', folder]);
  };
  const blank = join(OUT, 'blank');
  run('probe', {}, publish(join(DIST, 'probe')), blank);
  if (!readOp(blank, (o) => o.ok === 1)) fail('blank-page probe did not answer');
  const pageRef = publish(DIST);
  const small = join(OUT, 'small');
  run('render', { args: { width: 160, height: 90, ...(INFO.mesh ? { mesh: true } : {}) } }, pageRef, small);
  validateOp(readOp(small, (o) => o.trace && o.adapter), 160, 90);
  const image = (folder, width, height) => {
    const png = readdirSync(folder, { recursive: true }).map(String).find((f) => f.endsWith('.png'));
    if (!png) fail('the managed render returned no PNG');
    const file = join(folder, png);
    assert.deepEqual(pngSize(readFileSync(file)), [width, height], 'the returned PNG must match the requested screen');
    return file;
  };
  image(small, 160, 90);
  const input = JSON.parse(readFileSync(INPUT, 'utf8'));
  if (INFO.mesh) input.args = { ...input.args, mesh: true };
  const full = join(OUT, 'full');
  run('render', input, pageRef, full);
  const op = opResult();
  validateOp(op, input.args.width, input.args.height);
  mkdirSync(dirname(CAPTURE), { recursive: true });
  copyFileSync(image(full, input.args.width, input.args.height), CAPTURE);
  mkdirSync(dirname(RESULT), { recursive: true });
  writeFileSync(RESULT, JSON.stringify({ ...op, pageRef, capturedAt: new Date().toISOString() }, null, 2) + '\n');
  console.log(`trace-first-pixel: captured ${input.args.width}x${input.args.height}; adapter ${JSON.stringify(op.adapter)}`);
}

function client(args) {
  if (!process.env.NMC_GPU_CLIENT) fail('NMC_GPU_CLIENT is unset; GPU execution belongs to the managed runner');
  const res = spawnSync(process.execPath, [process.env.NMC_GPU_CLIENT, ...args], {
    cwd: ROOT, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024,
  });
  if (res.error || res.status !== 0) {
    let message = res.error?.message ?? `${res.stdout}\n${res.stderr}`;
    for (const [name, value] of Object.entries(process.env)) {
      if (/TOKEN|SECRET|KEY/.test(name) && value) message = message.split(value).join('[redacted]');
    }
    fail(`managed client ${args[0]} failed: ${message}`);
  }
  return res.stdout;
}

// Width and height from a PNG IHDR chunk.
export function pngSize(buf) {
  const sig = '89504e470d0a1a0a';
  if (buf.length < 24 || buf.subarray(0, 8).toString('hex') !== sig || buf.toString('ascii', 12, 16) !== 'IHDR') {
    return null;
  }
  return [buf.readUInt32BE(16), buf.readUInt32BE(20)];
}

// RGB(A) 8-bit non-interlaced PNG -> { width, height, channels, data } (filters 0-4).
export function decodePng(buf) {
  const size = pngSize(buf);
  if (!size) throw new Error('not a PNG');
  const [width, height] = size;
  const depth = buf[24], type = buf[25], interlace = buf[28];
  const channels = { 2: 3, 6: 4 }[type];
  if (depth !== 8 || !channels || interlace !== 0) throw new Error(`unsupported PNG (depth ${depth}, type ${type}, interlace ${interlace})`);
  const chunks = [];
  for (let at = 8; at + 8 <= buf.length;) {
    const len = buf.readUInt32BE(at), kind = buf.toString('ascii', at + 4, at + 8);
    if (kind === 'IDAT') chunks.push(buf.subarray(at + 8, at + 8 + len));
    if (kind === 'IEND') break;
    at += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(chunks));
  const stride = width * channels;
  const data = Buffer.alloc(stride * height);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? data[y * stride + x - channels] : 0;
      const b = y > 0 ? data[(y - 1) * stride + x] : 0;
      const c = x >= channels && y > 0 ? data[(y - 1) * stride + x - channels] : 0;
      const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
      const pred = [0, a, b, (a + b) >> 1, pa <= pb && pa <= pc ? a : pb <= pc ? b : c][filter];
      if (pred === undefined) throw new Error(`bad PNG filter ${filter}`);
      data[y * stride + x] = (line[x] + pred) & 255;
    }
  }
  return { width, height, channels, data };
}

// Pixel of a world point under the capture camera (same convention as checkView).
export function projectPixel(scene, point, width, height) {
  const { target, distance, yaw, pitch } = scene.camera;
  const right = [Math.cos(yaw), 0, -Math.sin(yaw)];
  const back = [Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch), Math.cos(pitch) * Math.cos(yaw)];
  const up = [-Math.sin(pitch) * Math.sin(yaw), Math.cos(pitch), -Math.sin(pitch) * Math.cos(yaw)];
  const v = point.map((p, a) => p - target[a]);
  const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
  const fy = Math.tan(Math.PI / 6), fx = fy * width / height, z = distance - dot(v, back);
  return [Math.round((dot(v, right) / (z * fx) + 1) / 2 * width), Math.round((1 - dot(v, up) / (z * fy)) / 2 * height)];
}

// Deterministic capture check, run before the visual judge: the background is dark at the
// four corners, the box (reddish) and the sphere (bluish) cover their projected centres, and
// their lit surfaces carry no dark speckles (back-face hits of a broken refinement).
export function checkCapture(image, scene) {
  const { width, height, channels, data } = image;
  const at = (x, y) => { const i = (y * width + x) * channels; return [data[i], data[i + 1], data[i + 2]]; };
  const problems = [];
  const m = Math.round(Math.min(width, height) * 0.03);
  for (const [x, y] of [[m, m], [width - 1 - m, m], [m, height - 1 - m], [width - 1 - m, height - 1 - m]]) {
    const px = at(x, y);
    if (Math.max(...px) > 80) problems.push(`background at ${x},${y} is ${px.join(',')}, not dark`);
  }
  const r = Math.max(2, Math.round(height / 40));
  for (const [name, hue] of [['box', 0], ['sphere', 2]]) {
    const [cx, cy] = projectPixel(scene, scene[name].center, width, height);
    const px = [];
    for (let y = cy - r; y <= cy + r; y++) for (let x = cx - r; x <= cx + r; x++) {
      if (x >= 0 && y >= 0 && x < width && y < height) px.push(at(x, y));
    }
    if (!px.length) { problems.push(`${name} centre ${cx},${cy} is off screen`); continue; }
    const luma = px.map((p) => Math.max(...p)).sort((a, b) => a - b);
    const median = luma[luma.length >> 1];
    const mean = [0, 1, 2].map((c) => px.reduce((s, p) => s + p[c], 0) / px.length);
    if (!(mean[hue] > mean[1] && mean[hue] > mean[2 - hue])) {
      problems.push(`${name} centre ${cx},${cy} is ${mean.map(Math.round).join(',')}, not ${hue ? 'blue' : 'red'}`);
    }
    const dark = luma.filter((v) => v < median / 2).length;
    if (dark > px.length * 0.005) problems.push(`${name} surface has ${dark} of ${px.length} dark speckle pixels`);
  }
  return problems;
}

// Stage 7119 preflight, without a GPU: the slab must contain the point where the eye-to-
// hidden-point ray crosses its mid plane and must miss the eye-to-box-centre ray, so the
// capture can show the mesh hiding the box surface while the box centre stays visible.
export function meshCoverage(scene, occlusion) {
  const { target, distance, yaw, pitch } = scene.camera;
  const back = [Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch), Math.cos(pitch) * Math.cos(yaw)];
  const eye = target.map((t, i) => t + back[i] * distance);
  const { center, halfExtents } = occlusion.mesh;
  const covers = (point) => {
    const d = point.map((p, i) => p - eye[i]);
    const t = (center[2] - eye[2]) / d[2];
    const hit = eye.map((e, i) => e + d[i] * t);
    return Math.abs(hit[0] - center[0]) <= halfExtents[0] && Math.abs(hit[1] - center[1]) <= halfExtents[1];
  };
  return { hidden: covers(occlusion.hidden), centre: covers(scene.box.center) };
}

// Additional-case check: the traced shapes still pass checkCapture, and the slab owns both
// its own centre pixel and the box pixel directly behind it. The second pixel can only turn
// green through trace.composite's depth test; the trace alone would leave the red box there.
export function checkOcclusion(image, scene, occlusion) {
  const { width, height, channels, data } = image;
  const at = (x, y) => { const i = (y * width + x) * channels; return [data[i], data[i + 1], data[i + 2]]; };
  const problems = checkCapture(image, scene);
  for (const [name, point] of [['mesh centre', occlusion.mesh.center], ['hidden box surface', occlusion.hidden]]) {
    const [x, y] = projectPixel(scene, point, width, height);
    if (x < 0 || y < 0 || x >= width || y >= height) { problems.push(`${name} is off screen`); continue; }
    const px = at(x, y);
    if (!(px[1] > px[0] + 8 && px[1] > px[2] + 8)) problems.push(`${name} at ${x},${y} is ${px.join(',')}, not the green mesh`);
  }
  return problems;
}

// The raw op result the GPU client wrote (the page fields with `trace`).
function readOp(folder, accepts) {
  if (!existsSync(folder)) return null;
  for (const f of readdirSync(folder, { recursive: true }).map(String).sort()) {
    if (!f.toLowerCase().endsWith('.json')) continue;
    let doc;
    try { doc = JSON.parse(readFileSync(join(folder, f), 'utf8')); } catch { continue; }
    const queue = [doc];
    for (let i = 0; i < queue.length && i < 32; i++) {
      const o = queue[i];
      if (!o || typeof o !== 'object') continue;
      if (accepts(o)) return o;
      for (const key of ['result', 'value', 'data', 'output', 'json']) if (o[key]) queue.push(o[key]);
    }
  }
  return null;
}

function opResult() { return readOp(join(OUT, 'full'), (o) => o.trace && o.adapter); }

function validateOp(op, width, height) {
  if (!op || op.hardware !== true || op.adapter.isFallbackAdapter !== false ||
      /swiftshader|llvmpipe|lavapipe|software|basic render/i.test(JSON.stringify(op.adapter))) {
    fail('the page did not return a core hardware GPU result');
  }
  assert.deepEqual(op.canvas, [width, height], 'the GPU canvas must match the requested screen');
  const { stats, commands } = op.trace;
  if (stats.sources !== 2 || stats.instances !== 2 || stats.width !== width || stats.height !== height) {
    fail('the GPU trace scene or full-resolution output dimensions differ from the requested scene');
  }
  if (commands?.volumes?.length !== 1 || commands.volumes[0].format !== 'r8snorm' ||
      commands.volumes[0].dimension !== '3d' || commands.volumes[0].mips !== 5) {
    fail('missing the five-mip R8Snorm 3D world volume');
  }
  const copies = new Set((commands.copies ?? []).map((copy) => copy.mip));
  if (![0, 1, 2, 3, 4].every((mip) => copies.has(mip))) {
    fail(`the volume received mip copies ${[...copies].sort().join(',')}, expected 0,1,2,3,4`);
  }
  const passes = commands.passes ?? [];
  for (const pass of ['raa-sdf-trace.stamp', 'raa-sdf-trace.mip', 'raa-sdf-trace.coarse', 'raa-sdf-trace.trace']) {
    if (!passes.includes(pass)) fail(`the frame did not record ${pass}`);
  }
  if (INFO.mesh !== Boolean(op.trace.mesh)) fail(`the page ${INFO.mesh ? 'did not draw' : 'drew'} the occlusion mesh`);
  if (JSON.stringify(op.trace.scene) !== JSON.stringify(SCENE)) fail('the page used another scene');
}

function git(args) {
  const res = spawnSync('git', args, { cwd: ROOT, encoding: 'utf8' });
  return res.status === 0 ? res.stdout.trim() : null;
}

// Camera convention and defaults: crates/raa-scene/src/camera.rs. These are view guards,
// never SDF evaluation or rendered geometry. Test all six planes, including a box enclosing the view.
export function checkView(scene, width, height) {
  const { target, distance, yaw, pitch } = scene.camera;
  const right = [Math.cos(yaw), 0, -Math.sin(yaw)];
  const back = [Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch), Math.cos(pitch) * Math.cos(yaw)];
  const up = [-Math.sin(pitch) * Math.sin(yaw), Math.cos(pitch), -Math.sin(pitch) * Math.cos(yaw)];
  const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
  const fy = Math.tan(Math.PI / 6), fx = fy * width / height;
  for (const name of ['box', 'sphere']) {
    const shape = scene[name];
    const half = shape.halfExtents ?? [shape.radius, shape.radius, shape.radius];
    const planes = Array.from({ length: 8 }, (_, corner) => {
      const v = half.map((h, a) => shape.center[a] + (corner & (1 << a) ? h : -h) - target[a]);
      const z = distance - dot(v, back), x = dot(v, right), y = dot(v, up);
      return [z - 0.1, 200 - z, z * fx - x, z * fx + x, z * fy - y, z * fy + y];
    });
    assert(!planes[0].some((_, i) => planes.every((p) => p[i] < 0)), `${name} is outside the capture view`);
  }
}

function preflight() {
  const input = JSON.parse(readFileSync(INPUT, 'utf8')).args;
  for (const [w, h] of [[160, 90], [input.width, input.height]]) checkView(SCENE, w, h);
  assert.throws(() => checkView({ ...SCENE, sphere: { ...SCENE.sphere, center: [1000, 0, 0] } }, 1280, 720));
  assert.throws(() => checkView({ ...SCENE, box: { ...SCENE.box, center: [0, 0, 1000] } }, 1280, 720));
  checkView({ ...SCENE, box: { ...SCENE.box, halfExtents: [1000, 1000, 1000] } }, 1280, 720);
  const coverage = meshCoverage(SCENE, OCCLUSION);
  assert(coverage.hidden, 'the occlusion mesh must cover the box surface behind it');
  assert(!coverage.centre, 'the occlusion mesh must leave the box centre visible');
  console.log('trace-first-pixel: both required shapes and the mesh occlusion case fit the small and full views');
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
  const png = readFileSync(CAPTURE);
  const size = pngSize(png);
  const input = JSON.parse(readFileSync(INPUT, 'utf8')).args;
  if (!size || size[0] !== input.width || size[1] !== input.height) {
    fail(`capture is ${size ? size.join('x') : 'not a PNG'}, expected the ${input.width}x${input.height} canvas`);
  }
  const image = decodePng(png);
  const problems = INFO.mesh ? checkOcclusion(image, SCENE, OCCLUSION) : checkCapture(image, SCENE);
  if (problems.length) fail(`capture content check failed:\n  ${problems.join('\n  ')}`);
  const op = opResult();
  if (!op) fail(`no op result with trace stats in ${OUT}/`);
  validateOp(op, size[0], size[1]);
  if (!result.pageRef || op.artifact.commit !== git(['rev-parse', 'HEAD'])) fail('the published artifact belongs to another commit');
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
    case: CASE,
    milestone: 'm10',
    capture: `proofs/captures/${NAME}${INFO.suffix}.png`,
    output: { width: size[0], height: size[1] },
    scene: SCENE,
    mesh: INFO.mesh ? OCCLUSION : null,
    volume: 'R8Snorm 3D world volume, GPU compute stamped (raa-sdf-trace TracePass)',
    trace: stats,
    adapter: result.adapter,
    code: { commit: git(['rev-parse', 'HEAD']), dirty: (git(['status', '--porcelain', '--untracked-files=no']) ?? '') !== '' },
    program: { bytes: programBytes(SCENE).length },
    artifact: { pageRef: result.pageRef, ...op.artifact },
    commands: op.trace.commands,
    capturedAt: result.capturedAt,
  };
  writeFileSync(RECORD, JSON.stringify(record, null, 2) + '\n');
  console.log(`trace-first-pixel: fresh ${size.join('x')} hardware capture of 2 traced instances; record ${RECORD}`);
}

function retain() {
  const out = process.argv[process.argv.indexOf('--out') + 1];
  if (!process.argv.includes('--out') || resolve(ROOT, out) !== EVIDENCE) {
    fail(`retain needs --out out/${NAME}`);
  }
  fresh();
  mkdirSync(EVIDENCE, { recursive: true });
  copyFileSync(CAPTURE, join(EVIDENCE, 'capture.png'));
  copyFileSync(RESULT, join(EVIDENCE, 'result.json'));
  const record = JSON.parse(readFileSync(RECORD, 'utf8'));
  record.capture = `out/${NAME}/capture.png`;
  writeFileSync(join(EVIDENCE, 'capture.trace.json'), JSON.stringify(record, null, 2) + '\n');
  console.log(`trace-first-pixel: retained PNG and scene/code/artifact record in ${EVIDENCE}/`);
}

const steps = { check: preflight, stage, clean, render, fresh, retain };
const step = steps[process.argv[2]];
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (!step) fail(`usage: node scripts/trace-first-pixel.mjs ${Object.keys(steps).join('|')}`);
  step();
}
