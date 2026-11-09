#!/usr/bin/env node
import { captureEvidence, attachedStages } from './lights-evidence.js';
// Metadata selection, managed transport, fail-closed freshness and judgment.
import { existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { spawn, execFileSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { inflateSync } from 'node:zlib';
import { setTimeout as delay } from 'node:timers/promises';
import { isAbsolute, join, relative, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs as parse } from 'node:util';

const OPS = ['select', 'check', 'clean', 'render', 'fresh', 'judge'];
const USAGE = 'usage: node web/common/lights-job.mjs <select|check|clean|render|fresh|judge> --page dist/lights [--out out/lights]';
const ASPECT = 16 / 9;
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const vec3 = (v) => Array.isArray(v) && v.length === 3 && v.every(isNum);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const add = (a, b) => a.map((v, i) => v + b[i]);
const mul = (a, s) => a.map((v) => v * s);
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const ensure = (ok, message) => { if (!ok) throw new Error(message); };
const unit = (a) => { const len = Math.hypot(...a); return len > 0 ? a.map((v) => v / len) : null; };

function readJson(path, hint) {
  ensure(existsSync(path), `${path} is missing: ${hint}`);
  try { return JSON.parse(readFileSync(path, 'utf8')); } catch (e) { throw new Error(`${path} is not valid JSON (${e.message}): ${hint}`); }
}

function parseArgs(argv) {
  const { values, positionals } = parse({ args: argv, allowPositionals: true,
    options: { page: { type: 'string' }, out: { type: 'string' } } });
  ensure(positionals.length === 1 && OPS.includes(positionals[0]) && values.page, USAGE);
  ensure(positionals[0] === 'check' || values.out, USAGE);
  return { op: positionals[0], page: values.page, out: values.out };
}

/** True when `box` overlaps the camera's view volume (conservative AABB/frustum test). */
export function inView(camera, box) {
  const forward = unit(sub(camera.target, camera.eye));
  const right = forward && unit(cross(forward, [0, 1, 0]));
  if (!right) return false;
  const up = cross(right, forward);
  const tanY = Math.tan((camera.fovYDeg * Math.PI) / 360);
  const tanX = tanY * (camera.aspect > 0 ? camera.aspect : ASPECT);
  const planes = [
    [forward, add(camera.eye, mul(forward, camera.near))],
    [mul(forward, -1), add(camera.eye, mul(forward, camera.far))],
    [sub(mul(forward, tanX), right), camera.eye],
    [add(mul(forward, tanX), right), camera.eye],
    [sub(mul(forward, tanY), up), camera.eye],
    [add(mul(forward, tanY), up), camera.eye],
  ];
  const center = box.min.map((v, i) => (v + box.max[i]) / 2);
  const half = box.min.map((v, i) => (box.max[i] - v) / 2);
  return planes.every(([normal, point]) => dot(normal, center) - dot(normal, point)
    + half.reduce((sum, h, i) => sum + Math.abs(normal[i]) * h, 0) >= 0);
}

function cameraProblems(camera) {
  if (!camera || !vec3(camera.eye) || !vec3(camera.target)) return ['no camera'];
  const problems = [];
  if (!isNum(camera.fovYDeg) || camera.fovYDeg <= 0 || camera.fovYDeg >= 180 || !isNum(camera.near)
    || camera.near <= 0 || !isNum(camera.far) || camera.far <= camera.near) problems.push('camera frustum is invalid');
  if (camera.aspect !== undefined && (!isNum(camera.aspect) || camera.aspect <= 0)) problems.push('camera aspect is invalid');
  const forward = vec3(camera.eye) && vec3(camera.target) ? unit(sub(camera.target, camera.eye)) : null;
  if (!forward || !unit(cross(forward, [0, 1, 0]))) problems.push('camera direction is invalid');
  return problems;
}

export function variantProblems(variant) {
  const selection = variant?.selection;
  const problems = [];
  if (typeof variant?.description !== 'string' || !variant.description.trim()) problems.push('no description');
  if (!isNum(variant?.timeMs) || variant.timeMs < 0) problems.push('timeMs is not a duration');
  if (variant?.evidence !== undefined && (!nonblank(variant.evidence?.label)
    || !/^[a-z0-9]+(?:-[a-z0-9]+)*\.png$/.test(variant.evidence?.filename))) problems.push('invalid functional image acceptance');
  if (variant?.judge !== true) problems.push('judge must be true');
  const cameraErrors = cameraProblems(selection?.camera);
  problems.push(...cameraErrors);
  if (!Array.isArray(selection?.subjects) || selection.subjects.length === 0) return [...problems, 'selection needs at least one subject'];
  selection.subjects.forEach((box, i) => {
    if (!box || !vec3(box.min) || !vec3(box.max) || box.min.some((x, k) => x > box.max[k])) problems.push(`subjects[${i}] is not a finite min<=max box`);
    else if (cameraErrors.length === 0 && !inView(selection.camera, box)) problems.push(`subjects[${i}] is outside the camera view`);
  });
  return problems;
}

// The module of an index entry must live inside the requested page build.
function sceneModulePath(page, file) {
  const root = resolve(page);
  const path = resolve(root, file);
  const rel = relative(root, path);
  return rel && !rel.startsWith('..') && !isAbsolute(rel) && existsSync(path) ? path : null;
}

async function loadScenes(page) {
  const indexPath = join(page, 'scenes.json');
  const index = readJson(indexPath, 'build the page first');
  ensure(Array.isArray(index) && index.length > 0, `${indexPath} is not a nonempty array`);
  const scenes = new Map();
  for (const [i, entry] of index.entries()) {
    ensure(entry && typeof entry.id === 'string' && entry.id.trim() && typeof entry.milestone === 'string'
      && typeof entry.file === 'string', `${indexPath} entry ${i} is not { id, milestone, file }`);
    const file = sceneModulePath(page, entry.file);
    ensure(file, `scene ${entry.id}: module ${entry.file} is not in the page build`);
    let def;
    try { def = (await import(pathToFileURL(file).href)).default; } catch (e) { throw new Error(`scene ${entry.id}: import failed (${e.message})`); }
    ensure(def?.id === entry.id && def.milestone === entry.milestone && typeof def.build === 'function', `scene ${entry.id}: metadata is invalid`);
    ensure(Array.isArray(def.variants) && def.variants.length > 0, `scene ${entry.id}: no variants`);
    const ids = def.variants.map((v) => v?.id);
    ids.forEach((id) => ensure(id === 'on' || id === 'off', `scene ${entry.id}: variant ${id} is not on or off`));
    ensure(new Set(ids).size === ids.length, `scene ${entry.id}: duplicate variant`);
    ensure(!scenes.has(def.id), `scene ${entry.id}: duplicate scene`);
    scenes.set(def.id, def);
  }
  return scenes;
}

function collect(scenes) {
  const pairs = [];
  const problems = [];
  for (const def of scenes.values()) for (const variant of def.variants) {
    const invalid = variantProblems(variant).map((p) => `${def.id}--${variant.id}: ${p}`);
    if (invalid.length > 0) problems.push(...invalid);
    else pairs.push({ scene: def.id, variant: variant.id, description: variant.description, timeMs: variant.timeMs,
      ...(variant.evidence ? { evidence: variant.evidence } : {}) });
  }
  ensure(problems.length === 0, `invalid selection:\n  ${problems.join('\n  ')}`);
  ensure(pairs.length > 0, 'selection is empty');
  return pairs;
}

export async function check(page) {
  return collect(await loadScenes(page));
}

// The selection records its produced build so later commands cannot run a stale one.
export async function select(page, out) {
  const pairs = await check(page);
  mkdirSync(out, { recursive: true });
  writeFileSync(join(out, 'selection.json'), `${JSON.stringify({ page: resolve(page), pairs }, null, 2)}\n`);
  return pairs;
}

const nonblank = (v) => typeof v === 'string' && v.trim().length > 0;
const hostId = (v) => nonblank(v) || (Number.isSafeInteger(v) && v > 0);
const iso = (v) => nonblank(v) && Number.isFinite(Date.parse(v)) && new Date(v).toISOString() === v;
const jsonWrite = (path, doc) => {
  const temp = `${path}.${randomUUID()}.tmp`;
  writeFileSync(temp, `${JSON.stringify(doc, null, 2)}\n`);
  renameSync(temp, path);
};

// Do not invent NMC_RUN_ID/NMC_COMMIT variables. Git is the worktree execution
// context; the installed client's NMC_GPU_REVISION is the host's launch revision.
function commitAt(page) {
  let commit;
  try { commit = execFileSync('git', ['-C', resolve(page), 'rev-parse', '--verify', 'HEAD'], { encoding: 'utf8' }).trim(); }
  catch (cause) { throw new Error('missing commit provenance: page must belong to its producing Git worktree', { cause }); }
  ensure(/^[0-9a-f]{40,64}$/.test(commit), 'missing commit provenance');
  ensure(!process.env.NMC_GPU_REVISION || process.env.NMC_GPU_REVISION === commit,
    'host NMC_GPU_REVISION differs from the producing worktree commit');
  return commit;
}

function buildIdentity(page) {
  const build = readJson(join(page, 'build.json'), 'rebuild lights with build-lights.sh for provenance');
  ensure(nonblank(build.buildId) && nonblank(build.commit) && iso(build.builtAt), 'missing build provenance');
  ensure(build.commit === commitAt(page), 'stale build commit provenance: rebuild lights');
  return build;
}

async function selectionAt(page, out) {
  const selection = readJson(join(out, 'selection.json'), 'select the produced page first');
  ensure(selection.page === resolve(page), 'selection belongs to a different page');
  const pairs = await check(page);
  ensure(Array.isArray(selection.pairs) && selection.pairs.length > 0, 'selection is empty');
  ensure(JSON.stringify(selection.pairs) === JSON.stringify(pairs), 'selection differs from produced metadata: select again');
  return pairs;
}

// All generated paths are contained and symlink-free, including existing ancestors.
function inside(root, name) {
  ensure(nonblank(name) && !name.includes('\\') && !name.includes(':') && !isAbsolute(name)
    && name.split('/').every((s) => s && s !== '.' && s !== '..'), `unsafe output path: ${name}`);
  const dest = resolve(root, name);
  ensure(!relative(resolve(root), dest).startsWith('..'), `escaping output path: ${name}`);
  let cursor = resolve(root);
  for (;;) {
    ensure(!lstatSync(cursor, { throwIfNoEntry: false })?.isSymbolicLink(), `symlink output path: ${cursor}`);
    const parent = resolve(cursor, '..');
    if (parent === cursor) break;
    cursor = parent;
  }
  cursor = resolve(root);
  for (const part of name.split('/')) {
    cursor = join(cursor, part);
    ensure(!lstatSync(cursor, { throwIfNoEntry: false })?.isSymbolicLink(), `symlink output path: ${cursor}`);
  }
  return dest;
}

function currentAttempt(out) {
  const pointer = readJson(inside(out, '.lights-current.json'), 'render a new lights run');
  ensure(/^\.lights-runs\/[0-9a-f-]{36}$/.test(pointer.attempt), 'invalid lights attempt pointer');
  inside(out, pointer.attempt);
  return pointer.attempt;
}

export async function clean(page, out) {
  await selectionAt(page, out);
  if (existsSync(inside(out, '.lights-current.json'))) {
    rmSync(inside(out, currentAttempt(out)), { recursive: true, force: true });
    rmSync(inside(out, '.lights-current.json'));
  }
  rmSync(inside(out, 'run.json'), { force: true });
}

const primaryEvidence = (pair) => pair.evidence ?? captureEvidence(pair.scene, pair.variant) ?? { filename: 'capture.png', label: pair.description };
const primaryFile = (pair) => primaryEvidence(pair).filename;
const stageFile = (pair, name) => pair.scene === 'attached-light' ? captureEvidence(pair.scene, pair.variant, name).filename : `${name}.png`;

const waitSeconds = () => {
  const n = process.env.NMC_GPU_WAIT_S === undefined ? 60 : Number(process.env.NMC_GPU_WAIT_S);
  ensure(Number.isFinite(n) && n > 0, 'NMC_GPU_WAIT_S must be positive seconds');
  return n;
};

async function managedClient(waitS) {
  const client = process.env.NMC_GPU_CLIENT;
  ensure(nonblank(client), 'NMC_GPU_CLIENT is missing: managed Windows Chrome is required');
  const until = Date.now() + waitS * 1000;
  while (!existsSync(client) && Date.now() < until) await delay(Math.min(100, Math.max(1, until - Date.now())));
  ensure(existsSync(client) && lstatSync(client).isFile(), `NMC_GPU_CLIENT unavailable after ${waitS}s`);
  return client;
}

// The installed CLI reports its queue/running transitions on stderr. Its running
// clock is separate from slot availability; never pass a wall-clock spawn timeout.
function clientCall(client, args, waitS, gpu = false) {
  return new Promise((resolveCall, reject) => {
    const child = spawn(process.execPath, [client, ...args], { stdio: ['ignore', 'pipe', 'pipe'], shell: false });
    let stdout = '', stderr = '', lineBuffer = '', failure = null, running = false;
    const stop = (message) => { failure ??= new Error(message); child.kill('SIGKILL'); };
    let timer = setTimeout(() => stop(`managed client ${gpu ? 'GPU waiting' : 'publication'} timeout after ${waitS}s`), waitS * 1000);
    child.stdout.on('data', (data) => { stdout += data; if (stdout.length > 1024 * 1024) stop('managed client stdout too large'); });
    child.stderr.on('data', (data) => {
      stderr = (stderr + data).slice(-16000);
      lineBuffer += data;
      const lines = lineBuffer.split(/\r?\n/);
      lineBuffer = lines.pop().slice(-16000);
      for (const line of lines) {
        if (gpu && /^GPU job .+: running \(/.test(line) && !running) {
          running = true;
          clearTimeout(timer);
          timer = setTimeout(() => stop('GPU job running timeout after 300s'), 300000);
        } else if (gpu && /^GPU job .+: done$/.test(line)) {
          // Downloads happen after the job finishes. Do not truncate a successful
          // job just because its separate file transfer crosses the running limit.
          clearTimeout(timer);
          timer = setTimeout(() => stop('managed output download timeout'), waitS * 1000);
        }
      }
    });
    child.on('error', (cause) => { failure = new Error(`managed client could not start: ${cause.message}`, { cause }); });
    child.on('close', (code, signal) => {
      clearTimeout(timer);
      if (failure) return reject(failure);
      const token = process.env.NMC_GPU_TOKEN;
      const details = token ? stderr.split(token).join('[redacted]') : stderr;
      if (code !== 0) return reject(new Error(`managed client failed (${code ?? signal}): ${details.trim()}`));
      resolveCall(stdout);
    });
  });
}

function pageReference(stdout) {
  const pick = (o) => [o, o?.result, o?.data].map((v) => v?.page_ref ?? v?.pageRef).find(nonblank);
  for (const text of [stdout, ...stdout.split(/\r?\n/)]) {
    try { const ref = pick(JSON.parse(text)); if (ref) return ref.trim(); } catch { /* CLI progress line */ }
    const match = /^\s*(?:page_ref|pageRef)\s*[:=]\s*["']?([^\s"']+)["']?\s*$/.exec(text);
    if (match) return match[1];
  }
  throw new Error('managed publication returned no page_ref');
}

function validateRun(op, pair, measured) {
  ensure(op?.scene === pair.scene && op?.variant === pair.variant, `wrong scene/variant: expected ${pair.scene}/${pair.variant}`);
  ensure(op.nonEmpty === true, `${pair.scene}/${pair.variant}: nonEmpty must be true`);
  const adapter = op.adapter;
  const keys = ['vendor', 'architecture', 'device', 'description'];
  ensure(op.hardware === true && adapter?.isFallbackAdapter === false
    && keys.every((k) => typeof adapter[k] === 'string')
    && !keys.some((k) => /software|swiftshader|llvmpipe|lavapipe|softpipe|fallback|basic render|microsoft basic|warp\b/i.test(adapter[k])),
  `${pair.scene}/${pair.variant}: hardware adapter required; software/fallback adapter refused`);
  ensure(keys.some((k) => nonblank(adapter[k])),
    `${pair.scene}/${pair.variant}: adapter identity is blank; capture must preserve WebGPU adapter info`);
  const size = measured ? [1920, 1080] : [1280, 720];
  ensure(JSON.stringify(op.canvas) === JSON.stringify(size), `invalid canvas dimensions for ${pair.scene}`);
  ensure(op.warmup === (measured ? 60 : 30) && op.frames === (measured ? 300 : 1), 'invalid warm-up/frame counts');
}

async function validateEvidence(op, pair, page) {
  const scenes = await loadScenes(page);
  const scene = scenes.get(pair.scene);
  const measured = !!scene.measure;
  validateRun(op, pair, measured);
  const browser = op.browser;
  ensure(browser && /\b(?:Headless)?Chrome\/\d+/.test(browser.userAgent)
    && /Windows NT/.test(browser.userAgent) && !/Edg\/|OPR\//.test(browser.userAgent)
    && /^Win(?:32|64|dows)$/.test(browser.platform), 'actual Chrome on Windows is required');
  ensure(isNum(browser.devicePixelRatio) && browser.devicePixelRatio === 1, 'invalid devicePixelRatio: DPR 1 is required');
  ensure(Array.isArray(op.gpuErrors), 'missing shader/device gpuErrors evidence');
  ensure(op.gpuErrors.length === 0, `shader/device GPU errors: ${op.gpuErrors.join('; ')}`);
  if (!measured) return;
  const config = scene.measure;
  ensure(config.width === 1920 && config.height === 1080 && config.warmup === 60 && config.frames === 300
    && config.devicePixelRatio === 1 && config.minFpsMean === 60, 'invalid measured scene configuration');
  ensure(nonblank(config.measurementFor) && op.measurement_for === config.measurementFor, 'missing/wrong measurement_for');
  // Consume the staged upstream summary implementation, never implement timing here.
  const { summarizeFrameIntervals } = await import(pathToFileURL(join(resolve(page), 'lib/runner.js')).href);
  for (const variant of ['on', 'off']) {
    const run = op.runs?.[variant];
    validateRun(run, { scene: pair.scene, variant }, true);
    ensure(Array.isArray(run.frameIntervalsMs) && run.frameIntervalsMs.length === 300
      && run.frameIntervalsMs.every((n) => isNum(n) && n > 0), `${variant}: invalid raw frame intervals`);
    const summary = summarizeFrameIntervals(run.frameIntervalsMs);
    for (const [key, value] of Object.entries(summary)) {
      ensure(isNum(run[key]) && run[key] > 0 && run[key] === value, `${variant}: invalid timing ${key}`);
    }
    if (run.gpuErrors !== undefined) ensure(Array.isArray(run.gpuErrors) && run.gpuErrors.length === 0, `${variant}: GPU errors`);
  }
  ensure(op.runs.on.fpsMean >= 60, 'many-lights on fpsMean must be >= 60');
  for (const key of ['frameIntervalsMs', 'frameIntervalMsMedian', 'frameIntervalMsP95', 'fpsMean', 'fpsMedian', 'fpsP95']) {
    ensure(JSON.stringify(op[key]) === JSON.stringify(op.runs[pair.variant][key]), `capture timing differs from ${pair.variant}: ${key}`);
  }
}

// PNG format validation (chunk CRCs and decoded scanlines), not an image comparison.
function validatePng(path, size) {
  const bytes = readFileSync(path);
  const bad = () => { throw new Error(`corrupt/empty PNG: ${path}`); };
  if (bytes.length < 45 || !bytes.subarray(0, 8).equals(Buffer.from('89504e470d0a1a0a', 'hex'))) bad();
  let offset = 8, header, palette = false, ended = false;
  const data = [];
  while (offset + 12 <= bytes.length) {
    const length = bytes.readUInt32BE(offset), end = offset + 8 + length;
    if (end + 4 > bytes.length) bad();
    const type = bytes.toString('ascii', offset + 4, offset + 8);
    let crc = 0xffffffff;
    for (const value of bytes.subarray(offset + 4, end)) {
      crc ^= value;
      for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
    }
    if (((crc ^ 0xffffffff) >>> 0) !== bytes.readUInt32BE(end)) bad();
    const chunk = bytes.subarray(offset + 8, end);
    if (!header && type !== 'IHDR') bad();
    if (type === 'IHDR') {
      if (header || length !== 13) bad();
      header = chunk;
    } else if (type === 'PLTE') {
      if (palette || data.length || length === 0 || length % 3 || length > 768) bad();
      palette = true;
    } else if (type === 'IDAT') data.push(chunk);
    else if (type === 'IEND') { if (length !== 0) bad(); ended = true; }
    else if (!/^[a-z]/.test(type)) bad(); // unknown critical PNG chunk
    offset = end + 4;
    if (ended) break;
  }
  if (!ended || offset !== bytes.length || !header || !data.length) bad();
  if (header.readUInt32BE(0) !== size[0] || header.readUInt32BE(4) !== size[1]) bad();
  const channels = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[header[9]], depth = header[8];
  const validDepths = { 0: [1, 2, 4, 8, 16], 2: [8, 16], 3: [1, 2, 4, 8], 4: [8, 16], 6: [8, 16] };
  if (!channels || !validDepths[header[9]].includes(depth) || header[10] || header[11] || header[12]
    || (header[9] === 3 && !palette)) bad();
  const stride = 1 + Math.ceil(size[0] * channels * depth / 8), expected = stride * size[1];
  let raw;
  try { raw = inflateSync(Buffer.concat(data), { maxOutputLength: expected }); } catch { bad(); }
  if (raw.length !== expected) bad();
  for (let row = 0; row < size[1]; row++) if (raw[row * stride] > 4) bad();
}

function outputFiles(dir, prefix = '') {
  return readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name)).flatMap((entry) => {
    const name = `${prefix}${entry.name}`, path = inside(dir, name);
    ensure(!entry.isSymbolicLink(), `symlink managed output: ${name}`);
    if (entry.isDirectory()) return outputFiles(path).map((child) => `${name}/${child}`);
    ensure(entry.isFile() && lstatSync(path).size > 0, `empty/missing output: ${name}`);
    return [name];
  });
}

async function pairOutput(page, dir, pair) {
  const files = outputFiles(dir).filter((name) => name !== 'completion.json');
  ensure(files.includes(primaryFile(pair)), 'missing explicit capture.png output');
  ensure(files.includes('result.json'), 'missing operation result.json output');
  const doc = readJson(join(dir, 'result.json'), 'managed render must produce its operation result');
  ensure(hostId(doc.job_id) && hostId(doc.client_id), 'missing managed job/client provenance');
  await validateEvidence(doc.result, pair, page);
  for (const file of files.filter((name) => name.endsWith('.png'))) validatePng(inside(dir, file), doc.result.canvas);
  if (pair.scene === 'attached-light') ensure(JSON.stringify(doc.result.lifecycle?.frames?.map(frame => frame.name))
    === JSON.stringify(attachedStages.map(([name]) => name)), 'incomplete attachment lifecycle');
  // Attached lifecycle filenames are authored by the page, not guessed from any PNG.
  for (const frame of doc.result.lifecycle?.frames ?? []) if (frame.name !== undefined) {
    ensure(files.includes(stageFile(pair, frame.name)), `missing lifecycle output: ${frame.name}.png`);
  }
  return { doc, files };
}

export async function render(page, out) {
  const pairs = await selectionAt(page, out);
  // Invalidate the previous completion before any operation that can fail.
  await clean(page, out);
  const attempt = `.lights-runs/${randomUUID()}`;
  mkdirSync(inside(out, attempt), { recursive: true });
  jsonWrite(inside(out, '.lights-current.json'), { attempt });
  const build = buildIdentity(page);
  const waitS = waitSeconds();
  const client = await managedClient(waitS);
  const pageRef = pageReference(await clientCall(client, ['publish', resolve(page)], waitS));
  // Before a producer returns, the only host-issued context is its publication.
  // One real managed job identifies the complete selection, never a clock
  // or an assumed NMC_RUN_ID. Its result remains alongside the pair receipts.
  const record = { pageRef, pairs: [], complete: false, buildId: build.buildId,
    runId: pageRef, images: pairs.flatMap(pair => [
      { scene: pair.scene, variant: pair.variant, ...primaryEvidence(pair) },
      ...(pair.scene === 'attached-light' ? attachedStages.map(([stage]) => ({ scene: pair.scene, variant: pair.variant, stage, ...captureEvidence(pair.scene, pair.variant, stage) })) : []),
    ]), commit: build.commit, startedAt: new Date().toISOString(), completedAt: null };
  jsonWrite(inside(out, 'run.json'), record);
  try {
    const input = inside(out, `${attempt}/input.json`);
    const batchDir = inside(out, attempt);
    jsonWrite(input, { args: { pairs: pairs.map(({ scene, variant }) => ({ scene, variant })) } });
    await clientCall(client, ['render', input, '--page-ref', pageRef, '-o', batchDir], waitS, true);
    const batch = readJson(join(batchDir, 'result.json'), 'managed job must return the complete selection');
    ensure(hostId(batch.job_id) && hostId(batch.client_id), 'missing managed job/client provenance');
    record.runId = String(batch.job_id);
    ensure(Array.isArray(batch.result?.steps) && Array.isArray(batch.result?.diagnostics), 'missing whole-flow diagnostics/steps');
    const errors = batch.result.diagnostics.map((d) => `${d.pair?.scene ?? 'page'}/${d.pair?.variant ?? ''} ${d.kind}: ${d.message}`);
    if (batch.result.steps.length !== pairs.length) errors.push('whole-flow step count differs from selection');
    for (const pair of pairs) {
      const namespace = `${attempt}/s-${encodeURIComponent(pair.scene)}/${pair.variant}`;
      const dir = inside(out, namespace);
      try {
        const steps = batch.result.steps.filter((s) => s.scene === pair.scene && s.variant === pair.variant);
        ensure(steps.length === 1, `${pair.scene}/${pair.variant}: missing/duplicate whole-flow step`);
        ensure(!steps[0].error, steps[0].error);
        mkdirSync(dir, { recursive: true });
        jsonWrite(join(dir, 'result.json'), { job_id: batch.job_id, client_id: batch.client_id, result: steps[0].result });
        const { doc, files } = await pairOutput(page, dir, pair);
        jsonWrite(join(dir, 'completion.json'), { complete: true, pageRef, buildId: record.buildId, runId: record.runId,
          commit: record.commit, scene: pair.scene, variant: pair.variant, jobId: doc.job_id, clientId: doc.client_id, files });
        record.pairs.push({ scene: pair.scene, variant: pair.variant,
          capture: `${namespace}/${primaryFile(pair)}`, result: `${namespace}/result.json` });
      } catch (error) { errors.push(`${pair.scene}/${pair.variant}: ${error.message}`); }
    }
    jsonWrite(join(batchDir, 'diagnostics.json'), { errors });
    ensure(errors.length === 0, `whole-flow failures:\n${errors.join('\n')}`);
    ensure(currentAttempt(out) === attempt, 'render attempt was superseded');
    record.complete = true;
    record.completedAt = new Date().toISOString();
    jsonWrite(inside(out, 'run.json'), record);
    return await fresh(page, out);
  } catch (cause) {
    record.complete = false;
    record.completedAt = null;
    if (currentAttempt(out) === attempt) jsonWrite(inside(out, 'run.json'), record);
    throw cause;
  }
}

export async function fresh(page, out) {
  const selection = await selectionAt(page, out);
  const build = buildIdentity(page);
  const record = readJson(inside(out, 'run.json'), 'render must complete successfully first');
  ensure(record.complete === true && iso(record.startedAt) && iso(record.completedAt)
    && Date.parse(record.completedAt) >= Date.parse(record.startedAt)
    && Date.parse(record.startedAt) >= Date.parse(build.builtAt), 'run is unsuccessful or incomplete');
  ensure(nonblank(record.pageRef) && nonblank(record.runId) && record.buildId === build.buildId
    && record.commit === build.commit, 'stale/missing run/build/commit provenance');
  const attempt = currentAttempt(out);
  ensure(Array.isArray(record.pairs) && record.pairs.length === selection.length, 'run pairs do not match selection');
  const batch = readJson(inside(out, `${attempt}/result.json`), 'missing whole-flow result');
  ensure(String(batch.job_id) === record.runId && hostId(batch.client_id), 'wrong managed job provenance');
  ensure(Array.isArray(batch.result?.steps) && batch.result.steps.length === selection.length
    && Array.isArray(batch.result?.diagnostics) && batch.result.diagnostics.length === 0, 'incomplete/failed whole-flow result');
  const expectedImages = selection.flatMap(pair => [
    { scene: pair.scene, variant: pair.variant, ...primaryEvidence(pair) },
    ...(pair.scene === 'attached-light' ? attachedStages.map(([stage]) => ({ scene: pair.scene, variant: pair.variant, stage, ...captureEvidence(pair.scene, pair.variant, stage) })) : []),
  ]);
  ensure(JSON.stringify(record.images) === JSON.stringify(expectedImages), 'capture manifest differs from produced metadata');
  ensure(new Set(record.images.map(image => image.filename)).size === record.images.length, 'capture filenames collide');
  let clientId;
  for (const pair of selection) {
    const matches = record.pairs.filter((p) => p.scene === pair.scene && p.variant === pair.variant);
    ensure(matches.length === 1, `missing/duplicate run pair: ${pair.scene}/${pair.variant}`);
    const run = matches[0];
    const namespace = `${attempt}/s-${encodeURIComponent(pair.scene)}/${pair.variant}`;
    ensure(run.capture === `${namespace}/${primaryFile(pair)}` && run.result === `${namespace}/result.json`, 'stale/unsafe pair paths');
    const dir = inside(out, namespace);
    const receipt = readJson(inside(dir, 'completion.json'), 'producer must validate successfully');
    ensure(receipt.complete === true && ['pageRef', 'buildId', 'runId', 'commit'].every((k) => receipt[k] === record[k])
      && receipt.scene === pair.scene && receipt.variant === pair.variant, 'stale/unsuccessful producer completion');
    const { doc, files } = await pairOutput(page, dir, pair);
    if (pair === selection[0]) ensure(record.runId === String(doc.job_id), 'wrong managed run provenance');
    const steps = batch.result.steps.filter((s) => s.scene === pair.scene && s.variant === pair.variant);
    ensure(steps.length === 1 && !steps[0].error && JSON.stringify(steps[0].result) === JSON.stringify(doc.result)
      && doc.client_id === batch.client_id, 'pair differs from whole-flow result');
    clientId ??= doc.client_id;
    ensure(receipt.jobId === doc.job_id && receipt.clientId === doc.client_id
      && String(doc.job_id) === record.runId && doc.client_id === clientId, 'wrong managed job provenance');
    ensure(JSON.stringify(receipt.files) === JSON.stringify(files), 'missing/changed producer output inventory');
  }
  ensure(currentAttempt(out) === attempt && JSON.stringify(readJson(join(out, 'run.json'), 'current run')) === JSON.stringify(record),
    'run changed during freshness validation');
  return record;
}

const ATTACHED_STAGES = attachedStages.map(([name, , label]) => [name, label]);

function judgeScore(text) {
  for (const chunk of [text.trim(), ...text.split(/\r?\n/).reverse()]) {
    if (!chunk.trim().startsWith('{')) continue;
    try { const score = JSON.parse(chunk).score; if (isNum(score)) return score; } catch { /* progress output */ }
  }
  const verdicts = [...text.matchAll(/\bJUDGE (?:PASS|FAIL): (\d+(?:\.\d+)?)\/10\b/g)];
  if (verdicts.length) return Number(verdicts.at(-1)[1]);
  const scores = [...text.matchAll(/(?<![-\w])score\W{0,4}(\d+(?:\.\d+)?)/gi)];
  return scores.length ? Number(scores.at(-1)[1]) : null;
}

function judgeOne(capture, description) {
  const client = process.env.NMC_JUDGE_CLIENT;
  ensure(nonblank(client), 'NMC_JUDGE_CLIENT is missing');
  return new Promise((resolveJudge, rejectJudge) => {
    const child = spawn('python3', [client, capture, '--description', description, '--min-score', '6', '--aspect', 'functional'],
      { stdio: ['ignore', 'pipe', 'pipe'], shell: false });
    let stdout = '', stderr = '';
    child.stdout.on('data', (data) => { stdout += data; });
    child.stderr.on('data', (data) => { stderr += data; });
    child.on('error', (cause) => rejectJudge(new Error(`judge process could not start: ${cause.message}`, { cause })));
    child.on('close', (exitCode, signal) => {
      const score = judgeScore(`${stdout}\n${stderr}`);
      if (exitCode !== 0) return rejectJudge(Object.assign(new Error(`judge process failed (${exitCode ?? signal}, score ${score ?? 'unknown'}): ${(stderr || stdout).trim()}`), { score, exitCode, stdout, stderr }));
      if (!isNum(score)) return rejectJudge(Object.assign(new Error('judge returned no parseable score verdict'), { exitCode, stdout, stderr }));
      if (score < 6 || score > 10) return rejectJudge(Object.assign(new Error(`judge score ${score} is outside the accepted range [6, 10]`), { score, exitCode, stdout, stderr }));
      resolveJudge({ score, exitCode, stdout, stderr });
    });
  });
}

export async function judge(page, out) {
  const run = await fresh(page, out);
  const selection = await selectionAt(page, out);
  ensure(selection.length === 14, `judge requires fourteen primary scene/variant captures; found ${selection.length}`);
  const record = { runId: run.runId, buildId: run.buildId, commit: run.commit, complete: false, images: [], errors: [] };
  jsonWrite(inside(out, 'judge.json'), record);
  const judgeImage = async (pair, capture, description) => {
    const entry = { scene: pair.scene, variant: pair.variant, capture, description, label: description, aspect: 'functional' };
    try { Object.assign(entry, await judgeOne(inside(out, capture), description)); }
    catch (error) {
      Object.assign(entry, { error: error.message, score: error.score ?? null, exitCode: error.exitCode ?? null,
        stdout: error.stdout ?? '', stderr: error.stderr ?? '' });
      record.errors.push(`${capture}: ${error.message}`);
    }
    record.images.push(entry);
    jsonWrite(inside(out, 'judge.json'), record);
  };
  const runPairs = new Map(run.pairs.map((pair) => [`${pair.scene}\0${pair.variant}`, pair]));
  for (const pair of selection) {
    const runPair = runPairs.get(`${pair.scene}\0${pair.variant}`);
    ensure(runPair, `missing run prerequisite for ${pair.scene}/${pair.variant}`);
    await judgeImage(pair, runPair.capture, primaryEvidence(pair).label);

    if (pair.scene === 'attached-light') {
      const resultPath = inside(out, runPair.result);
      const doc = readJson(resultPath, 'fresh render result is required for lifecycle judgment');
      const frames = doc.result?.lifecycle?.frames;
      ensure(Array.isArray(frames), `missing attachment lifecycle output for ${pair.scene}/${pair.variant}`);
      ensure(JSON.stringify(frames.map((frame) => frame.name)) === JSON.stringify(ATTACHED_STAGES.map(([name]) => name)),
        `incomplete attachment lifecycle for ${pair.scene}/${pair.variant}`);
      for (const [name] of ATTACHED_STAGES) {
        const relativeCapture = `${relative(resolve(out), resolve(resultPath)).replace(/result\.json$/, stageFile(pair, name))}`;
        const stagePath = inside(out, relativeCapture);
        ensure(existsSync(stagePath) && lstatSync(stagePath).isFile() && lstatSync(stagePath).size > 0,
          `missing lifecycle output: ${name}.png`);
        await judgeImage(pair, relativeCapture, captureEvidence(pair.scene, pair.variant, name).label);
      }
    }
  }
  ensure(JSON.stringify(readJson(inside(out, 'run.json'), 'current run')) === JSON.stringify(run),
    'run changed during judgment');
  ensure(record.images.length === 32, `judge requires 32 images; found ${record.images.length}`);
  ensure(record.errors.length === 0, `image judgment failures (${record.errors.length}/32):\n${record.errors.join('\n')}`);
  record.complete = true;
  jsonWrite(inside(out, 'judge.json'), record);
  return record;
}

export async function main(argv = process.argv.slice(2)) {
  const { op, page, out } = parseArgs(argv);
  const result = await ({ check, select, clean, render, fresh, judge })[op](page, out);
  const count = Array.isArray(result) ? result.length : result?.pairs?.length ?? result?.images?.length;
  const label = op === 'judge' ? ' images' : ' pairs';
  process.stdout.write(`lights-job: ${op}${count === undefined ? '' : ` ${count}${label}`}\n`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((error) => {
    process.stderr.write(`lights-job: ${error?.message ?? error}\n`);
    process.exit(1);
  });
}
