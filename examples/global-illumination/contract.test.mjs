import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { buildPage } from '../../scripts/build-page.mjs';
import { validateInventory, validateRequiredFacts } from './lib/inventory.js';
import { createGiPageRunner, waitGiSettled } from './lib/runtime.js';
import * as volume from './scenes/volume.js';

// Node-only contract checks of the global-illumination page (S40): the temporary build, the
// inventory and facts validators, the waitGiSettled gate and createGiPageRunner on a stub engine.

// The build fixture: a throwaway repository root with copies of the real page sources plus a
// stub dist/package/raa, so scripts/build-page.mjs stages the page without cargo or a GPU.
const REPO = new URL('../../', import.meta.url);
function write(root, relative, text) {
  const file = path.join(root, relative);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, text);
}
function copyTree(root, relative) {
  fs.cpSync(new URL(relative, REPO), path.join(root, relative), { recursive: true });
}
function copyFile(root, relative) {
  const file = path.join(root, relative);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.copyFileSync(new URL(relative, REPO), file);
}
function fixtureRoot(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'gi-build-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  for (const directory of ['examples/global-illumination/lib', 'examples/global-illumination/scenes', 'web/common']) {
    copyTree(root, directory);
  }
  for (const file of [
    'examples/global-illumination/index.html', 'examples/global-illumination/main.js',
    'examples/global-illumination/nmc.gpu.json', 'examples/virtual-geometry/lib/content.js',
    'examples/sky-weather/lib/capture.js', 'examples/sky-weather/lib/measure.js',
    'examples/sky-weather/lib/readiness.js', 'examples/sky-weather/lib/evidence.mjs',
  ]) copyFile(root, file);
  write(root, 'dist/package/raa/index.js', 'export async function loadRaa() { return {}; }\n');
  write(root, 'dist/package/raa/index.d.ts', 'export {};\n');
  write(root, 'dist/package/raa/pkg/raa_js_bg.wasm', 'wasm\n');
  return root;
}

test('build-page stages the page, its scene, shared and common files, scenes.json and a ready build.json', async (t) => {
  const root = fixtureRoot(t);
  const { out, scenes, receipt } = await buildPage({
    root,
    topic: 'global-illumination',
    stages: ['volume', 'lighting', 'virtual'],
    shared: ['examples/virtual-geometry/lib/content.js'],
    captureInputs: () => ({ stub: true }),
  });
  assert.equal(out, path.join(root, 'dist/global-illumination'));
  for (const relative of [
    'raa/index.js', 'common/page-runner.js', 'lib/runtime.js',
    'scenes/volume.js', 'shared/content.js', 'shared/evidence.mjs',
  ]) assert.ok(fs.existsSync(path.join(out, relative)), `missing ${relative}`);
  assert.deepEqual(scenes, [{ id: 'lighting', milestone: 'lighting', file: 'scenes/lighting.js' }, { id: 'volume', milestone: 'volume', file: 'scenes/volume.js' }]);
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(out, 'scenes.json'), 'utf8')), scenes);
  assert.match(fs.readFileSync(path.join(out, 'lib/runtime.js'), 'utf8'), /from '\.\.\/common\/page-runner\.js'/);
  const build = JSON.parse(fs.readFileSync(path.join(out, 'build.json'), 'utf8'));
  assert.equal(build.status, 'ready');
  assert.equal(build.error, null);
  assert.deepEqual(build.inputs, { stub: true });
  assert.deepEqual(build.files, receipt.files);
  assert.ok(receipt.files.includes('scenes/volume.js') && !receipt.files.includes('build.json'));
});

// Inventory and facts (contracts 'gi-page' inventory validation and 'gi-facts').
const PASS = { name: 'gi.volume', role: 'pass', enabled: true, order: 25, required: false };
const SETTINGS = {
  enabled: true, intensity: 1, environmentDiffuse: 0,
  localLightShadows: true, reflections: true, screenTraces: true, debugView: 'voxels',
};
const QUALITY = { resolution: 96, levels: 5, probeRays: 8 };
const STATS = {
  active: true, inactiveReason: null, quality: QUALITY, finestVoxelMetres: 0.2,
  voxelizedForwardInstances: 8, voxelizedVirtualInstances: 0, pendingRebuild: false,
  framesSinceRebuild: 240, allocatedBytes: 148635648, statsFrameIndex: 300, pending: false,
};
function facts() {
  return {
    settings: SETTINGS,
    quality: QUALITY,
    stats: STATS,
    instances: [{ name: 'room', instance: 0, globalIllumination: { requested: null, voxelized: true, driver: 'forward', code: null, message: null } }],
    assertions: [
      { name: 'gi-active', passed: true, detail: 'active true' },
      { name: 'debug-view', passed: true, detail: 'debugView voxels' },
    ],
    edits: [{ operation: 'setGlobalIllumination', accepted: true, error: null }],
    passes: [PASS],
    viewSizes: [{ view: 0, width: 1280, height: 720 }],
    readiness: { status: 'ready', stage: 'volume', error: null },
  };
}
const ON_ROW = volume.inventory.find(row => row.id === 'volume-voxels-on');

test('validateInventory validates the volume scene and refuses the absent lighting stage', () => {
  const rows = validateInventory([volume], 'volume');
  assert.equal(rows.length, 8);
  assert.deepEqual(rows.map(row => row.id), volume.inventory.map(row => row.id));
  assert.throws(() => validateInventory([volume], 'lighting'), /unavailable/);
});

test('validateRequiredFacts accepts complete facts and refuses pending, false, repeated and null evidence', () => {
  validateRequiredFacts(ON_ROW, facts(), volume.factNames);
  assert.throws(() => validateRequiredFacts(ON_ROW, { ...facts(), readiness: { status: 'pending', stage: 'volume', error: null } }, volume.factNames), /required/);
  const falseDetail = 'active false, pendingRebuild true';
  const failed = facts();
  failed.assertions = [{ name: 'gi-active', passed: false, detail: falseDetail }];
  assert.throws(() => validateRequiredFacts(ON_ROW, failed, volume.factNames),
    error => /required/.test(error.message) && error.message.includes(falseDetail));
  assert.throws(() => validateRequiredFacts(ON_ROW, { ...facts(), stats: null }, volume.factNames), /required/);
  const repeated = { ...ON_ROW, requiredFacts: [...ON_ROW.requiredFacts, 'runtime:ready'] };
  assert.throws(() => validateRequiredFacts(repeated, facts(), volume.factNames), /duplicate selector/);
});

// Readiness gate (contract 'gi-runtime', waitGiSettled). The budget double spends one step per
// render; the stats sequence returns the frame just rendered. Nothing here needs a browser.
function budgetDouble(limit) {
  let remaining = limit;
  return { deadlineMs: limit, remainingMs: () => remaining, spend() { remaining -= 1; } };
}
function gateFrame(statsAt, budget) {
  const renders = [];
  const engine = { giStats: () => statsAt(renders.length) };
  const render = async () => { renders.push(renders.length + 1); budget?.spend(); };
  return { engine, render, renders };
}
const PENDING = { active: true, inactiveReason: null, pendingRebuild: true, framesSinceRebuild: 0 };
const INACTIVE = { active: false, inactiveReason: 'noInstance', pendingRebuild: false, framesSinceRebuild: 0 };

test('waitGiSettled stays pending until the budget expires and never reports ready', async () => {
  const budget = budgetDouble(3);
  const frame = gateFrame(() => PENDING, budget);
  const verdict = await waitGiSettled(frame.engine, frame.render, budget);
  assert.equal(verdict.status, 'failed');
  assert.match(verdict.error, /not settled/);
  assert.equal(verdict.stats.pendingRebuild, true);
  assert.ok(frame.renders.length > 0);
  assert.equal(budget.remainingMs(), 0);
});

test('waitGiSettled fails on a non-null inactiveReason and never reports ready', async () => {
  const budget = budgetDouble(100);
  const frame = gateFrame(() => INACTIVE, budget);
  const verdict = await waitGiSettled(frame.engine, frame.render, budget);
  assert.equal(verdict.status, 'failed');
  assert.match(verdict.error, /inactive \(noInstance\)/);
  assert.equal(frame.renders.length, 1);
});

test('waitGiSettled fails with the render error message', async () => {
  const budget = budgetDouble(100);
  const engine = { giStats: () => null };
  const verdict = await waitGiSettled(engine, async () => { throw new Error('render exploded'); }, budget);
  assert.equal(verdict.status, 'failed');
  assert.match(verdict.error, /render exploded/);
  assert.equal(verdict.stats, null);
});

test('waitGiSettled is ready exactly on the frame that reaches 240 since the rebuild', async () => {
  const budget = budgetDouble(240);
  const frame = gateFrame(frames => ({
    active: true, inactiveReason: null, pendingRebuild: false, framesSinceRebuild: frames,
  }), budget);
  const verdict = await waitGiSettled(frame.engine, frame.render, budget);
  assert.equal(verdict.status, 'ready');
  assert.equal(verdict.stats.framesSinceRebuild, 240);
  assert.equal(frame.renders.length, 240);
  assert.equal(frame.renders.at(-1), 240);
});

// createGiPageRunner (contract 'gi-runtime') on a stub Engine and the real volume scene. The
// clock delivers frames only when the test drives it; the capture returns a fixed PNG.
const PNG = 'iVBORw0KGgo=';
const ADAPTER = { vendor: 'NVIDIA', architecture: 'Ada', device: 'RTX 3090', description: 'managed', isFallbackAdapter: false };
function frameClock() {
  const callbacks = new Map();
  let time = 0;
  let nextId = 1;
  return {
    now: () => time,
    requestFrame(callback) { const id = nextId++; callbacks.set(id, callback); return id; },
    cancelFrame(id) { callbacks.delete(id); },
    waiting: () => callbacks.size,
    deliver(stepMs = 100) {
      time += stepMs;
      const due = [...callbacks.values()];
      callbacks.clear();
      for (const callback of due) callback(time);
    },
  };
}
function statusDouble() {
  const texts = [];
  return { set textContent(text) { texts.push(text); }, get textContent() { return texts.at(-1); }, texts };
}
function volumeEngine(stats) {
  const state = { renders: 0, resizes: [], disposed: 0 };
  let handle = 0;
  return {
    state,
    engine: {
      setSky() {}, setWeather() {}, setLight() {}, setAmbient() {}, setCamera() {}, setGlobalIllumination() {},
      addMaterial: () => (handle += 1),
      addMesh: () => (handle += 1),
      addNode: () => (handle += 1),
      addMeshInstance: () => (handle += 1),
      resize(width, height) { state.resizes.push([width, height]); },
      render() { state.renders += 1; },
      dispose() { state.disposed += 1; },
      passes: () => [PASS],
      stats: () => ({ frames: state.renders, triangles: 12, hardware: true, adapter: ADAPTER }),
      giStats: () => ({ ...stats }),
      giQuality: () => ({ ...QUALITY }),
      globalIllumination: () => ({ ...SETTINGS, enabled: stats.active, debugView: 'none' }),
      instanceGlobalIllumination: () => ({ requested: null, voxelized: false, driver: null, code: null, message: null }),
    },
  };
}
function harness(stats) {
  const clock = frameClock();
  const canvas = { width: 1280, height: 720, style: {} };
  const status = statusDouble();
  const { state, engine } = volumeEngine(stats);
  const captures = [];
  const runner = createGiPageRunner({
    canvas,
    status,
    scenes: new Map([[volume.default.id, volume]]),
    lib: {},
    load: async () => ({ createEngine: async () => engine }),
    clock,
    capture: async (...args) => { captures.push(args); return PNG; },
  });
  return { runner, clock, status, state, captures };
}
function runInput() {
  return {
    selectionId: 'selection-1', attemptId: 'attempt-1', buildId: 'build-1',
    scenario: 'volume', mode: 'voxels-off', width: 1280, height: 720, remainingMs: 300000,
  };
}
async function flush() {
  await new Promise(resolve => setImmediate(resolve));
}
async function settleRun(clock, promise) {
  let outcome = null;
  promise.then(value => { outcome = { value }; }, error => { outcome = { error }; });
  for (let turn = 0; outcome === null; turn += 1) {
    assert.ok(turn < 4000, 'the run never settled');
    await flush();
    if (outcome === null && clock.waiting() > 0) clock.deliver();
  }
  return outcome;
}

test('createGiPageRunner: the -off variant accepts disabled GI with no allocation and answers ready facts', async () => {
  const disabled = { ...STATS, active: false, inactiveReason: 'disabled', framesSinceRebuild: 0, allocatedBytes: 0 };
  const h = harness(disabled);
  const outcome = await settleRun(h.clock, h.runner.run(runInput()));
  assert.ok('value' in outcome, String(outcome.error?.message));
  const result = outcome.value;
  assert.deepEqual(result.facts.readiness, { status: 'ready', stage: 'volume', error: null });
  assert.match(h.status.textContent, /^ready: volume\/voxels-off/);
  assert.equal(result.images.length, 1);
  assert.equal(result.images[0].filename, 'gi-room-voxels-off-p3-frame.png');
  assert.equal(result.measurements.length, 1);
  assert.equal(h.captures.length, 1);
});

test('createGiPageRunner: the -off variant refuses allocatedBytes 1 and stays failed, never ready', async () => {
  const h = harness({ ...STATS, active: false, inactiveReason: 'disabled', framesSinceRebuild: 0, allocatedBytes: 1 });
  const outcome = await settleRun(h.clock, h.runner.run(runInput()));
  assert.match(String(outcome.error?.message), /expected GI disabled/);
  assert.match(h.status.textContent, /^failed:/);
  assert.equal(h.status.texts.some(text => text.startsWith('ready')), false);
  assert.equal(h.captures.length, 0);
});
