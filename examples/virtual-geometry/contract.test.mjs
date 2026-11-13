import test from 'node:test';
import assert from 'node:assert/strict';
import { validateInventory, validateRequiredFacts } from './lib/inventory.js';
import { createVgPageRunner } from './lib/runtime.js';

// Synthetic scene namespaces only: no test imports a real scene, and every test builds its own.
const camera = { eye: [0, 1, 5], target: [0, 1, 0], fovYDeg: 60, near: .1, far: 100 };
const selection = { camera, subjects: [{ name: 'floor', min: [-100, -1, -100], max: [100, 0, 100] }, { name: 'mesh', min: [-1, 0, -1], max: [1, 2, 1] }] };
const selectors = ['snapshot:settings', 'snapshot:stats', 'snapshot:meshes', 'snapshot:instances', 'runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'assertion:drawn'];
function row(scenario, mode, n, extra = {}) {
  return { id: `${scenario}-row-${n}`, criterionName: 'Cluster patches', scenario, mode, filename: `${scenario}-row-${n}.png`, expectedVisibleBehavior: 'Cluster patches are visible', heldConstant: ['floor'], requiredFacts: ['runtime:ready', 'assertion:drawn'], pair: null, ...extra };
}
function scene(milestone, rows = 1) {
  const id = `${milestone}-scene`;
  const variants = [{ id: 'default', description: 'Default', references: [], judge: true, selection }];
  const inventory = Array.from({ length: rows }, (_, n) => row(id, 'default', n + 1));
  return { default: { id, milestone, title: 'Scene', variants, build() {} }, inventory, factNames: ['drawn'], observe: async () => ({}) };
}
// A bake scene whose single variant set is a measurement pair; states are [id, state, preset].
function paired(states) {
  const s = scene('bake');
  s.default.variants = states.map(([id, state, preset]) => ({ id, selection, measurement: { pair: 'fog', state, preset, features: { fog: state === 'on' } } }));
  s.inventory[0].mode = states[0][0];
  s.inventory[0].pair = 'fog';
  return s;
}
function facts() {
  return {
    settings: { enabled: true, errorPixels: 1, softwareRaster: true, occlusion: true, shadows: true, runtimeBake: true, instanceDefault: true, maxVisibleClusters: 262144, debugView: 'none' },
    stats: { active: false, inactiveReason: 'disabled', virtualInstances: 0, forwardInstances: 1, refused: {}, bakingMeshes: 0, readyMeshes: 1, drawnClusters: null, drawnTriangles: null, hardwareClusters: null, softwareClusters: null, shadowClusters: null, droppedClusters: null, statsFrameIndex: null, pending: false },
    meshes: [{ name: 'high-poly', mesh: 0, clusters: { state: 'ready' }, cuts: [{ errorMetres: 0, clusters: 4, triangles: 100 }], states: [{ state: 'ready', progress: 1 }] }],
    instances: [{ name: 'high-poly-0', instance: 0, virtualGeometry: { requested: null, path: 'virtual', pending: false, code: null, message: null } }],
    assertions: [{ name: 'drawn', passed: true, detail: 'Clusters drawn' }],
    edits: [{ operation: 'enable', accepted: true, error: null }],
    passes: [{ name: 'vg.cull', role: 'pass', enabled: true, order: 21, required: false }],
    viewSizes: [{ view: 0, width: 1280, height: 720 }],
    readiness: { status: 'ready', stage: 'bake', error: null },
  };
}

test('the bake stage selects bake rows and passes selects both stages in scene order', () => {
  const all = [scene('bake'), scene('passes', 2)];
  assert.deepEqual(validateInventory(all, 'bake').map(r => r.id), ['bake-scene-row-1']);
  assert.deepEqual(validateInventory(all, 'passes').map(r => r.id), ['bake-scene-row-1', 'passes-scene-row-1', 'passes-scene-row-2']);
});

test('a stage with no scene of its own is unavailable and unknown stages are refused', () => {
  assert.throws(() => validateInventory([scene('bake')], 'passes'), /unavailable/);
  assert.throws(() => validateInventory([scene('bake')], 'weather'), /stage/);
});

test('inventory refuses duplicate rows, bad filenames, unknown modes, holds and selectors', () => {
  for (const [mutate, pattern] of [
    [s => s.inventory.push({ ...s.inventory[0], filename: 'bake-row-2.png' }), /invalid inventory row bake-scene-row-1/], // duplicate row id
    [s => s.inventory.push({ ...s.inventory[0], id: 'bake-scene-row-2' }), /invalid inventory row bake-scene-row-2/], // duplicate filename
    [s => s.inventory[0].filename = 'bake.png', /invalid inventory row/], // filename without a hyphen
    [s => s.inventory[0].mode = 'missing', /invalid inventory row/], // unknown variant
    [s => s.inventory[0].heldConstant = [''], /invalid inventory row/], // empty held-constant entry
    [s => s.inventory[0].requiredFacts.push('snapshot:sky'), /unknown selector snapshot:sky/], // unknown selector
    [s => s.inventory[0].requiredFacts.push('assertion:undeclared'), /unknown selector assertion:undeclared/], // assertion not declared in factNames
    [s => s.inventory[0].requiredFacts.push('measurement:fog'), /unknown selector measurement:fog/], // measurement selector on a row without pair
    [s => s.default.variants[0].selection = { camera, subjects: [{ name: 'behind', min: [0, 0, 10], max: [1, 1, 11] }] }, /outside camera frustum/], // wholly behind the camera
  ]) {
    const s = scene('bake');
    mutate(s);
    assert.throws(() => validateInventory([s], 'bake'), pattern);
  }
});

test('measurement pairs need one on and one off variant named -on and -off with one preset', () => {
  assert.equal(validateInventory([paired([['fog-on', 'on', 'day'], ['fog-off', 'off', 'day']])], 'bake').length, 1);
  assert.throws(() => validateInventory([paired([['fog-on', 'on', 'day'], ['fog-other-on', 'on', 'day']])], 'bake'), /pair/);
  assert.throws(() => validateInventory([paired([['fog-enabled', 'on', 'day'], ['fog-off', 'off', 'day']])], 'bake'), /measurement/);
  assert.throws(() => validateInventory([paired([['fog-on', 'on', 'day'], ['fog-off', 'off', 'night']])], 'bake'), /pair/);
});

test('a floor box larger than the view is in frame and a box wholly behind the camera is not', () => {
  const s = scene('bake');
  s.default.variants[0].selection = { camera, subjects: [{ name: 'floor', min: [-60, -1, -60], max: [60, 0, 60] }] };
  assert.equal(validateInventory([s], 'bake').length, 1);
  s.default.variants[0].selection = { camera, subjects: [{ name: 'behind', min: [-1, 0, 8], max: [1, 1, 9] }] };
  assert.throws(() => validateInventory([s], 'bake'), /outside camera frustum/);
});

test('required facts pass complete facts and refuse missing, null, empty, failed and repeated evidence', () => {
  const row0 = { ...scene('bake').inventory[0], requiredFacts: selectors };
  validateRequiredFacts(row0, facts(), ['drawn']);
  for (const [mutate, pattern] of [
    [f => f.settings = null, /snapshot:settings/],
    [f => f.stats = null, /snapshot:stats/],
    [f => f.meshes = [], /snapshot:meshes/],
    [f => f.instances = [], /snapshot:instances/],
    [f => f.passes = [], /runtime:passes/],
    [f => f.viewSizes = [], /runtime:viewSizes/],
    [f => f.readiness.status = 'pending', /runtime:ready/],
    [f => f.readiness.error = 'originating failure', /runtime:ready/],
    [f => f.assertions[0] = { name: 'drawn', passed: false, detail: '0 of 4 clusters drawn' }, /0 of 4 clusters drawn/],
    [f => f.assertions.push({ ...f.assertions[0] }), /2 reported/],
    [f => f.edits = [{ operation: 'enable' }], /SceneObservation/],
  ]) {
    const f = facts();
    mutate(f);
    assert.throws(() => validateRequiredFacts(row0, f, ['drawn']), pattern);
  }
  assert.throws(() => validateRequiredFacts({ ...row0, requiredFacts: ['runtime:ready', 'runtime:ready'] }, facts(), ['drawn']), /duplicate/);
  assert.throws(() => validateRequiredFacts({ ...row0, requiredFacts: ['measurement:fog'] }, facts(), ['drawn']), /measurement:fog/);
  const withPair = { ...row0, pair: 'fog', requiredFacts: ['measurement:fog'] };
  validateRequiredFacts(withPair, facts(), ['drawn']);
});

// Runner readiness tests (contract 'vg-runtime', createVgPageRunner). Every test builds its own canvas,
// status, clock, Engine double and scene namespaces; the clock delivers frames only when a test drives it.
const PNG = 'iVBORw0KGgo=';
const PASS = { name: 'forward', role: 'pass', enabled: true, order: 60, required: true };
const ADAPTER = { vendor: 'NVIDIA', architecture: 'Ada', device: 'RTX 3090', description: 'managed', isFallbackAdapter: false };
const RUNNER_FACTS = ['runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'assertion:drawn'];

// The clock double of contract 'vg-runtime': requestFrame queues a callback; deliver() advances the time and runs the queued frames.
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

// A status element that records every text assigned to it.
function statusDouble() {
  const texts = [];
  return { set textContent(text) { texts.push(text); }, get textContent() { return texts.at(-1); }, texts };
}

// The Engine double; `state` is the test's record of its frames, resizes and disposal.
function engineDouble(state) {
  return {
    resize(width, height) { state.resizes.push([width, height]); },
    render(timeMs) { state.frames += 1; state.times.push(timeMs); },
    stats: () => ({ frames: state.frames, triangles: 12, hardware: true, adapter: ADAPTER }),
    passes: () => [PASS],
    dispose() { state.disposed += 1; },
  };
}

// A runner over the given scenes with the clock, capture and Engine doubles of this file.
function harness(scenes) {
  const clock = frameClock();
  const canvas = { width: 1280, height: 720, style: {} };
  const status = statusDouble();
  const engines = [];
  const captures = [];
  const load = async () => ({
    createEngine: async (_canvas, options) => {
      const state = { frames: 0, times: [], resizes: [], disposed: 0, options };
      engines.push(state);
      return engineDouble(state);
    },
  });
  const capture = async (...args) => { captures.push(args); return PNG; };
  const runner = createVgPageRunner({ canvas, status, load, scenes, lib: {}, clock, capture });
  return { runner, clock, status, engines, captures };
}

// A ready hook that reports 'baking' until its call number `frames`, then ready with `stage`.
function readyFrom(frames, stage = 'ready') {
  let calls = 0;
  return () => {
    calls += 1;
    return calls >= frames ? { ready: true, stage } : { ready: false, stage: 'baking' };
  };
}

function observation() {
  return { meshes: [], instances: [], assertions: [{ name: 'drawn', passed: true, detail: 'Clusters drawn' }], edits: [] };
}

// A scene namespace of milestone 'bake' built from the test's own build, ready hook and observation.
function runnerScene(id, { build = async () => ({}), ready = readyFrom(1), observe = async () => observation(), rows = 1 } = {}) {
  const variants = [{ id: 'default', description: 'Default', references: [], judge: true, selection }];
  const inventory = Array.from({ length: rows }, (_, n) => row(id, 'default', n + 1, { requiredFacts: RUNNER_FACTS }));
  return { default: { id, milestone: 'bake', title: 'Runner scene', variants, build, ready }, inventory, factNames: ['drawn'], observe };
}

function runInput(scenario, overrides = {}) {
  return { selectionId: 'selection-1', attemptId: 'attempt-1', buildId: 'build-1', scenario, mode: 'default', width: 1280, height: 720, remainingMs: 300000, ...overrides };
}

async function flush() {
  await new Promise(resolve => setImmediate(resolve));
}

// Delivers one frame whenever the run waits for one, until it settles; returns { value } or { error }.
async function settle(clock, promise) {
  let outcome = null;
  promise.then(value => { outcome = { value }; }, error => { outcome = { error }; });
  for (let turn = 0; outcome === null; turn += 1) {
    assert.ok(turn < 1000, 'the run never settled');
    await flush();
    if (outcome === null && clock.waiting() > 0) clock.deliver();
  }
  return outcome;
}

test('readiness: a scene that never reports ready fails at the deadline and shows failed, never ready', async () => {
  let observed = 0;
  const scene = runnerScene('pending-scene', {
    ready: () => ({ ready: false, stage: 'baking' }),
    observe: async () => { observed += 1; return observation(); },
  });
  const h = harness(new Map([[scene.default.id, scene]]));
  const outcome = await settle(h.clock, h.runner.run(runInput('pending-scene', { remainingMs: 1000 })));
  assert.match(String(outcome.error?.message), /readiness: baking not reached/);
  assert.match(h.status.textContent, /^failed:/);
  assert.equal(h.status.texts.some(text => text.startsWith('ready')), false);
  assert.equal(observed, 0);
  assert.equal(h.captures.length, 0);
});

test('readiness: a failed build rejects with its own error, shows failed: boom and renders nothing', async () => {
  const boom = new Error('boom');
  let readyCalls = 0;
  let observed = 0;
  const failing = runnerScene('failing-scene', {
    build: async () => { throw boom; },
    ready: () => { readyCalls += 1; return { ready: true, stage: 'ready' }; },
    observe: async () => { observed += 1; return observation(); },
  });
  const working = runnerScene('ready-scene');
  const h = harness(new Map([[failing.default.id, failing], [working.default.id, working]]));
  const outcome = await settle(h.clock, h.runner.run(runInput('failing-scene')));
  assert.strictEqual(outcome.error, boom);
  assert.equal(h.status.textContent, 'failed: boom');
  assert.equal(h.status.texts.some(text => text.startsWith('ready')), false);
  assert.equal(h.engines[0].frames, 0);
  assert.equal(readyCalls, 0);
  assert.equal(observed, 0);
  assert.equal(h.captures.length, 0);
  // The failed preparation does not block the next run on the same runner.
  const next = await settle(h.clock, h.runner.run(runInput('ready-scene')));
  assert.ok('value' in next, 'the next run must resolve');
  assert.equal(next.value.facts.readiness.status, 'ready');
  assert.equal(h.engines[0].disposed, 1);
});

test('readiness: a second run while the first is pending rejects and the first run still completes', async () => {
  const scene = runnerScene('pending-scene', { ready: readyFrom(30) });
  const h = harness(new Map([[scene.default.id, scene]]));
  const first = h.runner.run(runInput('pending-scene'));
  await flush();
  assert.equal(h.clock.waiting(), 1);
  await assert.rejects(h.runner.run(runInput('pending-scene')), /already in progress/);
  const outcome = await settle(h.clock, first);
  assert.ok('value' in outcome, 'the first run must still resolve');
  assert.equal(outcome.value.facts.readiness.status, 'ready');
});

test('readiness: a scene ready from frame 40 resolves with the bake milestone as its stage and one image per row', async () => {
  let framesAtObservation = -1;
  const scene = runnerScene('bake-scene', {
    ready: readyFrom(40),
    observe: async engine => { framesAtObservation = engine.stats().frames; return observation(); },
    rows: 2,
  });
  const h = harness(new Map([[scene.default.id, scene]]));
  const outcome = await settle(h.clock, h.runner.run(runInput('bake-scene')));
  assert.ok('value' in outcome, 'the run must resolve');
  const result = outcome.value;
  assert.deepEqual(result.facts.readiness, { status: 'ready', stage: 'bake', error: null });
  assert.ok(framesAtObservation >= 40, `observation taken after ${framesAtObservation} frames`);
  assert.deepEqual(result.provenance, { selectionId: 'selection-1', attemptId: 'attempt-1', buildId: 'build-1', scenario: 'bake-scene', mode: 'default' });
  assert.deepEqual(result.images.map(image => image.filename), scene.inventory.map(item => item.filename));
  assert.deepEqual(result.images.map(image => image.observation), scene.inventory.map(item => item.expectedVisibleBehavior));
  assert.ok(result.images.every(image => image.data === PNG && image.mime === 'image/png'));
  assert.equal(h.captures.length, 1);
  assert.equal(result.facts.settings, null);
  assert.equal(result.facts.stats, null);
  assert.deepEqual(result.facts.passes, [PASS]);
  assert.deepEqual(result.facts.viewSizes, [{ view: 0, width: 1280, height: 720 }]);
  assert.match(h.status.textContent, /^ready: /);
});

test('dispose is idempotent, disposes the Engine once and a run after dispose rejects', async () => {
  const scene = runnerScene('bake-scene');
  const h = harness(new Map([[scene.default.id, scene]]));
  assert.ok('value' in (await settle(h.clock, h.runner.run(runInput('bake-scene')))));
  h.runner.dispose();
  h.runner.dispose();
  assert.equal(h.runner.engine(), null);
  assert.equal(h.engines[0].disposed, 1);
  await assert.rejects(h.runner.run(runInput('bake-scene')), /disposed/);
});

test('scenes/bake.js: the real module validates the four bake rows in order, no behaviour names a fixture, and importing it twice has no side effect', async () => {
  const first = await import('./scenes/bake.js');
  const again = await import('./scenes/bake.js?again');
  const rows = validateInventory([first], 'bake');
  assert.deepEqual(rows.map(r => r.filename), ['bake-finest-cut-cluster-colours.png', 'bake-three-cuts-coarser-left-to-right.png', 'bake-foliage-thinned-keeps-coverage.png', 'bake-sliced-progress-then-complete.png']);
  for (const r of rows) {
    assert.ok(r.expectedVisibleBehavior.length > 0);
    for (const fixture of r.heldConstant) assert.ok(!r.expectedVisibleBehavior.includes(fixture), `${r.id} names ${fixture}`);
  }
  assert.equal(again.default.id, 'bake');
});
