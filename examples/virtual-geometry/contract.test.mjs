import test from 'node:test';
import assert from 'node:assert/strict';
import { validateInventory, validateRequiredFacts } from './lib/inventory.js';

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
