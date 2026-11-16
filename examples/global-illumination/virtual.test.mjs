import test from 'node:test';
import assert from 'node:assert/strict';
import * as virtual from './scenes/virtual.js';
import * as volume from './scenes/volume.js';
import * as lighting from './scenes/lighting.js';
import * as temporal from './scenes/temporal.js';
import * as areaShadows from './scenes/area-shadows.js';
import * as areaStochastic from './scenes/area-stochastic.js';
import * as areaRestir from './scenes/area-restir.js';
import * as ssgi from './scenes/ssgi.js';
import * as giQuality from './scenes/gi-quality.js';
import { validateInventory } from './lib/inventory.js';

// Virtual stage of the global-illumination page, checked against contract 'gi-virtual-rows'. The
// literals below repeat that contract; they are not read back from scenes/virtual.js. Node only:
// no engine, no WebGPU.

// [id, filename] of the five rows, in contract order.
const ROWS = [
  ['virtual-field-on', 'gi-virtual-field-bumpy-spheres-take-red-bounce-from-wall.png'],
  ['virtual-field-off', 'gi-virtual-field-off-p3-frame-no-bounce.png'],
  ['virtual-shared-grid-on', 'gi-virtual-shared-grid-spheres-and-boxes-share-red-bounce.png'],
  ['virtual-shared-grid-off', 'gi-virtual-shared-grid-off-p3-frame-no-bounce.png'],
  ['virtual-shared-grid-voxels', 'gi-virtual-shared-grid-voxels-spheres-and-boxes-same-0-2m-cubes.png'],
];
const ON_FACTS = ['snapshot:settings', 'snapshot:quality', 'snapshot:stats', 'snapshot:instances',
  'runtime:passes', 'runtime:viewSizes', 'runtime:ready',
  'assertion:gi-active', 'assertion:vg-active', 'assertion:virtual-voxelized'];
const OFF_FACTS = ['snapshot:settings', 'snapshot:quality', 'snapshot:stats',
  'runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'assertion:gi-off', 'assertion:vg-active'];
const GRID_ON_FACTS = [...ON_FACTS, 'assertion:forward-voxelized'];
const VOXEL_FACTS = [...GRID_ON_FACTS, 'assertion:debug-view'];
const FACTS = {
  'virtual-field-on': ON_FACTS,
  'virtual-field-off': OFF_FACTS,
  'virtual-shared-grid-on': GRID_ON_FACTS,
  'virtual-shared-grid-off': OFF_FACTS,
  'virtual-shared-grid-voxels': VOXEL_FACTS,
};

const variantOf = id => virtual.default.variants.find(variant => variant.id === id);
const rowOf = id => virtual.inventory.find(row => row.id === id);
const outcome = result => Object.fromEntries(result.assertions.map(a => [a.name, a.passed]));

test('rows: validateInventory returns the five contract ids and filenames on scenario virtual', () => {
  const rows = validateInventory([virtual], 'virtual');
  assert.deepEqual(rows.map(row => [row.id, row.filename]), ROWS);
  for (const row of rows) assert.equal(row.scenario, 'virtual');
  assert.deepEqual(virtual.default.engine, { msaa: 1 });
});

test('pairs: field and shared-grid on/off share a pair and preset; the voxel row has neither', () => {
  for (const [on, off, pair] of [
    ['field-on', 'field-off', 'gi-virtual-field'],
    ['shared-grid-on', 'shared-grid-off', 'gi-virtual-shared-grid'],
  ]) {
    assert.deepEqual(variantOf(on).measurement, {
      pair, state: 'on', preset: pair,
      features: { globalIllumination: true, virtualGeometry: true },
    });
    assert.deepEqual(variantOf(off).measurement, {
      pair, state: 'off', preset: pair,
      features: { globalIllumination: false, virtualGeometry: true },
    });
    assert.deepEqual(virtual.inventory.filter(row => row.pair === pair).map(row => row.mode), [on, off]);
  }
  assert.equal(variantOf('shared-grid-voxels').measurement, undefined);
  assert.equal(rowOf('virtual-shared-grid-voxels').pair, null);
});

test('requiredFacts equal the contract lists; the GI-off rows omit snapshot:instances', () => {
  for (const row of virtual.inventory) assert.deepEqual(row.requiredFacts, FACTS[row.id], row.id);
  for (const id of ['virtual-field-off', 'virtual-shared-grid-off']) {
    assert.ok(rowOf(id).requiredFacts.includes('assertion:gi-off'));
    assert.ok(rowOf(id).requiredFacts.includes('assertion:vg-active'));
    assert.ok(!rowOf(id).requiredFacts.includes('snapshot:instances'));
  }
  assert.ok(rowOf('virtual-shared-grid-on').requiredFacts.includes('assertion:forward-voxelized'));
  assert.ok(rowOf('virtual-shared-grid-voxels').requiredFacts.includes('assertion:debug-view'));
});

test('validateInventory accepts every variant and refuses a field-on camera turned behind the field', () => {
  assert.equal(validateInventory([virtual], 'virtual').length, ROWS.length);
  const turnedAway = variant => variant.id !== 'field-on' ? variant
    : { ...variant, selection: { ...variant.selection, camera: { ...variant.selection.camera, target: [1.5, 1, 40] } } };
  const turned = { ...virtual.default, variants: virtual.default.variants.map(turnedAway) };
  assert.throws(() => validateInventory([{ ...virtual, default: turned }], 'virtual'), /outside camera frustum/);
});

test('the virtual stage clashes with no other stage id or filename', () => {
  const rows = validateInventory([virtual, volume, lighting, temporal, areaShadows, areaStochastic,
    areaRestir, ssgi, giQuality], 'virtual');
  assert.deepEqual(rows.slice(0, ROWS.length).map(row => row.id), ROWS.map(([id]) => id));
  assert.equal(new Set(rows.map(row => row.id)).size, rows.length);
  assert.equal(new Set(rows.map(row => row.filename)).size, rows.length);
});

const INSTANCES = [
  { name: 'ground', instance: 0 }, { name: 'wall', instance: 1 }, { name: 'field-0', instance: 10 },
  { name: 'box-a', instance: 40 }, { name: 'box-b', instance: 41 },
];
const FIELD = Array.from({ length: 16 }, (_, i) => 10 + i);
const BOXES = [40, 41];
const SETTLED = {
  active: true, inactiveReason: null, quality: { resolution: 96, levels: 5, probeRays: 8 },
  finestVoxelMetres: 0.2, voxelizedForwardInstances: 3, voxelizedVirtualInstances: 16,
  pendingRebuild: false, framesSinceRebuild: 240, allocatedBytes: 148635648, statsFrameIndex: 300,
  pending: false,
};
const DISABLED = {
  ...SETTLED, active: false, inactiveReason: 'disabled', voxelizedForwardInstances: 0,
  voxelizedVirtualInstances: 0, framesSinceRebuild: 0, allocatedBytes: 0,
};
const VG = { active: true, virtualInstances: 16 };
// Each stub engine reports the four getters observe reads; drivers override per instance.
function stub({ stats = SETTLED, debugView = 'none', drivers = {} } = {}) {
  const driver = instance => drivers[instance] ?? (FIELD.includes(instance) ? 'virtual' : 'forward');
  return {
    giStats: () => ({ ...stats, quality: { ...stats.quality } }),
    globalIllumination: () => ({ enabled: stats.active, debugView }),
    virtualGeometryStats: () => ({ ...VG }),
    instanceGlobalIllumination: instance => ({ driver: driver(instance) }),
  };
}
const STATE = { instances: INSTANCES, fieldInstances: FIELD, boxInstances: BOXES };
const observed = (engine, id) => virtual.observe(engine, variantOf(id), STATE);

test('observe reports factNames in order; active stats pass every assertion except gi-off', async () => {
  const active = await observed(stub(), 'shared-grid-on');
  assert.deepEqual(active.assertions.map(a => a.name), virtual.factNames);
  assert.deepEqual(active.instances, INSTANCES);
  assert.deepEqual(active.edits, []);
  assert.deepEqual(outcome(active), {
    'gi-active': true, 'gi-off': false, 'vg-active': true,
    'virtual-voxelized': true, 'forward-voxelized': true, 'debug-view': true,
  });
  for (const patch of [{ pendingRebuild: true }, { framesSinceRebuild: 239 }, { allocatedBytes: 148635647 }]) {
    const result = await observed(stub({ stats: { ...SETTLED, ...patch } }), 'field-on');
    assert.equal(outcome(result)['gi-active'], false, JSON.stringify(patch));
  }
});

test('observe fails virtual-voxelized when one field instance reports driver forward', async () => {
  const drivers = { [FIELD[5]]: 'forward' };
  const result = await observed(stub({ drivers }), 'field-on');
  assert.equal(outcome(result)['virtual-voxelized'], false);
});

test('observe passes gi-off only with the disabled stats and no allocation', async () => {
  assert.equal(outcome(await observed(stub({ stats: DISABLED }), 'field-off'))['gi-off'], true);
  for (const patch of [{ allocatedBytes: 1 }, { inactiveReason: 'noInstance' }, { active: true }]) {
    const result = await observed(stub({ stats: { ...DISABLED, ...patch } }), 'field-off');
    assert.equal(outcome(result)['gi-off'], false, JSON.stringify(patch));
  }
});

function readyEngine({ clusters = {}, stats = { active: true, statsFrameIndex: 3 } } = {}) {
  const calls = [];
  return {
    calls,
    meshClusters: mesh => ({ state: 'ready', ...clusters[mesh] }),
    virtualGeometryStats: () => ({ ...stats }),
    setGlobalIllumination: patch => { calls.push(patch); },
  };
}

test('ready: baking and vg-pending are never ready and never apply GI', () => {
  const state = { clustered: [7], giApplied: false };
  const baking = readyEngine({ clusters: { 7: { state: 'baking' } } });
  assert.deepEqual(virtual.default.ready(baking, state, variantOf('field-on')), { ready: false, stage: 'baking' });
  assert.deepEqual(baking.calls, []);
  for (const stats of [{ active: false, statsFrameIndex: null }, { active: true, statsFrameIndex: null }]) {
    const pending = readyEngine({ stats });
    assert.deepEqual(virtual.default.ready(pending, state, variantOf('field-on')), { ready: false, stage: 'vg-pending' });
    assert.deepEqual(pending.calls, []);
  }
});

test('ready: applies variant.gi once when VG is active, then stays ready without applying again', () => {
  const variant = variantOf('field-on');
  const state = { clustered: [7], giApplied: false };
  const engine = readyEngine();
  assert.deepEqual(virtual.default.ready(engine, state, variant), { ready: true, stage: 'vg-active' });
  assert.deepEqual(engine.calls, [variant.gi]);
  assert.equal(state.giApplied, true);
  assert.deepEqual(virtual.default.ready(engine, state, variant), { ready: true, stage: 'vg-active' });
  assert.deepEqual(engine.calls, [variant.gi]);
});

test('ready: a refused cluster bake throws and never reports ready', () => {
  const clusters = { 7: { state: 'refused', code: 'cluster.empty', message: 'no triangles' } };
  const engine = readyEngine({ clusters });
  assert.throws(() => virtual.default.ready(engine, { clustered: [7], giApplied: false }, variantOf('field-on')),
    /virtual: cluster bake refused: cluster\.empty/);
  assert.deepEqual(engine.calls, []);
});
