import test from 'node:test';
import assert from 'node:assert/strict';
import * as temporal from './scenes/temporal.js';
import * as area from './scenes/area-shadows.js';
import * as lighting from './scenes/lighting.js';
import * as volume from './scenes/volume.js';
import { validateInventory } from './lib/inventory.js';

// Temporal anti-aliasing rows over a swaying camera, checked against contract 'gi-taa-rows'. The literals below
// repeat that contract; they are not read back from scenes/temporal.js. Node only: no engine, no WebGPU.

const ON_FACTS = ['snapshot:settings', 'snapshot:stats', 'runtime:ready', 'assertion:gi-active', 'assertion:taa-pass'];
const ORBIT_TARGET = [2.78, 2.73, 0];
const ORBIT_DISTANCE = 7.9;
const SETTLED = {
  active: true, inactiveReason: null, quality: { resolution: 96, levels: 5, probeRays: 8 },
  finestVoxelMetres: 0.2, voxelizedForwardInstances: 8, voxelizedVirtualInstances: 0,
  pendingRebuild: false, framesSinceRebuild: 240, allocatedBytes: 148635648, statsFrameIndex: 300, pending: false,
};
const DISABLED = { ...SETTLED, active: false, inactiveReason: 'disabled', voxelizedForwardInstances: 0, framesSinceRebuild: 0, allocatedBytes: 0 };
const TAA_ROW = { name: 'taa', role: 'pass', enabled: true, order: 100 };
// mode, variant.taa, filename, criterionName, expectedVisibleBehavior
const ROWS = [
  ['taa-off', false, 'gi-taa-off-swaying-camera-jagged-stair-step-edges.png', 'M5 temporal accumulation: TAA off',
    'Temporal anti-aliasing is off and the frame has no multisampling: the outlines of the two white blocks, the edges of the ceiling light fixture, the wall corners and the outline of the orange sphere are jagged, with visible stair-stepped pixel edges along every slanted line.'],
  ['taa-on', true, 'gi-taa-on-swaying-camera-smooth-edges-no-ghost-trail.png', 'M5 temporal accumulation: TAA on',
    'Temporal anti-aliasing is on: the outlines of the two white blocks, the ceiling light fixture, the wall corners and the orange sphere are smooth anti-aliased lines with no visible stair steps; the orange sphere is crisp with no smeared trail, doubled copy or dark ghost outline beside it; walls and blocks stay sharp, not blurred.'],
];

const variantOf = id => temporal.default.variants.find(v => v.id === id);
const named = (calls, name) => calls.filter(call => call[0] === name).map(call => call.slice(1));
const outcome = result => Object.fromEntries(result.assertions.map(a => [a.name, a.passed]));
// A recording engine: every method returns a fresh id and is logged in `calls`.
function recordingEngine() {
  const calls = [];
  let next = 1;
  return new Proxy({}, {
    get: (target, name) => (name === 'calls' ? calls : (...args) => { calls.push([name, ...args]); return next++; }),
  });
}
// A stub engine with the two methods observe reads.
function stub({ stats = SETTLED, passes = [TAA_ROW] } = {}) {
  return { giStats: () => ({ ...stats }), passes: () => passes.map(pass => ({ ...pass })) };
}
const observed = (id, engine) => temporal.observe(engine, variantOf(id), { instances: [], moved: false, blockNode: null });
// The clipmap origin of one eye position at a level (clipmap.rs:72-78), per axis.
const clipmapOrigin = (eye, level) => {
  const voxel = 0.2 * 2 ** level;
  return eye.map(c => Math.floor(Math.floor(c / voxel) / 8) * 8 - 48);
};

test('the two inventory rows equal the contract rows, row by row', () => {
  assert.deepEqual(temporal.inventory.map(row => row.id), ROWS.map(([mode]) => `temporal-${mode}`));
  ROWS.forEach(([mode, , filename, criterionName, behaviour], index) => {
    const row = temporal.inventory[index];
    assert.equal(row.mode, mode);
    assert.equal(row.scenario, 'temporal');
    assert.equal(row.filename, filename);
    assert.equal(row.criterionName, criterionName);
    assert.equal(row.expectedVisibleBehavior, behaviour);
    assert.equal(row.producer, 'raa-taa');
    assert.deepEqual(row.requiredFacts, ON_FACTS);
    assert.equal(row.pair, null);
    assert.equal(row.heldConstant.at(-1), 'orange sphere path along the floor, excluded from GI');
  });
});

test('the variants carry the contract settings, msaa 1, the TAA state and GI on, unmeasured and unpaired', () => {
  ROWS.forEach(([mode, taa]) => {
    const variant = variantOf(mode);
    assert.equal(variant.arrangement, 'cornell-box');
    assert.deepEqual(variant.engine, { msaa: 1 });
    assert.equal(variant.taa, taa);
    assert.deepEqual(variant.gi, { enabled: true, debugView: 'none' });
    assert.deepEqual(variant.quality, { resolution: 96, levels: 5, probeRays: 8 });
    assert.equal(variant.giExpected, 'active');
    assert.deepEqual(variant.areaShadows, { tier: 'hero', heroCount: 1, heroDistance: 25, fadeDistance: 80, band: 5 });
    assert.deepEqual(variant.selection.camera, { eye: [2.78, 2.73, -7.9], target: [2.78, 2.73, 0], fovYDeg: 39.3, near: 0.05, far: 200 });
    assert.deepEqual(variant.selection.subjects[2], { name: 'sliding-sphere', min: [0.75, 0, 0.1], max: [4.75, 0.5, 0.6] });
    assert.deepEqual(variant.references, []);
    assert.equal(variant.judge, true);
    assert.equal(variant.measurement, undefined);
    assert.equal(variant.offMode, undefined);
  });
});

test('validateInventory accepts the scenes and returns rows that include both temporal ids', () => {
  const rows = validateInventory([temporal, area, lighting, volume], 'lighting');
  const ids = rows.map(row => row.id);
  for (const [mode] of ROWS) assert.ok(ids.includes(`temporal-${mode}`), `missing temporal-${mode}`);
});

test('build on a recording engine sets the TAA pass, the orbit, the GI state and the sphere outside GI', async () => {
  for (const [mode, taa] of ROWS) {
    const engine = recordingEngine();
    const state = await temporal.default.build(engine, {}, variantOf(mode));
    const calls = engine.calls;
    assert.deepEqual(named(calls, 'setPassEnabled'), [['taa', taa]], mode);
    assert.deepEqual(named(calls, 'setOrbit'), [[ORBIT_TARGET, ORBIT_DISTANCE, Math.PI, 0]], mode);
    assert.deepEqual(named(calls, 'setGlobalIllumination').at(-1), [{ enabled: true, debugView: 'none' }], mode);
    const fixture = state.instances.find(item => item.name === 'fixture').instance;
    const sphereInstance = state.instances.find(item => item.name === 'sliding-sphere').instance;
    assert.deepEqual(named(calls, 'setInstanceGlobalIllumination'), [[fixture, false], [sphereInstance, false]], mode);
    assert.equal(state.frame, 0);
    assert.equal(state.moved, false);
    assert.equal(state.blockNode, null);
    assert.equal(typeof state.sphereNode, 'number');
  }
});

test('400 frame-hook calls sway within PI +- 0.02, keep the clipmap origin and slide the sphere inside its box', () => {
  const engine = recordingEngine();
  const variant = variantOf('taa-on');
  const state = { instances: [], moved: false, blockNode: null, frame: 0, sphereNode: 7 };
  for (let i = 0; i < 400; i += 1) temporal.default.frame(engine, 0, variant, state);
  assert.equal(state.frame, 400);
  const orbits = named(engine.calls, 'orbit');
  assert.equal(orbits.length, 400);
  let yaw = Math.PI;
  const base = [2.78 + 7.9 * Math.sin(Math.PI), 2.73, 7.9 * Math.cos(Math.PI)];
  const baseOrigins = [0, 1, 2, 3, 4].map(level => clipmapOrigin(base, level));
  for (const [dYaw, dPitch, zoom] of orbits) {
    assert.ok(dYaw === 0.002 || dYaw === -0.002, `yaw step ${dYaw}`);
    assert.equal(dPitch, 0);
    assert.equal(zoom, 1);
    yaw += dYaw;
    assert.ok(Math.abs(yaw - Math.PI) <= 0.02 + 1e-9, `yaw ${yaw} leaves PI +- 0.02`);
    const eye = [ORBIT_TARGET[0] + ORBIT_DISTANCE * Math.sin(yaw), ORBIT_TARGET[1], ORBIT_TARGET[2] + ORBIT_DISTANCE * Math.cos(yaw)];
    [0, 1, 2, 3, 4].forEach(level => assert.deepEqual(clipmapOrigin(eye, level), baseOrigins[level], `level ${level} origin moved at yaw ${yaw}`));
  }
  const transforms = named(engine.calls, 'setTransforms');
  assert.equal(transforms.length, 400);
  for (const [nodes, values] of transforms) {
    assert.deepEqual(Array.from(nodes), [7]);
    assert.ok(values[0] >= 1.0 && values[0] <= 4.5, `sphere x ${values[0]} leaves its travel box`);
    assert.deepEqual(Array.from(values.slice(1, 3)), [0.25, Math.fround(0.35)]);
  }
});

test('observe passes taa-pass when the taa row matches, and fails when enabled differs or the row is missing', async () => {
  const on = await observed('taa-on', stub());
  assert.deepEqual(on.assertions.map(a => a.name), temporal.factNames);
  assert.deepEqual(temporal.factNames, ['gi-active', 'taa-pass']);
  assert.equal(outcome(on)['taa-pass'], true);
  assert.match(on.assertions[1].detail, /taa rows 1 .*enabled true .*order 100/);
  assert.deepEqual(on.edits, []);
  const off = outcome(await observed('taa-off', stub({ passes: [{ ...TAA_ROW, enabled: false }] })));
  assert.equal(off['taa-pass'], true);
  const wrong = outcome(await observed('taa-on', stub({ passes: [{ ...TAA_ROW, enabled: false }] })));
  assert.equal(wrong['taa-pass'], false);
  const missing = outcome(await observed('taa-on', stub({ passes: [] })));
  assert.equal(missing['taa-pass'], false);
});

test('observe reports gi-active by the rule of the settled and disabled stats', async () => {
  const on = outcome(await observed('taa-on', stub()));
  assert.equal(on['gi-active'], true);
  const starved = outcome(await observed('taa-on', stub({ stats: { ...SETTLED, allocatedBytes: 1 } })));
  assert.equal(starved['gi-active'], false);
  const unsettled = outcome(await observed('taa-on', stub({ stats: { ...SETTLED, framesSinceRebuild: 10 } })));
  assert.equal(unsettled['gi-active'], false);
  const off = outcome(await observed('taa-off', stub({ stats: DISABLED })));
  assert.equal(off['gi-active'], false);
});

test('ready returns the lighting stage with ready true', () => {
  assert.deepEqual(temporal.default.ready({}, { instances: [], moved: false, blockNode: null }, variantOf('taa-on')), { ready: true, stage: 'lighting' });
});
