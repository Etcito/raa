import test from 'node:test';
import assert from 'node:assert/strict';
import * as stochastic from './scenes/area-stochastic.js';
import * as area from './scenes/area-shadows.js';
import * as lighting from './scenes/lighting.js';
import * as volume from './scenes/volume.js';
import * as temporal from './scenes/temporal.js';
import { validateInventory } from './lib/inventory.js';

// Mid-tier stochastic-shadow rows, checked against contract 'gi-area-stochastic-rows'. The literals below
// repeat that contract; they are not read back from scenes/area-stochastic.js. Node only: no engine, no WebGPU.

const CORNELL_HELD = ['Cornell box geometry and wall colours', 'single ceiling area light and its fixture', 'camera', 'exposure'];
const BASE_FACTS = ['snapshot:settings', 'snapshot:stats', 'runtime:ready', 'assertion:gi-active', 'assertion:area-stochastic', 'assertion:area-shadow-pass'];
const SDF_FACTS = [...BASE_FACTS, 'assertion:sdf-content'];
const MOVING_HELD = 'camera swaying by at most 0.02 rad about the box centre';
const SDF_HELD = 'an SDF sphere at the front right and an SDF box at the back left, drawn by the SDF trace pass with their own shading';
const STILL_BEHAVIOUR = 'The two white blocks cast soft shadows of the ceiling area light on the floor and on the red wall. Each block casts ONE continuous shadow whose penumbra is a single smooth gradient, sharp where the shadow meets the block base and wider and softer with distance from the block, like the shadows in the reference; never four (or nine) separate overlapping copies of the shadow, no banding or stepped edges. The penumbrae are clean: no grain, speckles or blotchy noise. Bounce light lights the ceiling and the red and green walls tint the nearby white surfaces. Brightness differences from the path-traced reference are not faults.';
const MOVING_BEHAVIOUR = 'The camera is swaying slightly. The shadows of the two blocks are still single smooth penumbrae, sharp at the block bases and widening with distance; there is no grain, no speckle, no shimmering blotches and no smeared, doubled or trailing ghost copy of any shadow edge; walls and blocks stay sharp. Brightness differences from the path-traced reference are not faults.';
const SDF_BEHAVIOUR = 'Besides the two white blocks, a sphere stands on the floor at the front right and a box at the back left. The sphere and the box each cast a soft shadow of the ceiling area light onto the floor: one smooth penumbra per object, darkest at the object base and widening away from it, with no grain or noise, like the block shadows. The colour or brightness of the sphere and box themselves is not judged. Brightness differences from the path-traced reference are not faults.';
// mode, stochasticCells, motion, sdf, filename, criterionName, expectedVisibleBehavior
const ROWS = [
  ['stochastic-2x2-still', 4, false, false, 'gi-area-stochastic-2x2-still-one-smooth-rectangular-penumbra.png', 'M6 area light stochastic shadows: 2x2 cells, still camera', STILL_BEHAVIOUR],
  ['stochastic-3x3-still', 9, false, false, 'gi-area-stochastic-3x3-still-one-smooth-rectangular-penumbra.png', 'M6 area light stochastic shadows: 3x3 cells, still camera', STILL_BEHAVIOUR],
  ['stochastic-2x2-moving', 4, true, false, 'gi-area-stochastic-2x2-moving-camera-no-crawling-noise-in-penumbra.png', 'M6 area light stochastic shadows: 2x2 cells, moving camera', MOVING_BEHAVIOUR],
  ['stochastic-sdf-occluder', 4, false, true, 'gi-area-stochastic-sdf-sphere-and-box-cast-soft-penumbrae-on-floor.png', 'M6 area light stochastic shadows: SDF occluders', SDF_BEHAVIOUR],
];
const SDF_SUBJECTS = [
  { name: 'sdf-sphere', min: [3.65, 0, 0.65], max: [4.75, 1.1, 1.75] },
  { name: 'sdf-box', min: [0.75, 0, 3.85], max: [1.65, 1.0, 4.75] },
];
const SETTLED = {
  active: true, inactiveReason: null, quality: { resolution: 96, levels: 5, probeRays: 8 },
  finestVoxelMetres: 0.2, voxelizedForwardInstances: 8, voxelizedVirtualInstances: 0,
  pendingRebuild: false, framesSinceRebuild: 240, allocatedBytes: 148635648, statsFrameIndex: 300, pending: false,
};
const AREA = { tier: 'auto', heroCount: 0, stochasticCount: 1, stochasticCells: 4, activeStochastic: 1 };
const PASS = { name: 'gi.area-shadow', role: 'pass', enabled: true, order: 43, required: false };

const variantOf = id => stochastic.default.variants.find(v => v.id === id);
const named = (calls, name) => calls.filter(call => call[0] === name).map(call => call.slice(1));
const outcome = result => Object.fromEntries(result.assertions.map(a => [a.name, a.passed]));
const shadowOf = cells => ({
  tier: 'auto', heroCount: 0, heroDistance: 25, fadeDistance: 80, band: 5,
  stochasticCount: 1, stochasticDistance: 40, stochasticCells: cells,
});
const heldOf = (cells, motion, sdf) => {
  const held = [...CORNELL_HELD, `area-light shadow tier auto with stochasticCount 1 and stochasticCells ${cells}`];
  if (motion) held.push(MOVING_HELD);
  if (sdf) held.push(SDF_HELD);
  return held;
};
// A recording engine: every method returns a fresh id and is logged in `calls`.
function recordingEngine() {
  const calls = [];
  let next = 1;
  return new Proxy({}, {
    get: (target, name) => (name === 'calls' ? calls : (...args) => { calls.push([name, ...args]); return next++; }),
  });
}
// A stub engine with the four methods observe reads.
function stub({ stats = SETTLED, areaShadows = AREA, passes = [PASS], trace = { instances: 2 } } = {}) {
  return {
    giStats: () => ({ ...stats }),
    areaLightShadows: () => ({ ...areaShadows }),
    passes: () => passes.map(row => ({ ...row })),
    traceStats: () => ({ ...trace }),
  };
}
const observed = (id, engine) => stochastic.observe(engine, variantOf(id), { instances: [], moved: false, blockNode: null, frame: 0 });

test('the four inventory rows equal the contract rows, row by row', () => {
  assert.deepEqual(stochastic.inventory.map(row => row.id), ROWS.map(([mode]) => `area-stochastic-${mode}`));
  ROWS.forEach(([mode, cells, motion, sdf, filename, criterionName, behaviour], index) => {
    const row = stochastic.inventory[index];
    assert.equal(row.mode, mode);
    assert.equal(row.scenario, 'area-stochastic');
    assert.equal(row.filename, filename);
    assert.equal(row.criterionName, criterionName);
    assert.equal(row.expectedVisibleBehavior, behaviour);
    assert.deepEqual(row.requiredFacts, sdf ? SDF_FACTS : BASE_FACTS);
    assert.equal(row.pair, null);
    assert.equal(row.producer, 'raa-gi-area-stochastic');
    assert.equal(row.judgeReference, '/refs/cornell-box-reference.png');
    assert.equal(row.judgeAspect, 'functional');
    assert.deepEqual(row.heldConstant, heldOf(cells, motion, sdf));
  });
});

test('the variants carry the contract settings, an active mid tier and no measurement', () => {
  ROWS.forEach(([mode, cells, motion, sdf]) => {
    const variant = variantOf(mode);
    assert.equal(variant.arrangement, 'cornell-box');
    assert.deepEqual(variant.gi, { enabled: true, debugView: 'none' });
    assert.deepEqual(variant.quality, { resolution: 96, levels: 5, probeRays: 8 });
    assert.equal(variant.giExpected, 'active');
    assert.deepEqual(variant.areaShadows, shadowOf(cells));
    assert.equal(variant.motion, motion);
    assert.equal(variant.sdf, sdf);
    assert.equal(variant.judge, true);
    assert.equal(variant.measurement, undefined);
    assert.deepEqual(variant.selection.camera, lighting.CORNELL_SELECTION.camera);
    assert.deepEqual(variant.selection.subjects.slice(0, 2), lighting.CORNELL_SELECTION.subjects);
    assert.deepEqual(variant.selection.subjects.slice(2), sdf ? SDF_SUBJECTS : []);
  });
});

test('validateInventory accepts the lighting scenes and returns rows that include the four ids', () => {
  const rows = validateInventory([stochastic, area, lighting, volume, temporal], 'lighting');
  const ids = rows.map(row => row.id);
  for (const [mode] of ROWS) assert.ok(ids.includes(`area-stochastic-${mode}`), `missing area-stochastic-${mode}`);
});

test('build on a recording engine sets the mid tier, the trace scene and the orbit only on their rows', async () => {
  for (const [mode, cells, motion, sdf] of ROWS) {
    const engine = recordingEngine();
    const variant = variantOf(mode);
    const state = await stochastic.build(engine, {}, variant);
    const calls = engine.calls;
    assert.deepEqual(named(calls, 'setGiQuality'), [[{ resolution: 96, levels: 5, probeRays: 8 }]]);
    assert.deepEqual(named(calls, 'setAreaLightShadows'), [[shadowOf(cells)]], mode);
    assert.deepEqual(named(calls, 'setCamera'), [[variant.selection.camera]], mode);
    assert.deepEqual(named(calls, 'setGlobalIllumination').at(-1), [{ enabled: true, debugView: 'none' }], mode);
    const traces = named(calls, 'setTraceScene');
    if (sdf) {
      assert.equal(traces.length, 1, mode);
      const bytes = traces[0][0];
      assert.ok(bytes instanceof Uint8Array, mode);
      assert.equal(new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(0, true), 0x50544453, mode);
    } else assert.equal(traces.length, 0, mode);
    const orbits = named(calls, 'setOrbit');
    if (motion) assert.deepEqual(orbits, [[[2.78, 2.73, 0], 8, Math.PI, 0]], mode);
    else assert.equal(orbits.length, 0, mode);
    const lights = named(calls, 'addLight');
    assert.equal(lights.length, 1, mode);
    assert.equal(lights[0][0].kind, 'area', mode);
    assert.equal(state.moved, false, mode);
    assert.equal(state.blockNode, null, mode);
    assert.equal(state.frame, 0, mode);
    assert.ok(state.instances.some(item => item.name === 'cornell-white'), mode);
  }
});

test('frame calls orbit only on the moving row and keeps the accumulated yaw within one snap cell', () => {
  const still = recordingEngine();
  const stillState = { instances: [], moved: false, blockNode: null, frame: 0 };
  for (let index = 0; index < 5; index += 1) stochastic.frame(still, 0, variantOf('stochastic-2x2-still'), stillState);
  assert.equal(named(still.calls, 'orbit').length, 0);
  const engine = recordingEngine();
  const state = { instances: [], moved: false, blockNode: null, frame: 0 };
  for (let index = 0; index < 400; index += 1) stochastic.frame(engine, 0, variantOf('stochastic-2x2-moving'), state);
  const orbits = named(engine.calls, 'orbit');
  assert.equal(orbits.length, 400);
  assert.equal(state.frame, 400);
  let yaw = 0;
  for (const [yawStep, pitchStep, rollStep] of orbits) {
    assert.deepEqual([pitchStep, rollStep], [0, 1]);
    assert.ok(Math.abs(yawStep) === 0.002);
    yaw += yawStep;
    assert.ok(yaw >= -0.02 - 1e-9 && yaw <= 0.02 + 1e-9, `yaw ${yaw}`);
  }
  assert.ok(Math.abs(yaw) < 1e-9, `final yaw ${yaw}`);
});

test('observe passes the four assertions on the SDF row and each one fails on its own defect', async () => {
  const result = await observed('stochastic-sdf-occluder', stub());
  assert.deepEqual(result.assertions.map(assertionRow => assertionRow.name), stochastic.factNames);
  assert.deepEqual(result.assertions.map(assertionRow => assertionRow.name), ['gi-active', 'area-stochastic', 'area-shadow-pass', 'sdf-content']);
  assert.deepEqual(Object.values(outcome(result)), [true, true, true, true]);
  assert.deepEqual(result.edits, []);
  assert.match(result.assertions[1].detail, /tier auto .*heroCount 0 .*stochasticCount 1 .*stochasticCells 4 .*activeStochastic 1/);
  const zero = outcome(await observed('stochastic-2x2-still', stub({ areaShadows: { ...AREA, activeStochastic: 0 } })));
  assert.equal(zero['area-stochastic'], false);
  const ordered = outcome(await observed('stochastic-2x2-still', stub({ passes: [{ ...PASS, order: 42 }] })));
  assert.equal(ordered['area-shadow-pass'], false);
  const disabled = outcome(await observed('stochastic-2x2-still', stub({ passes: [{ ...PASS, enabled: false }] })));
  assert.equal(disabled['area-shadow-pass'], false);
  const content = outcome(await observed('stochastic-2x2-still', stub({ trace: { instances: 2 } })));
  assert.equal(content['sdf-content'], false);
  const still = outcome(await observed('stochastic-2x2-still', stub({ trace: { instances: 0 } })));
  assert.equal(still['sdf-content'], true);
});

test('ready returns the lighting stage with ready true', () => {
  assert.deepEqual(stochastic.default.ready({}, { instances: [], moved: false, blockNode: null, frame: 0 }, variantOf('stochastic-2x2-still')), { ready: true, stage: 'lighting' });
});
