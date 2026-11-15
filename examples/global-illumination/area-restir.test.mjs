import test from 'node:test';
import assert from 'node:assert/strict';
import * as restir from './scenes/area-restir.js';
import * as stochastic from './scenes/area-stochastic.js';
import * as area from './scenes/area-shadows.js';
import * as lighting from './scenes/lighting.js';
import * as volume from './scenes/volume.js';
import * as temporal from './scenes/temporal.js';
import { validateInventory } from './lib/inventory.js';

// ReSTIR ultra-tier rows, checked against contract 'gi-area-restir-rows'. The literals below repeat that contract;
// they are not read back from scenes/area-restir.js. Node only: no engine, no WebGPU.

const CORNELL_HELD = ['Cornell box geometry and wall colours', 'single ceiling area light and its fixture', 'camera', 'exposure'];
const BASE_FACTS = ['snapshot:settings', 'snapshot:stats', 'runtime:ready', 'assertion:gi-active', 'assertion:area-restir', 'assertion:area-shadow-pass'];
const SDF_FACTS = [...BASE_FACTS, 'assertion:sdf-content'];
const MOVING_HELD = 'camera swaying by at most 0.02 rad about the box centre';
const SDF_HELD = 'an SDF sphere at the front right and an SDF box at the back left, drawn by the SDF trace pass with their own shading';
const STILL_BEHAVIOUR = 'The two white blocks cast soft shadows of the ceiling area light on the floor and on the red wall. Each block casts ONE continuous shadow whose penumbra is a single smooth gradient, sharp where the shadow meets the block base and wider and softer with distance from the block, like the shadows in the reference. The penumbrae and the lit surfaces are clean: no grain, no speckles, no blotchy or splotchy correlated patches, no bright fireflies and no banding or stepped edges. Bounce light lights the ceiling. Brightness and colour-bleed differences from the path-traced reference are not faults.';
const MOVING_BEHAVIOUR = 'The camera is swaying slightly. The shadows of the two blocks stay single smooth penumbrae, sharp at the block bases and widening with distance; there is no boiling (no flickering bright or dark blotches), no grain or speckle, no blotchy correlated patches and no smeared, doubled or trailing ghost copy of any shadow edge; walls and blocks stay sharp. Brightness and colour-bleed differences from the path-traced reference are not faults.';
const SDF_BEHAVIOUR = 'Besides the two white blocks, a sphere stands on the floor at the front right and a box at the back left. The sphere and the box each cast a soft shadow of the ceiling area light onto the floor: one smooth penumbra per object, darkest at the object base and widening away from it, with no grain, blotches or boiling noise, like the block shadows. The colour or brightness of the sphere and box themselves is not judged. Brightness and colour-bleed differences from the path-traced reference are not faults.';
// mode, motion, sdf, filename, criterionName, expectedVisibleBehavior
const ROWS = [
  ['restir-still', false, false, 'gi-area-restir-still-converged-one-soft-rectangular-penumbra-no-blotches.png', 'M7 ReSTIR area light shadows: still camera after convergence', STILL_BEHAVIOUR],
  ['restir-moving', true, false, 'gi-area-restir-moving-camera-no-boiling-or-blotchy-noise-in-penumbra.png', 'M7 ReSTIR area light shadows: moving camera', MOVING_BEHAVIOUR],
  ['restir-sdf-occluder', false, true, 'gi-area-restir-sdf-sphere-and-box-cast-soft-penumbrae-on-floor.png', 'M7 ReSTIR area light shadows: SDF occluders', SDF_BEHAVIOUR],
];
const DESCRIPTIONS = {
  'restir-still': 'The Cornell box with the ReSTIR ultra-tier area-light shadows and a still camera after convergence.',
  'restir-moving': 'The Cornell box with the ReSTIR ultra-tier area-light shadows while the camera sways slightly.',
  'restir-sdf-occluder': 'The Cornell box with the ReSTIR ultra tier and an SDF sphere and box in front of the ceiling area light.',
};
const SDF_SUBJECTS = [
  { name: 'sdf-sphere', min: [3.65, 0, 0.65], max: [4.75, 1.1, 1.75] },
  { name: 'sdf-box', min: [0.75, 0, 3.85], max: [1.65, 1.0, 4.75] },
];
const SETTLED = {
  active: true, inactiveReason: null, quality: { resolution: 96, levels: 5, probeRays: 8 },
  finestVoxelMetres: 0.2, voxelizedForwardInstances: 8, voxelizedVirtualInstances: 0,
  pendingRebuild: false, framesSinceRebuild: 240, allocatedBytes: 148635648, statsFrameIndex: 300, pending: false,
};
const AREA = { tier: 'auto', heroCount: 0, stochasticCount: 1, stochasticMode: 'restir', activeStochastic: 1 };
const PASS = { name: 'gi.area-shadow', role: 'pass', enabled: true, order: 43, required: false };

const variantOf = id => restir.default.variants.find(v => v.id === id);
const named = (calls, name) => calls.filter(call => call[0] === name).map(call => call.slice(1));
const outcome = result => Object.fromEntries(result.assertions.map(a => [a.name, a.passed]));
const shadowOf = () => ({
  tier: 'auto', heroCount: 0, heroDistance: 25, fadeDistance: 80, band: 5,
  stochasticCount: 1, stochasticDistance: 40, stochasticCells: 4, stochasticMode: 'restir',
});
const heldOf = (motion, sdf) => {
  const held = [...CORNELL_HELD, 'area-light shadow tier auto with stochasticCount 1 and stochasticMode restir (ultra tier)'];
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
const observed = (id, engine) => restir.observe(engine, variantOf(id), { instances: [], moved: false, blockNode: null, frame: 0 });

test('the three inventory rows equal the contract rows, field by field', () => {
  assert.deepEqual(restir.inventory.map(row => row.id), ROWS.map(([mode]) => `area-restir-${mode}`));
  ROWS.forEach(([mode, motion, sdf, filename, criterionName, behaviour], index) => {
    const row = restir.inventory[index];
    assert.equal(row.mode, mode);
    assert.equal(row.scenario, 'area-restir');
    assert.equal(row.filename, filename);
    assert.equal(row.criterionName, criterionName);
    assert.equal(row.expectedVisibleBehavior, behaviour);
    assert.deepEqual(row.requiredFacts, sdf ? SDF_FACTS : BASE_FACTS);
    assert.equal(row.pair, null);
    assert.equal(row.producer, 'raa-gi-area-restir');
    assert.equal(row.judgeReference, '/refs/cornell-box-reference.png');
    assert.equal(row.judgeAspect, 'functional');
    assert.deepEqual(row.heldConstant, heldOf(motion, sdf));
  });
});

test('the variants carry the contract settings, the ReSTIR mode and no measurement', () => {
  ROWS.forEach(([mode, motion, sdf]) => {
    const variant = variantOf(mode);
    assert.equal(variant.description, DESCRIPTIONS[mode]);
    assert.equal(variant.arrangement, 'cornell-box');
    assert.deepEqual(variant.gi, { enabled: true, debugView: 'none' });
    assert.deepEqual(variant.quality, { resolution: 96, levels: 5, probeRays: 8 });
    assert.equal(variant.giExpected, 'active');
    assert.deepEqual(variant.areaShadows, shadowOf());
    assert.equal(variant.motion, motion);
    assert.equal(variant.sdf, sdf);
    assert.equal(variant.judge, true);
    assert.equal(variant.measurement, undefined);
    assert.deepEqual(variant.selection.camera, lighting.CORNELL_SELECTION.camera);
    assert.deepEqual(variant.selection.subjects, sdf ? [...lighting.CORNELL_SELECTION.subjects, ...SDF_SUBJECTS] : lighting.CORNELL_SELECTION.subjects);
  });
});

test('validateInventory accepts the lighting scenes and returns rows that include the three ids', () => {
  const rows = validateInventory([restir, stochastic, area, lighting, volume, temporal], 'lighting');
  const ids = rows.map(row => row.id);
  for (const [mode] of ROWS) assert.ok(ids.includes(`area-restir-${mode}`), `missing area-restir-${mode}`);
});

test('build on a recording engine sets the ReSTIR shadows once, the trace scene on the SDF row and the orbit on the moving row', async () => {
  for (const [mode, motion, sdf] of ROWS) {
    const engine = recordingEngine();
    const variant = variantOf(mode);
    const state = await restir.default.build(engine, {}, variant);
    const calls = engine.calls;
    assert.deepEqual(named(calls, 'setAreaLightShadows'), [[shadowOf()]], mode);
    assert.deepEqual(named(calls, 'setCamera'), [[variant.selection.camera]], mode);
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
    assert.equal(state.moved, false, mode);
    assert.ok(state.instances.some(item => item.name === 'cornell-white'), mode);
  }
});

test('observe passes the four assertions on the SDF row and each defect fails its own assertion', async () => {
  const result = await observed('restir-sdf-occluder', stub());
  assert.deepEqual(result.assertions.map(assertionRow => assertionRow.name), restir.factNames);
  assert.deepEqual(restir.factNames, ['gi-active', 'area-restir', 'area-shadow-pass', 'sdf-content']);
  assert.deepEqual(Object.values(outcome(result)), [true, true, true, true]);
  assert.deepEqual(result.edits, []);
  assert.match(result.assertions[1].detail, /tier auto .*heroCount 0 .*stochasticCount 1 .*stochasticMode restir .*activeStochastic 1/);
  const stratified = outcome(await observed('restir-still', stub({ areaShadows: { ...AREA, stochasticMode: 'stratified' } })));
  assert.equal(stratified['area-restir'], false);
  const zero = outcome(await observed('restir-still', stub({ areaShadows: { ...AREA, activeStochastic: 0 } })));
  assert.equal(zero['area-restir'], false);
  const ordered = outcome(await observed('restir-still', stub({ passes: [{ ...PASS, order: 42 }] })));
  assert.equal(ordered['area-shadow-pass'], false);
  const content = outcome(await observed('restir-still', stub({ trace: { instances: 2 } })));
  assert.equal(content['sdf-content'], false);
  const still = outcome(await observed('restir-still', stub({ trace: { instances: 0 } })));
  assert.equal(still['sdf-content'], true);
});

test('the default export is the ReSTIR scene in the lighting stage, with the imported build and frame', () => {
  assert.equal(restir.default.id, 'area-restir');
  assert.equal(restir.default.milestone, 'lighting');
  assert.equal(restir.default.build, stochastic.build);
  assert.equal(restir.default.frame, stochastic.frame);
  assert.deepEqual(restir.default.ready(), { ready: true, stage: 'lighting' });
});
