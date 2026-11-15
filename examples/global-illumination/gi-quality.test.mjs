import test from 'node:test';
import assert from 'node:assert/strict';
import * as quality from './scenes/gi-quality.js';
import * as areaStochastic from './scenes/area-stochastic.js';
import * as lighting from './scenes/lighting.js';
import { validateInventory } from './lib/inventory.js';

// GI quality rows, checked against contract 'gi-quality-rows'. The literals below repeat that contract; they are
// not read back from scenes/gi-quality.js. Node only: no engine, no WebGPU.

const CORNELL_HELD = ['Cornell box geometry and wall colours', 'single ceiling area light and its fixture', 'camera', 'exposure'];
const SHADOW_BASE = { tier: 'auto', heroCount: 0, heroDistance: 25, fadeDistance: 80, band: 5, stochasticDistance: 40, stochasticCells: 4 };
const ALL_ON_SHADOWS = { ...SHADOW_BASE, stochasticCount: 1, stochasticMode: 'restir' };
const CONE_SHADOWS = { ...SHADOW_BASE, stochasticCount: 0 };
const VOLUME_HELD = 'global illumination volume and screen probes at quality 96/5';
const ALL_ON = { enabled: true, debugView: 'none', screenSpace: true, screenSpaceQuality: 'high' };
// mode, filename, criterionName, gi, areaShadows, giExpected, requiredFacts, held, producer, judgeAspect, behaviour
const ROWS = [
  { mode: 'cornell-all-on', filename: 'gi-quality-cornell-box-all-parts-on-bright-bounce-and-strong-colour-bleed.png',
    criterionName: 'M9 GI quality: Cornell box with every lighting part on against the path-traced reference',
    gi: ALL_ON, areaShadows: ALL_ON_SHADOWS, giExpected: 'active', producer: 'raa-gi-screen', judgeAspect: 'look',
    requiredFacts: ['snapshot:settings', 'snapshot:stats', 'runtime:ready', 'assertion:gi-active', 'assertion:all-parts'],
    held: [...CORNELL_HELD, VOLUME_HELD, 'screen-space indirect light quality high',
      'area-light shadow tier auto with stochasticCount 1 and stochasticMode restir (ultra tier)'],
    behaviour: 'Every lighting part is on: the ceiling area light with its soft shadows, global illumination and screen-space indirect light. Compared with the path-traced reference: the ceiling is clearly lit by warm bounce light, a light warm orange like the reference, not a dark brown; the floor is the brightest large surface, a light warm cream; the front faces of the two white blocks, which the light does not reach directly, are clearly lit by warm bounce light from the floor (the short block front is a visible warm tan, not black) and stay crisp, not smeared; the tall block side facing the red wall is strongly tinted red and the short block side facing the green wall clearly green, and the floor and ceiling near each wall take the same tint, about as strongly as in the reference; the back wall is evenly lit with no dark lobe or blotch beside the tall block other than the soft shadow the block itself casts; each block casts one soft shadow whose penumbra widens with distance, with no staircase or voxel-stepped edges, grain or blotches. Small differences in fine shadow shape and slight overall brightness differences are not faults.' },
  { mode: 'cornell-all-off', filename: 'gi-quality-cornell-box-all-parts-off-direct-light-only.png',
    criterionName: 'M9 GI quality: Cornell box with GI off, direct light only',
    gi: { enabled: false, debugView: 'none' }, areaShadows: ALL_ON_SHADOWS, giExpected: 'disabled',
    producer: 'raa-area-ltc', judgeAspect: 'functional',
    requiredFacts: ['snapshot:settings', 'snapshot:stats', 'runtime:ready', 'assertion:gi-off'],
    held: [...CORNELL_HELD, 'global illumination disabled'],
    behaviour: 'Global illumination is off; the same box, area light and exposure. Compared with the reference only for how the direct light of the ceiling light is distributed: the floor under the light is the brightest region and stays well lit towards the front of the box, the back wall is evenly lit in its middle and falls off towards its top corners, and the tops of the two blocks are bright; the ceiling and the faces turned away from the light stay dark and there is no red or green colour bleeding. Differences from the path-traced reference in overall brightness, missing bounce light, shadows or colour bleeding are not faults here.' },
  { mode: 'cornell-cone-tier-on', filename: 'gi-quality-cornell-box-cone-tier-soft-block-shadows-no-voxel-steps.png',
    criterionName: 'M9 GI quality: far cone shadow tier soft, without voxel steps',
    gi: { enabled: true, debugView: 'none' }, areaShadows: CONE_SHADOWS, giExpected: 'active',
    producer: 'raa-shader-gi', judgeAspect: 'functional',
    requiredFacts: ['snapshot:settings', 'snapshot:stats', 'runtime:ready', 'assertion:gi-active', 'assertion:cone-tier'],
    held: [...CORNELL_HELD, VOLUME_HELD, 'area-light shadow tier auto with stochasticCount 0 (far cone tier only)'],
    behaviour: 'Only the far cone shadow tier shades the area light: each white block casts one soft shadow of the ceiling light on the floor, darkest at the block base and widening into a smooth gradient with distance; the shadow edges are smooth curves with no staircase, voxel-sized steps or blocky contours, and the open floor and walls away from the blocks show no dark blotches or contour bands. Bounce light lights the ceiling. Brightness and colour-bleed differences from the path-traced reference are not faults.' },
];
const DESCRIPTIONS = {
  'cornell-all-on': 'The Cornell box with every lighting part on: the area light, the soft shadow tiers, ReSTIR, SSGI and the voxel GI.',
  'cornell-all-off': 'The same Cornell box with every lighting part off: the direct light of the ceiling area light only.',
  'cornell-cone-tier-on': 'The Cornell box with the far cone shadow tier alone over the voxel global illumination.',
};
const SETTLED = {
  active: true, inactiveReason: null, quality: { resolution: 96, levels: 5, probeRays: 8 },
  finestVoxelMetres: 0.2, voxelizedForwardInstances: 8, voxelizedVirtualInstances: 0,
  pendingRebuild: false, framesSinceRebuild: 240, allocatedBytes: 148635648, statsFrameIndex: 300, pending: false,
};
const AREA_ON = { tier: 'auto', stochasticMode: 'restir', stochasticCount: 1, activeStochastic: 1 };
const SSGI_PASS = { name: 'gi.ssgi', role: 'pass', enabled: true, order: 44, required: false };
const SHADOW_PASS = { name: 'gi.area-shadow', role: 'pass', enabled: true, order: 43, required: false };

const variantOf = id => quality.default.variants.find(variant => variant.id === id);
const named = (calls, name) => calls.filter(call => call[0] === name).map(call => call.slice(1));
// A recording engine: every method returns a fresh id and is logged in `calls`.
function recordingEngine() {
  const calls = [];
  let next = 1;
  return new Proxy({}, {
    get: (target, name) => (name === 'calls' ? calls : (...args) => { calls.push([name, ...args]); return next++; }),
  });
}
// A stub engine with the four getters observe reads.
function stub({ stats = SETTLED, settings = { screenSpace: true, screenSpaceQuality: 'high' }, area = AREA_ON, passes = [SSGI_PASS, SHADOW_PASS] } = {}) {
  return {
    giStats: () => ({ ...stats }),
    globalIllumination: () => ({ ...settings }),
    areaLightShadows: () => ({ ...area }),
    passes: () => passes.map(row => ({ ...row })),
  };
}
const observed = (engine, mode = 'cornell-all-on') => quality.observe(engine, variantOf(mode), { instances: [] });
const passesFact = async (name, engine) => (await observed(engine)).assertions.find(item => item.name === name).passed;

test('the three inventory rows equal the contract rows, field by field', () => {
  assert.deepEqual(quality.inventory.map(row => row.id), ROWS.map(row => `gi-quality-${row.mode}`));
  ROWS.forEach((wanted, index) => {
    const row = quality.inventory[index];
    assert.equal(row.scenario, 'gi-quality');
    assert.equal(row.mode, wanted.mode);
    assert.equal(row.filename, wanted.filename);
    assert.equal(row.criterionName, wanted.criterionName);
    assert.equal(row.expectedVisibleBehavior, wanted.behaviour);
    assert.deepEqual(row.requiredFacts, wanted.requiredFacts);
    assert.equal(row.pair, null);
    assert.deepEqual(row.heldConstant, wanted.held);
    assert.equal(row.producer, wanted.producer);
    assert.equal(row.judgeReference, '/refs/cornell-box-reference.png');
    assert.equal(row.judgeAspect, wanted.judgeAspect);
  });
});

test('the variants carry the contract gi, areaShadows, giExpected, quality and Cornell selection', () => {
  ROWS.forEach(wanted => {
    const variant = variantOf(wanted.mode);
    assert.ok(variant, wanted.mode);
    assert.equal(variant.description, DESCRIPTIONS[wanted.mode]);
    assert.deepEqual(variant.gi, wanted.gi);
    assert.deepEqual(variant.areaShadows, wanted.areaShadows);
    assert.equal(variant.giExpected, wanted.giExpected);
    assert.deepEqual(variant.quality, { resolution: 96, levels: 5, probeRays: 8 });
    assert.deepEqual(variant.selection, lighting.CORNELL_SELECTION);
    assert.equal(variant.arrangement, 'cornell-box');
    assert.equal(variant.judge, true);
    assert.deepEqual(variant.references, []);
    assert.equal(variant.motion, false);
    assert.equal(variant.sdf, false);
    assert.equal(variant.measurement, undefined);
  });
});

test('validateInventory accepts the gi-quality scene in the lighting stage and returns its three rows', () => {
  const rows = validateInventory([quality], 'lighting');
  assert.deepEqual(rows.map(row => row.id), quality.inventory.map(row => row.id));
});

test('build sets the 96/5 quality, the row areaShadows and the row GI through the area-stochastic build', async () => {
  for (const wanted of ROWS) {
    const engine = recordingEngine();
    const state = await quality.default.build(engine, {}, variantOf(wanted.mode));
    assert.deepEqual(named(engine.calls, 'setGiQuality'), [[{ resolution: 96, levels: 5, probeRays: 8 }]], wanted.mode);
    assert.deepEqual(named(engine.calls, 'setAreaLightShadows'), [[wanted.areaShadows]], wanted.mode);
    assert.deepEqual(named(engine.calls, 'setGlobalIllumination').at(-1), [wanted.gi], wanted.mode);
    assert.deepEqual(named(engine.calls, 'setCamera'), [[lighting.CORNELL_SELECTION.camera]], wanted.mode);
    assert.equal(state.frame, 0, wanted.mode);
    assert.ok(state.instances.some(item => item.name === 'cornell-white'), wanted.mode);
  }
});

test('observe reports factNames in order and each part fails on its own defect only', async () => {
  assert.deepEqual(quality.factNames, ['gi-active', 'gi-off', 'all-parts', 'cone-tier']);
  const on = await observed(stub());
  assert.deepEqual(on.assertions.map(item => item.name), quality.factNames);
  assert.equal(on.assertions.find(item => item.name === 'gi-active').passed, true);
  assert.equal(on.assertions.find(item => item.name === 'all-parts').passed, true);
  assert.deepEqual(on.instances, []);
  assert.deepEqual(on.edits, []);
  assert.equal(await passesFact('all-parts', stub({ settings: { screenSpace: false, screenSpaceQuality: 'high' } })), false);
  assert.equal(await passesFact('all-parts', stub({ area: { ...AREA_ON, stochasticMode: 'stochastic' } })), false);
  assert.equal(await passesFact('all-parts', stub({ passes: [SHADOW_PASS] })), false);
  assert.equal(await passesFact('all-parts', stub({ passes: [SSGI_PASS] })), false);
  assert.equal(await passesFact('cone-tier', stub({ area: { ...AREA_ON, stochasticCount: 0, activeStochastic: 0 } })), true);
  assert.equal(await passesFact('cone-tier', stub()), false);
  assert.equal(await passesFact('cone-tier', stub({ area: { ...AREA_ON, stochasticCount: 0, activeStochastic: 1 } })), false);
  const off = { ...SETTLED, active: false, inactiveReason: 'disabled', allocatedBytes: 0 };
  assert.equal(await passesFact('gi-off', stub({ stats: off })), true);
  assert.equal(await passesFact('gi-off', stub()), false);
  assert.equal(await passesFact('gi-off', stub({ stats: { ...off, allocatedBytes: 1 } })), false);
  assert.equal(await passesFact('gi-off', stub({ stats: { ...off, inactiveReason: 'noInstance' } })), false);
});

test('the default export is the gi-quality scene in the lighting stage, reusing the area-stochastic build and frame', () => {
  assert.equal(quality.default.id, 'gi-quality');
  assert.equal(quality.default.milestone, 'lighting');
  assert.equal(quality.default.title, 'GI quality: the Cornell box with every lighting part');
  assert.equal(quality.default.build, areaStochastic.build);
  assert.equal(quality.default.frame, areaStochastic.frame);
  assert.deepEqual(quality.default.ready(), { ready: true, stage: 'lighting' });
});
