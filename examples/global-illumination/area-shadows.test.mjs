import test from 'node:test';
import assert from 'node:assert/strict';
import * as area from './scenes/area-shadows.js';
import * as lighting from './scenes/lighting.js';
import * as volume from './scenes/volume.js';
import { validateInventory } from './lib/inventory.js';

// Cornell box area-shadow rows, checked against contract 'gi-area-shadow-rows'. The literals below repeat
// that contract; they are not read back from scenes/area-shadows.js. Node only: no engine, no WebGPU.

const CORNELL_HELD = ['Cornell box geometry and wall colours', 'single ceiling area light and its fixture', 'camera', 'exposure'];
const ON_FACTS = ['snapshot:settings', 'snapshot:stats', 'runtime:ready', 'assertion:gi-active', 'assertion:area-tier'];
const OFF_FACTS = ['snapshot:settings', 'snapshot:stats', 'runtime:ready', 'assertion:gi-off', 'assertion:area-tier'];
const SHADOW = { heroDistance: 25, fadeDistance: 80, band: 5 };
// mode, gi.enabled, tier, heroCount, giExpected, expected activeHeroes, filename, producer
const ROWS = [
  ['hero-gi-on', true, 'hero', 1, 'active', 1, 'gi-area-shadow-hero-pcss-penumbra-widens-from-block-base.png', 'raa-area-ltc'],
  ['hero-gi-off', false, 'hero', 1, 'disabled', 1, 'gi-area-shadow-hero-pcss-gi-off-soft-shadows-without-bounce.png', 'raa-area-ltc'],
  ['cone-gi-on', true, 'cone', 0, 'active', 0, 'gi-area-shadow-cone-trace-smooth-penumbra-no-voxel-steps.png', 'raa-gi-screen'],
  ['fade-gi-on', true, 'fade', 0, 'active', 0, 'gi-area-shadow-far-fade-light-through-gi-volume-only.png', 'raa-area-ltc'],
  ['auto-gi-on', true, 'auto', 1, 'active', 1, 'gi-area-shadow-auto-selection-soft-shadows-and-bounce.png', 'raa-area-ltc'],
];
const SETTLED = {
  active: true, inactiveReason: null, quality: { resolution: 96, levels: 5, probeRays: 8 },
  finestVoxelMetres: 0.2, voxelizedForwardInstances: 8, voxelizedVirtualInstances: 0,
  pendingRebuild: false, framesSinceRebuild: 240, allocatedBytes: 148635648, statsFrameIndex: 300, pending: false,
};
const DISABLED = { ...SETTLED, active: false, inactiveReason: 'disabled', voxelizedForwardInstances: 0, framesSinceRebuild: 0, allocatedBytes: 0 };
const AREA = { tier: 'hero', heroCount: 1, heroDistance: 25, fadeDistance: 80, band: 5, activeHeroes: 1 };

const variantOf = id => area.default.variants.find(v => v.id === id);
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
function stub({ stats = SETTLED, area = AREA } = {}) {
  return { giStats: () => ({ ...stats }), areaLightShadows: () => ({ ...area }) };
}
const observed = (id, engine) => area.observe(engine, variantOf(id), { instances: [], moved: false, blockNode: null });

test('the five inventory rows equal the contract rows, row by row', () => {
  assert.deepEqual(area.inventory.map(row => row.id), ROWS.map(([mode]) => `area-shadows-${mode}`));
  ROWS.forEach(([mode, gi, tier, heroCount, , , filename, producer], index) => {
    const row = area.inventory[index];
    assert.equal(row.mode, mode);
    assert.equal(row.scenario, 'area-shadows');
    assert.equal(row.filename, filename);
    assert.equal(row.producer, producer);
    assert.equal(row.criterionName, tier === 'auto' ? 'M4 area light soft shadows: automatic selection' : `M4 area light soft shadows: ${tier} tier`);
    assert.deepEqual(row.requiredFacts, gi ? ON_FACTS : OFF_FACTS);
    assert.equal(row.pair, null);
    assert.equal(row.judgeReference, '/refs/cornell-box-reference.png');
    assert.equal(row.judgeAspect, 'functional');
    assert.deepEqual(row.heldConstant.slice(0, CORNELL_HELD.length), CORNELL_HELD);
    assert.equal(row.heldConstant[CORNELL_HELD.length], `area-light shadow tier ${tier} with heroCount ${heroCount}`);
  });
});

test('the variants carry the contract settings, GI state and expected heroes, unmeasured and unpaired', () => {
  ROWS.forEach(([mode, gi, tier, heroCount, giExpected, expectedHeroes]) => {
    const variant = variantOf(mode);
    assert.equal(variant.arrangement, 'cornell-box');
    assert.deepEqual(variant.gi, { enabled: gi, debugView: 'none' });
    assert.deepEqual(variant.quality, { resolution: 96, levels: 5, probeRays: 8 });
    assert.equal(variant.giExpected, giExpected);
    assert.deepEqual(variant.areaShadows, { tier, heroCount, ...SHADOW });
    assert.equal(variant.expectedHeroes, expectedHeroes);
    assert.equal(variant.measurement, undefined);
    assert.equal(variant.offMode, undefined);
    assert.equal(variant.judge, true);
  });
});

test('validateInventory accepts the scenes and returns rows that include the five ids', () => {
  const rows = validateInventory([area, lighting, volume], 'lighting');
  const ids = rows.map(row => row.id);
  for (const [mode] of ROWS) assert.ok(ids.includes(`area-shadows-${mode}`), `missing area-shadows-${mode}`);
});

test('build on a recording engine sets the tier, the GI state, the fixture exclusion and one area light', async () => {
  for (const [mode, gi, tier, heroCount] of ROWS) {
    const engine = recordingEngine();
    const state = await area.build(engine, {}, variantOf(mode));
    const calls = engine.calls;
    assert.deepEqual(named(calls, 'setGiQuality'), [[{ resolution: 96, levels: 5, probeRays: 8 }]]);
    assert.deepEqual(named(calls, 'setAreaLightShadows'), [[{ tier, heroCount, ...SHADOW }]], mode);
    assert.deepEqual(named(calls, 'setGlobalIllumination').at(-1), [{ enabled: gi, debugView: 'none' }], mode);
    assert.deepEqual(named(calls, 'setCamera'), [[lighting.CORNELL_SELECTION.camera]]);
    const fixture = state.instances.find(item => item.name === 'fixture').instance;
    assert.deepEqual(named(calls, 'setInstanceGlobalIllumination'), [[fixture, false]], mode);
    const lights = named(calls, 'addLight');
    assert.equal(lights.length, 1, mode);
    assert.equal(lights[0][0].kind, 'area');
    assert.equal(state.moved, false);
    assert.equal(state.blockNode, null);
  }
});

test('observe passes area-tier with matching tier, heroCount and activeHeroes, and fails when activeHeroes differs', async () => {
  const result = await observed('hero-gi-on', stub());
  assert.deepEqual(result.assertions.map(a => a.name), area.factNames);
  assert.deepEqual(result.assertions.map(a => a.name), ['gi-active', 'gi-off', 'area-tier']);
  assert.equal(outcome(result)['area-tier'], true);
  assert.deepEqual(result.edits, []);
  assert.match(result.assertions[2].detail, /tier hero .*heroCount 1 .*activeHeroes 1/);
  const off = outcome(await observed('hero-gi-on', stub({ area: { ...AREA, activeHeroes: 0 } })));
  assert.equal(off['area-tier'], false);
  const wrongTier = outcome(await observed('cone-gi-on', stub()));
  assert.equal(wrongTier['area-tier'], false);
});

test('observe follows the lighting rules for gi-active and gi-off on GI-on and GI-off rows', async () => {
  const on = outcome(await observed('cone-gi-on', stub()));
  assert.equal(on['gi-active'], true);
  assert.equal(on['gi-off'], false);
  const starved = outcome(await observed('cone-gi-on', stub({ stats: { ...SETTLED, allocatedBytes: 1 } })));
  assert.equal(starved['gi-active'], false);
  const off = outcome(await observed('hero-gi-off', stub({ stats: DISABLED, area: { ...AREA, activeHeroes: 1 } })));
  assert.equal(off['gi-active'], false);
  assert.equal(off['gi-off'], true);
  assert.equal(off['area-tier'], true);
});

test('ready returns the lighting stage with ready true', () => {
  assert.deepEqual(area.default.ready({}, { instances: [], moved: false, blockNode: null }, variantOf('hero-gi-on')), { ready: true, stage: 'lighting' });
});
