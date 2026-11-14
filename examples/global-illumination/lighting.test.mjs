import test from 'node:test';
import assert from 'node:assert/strict';
import * as lighting from './scenes/lighting.js';
import * as volume from './scenes/volume.js';
import { validateInventory } from './lib/inventory.js';

// Lighting stage of the global-illumination page, checked against contract gi-lighting-rows. The literals
// below repeat that contract; they are not read back from scenes/lighting.js. Node only: no engine, no WebGPU.

const FILENAME = /^[a-z0-9]+(?:-[a-z0-9]+)+\.png$/;
const ON_FACTS = ['snapshot:settings', 'snapshot:quality', 'snapshot:stats', 'snapshot:instances',
  'runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'assertion:gi-active'];
const OFF_FACTS = ['snapshot:settings', 'snapshot:quality', 'snapshot:stats',
  'runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'assertion:gi-off'];
const ROOM = ['room shell with red left wall and green right wall', 'white box and sphere', 'sun through the ceiling opening', 'camera'];
const HELD = {
  'color-bleed': ROOM,
  'interior-bounce': ROOM,
  quality: ROOM,
  'emissive-panel': ['room shell', 'emissive panel', 'white box and sphere', 'night sky without sun', 'camera'],
  'local-light-shadow': ['floor', '0.6 m wall', 'point and spot light placement', 'night sky without sun', 'camera'],
  'sky-overhang': ['open ground', '3 m overhang on four pillars', 'white box under the overhang', 'physical day sky and sun', 'camera'],
  'glossy-reflection': ['glossy floor', 'orange glowing sphere above the frame', 'day sky', 'camera'],
  'forward-content': ['room shell with red left wall', 'skinned figure, squashed shape, transparent pane and fur ball', 'camera'],
  'moved-instance': ['room shell', 'orange block and its two positions', 'camera'],
  'cornell-box': ['Cornell box geometry and wall colours', 'single ceiling area light and its fixture', 'camera', 'exposure'],
};
// The Cornell rows name their own fact lists (contract: snapshot:settings and snapshot:stats for the pair).
const CORNELL_FACTS = ['snapshot:settings', 'snapshot:stats'];
// Rows whose held constants extend their arrangement's: the GI-off frame's flat sky ambient (the no-GI baseline).
const EXTRA_HELD = { 'quality-off': ['flat uncoloured sky ambient fill of the frame without global illumination'] };
// The 21 rows in contract order: [mode, arrangement, filename, criterionName, producer, extra fact, expectedVisibleBehavior].
const TABLE = [
  ['color-bleed-on', 'color-bleed', 'gi-color-bleed-white-box-tinted-red-green.png', 'P4-Q3 colour bleeding', 'raa-gi-screen', null,
    'The white box and the white floor beside the coloured walls take a visible red tint on the side facing the red left wall and a green tint on the side facing the green right wall.'],
  ['color-bleed-off', 'color-bleed', 'gi-color-bleed-off-white-box-neutral.png', 'P4-Q8 GI off frame', 'raa-gi-core', null,
    'The white box and the white floor stay neutral white-grey, with no red or green tint picked up from the coloured walls.'],
  ['interior-bounce-on', 'interior-bounce', 'gi-interior-bounce-lit-from-sunlit-patch.png', 'P4-Q3 interior bounce', 'raa-gi-screen', null,
    'The walls, ceiling and objects away from the bright sunlit floor patch are lit by soft bounce light coming from that patch, so the shadowed interior is clearly visible.'],
  ['interior-bounce-off', 'interior-bounce', 'gi-interior-bounce-off-dark-interior.png', 'P4-Q8 GI off frame', 'raa-gi-core', null,
    'Away from the bright sunlit floor patch the interior stays dark, lit only by a faint flat ambient term.'],
  ['emissive-panel-on', 'emissive-panel', 'gi-emissive-panel-lights-nearby-surfaces.png', 'P4-Q3 emissive panel light', 'raa-gi-screen', null,
    'The floor, the back wall and the white box and sphere next to the glowing panel are lit by warm light spilling from it, brightest close to the panel and fading with distance.'],
  ['emissive-panel-off', 'emissive-panel', 'gi-emissive-panel-off-no-spill.png', 'P4-Q8 GI off frame', 'raa-gi-core', null,
    'Only the panel itself glows; the floor, walls, box and sphere around it stay dark, with no light spilling from the panel.'],
  ['local-light-shadow-on', 'local-light-shadow', 'gi-local-lights-stopped-by-wall.png', 'P4-Q4 local lights stopped by a wall', 'raa-gi-screen', 'assertion:local-lights',
    'The floor on the camera side of the wall stays dark in the wall shadow: the point and spot lights behind the wall light only the far side, whose glow shows past the edges of the wall.'],
  ['local-light-shadow-off', 'local-light-shadow', 'gi-local-lights-off-leak-through-wall.png', 'P4-Q8 GI off frame', 'raa-gi-core', 'assertion:local-lights',
    'Light from the point and spot lights behind the wall leaks through it: the floor on the camera side of the wall is lit as if the wall were not there.'],
  ['sky-overhang-on', 'sky-overhang', 'gi-sky-open-ground-kept-overhang-bounce.png', 'P4-Q5 sky ground kept and overhang bounce', 'raa-gi-screen', null,
    'The sunlit open ground keeps its normal daylight brightness, while the ground and the white box under the overhang receive visible soft bounce light and are clearly brighter than plain shadow.'],
  ['sky-overhang-off', 'sky-overhang', 'gi-sky-overhang-off.png', 'P4-Q8 GI off frame', 'raa-gi-core', null,
    'The ground and the white box under the overhang sit in flat dark shadow lit only by the ambient light of the sky.'],
  ['glossy-reflection-on', 'glossy-reflection', 'gi-glossy-floor-reflects-offscreen-object.png', 'P4-Q5 glossy floor volume reflection', 'raa-ssr-gi-fallback', 'assertion:ssr-enabled',
    'The glossy floor clearly reflects a bright orange glowing object that is itself outside the picture, above its top edge.'],
  ['glossy-reflection-off', 'glossy-reflection', 'gi-glossy-floor-off-env-only.png', 'P4-Q8 GI off frame', 'raa-gi-core', 'assertion:ssr-enabled',
    'The glossy floor reflects only the sky environment; no reflection of an orange glowing object appears on it.'],
  ['quality-64-3-on', 'quality', 'gi-quality-64-3-room-lit.png', 'P4-Q6 quality 64/3', 'raa-gi-screen', 'assertion:quality-applied',
    'The red-and-green room is lit by global illumination at the low quality: the walls, floor, box and sphere away from the sunlit patch show soft bounce light tinted by the coloured walls.'],
  ['quality-96-5-on', 'quality', 'gi-quality-96-5-room-lit.png', 'P4-Q6 quality 96/5', 'raa-gi-screen', 'assertion:quality-applied',
    'The red-and-green room is lit by global illumination at the high quality: the walls, floor, box and sphere away from the sunlit patch show soft bounce light tinted by the coloured walls.'],
  ['quality-off', 'quality', 'gi-quality-off-room.png', 'P4-Q8 GI off frame', 'raa-gi-core', null,
    'The same red-and-green room without global illumination: no coloured bounce tint appears anywhere; the box, the sphere and the floor beside the red and green walls stay neutral, lit away from the sunlit patch only by a flat, uncoloured ambient fill.'],
  ['forward-content-on', 'forward-content', 'gi-skinned-deformed-transparent-fur-receive-bounce.png', 'P4-Q6 forward content receives GI', 'raa-gi-screen', 'assertion:forward-content',
    'The skinned figure, the squashed shape, the transparent pane and the fur ball beside the red wall all show a reddish bounce tint and soft indirect light on their sides facing the wall, like the static surfaces around them.'],
  ['forward-content-off', 'forward-content', 'gi-forward-content-off.png', 'P4-Q8 GI off frame', 'raa-gi-core', 'assertion:forward-content',
    'The skinned figure, the squashed shape, the transparent pane and the fur ball beside the red wall show no red bounce tint; their shadowed sides stay dark.'],
  ['moved-instance-on', 'moved-instance', 'gi-moved-block-lights-new-place.png', 'P4-Q6 moved instance relit', 'raa-gi-screen', 'assertion:block-moved',
    'The bright orange block stands in its new place near the right wall, and the floor and wall around it there glow orange from its bounce light; no orange glow remains at its old place in the far left corner.'],
  ['moved-instance-off', 'moved-instance', 'gi-moved-block-off.png', 'P4-Q8 GI off frame', 'raa-gi-core', 'assertion:block-moved',
    'The bright orange block stands in its new place near the right wall with no orange light spilling onto the floor or wall around it.'],
  ['cornell-box-on', 'cornell-box', 'gi-cornell-box-on-bounce-lit.png', 'P4-Q3 Cornell box GI bounce visible', 'raa-gi-screen', null,
    'Global illumination is visibly at work in the Cornell box, compared with the same box with GI off: the ceiling, dark without GI, is lit by bounce light; the red and green walls bleed their colour onto the white blocks and the floor (a red tint on faces near the red left wall, a green tint on faces near the green right wall); the front faces of the two blocks, which the ceiling light does not reach directly, are lit by bounce light instead of staying black; no bright blotch or ring around the ceiling light and no light leaking through the walls or ceiling. The reference image is a path-traced Cornell box given only as context for the scene: hard shadow edges, a different overall brightness, a different shape of the lit area under the light, visible voxel-scale smoothness in the bounce, or any other difference from the reference are not faults here.'],
  ['cornell-box-off', 'cornell-box', 'gi-cornell-box-off-direct-only.png', 'P4-Q8 GI off frame', 'raa-gi-core', null,
    'The same Cornell box with global illumination off and the same exposure: only surfaces the ceiling light reaches directly are lit; the ceiling and the surfaces facing away from the light stay dark; there is no red or green colour bleeding on the white surfaces, and the blocks cast no shadows because RAA local lights are unshadowed without GI.'],
];
const MODES = TABLE.map(row => row[0]);
const QUALITY_ON = ['quality-64-3-on', 'quality-96-5-on'];
const pick = (object, keys) => Object.fromEntries(keys.map(key => [key, object[key]]));
const variantOf = id => lighting.default.variants.find(v => v.id === id);

const expectedRow = ([mode, arrangement, filename, criterionName, producer, extra, expectedVisibleBehavior]) => {
  const on = mode.endsWith('-on');
  return {
    criterionName,
    scenario: 'lighting',
    expectedVisibleBehavior,
    requiredFacts: arrangement === 'cornell-box' ? CORNELL_FACTS : [...(on ? ON_FACTS : OFF_FACTS), ...(extra ? [extra] : [])],
    heldConstant: [...HELD[arrangement], ...(EXTRA_HELD[mode] ?? [])],
    producer,
    pair: mode === 'quality-64-3-on' ? null : `gi-lighting-${arrangement}`,
    // Only the Cornell GI-on row carries the reference image, as judge context (aspect functional).
    ...(mode === 'cornell-box-on' ? { judgeReference: '/refs/cornell-box-reference.png', judgeAspect: 'functional' } : {}),
  };
};

test('exports: identity, factNames, functions and the 21 variant ids in order', () => {
  assert.equal(lighting.default.id, 'lighting');
  assert.equal(lighting.default.milestone, 'lighting');
  assert.equal(lighting.default.title, 'GI lighting');
  assert.deepEqual(lighting.factNames, [
    'gi-active', 'gi-off', 'local-lights', 'ssr-enabled', 'quality-applied', 'forward-content', 'block-moved',
  ]);
  assert.equal(typeof lighting.default.build, 'function');
  assert.equal(typeof lighting.default.ready, 'function');
  assert.equal(typeof lighting.observe, 'function');
  assert.deepEqual(lighting.default.variants.map(v => v.id), MODES);
});

test('inventory rows: [id, mode, filename] equal the 21 contract rows in order', () => {
  assert.deepEqual(lighting.inventory.map(r => [r.id, r.mode, r.filename]),
    TABLE.map(([mode, , filename]) => [`lighting-${mode}`, mode, filename]));
  for (const row of lighting.inventory) {
    assert.match(row.filename, FILENAME);
    assert.equal(row.scenario, 'lighting');
  }
});

test('rows carry the contract criterion, scenario, behaviour, facts, held constants, producer and pair', () => {
  for (const row of lighting.inventory) {
    const expected = expectedRow(TABLE.find(([mode]) => mode === row.mode));
    assert.deepEqual(pick(row, Object.keys(expected)), expected);
  }
});

test('variants carry the contract gi patch, giExpected, quality, arrangement, offMode and measurement', () => {
  for (const [mode, arrangement] of TABLE) {
    const v = variantOf(mode);
    const on = mode.endsWith('-on');
    assert.deepEqual(v.gi, { enabled: on, debugView: 'none' });
    assert.equal(v.giExpected, on ? 'active' : 'disabled');
    assert.equal(v.arrangement, arrangement);
    assert.deepEqual(v.quality, mode === 'quality-64-3-on'
      ? { resolution: 64, levels: 3, probeRays: 6 } : { resolution: 96, levels: 5, probeRays: 8 });
    assert.deepEqual([v.references, v.judge], [[], true]);
    if (on) assert.equal(v.offMode, QUALITY_ON.includes(mode) ? 'quality-off' : `${arrangement}-off`);
    else assert.equal(v.offMode, undefined);
    if (mode === 'quality-64-3-on') {
      assert.equal(v.measurement, undefined);
    } else {
      assert.deepEqual(v.measurement, {
        pair: `gi-lighting-${arrangement}`,
        state: on ? 'on' : 'off',
        preset: `lighting-${arrangement}`,
        features: { globalIllumination: on },
      });
    }
  }
});

test('measured pairs: one -on and one -off variant on the same preset; quality-64-3-on has no pair', () => {
  const measured = lighting.default.variants.filter(v => v.measurement);
  const pairs = [...new Set(measured.map(v => v.measurement.pair))];
  assert.equal(pairs.length, 10);
  for (const pair of pairs) {
    const members = measured.filter(v => v.measurement.pair === pair);
    assert.deepEqual(members.map(v => v.measurement.state).sort(), ['off', 'on']);
    assert.equal(new Set(members.map(v => v.measurement.preset)).size, 1);
    assert.ok(members.every(v => v.id.endsWith(`-${v.measurement.state}`)));
  }
  assert.equal(variantOf('quality-96-5-on').measurement.pair, 'gi-lighting-quality');
  assert.equal(variantOf('quality-off').measurement.pair, 'gi-lighting-quality');
  assert.equal(variantOf('quality-64-3-on').measurement, undefined);
  assert.equal(lighting.inventory.find(r => r.mode === 'quality-64-3-on').pair, null);
});

test('both quality -on rows name quality-off as offMode and share its arrangement', () => {
  const off = variantOf('quality-off');
  assert.equal(off.arrangement, 'quality');
  for (const id of QUALITY_ON) {
    const v = variantOf(id);
    assert.equal(v.offMode, 'quality-off');
    assert.equal(v.arrangement, off.arrangement);
  }
});

test('each off variant equals its on partner except id, gi.enabled, giExpected, description, measurement.state and offMode', () => {
  // measurement.features.globalIllumination necessarily differs too (true on, false off); it is checked below.
  const core = v => {
    const { id, description, giExpected, gi, measurement, offMode, ...rest } = v;
    return rest;
  };
  const measurementCore = m => ({ pair: m.pair, preset: m.preset });
  const offs = lighting.default.variants.filter(v => v.id.endsWith('-off'));
  assert.equal(offs.length, 10);
  for (const off of offs) {
    const partner = variantOf(off.id === 'quality-off' ? 'quality-96-5-on' : off.id.replace(/-off$/, '-on'));
    assert.equal(partner.offMode, off.id);
    assert.equal(off.offMode, undefined);
    assert.equal(off.giExpected, 'disabled');
    assert.equal(off.gi.enabled, false);
    assert.equal(partner.giExpected, 'active');
    assert.equal(partner.gi.enabled, true);
    assert.notEqual(off.description, partner.description);
    assert.deepEqual(core(off), core(partner));
    assert.equal(off.gi.debugView, partner.gi.debugView);
    assert.deepEqual(measurementCore(off.measurement), measurementCore(partner.measurement));
    assert.deepEqual([off.measurement.state, off.measurement.features.globalIllumination], ['off', false]);
    assert.deepEqual([partner.measurement.state, partner.measurement.features.globalIllumination], ['on', true]);
  }
});

test('validateInventory([volume, lighting], "lighting") returns the 8 volume ids then the 21 lighting ids', () => {
  const volumeIds = ['volume-voxels-on', 'volume-voxels-off', 'volume-radiance-on', 'volume-radiance-off',
    'volume-distance-on', 'volume-distance-off', 'volume-levels-on', 'volume-levels-off'];
  const rows = validateInventory([volume, lighting], 'lighting');
  assert.deepEqual(rows.map(r => r.id), [...volumeIds, ...MODES.map(mode => `lighting-${mode}`)]);
});

test('a glossy namespace whose only subject is the orange sphere box is refused as outside camera frustum', () => {
  // The sphere has radius 1 at [0, 6, -8]: its box lies above the top plane of the glossy camera.
  const orangeBox = { name: 'orange', min: [-1, 5, -9], max: [1, 7, -7] };
  const subjectOnly = v => (v.arrangement !== 'glossy-reflection' ? v : { ...v, selection: { ...v.selection, subjects: [orangeBox] } });
  const patched = { ...lighting.default, variants: lighting.default.variants.map(subjectOnly) };
  assert.throws(() => validateInventory([{ ...lighting, default: patched }], 'lighting'), /outside camera frustum/);
});

const INSTANCES = [
  { name: 'room', instance: 0 }, { name: 'skinned', instance: 1 }, { name: 'deformed', instance: 2 },
  { name: 'transparent', instance: 3 }, { name: 'fur', instance: 4 },
];
const SETTLED = {
  active: true, inactiveReason: null, quality: { resolution: 96, levels: 5, probeRays: 8 },
  finestVoxelMetres: 0.2, voxelizedForwardInstances: 8, voxelizedVirtualInstances: 0,
  pendingRebuild: false, framesSinceRebuild: 240, allocatedBytes: 148635648, statsFrameIndex: 300, pending: false,
};
const DISABLED = {
  ...SETTLED, active: false, inactiveReason: 'disabled', voxelizedForwardInstances: 0,
  framesSinceRebuild: 0, allocatedBytes: 0,
};
// A stub engine with the methods observe and ready read; `calls` records setTransform.
function stub({ stats = SETTLED, quality = stats.quality, reflections = true, activeLights = 2,
  passes = [{ name: 'gi.volume', enabled: true }, { name: 'ssr', enabled: true }], voxelized = false, calls = [] } = {}) {
  return {
    giStats: () => ({ ...stats }),
    giQuality: () => ({ ...quality }),
    globalIllumination: () => ({
      enabled: stats.active, intensity: 1, environmentDiffuse: 0,
      localLightShadows: true, reflections, screenTraces: true, debugView: 'none',
    }),
    lightStats: () => ({
      activeLights, attachedLights: 0, tileColumns: 1, tileRows: 1, tileCapacity: 127,
      droppedLightReferences: 0, statsFrameIndex: 300, pending: false,
    }),
    passes: () => passes,
    instanceGlobalIllumination: () => ({ requested: null, voxelized, driver: voxelized ? 'forward' : null, code: null, message: null }),
    setTransform: (...args) => calls.push(args),
  };
}
const stateOf = (overrides = {}) => ({ instances: INSTANCES, moved: false, blockNode: null, ...overrides });
const observed = (id, engine, overrides) => lighting.observe(engine, variantOf(id), stateOf(overrides));
const outcome = result => Object.fromEntries(result.assertions.map(a => [a.name, a.passed]));

test('observe reports the seven factNames in order, with the instances and no edits', async () => {
  const result = await observed('color-bleed-on', stub());
  assert.deepEqual(result.assertions.map(a => a.name), lighting.factNames);
  assert.deepEqual(result.instances, INSTANCES);
  assert.deepEqual(result.edits, []);
});

test('gi-active passes on a settled 96/5 stub and on a settled 64/3 stub with allocatedBytes 27262976', async () => {
  assert.equal(outcome(await observed('color-bleed-on', stub()))['gi-active'], true);
  const low = { ...SETTLED, quality: { resolution: 64, levels: 3, probeRays: 6 }, allocatedBytes: 27262976 };
  const result = await observed('quality-64-3-on', stub({ stats: low, quality: { resolution: 64, levels: 3, probeRays: 6 } }));
  assert.equal(outcome(result)['gi-active'], true);
});

test('gi-off passes on the disabled stub and refuses gi-active there', async () => {
  const result = outcome(await observed('color-bleed-off', stub({ stats: DISABLED })));
  assert.equal(result['gi-off'], true);
  assert.equal(result['gi-active'], false);
});

test('quality-applied passes when giQuality matches the variant and fails on a different quality', async () => {
  assert.equal(outcome(await observed('quality-96-5-on', stub()))['quality-applied'], true);
  const other = stub({ quality: { resolution: 64, levels: 3, probeRays: 6 } });
  assert.equal(outcome(await observed('quality-96-5-on', other))['quality-applied'], false);
});

test('local-lights passes with two active local lights and fails with one', async () => {
  assert.equal(outcome(await observed('local-light-shadow-on', stub()))['local-lights'], true);
  assert.equal(outcome(await observed('local-light-shadow-on', stub({ activeLights: 1 })))['local-lights'], false);
});

test('forward-content passes with the four named instances voxelized false and fails otherwise', async () => {
  assert.equal(outcome(await observed('forward-content-on', stub()))['forward-content'], true);
  assert.equal(outcome(await observed('forward-content-on', stub({ voxelized: true })))['forward-content'], false);
  const missing = await observed('forward-content-on', stub(), { instances: INSTANCES.slice(0, 3) });
  assert.equal(outcome(missing)['forward-content'], false);
});

test('block-moved follows state.moved', async () => {
  assert.equal(outcome(await observed('moved-instance-on', stub(), { moved: true }))['block-moved'], true);
  assert.equal(outcome(await observed('moved-instance-on', stub()))['block-moved'], false);
});

test('ssr-enabled needs an enabled ssr pass entry and reflections true', async () => {
  assert.equal(outcome(await observed('glossy-reflection-on', stub()))['ssr-enabled'], true);
  const passOff = stub({ passes: [{ name: 'ssr', enabled: false }] });
  assert.equal(outcome(await observed('glossy-reflection-on', passOff))['ssr-enabled'], false);
  const noReflections = stub({ reflections: false });
  assert.equal(outcome(await observed('glossy-reflection-on', noReflections))['ssr-enabled'], false);
});

test('ready: moved-instance-on waits for a settled volume, then moves the block once', () => {
  const variant = variantOf('moved-instance-on');
  const calls = [];
  const waiting = stateOf({ blockNode: 7 });
  const pending = stub({ stats: { ...SETTLED, pendingRebuild: true, framesSinceRebuild: 0 }, calls });
  assert.deepEqual(lighting.default.ready(pending, waiting, variant), { ready: false, stage: 'lighting' });
  assert.equal(waiting.moved, false);
  assert.equal(calls.length, 0);
  const settled = stateOf({ blockNode: 7 });
  assert.deepEqual(lighting.default.ready(stub({ calls }), settled, variant), { ready: true, stage: 'lighting' });
  assert.deepEqual(lighting.default.ready(stub({ calls }), settled, variant), { ready: true, stage: 'lighting' });
  assert.equal(settled.moved, true);
  assert.deepEqual(calls, [[7, [2.0, 0.4, -3.2], [0, 0, 0, 1], [1, 1, 1]]]);
});

test('ready: moved-instance-off moves the block on its first call', () => {
  const calls = [];
  const st = stateOf({ blockNode: 7 });
  const result = lighting.default.ready(stub({ stats: DISABLED, calls }), st, variantOf('moved-instance-off'));
  assert.deepEqual(result, { ready: true, stage: 'lighting' });
  assert.equal(st.moved, true);
  assert.equal(calls.length, 1);
});

test('ready: other arrangements are ready at once and never move a block', () => {
  const calls = [];
  const result = lighting.default.ready(stub({ stats: DISABLED, calls }), stateOf(), variantOf('color-bleed-off'));
  assert.deepEqual(result, { ready: true, stage: 'lighting' });
  assert.equal(calls.length, 0);
});

// The Cornell box (S40 amendment): build against a recording engine, then check the calls it makes.
function recordingEngine() {
  const calls = [];
  let next = 1;
  const engine = new Proxy({}, {
    get: (target, name) => (name === 'calls' ? calls : (...args) => { calls.push([name, ...args]); return next++; }),
  });
  return engine;
}
async function cornellCalls(id) {
  const engine = recordingEngine();
  const state = await lighting.build(engine, {}, variantOf(id));
  return { calls: engine.calls, state };
}
const named = (calls, name) => calls.filter(call => call[0] === name).map(call => call.slice(1));
const srgb = v => (v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055);

test('Cornell rows: 21 rows, the pair gi-lighting-cornell-box, judgeReference on the on row only', () => {
  const on = lighting.inventory.find(r => r.mode === 'cornell-box-on');
  const off = lighting.inventory.find(r => r.mode === 'cornell-box-off');
  assert.equal(lighting.inventory.length, 21);
  assert.equal(on.judgeReference, '/refs/cornell-box-reference.png');
  assert.equal(on.judgeAspect, 'functional');
  assert.equal(off.judgeReference, undefined);
  assert.equal(off.judgeAspect, undefined);
  assert.equal(on.pair, 'gi-lighting-cornell-box');
  assert.equal(off.pair, 'gi-lighting-cornell-box');
  assert.deepEqual(on.requiredFacts, ['snapshot:settings', 'snapshot:stats']);
  assert.deepEqual(off.requiredFacts, ['snapshot:settings', 'snapshot:stats']);
  assert.equal(on.filename, 'gi-cornell-box-on-bounce-lit.png');
  assert.equal(off.filename, 'gi-cornell-box-off-direct-only.png');
  assert.equal(variantOf('cornell-box-on').offMode, 'cornell-box-off');
  assert.deepEqual(variantOf('cornell-box-on').measurement.features, { globalIllumination: true });
  assert.deepEqual(variantOf('cornell-box-off').measurement.features, { globalIllumination: false });
  assert.deepEqual(variantOf('cornell-box-on').selection.camera, { eye: [2.78, 2.73, -8], target: [2.78, 2.73, 0], fovYDeg: 39.3, near: 0.05, far: 200 });
});

test('Cornell light and fixture: CORNELL_LIGHT_Y 5.088, the fixture at 5.4875 and excluded from GI', async () => {
  assert.equal(lighting.CORNELL_LIGHT_Y, 5.088);
  assert.equal(lighting.CORNELL_FIXTURE_Y, 5.4875);
  for (const id of ['cornell-box-on', 'cornell-box-off']) {
    const { calls, state } = await cornellCalls(id);
    const lights = named(calls, 'addLight');
    assert.equal(lights.length, 1, 'the only light');
    const [light] = lights;
    assert.equal(light[0].kind, 'area');
    assert.deepEqual(light[0].position, [2.78, 5.088, 2.795]);
    assert.deepEqual(light[0].direction, [0, -1, 0]);
    assert.deepEqual(light[0].size, [1.05, 1.3]);
    assert.equal(light[0].range, 30);
    assert.ok(Math.abs(light[0].intensity - 18.4 * 1.365) < 1e-9);
    assert.deepEqual(light[0].color, [1, 0.598, 0.149].map(srgb));
    assert.deepEqual(named(calls, 'setLight'), [[{ direction: [0, -1, 0], color: [1, 1, 1], intensity: 0 }]]);
    assert.deepEqual(named(calls, 'setAmbient'), [[{ color: [0, 0, 0], intensity: 0 }]]);
    assert.deepEqual(named(calls, 'setEnvironmentLighting'), [[{ diffuse: 0, specular: 0 }]]);
    // The fixture is the flat quad at CORNELL_FIXTURE_Y, excluded from GI, and the last instance of the scene.
    const fixture = state.instances.find(item => item.name === 'fixture');
    assert.ok(fixture, 'fixture instance');
    assert.deepEqual(named(calls, 'setInstanceGlobalIllumination'), [[fixture.instance, false]]);
    const meshes = named(calls, 'addMesh').map(([desc]) => desc);
    const fixtureMesh = meshes.find(desc => desc.positions.every((v, i) => i % 3 !== 1 || Math.abs(v - 5.4875) < 1e-5));
    assert.ok(fixtureMesh, 'a mesh at CORNELL_FIXTURE_Y');
    assert.ok(fixtureMesh.normals.every((v, i) => Math.abs(v - (i % 3 === 1 ? -1 : 0)) < 1e-6), 'facing down');
  }
});

test('Cornell tone mapping: the shared exposure is the derived constant, and the derivation holds', async () => {
  assert.equal(lighting.CORNELL_EXPOSURE, 2.1666);
  for (const id of ['cornell-box-on', 'cornell-box-off']) {
    const { calls } = await cornellCalls(id);
    assert.deepEqual(named(calls, 'setToneMapping'), [[{ exposureEv: Math.log2(2.1666) }]]);
    // The fixture reads 4 after the exposure: its linear colour value is 4 / exposure.
    const fixtureMaterial = named(calls, 'addMaterial').find(([desc]) => desc.model === 'flat');
    assert.deepEqual(fixtureMaterial[0].emissive.map(v => Math.round(v * 1e6)), [4 / 2.1666, 4 / 2.1666, 4 / 2.1666].map(v => Math.round(v * 1e6)));
    assert.equal(fixtureMaterial[0].color.every(v => v === 0), true);
  }
  // The back-wall centre under the S17 area formula (light at 5.487, representative point, Rec. 709 luminance)
  // and the target luminance of the reference patch after inverting AgX (0.5797): exposure = Y_t / (1.35 Y_d).
  const P = [2.78, 2.74, 5.592], L = [2.78, 5.487, 2.795];
  const v = P.map((p, i) => p - L[i]);
  const along = [0, 0, -1], across = [1, 0, 0];   // travel (0, -1, 0): tangent -z, bitangent +x
  const at = Math.min(Math.max(v[2] * along[2], -0.525), 0.525);
  const ab = Math.min(Math.max(v[0] * across[0], -0.65), 0.65);
  const R = L.map((l, i) => l + along[i] * at + across[i] * ab);
  const toLight = R.map((r, i) => r - P[i]);
  const d2 = Math.max(toLight.reduce((sum, x) => sum + x * x, 0), 0.04);
  // The white back wall's normal is (0, 0, -1), so n.l is -toLight.z over the distance.
  const ndotl = Math.max(-toLight[2] / Math.sqrt(d2), 0);
  const q = Math.sqrt(d2) / 30;
  const window = Math.min(Math.max(1 - q ** 4, 0), 1) ** 2;
  const direct = [0.886, 0.699, 0.666].map((a, i) => (a / Math.PI) * [1, 0.598, 0.149][i] * 18.4 * 1.365 * window / d2 * ndotl);
  const luminance = direct[0] * 0.2126 + direct[1] * 0.7152 + direct[2] * 0.0722;
  assert.ok(Math.abs(0.5797 / (1.35 * luminance) - lighting.CORNELL_EXPOSURE) < 5e-4, `derived ${0.5797 / (1.35 * luminance)}`);
});

test('Cornell geometry: wall normals point into the box, block tops face up, windings agree', async () => {
  const { calls } = await cornellCalls('cornell-box-on');
  // Meshes are recorded in creation order: the fixture, the white group, the red and the green wall.
  const [, white, red, green] = named(calls, 'addMesh').map(([desc]) => desc);
  // The red wall (x about 5.5) faces -x into the box and the green wall (x 0) faces +x; the red quad is not exactly
  // planar, so its normal is only close to the axis.
  for (let i = 0; i < red.normals.length; i += 3) assert.ok(red.normals[i] < -0.99);
  for (let i = 0; i < green.normals.length; i += 3) assert.ok(Math.abs(green.normals[i] - 1) < 1e-6);
  assert.equal(red.indices.length, 6);
  assert.equal(green.indices.length, 6);
  // The white mesh starts with the floor (vertices 0-3, +y), the ceiling (4-7, -y) and the back (8-11, -z);
  // the short block's top follows (12-15, +y).
  const normalOf = vertex => [0, 1, 2].map(k => white.normals[vertex * 3 + k]);
  const near = (a, b) => a.every((x, k) => Math.abs(x - b[k]) < 1e-6);
  assert.ok(near(normalOf(0), [0, 1, 0]), 'floor');
  assert.ok(near(normalOf(4), [0, -1, 0]), 'ceiling');
  assert.ok(near(normalOf(8), [0, 0, -1]), 'back');
  assert.ok(near(normalOf(12), [0, 1, 0]), 'short block top');
  // Every triangle's winding agrees with its vertex normal (counter-clockwise front faces).
  for (const mesh of [white, red, green]) {
    for (let t = 0; t < mesh.indices.length; t += 3) {
      const corner = i => [0, 1, 2].map(k => mesh.positions[mesh.indices[t + i] * 3 + k]);
      const [a, b, c] = [corner(0), corner(1), corner(2)];
      const e1 = a.map((x, k) => b[k] - x), e2 = a.map((x, k) => c[k] - x);
      const cross = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
      const n = [0, 1, 2].map(k => mesh.normals[mesh.indices[t] * 3 + k]);
      assert.ok(cross[0] * n[0] + cross[1] * n[1] + cross[2] * n[2] > 0, `triangle ${t / 3} winding`);
    }
  }
});
