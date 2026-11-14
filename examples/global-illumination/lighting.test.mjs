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
const CORNELL_HELD = ['Cornell box geometry and wall colours', 'single ceiling area light and its fixture', 'camera', 'exposure'];
const AREA_HELD = ['dark glossy floor', 'rectangular area light panel and its glowing fixture',
  'black sky without sun or ambient', 'camera'];
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
  'cornell-box': CORNELL_HELD,
  'cornell-box-linear': [...CORNELL_HELD, 'linear tone mapping without dither at a quarter of the Cornell exposure'],
  'area-glossy-ltc': AREA_HELD,
  'area-glossy-form-factor': AREA_HELD,
};
// The Cornell rows name their own fact lists (contract: snapshot:settings and snapshot:stats for the pair).
const CORNELL_FACTS = ['snapshot:settings', 'snapshot:stats'];
// Rows whose held constants extend their arrangement's: the GI-off frame's flat sky ambient (the no-GI baseline).
const EXTRA_HELD = { 'quality-off': ['flat uncoloured sky ambient fill of the frame without global illumination'] };
// The 25 rows in contract order: [mode, arrangement, filename, criterionName, producer, extra fact, expectedVisibleBehavior].
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
    'Global illumination is visibly at work in the Cornell box, compared with the same box with GI off: the ceiling, dark without GI, is lit by bounce light; the red and green walls bleed their colour onto the white blocks and the floor (a red tint on faces near the red left wall, a green tint on faces near the green right wall); the front faces of the two blocks, which the ceiling light does not reach directly, are lit by bounce light instead of staying black; no bright blotch or ring around the ceiling light and no light leaking through the walls or ceiling; the direct light is distributed like the reference: the floor under the ceiling light is the brightest region and the back wall falls off towards its top corners. The reference image is a path-traced Cornell box given as context for the scene and for that direct-light distribution: hard shadow edges, a different overall brightness, visible voxel-scale smoothness in the bounce, or any other difference from the reference except the direct-light distribution are not faults here.'],
  ['cornell-box-off', 'cornell-box', 'gi-cornell-box-off-direct-only.png', 'P4-Q8 GI off frame', 'raa-gi-core', null,
    'The same Cornell box with global illumination off and the same exposure, compared with the reference only for the distribution of the direct light of the ceiling light: the floor under the light is the brightest region and stays well lit towards the front of the box, the back wall is evenly lit in its middle and falls off towards its top corners, and the tops of the two blocks are bright; the ceiling and the surfaces facing away from the light stay dark, there is no red or green colour bleeding, and the blocks cast no shadows because RAA local lights are unshadowed without GI. Differences from the path-traced reference in overall brightness, missing bounce light, missing shadows or missing colour bleeding are not faults here.'],
  ['cornell-box-linear-on', 'cornell-box-linear', 'gi-cornell-box-linear-on-hdr-ratio-probe.png', 'P4-Q3 Cornell linear HDR probe', 'raa-gi-screen', null,
    'Global illumination on, shown with linear tone mapping at a quarter of the usual exposure (a darker image with flat contrast): the floor under the ceiling light is the brightest surface, the back wall is lit and darkens towards its top corners, the ceiling is lit only by bounce light, and the red and green walls tint the nearby white surfaces; nothing is clipped to pure white except the light fixture itself.'],
  ['cornell-box-linear-off', 'cornell-box-linear', 'gi-cornell-box-linear-off-hdr-ratio-probe.png', 'P4-Q8 GI off frame', 'raa-gi-core', null,
    'Global illumination off, shown with linear tone mapping at a quarter of the usual exposure: only surfaces the ceiling light reaches directly are lit; the floor under the light is the brightest surface and the back wall darkens towards its top corners; the ceiling stays black and there is no red or green colour bleeding; nothing is clipped to pure white except the light fixture itself.'],
  ['area-glossy-ltc-off', 'area-glossy-ltc', 'gi-area-light-glossy-floor-stretched-rectangle-reflection.png', 'M3 LTC area light: specular reflection is a stretched rectangle', 'raa-area-ltc', null,
    'On the dark glossy floor in front of the glowing rectangular light panel, the reflection of the panel appears as a bright elongated rectangle, a streak as wide as the panel and stretched towards the camera, surrounded by a softer pool of diffuse light on the floor; the reflection is not a small round dot or point-like highlight.'],
  ['area-glossy-form-factor-off', 'area-glossy-form-factor', 'gi-area-light-form-factor-mode-diffuse-pool-no-reflection.png', 'M3 area light diffuse-only form-factor mode', 'raa-area-ltc', null,
    'The same dark glossy floor and glowing rectangular panel in the cheap diffuse-only area-light mode: the floor shows only a soft pool of diffuse light in front of the panel, brightest nearest to it and fading with distance; no bright reflection streak or highlight of the panel appears on the floor.'],
];
const MODES = TABLE.map(row => row[0]);
const QUALITY_ON = ['quality-64-3-on', 'quality-96-5-on'];
const UNMEASURED = ['quality-64-3-on', 'cornell-box-linear-on', 'cornell-box-linear-off',
  'area-glossy-ltc-off', 'area-glossy-form-factor-off'];
// The area-light rows are the only off rows without an on partner.
const UNPAIRED_OFF = ['area-glossy-ltc-off', 'area-glossy-form-factor-off'];
// Both Cornell rows carry the reference image as judge context, for the direct-light distribution.
const REFERENCE_ROWS = ['cornell-box-on', 'cornell-box-off'];
const pick = (object, keys) => Object.fromEntries(keys.map(key => [key, object[key]]));
const variantOf = id => lighting.default.variants.find(v => v.id === id);

const expectedRow = ([mode, arrangement, filename, criterionName, producer, extra, expectedVisibleBehavior]) => {
  const on = mode.endsWith('-on');
  return {
    criterionName,
    scenario: 'lighting',
    expectedVisibleBehavior,
    requiredFacts: ['cornell-box', 'cornell-box-linear'].includes(arrangement) ? CORNELL_FACTS : [...(on ? ON_FACTS : OFF_FACTS), ...(extra ? [extra] : [])],
    heldConstant: [...HELD[arrangement], ...(EXTRA_HELD[mode] ?? [])],
    producer,
    pair: UNMEASURED.includes(mode) ? null : `gi-lighting-${arrangement}`,
    // Both Cornell rows carry the reference image, as judge context for the direct-light distribution.
    ...(REFERENCE_ROWS.includes(mode) ? { judgeReference: '/refs/cornell-box-reference.png', judgeAspect: 'functional' } : {}),
  };
};

test('exports: identity, factNames, functions and the 25 variant ids in order', () => {
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

test('inventory rows: [id, mode, filename] equal the 25 contract rows in order', () => {
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
    if (UNMEASURED.includes(mode)) {
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
  const offs = lighting.default.variants.filter(v => v.id.endsWith('-off') && !UNPAIRED_OFF.includes(v.id));
  assert.equal(offs.length, 11);
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
    if (off.measurement === undefined) {
      // The unmeasured probe rows carry no pair on either side.
      assert.equal(partner.measurement, undefined);
    } else {
      assert.deepEqual(measurementCore(off.measurement), measurementCore(partner.measurement));
      assert.deepEqual([off.measurement.state, off.measurement.features.globalIllumination], ['off', false]);
      assert.deepEqual([partner.measurement.state, partner.measurement.features.globalIllumination], ['on', true]);
    }
  }
});

test('validateInventory([volume, lighting], "lighting") returns the 8 volume ids then the 25 lighting ids', () => {
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

test('Cornell rows: 25 rows, the pair gi-lighting-cornell-box, the reference on both rows only', () => {
  const on = lighting.inventory.find(r => r.mode === 'cornell-box-on');
  const off = lighting.inventory.find(r => r.mode === 'cornell-box-off');
  assert.equal(lighting.inventory.length, 25);
  assert.equal(on.judgeReference, '/refs/cornell-box-reference.png');
  assert.equal(on.judgeAspect, 'functional');
  assert.equal(off.judgeReference, '/refs/cornell-box-reference.png');
  assert.equal(off.judgeAspect, 'functional');
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
  assert.equal(lighting.CORNELL_EXPOSURE, 3.3743);
  for (const id of ['cornell-box-on', 'cornell-box-off']) {
    const { calls } = await cornellCalls(id);
    assert.deepEqual(named(calls, 'setToneMapping'), [[{ exposureEv: Math.log2(3.3743) }]]);
    // The fixture reads 4 after the exposure: its linear colour value is 4 / exposure.
    const fixtureMaterial = named(calls, 'addMaterial').find(([desc]) => desc.model === 'flat');
    assert.deepEqual(fixtureMaterial[0].emissive.map(v => Math.round(v * 1e6)), [4 / 3.3743, 4 / 3.3743, 4 / 3.3743].map(v => Math.round(v * 1e6)));
    assert.equal(fixtureMaterial[0].color.every(v => v === 0), true);
  }
  // The back-wall centre under the M3 area model (light at 5.487): diffuse = albedo x radiance x F x W, with
  // radiance 18.4 x colour, F the horizon-clipped form factor of contract area-form-factor-wgsl and W the range
  // window at the representative point; the target luminance of the reference patch after inverting AgX is
  // 0.5797, and exposure = Y_t / (1.35 Y_d).
  const P = [2.78, 2.74, 5.592], L = [2.78, 5.487, 2.795], n = [0, 0, -1];
  const T = [0, 0, -0.525], B = [0.65, 0, 0];   // travel (0, -1, 0): tangent -z (1.05 m), bitangent +x (1.30 m)
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const unit = v => v.map(x => x / Math.hypot(...v));
  // three.js LTC_EdgeVectorFormFactor: the rational fit of theta / sin(theta), 0.5 / sin(theta) past the horizon.
  const edge = (a, b) => {
    const x = dot(a, b), y = Math.abs(x);
    const v = (0.8543985 + (0.4965155 + 0.0145206 * y) * y) / (3.4175940 + (4.1616724 + y) * y);
    const s = x > 0 ? v : 0.5 / Math.sqrt(Math.max(1 - x * x, 1e-7)) - v;
    return cross(a, b).map(c => c * s);
  };
  // Corners p - T - B, p - T + B, p + T + B, p + T - B relative to the receiver (counter-clockwise from below).
  const p = L.map((l, i) => l - P[i]);
  const u = [[-1, -1], [-1, 1], [1, 1], [1, -1]].map(([st, sb]) => unit(p.map((x, i) => x + st * T[i] + sb * B[i])));
  const f = u.reduce((sum, a, i) => edge(a, u[(i + 1) % 4]).map((x, k) => x + sum[k]), [0, 0, 0]);
  const length = Math.hypot(...f);
  const F = Math.max((length * length + dot(f, n)) / (length + 1), 0);
  assert.ok(Math.abs(F - 0.013992) < 1e-6, `form factor ${F}`);
  // The window W of the representative point (the light centre moved 0.525 m towards the wall along z).
  const distance = Math.hypot(0, 5.487 - P[1], 3.32 - P[2]);
  const window = Math.min(Math.max(1 - (distance / 30) ** 4, 0), 1) ** 2;
  const direct = [0.886, 0.699, 0.666].map((a, i) => a * 18.4 * [1, 0.598, 0.149][i] * F * window);
  const luminance = direct[0] * 0.2126 + direct[1] * 0.7152 + direct[2] * 0.0722;
  assert.ok(Math.abs(0.5797 / (1.35 * luminance) - lighting.CORNELL_EXPOSURE) < 5e-4, `derived ${0.5797 / (1.35 * luminance)}`);
});

test('Cornell linear probe rows: quarter-exposure linear tone mapping over the cornell-box selection', async () => {
  const cornell = variantOf('cornell-box-on');
  for (const [id, enabled] of [['cornell-box-linear-on', true], ['cornell-box-linear-off', false]]) {
    const variant = variantOf(id);
    assert.equal(variant.gi.enabled, enabled);
    assert.deepEqual(variant.selection, cornell.selection);
    const { calls } = await cornellCalls(id);
    assert.deepEqual(named(calls, 'setToneMapping'), [
      [{ exposureEv: Math.log2(lighting.CORNELL_EXPOSURE) }],
      [{ mapper: 'linear', dither: false, exposureEv: Math.log2(lighting.CORNELL_EXPOSURE * 0.25) }],
    ]);
  }
  assert.equal(variantOf('cornell-box-linear-on').offMode, 'cornell-box-linear-off');
  assert.equal(lighting.CORNELL_LIGHT_X, 2.78);
  assert.equal(lighting.CORNELL_LIGHT_Z, 2.795);
  assert.deepEqual(lighting.CORNELL_LIGHT_SIZE, [1.05, 1.3]);
  assert.equal(lighting.CORNELL_LINEAR_EXPOSURE_SCALE, 0.25);
});

test('references: the linear and area rows carry no judgeReference or judgeAspect', () => {
  for (const mode of ['cornell-box-linear-on', 'cornell-box-linear-off', 'area-glossy-ltc-off', 'area-glossy-form-factor-off']) {
    const row = lighting.inventory.find(r => r.mode === mode);
    assert.equal(row.judgeReference, undefined);
    assert.equal(row.judgeAspect, undefined);
  }
});

test('area-glossy rows: area-light shading, one area light, the floor and the GI-excluded fixture', async () => {
  for (const [id, mode] of [['area-glossy-ltc-off', 'ltc'], ['area-glossy-form-factor-off', 'formFactor']]) {
    const { calls, state } = await cornellCalls(id);
    assert.deepEqual(named(calls, 'setAreaLightShading'), [[{ mode }]]);
    const lights = named(calls, 'addLight');
    assert.equal(lights.length, 1, 'exactly one light');
    assert.deepEqual(lights[0][0], {
      kind: 'area', position: [0, 1.2, -6], direction: [0, -0.5, 1], color: [1, 0.95, 0.88],
      intensity: 24, range: 15, size: [2, 0.5],
    });
    assert.deepEqual(named(calls, 'setEnvironmentLighting'), [[{ diffuse: 0, specular: 0 }]]);
    assert.deepEqual(named(calls, 'setAmbient'), [[{ color: [0, 0, 0], intensity: 0 }]]);
    assert.deepEqual(named(calls, 'setToneMapping'), [], 'default tone mapping');
    assert.deepEqual(state.instances.map(item => item.name), ['floor', 'fixture']);
    const fixture = state.instances.find(item => item.name === 'fixture');
    assert.deepEqual(named(calls, 'setInstanceGlobalIllumination'), [[fixture.instance, false]]);
  }
});

test('area-glossy variants: GI off, expected disabled, unmeasured, no offMode and OFF_FACTS', () => {
  for (const id of ['area-glossy-ltc-off', 'area-glossy-form-factor-off']) {
    const variant = variantOf(id);
    assert.deepEqual(variant.gi, { enabled: false, debugView: 'none' });
    assert.equal(variant.giExpected, 'disabled');
    assert.equal(variant.measurement, undefined);
    assert.equal(variant.offMode, undefined);
    const row = lighting.inventory.find(r => r.mode === id);
    assert.deepEqual(row.requiredFacts, OFF_FACTS);
    assert.equal(row.pair, null);
    assert.deepEqual(row.heldConstant, AREA_HELD);
  }
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
