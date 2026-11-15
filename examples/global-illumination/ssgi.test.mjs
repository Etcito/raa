import test from 'node:test';
import assert from 'node:assert/strict';
import * as ssgi from './scenes/ssgi.js';
import * as restir from './scenes/area-restir.js';
import * as stochastic from './scenes/area-stochastic.js';
import * as area from './scenes/area-shadows.js';
import * as lighting from './scenes/lighting.js';
import * as volume from './scenes/volume.js';
import * as temporal from './scenes/temporal.js';
import { validateInventory } from './lib/inventory.js';

// Screen-space GI rows, checked against contract 'gi-ssgi-rows'. The literals below repeat that contract; they are
// not read back from scenes/ssgi.js. Node only: no engine, no WebGPU.

const CORNELL_HELD = ['Cornell box geometry and wall colours', 'single ceiling area light and its fixture', 'camera', 'exposure'];
const ROOM_HELD = ['room shell with red left wall and green right wall, closed at the front behind the camera', 'white box and sphere', 'sun through the ceiling opening', 'camera'];
const VOLUME_HELD = 'global illumination volume and screen probes at quality 96/5';
const MOVING_HELD = 'camera swaying by at most 0.02 rad about the box centre';
const FACTS = ['snapshot:settings', 'snapshot:stats', 'runtime:ready', 'assertion:gi-active', 'assertion:ssgi-on', 'assertion:ssgi-pass'];
const ROOM_SELECTION = {
  camera: { eye: [0, 1.2, -1.6], target: [0.4, 0.35, -5.6], fovYDeg: 60, near: 0.05, far: 200 },
  subjects: [
    { name: 'box', min: [0.7, 0, -5.7], max: [1.7, 1, -4.7] },
    { name: 'sphere', min: [-1.8, 0, -5.2], max: [-0.6, 1.2, -4.0] },
    { name: 'floor-wall-corners', min: [-3, 0, -8], max: [3, 0.3, -4] },
  ],
};
// mode, arrangement, screen-space quality, motion, filename, criterionName
const ROWS = [
  ['cornell-on', 'cornell-box', 'high', false, 'gi-ssgi-cornell-box-on-contact-bleed-and-floor-lit-block-fronts.png', 'M8 screen-space GI: Cornell box contact bleed and floor-lit block fronts'],
  ['room-on', 'color-bleed', 'medium', false, 'gi-ssgi-room-on-box-and-sphere-bases-and-wall-corners-show-contact-bounce.png', 'M8 screen-space GI: room contact bounce at box, sphere and wall corners'],
  ['cornell-moving-on', 'cornell-box', 'medium', true, 'gi-ssgi-cornell-box-moving-camera-no-halos-ghosts-or-crawling-near-blocks.png', 'M8 screen-space GI: moving camera stability'],
];
const DESCRIPTIONS = {
  'cornell-on': 'The Cornell box with screen-space indirect light (visibility bitmask, quality high) on top of the voxel global illumination.',
  'room-on': 'The red-and-green room with screen-space indirect light (quality medium) on top of the voxel global illumination.',
  'cornell-moving-on': 'The Cornell box with screen-space indirect light (quality medium) while the camera sways slightly.',
};
const BEHAVIOUR = {
  'cornell-on': 'Screen-space indirect light adds near-field bounce detail on top of the voxel global illumination. Where the two white blocks stand on the floor, a soft, narrow contact band darkens the floor right at each block base, with no hard black line. Close to where surfaces meet, the floor and block faces near the red left wall pick up a red tint and those near the green right wall a green tint, and the floor-to-wall and wall-to-wall corners show a soft darkening band tinted by the adjacent wall. The front faces of the two blocks, which the ceiling light does not reach directly, are clearly lit by warm bounce light from the bright floor in front of them, brighter towards their bottom edge. No dark or bright halo outlines the blocks against the walls, there is no grain, speckle or banding, and nothing streaks or smears at the image edges. The reference is a path-traced Cornell box given as context for the scene: overall brightness, the overall strength of colour bleeding across the box, shadow shapes and voxel-scale smoothness of the far bounce are not faults here.',
  'room-on': 'Screen-space indirect light adds near-field bounce detail on top of the voxel global illumination in the red-and-green room. At the bases of the white box and the white sphere the floor shows a soft, narrow contact shadow, and the lower faces of the box and the underside of the sphere pick up soft bounce light from the floor. The floor-to-wall corners show a soft contact darkening tinted by the wall colour: red along the red left wall, green along the green right wall. No dark or bright halo outlines the box or the sphere, there is no grain, speckle or banding, and nothing streaks or smears at the image edges. The overall strength of colour bleeding across the room is not judged.',
  'cornell-moving-on': 'The camera sways slightly while screen-space indirect light is on over the voxel global illumination. The contact darkening at the block bases and the floor bounce on the block fronts stay stable and smooth: there is no crawling or flickering noise, no dark or bright halo outlining the blocks, no smeared, doubled or trailing ghost copy of the block edges or of their contact bands, and no streaks at the image edges where new floor comes into view. Overall brightness and colour-bleed differences from the path-traced reference are not faults.',
};
const HELD = {
  'cornell-on': [...CORNELL_HELD, VOLUME_HELD, 'screen-space indirect light quality high'],
  'room-on': [...ROOM_HELD, VOLUME_HELD, 'screen-space indirect light quality medium'],
  'cornell-moving-on': [...CORNELL_HELD, VOLUME_HELD, 'screen-space indirect light quality medium', MOVING_HELD],
};
const SETTLED = {
  active: true, inactiveReason: null, quality: { resolution: 96, levels: 5, probeRays: 8 },
  finestVoxelMetres: 0.2, voxelizedForwardInstances: 8, voxelizedVirtualInstances: 0,
  pendingRebuild: false, framesSinceRebuild: 240, allocatedBytes: 148635648, statsFrameIndex: 300, pending: false,
};
const PASS = { name: 'gi.ssgi', role: 'pass', enabled: true, order: 44, required: false };

const variantOf = id => ssgi.default.variants.find(v => v.id === id);
const named = (calls, name) => calls.filter(call => call[0] === name).map(call => call.slice(1));
const failed = result => result.assertions.filter(item => !item.passed).map(item => item.name);
// A recording engine: every method returns a fresh id and is logged in `calls`.
function recordingEngine() {
  const calls = [];
  let next = 1;
  return new Proxy({}, {
    get: (target, name) => (name === 'calls' ? calls : (...args) => { calls.push([name, ...args]); return next++; }),
  });
}
// A stub engine with the three methods observe reads; `settings` is what globalIllumination() reports.
function stub({ stats = SETTLED, settings = { screenSpace: true, screenSpaceQuality: 'high' }, passes = [PASS] } = {}) {
  return {
    giStats: () => ({ ...stats }),
    globalIllumination: () => ({ ...settings }),
    passes: () => passes.map(row => ({ ...row })),
  };
}
const observed = (id, engine) => ssgi.observe(engine, variantOf(id), { instances: [], moved: false, blockNode: null, frame: 0 });

test('the three inventory rows equal the contract rows, field by field', () => {
  assert.deepEqual(ssgi.inventory.map(row => row.id), ROWS.map(([mode]) => `ssgi-${mode}`));
  ROWS.forEach(([mode, arrangement, , , filename, criterionName], index) => {
    const row = ssgi.inventory[index];
    assert.equal(row.scenario, 'ssgi');
    assert.equal(row.mode, mode);
    assert.equal(row.filename, filename);
    assert.equal(row.criterionName, criterionName);
    assert.equal(row.expectedVisibleBehavior, BEHAVIOUR[mode]);
    assert.deepEqual(row.requiredFacts, FACTS);
    assert.equal(row.pair, null);
    assert.equal(row.producer, 'raa-gi-ssgi');
    assert.deepEqual(row.heldConstant, HELD[mode]);
    if (arrangement === 'cornell-box') {
      assert.equal(row.judgeReference, '/refs/cornell-box-reference.png');
      assert.equal(row.judgeAspect, 'functional');
    } else {
      assert.equal('judgeReference' in row, false);
      assert.equal('judgeAspect' in row, false);
    }
  });
});

test('the variants carry the contract gi object, quality, judge flags, selection and no measurement', () => {
  ROWS.forEach(([mode, arrangement, quality, motion]) => {
    const variant = variantOf(mode);
    assert.equal(variant.description, DESCRIPTIONS[mode]);
    assert.equal(variant.arrangement, arrangement);
    assert.deepEqual(variant.gi, { enabled: true, debugView: 'none', screenSpace: true, screenSpaceQuality: quality });
    assert.deepEqual(variant.quality, { resolution: 96, levels: 5, probeRays: 8 });
    assert.equal(variant.giExpected, 'active');
    assert.equal(variant.judge, true);
    assert.deepEqual(variant.references, []);
    assert.equal(variant.measurement, undefined);
    assert.equal(variant.motion, motion);
    assert.deepEqual(variant.selection, arrangement === 'cornell-box' ? lighting.CORNELL_SELECTION : ROOM_SELECTION);
  });
});

test('validateInventory accepts the lighting scenes and returns rows that include the three ids', () => {
  const rows = validateInventory([ssgi, restir, stochastic, area, lighting, volume, temporal], 'lighting');
  const ids = rows.map(row => row.id);
  for (const [mode] of ROWS) assert.ok(ids.includes(`ssgi-${mode}`), `missing ssgi-${mode}`);
});

test('build on a recording engine sets the quality and camera once, the GI patch last and the sway on the moving row only', async () => {
  for (const [mode, arrangement, , motion] of ROWS) {
    const engine = recordingEngine();
    const variant = variantOf(mode);
    const state = await ssgi.default.build(engine, {}, variant);
    const calls = engine.calls;
    assert.deepEqual(named(calls, 'setGiQuality'), [[{ resolution: 96, levels: 5, probeRays: 8 }]], mode);
    assert.deepEqual(named(calls, 'setCamera'), [[variant.selection.camera]], mode);
    assert.deepEqual(named(calls, 'setGlobalIllumination').at(-1), [variant.gi], mode);
    const orbits = named(calls, 'setOrbit');
    if (motion) assert.deepEqual(orbits, [[[2.78, 2.73, 0], 8, Math.PI, 0]], mode);
    else assert.equal(orbits.length, 0, mode);
    assert.equal(state.frame, 0, mode);
    const expected = arrangement === 'color-bleed' ? 'box' : 'cornell-white';
    assert.ok(state.instances.some(item => item.name === expected), mode);
  }
});

test('observe passes the three assertions in factNames order, and each defect fails its own assertion only', async () => {
  assert.deepEqual(ssgi.factNames, ['gi-active', 'ssgi-on', 'ssgi-pass']);
  for (const [mode, , quality] of ROWS) {
    const result = await observed(mode, stub({ settings: { screenSpace: true, screenSpaceQuality: quality } }));
    assert.deepEqual(result.assertions.map(item => item.name), ssgi.factNames);
    assert.deepEqual(failed(result), [], mode);
    assert.deepEqual(result.edits, []);
  }
  const cornell = 'cornell-on';
  assert.deepEqual(failed(await observed(cornell, stub({ settings: { screenSpace: false, screenSpaceQuality: 'high' } }))), ['ssgi-on']);
  assert.deepEqual(failed(await observed(cornell, stub({ settings: { screenSpace: true, screenSpaceQuality: 'low' } }))), ['ssgi-on']);
  assert.deepEqual(failed(await observed(cornell, stub({ passes: [{ ...PASS, order: 43 }] }))), ['ssgi-pass']);
  assert.deepEqual(failed(await observed(cornell, stub({ passes: [] }))), ['ssgi-pass']);
  assert.deepEqual(failed(await observed(cornell, stub({ stats: { ...SETTLED, pendingRebuild: true } }))), ['gi-active']);
});

test('the default export is the SSGI scene in the lighting stage, with the imported frame and ready', () => {
  assert.equal(ssgi.default.id, 'ssgi');
  assert.equal(ssgi.default.milestone, 'lighting');
  assert.equal(ssgi.default.frame, stochastic.frame);
  assert.deepEqual(ssgi.default.ready(), { ready: true, stage: 'lighting' });
});
