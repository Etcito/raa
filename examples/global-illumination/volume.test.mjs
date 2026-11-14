import test from 'node:test';
import assert from 'node:assert/strict';
import * as volume from './scenes/volume.js';
import { validateInventory } from './lib/inventory.js';
import { box, plane, roomShell, sphere, volumeRoom } from './lib/content.js';

// Volume stage of the global-illumination page, checked against contract gi-volume-rows. The
// literals below repeat that contract; they are not read back from scenes/volume.js. Node only:
// no engine, no WebGPU.

const VIEWS = ['voxels', 'radiance', 'distance', 'levels'];
const PRESET = { voxels: 'volume-room', radiance: 'volume-room', distance: 'volume-room', levels: 'volume-room-elevated' };
const HELD = ['room shell, box and sphere', 'ground plane', 'camera'];
const ON_FACTS = ['snapshot:settings', 'snapshot:quality', 'snapshot:stats', 'snapshot:instances',
  'runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'assertion:gi-active', 'assertion:debug-view'];
const OFF_FACTS = ['snapshot:settings', 'snapshot:quality', 'snapshot:stats',
  'runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'assertion:gi-off'];
const BEHAVIOUR = {
  voxels: 'Blocky voxel cubes about 0.2 m wide cover the visible surfaces, each cube coloured like the surface it samples: red cubes on the left wall, green cubes on the right wall and white cubes on the floor, back wall, box and sphere.',
  radiance: 'Voxel radiance view: the voxels in the sunlit floor patch under the ceiling opening are bright, and the voxels of the walls, floor, box and sphere away from the patch show dimmer bounce light, tinted red near the red wall and green near the green wall.',
  distance: 'Distance field view: repeating light and dark contour bands about 0.4 m apart run parallel to every wall, the floor, the box and the sphere.',
  levels: 'Seen from above, the room and the ground are tinted in concentric coloured bands nested around the point under the camera: red nearest, then orange, yellow, green and blue farther out.',
  off: 'No debug overlay: the surfaces render with normal smooth shading, without voxel blocks, contour bands or coloured level tints.',
};
// [id, mode, filename] of the eight rows, in contract order.
const ROWS = [
  ['volume-voxels-on', 'voxels-on', 'gi-room-voxels-debug-shows-0-2m-blocks.png'],
  ['volume-voxels-off', 'voxels-off', 'gi-room-voxels-off-p3-frame.png'],
  ['volume-radiance-on', 'radiance-on', 'gi-room-radiance-debug-bounce-filled.png'],
  ['volume-radiance-off', 'radiance-off', 'gi-room-radiance-off-p3-frame.png'],
  ['volume-distance-on', 'distance-on', 'gi-room-distance-debug-contours-follow-walls.png'],
  ['volume-distance-off', 'distance-off', 'gi-room-distance-off-p3-frame.png'],
  ['volume-levels-on', 'levels-on', 'gi-room-levels-debug-nested-around-camera.png'],
  ['volume-levels-off', 'levels-off', 'gi-room-levels-off-p3-frame.png'],
];
const FILENAME = /^[a-z0-9]+(?:-[a-z0-9]+)+\.png$/;

const variantOf = id => volume.default.variants.find(v => v.id === id);
const pick = (object, keys) => Object.fromEntries(keys.map(key => [key, object[key]]));
const expectedRow = (view, state) => {
  const on = state === 'on';
  return {
    criterionName: on ? `P4-Q2 ${view} debug view` : 'P4-Q8 GI off frame',
    scenario: 'volume',
    expectedVisibleBehavior: on ? BEHAVIOUR[view] : BEHAVIOUR.off,
    requiredFacts: on ? ON_FACTS : OFF_FACTS,
    heldConstant: HELD,
    producer: on ? 'raa-gi-volume' : 'raa-gi-core',
    pair: `gi-volume-${view}`,
  };
};

test('scene exports: identity, factNames and variant order', () => {
  assert.equal(volume.default.id, 'volume');
  assert.equal(volume.default.milestone, 'volume');
  assert.equal(volume.default.title, 'GI volume debug views');
  assert.equal(typeof volume.default.build, 'function');
  assert.deepEqual(volume.factNames, ['gi-active', 'gi-off', 'debug-view']);
  assert.deepEqual(volume.default.variants.map(v => v.id), ROWS.map(([, mode]) => mode));
});

test('inventory rows: ids, modes and filenames equal the contract list in order', () => {
  assert.deepEqual(volume.inventory.map(r => [r.id, r.mode, r.filename]), ROWS);
  for (const row of volume.inventory) {
    assert.match(row.filename, FILENAME);
    assert.equal(row.scenario, 'volume');
  }
});

test('variants carry the contract gi patch, giExpected, selection and measurement', () => {
  const room = volumeRoom();
  for (const view of VIEWS) {
    for (const state of ['on', 'off']) {
      const on = state === 'on';
      const elevated = view === 'levels';
      const v = variantOf(`${view}-${state}`);
      assert.deepEqual(v.gi, on ? { enabled: true, debugView: view } : { enabled: false, debugView: 'none' });
      assert.equal(v.giExpected, on ? 'active' : 'disabled');
      assert.deepEqual(v.selection, {
        camera: elevated ? room.elevatedCamera : room.camera,
        subjects: elevated ? room.elevatedSubjects : room.subjects,
      });
      assert.deepEqual(v.measurement, {
        pair: `gi-volume-${view}`,
        state,
        preset: PRESET[view],
        features: { globalIllumination: on, debugView: on ? view : 'none' },
      });
      assert.ok(v.description.length > 0);
      assert.deepEqual(v.references, []);
    }
  }
});

test('each debug view pairs one -on and one -off variant and row on the same preset', () => {
  for (const view of VIEWS) {
    const variants = volume.default.variants.filter(v => v.measurement?.pair === `gi-volume-${view}`);
    assert.deepEqual(variants.map(v => v.id), [`${view}-on`, `${view}-off`]);
    assert.deepEqual(variants.map(v => v.measurement.state), ['on', 'off']);
    assert.deepEqual(variants.map(v => v.measurement.preset), [PRESET[view], PRESET[view]]);
    const rows = volume.inventory.filter(r => r.pair === `gi-volume-${view}`);
    assert.deepEqual(rows.map(r => r.mode), [`${view}-on`, `${view}-off`]);
  }
});

test('rows carry the contract criterion, behaviour, facts, held constants, producer and pair', () => {
  for (const row of volume.inventory) {
    const [view, state] = row.mode.split('-');
    const expected = expectedRow(view, state);
    assert.equal(row.id, `volume-${row.mode}`);
    assert.deepEqual(pick(row, Object.keys(expected)), expected);
  }
});

test('requiredFacts: -on rows add the GI and debug-view assertions, -off rows omit snapshot:instances', () => {
  for (const row of volume.inventory) {
    if (row.mode.endsWith('-on')) {
      assert.ok(row.requiredFacts.includes('assertion:gi-active'));
      assert.ok(row.requiredFacts.includes('assertion:debug-view'));
    } else {
      assert.ok(row.requiredFacts.includes('assertion:gi-off'));
      assert.ok(!row.requiredFacts.includes('snapshot:instances'));
    }
  }
});

test('validateInventory accepts the volume stage and returns its eight rows', () => {
  const rows = validateInventory([volume], 'volume');
  assert.deepEqual(rows.map(r => r.id), ROWS.map(([id]) => id));
});

test('validateInventory refuses a levels-on camera turned away from its subjects', () => {
  const { eye, target } = volumeRoom().elevatedCamera;
  // Reflect the target through the eye: the subjects now lie behind the camera.
  const behind = target.map((t, i) => 2 * eye[i] - t);
  const turnedAway = v => v.id !== 'levels-on' ? v
    : { ...v, selection: { ...v.selection, camera: { ...v.selection.camera, target: behind } } };
  const turned = { ...volume.default, variants: volume.default.variants.map(turnedAway) };
  assert.throws(() => validateInventory([{ ...volume, default: turned }], 'volume'), /outside camera frustum/);
});

test('room shell has five sub-meshes; room, box, plane and sphere meshes index triangles in range', () => {
  const room = volumeRoom();
  const shell = roomShell(room.room);
  assert.equal(shell.subMeshes.length, 5);
  const meshes = [
    shell,
    box(room.box.center, room.box.half),
    plane(room.ground.width, room.ground.depth),
    sphere(room.sphere.radius, room.sphere.segments),
  ];
  for (const mesh of meshes) {
    const vertices = mesh.positions.length / 3;
    assert.ok(mesh.positions instanceof Float32Array && mesh.normals instanceof Float32Array);
    assert.ok(mesh.indices instanceof Uint32Array);
    assert.equal(mesh.indices.length % 3, 0);
    assert.ok(mesh.indices.every(i => i < vertices));
  }
  for (const part of shell.subMeshes) {
    assert.ok(part.count > 0 && part.first % 3 === 0 && part.count % 3 === 0);
    assert.ok(part.first + part.count <= shell.indices.length);
  }
});

const SETTLED = {
  active: true, inactiveReason: null, quality: { resolution: 96, levels: 5, probeRays: 8 },
  finestVoxelMetres: 0.2, voxelizedForwardInstances: 8, voxelizedVirtualInstances: 0,
  pendingRebuild: false, framesSinceRebuild: 240, allocatedBytes: 148635648, statsFrameIndex: 300, pending: false,
};
const DISABLED = {
  ...SETTLED, active: false, inactiveReason: 'disabled', voxelizedForwardInstances: 0,
  framesSinceRebuild: 0, allocatedBytes: 0,
};
const INSTANCES = [
  { name: 'room', instance: 0 }, { name: 'box', instance: 1 }, { name: 'sphere', instance: 2 }, { name: 'ground', instance: 3 },
];
const stubEngine = (stats, debugView) => ({
  giStats: () => ({ ...stats }),
  globalIllumination: () => ({
    enabled: stats.active, intensity: 1, environmentDiffuse: 0,
    localLightShadows: true, reflections: true, screenTraces: true, debugView,
  }),
});
const observed = (stats, debugView, id) =>
  volume.observe(stubEngine(stats, debugView), variantOf(id), { instances: INSTANCES });
const outcome = result => Object.fromEntries(result.assertions.map(a => [a.name, a.passed]));

test('observe on a settled stub passes gi-active and debug-view and refuses gi-off', async () => {
  const result = await observed(SETTLED, 'voxels', 'voxels-on');
  assert.deepEqual(result.assertions.map(a => a.name), ['gi-active', 'gi-off', 'debug-view']);
  assert.deepEqual(outcome(result), { 'gi-active': true, 'gi-off': false, 'debug-view': true });
  assert.deepEqual(result.instances, INSTANCES);
  assert.deepEqual(result.edits, []);
});

test('observe on the disabled stub passes gi-off and refuses gi-active', async () => {
  const result = await observed(DISABLED, 'none', 'voxels-off');
  assert.deepEqual(outcome(result), { 'gi-active': false, 'gi-off': true, 'debug-view': true });
});

test('observe refuses gi-active on a pending, young or under-allocated volume and debug-view on a wrong view', async () => {
  const patches = [{ pendingRebuild: true }, { framesSinceRebuild: 239 }, { allocatedBytes: 148635647 }, { voxelizedForwardInstances: 7 }];
  for (const patch of patches) {
    const result = await observed({ ...SETTLED, ...patch }, 'voxels', 'voxels-on');
    assert.equal(outcome(result)['gi-active'], false, JSON.stringify(patch));
  }
  const wrong = await observed(SETTLED, 'radiance', 'voxels-on');
  assert.equal(outcome(wrong)['debug-view'], false);
});
