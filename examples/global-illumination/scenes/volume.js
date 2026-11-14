// Volume stage (S40) of the global-illumination page. Importing this module has no side
// effects, so node and the page can read its metadata. The stage is the closed 6 x 3 x 8 m
// room with a 1.2 m ceiling opening, a white box and sphere inside and a long ground plane.
// Each variant turns the GI volume on or off and picks a debug view; observe reports the GI
// evidence of contract 'gi-volume-rows'. Geometry comes from '../lib/content.js'.
import { box, plane, roomShell, sphere, volumeRoom } from '../lib/content.js';

const room = volumeRoom();
// The volume-levels rows look down on the room and the ground from the elevated camera.
const cameraSelection = { camera: room.camera, subjects: room.subjects };
const elevatedSelection = { camera: room.elevatedCamera, subjects: room.elevatedSubjects };
const heldConstant = ['room shell, box and sphere', 'ground plane', 'camera'];

const BEHAVIOURS = {
  voxels: 'Blocky voxel cubes about 0.2 m wide cover the visible surfaces, each cube coloured '
    + 'like the surface it samples: red cubes on the left wall, green cubes on the right wall and '
    + 'white cubes on the floor, back wall, box and sphere.',
  radiance: 'Voxel radiance view: the voxels in the sunlit floor patch under the ceiling opening '
    + 'are bright, and the voxels of the walls, floor, box and sphere away from the patch show '
    + 'dimmer bounce light, tinted red near the red wall and green near the green wall.',
  distance: 'Distance field view: repeating light and dark contour bands about 0.4 m apart run '
    + 'parallel to every wall, the floor, the box and the sphere.',
  levels: 'Seen from above, the room and the ground are tinted in concentric coloured bands '
    + 'nested around the point under the camera: red nearest, then orange, yellow, green and blue '
    + 'farther out.',
};
const OFF_BEHAVIOUR = 'No debug overlay: the surfaces render with normal smooth shading, without '
  + 'voxel blocks, contour bands or coloured level tints.';
const DESCRIPTIONS = {
  voxels: 'GI volume debug view: blocky 0.2 m voxel cubes coloured like the surfaces they sample.',
  radiance: 'GI volume debug view: voxel radiance bright under the ceiling opening and tinted bounce light on the walls.',
  distance: 'GI volume distance field: light and dark contour bands parallel to every surface.',
  levels: 'GI volume levels from above: concentric coloured bands nested around the point under the camera.',
  off: 'The same room with global illumination disabled: the P1-P3 forward frame with no debug overlay.',
};

// [debug view, selection, measurement preset]
const VIEWS = [
  ['voxels', cameraSelection, 'volume-room'],
  ['radiance', cameraSelection, 'volume-room'],
  ['distance', cameraSelection, 'volume-room'],
  ['levels', elevatedSelection, 'volume-room-elevated'],
];

const variant = ([view, selection, preset], state) => {
  const on = state === 'on';
  return {
    id: `${view}-${state}`,
    description: on ? DESCRIPTIONS[view] : DESCRIPTIONS.off,
    references: [],
    judge: true,
    gi: { enabled: on, debugView: on ? view : 'none' },
    giExpected: on ? 'active' : 'disabled',
    selection,
    measurement: {
      pair: `gi-volume-${view}`, state, preset,
      features: { globalIllumination: on, debugView: on ? view : 'none' },
    },
  };
};

const ON_FACTS = [
  'snapshot:settings', 'snapshot:quality', 'snapshot:stats', 'snapshot:instances',
  'runtime:passes', 'runtime:viewSizes', 'runtime:ready',
  'assertion:gi-active', 'assertion:debug-view',
];
const OFF_FACTS = [
  'snapshot:settings', 'snapshot:quality', 'snapshot:stats',
  'runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'assertion:gi-off',
];
const FILES = {
  voxels: ['gi-room-voxels-debug-shows-0-2m-blocks.png', 'gi-room-voxels-off-p3-frame.png'],
  radiance: ['gi-room-radiance-debug-bounce-filled.png', 'gi-room-radiance-off-p3-frame.png'],
  distance: ['gi-room-distance-debug-contours-follow-walls.png', 'gi-room-distance-off-p3-frame.png'],
  levels: ['gi-room-levels-debug-nested-around-camera.png', 'gi-room-levels-off-p3-frame.png'],
};
const row = ([view], state) => ({
  id: `volume-${view}-${state}`,
  criterionName: state === 'on' ? `P4-Q2 ${view} debug view` : 'P4-Q8 GI off frame',
  scenario: 'volume',
  mode: `${view}-${state}`,
  filename: FILES[view][state === 'on' ? 0 : 1],
  expectedVisibleBehavior: state === 'on' ? BEHAVIOURS[view] : OFF_BEHAVIOUR,
  requiredFacts: state === 'on' ? ON_FACTS : OFF_FACTS,
  pair: `gi-volume-${view}`,
  heldConstant,
  producer: state === 'on' ? 'raa-gi-volume' : 'raa-gi-core',
});

export const factNames = ['gi-active', 'gi-off', 'debug-view'];
export const inventory = VIEWS.flatMap(view => [row(view, 'on'), row(view, 'off')]);
const variants = VIEWS.flatMap(view => [variant(view, 'on'), variant(view, 'off')]);

const nodeAt = (engine, position = [0, 0, 0]) =>
  engine.addNode(null, position, [0, 0, 0, 1], [1, 1, 1]);
const meshWith = (engine, desc, material) => engine.addMesh({
  ...desc,
  subMeshes: [{ first: 0, count: desc.indices.length, material }],
});

/**
 * `build(engine, lib, variant)` builds the volume stage and returns the instance groups of
 * the four named objects. Today it sets the day sky, clear weather, the room's sun and the
 * low ambient, creates the red, green, white and ground materials, adds the room shell
 * (sub-meshes floor, ceiling, back, left, right with materials white, white, white, red,
 * green), the white box and sphere and the ground plane at y -0.42, sets the variant camera
 * and applies `variant.gi` through `setGlobalIllumination`.
 * Handoff: gi-page: page entry, build script and p4-gi commands calls
 * `build(engine, lib, variant) -> Promise<{ instances: Array<{ name: string, instance: number }> }>`
 * exported by examples/global-illumination/scenes/volume.js; the page loads the scene module
 * through scenes.json and createPageRunner attaches it by calling build before observe;
 * state/keys: the returned `instances` entries room, box, sphere and ground with their
 * instance-group handles.
 * Upstream: examples/global-illumination/lib/content.js:169 volumeRoom hands this stage its
 * room descriptor and builders (roomShell at :140, box at :55, plane at :65, sphere at :77).
 */
export async function build(engine, lib, variant) {
  const { colors, room: size, sun } = room;
  engine.setSky({ preset: 'day' });
  engine.setWeather({ preset: 'clear' });
  engine.setLight({ direction: sun.direction, color: sun.color, intensity: sun.intensity });
  engine.setAmbient({ color: [0.6, 0.65, 0.75], intensity: 0.05 });
  const material = color => engine.addMaterial({ color, roughness: 0.85 });
  const red = material(colors.red), green = material(colors.green);
  const white = material(colors.white), groundMaterial = material(colors.ground);
  const shell = roomShell(size);
  const wallMaterials = [white, white, white, red, green];
  const roomMesh = engine.addMesh({
    ...shell,
    subMeshes: shell.subMeshes.map((part, index) => ({ ...part, material: wallMaterials[index] })),
  });
  const instances = [
    { name: 'room', instance: engine.addMeshInstance(nodeAt(engine), roomMesh) },
    { name: 'box', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, box(room.box.center, room.box.half), white)) },
    { name: 'sphere', instance: engine.addMeshInstance(nodeAt(engine, room.sphere.center), meshWith(engine, sphere(room.sphere.radius, room.sphere.segments), white)) },
    { name: 'ground', instance: engine.addMeshInstance(nodeAt(engine, [0, room.ground.y, 0]), meshWith(engine, plane(room.ground.width, room.ground.depth), groundMaterial)) },
  ];
  engine.setCamera(variant.selection.camera);
  engine.setGlobalIllumination(variant.gi);
  return { instances };
}

const check = (name, passed, detail) => ({ name, passed: passed === true, detail });

/**
 * `observe(engine, variant, state)` reports the GI evidence of the built volume stage.
 * Today it reads `engine.giStats()` and `engine.globalIllumination()` and returns the three
 * contract assertions with a detail that names every value it read. `gi-active` requires an
 * active volume with no pending rebuild, at least 240 frames since the rebuild, the 96/5
 * allocation and the eight voxelized forward instances (five room sub-meshes, box, sphere
 * and ground). `gi-off` requires an inactive volume with reason 'disabled' and no
 * allocation. `debug-view` compares the accepted debug view with the variant's. Assertions
 * that do not apply to the variant are still reported with the same computation.
 * Handoff: gi-page: volume.test.mjs calls
 * `observe(engine: object, variant: object, state: { instances: Array<{ name: string, instance: number }> }) -> Promise<{ instances: Array<{ name: string, instance: number }>, assertions: Array<{ name: string, passed: boolean, detail: string }>, edits: [] }>`
 * exported by examples/global-illumination/scenes/volume.js; the test attaches it by
 * importing this module's namespace and calling observe on a stub engine; state/keys: this
 * module's `inventory`, `factNames` and `default.variants`, the `state.instances` entries
 * (room, box, sphere, ground) and the assertion names gi-active, gi-off and debug-view.
 */
export async function observe(engine, variant, state) {
  const stats = engine.giStats();
  const settings = engine.globalIllumination();
  const assertions = [
    check('gi-active',
      stats.active === true && stats.pendingRebuild === false && stats.framesSinceRebuild >= 240
        && stats.allocatedBytes >= 148635648 && stats.voxelizedForwardInstances >= 8,
      `active ${stats.active}, pendingRebuild ${stats.pendingRebuild}, framesSinceRebuild ${stats.framesSinceRebuild}, `
        + `allocatedBytes ${stats.allocatedBytes}, voxelizedForwardInstances ${stats.voxelizedForwardInstances}`),
    check('gi-off',
      stats.active === false && stats.inactiveReason === 'disabled' && stats.allocatedBytes === 0,
      `active ${stats.active}, inactiveReason ${stats.inactiveReason}, allocatedBytes ${stats.allocatedBytes}`),
    check('debug-view',
      settings.debugView === variant.gi.debugView,
      `debugView ${settings.debugView}, expected ${variant.gi.debugView}`),
  ];
  return { instances: state.instances, assertions, edits: [] };
}

export default {
  id: 'volume',
  milestone: 'volume',
  title: 'GI volume debug views',
  variants,
  build,
};
