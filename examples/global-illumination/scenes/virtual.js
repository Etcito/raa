// Virtual stage (S40, contract 'gi-virtual-rows') of the global-illumination page: driver B
// virtualized geometry voxelized through P3 culling into the shared GI grid. Importing this
// module has no side effects. A ground plane, a red wall on the left, sixteen bumpy high-poly
// spheres from one clustered mesh (driver B) and, on the shared-grid variants, two boxes opted
// out of virtualized geometry (driver A), both drivers filling the one GI grid. The
// sphere mesh comes from the shared virtual-geometry content; the rest from '../lib/content.js'.
import { highPolyMesh } from '../../virtual-geometry/lib/content.js';
import { box, plane, volumeRoom } from '../lib/content.js';

const sun = volumeRoom().sun;
const SUN_DIRECTION = [-0.55, -0.75, -0.35];
const CAMERA = { eye: [1.5, 7, 15], target: [1.5, 1, 0], fovYDeg: 50, near: 0.1, far: 200 };
const SUBJECTS = [
  { name: 'ground', min: [-20, 0, -20], max: [20, 0, 20] },
  { name: 'wall', min: [-6.25, 0, -6], max: [-5.75, 4, 6] },
  { name: 'field', min: [-2.6, 0, -4.85], max: [7.1, 2.2, 4.85] },
];
const BOX_SUBJECT = { name: 'boxes', min: [-4.8, 0, -3.1], max: [-3.6, 1.2, 3.1] };
// GI clipmap level 0 is the 0.2 m level: 96 voxels = 19.2 m wide, centred on the camera snapped
// to 1.6 m (raa-gi `clipmap::snapped_origin`), and a point within 1.5 voxels of its faces is
// traced in level 1 (`raa_gi_level_at`). From CAMERA (z 15) level 0 spans z 4.8..24: the wall,
// the boxes and every sphere lie in the 0.4 m level 1. The voxel debug view walks level 0 only
// (`debug_view.wgsl`), so it showed them smooth, and the lit row's bounce came from 0.4 m voxels
// only. Every shared-grid row (lit, off and voxels; the on/off pair keeps one camera) looks down
// on the stage from GRID_CAMERA instead: its eye snaps to (0, 8, 3.2), so level 0 spans
// x -9.6..9.6, y -1.6..17.6 and z -6.4..12.8 and holds the wall, the boxes and the sixteen
// spheres whole; the spheres, the boxes and the wall's half next to them are framed in the
// 1280x720 view. Level 0 can hold the 12 m wall only while the eye stays at z < 4.8, hence the
// high, steep view. Pinned by crates/raa-gi/tests/virtual_variants.rs.
const GRID_CAMERA = { eye: [0.4, 9, 4.5], target: [0.8, 0.5, -0.5], fovYDeg: 72, near: 0.1, far: 200 };
const SELECTIONS = {
  field: { camera: CAMERA, subjects: SUBJECTS },
  'shared-grid': { camera: GRID_CAMERA, subjects: [...SUBJECTS, BOX_SUBJECT] },
};
const FIELD_FIXTURES = ['ground plane', 'red wall panel on the left',
  'sixteen bumpy high-poly spheres in a 4 x 4 grid', 'day sky preset', 'sun direction'];
const FIELD_HELD = [...FIELD_FIXTURES, 'camera'];
const GRID_HELD = [...FIELD_FIXTURES,
  'high camera looking steeply down on the stage from above the front row',
  'two white forward boxes beside the wall'];
const FIELD_X = [-1.5, 1, 3.5, 6];
const FIELD_Z = [-3.75, -1.25, 1.25, 3.75];

const DESCRIPTIONS = {
  'field-on': 'Dense virtualized high-poly field beside a red wall, lit by voxel GI bounce light.',
  'field-off': 'The same virtualized field with global illumination off.',
  'shared-grid-on': 'Virtualized field and forward boxes in one GI voxel grid, both bounce-lit.',
  'shared-grid-off': 'The shared-grid scene with global illumination off.',
  'shared-grid-voxels': 'GI voxel debug view of the virtualized field and the forward boxes.',
};
const BEHAVIOURS = {
  'field-on': 'The sixteen bumpy spheres are lit by bounce light: the sides that face the red wall '
    + 'on the left, which the sun does not reach, show a clear red tint from light bounced off the '
    + 'sunlit wall, strongest on the spheres nearest the wall and fading toward the right; their '
    + 'shadowed undersides and the ground between them are softly lit, not black. The sphere '
    + 'surfaces are complete, with no holes, missing patches or cracks.',
  'field-off': 'Global illumination is off, the P1-P3 virtualized frame: the sides of the spheres '
    + 'that face the red wall are dark grey with no red tint, the shadowed undersides are darker '
    + 'than in a bounce-lit image, and the sphere surfaces are complete, with no holes or cracks.',
  'shared-grid-on': 'The bumpy spheres and the two white boxes beside the red wall are lit by the '
    + 'same bounce light: the box faces turned toward the red wall and the wall-facing sides of the '
    + 'nearest spheres show a similar red tint, and the shadowed sides of both are softly lit '
    + 'rather than black, with no light leaks or blotches.',
  'shared-grid-off': 'Global illumination is off: the faces of the boxes and the sides of the '
    + 'spheres turned toward the red wall are dark with no red tint; direct sun, sky and ambient '
    + 'light only.',
  'shared-grid-voxels': 'GI voxel debug view: the bumpy spheres, the two boxes, the red wall and '
    + 'the ground all appear as blocky voxel cubes about 0.2 m wide in one aligned grid, white '
    + 'cubes on the spheres and boxes, red cubes on the wall and grey cubes on the ground; every '
    + 'sphere is covered by cubes like the boxes, none is left smooth or missing from the voxels.',
};
const OFF_FACTS = ['snapshot:settings', 'snapshot:quality', 'snapshot:stats',
  'runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'assertion:gi-off', 'assertion:vg-active'];
const ON_FACTS = ['snapshot:settings', 'snapshot:quality', 'snapshot:stats', 'snapshot:instances',
  'runtime:passes', 'runtime:viewSizes', 'runtime:ready',
  'assertion:gi-active', 'assertion:vg-active', 'assertion:virtual-voxelized'];
const GRID_ON_FACTS = [...ON_FACTS, 'assertion:forward-voxelized'];
const VOXEL_FACTS = [...GRID_ON_FACTS, 'assertion:debug-view'];

// [id, content, GI state, debug view, measurement pair]
const VARIANTS = [
  ['field-on', 'field', 'on', 'none', 'gi-virtual-field'],
  ['field-off', 'field', 'off', 'none', 'gi-virtual-field'],
  ['shared-grid-on', 'shared-grid', 'on', 'none', 'gi-virtual-shared-grid'],
  ['shared-grid-off', 'shared-grid', 'off', 'none', 'gi-virtual-shared-grid'],
  ['shared-grid-voxels', 'shared-grid', 'on', 'voxels', null],
];
const variant = ([id, content, state, debugView, pair]) => ({
  id,
  description: DESCRIPTIONS[id],
  references: [],
  judge: true,
  content,
  gi: { enabled: state === 'on', debugView },
  giExpected: state === 'on' ? 'active' : 'disabled',
  selection: SELECTIONS[content],
  ...(pair === null ? {} : {
    measurement: {
      pair, state, preset: pair,
      features: { globalIllumination: state === 'on', virtualGeometry: true },
    },
  }),
});
const variants = VARIANTS.map(variant);

// [mode, filename, criterionName, required facts, pair, producer]
const ROWS = [
  ['field-on', 'gi-virtual-field-bumpy-spheres-take-red-bounce-from-wall.png',
    'P4-Q7 VG field lit by bounce light', ON_FACTS, 'gi-virtual-field', 'raa-gi-virtual'],
  ['field-off', 'gi-virtual-field-off-p3-frame-no-bounce.png',
    'P4-Q8 GI off frame of the VG field', OFF_FACTS, 'gi-virtual-field', 'raa-gi-core'],
  ['shared-grid-on', 'gi-virtual-shared-grid-spheres-and-boxes-share-red-bounce.png',
    'P4-Q7 VG and forward share one grid', GRID_ON_FACTS, 'gi-virtual-shared-grid', 'raa-gi-virtual'],
  ['shared-grid-off', 'gi-virtual-shared-grid-off-p3-frame-no-bounce.png',
    'P4-Q8 GI off frame of the shared grid', OFF_FACTS, 'gi-virtual-shared-grid', 'raa-gi-core'],
  ['shared-grid-voxels', 'gi-virtual-shared-grid-voxels-spheres-and-boxes-same-0-2m-cubes.png',
    'P4-Q7 VG field in the voxels', VOXEL_FACTS, null, 'raa-gi-virtual'],
];
const row = ([mode, filename, criterionName, requiredFacts, pair, producer]) => ({
  id: `virtual-${mode}`,
  criterionName,
  scenario: 'virtual',
  mode,
  filename,
  expectedVisibleBehavior: BEHAVIOURS[mode],
  requiredFacts,
  pair,
  heldConstant: mode.startsWith('shared-grid') ? GRID_HELD : FIELD_HELD,
  producer,
});

export const factNames = ['gi-active', 'gi-off', 'vg-active', 'virtual-voxelized',
  'forward-voxelized', 'debug-view'];
export const inventory = ROWS.map(row);

const nodeAt = (engine, position = [0, 0, 0]) =>
  engine.addNode(null, position, [0, 0, 0, 1], [1, 1, 1]);
const meshWith = (engine, desc, material) => engine.addMesh({
  ...desc,
  subMeshes: [{ first: 0, count: desc.indices.length, material }],
});

/**
 * `build(engine, lib, variant)` builds the virtual stage and returns its instance groups.
 * Today it sets the day sky, clear weather, the stage sun and the low ambient, creates the
 * ground, red-wall and white materials, adds the ground plane at y 0, the red wall at x -6,
 * the one bumpy sphere mesh with its baked clusters attached, the sixteen field instances and,
 * for content 'shared-grid', the two boxes without clusters, each opted out of virtualized
 * geometry with `setInstanceVirtualGeometry(box, false)` so that they stay on the forward path
 * and GI driver A voxelizes them beside the driver B spheres; it then sets the variant
 * camera, enables virtualized geometry and starts global illumination disabled.
 */
export async function build(engine, lib, variant) {
  engine.setSky({ preset: 'day' });
  engine.setWeather({ preset: 'clear' });
  engine.setLight({ direction: SUN_DIRECTION, color: sun.color, intensity: sun.intensity });
  engine.setAmbient({ color: [0.6, 0.65, 0.75], intensity: 0.05 });
  const ground = engine.addMaterial({ color: [0.55, 0.55, 0.55], roughness: 0.9 });
  const red = engine.addMaterial({ color: [0.8, 0.1, 0.08], roughness: 0.85 });
  const white = engine.addMaterial({ color: [0.85, 0.85, 0.85], roughness: 0.6 });
  const instances = [
    { name: 'ground', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, plane(40, 40), ground)) },
    { name: 'wall', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, box([-6, 2, 0], [0.25, 2, 6]), red)) },
  ];
  const desc = highPolyMesh();
  const field = meshWith(engine, desc, white);
  engine.attachMeshClusters(field, [lib.raa.bakeClusters({
    positions: desc.positions, normals: desc.normals, uvs: desc.uvs, indices: desc.indices,
  })]);
  const fieldInstances = [];
  for (const z of FIELD_Z) {
    for (const x of FIELD_X) {
      const instance = engine.addMeshInstance(nodeAt(engine, [x, 1.1, z]), field);
      instances.push({ name: `field-${fieldInstances.length}`, instance });
      fieldInstances.push(instance);
    }
  }
  const boxInstances = [];
  if (variant.content === 'shared-grid') {
    for (const [name, z] of [['box-a', -2.5], ['box-b', 2.5]]) {
      const instance = engine.addMeshInstance(nodeAt(engine),
        meshWith(engine, box([-4.2, 0.6, z], [0.6, 0.6, 0.6]), white));
      // Driver A content: the box is opted out of virtualized geometry, so the runtime bake
      // (instanceDefault true) never moves it to the virtual path and GI voxelizes it forward.
      engine.setInstanceVirtualGeometry(instance, false);
      instances.push({ name, instance });
      boxInstances.push(instance);
    }
  }
  engine.setCamera(variant.selection.camera);
  engine.setVirtualGeometry({ enabled: true });
  engine.setGlobalIllumination({ enabled: false, debugView: 'none' });
  return { instances, clustered: [field], fieldInstances, boxInstances, giApplied: false };
}

/**
 * `ready(engine, state, variant)` is the readiness gate of the virtual stage. Today it throws
 * `virtual: cluster bake refused: <code> <message>` for a refused mesh of `state.clustered`,
 * reports stage 'baking' while a mesh is not ready, reports stage 'vg-pending' while
 * `virtualGeometryStats()` is inactive or has no published stats frame, and otherwise applies
 * `variant.gi` through `setGlobalIllumination` once (setting `state.giApplied`) and reports
 * stage 'vg-active' ready, so GI is allocated only after virtualized geometry runs.
 */
export function ready(engine, state, variant) {
  for (const mesh of state.clustered) {
    const clusters = engine.meshClusters(mesh);
    if (clusters.state === 'refused') {
      throw new Error(`virtual: cluster bake refused: ${clusters.code} ${clusters.message}`);
    }
  }
  if (state.clustered.some(mesh => engine.meshClusters(mesh).state !== 'ready')) {
    return { ready: false, stage: 'baking' };
  }
  const stats = engine.virtualGeometryStats();
  if (stats.active !== true || stats.statsFrameIndex === null) {
    return { ready: false, stage: 'vg-pending' };
  }
  if (!state.giApplied) {
    engine.setGlobalIllumination(variant.gi);
    state.giApplied = true;
  }
  return { ready: true, stage: 'vg-active' };
}

const check = (name, passed, detail) => ({ name, passed: passed === true, detail });

/**
 * `observe(engine, variant, state)` reports the six assertions of contract 'gi-virtual-rows'
 * for the built virtual stage. Today it reads `engine.giStats()`, `globalIllumination()`,
 * `virtualGeometryStats()` and `instanceGlobalIllumination(instance)` for every state field and
 * box instance, and returns `{ instances: state.instances, assertions, edits: [] }` with each
 * detail naming every value it read: gi-active (active, no pending rebuild, >= 240 frames since
 * the rebuild, the 96/5 allocation), gi-off (inactive, 'disabled', no allocation), vg-active
 * (>= 16 virtual instances), virtual-voxelized, forward-voxelized and debug-view.
 * Handoff: card 'GI page: virtual.test.mjs and the p4-gi-virtual-* commands' calls
 * `observe(engine, variant, state) -> Promise<{ instances, assertions, edits: [] }>` exported
 * here, together with `build(engine, lib, variant) -> Promise<{ instances, clustered,
 * fieldInstances, boxInstances, giApplied }>` and `default.ready(engine, state, variant) ->
 * { ready: boolean, stage: string }`; state/keys: this module's `inventory` (5 rows keyed by
 * id), `factNames` (6 assertion names) and `default.variants` (5 variants keyed by id), and the
 * state fields instances, clustered, fieldInstances, boxInstances and giApplied.
 */
export async function observe(engine, variant, state) {
  const stats = engine.giStats();
  const settings = engine.globalIllumination();
  const vg = engine.virtualGeometryStats();
  const field = state.fieldInstances.map(instance => engine.instanceGlobalIllumination(instance));
  const boxes = state.boxInstances.map(instance => engine.instanceGlobalIllumination(instance));
  const assertions = [
    check('gi-active',
      stats.active === true && stats.pendingRebuild === false && stats.framesSinceRebuild >= 240
        && stats.allocatedBytes >= 148635648,
      `active ${stats.active}, pendingRebuild ${stats.pendingRebuild}, framesSinceRebuild ${stats.framesSinceRebuild}, `
        + `allocatedBytes ${stats.allocatedBytes}`),
    check('gi-off',
      stats.active === false && stats.inactiveReason === 'disabled' && stats.allocatedBytes === 0,
      `active ${stats.active}, inactiveReason ${stats.inactiveReason}, allocatedBytes ${stats.allocatedBytes}`),
    check('vg-active',
      vg.active === true && vg.virtualInstances >= 16,
      `active ${vg.active}, virtualInstances ${vg.virtualInstances}`),
    check('virtual-voxelized',
      stats.voxelizedVirtualInstances > 0 && field.every(entry => entry.driver === 'virtual'),
      `voxelizedVirtualInstances ${stats.voxelizedVirtualInstances}, `
        + `field drivers ${field.map(entry => entry.driver).join(',')}`),
    check('forward-voxelized',
      stats.voxelizedForwardInstances > 0 && state.boxInstances.length === 2
        && boxes.every(entry => entry.driver === 'forward'),
      `voxelizedForwardInstances ${stats.voxelizedForwardInstances}, boxes ${boxes.length}, `
        + `box drivers ${boxes.map(entry => entry.driver).join(',')}`),
    check('debug-view',
      settings.debugView === variant.gi.debugView,
      `debugView ${settings.debugView}, expected ${variant.gi.debugView}`),
  ];
  return { instances: state.instances, assertions, edits: [] };
}

export default {
  id: 'virtual',
  milestone: 'virtual',
  title: 'Driver B: virtualized geometry in the GI voxels',
  engine: { msaa: 1 },
  variants,
  build,
  ready,
};
