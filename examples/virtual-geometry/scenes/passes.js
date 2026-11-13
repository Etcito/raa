// Milestone-2 passes scene of the virtual-geometry page (S33 `passes` stage): seven presets, their
// on/off pairs, the Q15 toggle variants, the runtime-bake states and the four debug views. The
// inventory below is the only source of selection, filenames and judge descriptions. No imports and
// no top-level side effect: the page passes its helpers as `lib` (raa, content, bakeSection) and the
// Engine methods are the m29 ones. Every variant uses one viewport and `engine: { msaa: 1 }`.

const SETS = {
  field: ['ground plane', 'day sky preset', 'camera', 'same bumpy sphere material', 'sun direction and local lights'],
  lod: ['ground plane', 'day sky preset', 'camera', 'five copies of one bumpy sphere at fixed places', 'sun direction'],
  walls: ['ground plane', 'day sky preset', 'camera', 'wall placement and size', 'field of bumpy spheres behind the walls'],
  shadows: ['ground plane', 'day sky preset', 'camera', 'sun direction', 'forward slab and wall placement'],
  mixed: ['ground plane', 'day sky preset', 'camera', 'three local lights', 'order and placement of the seven objects'],
  bake: ['ground plane', 'day sky preset', 'camera', 'one bumpy sphere and its material', 'sun direction'],
};
const HELD = { 'dense-field': SETS.field, 'continuous-lod': SETS.lod, occlusion: SETS.walls, shadows: SETS.shadows, mixed: SETS.mixed, 'runtime-bake': SETS.bake, 'debug-views': SETS.lod };
const FIELD = 'the dense field of bumpy spheres is fully drawn, lit by the sun with soft cascade shadows, fogged toward the horizon, local light pools visible and the bloom and grading applied';
const LOD = 'five copies of one bumpy sphere recede from near to far with a smooth silhouette and no crack, hole or popping seam; the nearest shows the finest surface detail';
const OCCLUSION = 'three large walls stand in front of the field and hide most of the spheres; only the walls and the few spheres in front of them are visible';
const SHADOWS = 'the bumpy spheres cast soft shadows on the ground and on the wall, and the forward slab casts a shadow across a sphere, all in the sun cascades';
const MIXED = 'three bumpy spheres on the left sit beside four forward objects of the same size: a skinned one, a squashed one, a transparent one and a furry one, all lit by the same three local lights and correctly in front of or behind each other';
const BAKE = 'one bumpy sphere on the ground in the sun';
const DEBUG = 'the same five bumpy spheres as the continuous detail scene, coloured by the debug view';

// [variant id, scene, virtual-geometry settings (null = disabled), pair, filename stem, criterion, expected behaviour, assertions]
const ROWS = [
  ['dense-field-on', 'dense-field', {}, 'dense-field', 'dense-field-vg-on-many-high-poly-instances', 'Dense field through virtualized geometry', `${FIELD}.`, ['vg-active']],
  ['dense-field-off', 'dense-field', null, 'dense-field', 'dense-field-vg-off-forward-path', 'Dense field on the forward path', `${FIELD}, drawn by the forward path.`, ['vg-inactive-forward']],
  ['dense-field-hardware-only', 'dense-field', { softwareRaster: false }, null, 'dense-field-vg-on-hardware-raster-only', 'Dense field with hardware raster only', `${FIELD}; every cluster uses the hardware rasteriser.`, ['vg-active']],
  ['continuous-lod-on', 'continuous-lod', {}, 'continuous-lod', 'continuous-lod-vg-on-five-distances-no-cracks', 'One mesh at five distances', `${LOD}.`, ['vg-active', 'lod-triangles-fall-with-distance']],
  ['continuous-lod-off', 'continuous-lod', null, 'continuous-lod', 'continuous-lod-vg-off-forward-path', 'One mesh at five distances on forward', `${LOD}, drawn by the forward path.`, ['vg-inactive-forward']],
  ['occlusion-on', 'occlusion', {}, 'occlusion', 'occlusion-vg-on-walls-hide-most-instances', 'Walls hide most instances', `${OCCLUSION}.`, ['vg-active', 'occlusion-reduces-drawn-clusters']],
  ['occlusion-off', 'occlusion', null, 'occlusion', 'occlusion-vg-off-forward-path', 'Walls hide most instances on forward', `${OCCLUSION}, drawn by the forward path.`, ['vg-inactive-forward']],
  ['occlusion-test-off', 'occlusion', { occlusion: false }, null, 'occlusion-vg-on-occlusion-test-off-same-content', 'Occlusion test off shows the same content', `${OCCLUSION}; the picture equals the one with the occlusion test on.`, ['vg-active']],
  ['shadows-on', 'shadows', {}, 'shadows', 'shadows-vg-on-cast-and-received-in-cascades', 'Shadows cast and received in the cascades', `${SHADOWS}.`, ['vg-active', 'shadow-clusters-drawn']],
  ['shadows-off', 'shadows', null, 'shadows', 'shadows-vg-off-forward-path', 'Shadows on the forward path', `${SHADOWS}, drawn by the forward path.`, ['vg-inactive-forward']],
  ['shadows-vg-shadows-off', 'shadows', { shadows: false }, null, 'shadows-vg-on-virtual-casters-off', 'Virtualized casters off', 'the bumpy spheres cast no shadow on the ground or on the wall; the forward slab still casts its shadow, which falls across a sphere.', ['vg-active']],
  ['mixed-on', 'mixed', {}, 'mixed', 'mixed-vg-on-beside-forward-skinned-transparent-fur', 'Virtualized beside forward content', `${MIXED}.`, ['vg-active', 'refusals-reported']],
  ['mixed-off', 'mixed', null, 'mixed', 'mixed-vg-off-forward-path', 'Mixed content on forward', `${MIXED}, all drawn by the forward path.`, ['vg-inactive-forward']],
  ['runtime-bake-baking', 'runtime-bake', {}, null, 'runtime-bake-forward-while-baking', 'Mesh visible on forward while it bakes', `${BAKE}, drawn by the forward path while its clusters bake.`, ['bake-states-in-order']],
  ['runtime-bake-on', 'runtime-bake', {}, 'runtime-bake', 'runtime-bake-virtual-when-ready', 'Mesh virtualized once baked', `${BAKE}, drawn through virtualized geometry once the bake finished.`, ['vg-active', 'bake-states-in-order']],
  ['runtime-bake-off', 'runtime-bake', null, 'runtime-bake', 'runtime-bake-vg-off-forward-path', 'Runtime bake mesh on forward', `${BAKE}, drawn by the forward path.`, ['vg-inactive-forward']],
  ['debug-clusters-on', 'continuous-lod', { debugView: 'clusters' }, 'debug-views', 'debug-clusters-coloured-patches', 'Debug view: clusters', `${DEBUG}: each cluster is a small patch of its own flat colour.`, ['vg-active', 'debug-view-set']],
  ['debug-lod', 'continuous-lod', { debugView: 'lod' }, null, 'debug-lod-level-ramp-by-distance', 'Debug view: level of detail', `${DEBUG}: the colour ramps with the level of detail, one colour band per level, changing with distance.`, ['vg-active', 'debug-view-set']],
  ['debug-raster', 'continuous-lod', { debugView: 'raster' }, null, 'debug-raster-hardware-versus-software', 'Debug view: raster', `${DEBUG}: clusters drawn by the hardware rasteriser and by the software rasteriser have two clearly different colours.`, ['vg-active', 'debug-view-set']],
  ['debug-occlusion', 'continuous-lod', { debugView: 'occlusion' }, null, 'debug-occlusion-first-versus-second-phase', 'Debug view: occlusion', `${DEBUG}: clusters drawn in the first phase and clusters recovered by the second phase have two different colours.`, ['vg-active', 'debug-view-set']],
  ['debug-views-off', 'continuous-lod', null, 'debug-views', 'debug-views-off-normal-shading', 'Debug views off', `${DEBUG.replace('coloured by the debug view', 'with normal shading')}.`, ['vg-inactive-forward', 'debug-view-set']],
];

const camera = (eye, target, fovYDeg = 50, far = 200) => ({ eye, target, fovYDeg, near: .1, far });
const box = (name, min, max) => ({ name, min, max });
const ball = (name, [x, y, z], r = 1.06) => box(name, [x - r, y - r, z - r], [x + r, y + r, z + r]);
const SUN = [-.4, -.6, -.65];
const SELECTION = {
  'dense-field': { camera: camera([0, 7, 24], [0, 0, -6]), subjects: [box('field', [-19, 0, -19], [19, 1, 19])] },
  'continuous-lod': {
    camera: camera([0, 2, 7], [0, 1, -50], 50, 300),
    // The copies fan out to the right with distance so that none hides behind a nearer one
    // from this camera (their projected discs stay at least 27 px apart at 1280x720).
    subjects: [ball('copy-1', [-3, 1.06, 0]), ball('copy-2', [-.5, 1.06, -10]), ball('copy-3', [4, 1.06, -24]), ball('copy-4', [12, 1.06, -45]), ball('copy-5', [26, 1.06, -80])],
  },
  occlusion: { camera: camera([0, 1.6, 14], [0, 1.2, -4]), subjects: [box('walls', [-9, 0, 3.85], [9, 3, 4.15]), box('hidden-field', [-12.5, 0, -15], [12.5, 1.1, -1.5]), box('front-row', [-4.5, 0, 7], [4.5, 1.1, 9])] },
  shadows: {
    camera: camera([0, 3.2, 9], [0, .9, -1]),
    subjects: [ball('left-sphere', [-1.8, 1.06, 0]), ball('right-sphere', [1.8, 1.06, 0]), box('forward-slab', [-3.5, 0, 1.25], [-2.9, 3, 1.55]), box('forward-wall', [0, 0, -3.65], [3, 2.4, -3.35])],
  },
  mixed: { camera: camera([1.5, 3, 16], [1.5, 1, 0]), subjects: [-7.5, -4.5, -1.5, 1.5, 4.5, 7.5, 10.5].map((x, i) => ball(`object-${i + 1}`, [x, 1.06, 0])) },
  'runtime-bake': { camera: camera([0, 2, 6], [0, 1, 0]), subjects: [ball('sphere', [0, 1.06, 0])] },
};
const variants = ROWS.map(([id, scene, vg, pair, , , description]) => ({
  id, scene, vg, description, references: [], judge: true, selection: SELECTION[scene],
  ...(pair === null ? {} : { measurement: { pair, state: id.endsWith('-on') ? 'on' : 'off', preset: pair, features: { virtualGeometry: vg !== null } } }),
}));
const ASSERTIONS = Object.fromEntries(ROWS.map(row => [row[0], row[7]]));
export const factNames = ['vg-active', 'vg-inactive-forward', 'lod-triangles-fall-with-distance', 'occlusion-reduces-drawn-clusters', 'shadow-clusters-drawn', 'refusals-reported', 'bake-states-in-order', 'debug-view-set'];
const row = ([id, scene, vg, pair, stem, criterionName, expectedVisibleBehavior, assertions]) => ({
  id: stem, criterionName, scenario: 'passes', mode: id, filename: `${stem}.png`, expectedVisibleBehavior,
  heldConstant: HELD[scene === 'continuous-lod' && id.startsWith('debug') ? 'debug-views' : scene],
  requiredFacts: [
    'snapshot:settings', 'snapshot:stats', 'snapshot:instances', 'runtime:passes', 'runtime:viewSizes', 'runtime:ready',
    ...assertions.map(name => `assertion:${name}`), ...(scene === 'runtime-bake' ? ['snapshot:meshes'] : []), ...(pair === null ? [] : [`measurement:${pair}`]),
  ],
  pair,
});
export const inventory = ROWS.map(row);

const meshOf = (engine, desc, material) => engine.addMesh({ ...desc, subMeshes: [{ first: 0, count: desc.indices.length, material }] });
const put = (engine, mesh, position, scale = [1, 1, 1]) => engine.addMeshInstance(engine.addNode(null, position, [0, 0, 0, 1], scale), mesh);
const forward = (engine, instance) => {
  engine.setInstanceVirtualGeometry(instance, false);
  return instance;
};
const note = (state, name, instance) => { state.instances.push({ name, instance }); return instance; };

function environment(engine, selected, { fog = false, post = false, lights = [], sun = SUN } = {}) {
  engine.setCamera(selected.selection.camera);
  engine.setSky({ preset: 'day' });
  engine.setLight({ direction: sun, color: [1, 1, 1], intensity: 3 });
  engine.setPassEnabled('shadow.csm', true);
  if (fog) {
    engine.setFog({ density: .01, heightFalloff: .1, baseHeight: 0 });
    engine.setPassEnabled('fog', true);
  }
  if (post) {
    engine.setBloom({ strength: .25 });
    engine.setGrading({ preset: 'neutral', contrast: 1.05, saturation: 1.1 });
    engine.setPassEnabled('bloom', true);
    engine.setPassEnabled('grading', true);
  }
  for (const position of lights) engine.addLight({ kind: 'point', position, color: [1, .8, .55], intensity: 4, range: 9 });
}

// The ground stays on the forward path in every variant, so a pair differs only by the feature.
function ground(engine, lib, state, scale = [1, 1, 1]) {
  const material = engine.addSurfaceMaterial({ color: [.35, .4, .3], roughness: .95 });
  note(state, 'ground', forward(engine, put(engine, meshOf(engine, lib.content.groundPlane(), material), [0, 0, 0], scale)));
}

// One high-poly mesh with its clusters attached once when the variant virtualizes.
function sphereMesh(engine, lib, variant, state, material, attach = true) {
  const desc = lib.content.highPolyMesh(), mesh = meshOf(engine, desc, material);
  if (attach && variant.vg !== null) {
    engine.attachMeshClusters(mesh, [lib.bakeSection('high-poly', desc)]);
    state.clustered.push(mesh);
  }
  state.meshes.push({ name: 'high-poly', mesh });
  return mesh;
}

const BUILDERS = {
  'dense-field'(engine, lib, variant, state) {
    environment(engine, variant, { fog: true, post: true, lights: [[-6, 2.2, 8], [7, 2.2, 0], [0, 2.2, -10]] });
    ground(engine, lib, state);
    const mesh = sphereMesh(engine, lib, variant, state, engine.addSurfaceMaterial({ color: [.75, .7, .62], roughness: .6 }));
    for (let i = 0; i < 1024; i++) {
      const instance = put(engine, mesh, [(i % 32 - 15.5) * 1.2, .45, (Math.floor(i / 32) - 15.5) * 1.2], [.45, .45, .45]);
      if (i === 0 || i === 1023) note(state, `field-${i}`, instance);
    }
  },
  'continuous-lod'(engine, lib, variant, state) {
    environment(engine, variant);
    ground(engine, lib, state, [5, 1, 5]);
    const mesh = sphereMesh(engine, lib, variant, state, engine.addSurfaceMaterial({ color: [.75, .7, .62], roughness: .6 }));
    state.copies = SELECTION['continuous-lod'].subjects.map(subject => note(state, subject.name, put(engine, mesh, [(subject.min[0] + subject.max[0]) / 2, 1.06, (subject.min[2] + subject.max[2]) / 2])));
  },
  occlusion(engine, lib, variant, state) {
    environment(engine, variant);
    ground(engine, lib, state, [2, 1, 2]);
    const mesh = sphereMesh(engine, lib, variant, state, engine.addSurfaceMaterial({ color: [.75, .7, .62], roughness: .6 }));
    const wall = meshOf(engine, lib.content.occluderWall(), engine.addSurfaceMaterial({ color: [.55, .5, .5], roughness: .9 }));
    [-6, 0, 6].forEach(x => note(state, `wall-${x}`, forward(engine, put(engine, wall, [x, 0, 4]))));
    for (let i = 0; i < 52; i++) put(engine, mesh, [(i % 13 - 6) * 2, .53, -2 - 4 * Math.floor(i / 13)], [.5, .5, .5]);
    [-4, 0, 4].forEach(x => note(state, `front-${x}`, put(engine, mesh, [x, .53, 8], [.5, .5, .5])));
  },
  shadows(engine, lib, variant, state) {
    environment(engine, variant, { sun: [.45, -.6, -.65] });
    ground(engine, lib, state, [.5, 1, .5]);
    const mesh = sphereMesh(engine, lib, variant, state, engine.addSurfaceMaterial({ color: [.75, .7, .62], roughness: .6 }));
    const wall = meshOf(engine, lib.content.occluderWall(), engine.addSurfaceMaterial({ color: [.6, .55, .5], roughness: .9 }));
    note(state, 'left-sphere', put(engine, mesh, [-1.8, 1.06, 0]));
    note(state, 'right-sphere', put(engine, mesh, [1.8, 1.06, 0]));
    note(state, 'forward-slab', forward(engine, put(engine, wall, [-3.2, 0, 1.4], [.1, 1, 1])));
    note(state, 'forward-wall', forward(engine, put(engine, wall, [1.5, 0, -3.5], [.5, .8, 1])));
  },
  mixed(engine, lib, variant, state) {
    environment(engine, variant, { lights: [[-4, 3, 4], [3, 3, 4], [9, 3, 4]] });
    ground(engine, lib, state, [1, 1, 1]);
    const plain = engine.addSurfaceMaterial({ color: [.75, .7, .62], roughness: .6 });
    const mesh = sphereMesh(engine, lib, variant, state, plain);
    [-7.5, -4.5, -1.5].forEach((x, i) => note(state, `virtual-${i + 1}`, put(engine, mesh, [x, 1.06, 0])));
    const desc = lib.content.highPolyMesh(), count = desc.positions.length / 3;
    const rig = new Uint16Array(4 * count), weights = new Float32Array(4 * count);
    for (let i = 0; i < count; i++) weights[4 * i] = 1;
    const skinned = engine.addMesh({ ...desc, joints: rig, weights, subMeshes: [{ first: 0, count: desc.indices.length, material: plain }] });
    const skin = put(engine, skinned, [1.5, 1.06, 0]);
    engine.setSkinPose(skin, new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]));
    note(state, 'skinned', skin);
    const squash = put(engine, mesh, [4.5, 1.06, 0]);
    engine.setSquash(squash, { amount: .25 });
    note(state, 'deformed', squash);
    const glass = engine.addSurfaceMaterial({ color: [.4, .7, .9], roughness: .2, blend: 'transparent', opacity: .5 });
    note(state, 'transparent', put(engine, meshOf(engine, desc, glass), [7.5, 1.06, 0]));
    const fur = engine.addSurfaceMaterial({ color: [.7, .5, .3], roughness: .9, furLength: .06, furShells: 16 });
    note(state, 'fur', put(engine, meshOf(engine, desc, fur), [10.5, 1.06, 0]));
  },
  'runtime-bake'(engine, lib, variant, state) {
    environment(engine, variant);
    ground(engine, lib, state);
    engine.setClusterBakeBudget(.5);
    // No clusters are attached: the mesh bakes at run time when the feature is on.
    const mesh = sphereMesh(engine, lib, variant, state, engine.addSurfaceMaterial({ color: [.75, .7, .62], roughness: .6 }), false);
    note(state, 'high-poly', put(engine, mesh, [0, 1.06, 0]));
    state.bakeMesh = mesh;
    if (variant.vg !== null) state.clustered.push(mesh);
    state.samples = [engine.meshClusters(mesh)];
  },
};

const check = (name, passed, detail) => ({ name, passed: passed === true, detail });
const MAX_FRAMES = 240;
const frame = engine => {
  engine.render(performance.now());
  return new Promise(resolve => requestAnimationFrame(resolve));
};
// Renders until two newer statistics frames are published and no readback is outstanding (the
// first may predate the change just made); null when MAX_FRAMES frames are not enough.
async function settle(engine) {
  let last = engine.virtualGeometryStats().statsFrameIndex ?? -1, advances = 0;
  for (let n = 0; n < MAX_FRAMES; n++) {
    await frame(engine);
    const stats = engine.virtualGeometryStats();
    if (stats.statsFrameIndex !== null && stats.statsFrameIndex > last) { last = stats.statsFrameIndex; advances++; }
    if (advances >= 2 && !stats.pending) return stats;
  }
  return null;
}
const number = value => typeof value === 'number';

const ASSERT = {
  'vg-active'(engine) {
    const stats = engine.virtualGeometryStats();
    return check('vg-active', stats.active === true && stats.virtualInstances > 0, `active ${stats.active}, virtual instances ${stats.virtualInstances}`);
  },
  'vg-inactive-forward'(engine) {
    const stats = engine.virtualGeometryStats();
    return check('vg-inactive-forward', stats.active === false && stats.inactiveReason === 'disabled', `active ${stats.active}, reason ${stats.inactiveReason}`);
  },
  async 'lod-triangles-fall-with-distance'(engine, variant, state) {
    const only = async index => {
      state.copies.forEach((copy, i) => engine.setVisible(copy, i === index));
      return settle(engine);
    };
    const near = await only(0), far = await only(state.copies.length - 1);
    state.copies.forEach(copy => engine.setVisible(copy, true));
    if (!near || !far) return check('lod-triangles-fall-with-distance', false, `statistics not published within ${MAX_FRAMES} frames`);
    return check('lod-triangles-fall-with-distance', number(near.drawnTriangles) && number(far.drawnTriangles) && far.drawnTriangles < near.drawnTriangles, `near ${near.drawnTriangles} triangles, far ${far.drawnTriangles}`);
  },
  async 'occlusion-reduces-drawn-clusters'(engine) {
    const tested = await settle(engine);
    engine.setVirtualGeometry({ occlusion: false });
    const untested = await settle(engine);
    engine.setVirtualGeometry({ occlusion: true });
    if (!tested || !untested) return check('occlusion-reduces-drawn-clusters', false, `statistics not published within ${MAX_FRAMES} frames`);
    return check('occlusion-reduces-drawn-clusters', number(tested.drawnClusters) && number(untested.drawnClusters) && tested.drawnClusters < untested.drawnClusters, `occlusion on ${tested.drawnClusters} clusters, off ${untested.drawnClusters}`);
  },
  'shadow-clusters-drawn'(engine) {
    const stats = engine.virtualGeometryStats();
    return check('shadow-clusters-drawn', number(stats.shadowClusters) && stats.shadowClusters > 0, `shadow clusters ${stats.shadowClusters}`);
  },
  'refusals-reported'(engine) {
    const refused = engine.virtualGeometryStats().refused ?? {};
    const codes = ['vg.skinned_unsupported', 'vg.deformed_unsupported', 'vg.transparent_material'];
    return check('refusals-reported', codes.every(code => refused[code] > 0), `refused ${JSON.stringify(refused)}`);
  },
  'bake-states-in-order'(engine, variant, state) {
    const samples = state.samples, rank = { none: 0, baking: 1, ready: 2 };
    const baking = samples.filter(sample => sample.state === 'baking');
    const ordered = samples.every((sample, i) => sample.state in rank && (i === 0 || rank[sample.state] >= rank[samples[i - 1].state]));
    const rising = baking.every((sample, i) => sample.progress < 1 && (i === 0 || sample.progress >= baking[i - 1].progress));
    const last = samples.at(-1), instance = engine.instanceVirtualGeometry(state.instances.find(i => i.name === 'high-poly').instance);
    const settled = variant.id === 'runtime-bake-baking'
      ? last.state === 'baking' && instance.path === 'forward' && instance.pending === true
      : last.state === 'ready' && instance.path === 'virtual' && instance.pending === false;
    return check('bake-states-in-order', samples[0].state === 'none' && baking.length > 0 && ordered && rising && settled, `states ${samples.map(s => s.state).filter((s, i, all) => s !== all[i - 1]).join(' -> ')}; baking samples ${baking.length}; instance ${instance.path}, pending ${instance.pending}`);
  },
  'debug-view-set'(engine, variant) {
    const expected = variant.vg?.debugView ?? 'none', actual = engine.virtualGeometry().debugView;
    return check('debug-view-set', actual === expected, `debug view ${actual}, expected ${expected}`);
  },
};

export async function observe(engine, variant, state) {
  const assertions = [];
  for (const name of ASSERTIONS[variant.id]) assertions.push(await ASSERT[name](engine, variant, state));
  return {
    meshes: state.meshes.map(({ name, mesh }) => ({ name, mesh, clusters: null, cuts: [], states: (state.samples ?? []).map(({ state: s, progress }) => ({ state: s, progress })) })),
    instances: state.instances,
    assertions,
    edits: [],
  };
}

export default {
  id: 'passes', milestone: 'passes', title: 'Virtualized geometry in the frame: seven presets with on/off pairs',
  engine: { msaa: 1 },
  variants,
  async build(engine, lib, variant) {
    const state = { scene: variant.scene, instances: [], meshes: [], clustered: [], copies: [], samples: null, bakeMesh: null };
    BUILDERS[variant.scene](engine, lib, variant, state);
    if (variant.vg !== null) engine.setVirtualGeometry({ enabled: true, ...variant.vg });
    return state;
  },
  // Readiness conditions only: mesh state, active feature and a published statistics frame.
  ready(engine, state, variant) {
    if (state.samples) state.samples.push(engine.meshClusters(state.bakeMesh));
    if (variant.id === 'runtime-bake-baking') {
      const sample = state.samples.at(-1);
      return { ready: sample.state === 'baking' && sample.progress > 0, stage: sample.state };
    }
    if (variant.vg === null) return { ready: true, stage: 'forward' };
    const stats = engine.virtualGeometryStats();
    const baked = state.clustered.every(mesh => engine.meshClusters(mesh).state === 'ready');
    return { ready: baked && stats.active === true && stats.statsFrameIndex !== null, stage: baked ? (stats.active ? 'active' : 'inactive') : 'baking' };
  },
};
