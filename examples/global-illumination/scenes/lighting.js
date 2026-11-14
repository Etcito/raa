// Lighting stage (S40) of the global-illumination page, rows 1-10: colour bleeding, interior
// bounce, the emissive panel, local-light shadows and the sky overhang. Importing this module has
// no side effects, so node and the page can read its metadata. Every row is an on/off pair on one
// arrangement, built from lib/content.js builders; observe reports the evidence of contract
// 'gi-lighting-rows'. Rows 11-19 are appended to ROWS and ARRANGEMENTS by the next share.
import { box, panel, plane, roomShell, sphere, volumeRoom } from '../lib/content.js';

const room = volumeRoom();
const DEFAULT_QUALITY = { resolution: 96, levels: 5, probeRays: 8 };
const VOLUME_BYTES = {
  '96/5': 148635648, '96/4': 120324096, '96/3': 92012544,
  '64/5': 44040192, '64/4': 35651584, '64/3': 27262976,
};
const ON_FACTS = [
  'snapshot:settings', 'snapshot:quality', 'snapshot:stats', 'snapshot:instances',
  'runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'assertion:gi-active',
];
const OFF_FACTS = [
  'snapshot:settings', 'snapshot:quality', 'snapshot:stats',
  'runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'assertion:gi-off',
];

export const factNames = [
  'gi-active', 'gi-off', 'local-lights', 'ssr-enabled', 'quality-applied', 'forward-content', 'block-moved',
];

/**
 * `ROWS` lists the rows in table order (contract gi-lighting-rows). Each entry's `measured` flag
 * gives it a PAIR_RUN measurement, and its `extraFacts` are appended to ON_FACTS or OFF_FACTS.
 * Today ROWS holds rows 1-10 only, and `ready` (a member of the default export) returns
 * { ready: true, stage: 'lighting' } for every variant.
 * Handoff: gi page: scenes/lighting.js rows 11-19 (glossy, quality, forward content, moved block)
 * appends its rows to `ROWS` (examples/global-illumination/scenes/lighting.js:37), its arrangements
 * to `ARRANGEMENTS` and replaces the moved-block logic of `ready`; state/keys: today `build` returns
 * `moved: false` and `blockNode: null` for every row, and no arrangement moves a block.
 */
const ROWS = [
  { mode: 'color-bleed-on', arrangement: 'color-bleed', filename: 'gi-color-bleed-white-box-tinted-red-green.png',
    criterionName: 'P4-Q3 colour bleeding', producer: 'raa-gi-screen', extraFacts: [], measured: true },
  { mode: 'color-bleed-off', arrangement: 'color-bleed', filename: 'gi-color-bleed-off-white-box-neutral.png',
    criterionName: 'P4-Q8 GI off frame', producer: 'raa-gi-core', extraFacts: [], measured: true },
  { mode: 'interior-bounce-on', arrangement: 'interior-bounce', filename: 'gi-interior-bounce-lit-from-sunlit-patch.png',
    criterionName: 'P4-Q3 interior bounce', producer: 'raa-gi-screen', extraFacts: [], measured: true },
  { mode: 'interior-bounce-off', arrangement: 'interior-bounce', filename: 'gi-interior-bounce-off-dark-interior.png',
    criterionName: 'P4-Q8 GI off frame', producer: 'raa-gi-core', extraFacts: [], measured: true },
  { mode: 'emissive-panel-on', arrangement: 'emissive-panel', filename: 'gi-emissive-panel-lights-nearby-surfaces.png',
    criterionName: 'P4-Q3 emissive panel light', producer: 'raa-gi-screen', extraFacts: [], measured: true },
  { mode: 'emissive-panel-off', arrangement: 'emissive-panel', filename: 'gi-emissive-panel-off-no-spill.png',
    criterionName: 'P4-Q8 GI off frame', producer: 'raa-gi-core', extraFacts: [], measured: true },
  { mode: 'local-light-shadow-on', arrangement: 'local-light-shadow', filename: 'gi-local-lights-stopped-by-wall.png',
    criterionName: 'P4-Q4 local lights stopped by a wall', producer: 'raa-gi-screen',
    extraFacts: ['assertion:local-lights'], measured: true },
  { mode: 'local-light-shadow-off', arrangement: 'local-light-shadow', filename: 'gi-local-lights-off-leak-through-wall.png',
    criterionName: 'P4-Q8 GI off frame', producer: 'raa-gi-core', extraFacts: ['assertion:local-lights'], measured: true },
  { mode: 'sky-overhang-on', arrangement: 'sky-overhang', filename: 'gi-sky-open-ground-kept-overhang-bounce.png',
    criterionName: 'P4-Q5 sky ground kept and overhang bounce', producer: 'raa-gi-screen', extraFacts: [], measured: true },
  { mode: 'sky-overhang-off', arrangement: 'sky-overhang', filename: 'gi-sky-overhang-off.png',
    criterionName: 'P4-Q8 GI off frame', producer: 'raa-gi-core', extraFacts: [], measured: true },
];

// Judge-held fixtures per arrangement: the judge never scores them.
const ROOM_HELD = ['room shell with red left wall and green right wall', 'white box and sphere',
  'sun through the ceiling opening', 'camera'];
const HELD = {
  'color-bleed': ROOM_HELD,
  'interior-bounce': ROOM_HELD,
  'emissive-panel': ['room shell', 'emissive panel', 'white box and sphere', 'night sky without sun', 'camera'],
  'local-light-shadow': ['floor', '0.6 m wall', 'point and spot light placement', 'night sky without sun', 'camera'],
  'sky-overhang': ['open ground', '3 m overhang on four pillars', 'white box under the overhang',
    'physical day sky and sun', 'camera'],
};

// Camera and subject bounds per arrangement, in metres (camera near 0.05, far 200).
const lens = (eye, target, fovYDeg) => ({ eye, target, fovYDeg, near: 0.05, far: 200 });
const roomSubject = { name: 'room', min: [-3.4, -0.4, -8.4], max: [3.4, 3.4, 0] };
const SELECTIONS = {
  'color-bleed': {
    camera: lens([0, 1.4, -1.2], [0, 0.6, -5.4], 60),
    subjects: [roomSubject, { name: 'box', min: [0.7, 0, -5.7], max: [1.7, 1, -4.7] }],
  },
  'interior-bounce': {
    camera: lens([2.4, 1.5, -1.0], [-2.0, 1.0, -7.5], 70),
    subjects: [roomSubject, { name: 'sphere', min: [-1.8, 0, -5.2], max: [-0.6, 1.2, -4.0] }],
  },
  'emissive-panel': {
    camera: lens([0, 1.6, -1.5], [0, 0.8, -7], 65),
    subjects: [
      { name: 'panel', min: [-0.8, 0.4, -7.45], max: [0.8, 1.4, -7.35] },
      { name: 'box', min: [0.7, 0, -6.7], max: [1.7, 1, -5.7] },
    ],
  },
  'local-light-shadow': {
    camera: lens([0, 2.0, 1.8], [0, 0.4, -4], 65),
    subjects: [
      { name: 'floor-front', min: [-3, 0, -3.7], max: [3, 0, 1] },
      { name: 'wall', min: [-3, 0, -4.3], max: [3, 2.5, -3.7] },
    ],
  },
  'sky-overhang': {
    camera: lens([7, 2.2, 8], [0, 1.0, 0], 60),
    subjects: [
      { name: 'ground', min: [-10, 0, -10], max: [10, 0, 10] },
      { name: 'overhang', min: [-2.2, 0, -2.2], max: [2.2, 3.3, 2.2] },
    ],
  },
};

// Quality per mode; a mode without an entry uses DEFAULT_QUALITY.
const QUALITY = {};

const BEHAVIOURS = {
  'color-bleed-on': 'The white box and the white floor beside the coloured walls take a visible red tint '
    + 'on the side facing the red left wall and a green tint on the side facing the green right wall.',
  'color-bleed-off': 'The white box and the white floor stay neutral white-grey, with no red or green tint '
    + 'picked up from the coloured walls.',
  'interior-bounce-on': 'The walls, ceiling and objects away from the bright sunlit floor patch are lit by soft '
    + 'bounce light coming from that patch, so the shadowed interior is clearly visible.',
  'interior-bounce-off': 'Away from the bright sunlit floor patch the interior stays dark, lit only by a faint '
    + 'flat ambient term.',
  'emissive-panel-on': 'The floor, the back wall and the white box and sphere next to the glowing panel are lit '
    + 'by warm light spilling from it, brightest close to the panel and fading with distance.',
  'emissive-panel-off': 'Only the panel itself glows; the floor, walls, box and sphere around it stay dark, with '
    + 'no light spilling from the panel.',
  'local-light-shadow-on': 'The floor on the camera side of the wall stays dark in the wall shadow: the point and '
    + 'spot lights behind the wall light only the far side, whose glow shows past the edges of the wall.',
  'local-light-shadow-off': 'Light from the point and spot lights behind the wall leaks through it: the floor on '
    + 'the camera side of the wall is lit as if the wall were not there.',
  'sky-overhang-on': 'The sunlit open ground keeps its normal daylight brightness, while the ground and the white '
    + 'box under the overhang receive visible soft bounce light and are clearly brighter than plain shadow.',
  'sky-overhang-off': 'The ground and the white box under the overhang sit in flat dark shadow lit only by the '
    + 'ambient light of the sky.',
};
const DESCRIPTIONS = {
  'color-bleed-on': 'Global illumination on: the white box and floor take bounce light from the red and green walls.',
  'color-bleed-off': 'The same red and green room with global illumination disabled: no coloured bounce light.',
  'interior-bounce-on': 'Global illumination on: the shadowed interior is lit by bounce light from the sunlit patch.',
  'interior-bounce-off': 'The same interior with global illumination disabled: only the flat ambient term lights it.',
  'emissive-panel-on': 'Global illumination on: the emissive panel spills warm light onto the nearby surfaces.',
  'emissive-panel-off': 'The same night room with global illumination disabled: the panel glows and lights nothing.',
  'local-light-shadow-on': 'Global illumination on: a point and a spot light behind a 0.6 m wall are stopped by it.',
  'local-light-shadow-off': 'The same lights with global illumination disabled: they leak through the wall.',
  'sky-overhang-on': 'Global illumination on under the physical day sky: the overhang region gains soft bounce light.',
  'sky-overhang-off': 'The same sky and overhang with global illumination disabled: flat shadow under the overhang.',
};

const nodeAt = (engine, position = [0, 0, 0]) => engine.addNode(null, position, [0, 0, 0, 1], [1, 1, 1]);
const meshWith = (engine, desc, material) => engine.addMesh({
  ...desc,
  subMeshes: [{ first: 0, count: desc.indices.length, material }],
});
const matte = (engine, color) => engine.addMaterial({ color, roughness: 0.85 });
const pairOf = row => (row.measured ? `gi-lighting-${row.arrangement}` : null);

// Joins MeshDesc parts into one MeshDesc, so that one instance group holds all of them.
function joinMeshes(parts) {
  const total = key => parts.reduce((sum, part) => sum + part[key].length, 0);
  const out = {
    positions: new Float32Array(total('positions')),
    normals: new Float32Array(total('normals')),
    indices: new Uint32Array(total('indices')),
  };
  let vertex = 0, index = 0;
  for (const part of parts) {
    out.positions.set(part.positions, vertex * 3);
    out.normals.set(part.normals, vertex * 3);
    out.indices.set(part.indices.map(i => i + vertex), index);
    vertex += part.positions.length / 3;
    index += part.indices.length;
  }
  return out;
}

// Clear weather, the room's sun and ambient; the night variants have no sun and a dim ambient.
function environment(engine, night) {
  engine.setSky({ preset: night ? 'night' : 'day' });
  engine.setWeather({ preset: 'clear' });
  engine.setLight({ direction: room.sun.direction, color: room.sun.color, intensity: night ? 0 : room.sun.intensity });
  engine.setAmbient({ color: [0.6, 0.65, 0.75], intensity: night ? 0.02 : 0.05 });
}

/**
 * The shared room of volume.js: sky, sun and ambient, the walls (sub-meshes floor, ceiling, back, left,
 * right with materials white, white, white, red, green) on roomShell(room.room) and the 'room' instance.
 * Returns the white material for the objects placed in the room.
 */
function roomScene(engine, state, { night = false } = {}) {
  environment(engine, night);
  const red = matte(engine, room.colors.red), green = matte(engine, room.colors.green);
  const white = matte(engine, room.colors.white);
  const shell = roomShell(room.room);
  const wallMaterials = [white, white, white, red, green];
  const roomMesh = engine.addMesh({
    ...shell,
    subMeshes: shell.subMeshes.map((part, index) => ({ ...part, material: wallMaterials[index] })),
  });
  state.instances.push({ name: 'room', instance: engine.addMeshInstance(nodeAt(engine), roomMesh) });
  return white;
}

// The room with a white box and a white sphere: the colour-bleed and interior-bounce arrangements.
function roomWithObjects(engine, variant, state) {
  const white = roomScene(engine, state);
  state.instances.push(
    { name: 'box', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, box(room.box.center, room.box.half), white)) },
    { name: 'sphere', instance: engine.addMeshInstance(nodeAt(engine, room.sphere.center), meshWith(engine, sphere(room.sphere.radius, room.sphere.segments), white)) },
  );
}

// Each arrangement adds its content and pushes { name, instance } entries into state.instances.
const ARRANGEMENTS = {
  'color-bleed': roomWithObjects,
  'interior-bounce': roomWithObjects,
  'emissive-panel': (engine, variant, state) => {
    const white = roomScene(engine, state, { night: true });
    const glow = engine.addSurfaceMaterial({ color: [1, 0.92, 0.8], emissive: [8, 7.2, 6], roughness: 0.9 });
    state.instances.push(
      { name: 'panel', instance: engine.addMeshInstance(nodeAt(engine, [0, 0.9, -7.4]), meshWith(engine, panel(1.6, 1.0), glow)) },
      { name: 'box', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, box([1.2, 0.5, -6.2], [0.5, 0.5, 0.5]), white)) },
      { name: 'sphere', instance: engine.addMeshInstance(nodeAt(engine, [-1.3, 0.6, -6.4]), meshWith(engine, sphere(0.6, 32), white)) },
    );
  },
  'local-light-shadow': (engine, variant, state) => {
    environment(engine, true);
    const floor = engine.addMaterial({ color: [0.7, 0.7, 0.68], roughness: 0.9 });
    const white = matte(engine, room.colors.white);
    state.instances.push(
      { name: 'floor', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, plane(14, 14), floor)) },
      { name: 'wall', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, box([0, 1.25, -4], [3, 1.25, 0.3]), white)) },
    );
    engine.addLight({ kind: 'point', position: [-0.9, 1.2, -5.6], color: [1, 0.85, 0.6], intensity: 20, range: 8 });
    engine.addLight({ kind: 'spot', position: [0.9, 2.4, -5.8], direction: [0, -0.55, 1], color: [0.7, 0.85, 1], intensity: 30, range: 8, angleDeg: 35 });
  },
  'sky-overhang': (engine, variant, state) => {
    engine.setSky({ model: 'physical', preset: 'day' });
    engine.setLight({ direction: [0.45, -1, 0.25], color: [1, 0.96, 0.9], intensity: 3 });
    const ground = matte(engine, [0.42, 0.42, 0.4]);
    const white = matte(engine, room.colors.white);
    const pillars = [[1.9, 1.9], [-1.9, 1.9], [1.9, -1.9], [-1.9, -1.9]].map(([x, z]) => box([x, 1.5, z], [0.15, 1.5, 0.15]));
    const overhang = joinMeshes([box([0, 3.15, 0], [2.2, 0.15, 2.2]), ...pillars]);
    state.instances.push(
      { name: 'ground', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, plane(60, 60), ground)) },
      { name: 'overhang', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, overhang, white)) },
      { name: 'box', instance: engine.addMeshInstance(nodeAt(engine), meshWith(engine, box([0.3, 0.4, 0.2], [0.4, 0.4, 0.4]), white)) },
    );
  },
};

// One variant per row. An on variant names its off partner (offMode); an off variant shares the
// arrangement, quality and selection of its on partner and disables GI.
const variants = ROWS.map(row => {
  const on = row.mode.endsWith('-on');
  return {
    id: row.mode,
    description: DESCRIPTIONS[row.mode],
    references: [],
    judge: true,
    arrangement: row.arrangement,
    gi: { enabled: on, debugView: 'none' },
    quality: QUALITY[row.mode] ?? { ...DEFAULT_QUALITY },
    giExpected: on ? 'active' : 'disabled',
    selection: SELECTIONS[row.arrangement],
    ...(on ? { offMode: `${row.arrangement}-off` } : {}),
    ...(row.measured ? { measurement: { pair: pairOf(row), state: on ? 'on' : 'off', preset: `lighting-${row.arrangement}`, features: { globalIllumination: on } } } : {}),
  };
});
export const inventory = ROWS.map(row => ({
  id: `lighting-${row.mode}`,
  criterionName: row.criterionName,
  scenario: 'lighting',
  mode: row.mode,
  filename: row.filename,
  expectedVisibleBehavior: BEHAVIOURS[row.mode],
  requiredFacts: [...(row.mode.endsWith('-on') ? ON_FACTS : OFF_FACTS), ...row.extraFacts],
  pair: pairOf(row),
  heldConstant: HELD[row.arrangement],
  producer: row.producer,
}));

/**
 * `build(engine, lib, variant)` sets the variant's quality, adds its arrangement, sets the variant camera and
 * applies `variant.gi`, then returns the state { instances, moved: false, blockNode: null }.
 */
export async function build(engine, lib, variant) {
  const state = { instances: [], moved: false, blockNode: null };
  engine.setGiQuality(variant.quality);
  ARRANGEMENTS[variant.arrangement](engine, variant, state);
  engine.setCamera(variant.selection.camera);
  engine.setGlobalIllumination(variant.gi);
  return state;
}

const check = (name, passed, detail) => ({ name, passed: passed === true, detail });

/**
 * `observe(engine, variant, state)` reports the seven assertions of contract 'gi-lighting-rows', in factNames
 * order, for any variant. Today it reads giStats(), giQuality(), globalIllumination(), lightStats(), passes(),
 * instanceGlobalIllumination() and the state's instances and moved flag. `gi-active` and `gi-off` read the volume
 * as volume.js does; `local-lights` needs two active local lights; `ssr-enabled` needs the 'ssr' pass enabled and
 * reflections on; `quality-applied` compares giQuality() with the variant's quality; `forward-content` needs the
 * skinned, deformed, transparent and fur instances, none voxelized; `block-moved` needs state.moved. Each detail
 * names every value read, and assertions that do not apply to the variant are still reported.
 * Handoff: gi page: lighting.test.mjs and p4-gi-lighting commands calls
 * `observe(engine: object, variant: object, state: { instances: Array<{ name: string, instance: number }>, moved: boolean, blockNode: number | null }) -> Promise<{ instances: Array<{ name: string, instance: number }>, assertions: Array<{ name: string, passed: boolean, detail: string }>, edits: [] }>`
 * exported by examples/global-illumination/scenes/lighting.js:309 observe; the test attaches it with
 * `import * as lighting from './scenes/lighting.js'` and calls observe on a stub engine; state/keys: the namespace
 * keys `inventory`, `factNames`, `observe`, `default.variants`, `default.build` and `default.ready`, the state keys
 * `instances`, `moved` and `blockNode`, and the assertion names of `factNames`.
 */
export async function observe(engine, variant, state) {
  const stats = engine.giStats();
  const settings = engine.globalIllumination();
  const quality = engine.giQuality();
  const light = engine.lightStats();
  const ssr = engine.passes().find(pass => pass.name === 'ssr');
  const key = `${stats.quality.resolution}/${stats.quality.levels}`;
  const expectedBytes = VOLUME_BYTES[key];
  const wanted = variant.quality;
  const forward = ['skinned', 'deformed', 'transparent', 'fur'].map(name => {
    const entry = state.instances.find(item => item.name === name);
    return { name, present: entry !== undefined, voxelized: entry ? engine.instanceGlobalIllumination(entry.instance).voxelized : null };
  });
  const assertions = [
    check('gi-active',
      stats.active === true && stats.pendingRebuild === false && stats.framesSinceRebuild >= 240
        && stats.allocatedBytes >= expectedBytes && stats.voxelizedForwardInstances >= 1,
      `active ${stats.active}, pendingRebuild ${stats.pendingRebuild}, framesSinceRebuild ${stats.framesSinceRebuild}, `
        + `quality ${key}, allocatedBytes ${stats.allocatedBytes} (expected at least ${expectedBytes}), `
        + `voxelizedForwardInstances ${stats.voxelizedForwardInstances}`),
    check('gi-off',
      stats.active === false && stats.inactiveReason === 'disabled' && stats.allocatedBytes === 0,
      `active ${stats.active}, inactiveReason ${stats.inactiveReason}, allocatedBytes ${stats.allocatedBytes}`),
    check('local-lights', light.activeLights === 2, `activeLights ${light.activeLights}, expected 2`),
    check('ssr-enabled', ssr?.enabled === true && settings.reflections === true,
      `ssr pass ${ssr ? `enabled ${ssr.enabled}` : 'absent'}, reflections ${settings.reflections}`),
    check('quality-applied',
      quality.resolution === wanted.resolution && quality.levels === wanted.levels && quality.probeRays === wanted.probeRays,
      `quality ${quality.resolution}/${quality.levels}/${quality.probeRays}, `
        + `expected ${wanted.resolution}/${wanted.levels}/${wanted.probeRays}`),
    check('forward-content', forward.every(item => item.present && item.voxelized === false),
      forward.map(item => `${item.name} ${item.present ? `voxelized ${item.voxelized}` : 'absent'}`).join(', ')),
    check('block-moved', state.moved === true, `moved ${state.moved}`),
  ];
  return { instances: state.instances, assertions, edits: [] };
}

// Rows 1-10 have no moved block to wait for: every variant is ready as soon as build returns.
function ready(engine, state, variant) {
  return { ready: true, stage: 'lighting' };
}

export default {
  id: 'lighting',
  milestone: 'lighting',
  title: 'GI lighting',
  variants,
  build,
  ready,
};
