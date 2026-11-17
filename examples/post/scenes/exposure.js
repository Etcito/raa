// The exposure-room scene (S49, post page): a closed 6 x 3 x 6 m room whose only opening is in
// its east wall, facing a sunlit exterior with a facade, a tiled ground, three poles and two cubes.
// Two views, interior and exterior; each has an -off variant at the fixed exposure and an -on variant
// with automatic exposure, which the runner draws until it settles. Importing this module has no side
// effects. Geometry comes from '../lib/content.js': roomWithOpening at examples/post/lib/content.js:104
// (its doc comment, lines 90-103, lists the builders), with tiledGround, facadeCourses, box and plane.
import { box, facadeCourses, plane, roomWithOpening, tiledGround } from '../lib/content.js';

// Estimates behind the values below; they are not measured (the judge rows measure them). Scene-linear
// radiance of a sunlit diffuse surface is albedo x sun x cos / pi; the fixed exposure multiplies it by 2
// (EV 1), then AgX maps it to display. The direct term was calibrated on the exterior-off capture at
// intensity 14: facade red ~1.7 and ground ~0.6 before the fixed exposure, matching albedo x sun x cos / pi
// within about 10 percent.
//  - sunlit ground at intensity 70 (tile albedo 0.21 and 0.11, cos 0.63): ~3 and ~1.5 before the fixed
//    exposure, 5 and 4 stops above middle grey 0.18 at EV 1: the chequer still reads, as the judge saw it;
//  - at intensity 280: ~12 and ~6 before the fixed exposure, 7 and 6 stops; the facade (albedo 0.52,
//    cos 0.69) ~32 before it, 8.5 stops. AgX is near white from about +5.5 stops and reaches its end at +6.5, so the
//    ground and facade clip to flat white;
//  - the sun patch the floor takes through the opening (red albedo 0.033, cos 0.63): ~1.9 before the
//    fixed exposure, bright like the opening; the sun through the opening is the room's only direct light;
//  - the rest of the room gets only the sky fill (ENV_LIGHTING, unoccluded by the walls): the first values
//    (diffuse 0.02, specular 0.05) left walls at ~0.03 before the fixed exposure, mid grey in the judged
//    capture. At a sixth of that, walls are ~0.005 before it (~0.01 at EV 1): dark, the target;
//  - auto exposure, 1.2 * 0.18 / average, is clamped to maxEv 6 (a factor 64, 128 with EV 1). The interior
//    average (~0.002) hits the clamp, so the walls come up to ~0.3 and stay readable. The exterior average
//    (sunlit, ~4) gives EV ~ -4, within minEv -6, so the sunlit surfaces keep their look at any sun.

// Tone mapping exposureEv shared by all four variants: the fixed exposure of the -off rows.
export const FIXED_EXPOSURE_EV = 1;
// The sun stands on +X: it travels towards -X, -Y and +Z (estimates above). Intensity 280 clips the
// sunlit ground and facade at the fixed exposure (four times the first 70); the floor patch is bright.
export const SUN = { direction: [-0.6, -0.55, 0.3], color: [1, 0.97, 0.92], intensity: 280 };
// Ambient, scaled down with the sky fill. With a sky environment on, the ambient slot takes its light from
// the sky and does not read this value (crates/raa-ibl/src/wgsl/ambient.wgsl:36-37); it is small for the case without one.
export const AMBIENT = { color: [0.55, 0.6, 0.7], intensity: 0.0015 };
// Sky fill, the room's light besides the sun patch: the diffuse term (ambient.wgsl:46-47) has no
// occlusion, so these two intensities set how dark the interior is at the fixed exposure. About a sixth of
// the first values (0.02 and 0.05); auto exposure lifts the interior in the -on rows.
export const ENV_LIGHTING = { diffuse: 0.003, specular: 0.008 };
// Bloom is off: its composite mixes 4 % of the blurred frame into every pixel (raa-bloom's default
// strength 0.04, crates/raa-bloom/src/desc.rs:17-24), which would lift the dark room and its shadows
// with the sun. setBloom merges over the default, so only the strength is set.
export const BLOOM = { strength: 0 };

// The room: interior x in [-3, 3], y in [0, 3], z in [-6, 0]; the opening is in the east wall.
const ROOM = { width: 6, height: 3, depth: 6, wall: 0.3, opening: { zMin: -4, zMax: -2, yMin: 0.3, yMax: 2.5 } };
// sRGB base colours; every material has roughness 0.85. The floor is dark (about a fifth of the linear
// albedo of its first version), which keeps the sun patch on it near the opening's brightness at intensity 280.
export const COLORS = {
  walls: [0.8, 0.78, 0.72], floor: [0.2, 0.12, 0.07], table: [0.5, 0.35, 0.2], couch: [0.3, 0.35, 0.5],
  red: [0.8, 0.1, 0.08], green: [0.15, 0.6, 0.15], blue: [0.15, 0.25, 0.75],
  tileEven: [0.5, 0.5, 0.47], tileOdd: [0.36, 0.36, 0.34], base: [0.42, 0.42, 0.4],
  courseEven: [0.75, 0.55, 0.42], courseOdd: [0.65, 0.47, 0.36], pole: [0.2, 0.2, 0.22],
  yellow: [0.85, 0.7, 0.15], teal: [0.15, 0.6, 0.6],
};

// Camera and subjects of each view. The interior camera stands in the room and looks across it with
// the opening at the right of the frame; the exterior camera faces the sunlit facade and the ground.
const OPENING = { name: 'opening', min: [3, 0.3, -4], max: [3.3, 2.5, -2] };
const VIEWS = {
  interior: {
    camera: { eye: [-2.2, 1.6, -0.6], target: [0.8, 1, -5.5], fovYDeg: 70, near: 0.05, far: 200 },
    subjects: [
      { name: 'room-interior', min: [-3, 0, -6], max: [3, 3, 0] },
      OPENING,
      { name: 'table', min: [-1.3, 0, -4], max: [0.3, 0.79, -3] },
      { name: 'cubes', min: [0.55, 0, -5.45], max: [1.75, 1, -4.95] },
    ],
  },
  exterior: {
    camera: { eye: [14, 2.2, 4], target: [3.3, 1.4, -3], fovYDeg: 60, near: 0.1, far: 300 },
    subjects: [
      { name: 'facade', min: [3.3, 0, -6.3], max: [3.32, 3.3, 0.3] },
      OPENING,
      { name: 'ground', min: [3.3, 0, -16], max: [23.3, 0, 10] },
      { name: 'poles', min: [5.94, 0, -5.06], max: [12.06, 3, 2.06] },
    ],
  },
};

// One variant per view and state. -on sets settle (the runner draws until settled); -off has no settle key.
function variantOf(view, state) {
  const on = state === 'on';
  return {
    id: `${view}-${state}`,
    selection: VIEWS[view],
    ...(on ? { settle: true } : {}),
    settings: { toneMapping: { exposureEv: FIXED_EXPOSURE_EV }, autoExposure: { enabled: on } },
    measurement: { pair: `exposure-${view}`, state, preset: `exposure-room-${view}`, features: { autoExposure: on } },
  };
}
const variants = ['interior', 'exterior'].flatMap(view => ['on', 'off'].map(state => variantOf(view, state)));

const REQUIRED_FACTS = [
  'runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'fact:exposure-settled', 'fact:exposure-ev',
  'assertion:exposure-settled', 'assertion:exposure-ev', 'assertion:exposure-enabled',
];
const HELD_CONSTANT = {
  interior: ['room geometry with the east wall opening', 'sky, sun, ambient and environment lighting',
    'interior table, couch and cubes', 'tone mapping exposureEv 1', 'interior camera'],
  exterior: ['room geometry with the east wall opening', 'sky, sun, ambient and environment lighting',
    'tiled ground, facade courses, poles and cubes', 'tone mapping exposureEv 1', 'exterior camera'],
};
const row = (view, state, filename, expectedVisibleBehavior) => ({
  id: `exposure-${view}-${state}`,
  criterionName: `P5-Q2 exposure ${view} ${state}`,
  scenario: 'exposure-room',
  mode: `${view}-${state}`,
  filename,
  expectedVisibleBehavior,
  requiredFacts: REQUIRED_FACTS,
  pair: `exposure-${view}`,
  heldConstant: HELD_CONSTANT[view],
  producer: state === 'on' ? 'raa-exposure' : 'post-page',
});

export const inventory = [
  row('interior', 'on', 'exposure-interior-on-room-readable.png', 'Inside a room lit only through a wall opening onto a sunlit exterior: the room\'s walls, floor and objects are clearly visible with readable shading; the opening is bright but the interior is not black.'),
  row('interior', 'off', 'exposure-interior-off-room-too-dark.png', 'The same interior view at the fixed exposure: the room is very dark, walls and objects inside are hard to make out, only the opening is bright.'),
  row('exterior', 'on', 'exposure-exterior-on-sunlit-facade-readable.png', 'Outside the same building in direct sun: the facade, the opening and the ground keep visible detail and texture; nothing large is clipped to white.'),
  row('exterior', 'off', 'exposure-exterior-off-sunlit-facade-blown-out.png', 'The same exterior view at the fixed exposure: large areas of the sunlit ground and facade are clipped to flat white with little detail.'),
];

export const factNames = ['exposure-settled', 'exposure-ev', 'exposure-enabled'];

// Adds one mesh whose sub-meshes take `materials` (one per sub-mesh, or one for all of them) and one
// instance of it at `at`; returns the instance handle.
function addPart(engine, desc, materials, at) {
  const list = Array.isArray(materials) ? materials : [materials];
  const parts = desc.subMeshes ?? [{ first: 0, count: desc.indices.length }];
  const mesh = engine.addMesh({ ...desc, subMeshes: parts.map((part, i) => ({ ...part, material: list[i] ?? list[0] })) });
  return engine.addMeshInstance(engine.addNode(null, at, [0, 0, 0, 1], [1, 1, 1]), mesh);
}

/**
 * `build(engine, lib, variant)` builds the room, its interior objects and the exterior, and sets the
 * variant camera. Today it sets the day sky, clear weather, SUN, AMBIENT, ENV_LIGHTING and BLOOM,
 * enables shadow.csm, and adds each part with its material; it sets no tone mapping and no automatic
 * exposure, which the runtime applies from variant.settings. It returns one named instance per part.
 */
export async function build(engine, lib, variant) {
  engine.setSky({ preset: 'day' });
  engine.setWeather({ preset: 'clear' });
  engine.setLight(SUN);
  engine.setAmbient(AMBIENT);
  engine.setEnvironmentLighting(ENV_LIGHTING);
  engine.setBloom(BLOOM);
  engine.setPassEnabled('shadow.csm', true);
  const m = Object.fromEntries(Object.entries(COLORS).map(([name, color]) => [name, engine.addMaterial({ color, roughness: 0.85 })]));
  const instances = [];
  const place = (name, desc, materials, at = [0, 0, 0]) => instances.push({ name, instance: addPart(engine, desc, materials, at) });
  const cube = (center, half) => box(center, [half, half, half]);
  place('room', roomWithOpening(ROOM), [m.floor, m.walls, m.walls]);
  place('table-top', box([-0.5, 0.75, -3.5], [0.8, 0.04, 0.5]), m.table);
  [[-1.2, -3.9], [0.2, -3.9], [-1.2, -3.1], [0.2, -3.1]].forEach(([x, z], i) => place(`table-leg-${i}`, box([x, 0.355, z], [0.04, 0.355, 0.04]), m.table));
  place('couch', box([-2.3, 0.4, -3], [0.5, 0.4, 1]), m.couch);
  place('cube-red', cube([0.8, 0.25, -5.2], 0.25), m.red);
  place('cube-green', cube([1.5, 0.25, -5.2], 0.25), m.green);
  place('cube-blue', cube([1.15, 0.75, -5.2], 0.25), m.blue);
  place('ground-tiles', tiledGround({ x0: 3.3, z0: -16, columns: 20, rows: 26, size: 1 }), [m.tileEven, m.tileOdd]);
  place('ground-base', plane(160, 160), m.base, [0, -0.01, 0]);
  place('facade', facadeCourses({ x: 3.3, zMin: -6.3, zMax: 0.3, yMin: 0, yMax: 3.3, course: 0.25, depth: 0.02, skip: ROOM.opening }), [m.courseEven, m.courseOdd]);
  [[6, -1], [9, -5], [12, 2]].forEach(([x, z], i) => place(`pole-${i}`, box([x, 1.5, z], [0.06, 1.5, 0.06]), m.pole));
  place('cube-yellow', cube([7, 0.4, -7], 0.4), m.yellow);
  place('cube-teal', cube([10, 0.4, 1], 0.4), m.teal);
  engine.setCamera(variant.selection.camera);
  return { instances };
}

/**
 * `observe(engine, variant, state)` reads the automatic exposure state of viewport 0 and the pass
 * settings, and returns the three assertions of the rows: settled; the exposure EV, a finite number
 * with an -on variant and null with an -off variant; and the enabled flag the variant names. Each
 * detail names every value it read.
 */
export async function observe(engine, variant, state) {
  const s = engine.autoExposureState();
  const a = engine.autoExposure();
  const on = variant.settings.autoExposure.enabled;
  const read = `settled ${s.settled}, pending ${s.pending}, exposureEv ${s.exposureEv}, `
    + `averageLuminance ${s.averageLuminance}, enabled ${a.enabled}`;
  return {
    values: { 'exposure-settled': s.settled, 'exposure-ev': s.exposureEv },
    assertions: [
      { name: 'exposure-settled', passed: s.settled === true, detail: read },
      { name: 'exposure-ev', passed: on ? Number.isFinite(s.exposureEv) : s.exposureEv === null, detail: `${read}; expected ${on ? 'a finite exposureEv' : 'exposureEv null'}` },
      { name: 'exposure-enabled', passed: a.enabled === on, detail: `${read}; expected enabled ${on}` },
    ],
  };
}

export default {
  id: 'exposure-room',
  milestone: 'exposure',
  title: 'Automatic exposure: room with a sunlit opening',
  variants,
  build,
};
