// Lookdev scene m13-liquids-particles (chain milestone 6, R13): animated liquid surfaces.
// Water (raa.liquid.water): a basin with a tiled floor, a sloped sand beach and rocks breaking
// the surface (shore foam), Gerstner waves with crest foam. Slime (raa.liquid.slime): a
// neon-green river in a channel with inflatable pads. Mud (raa.liquid.mud): an opaque wet pit
// with bubbles between rocks. waves-t0 / waves-t1 show the water set at two times.
// Variants are grouped in sets (settings.set), one builder per set in SETS.
// Pure module (S14): no top-level side effects, node reads id/milestone/variants.

const BACKGROUND = [0.3, 0.34, 0.4]; // display-referred (S12), hidden by the sky
const SUN = { direction: [-0.8067, -0.5736, 0.1422], color: [1, 0.93, 0.82], intensity: 7 };
const AMBIENT = { color: [0.55, 0.6, 0.7], intensity: 0.35 };
const DAY_SKY = {
  zenith: [0.03, 0.17, 0.95],
  horizon: [0.32, 0.55, 0.95],
  ground: [0.45, 0.62, 0.92],
  intensity: 1.6,
  horizonSoftness: 5,
  sunDiscSize: 0.022,
  sunDiscIntensity: 20,
  clouds: { coverage: 0.5, density: 2.5, scale: 1800 },
};
const SHADOWS = { softness: 1, distance: 30 };
const ENV = { diffuse: 0.7, specular: 1 };

// Orbit cameras: [target, distance m, yaw rad, pitch rad].
const CAMERAS = {
  water: [[0, -0.1, -0.3], 5.2, 0.35, 0.42],
  slime: [[0, 0, 0], 4.6, 0.2, 0.5],
  mud: [[0, -0.1, 0], 5.0, -0.3, 0.5],
};

const LIQUID_SIZE = 6; // m
const LIQUID_SEGMENTS = 160;
const FLOOR_Y = -0.5;
const TILES_N = 12;

const WATER = {
  deepColor: [0.0, 0.5, 0.75],
  shallowColor: [0.35, 0.95, 0.85],
  clarity: 0.5,
  foamWidth: 0.2,
  foamAmount: 0.9,
};
const WATER_IOR = 1.33;
const WATER_THICKNESS = 0.5;

const SLIME = { color: [0.42, 1, 0.12], absorption: 5, glossiness: 0.92, bubbleScale: 0.05, foamAmount: 0.3 };
const SLIME_IOR = 1.38;
const SLIME_THICKNESS = 0.3;
const SLIME_PADS = [
  { r: 0.4, h: 0.45, x: -1.2, z: 0.2, color: [1, 0.3, 0.55] },
  { r: 0.3, h: 0.4, x: 0.7, z: -0.4, color: [1, 0.55, 0.1] },
  { r: 0.35, h: 0.42, x: 1.7, z: 0.5, color: [0.25, 0.5, 1] },
];

const MUD = { color: [0.32, 0.22, 0.14], wetness: 0.9, bubbleScale: 0.2, bubbleRate: 0.8, foamAmount: 0.15 };
// Rocks: [x, z, radius, squash in y, y centre]
const ROCKS = [
  [-1.2, -0.9, 0.5, 0.7, -0.05],
  [1.5, -1.6, 0.6, 0.8, 0.0],
  [0.4, 0.9, 0.3, 0.7, -0.05],
];
const MUD_ROCKS = [
  [-2.2, -0.6, 0.7, 0.8, 0.1],
  [-1.0, -2.2, 0.55, 0.7, 0.05],
  [1.9, -1.2, 0.65, 0.8, 0.1],
  [2.3, 1.2, 0.5, 0.7, 0.05],
  [-1.6, 2.0, 0.6, 0.75, 0.1],
  [0.8, 2.3, 0.45, 0.7, 0.05],
];

const identityQuat = [0, 0, 0, 1];

function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function place(engine, mesh, t, scale = [1, 1, 1]) {
  const node = engine.addNode(null, t, identityQuat, scale);
  return engine.addMeshInstance(node, mesh);
}

function upload(engine, desc, material) {
  const subMeshes = [{ first: 0, count: desc.indices.length, material }];
  return engine.addMesh({ ...desc, subMeshes });
}

function daylight(engine) {
  engine.setBackground(BACKGROUND);
  engine.setLight(SUN);
  engine.setAmbient(AMBIENT);
  engine.setSky({ preset: 'day', ...DAY_SKY });
  engine.setEnvironmentLighting(ENV);
  engine.setToneMapping({ look: 'punchy' });
  engine.setShadows(SHADOWS);
}

// Heightfield over the lib grid: y = height(x, z), vertex colour = colour(x, z, y), normals
// from central differences.
function heightfield(geometry, w, d, n, height, colour) {
  const desc = geometry.grid(w, d, n, n);
  const p = new Float32Array(desc.positions);
  const nrm = new Float32Array(desc.normals);
  const col = new Float32Array((p.length / 3) * 4);
  const e = 0.01;
  for (let i = 0; i < p.length / 3; i++) {
    const x = p[3 * i];
    const z = p[3 * i + 2];
    const y = height(x, z);
    p[3 * i + 1] = y;
    const nx = height(x - e, z) - height(x + e, z);
    const nz = height(x, z - e) - height(x, z + e);
    const l = Math.hypot(nx, 2 * e, nz);
    nrm[3 * i] = nx / l;
    nrm[3 * i + 1] = (2 * e) / l;
    nrm[3 * i + 2] = nz / l;
    col.set(colour(x, z, y), 4 * i);
  }
  return { ...desc, positions: p, normals: nrm, colors: col };
}

function rock(engine, geometry, [x, z, r, squash, y], color = [0.45, 0.42, 0.4]) {
  const m = engine.addSurfaceMaterial({ color, roughness: 0.85 });
  place(engine, upload(engine, geometry.uvSphere(r, 32, 16), m), [x, y, z], [1, squash, 1]);
}

// A liquid grid uploaded with its function material.
function liquid(engine, geometry, w, d, desc, t) {
  const mesh = upload(engine, geometry.grid(w, d, LIQUID_SEGMENTS, Math.round((LIQUID_SEGMENTS * d) / w)), engine.addSurfaceMaterial(desc));
  return place(engine, mesh, t);
}

function buildWater(engine, lib) {
  const { geometry, objects } = lib;
  daylight(engine);
  const tiles = engine.addSurfaceMaterial({ color: [1, 1, 1], roughness: 0.35 });
  place(engine, upload(engine, objects.floorTiles(TILES_N), tiles), [0, FLOOR_Y, 0]);
  // Sloped sand beach along the back (-z), rising from the floor through the water level.
  const sand = srgbToLinear(0.85);
  const beach = heightfield(geometry, LIQUID_SIZE, 2.2, 64,
    (x, z) => FLOOR_Y + Math.max(0, (-z + 1.1) / 2.2) ** 1.2 * 1.0 + 0.04 * Math.sin(x * 4 + z * 3),
    (x, z) => [sand, sand * 0.85, sand * 0.55, 1]);
  const sandMat = engine.addSurfaceMaterial({ color: [1, 1, 1], roughness: 0.95 });
  place(engine, upload(engine, beach, sandMat), [0, 0, -2.0]);
  for (const r of ROCKS) rock(engine, geometry, r);
  liquid(engine, geometry, LIQUID_SIZE, LIQUID_SIZE, {
    function: 'raa.liquid.water',
    params: WATER,
    blend: 'transmissive',
    transmission: 1,
    ior: WATER_IOR,
    thickness: WATER_THICKNESS,
  }, [0, 0, 0]);
}

function buildSlime(engine, lib) {
  const { geometry, objects } = lib;
  daylight(engine);
  const tiles = engine.addSurfaceMaterial({ color: [1, 1, 1], roughness: 0.35 });
  place(engine, upload(engine, objects.floorTiles(TILES_N), tiles), [0, FLOOR_Y, 0]);
  const W = 6;
  const D = 2.4;
  const wallMat = engine.addSurfaceMaterial({ color: [1, 0.42, 0.62], roughness: 0.3, clearcoat: 1 });
  for (const s of [-1, 1]) {
    const wall = geometry.roundedBox([W / 2 + 0.1, 0.4, 0.15], 0.05, 4);
    place(engine, upload(engine, wall, wallMat), [0, FLOOR_Y + 0.4, s * (D / 2 + 0.15)]);
  }
  for (const p of SLIME_PADS) {
    const m = engine.addSurfaceMaterial({ color: p.color, roughness: 0.25, clearcoat: 1 });
    const pad = geometry.lathe([[0, 0], [p.r, 0], [p.r, p.h * 0.7], [p.r * 0.75, p.h], [0, p.h]], 48);
    place(engine, upload(engine, pad, m), [p.x, FLOOR_Y, p.z]);
  }
  liquid(engine, geometry, W, D, {
    function: 'raa.liquid.slime',
    params: SLIME,
    blend: 'transmissive',
    transmission: 0.75,
    ior: SLIME_IOR,
    thickness: SLIME_THICKNESS,
  }, [0, -0.1, 0]);
}

function buildMud(engine, lib) {
  const { geometry } = lib;
  daylight(engine);
  const ground = engine.addSurfaceMaterial({ color: [0.5, 0.42, 0.32], roughness: 0.95 });
  const rim = heightfield(geometry, 8, 8, 64,
    (x, z) => -0.05 + 0.25 * Math.max(0, (Math.hypot(x, z * 1.1) - 2.8) / 1.2),
    () => [1, 1, 1, 1]);
  place(engine, upload(engine, rim, ground), [0, 0, 0]);
  for (const r of MUD_ROCKS) rock(engine, geometry, r);
  liquid(engine, geometry, LIQUID_SIZE, LIQUID_SIZE, {
    function: 'raa.liquid.mud',
    params: MUD,
    clearcoat: 1,
    clearcoatRoughness: 0.08,
  }, [0, 0, 0]);
}

const SETS = { water: buildWater, slime: buildSlime, mud: buildMud };

const BASIS = 'Engine lookdev capture of the liquids milestone. Score two things: that the '
  + 'capture shows what this description lists, and that the LIQUID look (moving waves, foam '
  + 'at shores and wave crests, refraction of what lies below, reflection of the sky) matches '
  + 'the reference. The reference is a finished illustration and is NOT the target for the '
  + 'scene: the sparse procedural staging (a few simple rocks, a plain tiled floor, simple '
  + 'pads), the absence of characters, buildings, particles, depth of field, fog and other '
  + 'post effects are expected at this milestone and must not lower the score. Shows: ';

const WATER_DESC = 'a square basin of water seen from above and in front, with Gerstner waves '
  + 'of several sizes and white foam on the wave crests; the water refracts and tints the '
  + 'coloured tile floor below it blue-green, reflects the blue sky with white clouds, and '
  + 'white foam lines the shore where it meets a sloped sand beach at the back and the grey '
  + 'rocks that break the surface';

const VARIANTS = [
  {
    id: 'water',
    description: `${BASIS}${WATER_DESC}`,
    references: ['ref-1', 'ref-3'],
    judge: true,
    settings: { set: 'water' },
  },
  {
    id: 'slime',
    description: `${BASIS}a neon-green translucent glossy slime river in a channel with pink `
      + 'inflatable walls, slow waves on its surface and small bubbles; three round glossy '
      + 'inflatable pads (pink, orange, blue) stand in it, and the slime refracts and tints '
      + 'the coloured tile floor of the channel',
    references: ['ref-1'],
    judge: true,
    settings: { set: 'slime' },
  },
  {
    id: 'mud',
    description: `${BASIS}an opaque dark brown wet mud pit with a slow swell, a glossy `
      + 'clearcoat sheen that reflects the sky and small popping bubbles, surrounded by '
      + 'rocks and a rim of lighter brown ground',
    references: ['ref-3'],
    judge: true,
    settings: { set: 'mud' },
  },
  {
    id: 'waves-t0',
    description: `Motion pair, first frame (time 1.0 s): ${WATER_DESC}.`,
    references: [],
    judge: false,
    timeMs: 1000,
    settings: { set: 'water' },
  },
  {
    id: 'waves-t1',
    description: 'Motion pair, second frame (time 1.6 s): the same view as waves-t0 with the '
      + 'waves moved to a different phase.',
    references: [],
    judge: false,
    timeMs: 1600,
    settings: { set: 'water' },
  },
];

export default {
  id: 'm13-liquids-particles',
  milestone: 'm13',
  title: 'Animated liquid surfaces and GPU particles',
  engine: { msaa: 4, toneMapper: 'agx' },
  variants: VARIANTS,

  async build(engine, lib, variant) {
    const settings = variant.settings ?? {};
    const set = settings.set ?? variant.id;
    const builder = SETS[set];
    if (!builder) throw new Error(`m13-liquids-particles: no builder for variant '${variant.id}'`);
    builder(engine, lib, settings);
    const [target, distance, yaw, pitch] = CAMERAS[settings.camera ?? set];
    engine.setOrbit(target, distance, yaw, pitch);
  },
};
