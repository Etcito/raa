// Lookdev scene m13-liquids-particles (chain milestone 6, R13): animated liquid surfaces.
// Water (raa.liquid.water): a basin with a tiled floor, a sloped sand beach and rocks breaking
// the surface (shore foam), Gerstner waves with crest foam. Slime (raa.liquid.slime): a
// neon-green river in a channel with inflatable pads. Mud (raa.liquid.mud): an opaque wet pit
// with bubbles between rocks. waves-t0 / waves-t1 show the water set at two times.
// Particles (raa-particles): splash (ripple impacts + splash emitters on the water set),
// snow, confetti, sparkles (night, neon) and the particles-on / particles-off pair.
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
  water: [[0, -0.1, -0.3], 5.0, 0.35, 0.45],
  slime: [[0, 0, 0], 4.0, 0.2, 0.44],
  mud: [[0, -0.05, 0], 4.3, -0.3, 0.5],
  snow: [[0, 0.8, -0.8], 6.0, 0.3, 0.28],
  confetti: [[0, 1.4, -0.4], 5.2, 0.15, 0.3],
  sparkles: [[0, 0.45, 0], 2.4, 0, 0.15],
  particles: [[0, 0.6, 0], 4.2, 0.25, 0.35],
};

const LIQUID_SIZE = 6; // m
const LIQUID_SEGMENTS = 160;
const FLOOR_Y = -0.5;
const TILES_N = 12;

// Waves are the crate's Gerstner layout (A, L, theta, Q); deep-water dispersion speeds.
// The amplitudes are raised above the built-in defaults so the swell and its crest foam
// read at capture distance (card 5018: a nearly flat surface).
const WATER = {
  wave0: [0.11, 3.0, 0.3, 0.65],
  wave1: [0.06, 1.7, 1.4, 0.55],
  wave2: [0.03, 1.0, -0.7, 0.5],
  wave3: [0.014, 0.6, 2.4, 0.4],
  speeds: [2.164, 1.629, 1.249, 0.968],
  deepColor: [0.0, 0.34, 0.64],
  shallowColor: [0.2, 0.84, 0.94],
  clarity: 0.68,
  foamWidth: 0.34,
  foamAmount: 1.0,
  detailStrength: 0.09,
  detailScale: 2.0,
};
// Pink inflatable rim of the water basin (three sides; the sand beach closes the back).
const BASIN_WALL = { half: 0.4, thickness: 0.15, color: [1, 0.42, 0.62] };
const WATER_IOR = 1.33;
const WATER_THICKNESS = 0.5;

// Slime: the water wave layout at about 0.75x amplitude and 0.25x speed, a visible slow
// swell at capture distance. `color` is sRGB (function colours are converted to linear):
// a neon green that keeps its saturation through the AgX tonemap. The absorption is low
// enough that the coloured tiles still read through the body as a strong green-tinted
// refraction (card 5031 saw only a faint tile tint at absorption 9).
const SLIME = {
  wave0: [0.083, 3.0, 0.3, 0.65],
  wave1: [0.045, 1.7, 1.4, 0.55],
  wave2: [0.0225, 1.0, -0.7, 0.5],
  wave3: [0.0105, 0.6, 2.4, 0.4],
  speeds: [0.541, 0.407, 0.312, 0.242],
  color: [0.36, 0.92, 0.1],
  absorption: 1.6,
  glossiness: 0.96,
  bubbleScale: 0.32,
  // Shore lace at the channel walls; little crest foam, which covered the river in a
  // noise-texture lace and hid the bubbles and the refracted tiles (card 5034).
  foamWidth: 0.14,
  foamAmount: 0.4,
  // A gentle detail normal: the old 0.32 mottled the sky reflection into a grey skin
  // (card 5031) instead of the glossy green surface the reference shows.
  detailStrength: 0.08,
  detailScale: 1.2,
};
const SLIME_IOR = 1.38;
const SLIME_THICKNESS = 0.24;
// Pads stand in the channel and rise above the slime surface (base at FLOOR_Y): the
// pink, orange and blue tops sit 0.32, 0.3 and 0.28 m above the slime at y = -0.1.
const SLIME_PADS = [
  { r: 0.42, h: 0.72, x: -1.7, z: 0.15, color: [1, 0.3, 0.55] },
  { r: 0.34, h: 0.7, x: 0.0, z: -0.1, color: [1, 0.55, 0.1] },
  { r: 0.38, h: 0.68, x: 1.65, z: 0.35, color: [0.25, 0.5, 1] },
];

// Mud: the water wave layout at about 0.85x amplitude and 0.08x speed (a slow viscous
// swell), raised from 0.6x so the swell reads at capture distance (card 5034). `color` is
// sRGB, converted to linear by the material table (linear about 0.16/0.066/0.017). The
// colour was never the cause of the mauve pit: on the card 4940 capture the pit centre
// (106, 66, 73) inverts through AgX punchy to HDR (0.22, 0.12, 0.14), so a blue sky term
// outweighed the brown. That term was the blue sky irradiance plus the sky mirrored by a
// strong clear coat; the shader now keeps the coat thin and MUD_LIGHT warms the key light
// and dims the sky terms, for a predicted pit HDR near (0.28, 0.13, 0.075), sRGB about
// (127, 70, 36) after AgX punchy.
const MUD = {
  wave0: [0.094, 3.0, 0.3, 0.65],
  wave1: [0.051, 1.7, 1.4, 0.55],
  wave2: [0.026, 1.0, -0.7, 0.5],
  wave3: [0.012, 0.6, 2.4, 0.4],
  speeds: [0.173, 0.130, 0.100, 0.0774],
  color: [0.44, 0.29, 0.14],
  wetness: 0.85,
  bubbleScale: 0.4,
  bubbleRate: 0.32,
  // Shore foam where the mud is shallower than foamWidth (the depth rides in the vertex
  // colour alpha, see mudMesh); faint churned streaks on the steepest crests.
  foamWidth: 0.12,
  foamAmount: 0.35,
  detailStrength: 0.1,
  detailScale: 0.6,
};
// Mud set lighting: a warmer sun and dimmer sky terms than daylight(). The sky irradiance
// of DAY_SKY carried about as much light as the sun on the pit, all of it blue (measured
// on the rim: light about (2.2, 2.4, 3.8)).
const MUD_LIGHT = {
  sun: { ...SUN, color: [1, 0.88, 0.72] },
  env: { diffuse: 0.45, specular: 0.3 },
};
// Ground height of the mud pit (the mud surface is y = 0): the bed 0.3 m down, rising
// through the surface about r = 2.7 m into a grassy bank.
function mudGround(x, z) {
  return -0.3 + 0.55 * Math.min(1, Math.max(0, (Math.hypot(x, z * 1.1) - 2.1) / 1.2));
}
// Glossy lime-green inflatable pads standing in the mud (ref-3's stepping pads):
// [x, z, radius, height above the bed]. Their tops sit 0.12..0.18 m above the mud.
const MUD_PADS = [
  [-1.45, -0.55, 0.62, 0.46],
  [0.75, -1.35, 0.5, 0.42],
  [1.45, 0.75, 0.56, 0.48],
  [-0.55, 1.35, 0.46, 0.44],
];
const MUD_PAD = { color: [0.55, 0.88, 0.16], roughness: 0.12, clearcoat: 1, clearcoatRoughness: 0.03 };
// Mud splash crowns around each pad wall: MUD_CROWN_POINTS emitters on a ring just outside
// the pad, each throwing opaque glossy chocolate blobs (opacity 1) up and outward. The
// earlier single emitter spawned 3..6 cm clear water droplets inside the pad volume (the
// splash kind's water shading ignores the palette), so no splash ever showed (card 5037).
const MUD_CROWN_POINTS = 6;
const MUD_CROWN = {
  kind: 'splash',
  spread: 0.3,
  rate: 45,
  lifetime: [0.25, 0.5],
  speed: [1.0, 2.1],
  size: [0.07, 0.13],
  drag: 0.4,
  colors: [[0.42, 0.22, 0.09], [0.5, 0.28, 0.12], [0.36, 0.18, 0.07]],
  opacity: 1,
  volume: [0.03, 0.01, 0.03],
  maxParticles: 64,
};
// A little clear white water spray between the crowns (ref-3's white splash spray).
const MUD_SPRAY_POINTS = 2;
const MUD_SPRAY = {
  kind: 'splash',
  spread: 0.4,
  rate: 70,
  lifetime: [0.3, 0.6],
  speed: [1.4, 2.6],
  size: [0.025, 0.05],
  colors: [[0.92, 0.97, 1]],
  volume: [0.03, 0.01, 0.03],
  maxParticles: 64,
};
// Rocks: [x, z, radius, squash in y, y centre]
const ROCKS = [
  [-1.2, -0.9, 0.5, 0.7, -0.05],
  [1.5, -1.6, 0.6, 0.8, 0.0],
  [0.4, 0.9, 0.3, 0.7, -0.05],
];

const identityQuat = [0, 0, 0, 1];

function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function place(engine, mesh, t, scale = [1, 1, 1], rotation = identityQuat) {
  const node = engine.addNode(null, t, rotation, scale);
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

function rock(engine, geometry, [x, z, r, squash, y], color = [0.42, 0.42, 0.43]) {
  const m = engine.addSurfaceMaterial({ color, roughness: 0.8 });
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
  // The pink inflatable rim of the basin, so the water reads as contained; the beach at
  // the back is its fourth shore.
  const wallMat = engine.addSurfaceMaterial({ color: BASIN_WALL.color, roughness: 0.3, clearcoat: 1 });
  const reach = LIQUID_SIZE / 2 + BASIN_WALL.thickness;
  const wallY = FLOOR_Y + BASIN_WALL.half;
  for (const sx of [-1, 1]) {
    const wall = geometry.roundedBox([BASIN_WALL.thickness, BASIN_WALL.half, reach], 0.06, 4);
    place(engine, upload(engine, wall, wallMat), [sx * (LIQUID_SIZE / 2 + BASIN_WALL.thickness), wallY, 0]);
  }
  const front = geometry.roundedBox([reach, BASIN_WALL.half, BASIN_WALL.thickness], 0.06, 4);
  place(engine, upload(engine, front, wallMat), [0, wallY, LIQUID_SIZE / 2 + BASIN_WALL.thickness]);
  return liquid(engine, geometry, LIQUID_SIZE, LIQUID_SIZE, {
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
  // Low rims (top 0.28 m above the slime): a taller wall hides the foam band along the
  // mesh border from the capture camera (the surface is 0.4 m below a 0.8 m wall's top).
  for (const s of [-1, 1]) {
    const wall = geometry.roundedBox([W / 2 + 0.1, 0.34, 0.15], 0.05, 4);
    place(engine, upload(engine, wall, wallMat), [0, FLOOR_Y + 0.34, s * (D / 2 + 0.15)]);
    const cap = geometry.roundedBox([0.15, 0.34, D / 2 + 0.3], 0.05, 4);
    place(engine, upload(engine, cap, wallMat), [s * (W / 2 + 0.15), FLOOR_Y + 0.34, 0]);
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
    transmission: 0.85,
    ior: SLIME_IOR,
    thickness: SLIME_THICKNESS,
  }, [0, -0.1, 0]);
}

// The mud grid; its vertex colour alpha is the mud depth above mudGround (metres, clamped
// to [0, 1]), which raa.liquid.mud reads for its shore foam. Around each pad the depth
// falls towards its wall, so the foam also rings the pads.
function mudMesh(geometry) {
  const depth = (p) => {
    let d = -mudGround(p[0], p[2]);
    for (const [x, z, r] of MUD_PADS) d = Math.min(d, 0.6 * (Math.hypot(p[0] - x, p[2] - z) - r));
    return Math.min(1, Math.max(0, d));
  };
  return geometry.grid(LIQUID_SIZE, LIQUID_SIZE, LIQUID_SEGMENTS, LIQUID_SEGMENTS, {
    color: (p) => [1, 1, 1, depth(p)],
  });
}

function buildMud(engine, lib) {
  const { geometry } = lib;
  daylight(engine);
  engine.setLight(MUD_LIGHT.sun);
  engine.setEnvironmentLighting(MUD_LIGHT.env);
  // The bank: lime grass above the mud line, a dark mud-stained band at the shore.
  const ground = engine.addSurfaceMaterial({ color: [1, 1, 1], roughness: 0.9 });
  const grass = [0.2, 0.46, 0.035];
  const stain = [0.1, 0.05, 0.02];
  const rim = heightfield(geometry, 8, 8, 96, mudGround, (x, z, y) => {
    const g = Math.min(1, Math.max(0, (y - 0.02) / 0.12));
    return [0, 1, 2].map((i) => stain[i] + (grass[i] - stain[i]) * g).concat(1);
  });
  place(engine, upload(engine, rim, ground), [0, 0, 0]);
  const padMat = engine.addSurfaceMaterial(MUD_PAD);
  let seed = 61;
  for (const [x, z, r, h] of MUD_PADS) {
    // A rounded inflatable disc: straight wall, quarter-round shoulder, slightly domed top.
    const shoulder = 0.35 * (h - 0.3);
    const profile = [[0, 0], [r, 0], [r, h - shoulder - 0.02]];
    for (let k = 1; k <= 6; k++) {
      const a = (k / 6) * (Math.PI / 2);
      profile.push([r - shoulder + shoulder * Math.cos(a), h - shoulder - 0.02 + shoulder * Math.sin(a)]);
    }
    profile.push([0.5 * r, h], [0, h + 0.01]);
    place(engine, upload(engine, geometry.lathe(profile, 64), padMat), [x, -0.3, z]);
    const ring = (count, phase, desc) => {
      for (let k = 0; k < count; k++) {
        const a = phase + (k / count) * Math.PI * 2;
        const c = Math.cos(a);
        const s = Math.sin(a);
        engine.addEmitter({
          ...desc,
          position: [x + (r + 0.06) * c, 0.04, z + (r + 0.06) * s],
          direction: [0.45 * c, 1, 0.45 * s],
          seed: seed++,
        });
      }
    };
    ring(MUD_CROWN_POINTS, 0.3, MUD_CROWN);
    ring(MUD_SPRAY_POINTS, 0.3 + Math.PI / MUD_CROWN_POINTS, MUD_SPRAY);
  }
  const mud = engine.addSurfaceMaterial({ function: 'raa.liquid.mud', params: MUD });
  const mesh = upload(engine, mudMesh(geometry), mud);
  place(engine, mesh, [0, 0, 0]);
}

// ---- Particle sets (raa-particles). Captures render at the variant's fixed timeMs; the
// simulation catches up from max(0, t - 4 s) in 1/60 s steps, so every set relies on
// rate-driven emitters (steady streams) rather than bursts.

// Splash: three ripple impacts on the water set at 0.3..0.8 s before the capture time
// (x, z in the water mesh's local space = world, the grid sits at the origin), each with a
// splash emitter throwing droplets up from the impact point.
const SPLASH_IMPACTS = [
  { x: -0.9, z: 0.7, ago: 0.3, strength: 1 },
  { x: 0.9, z: 0.2, ago: 0.55, strength: 0.9 },
  { x: 0.0, z: 1.6, ago: 0.8, strength: 0.8 },
];
const SPLASH_EMITTER = {
  kind: 'splash',
  direction: [0, 1, 0],
  spread: 0.45,
  rate: 250,
  lifetime: [0.5, 1.0],
  speed: [1.5, 3.2],
  size: [0.02, 0.045],
  colors: [[0.92, 0.97, 1]],
  maxParticles: 2048,
};

function buildSplash(engine, lib, settings, variant) {
  const water = buildWater(engine, lib);
  const t = (variant.timeMs ?? 1000) / 1000;
  engine.setLiquidImpacts(water, SPLASH_IMPACTS.map(({ x, z, ago, strength }) => ({ x, z, time: t - ago, strength })));
  SPLASH_IMPACTS.forEach(({ x, z }, i) => {
    engine.addEmitter({ ...SPLASH_EMITTER, position: [x, 0.02, z], seed: 11 + i });
  });
}

// Snow: a cool overcast day over snowy ground with white rounded blocks, pink and blue
// blocks with snow caps and snowballs (ref-2's palette), under a falling snow volume.
const OVERCAST = {
  sun: { direction: [-0.45, -0.7, -0.55], color: [0.85, 0.9, 1], intensity: 3 },
  ambient: { color: [0.75, 0.82, 0.95], intensity: 0.6 },
  sky: { preset: 'snow', clouds: { coverage: 0.75 } },
  env: { diffuse: 0.9, specular: 1 },
};
const SNOW_WHITE = [0.93, 0.95, 1];
// Blocks: [x, z, half extents, colour (null = snow white)]
const SNOW_BLOCKS = [
  [-1.8, -1.2, [0.5, 0.45, 0.5], null],
  [1.6, -1.8, [0.7, 0.6, 0.45], null],
  [0.3, -2.6, [0.45, 0.8, 0.45], [1, 0.35, 0.65]],
  [-0.6, 0.6, [0.3, 0.25, 0.3], null],
  [2.2, 0.4, [0.4, 0.35, 0.4], [0.25, 0.6, 1]],
  [-2.6, 1.0, [0.35, 0.3, 0.35], [0.25, 0.6, 1]],
];
// Snowballs: [x, z, radius]
const SNOWBALLS = [[0.9, -0.3, 0.45], [-1.2, -2.6, 0.7], [1.0, 1.4, 0.22]];
const SNOW_EMITTER = {
  kind: 'snow',
  position: [0, 2.5, -0.5],
  volume: [5, 2.5, 5],
  groundY: 0,
  rate: 2400,
  lifetime: [6, 10],
  size: [0.02, 0.045],
  colors: [[0.97, 0.98, 1]],
  maxParticles: 8192,
  seed: 21,
};

function overcast(engine) {
  engine.setBackground(BACKGROUND);
  engine.setLight(OVERCAST.sun);
  engine.setAmbient(OVERCAST.ambient);
  engine.setSky(OVERCAST.sky);
  engine.setEnvironmentLighting(OVERCAST.env);
  engine.setToneMapping({ look: 'punchy' });
  engine.setShadows(SHADOWS);
}

function buildSnow(engine, lib) {
  const { geometry } = lib;
  overcast(engine);
  const ground = heightfield(geometry, 16, 16, 96,
    (x, z) => 0.06 * Math.sin(x * 1.3 + 0.4) * Math.cos(z * 1.1) + 0.03 * Math.sin(x * 3.1 + z * 2.3),
    () => [1, 1, 1, 1]);
  place(engine, upload(engine, ground, engine.addSurfaceMaterial({ color: SNOW_WHITE, roughness: 0.9 })), [0, 0, 0]);
  const white = engine.addSurfaceMaterial({ color: SNOW_WHITE, roughness: 0.85 });
  for (const [x, z, half, color] of SNOW_BLOCKS) {
    const box = geometry.roundedBox(half, Math.min(...half) * 0.35, 4);
    const m = color ? engine.addSurfaceMaterial({ color, roughness: 0.3, clearcoat: 1 }) : white;
    place(engine, upload(engine, box, m), [x, half[1], z]);
    if (color) {
      const cap = geometry.roundedBox([half[0] + 0.03, 0.08, half[2] + 0.03], 0.07, 4);
      place(engine, upload(engine, cap, white), [x, 2 * half[1] + 0.03, z]);
    }
  }
  for (const [x, z, r] of SNOWBALLS) {
    place(engine, upload(engine, geometry.uvSphere(r, 48, 24), white), [x, r * 0.9, z]);
  }
  engine.addEmitter(SNOW_EMITTER);
}

// Confetti: two cannons on the coloured tiles in daylight shooting confetti upward and
// inward; the flakes tumble down and settle on the tiles (groundY 0, the tiles' top).
const CONFETTI_COLORS = [
  [1, 0.15, 0.45], [1, 0.8, 0.1], [0.2, 0.55, 1], [0.3, 0.9, 0.3], [1, 0.45, 0.1], [0.7, 0.3, 1],
];
// Cannons: [x, z, tilt about z rad (positive leans to -x)]
const CANNONS = [[-1.6, -0.6, -0.35], [1.6, -0.6, 0.35]];
const CANNON_LENGTH = 0.8;
const CONFETTI_EMITTER = {
  kind: 'confetti',
  spread: 0.3,
  rate: 420,
  lifetime: [2.5, 4.5],
  speed: [4.5, 7.5],
  size: [0.045, 0.08],
  drag: 1.5,
  groundY: 0,
  colors: CONFETTI_COLORS,
  maxParticles: 4096,
};

function buildConfetti(engine, lib) {
  const { geometry, objects } = lib;
  daylight(engine);
  const tiles = engine.addSurfaceMaterial({ color: [1, 1, 1], roughness: 0.35 });
  place(engine, upload(engine, objects.floorTiles(TILES_N), tiles), [0, 0, 0]);
  const barrel = engine.addSurfaceMaterial({ color: [1, 0.3, 0.6], roughness: 0.25, clearcoat: 1 });
  const ring = engine.addSurfaceMaterial({ color: [1, 0.8, 0.15], roughness: 0.25, clearcoat: 1 });
  CANNONS.forEach(([x, z, tilt], i) => {
    const q = [0, 0, Math.sin(tilt / 2), Math.cos(tilt / 2)];
    const dir = [-Math.sin(tilt), Math.cos(tilt), 0];
    const tube = geometry.lathe([[0, 0], [0.2, 0], [0.2, 0.06], [0.16, 0.12], [0.14, CANNON_LENGTH], [0.11, CANNON_LENGTH], [0, CANNON_LENGTH - 0.05]], 40);
    place(engine, upload(engine, tube, barrel), [x, 0, z], [1, 1, 1], q);
    const lip = geometry.lathe([[0.1, 0], [0.18, 0], [0.18, 0.08], [0.1, 0.08]], 40);
    const top = dir.map((d) => d * (CANNON_LENGTH - 0.06));
    place(engine, upload(engine, lip, ring), [x + top[0], top[1], z + top[2]], [1, 1, 1], q);
    const mouth = dir.map((d) => d * (CANNON_LENGTH + 0.05));
    engine.addEmitter({ ...CONFETTI_EMITTER, position: [x + mouth[0], mouth[1], z + mouth[2]], direction: dir, seed: 31 + i });
  });
}

// Sparkles: a night set (sky passes off, dark violet background, low ambient, bloom) with
// lib's neon letter in raa.neon and gold and white sparkles twinkling around it.
const NIGHT_BACKGROUND = [0.02, 0.012, 0.06];
const NIGHT = {
  moon: { direction: [-0.3, -0.85, -0.43], color: [0.8, 0.7, 1], intensity: 0.3 },
  ambient: { color: [0.5, 0.4, 0.9], intensity: 0.08 },
  env: { diffuse: 0.3, specular: 0.5 },
  bloom: { strength: 0.1, radius: 0.85 },
};
const SPARKLE_NEON = { color: [1, 0.12, 0.62], strength: 5, coreWhiteness: 1 };
const SPARKLE_EMITTERS = [
  { colors: [[1, 0.8, 0.3], [1, 0.65, 0.15]], rate: 160, seed: 41 },
  { colors: [[1, 1, 1], [0.85, 0.9, 1]], rate: 110, seed: 42 },
].map((e) => ({
  kind: 'sparkles',
  position: [0, 0.45, 0.05],
  volume: [0.7, 0.6, 0.35],
  lifetime: [0.6, 1.6],
  size: [0.015, 0.04],
  emissive: 8,
  maxParticles: 1024,
  ...e,
}));

function buildSparkles(engine, lib) {
  const { geometry, objects } = lib;
  engine.setBackground(NIGHT_BACKGROUND);
  engine.setLight(NIGHT.moon);
  engine.setAmbient(NIGHT.ambient);
  engine.setEnvironmentLighting(NIGHT.env);
  engine.setToneMapping({ look: 'punchy' });
  engine.setBloom(NIGHT.bloom);
  const deck = engine.addSurfaceMaterial({ color: [0.12, 0.06, 0.3], roughness: 0.2, clearcoat: 1, clearcoatRoughness: 0.05 });
  place(engine, upload(engine, geometry.roundedBox([2, 0.05, 1.5], 0.02, 2), deck), [0, -0.05, 0]);
  const neon = engine.addSurfaceMaterial({ function: 'raa.neon', params: SPARKLE_NEON, castShadow: false });
  place(engine, upload(engine, objects.neonTube(), neon), [0, 0, 0]);
  for (const e of SPARKLE_EMITTERS) engine.addEmitter(e);
}

// Particles on/off (Q15): one small daylight set with a confetti, a snow and a sparkle
// emitter; the off variant disables particles.sim, particles.opaque and particles.blend.
function buildParticlesToggle(engine, lib) {
  const { objects } = lib;
  daylight(engine);
  const tiles = engine.addSurfaceMaterial({ color: [1, 1, 1], roughness: 0.35 });
  place(engine, upload(engine, objects.floorTiles(8), tiles), [0, 0, 0]);
  engine.addEmitter({ ...CONFETTI_EMITTER, position: [-0.8, 0.05, 0], direction: [0.2, 1, 0], rate: 200, seed: 51 });
  engine.addEmitter({ ...SNOW_EMITTER, position: [0, 1.5, 0], volume: [2, 1.5, 2], rate: 600, seed: 52 });
  engine.addEmitter({ ...SPARKLE_EMITTERS[0], position: [0.9, 0.6, 0], volume: [0.4, 0.4, 0.4], seed: 53 });
}

const SETS = {
  water: buildWater,
  slime: buildSlime,
  mud: buildMud,
  splash: buildSplash,
  snow: buildSnow,
  confetti: buildConfetti,
  sparkles: buildSparkles,
  particles: buildParticlesToggle,
};

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
    description: `${BASIS}an opaque chocolate-brown wet mud pit with a slow swell, a wet `
      + 'glossy sheen with sun highlights and a faint reflection of the sky, small popping '
      + 'bubbles, churned frothy brown mud along its shore and around the pads, and crowns '
      + 'of rounded glossy chocolate mud splashes with a little white spray thrown up around '
      + 'every pad; round glossy lime-green inflatable pads stand in the mud, and a grassy '
      + 'lime-green bank surrounds the pit',
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

const PARTICLE_BASIS = 'Engine lookdev capture of the particles milestone. Score two things: '
  + 'that the capture shows what this description lists, and that the PARTICLES read as what '
  + 'they are (falling snow flakes, tumbling paper confetti, glowing twinkling sparkles) with '
  + 'the light and mood of the reference. The reference is a finished illustration and is NOT '
  + 'the target for the scene: the sparse procedural staging (a few simple blocks, a plain '
  + 'tiled floor, simple props), the absence of characters, buildings, depth of field, fog '
  + 'and other post effects are expected at this milestone and must not lower the score. '
  + 'Shows: ';

const PARTICLE_OFF = ['particles.sim', 'particles.opaque', 'particles.blend'];

const PARTICLE_VARIANTS = [
  {
    id: 'splash',
    description: `Splash check: ${WATER_DESC}; three ripple rings spread from impact points on `
      + 'the water and fountains of small white droplets spray up from them and fall back.',
    references: [],
    judge: false,
    timeMs: 1500,
    settings: { set: 'splash', camera: 'water' },
  },
  {
    id: 'snow',
    description: `${PARTICLE_BASIS}snow falling over a snowy landscape on a cool, bright `
      + 'overcast day: many small white snow flakes drift down through the whole frame, in '
      + 'front of white rounded snow blocks, pink and blue blocks with snow caps and big '
      + 'snowballs standing on white snowy ground, with flakes settling on the ground',
    references: ['ref-2'],
    judge: true,
    timeMs: 3000,
    settings: { set: 'snow' },
  },
  {
    id: 'confetti',
    description: `${PARTICLE_BASIS}two glossy pink party cannons with yellow rims standing on `
      + 'a floor of coloured tiles in sunny daylight under a blue sky, shooting streams of '
      + 'colourful paper confetti (pink, yellow, blue, green, orange, purple) up into the air; '
      + 'the small flakes tumble and flutter as they fall, and some lie scattered on the tiles',
    references: ['ref-1'],
    judge: true,
    timeMs: 2500,
    settings: { set: 'confetti' },
  },
  {
    id: 'sparkles',
    description: `${PARTICLE_BASIS}a night scene on a dark violet background: a glowing pink `
      + 'neon tube shaped like a script letter S stands on a dark glossy violet deck, with a '
      + 'soft bloom halo, and many tiny gold and white sparkles twinkle and glow around it',
    references: ['ref-4'],
    judge: true,
    timeMs: 2000,
    settings: { set: 'sparkles' },
    passes: { disable: ['sky.env', 'sky.background'] },
  },
  {
    id: 'particles-on',
    description: 'Particles on: coloured tiles in daylight with a confetti stream on the left, '
      + 'falling snow flakes over the set and gold sparkles on the right.',
    references: [],
    judge: false,
    timeMs: 2000,
    settings: { set: 'particles' },
  },
  {
    id: 'particles-off',
    description: 'Particles off: the same view as particles-on with the particle passes '
      + 'disabled (no confetti, snow or sparkles).',
    references: [],
    judge: false,
    timeMs: 2000,
    settings: { set: 'particles' },
    passes: { disable: PARTICLE_OFF },
  },
];

export default {
  id: 'm13-liquids-particles',
  milestone: 'm13',
  title: 'Animated liquid surfaces and GPU particles',
  engine: { msaa: 4, toneMapper: 'agx' },
  variants: [...VARIANTS, ...PARTICLE_VARIANTS],

  async build(engine, lib, variant) {
    const settings = variant.settings ?? {};
    const set = settings.set ?? variant.id;
    const builder = SETS[set];
    if (!builder) throw new Error(`m13-liquids-particles: no builder for variant '${variant.id}'`);
    builder(engine, lib, settings, variant);
    const [target, distance, yaw, pitch] = CAMERAS[settings.camera ?? set];
    engine.setOrbit(target, distance, yaw, pitch);
  },
};
