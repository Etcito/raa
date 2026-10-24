// Lookdev scene m11-transmission (chain milestone 5, R11): clear ice blocks of three
// thicknesses refracting and tinting a floor of saturated coloured tiles, a striped backdrop
// and coloured objects behind them (raa.ice, blend transmissive, thickness-dependent
// absorption); several interpenetrating translucent objects (raa.glass, blend transparent,
// plus an ice block) whose order the linked-list OIT resolves. On/off pairs for transmission
// (blend transmissive vs opaque) and for the `transparent.oit` pass (Q15). The slime and
// water sets: a slime pool with blobs (raa.slime) and a still water basin (raa.water).
// Variants are grouped in sets (settings.set), one builder per set in SETS.
// Pure module (S14): no top-level side effects, node reads id/milestone/variants.

const BACKGROUND = [0.3, 0.34, 0.4]; // display-referred (S12), hidden by the sky
// Sun from the upper right, slightly behind (as m09 and m10, 35 deg high).
const SUN = { direction: [-0.8067, -0.5736, 0.1422], color: [1, 0.93, 0.82], intensity: 7 };
const AMBIENT = { color: [0.55, 0.6, 0.7], intensity: 0.35 };
// The m08 `day` sky overrides, with m09's pale blue haze ground (as m10).
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

// Orbit cameras: [target, distance m, yaw rad, pitch rad] (as m10).
const CAMERAS = {
  // The three blocks side by side, low enough to see the backdrop through them.
  ice: [[0, 0.38, -0.2], 3.3, 0, 0.16],
  transmission: [[0, 0.36, -0.2], 2.2, 0.15, 0.16],
  oit: [[0, 0.42, 0], 2.2, 0.2, 0.18],
  overlap: [[0, 0.42, 0], 2.1, -0.25, 0.2],
  slime: [[0, 0.15, 0.1], 2.6, 0.2, 0.42],
  water: [[0, 0.1, 0], 3.2, 0.3, 0.55],
};

// Floor of lib's coloured tiles (n x n on a 0.5 m pitch), their vertex colours pushed away
// from their luma by SATURATION so refraction and tint read clearly through the ice.
const TILES_N = 12;
const SATURATION = 2.2;
const TILE_ROUGHNESS = 0.35;

// The ice: lib's iceBlock is 0.8 x 0.7 x 0.8 m; each block is scaled along Z to its depth
// (the thickness the light crosses from the camera) and stands at x on the tiles.
const ICE_DEPTH = 0.8;
const ICE_BLOCKS = [
  { depth: 0.15, x: -1.0, seed: 3 },
  { depth: 0.4, x: 0, seed: 7 },
  { depth: 0.8, x: 1.0, seed: 11 },
];
// raa.ice params (sRGB tint, absorption per metre, cloudiness, crack scale m): pale cyan
// over 1 / absorption metres, so the 0.8 m block is a deeper blue than the 0.15 m one.
const ICE = { tint: [0.55, 0.85, 0.97], absorption: 1.8, cloudiness: 0.2, crackScale: 0.22 };
const ICE_IOR = 1.31;
// Single block of the transmission pair (depth m).
const PAIR_DEPTH = 0.4;

// Striped backdrop behind the blocks: vertical stripes (sRGB colours cycled), each a thin
// box; half extents of one stripe, centre z of the backdrop.
const STRIPES = [[1, 0.85, 0.1], [0.95, 0.1, 0.45], [0.1, 0.35, 0.95], [1, 1, 1], [0.1, 0.75, 0.3]];
const STRIPE_COUNT = 15;
const STRIPE_HALF = [0.1, 0.55, 0.02];
const BACKDROP_Z = -1.3;
// Coloured objects between the blocks and the backdrop: spheres (radius, centre) and a
// lathe vase, sRGB colours.
const PROPS = {
  spheres: [
    { r: 0.18, t: [-0.75, 0.18, -0.8], color: [1, 0.35, 0.05] },
    { r: 0.14, t: [0.25, 0.14, -0.75], color: [0.6, 0.1, 0.9] },
    { r: 0.2, t: [1.25, 0.2, -0.85], color: [0.95, 0.9, 0.1] },
  ],
  vase: { t: [-0.2, 0, -0.9], color: [0.05, 0.7, 0.65] },
};

// Translucent glass (raa.glass, blend transparent): two tints (sRGB), opacity 0.4..0.6.
const GLASS_PINK = { tint: [1, 0.3, 0.6], opacity: 0.5 };
const GLASS_CYAN = { tint: [0.2, 0.8, 1], opacity: 0.45 };
const GLASS_AMBER = { tint: [1, 0.7, 0.15], opacity: 0.6 };
const GLASS_ROUGHNESS = 0.05;
// Overlap set: spheres (radius, centre, glass) interpenetrating, and an ice block (scale,
// centre, yaw) crossing them.
const OVERLAP_SPHERES = [
  { r: 0.36, t: [-0.22, 0.42, 0], glass: GLASS_PINK },
  { r: 0.32, t: [0.22, 0.4, 0.1], glass: GLASS_CYAN },
  { r: 0.26, t: [0.02, 0.62, -0.18], glass: GLASS_PINK },
  { r: 0.22, t: [0.05, 0.3, 0.3], glass: GLASS_CYAN },
];
const OVERLAP_ICE = { scale: [1.6, 0.9, 0.3], t: [0, 0.05, 0], yaw: 0.55 };
// OIT pair: three glass spheres in a depth row and a glass slab in front of them.
const OIT_SPHERES = [
  { r: 0.3, t: [-0.3, 0.3, -0.3], glass: GLASS_PINK },
  { r: 0.3, t: [0.05, 0.35, 0], glass: GLASS_CYAN },
  { r: 0.28, t: [0.35, 0.3, 0.25], glass: GLASS_AMBER },
];
const OIT_SLAB = { half: [0.35, 0.3, 0.03], t: [-0.15, 0.3, 0.45], yaw: 0.3, glass: GLASS_CYAN };

const yawQuat = (rad) => [0, Math.sin(rad / 2), 0, Math.cos(rad / 2)];

// Sets each sub-mesh material of `desc` (in order) and returns the mesh handle.
function upload(engine, desc, materials) {
  const subMeshes = (desc.subMeshes ?? [{ first: 0, count: desc.indices.length }]).map((s, i) => ({
    ...s,
    material: materials[Math.min(i, materials.length - 1)],
  }));
  return engine.addMesh({ ...desc, subMeshes });
}

function place(engine, mesh, t, r = yawQuat(0), scale = [1, 1, 1]) {
  const node = engine.addNode(null, t, r, scale);
  return engine.addMeshInstance(node, mesh);
}

// Daylight with the m08 sky, IBL and m09 shadows (as m10's daylight).
function daylight(engine) {
  engine.setBackground(BACKGROUND);
  engine.setLight(SUN);
  engine.setAmbient(AMBIENT);
  engine.setSky({ preset: 'day', ...DAY_SKY });
  engine.setEnvironmentLighting(ENV);
  engine.setToneMapping({ look: 'punchy' });
  engine.setShadows(SHADOWS);
}

// lib's floor tiles with their linear vertex colours saturated (in place, on a fresh desc).
function saturatedTiles(objects) {
  const desc = objects.floorTiles(TILES_N);
  const c = desc.colors;
  for (let i = 0; i < c.length; i += 4) {
    const luma = 0.2126 * c[i] + 0.7152 * c[i + 1] + 0.0722 * c[i + 2];
    for (let k = 0; k < 3; k++) c[i + k] = Math.min(1, Math.max(0, luma + (c[i + k] - luma) * SATURATION));
  }
  return desc;
}

// Coloured tiles, the striped backdrop and the props behind the transmissive objects.
function stage(engine, lib) {
  const { geometry, objects } = lib;
  daylight(engine);
  const tiles = engine.addSurfaceMaterial({ color: [1, 1, 1], roughness: TILE_ROUGHNESS });
  place(engine, upload(engine, saturatedTiles(objects), [tiles]), [0, 0, 0]);
  const stripes = [];
  for (let i = 0; i < STRIPE_COUNT; i++) {
    const color = STRIPES[i % STRIPES.length];
    const x = (i - (STRIPE_COUNT - 1) / 2) * 2 * STRIPE_HALF[0];
    const stripe = geometry.roundedBox(STRIPE_HALF, 0, 1, { color: [...color.map(srgbToLinear), 1] });
    stripes.push(translate(stripe, [x, STRIPE_HALF[1], 0]));
  }
  const paint = engine.addSurfaceMaterial({ color: [1, 1, 1], roughness: 0.5 });
  place(engine, upload(engine, geometry.merge(stripes), [paint]), [0, 0, BACKDROP_Z]);
  for (const s of PROPS.spheres) {
    const m = engine.addSurfaceMaterial({ color: s.color, roughness: 0.3 });
    place(engine, upload(engine, geometry.uvSphere(s.r, 48, 24), [m]), s.t);
  }
  const vase = geometry.lathe(
    [[0, 0], [0.12, 0], [0.12, 0], [0.16, 0.14], [0.09, 0.32], [0.11, 0.42], [0.11, 0.42], [0, 0.42]],
    48,
  );
  const glaze = engine.addSurfaceMaterial({ color: PROPS.vase.color, roughness: 0.2 });
  place(engine, upload(engine, vase, [glaze]), PROPS.vase.t);
}

function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

// A copy of `desc` moved by `t` (merge keeps the other arrays as they are).
function translate(desc, t) {
  const positions = new Float32Array(desc.positions);
  for (let i = 0; i < positions.length; i += 3) {
    for (let k = 0; k < 3; k++) positions[i + k] += t[k];
  }
  return { ...desc, positions };
}

// raa.ice material of a block `depth` m thick; `blend` 'transmissive' or 'opaque'.
function ice(engine, depth, blend = 'transmissive') {
  return engine.addSurfaceMaterial({
    function: 'raa.ice',
    params: ICE,
    blend,
    transmission: 1,
    ior: ICE_IOR,
    thickness: depth,
  });
}

// lib's ice block scaled along Z to `depth`, standing at (x, 0, z).
function iceBlock(engine, lib, depth, x, z, seed, material) {
  const mesh = upload(engine, lib.objects.iceBlock({ seed }), [material]);
  return place(engine, mesh, [x, 0, z], yawQuat(0), [1, 1, depth / ICE_DEPTH]);
}

// raa.glass, blend transparent, with the opacity of `glass`.
function glass(engine, g) {
  return engine.addSurfaceMaterial({
    function: 'raa.glass',
    params: { tint: g.tint, roughness: GLASS_ROUGHNESS },
    blend: 'transparent',
    opacity: g.opacity,
  });
}

function glassSphere(engine, geometry, s) {
  place(engine, upload(engine, geometry.uvSphere(s.r, 64, 32), [glass(engine, s.glass)]), s.t);
}

// Three blocks of increasing thickness over the tiles, the backdrop behind them.
function buildIce(engine, lib) {
  stage(engine, lib);
  for (const b of ICE_BLOCKS) iceBlock(engine, lib, b.depth, b.x, 0, b.seed, ice(engine, b.depth));
}

// One block, transmissive (settings.blend 'transmissive') or opaque ('opaque').
function buildTransmission(engine, lib, settings) {
  stage(engine, lib);
  iceBlock(engine, lib, PAIR_DEPTH, 0, 0, 7, ice(engine, PAIR_DEPTH, settings.blend));
}

// Glass spheres in a depth row and a glass slab in front (the oit pair).
function buildOit(engine, lib) {
  const { geometry } = lib;
  stage(engine, lib);
  for (const s of OIT_SPHERES) glassSphere(engine, geometry, s);
  const slab = geometry.roundedBox(OIT_SLAB.half, 0.02, 4);
  place(engine, upload(engine, slab, [glass(engine, OIT_SLAB.glass)]), OIT_SLAB.t, yawQuat(OIT_SLAB.yaw));
}

// Interpenetrating glass spheres in two tints crossed by a thin ice block.
function buildOverlap(engine, lib) {
  const { geometry } = lib;
  stage(engine, lib);
  for (const s of OVERLAP_SPHERES) glassSphere(engine, geometry, s);
  const depth = ICE_DEPTH * OVERLAP_ICE.scale[2];
  const mesh = upload(engine, lib.objects.iceBlock({ seed: 5 }), [ice(engine, depth)]);
  place(engine, mesh, OVERLAP_ICE.t, yawQuat(OVERLAP_ICE.yaw), OVERLAP_ICE.scale);
}

// Slime (raa.slime, blend transmissive): lib's pool scaled up, and squashed sphere blobs
// (radius, centre, squash in y) on the tiles, two of them in front of the pool and one
// pair overlapping.
const SLIME = { color: [0.45, 0.95, 0.15], absorption: 3, glossiness: 0.85, bubbleScale: 0.06 };
const SLIME_POOL_SCALE = 1.4;
const SLIME_BLOBS = [
  { r: 0.18, t: [-0.35, 0.09, 0.95], squash: 0.5 },
  { r: 0.14, t: [-0.15, 0.07, 1.05], squash: 0.5 },
  { r: 0.2, t: [0.55, 0.1, 0.9], squash: 0.45 },
];
// Water basin: inner half size (x, z) m, wall thickness and height m, water level m.
const BASIN = { half: [1.1, 0.8], wall: 0.12, height: 0.4, radius: 0.04 };
const WATER_LEVEL = 0.25;
const WATER = { deepColor: [0.05, 0.45, 0.55], shallowColor: [0.55, 0.92, 0.88], clarity: 0.7 };

function slime(engine) {
  return engine.addSurfaceMaterial({
    function: 'raa.slime',
    params: SLIME,
    blend: 'transmissive',
    transmission: 0.9,
    ior: 1.4,
    thickness: 0.3,
  });
}

// lib's slime pool (basin, slime surface) and three squashed slime blobs.
function buildSlime(engine, lib) {
  const { geometry, objects } = lib;
  stage(engine, lib);
  const basin = engine.addSurfaceMaterial({ color: [0.92, 0.9, 0.86], roughness: 0.4 });
  const material = slime(engine);
  const s = SLIME_POOL_SCALE;
  place(engine, upload(engine, objects.slimePool(), [basin, material]), [0, 0, 0], yawQuat(0), [s, s, s]);
  const blob = upload(engine, geometry.uvSphere(1, 64, 32), [material]);
  for (const b of SLIME_BLOBS) place(engine, blob, b.t, yawQuat(0), [b.r, b.r * b.squash, b.r]);
}

// Rounded-box basin walls around the tiles and a closed water slab filling it.
function buildWater(engine, lib) {
  const { geometry } = lib;
  stage(engine, lib);
  const [hx, hz] = BASIN.half;
  const { wall, height, radius } = BASIN;
  const hy = height / 2;
  const walls = [
    translate(geometry.roundedBox([hx + 2 * wall, hy, wall / 2], radius, 4), [0, hy, -hz - wall / 2]),
    translate(geometry.roundedBox([hx + 2 * wall, hy, wall / 2], radius, 4), [0, hy, hz + wall / 2]),
    translate(geometry.roundedBox([wall / 2, hy, hz], radius, 4), [-hx - wall / 2, hy, 0]),
    translate(geometry.roundedBox([wall / 2, hy, hz], radius, 4), [hx + wall / 2, hy, 0]),
  ];
  const stone = engine.addSurfaceMaterial({ color: [0.85, 0.82, 0.76], roughness: 0.6 });
  place(engine, upload(engine, geometry.merge(walls), [stone]), [0, 0, 0]);
  const water = engine.addSurfaceMaterial({
    function: 'raa.water',
    params: WATER,
    blend: 'transmissive',
    transmission: 1,
    ior: 1.33,
    thickness: WATER_LEVEL,
  });
  const slab = geometry.roundedBox([hx, WATER_LEVEL / 2, hz], 0.005, 2);
  place(engine, upload(engine, slab, [water]), [0, WATER_LEVEL / 2, 0]);
}

const SETS = {
  ice: buildIce,
  transmission: buildTransmission,
  oit: buildOit,
  overlap: buildOverlap,
  slime: buildSlime,
  water: buildWater,
};

const STAGE = 'on a floor of saturated coloured tiles (orange, green, red, blue) in daylight '
  + 'under a bright blue sky with white clouds, a backdrop of vertical yellow, pink, blue, '
  + 'white and green stripes and a few coloured spheres and a teal vase behind';
const GLASSES = 'interpenetrating translucent glass spheres in two tints (pink and cyan) '
  + 'and a thin pale-cyan ice block crossing them';

const VARIANTS = [
  {
    id: 'ice',
    description: 'clear pale-cyan ice blocks refract and tint the coloured tiles behind them; '
      + 'thicker blocks are deeper blue (thickness-dependent absorption); cracks and bubbles inside',
    references: ['ref-2'],
    judge: true,
    settings: { set: 'ice' },
  },
  {
    id: 'transmission-on',
    description: `R11: one clear ice block 0.4 m thick ${STAGE}; transmission is on (blend `
      + 'transmissive): the stripes and tiles show through the block, refracted and tinted cyan.',
    references: [],
    judge: false,
    settings: { set: 'transmission', blend: 'transmissive' },
  },
  {
    id: 'transmission-off',
    description: 'R11: the same view as transmission-on, but the block is drawn opaque (blend '
      + 'opaque): a solid pale ice-coloured block, nothing visible through it.',
    references: [],
    judge: false,
    settings: { set: 'transmission', blend: 'opaque' },
  },
  {
    id: 'oit-on',
    description: `R11: three translucent glass spheres (pink, cyan, amber) in a row and a cyan `
      + `glass slab in front of them ${STAGE}; the transparent.oit pass is on: every overlap `
      + 'blends in depth order, no holes.',
    references: [],
    judge: false,
    settings: { set: 'oit' },
  },
  {
    id: 'oit-off',
    description: 'R11: the same view as oit-on, but the transparent.oit pass is off: the '
      + 'translucent spheres and slab are not drawn, only the opaque stage remains.',
    references: [],
    judge: false,
    settings: { set: 'oit' },
    passes: { disable: ['transparent.oit'] },
  },
  {
    id: 'overlap',
    description: `R11: ${GLASSES} ${STAGE}; the order of every overlap is correct, with no `
      + 'sorting pops or holes where the objects interpenetrate.',
    references: [],
    judge: false,
    settings: { set: 'overlap' },
  },
  {
    id: 'slime',
    description: 'glossy translucent green slime pool and blobs; the tiles and objects behind and '
      + 'under the slime are visible, refracted and tinted green; bubbles inside; overlapping '
      + 'blobs composite in the right order',
    references: ['ref-1', 'ref-3'],
    judge: true,
    settings: { set: 'slime' },
  },
  {
    id: 'water',
    description: 'a basin of clear still water; the tiled bottom is visible through the water, '
      + 'slightly refracted and tinted blue-green with depth; the surface reflects the sky',
    references: ['ref-1', 'ref-3'],
    judge: true,
    settings: { set: 'water' },
  },
];

export default {
  id: 'm11-transmission',
  milestone: 'm11',
  title: 'Transmission, refraction with absorption and order-independent translucency',
  engine: { msaa: 4, toneMapper: 'agx' },
  variants: VARIANTS,

  async build(engine, lib, variant) {
    const settings = variant.settings ?? {};
    const builder = SETS[settings.set ?? variant.id];
    if (!builder) throw new Error(`m11-transmission: no builder for variant '${variant.id}'`);
    builder(engine, lib, settings);
    const [target, distance, yaw, pitch] = CAMERAS[settings.camera ?? settings.set ?? 'ice'];
    engine.setOrbit(target, distance, yaw, pitch);
  },
};
