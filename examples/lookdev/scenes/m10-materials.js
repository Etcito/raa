// Lookdev scene m10-materials (chain milestone 4, R10): glossy clear-coated inflatables (the
// inflatable hammer, balls and bumper rings in raa.inflatable and raa.candy) under the m08
// `day` sky, and neon tubes in raa.neon at night with physically based bloom. On/off pairs
// for the clear coat (material override, settings-driven) and the `bloom` pass (Q15).
// Variants are grouped in sets (settings.set), one builder per set in SETS.
// Pure module (S14): no top-level side effects, node reads id/milestone/variants.

const BACKGROUND = [0.3, 0.34, 0.4]; // display-referred (S12), hidden by the sky
// Sun from the upper right, slightly behind (as m09, 35 deg high).
const SUN = { direction: [-0.8067, -0.5736, 0.1422], color: [1, 0.93, 0.82], intensity: 7 };
// Cool dim moonlight for the night sets: the neon tubes are the main light.
const MOON = { direction: [-0.2955, -0.2924, 0.9096], color: [0.6, 0.7, 1], intensity: 0.3 };
const AMBIENT = { color: [0.55, 0.6, 0.7], intensity: 0.35 };
// The m08 `day` sky overrides, with m09's pale blue haze ground.
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
// BLOOM defaults, set explicitly so the capture documents them.
const BLOOM = { strength: 0.04, radius: 0.85 };
const SHADOWS = { softness: 1, distance: 30 };

// Orbit cameras: [target, distance m, yaw rad, pitch rad] (as m09).
const CAMERAS = {
  inflatables: [[0, 0.5, 0], 4.2, 0, 0.3],
  clearcoat: [[0, 0.6, 0], 2.2, 0.35, 0.2],
  neon: [[0, 0.55, 0], 3.6, 0, 0.12],
};

// Inflatable palette (sRGB): ref-1's pink, yellow, blue and white plastic.
const PINK = [0.95, 0.2, 0.55];
const YELLOW = [1, 0.78, 0.1];
const BLUE = [0.12, 0.45, 0.95];
const WHITE = [0.98, 0.96, 0.92];
// Balls: radius m, x, z, function, params.
const BALLS = [
  { r: 0.32, x: -0.35, z: 0.55, fn: 'raa.inflatable', params: { color: PINK, stripeColor: WHITE } },
  { r: 0.24, x: 0.45, z: 0.75, fn: 'raa.candy', params: { color: WHITE, stripeColor: PINK } },
  { r: 0.28, x: 1.15, z: 0.2, fn: 'raa.inflatable', params: { color: YELLOW, stripeColor: BLUE } },
  { r: 0.2, x: -1.0, z: 0.9, fn: 'raa.candy', params: { color: YELLOW, stripeColor: BLUE, stripeCount: 4 } },
];
// Bumper rings lying on the floor: major radius m, tube radius m, x, z, params.
const BUMPERS = [
  { R: 0.45, r: 0.16, x: 0.55, z: -0.7, params: { color: BLUE, stripeColor: YELLOW, seamCount: 12 } },
  { R: 0.35, r: 0.13, x: -1.5, z: 0.1, params: { color: YELLOW, stripeColor: PINK, seamCount: 10 } },
];
const HAMMER = { t: [-0.75, 0, -0.45], yaw: 0.4 };
// Floor slab (top at y 0): glossy blue by day, dark violet at night (sRGB).
const FLOOR = { half: [8, 0.05, 8], t: [0, -0.05, 0], day: [0.2, 0.5, 0.95], night: [0.08, 0.07, 0.16] };

// Neon colours (sRGB) and the shared tube radius (m).
const NEON_PINK = [1, 0.25, 0.7];
const NEON_CYAN = [0.15, 0.85, 1];
const NEON_YELLOW = [1, 0.8, 0.2];
const NEON_PURPLE = [0.6, 0.3, 1];
const TUBE_R = 0.025;

const yawQuat = (rad) => [0, Math.sin(rad / 2), 0, Math.cos(rad / 2)];

// Sets each sub-mesh material of `desc` (in order) and returns the mesh handle.
function upload(engine, desc, materials) {
  const subMeshes = (desc.subMeshes ?? [{ first: 0, count: desc.indices.length }]).map((s, i) => ({
    ...s,
    material: materials[Math.min(i, materials.length - 1)],
  }));
  return engine.addMesh({ ...desc, subMeshes });
}

function place(engine, mesh, t, r = yawQuat(0)) {
  const node = engine.addNode(null, t, r, [1, 1, 1]);
  return engine.addMeshInstance(node, mesh);
}

// Closed sweepTube path: a circle of radius r in the XZ plane, or XY when `upright`.
function circle(r, steps, upright = false) {
  const at = (a) => (upright ? [r * Math.cos(a), r * Math.sin(a), 0] : [r * Math.cos(a), 0, r * Math.sin(a)]);
  return Array.from({ length: steps }, (_, i) => at((2 * Math.PI * i) / steps));
}

// Polyline resampled at `per` samples per span (smooth sweepTube frames on straight runs).
function polyline(points, per) {
  const out = [];
  for (let i = 0; i + 1 < points.length; i++) {
    for (let s = 0; s < per; s++) {
      const t = s / per;
      out.push([0, 1, 2].map((k) => points[i][k] + (points[i + 1][k] - points[i][k]) * t));
    }
  }
  out.push(points[points.length - 1].slice());
  return out;
}

// Daylight with the m08 sky and m09 shadows (inflatable sets).
function daylight(engine) {
  engine.setBackground(BACKGROUND);
  engine.setLight(SUN);
  engine.setAmbient(AMBIENT);
  engine.setSky({ preset: 'day', ...DAY_SKY });
  engine.setEnvironmentLighting({ diffuse: 0.7, specular: 1 });
  engine.setToneMapping({ look: 'punchy' });
  engine.setShadows(SHADOWS);
}

// Night under the `night` preset with bloom (neon sets).
function night(engine) {
  engine.setBackground(BACKGROUND);
  engine.setLight(MOON);
  engine.setAmbient(AMBIENT);
  engine.setSky({ preset: 'night' });
  engine.setEnvironmentLighting({ diffuse: 0.6, specular: 1 });
  engine.setBloom(BLOOM);
}

// Surface material of function `fn`; `clearcoat` overrides the layer (undefined = keep).
function surface(engine, fn, params, clearcoat) {
  const override = clearcoat === undefined ? {} : { clearcoat };
  return engine.addSurfaceMaterial({ function: fn, params, ...override });
}

function floor(engine, geometry, color, roughness) {
  const m = engine.addSurfaceMaterial({ color, roughness });
  place(engine, upload(engine, geometry.roundedBox(FLOOR.half, 0.02, 2), [m]), FLOOR.t);
}

// Hammer (vertex-coloured stripes, stripe band off), balls and bumper rings on a glossy
// blue floor; `cc` overrides the clear coat of every inflatable (undefined = function's).
function buildInflatables(engine, lib, settings) {
  const { geometry, objects } = lib;
  const cc = settings.clearcoat;
  daylight(engine);
  floor(engine, geometry, FLOOR.day, 0.3);
  const hammer = surface(engine, 'raa.inflatable', { color: [1, 1, 1], stripeWidth: 0, seamDepth: 0.12 }, cc);
  place(engine, upload(engine, objects.inflatableHammer(), [hammer]), HAMMER.t, yawQuat(HAMMER.yaw));
  const balls = settings.set === 'clearcoat' ? BALLS.slice(0, 1) : BALLS;
  for (const b of balls) {
    const m = surface(engine, b.fn, b.params, cc);
    place(engine, upload(engine, geometry.uvSphere(b.r, 64, 32), [m]), [b.x, b.r, b.z]);
  }
  if (settings.set === 'clearcoat') return;
  for (const k of BUMPERS) {
    const ring = geometry.sweepTube(circle(k.R, 64), k.r, 24, { closed: true });
    place(engine, upload(engine, ring, [surface(engine, 'raa.inflatable', k.params, cc)]), [k.x, k.r, k.z]);
  }
}

// Neon tubes of four colours at night: the letter S, an upright ring, a zigzag and a bar,
// standing on a dark glossy floor that reflects their glow.
function buildNeon(engine, lib) {
  const { geometry, objects } = lib;
  night(engine);
  floor(engine, geometry, FLOOR.night, 0.25);
  const neon = (color) => engine.addSurfaceMaterial({ function: 'raa.neon', params: { color } });
  place(engine, upload(engine, objects.neonTube(), [neon(NEON_PINK)]), [-0.2, 0, 0]);
  const ring = geometry.sweepTube(circle(0.35, 64, true), TUBE_R, 16, { closed: true });
  place(engine, upload(engine, ring, [neon(NEON_CYAN)]), [0.85, 0.45, -0.2], yawQuat(-0.3));
  const zig = polyline([[-0.4, 0, 0], [-0.2, 0.25, 0], [0, 0, 0], [0.2, 0.25, 0], [0.4, 0, 0]], 12);
  const zigzag = geometry.sweepTube(zig, TUBE_R, 16);
  place(engine, upload(engine, zigzag, [neon(NEON_YELLOW)]), [-1.15, 0.2, -0.3], yawQuat(0.35));
  const bar = geometry.sweepTube(polyline([[-1.2, 0, 0], [1.2, 0, 0]], 24), TUBE_R, 16);
  place(engine, upload(engine, bar, [neon(NEON_PURPLE)]), [0, 1.15, -0.6]);
}

// Set name (variant.settings.set) to builder.
const SETS = { inflatables: buildInflatables, clearcoat: buildInflatables, neon: buildNeon };

const INFLATABLES = 'an upright striped inflatable toy hammer, shiny inflatable balls (pink, '
  + 'white with pink candy stripes, yellow with blue stripes) and two inflated bumper rings '
  + '(blue, yellow) lying on a glossy blue floor under a bright blue sky with white clouds';
const PLASTIC = 'glossy inflatable plastic with a crisp clear-coat highlight over a coloured '
  + 'base, inflated seams (soft pillowed bulges between the seams)';
const NEON = 'several glowing neon tubes at night (a pink script letter S, a cyan ring, a '
  + 'yellow zigzag and a purple bar) standing on a dark glossy floor under a dark navy night '
  + 'sky';
const GLOW = 'neon tubes glow with soft physically based bloom, white-hot cores, no halo '
  + 'banding';
// R10 judges the materials against the references; the set is a sparse procedural lookdev
// arrangement, so the judge is told what to score and what belongs to later milestones.
const BASIS = 'Engine lookdev capture of the materials milestone. Score two things: that the '
  + 'capture shows what this description lists, and that the MATERIALS (surface look, gloss, '
  + 'highlights, glow) match the reference. The reference is a finished illustration and is '
  + 'NOT the target for the scene: the sparse arrangement of a few simple procedural objects, '
  + 'the absence of characters, buildings, transparency, liquids, particles, depth of field, '
  + 'fog and other post effects are expected at this milestone and must not lower the score. '
  + 'Shows: ';

// The variant list: later sets append their variants here and their builder to SETS.
const VARIANTS = [
  {
    id: 'inflatables',
    description: `${BASIS}${INFLATABLES}; ${PLASTIC}; sky reflections on the curved plastic.`,
    references: ['ref-1'],
    judge: true,
    settings: { set: 'inflatables' },
  },
  {
    id: 'clearcoat-on',
    description: 'R10: a close view of the inflatable toy hammer and a pink inflatable ball on '
      + 'a glossy blue floor in daylight; the clear coat is on: a crisp, sharp white highlight '
      + 'and sharp sky reflection sit over the softer coloured base sheen.',
    references: [],
    judge: false,
    settings: { set: 'clearcoat', clearcoat: 1 },
  },
  {
    id: 'clearcoat-off',
    description: 'R10: the same close view as clearcoat-on, but the clear coat is off '
      + '(clearcoat 0): only the broad, softer highlight of the coloured base remains, no crisp '
      + 'top-layer highlight or sharp sky reflection, so the plastic looks duller.',
    references: [],
    judge: false,
    settings: { set: 'clearcoat', clearcoat: 0 },
  },
  {
    id: 'neon',
    description: `${BASIS}${NEON}; ${GLOW}; the glow tints the floor around each tube.`,
    references: ['ref-4'],
    judge: true,
    settings: { set: 'neon' },
  },
  {
    id: 'bloom-on',
    description: `R10: ${NEON}; bloom is on: ${GLOW}, a smooth glow fading into the dark sky.`,
    references: [],
    judge: false,
    settings: { set: 'neon' },
  },
  {
    id: 'bloom-off',
    description: `R10: the same view as bloom-on, ${NEON}, but bloom is off: the tubes are `
      + 'bright with white cores and hard edges, with no glow around them.',
    references: [],
    judge: false,
    settings: { set: 'neon' },
    passes: { disable: ['bloom'] },
  },
];

export default {
  id: 'm10-materials',
  milestone: 'm10',
  title: 'Material functions, clear coat, fur shells and emissive bloom',
  engine: { msaa: 4, toneMapper: 'agx' },
  variants: VARIANTS,

  async build(engine, lib, variant) {
    const settings = variant.settings ?? {};
    const builder = SETS[settings.set ?? variant.id];
    if (!builder) throw new Error(`m10-materials: no builder for variant '${variant.id}'`);
    builder(engine, lib, settings);
    const [target, distance, yaw, pitch] = CAMERAS[settings.camera ?? settings.set ?? 'inflatables'];
    engine.setOrbit(target, distance, yaw, pitch);
  },
};
