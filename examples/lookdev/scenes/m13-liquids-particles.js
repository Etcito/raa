// Lookdev scene m13-liquids-particles (chain milestone 6, R13): animated liquid surfaces.
// Water (raa.liquid.water): a basin with a tiled floor, a sloped sand beach and rocks breaking
// the surface (shore foam), Gerstner waves with crest foam. Slime (raa.liquid.slime): a
// neon-green river in a channel with inflatable pads. Mud (raa.liquid.mud): a deep-chocolate
// opaque wet pit with splash crowns, shore froth and bubbles between lime inflatable pads on
// a grassy bank. waves-t0 / waves-t1 show the water set at two times.
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
  // Slime looks down onto the river: at the old 0.44 rad pitch the Fresnel sky term turned
  // the whole surface into a grey film (card 4941 software preview).
  slime: [[0, -0.15, 0.1], 3.8, 0.2, 0.72],
  // Mud sits close, like ref-3: the pit fills the frame and the crowns stand against the
  // far mud. 29 deg down: at 21 deg (0.36 rad) the coat's Fresnel mirrored the blue sky
  // over most of the pit and the brown read lilac (card 5070).
  mud: [[0.1, 0.0, 0.3], 3.6, -0.3, 0.5],
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
  // A more saturated lime (card 4942's judge saw a flat, greyish film).
  color: [0.34, 0.95, 0.03],
  // Denser body (with transmission 0.6 below) so the river reads as neon-green slime with
  // the tiles shimmering through it, not as clear water over green tiles.
  absorption: 4.5,
  glossiness: 0.96,
  bubbleScale: 0.32,
  // Shore lace at the channel walls; little crest foam, which covered the river in a
  // noise-texture lace and hid the bubbles and the refracted tiles (card 5034).
  foamWidth: 0.1,
  foamAmount: 0.3,
  // A gentle detail normal: the old 0.32 mottled the sky reflection into a grey skin
  // (card 5031) instead of the glossy green surface the reference shows; 0.08 still read
  // as a mottled surface (card 4942).
  detailStrength: 0.05,
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

// Mud: one slow, long, low swell. `color` is sRGB, converted to linear by the material
// table: a deep chocolate within the reference's range (about #4a2414 to #5c2f1b). Card
// 5078's judge read the old milk chocolate (0.55, 0.32, 0.15) as a flat, lighter
// orange-rust plane: a bright diffuse hid the specular relief. A darker body lets the sun
// and sky speculars carry the wet swells. The shader shades the crests lighter and the
// troughs darker and adds a low-frequency fbm slope, so the swell breaks the sun highlight
// into bands at capture distance.
const MUD = {
  wave0: [0.12, 3.2, 0.3, 0.55],
  wave1: [0.07, 2.0, 1.4, 0.5],
  wave2: [0.02, 1.2, -0.7, 0.35],
  wave3: [0.008, 0.75, 2.4, 0.25],
  speeds: [0.24, 0.19, 0.145, 0.115],
  color: [0.34, 0.17, 0.1],
  wetness: 1,
  // Big, slow bubbles: a few viscous domes (about one per 5 m2 of pit) instead of the
  // invisible dense field of card 5078.
  bubbleScale: 0.52,
  bubbleRate: 0.35,
  // White shore froth where the mud is shallower than foamWidth (the depth rides in the
  // vertex colour alpha, see mudMesh): a soft irregular band along the shore and rings
  // around the pads, a little on the swell crests.
  foamWidth: 0.16,
  foamAmount: 0.35,
  // Low-frequency fbm slope (detailScale = 0.55 m^-1) plus a fine detail layer: the sun
  // highlight breaks into crests and troughs instead of a flat sheen.
  detailStrength: 0.2,
  detailScale: 0.55,
};
// Mud set lighting: a warmer sun and dimmer sky irradiance than daylight(). The sky
// irradiance of DAY_SKY carried about as much light as the sun on the pit, all of it blue
// (measured on the rim: light about (2.2, 2.4, 3.8)). The sun shines from behind the
// camera's left shoulder, 46 deg up (the camera sits at +z): the camera-facing sides of
// the crowns, drops and pads are lit and the pad shadows fall behind the pads. The old
// sun in front of the camera (travel +z) backlit every crown and threw the pad shadows
// over them, so the chocolate crowns rendered grey-black (card 5064 capture: crown pixels
// about (14..35, 13..35, 27..32), only the coat's sky reflection) and the shadowed pit
// read purple. The env specular gives the wet coat a Fresnel sky sheen.
const MUD_LIGHT = {
  sun: { ...SUN, direction: [0.6, -0.72, -0.35], color: [1, 0.88, 0.72] },
  env: { diffuse: 0.45, specular: 0.5 },
};
// Smoothstep of v over [a, b].
function smooth(a, b, v) {
  const t = Math.min(1, Math.max(0, (v - a) / (b - a)));
  return t * t * (3 - 2 * t);
}
// Ground height of the mud pit (the mud surface is y = 0): the bed 0.3 m down, then a
// steep grassy bank that reaches the mud line at r = 2.62 m and rises to a rounded ridge
// 0.61 m above the mud at r = 3.45 m, before the meadow settles at 0.36 m. Card 5078's
// judge saw the flat surround and its dark mud-stained skirt instead of a bank: the bank
// now rises fast enough just outside the crowns to frame the pit, and there is no stain
// band.
function mudGround(x, z) {
  const r = Math.hypot(x, z * 1.1);
  const bank = smooth(2.15, 3.25, r);
  const ridge = 0.25 * Math.exp(-(((r - 3.45) / 0.5) ** 2));
  return -0.3 + 0.66 * bank + ridge;
}
// Grass of the bank (user material function, the bank's vertex colour is its colour):
// broad clumps, tufts and fine blade speckle in brightness and hue, with a blade-scale
// normal and AO breakup; the fine detail fades out beyond 6 m so the far meadow does not
// alias. A flat colour read as a plain plane (card 5072).
const GRASS_WGSL = `
fn material_main(in: MaterialInput) -> MaterialOutput {
  var o = raa_default_output(in);
  let xz = in.position.xz;
  let near = 1.0 - smoothstep(6.0, 16.0, length(xz));
  let clump = raa_fbm3(vec3<f32>(xz.x * 1.3, 0.0, xz.y * 1.3), 4);
  let tuft = raa_noise3(vec3<f32>(xz.x * 11.0, 0.3, xz.y * 11.0));
  let blade = raa_noise3(vec3<f32>(xz.x * 55.0, 1.7, xz.y * 55.0)) * near;
  let warm = clamp(0.5 + 0.9 * clump + 0.3 * tuft, 0.0, 1.0);
  let hue = mix(vec3<f32>(0.75, 1.0, 0.85), vec3<f32>(1.2, 1.06, 0.55), warm);
  let light = clamp(0.82 + 0.3 * clump + 0.16 * tuft * near + 0.28 * blade, 0.4, 1.4);
  o.base_color = in.color.rgb * hue * light;
  let bx = raa_noise3(vec3<f32>(xz.x * 38.0, 5.1, xz.y * 38.0));
  let bz = raa_noise3(vec3<f32>(xz.x * 38.0, 9.3, xz.y * 38.0));
  o.normal = raa_perturb_normal(o.normal, in.tangent, normalize(vec3<f32>(0.45 * bx * near, 0.45 * bz * near, 1.0)));
  o.roughness = 0.78;
  o.metallic = 0.0;
  o.ao = clamp(0.75 + 0.25 * (0.5 + 0.5 * tuft) + 0.15 * blade, 0.55, 1.0);
  return o;
}
`;
// Wet chocolate of the splash tongues (user material function): large patches of lighter and
// darker mud, a fine lumpy normal and a glossy wet skin, so a tongue reads as mud rather
// than a smooth plastic lump (card 5078's judge).
const CROWN_WGSL = `
fn material_main(in: MaterialInput) -> MaterialOutput {
  var o = raa_default_output(in);
  let p = in.position;
  let blotch = raa_fbm3(vec3<f32>(p.x * 3.1, p.y * 4.0, p.z * 3.1), 3);
  let fine = raa_noise3(vec3<f32>(p.x * 24.0, p.y * 30.0, p.z * 24.0));
  let bx = raa_noise3(vec3<f32>(p.x * 19.0, 3.1, p.z * 19.0));
  let bz = raa_noise3(vec3<f32>(p.x * 19.0, 7.7, p.z * 19.0));
  o.base_color = o.base_color * (0.82 + 0.22 * blotch) * (0.92 + 0.12 * fine);
  o.normal = raa_perturb_normal(o.normal, in.tangent, normalize(vec3<f32>(0.2 * bx, 0.2 * bz, 1.0)));
  o.roughness = clamp(o.roughness + 0.1 * blotch, 0.18, 0.45);
  o.clearcoat = 0.65;
  o.clearcoat_roughness = 0.07;
  o.metallic = 0.0;
  return o;
}
`;
// Glossy lime-green inflatable pads floating in the mud (ref-3's stepping pads): low, wide
// discs, [x, z, radius, disc height]. Each sits PAD_SINK m into the local swell, so its
// rim stands 0.15..0.18 m above the mud and its flat top about 0.07..0.1 m. Card 5072's
// pucks (0.65 m tall, 1.0..1.2 m wide) read as tall cylinders; ref-3's pads are discs far
// wider than tall.
const MUD_PADS = [
  [-1.45, -0.55, 0.66, 0.24],
  [0.75, -1.35, 0.56, 0.22],
  [1.5, 0.75, 0.62, 0.24],
  [-0.55, 1.4, 0.52, 0.21],
];
const PAD_SINK = 0.06; // m below the local mud line
// Rim tube radius as a fraction of the pad radius: the rim stands 0.16 r (about 0.08..0.11
// m) above the flat top, the thick raised ring of ref-3's inflatable pads.
const PAD_RIM = 0.16;
const PAD_KEEL = 0.12; // m the pad's underside reaches below its base
// A satin lime plastic: the old mirror clearcoat (roughness 0.12, clearcoat 1) mirrored the
// blue sky over the tops, which read as concave sky-blue cups (card 5078).
const MUD_PAD = { color: [0.58, 0.9, 0.16], roughness: 0.26, clearcoat: 0.6, clearcoatRoughness: 0.12 };
// Wet band at the pads' bases: vertex colour multiplier at the mud line and its height m.
const MUD_PAD_WET = { tint: [0.42, 0.36, 0.24], height: 0.05 };
// Mud splash crowns around each pad wall are sculpted meshes, not particles: the splash
// emitter's droplets are camera-facing discs, which read as loose clusters of brown beads
// at capture distance (card 5039, attempt 9) instead of ref-3's crowns of mud. Per pad: a
// low collar of mud climbing the pad wall plus `lobes` separate splash tongues. Card 5078's
// judge read the old thin continuous sheet as "thin, tapered horn spikes": each tongue is
// now a thick sheet with a wide base and a bulbous, rounded tip bent outward (an elliptical
// swept tube, not a cone), so it reads as a heavy tongue of wet chocolate.
const MUD_CROWN = {
  // Deep chocolate like the pit and a little warmer; sRGB.
  color: [0.38, 0.2, 0.11],
  roughness: 0.26,
  clearcoat: 0.55,
  clearcoatRoughness: 0.08,
  lobes: 7, // separate tongues in a ring around every pad (6..8 per ref-3)
  height: [0.44, 0.56, 0.5], // m above the mud at the tongue tips, cycling per lobe
  flare: [0.24, 0.3, 0.27], // m the tip of a tongue stands out from the pad wall
  foot: 0.02, // m the tongue's base stands off the pad wall
  sink: 0.06, // m the tongue reaches below the mud line
  skirt: 0.24, // m the mud collar around the pad reaches out from the wall
  rise: 0.09, // m the mud climbs the pad wall
  // Elliptical cross-section: wide tangentially (a broad sheet) and thin radially.
  widthBase: 0.17, // m tangential half width at the base
  widthMid: 0.075, // m at the waist
  widthTip: 0.09, // m at the bulbous tip
  thickBase: 0.05, // m radial half thickness at the base
  thickMid: 0.022, // m at the waist
  thickTip: 0.06, // m at the bulbous tip
  capLength: 0.07, // m the rounded tip cap reaches past the last ring
  segments: 20, // around one tongue
  steps: 18, // rings along one tongue
  capSteps: 5, // rounded tip cap rings
};
// White spray (ref-3's splash spray): many small thin streaks, elongated along their
// flight, arcing out of the tip of every crown tongue (mudSprayArcs). Card 5078's judge saw
// large opaque blobs: the drops are now half the width and stretched 4x, and the emitter
// below adds a fine soft-edged particle spray around them.
const MUD_DROPS = {
  color: [1, 1, 1],
  roughness: 0.5,
  clearcoat: 0.25,
  clearcoatRoughness: 0.2,
  emissive: [0.35, 0.35, 0.35],
  count: 12,
  times: [0.02, 0.05, 0.08, 0.11, 0.15, 0.19, 0.23, 0.27],
  speed: [1.2, 2.4],
  radius: [0.004, 0.009], // m
  stretch: 4, // length over width along the flight
};
// Fine soft spray at the lobe tips: many small semi-transparent droplets on ballistic
// arcs, stretched along their velocity by the particle pass, so the plume reads as fine
// streaks rather than flat discs.
const MUD_SPRAY = {
  kind: 'splash',
  spread: 0.42,
  rate: 260,
  lifetime: [0.12, 0.3],
  speed: [1.0, 2.1],
  size: [0.006, 0.013],
  drag: 0.4,
  colors: [[0.95, 0.97, 1], [1, 1, 1]],
  opacity: 0.6,
  volume: [0.03, 0.01, 0.03],
  maxParticles: 96,
};
// Bubbles on the open mud: [x, z, radius m], and the ring of a popped one.
const MUD_BUBBLES = [
  [-0.25, 0.55, 0.11],
  [1.02, -0.32, 0.14],
  [-1.3, 0.95, 0.09],
];
const MUD_BUBBLE_RING = { x: 0.0, z: -0.15, radius: 0.14 };
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
// `warp` (optional) remaps each grid coordinate before the height is evaluated.
function heightfield(geometry, w, d, n, height, colour, warp = (v) => v) {
  const desc = geometry.grid(w, d, n, n);
  const p = new Float32Array(desc.positions);
  const nrm = new Float32Array(desc.normals);
  const col = new Float32Array((p.length / 3) * 4);
  const e = 0.01;
  for (let i = 0; i < p.length / 3; i++) {
    const x = warp(p[3 * i]);
    const z = warp(p[3 * i + 2]);
    const y = height(x, z);
    p[3 * i] = x;
    p[3 * i + 1] = y;
    p[3 * i + 2] = z;
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

// A lumpy boulder: a sphere displaced along its normal by a few low-frequency lobes, with
// normals rebuilt from the faces (accumulated per welded position, so the seam is smooth).
function boulder(geometry, r, seed) {
  const desc = geometry.uvSphere(r, 48, 24);
  const p = new Float32Array(desc.positions);
  const lobe = (a, b, c) => 0.12 * Math.sin(a * 3.1 + seed) * Math.cos(b * 2.3 + 1.7 * seed)
    + 0.07 * Math.sin(b * 5.3 + c * 4.1 + seed) + 0.04 * Math.cos(a * 7.7 - c * 6.1);
  for (let i = 0; i < p.length; i += 3) {
    const [a, b, c] = [p[i] / r, p[i + 1] / r, p[i + 2] / r];
    const k = 1 + lobe(a, b, c);
    p[i] *= k;
    p[i + 1] *= k * (b < -0.4 ? 0.85 : 1);
    p[i + 2] *= k;
  }
  const key = (i) => `${Math.round(p[3 * i] * 1e4)},${Math.round(p[3 * i + 1] * 1e4)},${Math.round(p[3 * i + 2] * 1e4)}`;
  const acc = new Map();
  const idx = desc.indices;
  for (let t = 0; t < idx.length; t += 3) {
    const [i0, i1, i2] = [idx[t], idx[t + 1], idx[t + 2]];
    const e1 = [0, 1, 2].map((k) => p[3 * i1 + k] - p[3 * i0 + k]);
    const e2 = [0, 1, 2].map((k) => p[3 * i2 + k] - p[3 * i0 + k]);
    const fn = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    for (const i of [i0, i1, i2]) {
      const a = acc.get(key(i)) ?? [0, 0, 0];
      acc.set(key(i), [a[0] + fn[0], a[1] + fn[1], a[2] + fn[2]]);
    }
  }
  const nrm = new Float32Array(desc.normals);
  for (let i = 0; i < p.length / 3; i++) {
    const a = acc.get(key(i));
    const l = a ? Math.hypot(...a) : 0;
    if (l > 0) nrm.set([a[0] / l, a[1] / l, a[2] / l], 3 * i);
  }
  return geometry.withTangents({ positions: p, normals: nrm, uvs: desc.uvs, indices: desc.indices });
}

function rock(engine, geometry, [x, z, r, squash, y], color = [0.5, 0.47, 0.42]) {
  const m = engine.addSurfaceMaterial({ color, roughness: 0.75 });
  place(engine, upload(engine, boulder(geometry, r, x * 3 + z), m), [x, y, z], [1, squash, 1]);
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
    transmission: 0.6,
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
  // Red is the swell fade: 0 at the bank (the vertex stage keeps the mud from poking through
  // the grass) and 1 in the pit, including around the pads where the foam depth is low. The
  // alpha is the mud depth the foam reads.
  const shore = (p) => smooth(0.005, 0.18, -mudGround(p[0], p[2]));
  return geometry.grid(LIQUID_SIZE, LIQUID_SIZE, LIQUID_SEGMENTS, LIQUID_SEGMENTS, {
    color: (p) => [shore(p), 1, 1, depth(p)],
  });
}

// Rest height of the mud waves at (x, z) and time t s: the CPU twin of raa-liquid's
// wave_height (src/waves.rs) over MUD's packing, so the crowns sit on the swell.
function mudHeight(x, z, t) {
  let y = 0;
  [MUD.wave0, MUD.wave1, MUD.wave2, MUD.wave3].forEach(([a, l, theta], i) => {
    const k = (2 * Math.PI) / l;
    y += a * Math.sin(k * (Math.cos(theta) * x + Math.sin(theta) * z - MUD.speeds[i] * t));
  });
  return y;
}

// Displaces a mesh by y += f(x, z) and transforms its normals with the inverse transpose
// of that map, (n.x - fx n.y, n.y, n.z - fz n.y), so a ring laid flat follows the swell.
function displaceY(geometry, desc, f) {
  const p = desc.positions;
  const n = desc.normals;
  const e = 0.005;
  for (let i = 0; i < p.length; i += 3) {
    const [x, z] = [p[i], p[i + 2]];
    const fx = (f(x + e, z) - f(x - e, z)) / (2 * e);
    const fz = (f(x, z + e) - f(x, z - e)) / (2 * e);
    p[i + 1] += f(x, z);
    const m = [n[i] - fx * n[i + 1], n[i + 1], n[i + 2] - fz * n[i + 1]];
    const l = Math.hypot(...m);
    [n[i], n[i + 1], n[i + 2]] = [m[0] / l, m[1] / l, m[2] / l];
  }
  delete desc.tangents;
  return geometry.withTangents(desc);
}

// The mud crown of one pad (x, z, radius r) at time t: a concave skirt of mud climbing the
// pad wall all around it (following the swell) plus one flared crown sheet (mudCrownSheet).
// Where lobe k of pad i (MUD_PADS index) peaks: its angle about the pad axis, its height
// above the mud and its outward flare.
function mudLobe(i, k) {
  const C = MUD_CROWN;
  const phase = 0.7 + 1.9 * i;
  return {
    a: phase + (k / C.lobes) * Math.PI * 2 + 0.35 * Math.sin(3 * k + phase),
    h: C.height[k % C.height.length],
    flare: C.flare[(k + i) % C.flare.length],
  };
}

// One splash tongue of pad i (pad-local XZ, world Y on the swell): an elliptical swept
// tube along a cubic Bezier that climbs from under the mud, leans outward and ends in a
// bulbous rounded club. Wide at the base, thin at the waist and thick and round at the
// tip: ref-3's heavy tongues of mud rather than card 5078's thin horn spikes. A tongue is
// closed (base fan + rounded cap) so its normals and shadows are well defined.
function mudTongue(geometry, [x, z, r], i, k, t) {
  const C = MUD_CROWN;
  const { a, h, flare } = mudLobe(i, k);
  const ca = Math.cos(a);
  const sa = Math.sin(a);
  const r0 = r + C.foot;
  const baseY = mudHeight(x + r0 * ca, z + r0 * sa, t);
  const path = [
    [r0, -C.sink],
    [r0 + 0.1 * flare, 0.52 * h],
    [r0 + 0.55 * flare, 1.02 * h],
    [r0 + flare, 0.98 * h],
  ];
  const at = (u) => {
    const v = 1 - u;
    const b = [v * v * v, 3 * v * v * u, 3 * v * u * u, u * u * u];
    const db = [-3 * v * v, 3 * v * v - 6 * v * u, 6 * v * u - 3 * u * u, 3 * u * u];
    const c = [0, 1].map((j) => b.reduce((m, bi, n) => m + bi * path[n][j], 0));
    const d = [0, 1].map((j) => db.reduce((m, bi, n) => m + bi * path[n][j], 0));
    const l = Math.hypot(d[0], d[1]) || 1;
    // Outward normal of the centreline in the radial plane: the tangent turned clockwise.
    return { c, t: [d[0] / l, d[1] / l], n: [d[1] / l, -d[0] / l] };
  };
  const section = (u) => {
    const taper = smooth(0, 0.55, u);
    const bulb = smooth(0.68, 0.92, u);
    return {
      w: (C.widthBase + (C.widthMid - C.widthBase) * taper) * (1 - bulb) + C.widthTip * bulb,
      d: (C.thickBase + (C.thickMid - C.thickBase) * taper) * (1 - bulb) + C.thickTip * bulb,
    };
  };
  const segs = C.segments;
  const positions = [];
  const uvs = [];
  const rings = [];
  for (let j = 0; j <= C.steps; j++) {
    const u = j / C.steps;
    const c = at(u);
    const { w, d } = section(u);
    const ring = [];
    for (let s = 0; s <= segs; s++) {
      const phi = (s / segs) * Math.PI * 2;
      const rho = c.c[0] + c.n[0] * d * Math.sin(phi);
      ring.push(positions.length);
      positions.push([
        rho * ca - w * Math.cos(phi) * sa,
        baseY + c.c[1] + c.n[1] * d * Math.sin(phi),
        rho * sa + w * Math.cos(phi) * ca,
      ]);
      uvs.push(s / segs, u);
    }
    rings.push(ring);
  }
  // Rounded tip: rings advance along the centreline tangent while shrinking to a point, so
  // the club ends in a bulb instead of a cut cone.
  const end = at(1);
  const { w: wEnd, d: dEnd } = section(1);
  for (let m = 1; m <= C.capSteps; m++) {
    const q = (m / C.capSteps) * (Math.PI / 2);
    const shrink = Math.cos(q);
    const shift = C.capLength * Math.sin(q);
    const rhoC = end.c[0] + end.t[0] * shift;
    const yC = baseY + end.c[1] + end.t[1] * shift;
    if (m === C.capSteps) {
      const pole = positions.length;
      positions.push([
        (end.c[0] + end.t[0] * C.capLength) * ca,
        baseY + end.c[1] + end.t[1] * C.capLength,
        (end.c[0] + end.t[0] * C.capLength) * sa,
      ]);
      uvs.push(0.5, 1);
      rings.push([pole]);
      break;
    }
    const ring = [];
    for (let s = 0; s <= segs; s++) {
      const phi = (s / segs) * Math.PI * 2;
      const rho = rhoC + end.n[0] * dEnd * shrink * Math.sin(phi);
      ring.push(positions.length);
      positions.push([
        rho * ca - wEnd * shrink * Math.cos(phi) * sa,
        yC + end.n[1] * dEnd * shrink * Math.sin(phi),
        rho * sa + wEnd * shrink * Math.cos(phi) * ca,
      ]);
      uvs.push(s / segs, 1 + m / C.capSteps);
    }
    rings.push(ring);
  }
  const base = at(0);
  const basePole = positions.length;
  positions.push([base.c[0] * ca, baseY + base.c[1], base.c[0] * sa]);
  uvs.push(0.5, 0);
  const indices = [];
  // Base fan facing down the tongue; then the ring quads and the tip fan.
  for (let s = 0; s < segs; s++) indices.push(basePole, rings[0][s + 1], rings[0][s]);
  for (let j = 0; j + 1 < rings.length; j++) {
    const lo = rings[j];
    const hi = rings[j + 1];
    if (hi.length === 1) {
      for (let s = 0; s < segs; s++) indices.push(hi[0], lo[s], lo[s + 1]);
    } else {
      for (let s = 0; s < segs; s++) {
        indices.push(lo[s], lo[s + 1], hi[s], lo[s + 1], hi[s + 1], hi[s]);
      }
    }
  }
  const key = (v) => `${Math.round(positions[v][0] * 1e5)},${Math.round(positions[v][1] * 1e5)},${Math.round(positions[v][2] * 1e5)}`;
  const acc = new Map();
  for (let q = 0; q < indices.length; q += 3) {
    const [i0, i1, i2] = [indices[q], indices[q + 1], indices[q + 2]];
    const p0 = positions[i0];
    const p1 = positions[i1];
    const p2 = positions[i2];
    const e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    const e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    for (const v of [i0, i1, i2]) {
      const kk = key(v);
      const cur = acc.get(kk) ?? [0, 0, 0];
      acc.set(kk, [cur[0] + n[0], cur[1] + n[1], cur[2] + n[2]]);
    }
  }
  const normals = new Float32Array(positions.length * 3);
  for (let v = 0; v < positions.length; v++) {
    const n = acc.get(key(v));
    const l = n ? Math.hypot(...n) : 0;
    if (l > 1e-12) normals.set([n[0] / l, n[1] / l, n[2] / l], 3 * v);
    else normals.set([0, 1, 0], 3 * v);
  }
  return geometry.withTangents({
    positions: new Float32Array(positions.flat()),
    normals,
    uvs: new Float32Array(uvs),
    indices: new Uint32Array(indices),
  });
}

function mudCrown(geometry, pad, i, t) {
  const C = MUD_CROWN;
  const [x, z, r] = pad;
  const phase = 0.7 + 1.9 * i;
  const surface = (lx, lz) => mudHeight(x + lx, z + lz, t);
  // Skirt: from C.skirt metres out (under the mud) up to C.rise on the wall, concave.
  const skirt = [];
  for (let j = 0; j <= 8; j++) {
    const s = j / 8;
    skirt.push([r - 0.01 + C.skirt * (1 - s), -0.03 + (C.rise + 0.03) * s ** 2.2]);
  }
  const ring = geometry.lathe(skirt, 72);
  const parts = [displaceY(geometry, ring, (lx, lz) => {
    const a = Math.atan2(lz, lx);
    const lift = Math.min(1, Math.max(0, (C.skirt + r - Math.hypot(lx, lz)) / C.skirt));
    return surface(lx, lz) + 0.018 * lift * Math.sin(5 * a + phase) * Math.cos(3 * a - phase);
  })];
  for (let k = 0; k < C.lobes; k++) parts.push(mudTongue(geometry, pad, i, k, t));
  const merged = geometry.merge(parts);
  delete merged.subMeshes;
  return merged;
}

// Deterministic hash in [0, 1).
function hash01(n) {
  const v = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return v - Math.floor(v);
}

// The white spray of pad i (x, z, radius r) at time t, pad-local XZ, world Y: per crown
// tongue MUD_DROPS.count thin streaks on ballistic arcs out of the tip, each an ellipsoid
// stretched along its velocity (normals by the inverse transpose of the stretch).
function mudSprayArcs(geometry, [x, z, r], i, t) {
  const D = MUD_DROPS;
  const unit = geometry.uvSphere(1, 12, 8);
  const parts = [];
  for (let k = 0; k < MUD_CROWN.lobes; k++) {
    const { a, h, flare } = mudLobe(i, k);
    const y0 = mudHeight(x + (r + 0.01) * Math.cos(a), z + (r + 0.01) * Math.sin(a), t);
    const rho = r + MUD_CROWN.foot + flare;
    const tip = [rho * Math.cos(a), y0 + 0.98 * h + 0.02, rho * Math.sin(a)];
    for (let j = 0; j < D.count; j++) {
      const n = 97 * i + 13 * k + j;
      const da = a + 0.5 * (hash01(n) - 0.5);
      const up = 0.9 + 0.7 * hash01(n + 0.3);
      const speed = D.speed[0] + (D.speed[1] - D.speed[0]) * hash01(n + 0.7);
      const l = Math.hypot(0.75, up);
      const v0 = [(0.75 * Math.cos(da) * speed) / l, (up * speed) / l, (0.75 * Math.sin(da) * speed) / l];
      const s = D.times[j % D.times.length];
      const pos = [tip[0] + v0[0] * s, tip[1] + v0[1] * s - 4.9 * s * s, tip[2] + v0[2] * s];
      const vel = [v0[0], v0[1] - 9.81 * s, v0[2]];
      const vl = Math.hypot(...vel);
      const u = vel.map((q) => q / vl);
      // An orthonormal frame (b1, u, b2) around the flight direction u.
      const ref = Math.abs(u[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
      let b1 = [u[1] * ref[2] - u[2] * ref[1], u[2] * ref[0] - u[0] * ref[2], u[0] * ref[1] - u[1] * ref[0]];
      const bl = Math.hypot(...b1);
      b1 = b1.map((q) => q / bl);
      const b2 = [u[1] * b1[2] - u[2] * b1[1], u[2] * b1[0] - u[0] * b1[2], u[0] * b1[1] - u[1] * b1[0]];
      // Later drops in the arc are smaller, as the sheet breaks into finer spray.
      const rad = (D.radius[1] - (D.radius[1] - D.radius[0]) * (j / Math.max(1, D.count - 1))) * (0.8 + 0.4 * hash01(n + 0.5));
      const len = rad * D.stretch;
      const p = new Float32Array(unit.positions.length);
      const nr = new Float32Array(unit.normals.length);
      for (let q = 0; q < p.length; q += 3) {
        const [lx, ly, lz] = [unit.positions[q], unit.positions[q + 1], unit.positions[q + 2]];
        const [nx, ny, nz] = [unit.normals[q] / rad, unit.normals[q + 1] / len, unit.normals[q + 2] / rad];
        for (let c = 0; c < 3; c++) {
          p[q + c] = pos[c] + b1[c] * lx * rad + u[c] * ly * len + b2[c] * lz * rad;
          nr[q + c] = b1[c] * nx + u[c] * ny + b2[c] * nz;
        }
        const m = Math.hypot(nr[q], nr[q + 1], nr[q + 2]);
        nr[q] /= m;
        nr[q + 1] /= m;
        nr[q + 2] /= m;
      }
      parts.push({ positions: p, normals: nr, uvs: unit.uvs, indices: unit.indices });
    }
  }
  const merged = geometry.merge(parts);
  delete merged.subMeshes;
  return merged;
}

// Mud set emitters at time t s: a cluster of white spray at the lip of every crown lobe
// (pad-local lobe tip, world position).
function mudEmitters(t) {
  const out = [];
  let seed = 61;
  MUD_PADS.forEach(([x, z, r], i) => {
    for (let k = 0; k < MUD_CROWN.lobes; k++) {
      const { a, h, flare } = mudLobe(i, k);
      const [c, sn] = [Math.cos(a), Math.sin(a)];
      const rho = r + MUD_CROWN.foot + flare;
      const y0 = mudHeight(x + (r + 0.01) * c, z + (r + 0.01) * sn, t);
      out.push({
        ...MUD_SPRAY,
        // The bulbous tip of the tongue (mudTongue: the centreline ends at 0.98 h).
        position: [x + rho * c, y0 + 0.98 * h + 0.02, z + rho * sn],
        direction: [0.7 * c, 1, 0.7 * sn],
        seed: seed++,
      });
    }
  });
  return out;
}

// The bank grid spans 8 m; its outermost cells (beyond 3.5 m, where the bank is flat) are
// stretched out to 40 m, so the grass runs to the horizon instead of ending in a square.
function bankWarp(v) {
  const a = Math.abs(v);
  return a <= 3.5 ? v : Math.sign(v) * (3.5 + (a - 3.5) * 73);
}

// Lathe profile of a pad: a low inflatable disc (height h above its base, radius r) with
// a slightly bulging wall and a raised inflatable tube rim (ref-3's pads): the rounded
// PAD_RIM shoulder runs over the top of the tube and down its inner side into a top
// PAD_RECESS lower, so the rim catches its own highlight ring. A disc with only a rounded
// shoulder read as a flat lime disc with no bevel (card 5077).
function mudPadProfile(r, h) {
  // A torus rim: its tube radius is PAD_RIM x the pad radius, so the rim stands about
  // 15..20% of the radius above the flat top (card 5078's pads were dished cups with a
  // shallow bevel). The flat top has its own +Y normal across the whole interior (the
  // repeated inner edge makes a hard crease in the lathe normals).
  const rr = Math.min(PAD_RIM * r, 0.55 * h);
  const wallTop = h - rr;
  const topY = wallTop + 0.008;
  // The underside reaches PAD_KEEL below the base, hidden in the mud on the pad's low side.
  const profile = [[0, -PAD_KEEL], [r - 0.02, -PAD_KEEL], [r, -PAD_KEEL + 0.02]];
  // The wall is split every 2 cm so the wet band follows the swell.
  for (let y = 0; y < wallTop; y += 0.02) profile.push([r + 0.012 * Math.sin((Math.PI * y) / wallTop), y]);
  profile.push([r, wallTop]);
  // The outer half of the torus: from its outer equator up and over its crown at h, down to
  // its inner equator at the top-face level.
  for (let k = 1; k <= 12; k++) {
    const a = (k / 12) * Math.PI;
    profile.push([r - rr + rr * Math.cos(a), wallTop + rr * Math.sin(a)]);
  }
  const rin = r - 2 * rr;
  // Flat top: a crease at the rim's inner edge, then planar rings to the centre.
  profile.push([rin, topY]);
  profile.push([rin, topY]);
  profile.push([0.6 * rin, topY + 0.001], [0.3 * rin, topY + 0.002], [0, topY + 0.003]);
  return profile;
}

// Pose of a floating pad at time t: the least-squares plane of the mud along its rim
// gives its tilt (a quaternion turning +y onto the plane normal) and its centre height;
// the base sits PAD_SINK below that plane. A level disc on the 0.25 m of swell across a
// pad was flooded on one side and stood clear on the other.
function mudPadPose([x, z, r], t) {
  let [mean, sx, sz] = [0, 0, 0];
  const n = 24;
  for (let k = 0; k < n; k++) {
    const a = (k / n) * Math.PI * 2;
    const y = mudHeight(x + r * Math.cos(a), z + r * Math.sin(a), t);
    mean += y / n;
    sx += (2 * y * Math.cos(a)) / (n * r);
    sz += (2 * y * Math.sin(a)) / (n * r);
  }
  const l = Math.hypot(sx, 1, sz);
  const [nx, ny, nz] = [-sx / l, 1 / l, -sz / l];
  const ql = Math.hypot(nz, nx, 1 + ny);
  return { base: mean - PAD_SINK, q: [nz / ql, 0, -nx / ql, (1 + ny) / ql] };
}

// v rotated by the unit quaternion q = [x, y, z, w].
function rotate([qx, qy, qz, qw], [vx, vy, vz]) {
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  return [
    vx + qw * tx + (qy * tz - qz * ty),
    vy + qw * ty + (qz * tx - qx * tz),
    vz + qw * tz + (qx * ty - qy * tx),
  ];
}

// Pad i's mesh in pad-local space (base at y 0, placed by its `pose`), with a wet band
// on its base: darker, mud-stained green up to MUD_PAD_WET.height above the local swell,
// fading softly into the dry pad (vertex colour times the glossy pad colour).
function mudPad(geometry, [x, z, r, h], pose, t) {
  const wet = (p) => {
    const w = rotate(pose.q, p);
    const above = w[1] + pose.base - mudHeight(x + w[0], z + w[2], t);
    const f = 1 - Math.min(1, Math.max(0, (above - 0.015) / (MUD_PAD_WET.height - 0.015)));
    const k = f * f * (3 - 2 * f);
    return [0, 1, 2].map((i) => 1 + (MUD_PAD_WET.tint[i] - 1) * k).concat(1);
  };
  return geometry.lathe(mudPadProfile(r, h), 72, { color: wet });
}

function buildMud(engine, lib, settings, variant) {
  const { geometry } = lib;
  const t = (variant?.timeMs ?? 1000) / 1000;
  daylight(engine);
  engine.setLight(MUD_LIGHT.sun);
  engine.setEnvironmentLighting(MUD_LIGHT.env);
  // The bank: lime grass right down to the waterline (no dark mud-stained skirt: card
  // 5078's judge read the stain ring as a brown apron around the pit).
  engine.addMaterialFunction('demo.grass', GRASS_WGSL);
  const ground = engine.addSurfaceMaterial({ function: 'demo.grass', color: [1, 1, 1], roughness: 0.8 });
  const grass = [0.17, 0.44, 0.03];
  const rim = heightfield(geometry, 8, 8, 96, mudGround, () => [grass[0], grass[1], grass[2], 1], bankWarp);
  place(engine, upload(engine, rim, ground), [0, 0, 0]);
  const padMat = engine.addSurfaceMaterial(MUD_PAD);
  engine.addMaterialFunction('demo.crown', CROWN_WGSL);
  const crownMat = engine.addSurfaceMaterial({
    function: 'demo.crown',
    color: MUD_CROWN.color,
    roughness: MUD_CROWN.roughness,
    clearcoat: MUD_CROWN.clearcoat,
    clearcoatRoughness: MUD_CROWN.clearcoatRoughness,
  });
  const dropMat = engine.addSurfaceMaterial({
    color: MUD_DROPS.color,
    roughness: MUD_DROPS.roughness,
    clearcoat: MUD_DROPS.clearcoat,
    clearcoatRoughness: MUD_DROPS.clearcoatRoughness,
    emissive: MUD_DROPS.emissive,
  });
  // A few popping bubbles on the open mud: dark glossy domes that follow the swell, plus
  // the ring a popped one leaves. The shader's sparse voronoi field alone was invisible at
  // capture distance (card 5078 judge).
  const bubbleMat = engine.addSurfaceMaterial({ color: [0.24, 0.12, 0.06], roughness: 0.04, clearcoat: 1, clearcoatRoughness: 0.02 });
  for (const [bx, bz, br] of MUD_BUBBLES) {
    const y = mudHeight(bx, bz, t);
    place(engine, upload(engine, geometry.uvSphere(1, 32, 16), bubbleMat), [bx, y - 0.005, bz], [br, br * 0.38, br]);
  }
  const ringPath = Array.from({ length: 32 }, (_, j) => {
    const a = (j / 32) * Math.PI * 2;
    return [MUD_BUBBLE_RING.x + MUD_BUBBLE_RING.radius * Math.cos(a), 0, MUD_BUBBLE_RING.z + MUD_BUBBLE_RING.radius * Math.sin(a)];
  });
  place(engine, upload(engine, geometry.sweepTube(ringPath, 0.012, 8, { closed: true }), bubbleMat),
    [0, mudHeight(MUD_BUBBLE_RING.x, MUD_BUBBLE_RING.z, t) + 0.001, 0]);
  MUD_PADS.forEach(([x, z, r, h], i) => {
    const pose = mudPadPose([x, z, r], t);
    place(engine, upload(engine, mudPad(geometry, [x, z, r, h], pose, t), padMat), [x, pose.base, z], [1, 1, 1], pose.q);
    place(engine, upload(engine, mudCrown(geometry, [x, z, r], i, t), crownMat), [x, 0, z]);
    place(engine, upload(engine, mudSprayArcs(geometry, [x, z, r], i, t), dropMat), [x, 0, z]);
  });
  for (const e of mudEmitters(t)) engine.addEmitter(e);
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

// Exposed for tests (examples/lookdev/test/): the mud set's emitters and crown mesh.
export const mudSet = {
  emitters: mudEmitters, crown: mudCrown, tongue: mudTongue, lobe: mudLobe, height: mudHeight,
  crownStyle: MUD_CROWN, pads: MUD_PADS, bubbles: MUD_BUBBLES, bubbleRing: MUD_BUBBLE_RING,
  pad: mudPad, padPose: mudPadPose, rotate, padProfile: mudPadProfile, spray: mudSprayArcs,
  drops: MUD_DROPS, ground: mudGround, material: MUD, grass: GRASS_WGSL,
};

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
      + 'glossy sheen with sun highlights and a faint reflection of the sky, a few small '
      + 'popping bubbles, a band of white froth along its shore and around the pads, and '
      + 'crowns of tall glossy chocolate mud splash sheets with thin white spray droplets '
      + 'arcing from their tips around every pad; low, wide, round glossy lime-green '
      + 'inflatable disc pads with a raised rounded rim float in the mud, and a grassy '
      + 'lime-green bank rises around the pit. Mud is opaque, so it shows no refraction of '
      + 'what lies below; its swell reads as light crests and darker troughs',
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
