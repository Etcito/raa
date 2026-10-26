// Lookdev scene m14-camera-effects (chain milestone 8, R14): an obstacle-course-like vista
// with every P2 feature, laid out along a long straight run seen from a low camera so depth
// of field, height fog and aerial perspective read with distance: a glossy tiled run
// (SSR), clear-coated inflatables (hammer, pads, an inflatable arch), ice blocks
// (transmission), a plush sphere (fur), neon tube arches (bloom), a liquid strip beside the
// run (raa.liquid.*), particles, a pillar gate between the camera and a low sun (sun
// shafts through the gaps) and big shapes 80-200 m away (aerial perspective).
// Four worlds, each through engine.applyEnvironment (sky, fog, post presets) with its own
// sun, palette, liquid and particles: one WORLDS entry per world, so tuning edits one entry.
// Shadows, AO and contact shadows stay at their defaults; DOF autofocuses on the hero
// hammer at f/2.8 through a 75 mm lens. The judged candy world raises SSR and the IBL
// specular so its clear-coated inflatables and tiles mirror the sky and the neon arches.
// Pure module (S14): no top-level side effects, node reads id/milestone/variants.

const COAT = { clearcoat: 1, clearcoatRoughness: 0.08 };
const FLOOR_ROUGHNESS = 0.08;
// Low camera at the start of the run looking down it (-Z); captures are 1280 x 720.
const CAMERA = { eye: [0, 0.65, 4.2], target: [0, 1.0, -24], fovYDeg: 50, near: 0.1, far: 400 };
const ASPECT = 16 / 9;
// Hero obstacle: the inflatable hammer, right of the run; DOF autofocus looks at its head.
const HERO = { t: [0.95, 0, -2.6], yaw: -0.5, scale: 1.4, focus: [0.95, 1.45, -2.6] };
const F_STOP = 4;
// The run: floorTiles(12) sections (6 x 6 m) on a plinth, from z = 4 to z = -26.
const RUN = { tiles: 12, sections: [1, -5, -11, -17, -23], plinth: [3.05, 0.3, 15], plinthZ: -11 };
const GROUND_Y = -0.6;
// Pillar gate across the far end, tall enough to stand in front of a sun 12 deg up.
const GATE = { z: -27, x0: -5.5, x1: 5.5, pitch: 1.0, half: [0.24, 4.2, 0.3], lintelY: 8.6 };
// Liquid strip beside the run (grid w x d at y, over a basin floor at GROUND_Y).
const STRIP = { x: 4.7, z: -9, w: 3, d: 18, y: -0.18, nx: 48, nz: 288 };
const PADS = [[-1.6, -6.5], [1.5, -9.5], [-1.2, -13], [1.7, -17.5]];
const NEON_ARCHES = [-0.5, -7.5, -14.5]; // z of each neon arch over the run
const INFLATABLE_ARCH_Z = -11;
const ICE_BLOCKS = [[-2.2, -3.6, 3], [-2.5, -4.5, 7], [2.3, -12.2, 11]];
const PLUSH = [-1.0, 0, -1.4];
// Distant shapes: [x, z, kind, size m]; kind 'peak' (lathe cone), 'dome' or 'tower'.
const VISTA = [
  [-55, -95, 'peak', 34], [40, -120, 'dome', 28], [-15, -160, 'tower', 22],
  [75, -180, 'peak', 50], [-95, -200, 'dome', 45], [20, -85, 'tower', 14],
];

// One entry per world. Colours sRGB; `sun.dir` is the travel direction.
const WORLDS = {
  candy: {
    // Keep blue visible near the low camera's horizon; warm light comes from the sun.
    sky: { preset: 'day', zenith: [0.055, 0.28, 0.85], horizon: [0.4, 0.65, 0.95],
      ground: [0.32, 0.35, 0.4], intensity: 1.15, horizonSoftness: 12,
      sunDiscSize: 0.012, sunDiscIntensity: 18,
      clouds: { coverage: 0.28, density: 0.8 } },
    fog: 'haze', post: 'candy',
    // Low, behind the gate: the sun travels towards the camera, 12 deg up, so from the
    // camera it sits behind the pillars (about 7.3 m up at the gate), below the lintel.
    sun: { dir: [0.12, -0.21, 0.97], color: [1, 0.95, 0.86], intensity: 4.5 },
    tiles: [[1, 0.08, 0.55], [0.04, 0.55, 1], [1, 0.77, 0.06], [0.1, 0.65, 1]],
    plinth: [1, 0.16, 0.58], ground: [0.25, 0.72, 0.3], vista: [0.72, 0.62, 0.95],
    inflatable: [[1, 0.035, 0.48], [1, 0.75, 0.025], [0.04, 0.38, 1]], stripe: [1, 1, 1],
    neon: [1, 0.3, 0.75], plush: [0.2, 0.3, 0.95], gate: [1, 0.85, 0.3],
    ice: { tint: [0.6, 0.9, 1], absorption: 1.8, cloudiness: 0.2, crackScale: 0.25 },
    liquid: { function: 'raa.liquid.slime', params: { color: [0.12, 1, 0.025] }, transmissive: false },
    particles: { kind: 'confetti', position: [0, 3.5, -7], volume: [2.5, 0.5, 4], direction: [0, 1, 0], spread: 0.6, rate: 300, speed: [1.5, 3], groundY: 0, colors: [[1, 0.15, 0.45], [1, 0.8, 0.1], [0.2, 0.55, 1], [0.3, 0.9, 0.3]] },
  },
  ice: {
    // Clear winter light rather than the snow sky's flat overcast illumination.
    icy: true,
    polished: true,
    // Saturated winter blue at the top (#2f7fe0) fading to the pale cyan-white horizon
    // of ref-2 (#bfe4ff); a white zenith or horizon reads as a washed-out sheet.
    sky: { preset: 'day', zenith: [0.028, 0.212, 0.745], horizon: [0.52, 0.78, 1],
      ground: [0.5, 0.7, 0.92], intensity: 0.95, horizonSoftness: 8,
      sunDiscSize: 0.012, sunDiscIntensity: 22,
      clouds: { coverage: 0.14, density: 0.5, color: [0.95, 0.98, 1] } },
    // Cold blue daylight grade: push saturation and keep only snow and the tube edges
    // above AgX's white shoulder. The old custom neutral grade left the frame pale.
    fog: 'snow', post: { grading: { preset: 'ice', contrast: 1.1,
      saturation: 1.32, vibrance: 0.22 }, bloom: { strength: 0.03, radius: 0.62 },
      toneMapping: { exposureEv: -0.1 } },
    // Low backlight through the gate catches the ice edges and falling snow.
    sun: { dir: [0.12, -0.21, 0.97], color: [0.94, 0.97, 1], intensity: 4.5 },
    tiles: [[0.02, 0.55, 1], [0.06, 0.75, 1], [0.02, 0.42, 0.95], [0.2, 0.82, 1]],
    plinth: [0.2, 0.62, 0.95], ground: [0.94, 0.97, 1], vista: [0.72, 0.83, 0.98],
    inflatable: [[0.025, 0.45, 0.98], [0.95, 0.06, 0.48], [0.8, 0.94, 1]], stripe: [1, 1, 1],
    neon: [0.1, 0.6, 1], plush: [0.06, 0.22, 0.85], gate: [0.16, 0.52, 0.92],
    ice: { tint: [0.1, 0.62, 1], absorption: 1.0, cloudiness: 0.02, crackScale: 0.4 },
    liquid: { function: 'raa.liquid.water', params: {}, transmissive: true },
    // Camera-facing white flakes across the hammer's focus plane, not just the sky.
    particles: { kind: 'snow', position: [0, 2.0, -2.6], volume: [3, 1.5, 3],
      direction: [0.08, -1, 0], spread: 0.25, gravity: 0.35, drag: 0.5,
      rate: 1800, lifetime: [6, 10], size: [0.055, 0.11], speed: [0.45, 0.85],
      emissive: 2, colors: [[1, 1, 1]] },
  },
  jungle: {
    jungle: true,
    polished: true,
    sky: { preset: 'day', zenith: [0.025, 0.35, 0.9], horizon: [0.1, 0.6, 0.9],
      ground: [0.2, 0.3, 0.16], intensity: 0.75, horizonSoftness: 10,
      sunDiscSize: 0.012, sunDiscIntensity: 8,
      clouds: { coverage: 0.18, density: 0.55, color: [0.9, 0.96, 1] } },
    fog: 'jungle',
    // Preserve the blue sky, brown mud and pink accents instead of a yellow wash.
    post: { grading: { preset: 'neutral', contrast: 1.08, saturation: 1.15,
      vibrance: 0.12 }, bloom: { strength: 0.004, radius: 0.45 },
      toneMapping: { exposureEv: -0.35 } },
    // Low, warm, behind the gate like candy, a little from the left.
    sun: { dir: [-0.15, -0.22, 0.96], color: [1, 0.78, 0.43], intensity: 3.2 },
    tiles: [[0.45, 0.95, 0.015], [1, 0.82, 0.015]],
    plinth: [0.42, 0.26, 0.14], ground: [0.3, 0.2, 0.1], vista: [0.12, 0.38, 0.16],
    inflatable: [[0.6, 0.95, 0.08], [1, 0.85, 0.08], [0.95, 0.3, 0.15]], stripe: [1, 0.95, 0.3],
    neon: [1, 0.7, 0.2], plush: [0.2, 0.3, 0.9], gate: [0.12, 0.46, 0.25],
    ice: { tint: [0.6, 0.95, 0.85], absorption: 1.5, cloudiness: 0.35, crackScale: 0.25 },
    liquid: { function: 'raa.liquid.mud', params: { color: [0.34, 0.17, 0.1] }, transmissive: false },
    particles: { kind: 'splash', position: [4.7, -0.15, -5], volume: [1, 0, 4], direction: [0, 1, 0], rate: 160, colors: [[0.4, 0.22, 0.12]], opacity: 1 },
  },
  neon: {
    night: true,
    polished: true,
    // Sky/Fog take LINEAR radiance. These sit about four times below the old #1a0f3a /
    // #3a1d6e decode: AgX lifts a linear 0.16 horizon into a pale lavender sheet, so a
    // dark violet night needs a low linear horizon. build() boosts only the IBL, which
    // keeps the near surfaces readable without brightening the background.
    sky: { preset: 'night', zenith: [0.018, 0.006, 0.028], horizon: [0.05, 0.012, 0.078],
      ground: [0.004, 0.0015, 0.006], intensity: 1, horizonSoftness: 5,
      sunDiscSize: 0.008, sunDiscIntensity: 0,
      clouds: { coverage: 0, density: 0 } },
    fog: 'night',
    // Saturated magenta/cyan night grade: lower exposure and a darker floor keep the tube
    // chroma inside AgX instead of bleaching it to white. Bloom is threshold-free here, so
    // its strength is the only threshold lever there is.
    post: { grading: { preset: 'neutral', contrast: 1.18, saturation: 1.32,
      vibrance: 0.2, lift: [0, 0, 0], shadowsTint: [0.5, 0.5, 0.5], vignette: 0.18 },
      bloom: { strength: 0.32, radius: 0.78 },
      // AgX's punchy look retains chroma in the luminous strips and their bloom.
      toneMapping: { exposureEv: -0.55, look: 'punchy' } },
    // A dim cool moon behind the gate: a violet rim on the pillars and the plush, and just
    // enough sun for faint violet shafts. The tubes are the actual light source.
    sun: { dir: [0.12, -0.21, 0.97], color: [0.45, 0.5, 1], intensity: 0.7 },
    tiles: [[0.055, 0.012, 0.13], [0.085, 0.014, 0.16], [0.012, 0.05, 0.13]],
    plinth: [0.045, 0.012, 0.10], ground: [0.02, 0.008, 0.045], vista: [0.07, 0.02, 0.16],
    inflatable: [[1, 0.035, 0.58], [0.025, 0.75, 1], [1, 0.65, 0.045]], stripe: [0.12, 0.04, 0.28],
    neon: [1, 0.08, 0.66], neon2: [0.04, 0.78, 1],
    plush: [0.2, 0.38, 1], gate: [0.055, 0.015, 0.12],
    ice: { tint: [0.3, 0.7, 1], absorption: 0.7, cloudiness: 0.04, crackScale: 0.4 },
    liquid: { function: 'raa.liquid.slime', params: { color: [0.04, 0.5, 1],
      glossiness: 1, absorption: 0.7, foamAmount: 0.03, detailStrength: 0.04 }, transmissive: true },
    particles: { kind: 'sparkles', position: [0, 2.6, -8], volume: [4, 2.2, 9],
      rate: 450, size: [0.03, 0.07], emissive: 7,
      colors: [[1, 0.04, 0.55], [0.04, 0.8, 1], [1, 0.65, 0.12]] },
  },
};

// Only the judged candy world uses this finish; diagnostic pairs keep their geometry.
const CANDY_LOOK = {
  ...WORLDS.candy,
  polished: true,
  candy: true,
  sky: { ...WORLDS.candy.sky, zenith: [0.055, 0.32, 0.9], horizon: [0.36, 0.65, 0.94],
    ground: [0.4, 0.38, 0.3], intensity: 1.1,
    sunDiscSize: 0.012, sunDiscIntensity: 20,
    clouds: { coverage: 0.32, density: 0.7 } },
  // The low sun sits in a gap of the gate (dir.x 0.077 puts it at x -2 m at the gate),
  // warm and bright enough to light the glossy inflatables and glow through the haze.
  sun: { dir: [0.077, -0.21, 0.97], color: [1, 0.9, 0.72], intensity: 6.5 },
  // Saturated pastel shapes: the vista reads mauve-pink in the pale blue aerial haze,
  // instead of the beige placeholders a warm grey turned into.
  vista: [0.8, 0.6, 0.96], ground: [0.32, 0.72, 0.35],
  plinth: [1, 0.08, 0.45], gate: [1, 0.72, 0.08],
  tiles: [[1, 0.045, 0.48], [0.025, 0.48, 1], [1, 0.72, 0.04], [0.04, 0.65, 1]],
  inflatable: [[1, 0.045, 0.48], [1, 0.76, 0.035], [0.025, 0.46, 1]],
  liquid: { function: 'raa.liquid.slime',
    params: { color: [0.12, 1, 0.025], glossiness: 1, absorption: 0.8,
      foamAmount: 0.08, detailStrength: 0.055 }, transmissive: true },
  particles: { ...WORLDS.candy.particles, position: [0, 2.8, -5],
    volume: [2.8, 0.6, 5], rate: 450, size: [0.045, 0.09],
    colors: [[1, 0.08, 0.48], [1, 0.78, 0.04], [0.05, 0.55, 1], [0.3, 1, 0.1]] },
};
const CANDY_COAT = { clearcoat: 1, clearcoatRoughness: 0.035 };
// The judged candy world's sharper coat: the reference's inflatables carry crisp white
// sky streaks. Keep CANDY_COAT for the already-accepted icy/jungle worlds.
const CANDY_GLOSS = { clearcoat: 1, clearcoatRoughness: 0.05 };

// A dense rounded cylinder with analytic profile normals, without pinched panel seams.
// Keep the shared hammer's dimensions, axis and placement (the capture contract).
function candyHammer(geometry, palette, headScale = 1) {
  const profile = (radius, half, rounding) => {
    const points = [[0, -half]];
    for (let i = 0; i <= 24; i++) {
      const a = -Math.PI / 2 + Math.PI / 2 * i / 24;
      points.push([radius - rounding + rounding * Math.cos(a),
        -half + rounding + rounding * Math.sin(a)]);
    }
    for (let i = 0; i <= 24; i++) {
      const a = Math.PI / 2 * i / 24;
      points.push([radius - rounding + rounding * Math.cos(a),
        half - rounding + rounding * Math.sin(a)]);
    }
    return [...points, [0, half]];
  };
  const color = (c) => [...c.map(srgbToLinear), 1];
  const handle = geometry.lathe(profile(0.065, 0.4875, 0.06), 64,
    { color: () => color(palette[2]) });
  for (let i = 1; i < handle.positions.length; i += 3) handle.positions[i] += 0.4875;
  const head = geometry.lathe(profile(0.19 * headScale, 0.36, 0.075 * headScale), 128, { color: (p) =>
    color(Math.abs(p[1]) > 0.285 ? palette[1]
      : Math.cos(6 * Math.atan2(-p[2], p[0])) > 0.5 ? [1, 1, 1] : palette[0]) });
  // Rotate the head Y -> X, including its normal and tangent frame.
  for (const [key, stride] of [['positions', 3], ['normals', 3], ['tangents', 4]]) {
    for (let i = 0; i < head[key].length; i += stride) {
      const x = head[key][i], y = head[key][i + 1];
      head[key][i] = y;
      head[key][i + 1] = -x + (key === 'positions' ? 1.08 : 0);
    }
  }
  return geometry.merge([handle, head]);
}

const identityQuat = [0, 0, 0, 1];
const yawQuat = (a) => [0, Math.sin(a / 2), 0, Math.cos(a / 2)];
const srgbToLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const sub = (a, b) => a.map((x, i) => x - b[i]);
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a) => a.map((x) => x / Math.hypot(...a));

// Screen [u, v] (0..1, v down) of world point p seen by CAMERA: the DOF autofocus point.
function screenUv(p, cam = CAMERA) {
  const fwd = unit(sub(cam.target, cam.eye));
  const right = unit(cross(fwd, [0, 1, 0]));
  const up = cross(right, fwd);
  const d = sub(p, cam.eye);
  const t = Math.tan((cam.fovYDeg * Math.PI) / 360);
  const z = dot(d, fwd);
  return [0.5 + dot(d, right) / z / (t * ASPECT) / 2, 0.5 - dot(d, up) / z / t / 2];
}

function place(engine, mesh, t, rotation = identityQuat, scale = [1, 1, 1]) {
  const node = engine.addNode(null, t, rotation, scale);
  return engine.addMeshInstance(node, mesh);
}

// Every sub-mesh of `desc` gets materials[i] (the last one repeats).
function upload(engine, desc, materials) {
  const subMeshes = (desc.subMeshes ?? [{ first: 0, count: desc.indices.length }]).map((s, i) => ({
    ...s,
    material: materials[Math.min(i, materials.length - 1)],
  }));
  return engine.addMesh({ ...desc, subMeshes });
}

const plain = (engine, color, roughness, extra = {}) => engine.addSurfaceMaterial({ color, roughness, ...extra });
const inflatable = (engine, color, stripeColor) =>
  engine.addSurfaceMaterial({ function: 'raa.inflatable', params: { color, stripeColor, seamDepth: 0.2 }, ...COAT });
const neon = (engine, color) =>
  engine.addSurfaceMaterial({ function: 'raa.neon', params: { color, strength: 5, coreWhiteness: 1 }, castShadow: false });

// lib's floor tiles recoloured with the world palette (one linear colour per tile, +-10%).
function tiles(objects, n, palette, alternate = false) {
  const desc = objects.floorTiles(n);
  const perTile = desc.positions.length / 3 / (n * n);
  const colors = new Float32Array(desc.positions.length / 3 * 4);
  for (let t = 0; t < n * n; t++) {
    const h = (Math.imul(t + 1, 2654435761) >>> 0) / 4294967296;
    const base = palette[alternate ? (t + Math.floor(t / n)) % palette.length : Math.floor(h * palette.length)];
    const k = 0.9 + 0.2 * ((Math.imul(t + 7, 40503) >>> 0) % 1000) / 1000;
    for (let v = t * perTile; v < (t + 1) * perTile; v++) {
      for (let c = 0; c < 3; c++) colors[4 * v + c] = srgbToLinear(base[c]) * k;
      colors[4 * v + 3] = 1;
    }
  }
  return { ...desc, colors };
}

// An arch path in the XY plane: legs `leg` m high, then a half circle of radius r.
function archPath(r, leg, steps = 48) {
  const pts = [[r, 0, 0], [r, leg, 0]];
  for (let i = 1; i < steps; i++) {
    const a = (Math.PI * i) / steps;
    pts.push([r * Math.cos(a), leg + r * Math.sin(a), 0]);
  }
  return [...pts, [-r, leg, 0], [-r, 0, 0]];
}

// Ref-4 uses broad glossy stepping tiles and substantial luminous pipe machinery.
// Keep this geometry exclusive to neon; the accepted worlds and diagnostic pairs
// retain the shared obstacle course.
// Keep coloured emitters below AgX's whitening shoulder. Bloom has no threshold,
// so these still glow; raising emission to 5–6 only bleaches their display colour.
function nightLight(engine, color, strength = 2) {
  return plain(engine, color, 0.15, { emissive: color.map(c => srgbToLinear(c) * strength),
    clearcoat: 1, castShadow: false });
}

function buildNeonRun(engine, { geometry }, w) {
  // Darker than the tubes so the reflection reads, and glossy enough for a sharp SSR hit.
  const floor = plain(engine, [0.055, 0.010, 0.13], 0.045,
    { ...COAT, clearcoatRoughness: 0.02, metallic: 0.2 });
  place(engine, upload(engine, geometry.roundedBox([4.2, 0.15, 24], 0.12, 4), [floor]),
    [0, -0.35, -19]);
  const tile = geometry.roundedBox([0.93, 0.13, 1.3], 0.12, 8);
  const faces = w.tiles.map(c => upload(engine, tile,
    [plain(engine, c, 0.04, { ...COAT, clearcoatRoughness: 0.02, metallic: 0.15 })]));
  const lights = [w.neon, w.neon2].map(c => nightLight(engine, c));
  const bars = lights.map(m => upload(engine,
    geometry.roundedBox([0.72, 0.026, 0.05], 0.024, 4), [m]));
  // Twin chevrons, in the same luminous material as each tile's front edge.
  const arrow = geometry.sweepTube([[-0.3, 0, 0.15], [0, 0, -0.08], [0.3, 0, 0.15]],
    0.03, 8, { caps: 'round' });
  const arrows = lights.map(m => upload(engine, arrow, [m]));
  for (let row = 0; row < 14; row++) for (let col = 0; col < 3; col++) {
    const x = (col - 1) * 2, z = 0.8 - row * 3;
    const hue = (row + col) % 2;
    place(engine, faces[(row + col) % faces.length], [x, -0.13, z]);
    place(engine, bars[hue], [x, 0.022, z + 1.1]);
    for (const dz of [-0.2, 0.2]) place(engine, arrows[hue], [x, 0.025, z + dz]);
  }
}

function ringPath(radius, steps = 72) {
  const pts = [];
  for (let i = 0; i <= steps; i++) {
    const a = (Math.PI * 2 * i) / steps;
    pts.push([radius * Math.cos(a), radius * Math.sin(a), 0]);
  }
  return pts;
}

function buildNeonArchitecture(engine, lib, w) {
  const { geometry } = lib;
  const violet = plain(engine, [0.24, 0.055, 0.45], 0.12,
    { ...COAT, metallic: 0.6 });
  const metal = plain(engine, [0.45, 0.25, 0.09], 0.18, { ...COAT, metallic: 0.7 });
  const lights = [w.neon, w.neon2].map(c => nightLight(engine, c));
  // Gas tubes carry a hot core and a far halo; the flat light strips above stay below
  // AgX's bleaching shoulder on purpose, so the tubes are the parts that really blow out.
  const gas = [w.neon, w.neon2].map(c => engine.addSurfaceMaterial({
    function: 'raa.neon', params: { color: c, strength: 4, coreWhiteness: 0.14 },
    castShadow: false,
  }));
  const tubeMesh = (path, radius, material) => upload(engine,
    geometry.sweepTube(path, radius, 24, { caps: 'round' }), [material]);
  const body = upload(engine, geometry.lathe([[0, 0], [0.34, 0], [0.34, 3.5], [0, 3.5]], 64), [violet]);
  const collar = upload(engine, geometry.lathe([[0, -0.1], [0.44, -0.1], [0.44, 0.1], [0, 0.1]], 64), [metal]);
  const band = upload(engine, geometry.lathe([[0, -0.035], [0.45, -0.035], [0.45, 0.035], [0, 0.035]], 64), [lights[1]]);
  // Split each cylinder quad into one checker material, avoiding interpolated
  // vertex colours which would turn the reference's checks into pastel gradients.
  const profile = [[0, 0], ...Array.from({ length: 9 }, (_, i) => [0.345, i * 0.4]), [0, 3.2]];
  const checks = geometry.lathe(profile, 64);
  const buckets = [[], []];
  for (let i = 0; i < checks.indices.length; i += 3) {
    const ids = Array.from(checks.indices.slice(i, i + 3));
    const p = [0, 1, 2].map(c => ids.reduce((sum, v) => sum + checks.positions[3 * v + c], 0) / 3);
    const sector = Math.floor((Math.atan2(p[2], p[0]) + Math.PI) / (Math.PI / 4));
    buckets[(sector + Math.floor(p[1] / 0.4)) % 2].push(...ids);
  }
  checks.indices = new Uint32Array([...buckets[0], ...buckets[1]]);
  checks.subMeshes = [{ first: 0, count: buckets[0].length },
    { first: buckets[0].length, count: buckets[1].length }];
  const checker = upload(engine, checks, [nightLight(engine, w.neon, 0.65),
    nightLight(engine, [0.55, 0.8, 1], 0.8)]);
  // Vertical gas strips edge every pillar so it is not a flat silhouette.
  const rimPath = [[0, 0, 0], [0, 3.4, 0]];
  const rims = gas.map(m => upload(engine, geometry.sweepTube(rimPath, 0.028, 12, { caps: 'round' }), [m]));
  const pipeHousing = tubeMesh([[0, 0, 0], [0, 0, -7.4]], 0.18, violet);
  const pipeMagenta = tubeMesh([[0, 0, 0], [0, 0, -7.4]], 0.055, gas[0]);
  const pipeCyan = tubeMesh([[0, 0, 0], [0, 0, -7.4]], 0.055, gas[1]);
  const riser = tubeMesh([[0, 0, 0], [0, 2.5, 0]], 0.035, gas[1]);
  for (const z of [-3, -11, -20, -30, -43, -58]) for (const side of [-1, 1]) {
    const x = side * 3.6;
    place(engine, (z === -3 || z === -20 || z === -43) ? checker : body, [x, 0, z]);
    for (const y of [0.15, 3.2]) {
      place(engine, collar, [x, y, z]);
      place(engine, band, [x, y + 0.12, z]);
    }
    for (const dx of [-0.31, 0.31]) {
      place(engine, rims[(dx < 0) === (side < 0) ? 0 : 1], [x + dx, 0.18, z + 0.39]);
    }
    // Thick violet pipe housings carry magenta and cyan strips towards the gate.
    place(engine, pipeHousing, [x, 2.5, z]);
    place(engine, side < 0 ? pipeMagenta : pipeCyan, [x - side * 0.17, 2.52, z]);
    place(engine, riser, [x, 0.4, z + 0.35]);
  }
  // Overhead magenta pipe, with a dark glossy housing and visible luminous face.
  const overheadHousing = tubeMesh([[-3.6, 0, 0], [3.6, 0, 0]], 0.22, violet);
  const overheadStrip = tubeMesh([[-3.6, 0, 0], [3.6, 0, 0]], 0.07, gas[0]);
  for (const z of [-5.5, -19, -33]) {
    place(engine, overheadHousing, [0, 3.5, z]);
    place(engine, overheadStrip, [0, 3.48, z + 0.2]);
  }
  const archPathMesh = geometry.sweepTube(archPath(2.7, 1.2), 0.09, 24, { caps: 'round' });
  const arches = gas.map(m => upload(engine, archPathMesh, [m]));
  for (const [i, z] of [-11, -24, -39].entries()) place(engine, arches[i % 2], [0, 0, z]);
  // A far double portal closes the run and gives the distant haze a luminous anchor.
  place(engine, upload(engine, geometry.sweepTube(ringPath(6.4), 0.1, 16, { caps: 'round' }), [gas[1]]),
    [0, 7.2, -50]);
  place(engine, upload(engine, geometry.sweepTube(ringPath(4.8), 0.08, 16, { caps: 'round' }), [gas[0]]),
    [0, 7.2, -50.4]);
  // Dark far towers with a vertical gas edge: readable depth instead of flat silhouettes.
  const tower = upload(engine, geometry.roundedBox([1, 1, 1], 0.08, 3), [violet]);
  const towerStrips = gas.map(m => upload(engine,
    geometry.roundedBox([0.05, 1, 0.05], 0.018, 2), [m]));
  const SKYLINE = [[-34, -96, 4, 13], [30, -104, 5, 10], [-18, -122, 6, 17],
    [18, -92, 3, 8], [46, -132, 7, 19], [-54, -118, 5, 12]];
  for (const [i, [x, z, half, tall]] of SKYLINE.entries()) {
    place(engine, tower, [x, tall - 0.4, z], identityQuat, [half, tall, half]);
    place(engine, towerStrips[i % 2], [x, tall - 0.4, z + half + 0.02],
      identityQuat, [1, tall, 1]);
  }
  const plush = engine.addSurfaceMaterial({
    function: 'raa.plush',
    params: { color: w.plush, tipColor: [0.55, 0.8, 1], sheenColor: [0.6, 0.85, 1],
      furLength: 0.12, furDensity: 8 },
    furLength: 0.12, furDensity: 8, furDroop: 0.3, furShells: 64, roughness: 0.95,
  });
  // Same 0.45 m sphere as objects.plushSphere, resting on the tile tops at y = 0.
  place(engine, upload(engine, geometry.uvSphere(0.45, 64, 32), [plush]), [PLUSH[0], 0.45, PLUSH[2]]);
  // A gas ring on the floor underlines the furry silhouette and lights the tufts.
  const plushRing = Array.from({ length: 49 }, (_, i) => {
    const a = (Math.PI * 2 * i) / 48;
    return [0.56 * Math.cos(a), 0, 0.56 * Math.sin(a)];
  });
  place(engine, upload(engine, geometry.sweepTube(plushRing, 0.018, 8, { caps: 'round' }), [gas[0]]),
    [PLUSH[0], 0.03, PLUSH[2]]);
  const star = upload(engine, geometry.uvSphere(1, 8, 4),
    [nightLight(engine, [0.35, 0.65, 1], 3)]);
  // A deterministic two-dimensional distribution, above and between distant pipes.
  const hash = n => ((Math.imul(n ^ (n >>> 16), 2246822519) >>> 0) % 10007) / 10007;
  for (let i = 0; i < 180; i++) {
    const r = 0.05 + hash(i * 19 + 91) * 0.09;
    place(engine, star, [(hash(i * 3 + 1) - 0.5) * 140, 5 + hash(i * 7 + 43) * 42, -110],
      identityQuat, [r, r, r]);
  }
}

function buildRun(engine, lib, w) {
  if (w.night) return buildNeonRun(engine, lib, w);
  const { geometry, objects } = lib;
  const ground = plain(engine, w.ground, 0.9);
  place(engine, upload(engine, geometry.roundedBox([260, 0.5, 260], 0.2, 1), [ground]), [0, GROUND_Y - 0.5, -120]);
  const plinth = plain(engine, w.plinth, w.candy ? 0.18 : 0.3, w.candy ? CANDY_GLOSS : COAT);
  place(engine, upload(engine, geometry.roundedBox(RUN.plinth, 0.1, 2), [plinth]), [0, -0.33, RUN.plinthZ]);
  // Candy's wet-look tiles: ior 2.0 lifts the dielectric F0 from 0.04 to 0.11, so SSR's
  // obstacle reflections clear the faint 4 % Fresnel instead of washing out.
  // Ice's tile run is a wet mirror: a fairly metallic, low-roughness base under a sharp
  // clearcoat, so SSR carries the hammer, plush and tube reflections into the blue tiles.
  const floor = plain(engine, [1, 1, 1], w.icy ? 0.18 : w.night ? 0.12 : FLOOR_ROUGHNESS,
    w.icy ? { clearcoat: 1, clearcoatRoughness: 0.04, metallic: 0.55 }
      : w.night ? { ...COAT, metallic: 0.5 } : w.candy ? { ior: 2.0 } : {});
  const run = upload(engine, tiles(objects, RUN.tiles, w.tiles, w.jungle), [floor]);
  for (const z of RUN.sections) place(engine, run, [0, 0, z]);
  if (w.night) {
    // Thin luminous inlays on the tile tops feed real SSR, not painted reflections.
    const edges = [w.neon, w.neon2].map(c => upload(engine,
      geometry.roundedBox([0.19, 0.006, 0.012], 0.005, 2),
      [plain(engine, c, 0.15, { emissive: c.map(v => srgbToLinear(v) * 5), castShadow: false })]));
    for (const z of RUN.sections) for (let row = 0; row < 12; row += 2) {
      for (let col = 0; col < 12; col += 2) place(engine, edges[(row / 2 + col / 2) % 2],
        [(col - 5.5) * 0.5, 0.008, z + (row - 5.5) * 0.5 + 0.21]);
    }
  }
}

function buildObstacles(engine, lib, w) {
  if (w.night) return buildNeonArchitecture(engine, lib, w);
  const { geometry, objects } = lib;
  const coat = w.candy ? { ...CANDY_GLOSS, ior: 1.8 } : w.polished ? CANDY_COAT : COAT;
  const hammer = engine.addSurfaceMaterial({
    function: 'raa.inflatable', params: { color: [1, 1, 1], stripeWidth: 0,
      seamDepth: w.polished ? 0.015 : 0.2 }, ...coat,
  });
  const s = HERO.scale;
  place(engine, upload(engine, (w.polished ? candyHammer(geometry, w.inflatable, w.jungle ? 2 : 1) : objects.inflatableHammer()), [hammer]), HERO.t, yawQuat(HERO.yaw), [s, s, s]);
  const pads = (w.jungle ? w.inflatable.slice(0, 2) : w.inflatable).map((c) => (w.polished || w.jungle)
    ? engine.addSurfaceMaterial({ function: 'raa.inflatable',
      params: { color: c, stripeWidth: 0, seamDepth: 0.015 }, ...coat })
    : inflatable(engine, c, w.stripe));
  // Candy pads are a little larger and rounder, so the clear-coat sky streak reads.
  const pad = geometry.roundedBox(w.candy ? [0.85, 0.26, 0.85] : w.polished ? [0.7, 0.24, 0.7] : [0.55, 0.18, 0.55],
    w.candy ? 0.24 : w.polished ? 0.23 : 0.14, w.candy ? 32 : w.polished ? 24 : 6);
  PADS.forEach(([x, z], i) => place(engine, upload(engine, pad, [pads[i % pads.length]]), [x, 0.18, z]));
  const arch = geometry.sweepTube(archPath(2.4, 0.4, w.polished ? 96 : 48), w.polished ? 0.42 : 0.3, w.polished ? 48 : 24, { caps: 'round' });
  place(engine, upload(engine, arch, [pads[1]]), [0, 0, INFLATABLE_ARCH_Z]);
  const tubes = [w.neon, w.neon2 ?? w.neon].map(c => w.night
    ? engine.addSurfaceMaterial({ function: 'raa.neon',
      params: { color: c, strength: 6, coreWhiteness: 0.15 }, castShadow: false })
    // Ice keeps a saturated cyan gas core (the white-core variant bleached the arch).
    : w.icy ? engine.addSurfaceMaterial({ function: 'raa.neon',
      params: { color: c, strength: 4, coreWhiteness: 0.15 }, castShadow: false })
    : neon(engine, c));
  const neonArch = geometry.sweepTube(archPath(2.1, 1.1), 0.05, 12, { caps: 'round' });
  NEON_ARCHES.forEach((z, i) => place(engine, upload(engine, neonArch, [tubes[i % 2]]), [0, 0, w.night ? z - 3 : z]));
  const ice = engine.addSurfaceMaterial({
    function: 'raa.ice', params: w.ice, blend: 'transmissive', transmission: 1, ior: 1.31, thickness: w.icy ? 1.4 : 0.8, roughness: w.icy ? 0.03 : 0.04, castShadow: false,
  });
  const blocks = w.icy ? [[-2.0, -2.8, 3], [-2.25, -5.3, 7], [2.1, -5.5, 11]] : ICE_BLOCKS;
  for (const [x, z, seed] of blocks) place(engine, upload(engine, objects.iceBlock({ seed }), [ice]),
    [x, 0, z], yawQuat(seed * 0.3), w.icy ? [1.8, 2.1, 1.8] : [1, 1, 1]);
  const plush = engine.addSurfaceMaterial({
    function: 'raa.plush',
    // Sparse long tufts resolve at capture size; dense subpixel strands look smooth.
    // Candy gets the tufted shells too: the judged capture needs a furry silhouette and
    // sheen, not the short pile that read as a smooth blue ball.
    params: { color: w.plush,
      furLength: w.candy ? 0.08 : w.icy ? 0.12 : (w.jungle || w.night) ? 0.09 : 0.022,
      furDensity: w.candy ? 8 : w.icy ? 10 : (w.jungle || w.night) ? 6 : 12,
      ...(w.icy ? { tipColor: [0.16, 0.45, 1], sheenColor: [0.5, 0.75, 1] }
        : w.candy ? { tipColor: [0.58, 0.74, 1], sheenColor: [0.82, 0.9, 1] } : {}) },
    furLength: w.candy ? 0.08 : w.icy ? 0.12 : (w.jungle || w.night) ? 0.09 : 0.022,
    furDensity: w.candy ? 8 : w.icy ? 10 : (w.jungle || w.night) ? 6 : 12, furDroop: 0.2,
    ...(w.icy || w.candy ? { furShells: 64 } : {}),
    ...(w.jungle || w.night ? { furShells: 64, roughness: 0.95 } : {}),
  });
  place(engine, upload(engine, objects.plushSphere(), [plush]), w.icy ? [-0.9, 0, -2.6] : PLUSH);
}

// The liquid strip in its basin beside the run.
function buildStrip(engine, lib, w) {
  if (w.night) return; // Ref-4 has luminous pipes, not a daytime liquid basin.
  const { geometry } = lib;
  const wall = w.polished ? plain(engine, w.inflatable[0], 0.18,
    w.candy ? { ...CANDY_GLOSS, ior: 1.8 } : CANDY_COAT)
    : inflatable(engine, w.inflatable[0], w.stripe);
  for (const sx of [-1, 1]) {
    // Lower the candy rim to expose the green surface from the low camera.
    const side = geometry.roundedBox([0.15, w.jungle ? 0.12 : w.polished ? 0.28 : 0.42, STRIP.d / 2 + 0.3], w.jungle ? 0.1 : w.polished ? 0.14 : 0.06, w.polished ? 16 : 4);
    place(engine, upload(engine, side, [wall]), [STRIP.x + sx * (STRIP.w / 2 + 0.15), GROUND_Y + 0.42, STRIP.z]);
  }
  const { function: fn, params, transmissive } = w.liquid;
  const blend = transmissive ? { blend: 'transmissive', transmission: w.polished ? 0.3 : 0.95, ior: 1.34, thickness: 0.4 } : {};
  const m = w.jungle ? plain(engine, [0.48, 0.25, 0.12], 0.72, { transmission: 0 })
    : engine.addSurfaceMaterial({ function: fn, params, ...blend,
      ...(w.candy ? CANDY_GLOSS : {}) });
  const surfaceY = w.jungle ? 0.04 : STRIP.y;
  place(engine, upload(engine, geometry.grid(STRIP.w, STRIP.d, STRIP.nx, STRIP.nz), [m]), [STRIP.x, surfaceY, STRIP.z]);
  if (w.jungle) {
    // Visible even at the fixed capture time, alongside the simulated splash particles.
    const mudBlob = upload(engine, geometry.uvSphere(1, 24, 16), [m]);
    for (let i = 0; i < 6; i++) {
      const x = STRIP.x + (i % 2 ? 1.15 : -1.15);
      const z = -0.8 - i * 2.2;
      const h = 0.28 + (i % 3) * 0.12;
      place(engine, mudBlob, [x, surfaceY + h * 0.6, z], yawQuat(i), [0.28, h, 0.22]);
      place(engine, mudBlob, [x - 0.12, surfaceY + h * 1.8, z + 0.1],
        identityQuat, [0.12, 0.16, 0.12]);
    }
  }
}

// The pillar gate across the far end and the distant big shapes.
function buildBackdrop(engine, lib, w) {
  if (w.night) return; // The neon architecture supplies pillars and stars, without mountains.
  const { geometry } = lib;
  const gate = plain(engine, w.gate, w.candy ? 0.22 : 0.35,
    w.candy ? { ...CANDY_GLOSS, ior: 1.9 } : COAT);
  // Seven separated, substantial columns silhouette against the low winter sun.
  const gateShape = w.icy ? { ...GATE, x0: -5.1, x1: 5.1, pitch: 1.7,
    half: [0.32, 5.4, 0.45], z: -21 } : w.jungle
      ? { ...GATE, x0: -5, x1: 5, half: [0.6, 4.2, 0.65], pitch: 2, z: -22 } : GATE;
  const pillar = upload(engine, geometry.roundedBox(gateShape.half, 0.08, 3), [gate]);
  for (let x = gateShape.x0; x <= gateShape.x1 + 1e-6; x += gateShape.pitch) place(engine, pillar, [x, gateShape.half[1] - 0.05, gateShape.z]);
  const lintel = geometry.roundedBox([(GATE.x1 - GATE.x0) / 2 + 0.6, 0.45, 0.45], 0.12, 3);
  if (!w.icy) place(engine, upload(engine, lintel, [gate]), [0, GATE.lintelY, gateShape.z]);
  if (w.night) {
    const stripes = [w.neon, w.neon2].map(c => upload(engine,
      geometry.roundedBox([0.035, 3.9, 0.025], 0.02, 3),
      [plain(engine, c, 0.15, { emissive: c.map(v => srgbToLinear(v) * 6), castShadow: false })]));
    for (let i = 0; i < 12; i++) place(engine, stripes[i % 2],
      [GATE.x0 + i * GATE.pitch, 4.15, GATE.z + 0.32]);
    // Sparse fixed stars above the vista, separate from the existing airborne sparkles.
    const star = upload(engine, geometry.uvSphere(1, 8, 4),
      [plain(engine, [0.4, 0.55, 1], 1, { emissive: [0.5, 0.8, 2], castShadow: false })]);
    for (let i = 0; i < 48; i++) {
      const hash = n => ((Math.imul(n, 2654435761) >>> 0) % 10000) / 10000;
      const r = 0.035 + hash(i + 99) * 0.05;
      place(engine, star, [(hash(i + 1) - 0.5) * 170, 22 + hash(i + 41) * 55, -140],
        identityQuat, [r, r, r]);
    }
  }
  const far = plain(engine, w.vista, 0.8);
  const shapes = {
    peak: (s) => geometry.lathe([[0, 0], [s, 0], [s * 0.12, s * 1.3], [0, s * 1.35]], 48),
    dome: (s) => geometry.uvSphere(s, 48, 24),
    tower: (s) => geometry.roundedBox([s * 0.3, s * 1.6, s * 0.3], s * 0.08, 3),
  };
  for (const [x, z, kind, size] of VISTA) {
    const y = kind === 'tower' ? GROUND_Y + size * 1.6 : GROUND_Y - (kind === 'dome' ? size * 0.35 : 0);
    place(engine, upload(engine, shapes[kind](size), [far]), [x, y, z]);
  }
  if (w.icy) {
    // Ref-2's rolling giant snowballs: white glossy spheres off the run, fading into the
    // snowy aerial haze behind the pillar gate.
    const snow = plain(engine, [0.93, 0.97, 1], 0.5, { clearcoat: 1, clearcoatRoughness: 0.22 });
    const ball = (r) => upload(engine, geometry.uvSphere(r, 48, 24), [snow]);
    for (const [x, y, z, r] of [[-9, 4.2, -20, 2.6], [-17, 3.6, -38, 4.2],
      [15, 2.8, -44, 3.4], [-27, 8, -62, 7], [28, 10, -78, 9], [-12, 2.2, -30, 1.8]]) {
      place(engine, ball(r), [x, y, z]);
    }
  }
  if (w.jungle) {
    // Simple palm silhouettes frame the run without introducing characters.
    const trunk = upload(engine, geometry.roundedBox([0.22, 2.5, 0.22], 0.18, 8),
      [plain(engine, [0.42, 0.28, 0.12], 0.85)]);
    const leaf = upload(engine, geometry.uvSphere(1, 24, 12),
      [plain(engine, [0.25, 0.65, 0.025], 0.55)]);
    for (const [x, z] of [[-5, -7], [8, -10], [-7, -17], [9, -23], [-9, -30], [12, -36]]) {
      place(engine, trunk, [x, 1.9, z]);
      for (let i = 0; i < 6; i++) {
        const a = i * Math.PI / 3;
        place(engine, leaf, [x + Math.cos(a) * 1.1, 4.5, z + Math.sin(a) * 1.1],
          yawQuat(-a), [1.8, 0.16, 0.48]);
      }
    }
  }
}

// Judge text shared by the four worlds; each variant adds the mood of its reference.
const VISTA_TEXT = 'A low camera looks down a long straight obstacle run of glossy coloured tiles '
  + 'that mirror the obstacles (screen-space reflections), past a striped inflatable hammer in '
  + 'sharp focus on the right, clear-coated pads and an inflatable arch, ice blocks, a furry plush '
  + 'ball, glowing neon tube arches, a liquid strip beside the run and particles in the air, to a '
  + 'row of tall pillars with light shafts between them and big distant shapes fading into haze. '
  + 'The far run is softly out of focus (depth of field) and the colours are graded. Expected: '
  + 'simple procedural shapes and no characters; score the mood, gloss, depth haze and grading.';
const WORLD_TEXT = {
  candy: 'Target ref-1: a sunny pastel candy world, a blue sky with soft haze, a low warm sun '
    + 'behind the gate, glossy pink, yellow and blue inflatables and bright green slime.',
  ice: 'Target ref-2: an icy snowy blue-white world, crisp cold light, falling snow, clear '
    + 'blue ice blocks, a blue plush ball, white snowy haze over the distance.',
  jungle: 'Target ref-3: a humid green jungle, warm low sun through the pillars, lime and '
    + 'yellow inflatables, a brown mud strip with splashes, thick green humid haze.',
  neon: 'Target ref-4: a neon night world, a dark violet sky, magenta and cyan glowing tubes '
    + 'with strong bloom reflected in the dark glossy floor, sparkles and violet night haze.',
};
// Ref-4's geometry is different from refs 1-3. Retain the same visual quality
// criteria without asking the judge for the explicitly excluded daytime props.
const NEON_VISTA_TEXT = 'A low camera looks down a long straight run of dark glossy violet '
  + 'stepping tiles with luminous chevrons and screen-space reflections of saturated magenta '
  + 'and cyan tubes. Glossy violet pipes, checker pillars and a magenta pipe overhead frame '
  + 'the run. Glowing tube arches lead to distant pillars fading into violet night haze. '
  + 'Star points and airborne sparkles punctuate a dark violet sky. The far run is softly '
  + 'out of focus (depth of field) and the colours are graded. Expected: simple procedural '
  + 'shapes and no characters; score the mood, gloss, depth haze and grading.';

const VARIANTS = Object.keys(WORLDS).map((world, i) => ({
  id: `world-${world}`,
  description: `${world === 'neon' ? NEON_VISTA_TEXT : VISTA_TEXT} ${WORLD_TEXT[world]}`,
  references: [`ref-${i + 1}`],
  judge: true,
  settings: { world, ...(world === 'candy' ? {
    // See over the basin rim and bring the hammer forward without losing the blue sky.
    camera: { ...CAMERA, eye: [0.35, 1.45, 2.5], target: [0.35, 1.7, -24], fovYDeg: 56 },
    // A longer, faster lens: the hammer stays sharp while the far run visibly softens.
    dof: { fStop: 2.8, focalLength: 0.075, focus: HERO.focus },
    // Pastel blue-white aerial haze: keep the blue sky and the near run's colour, fade
    // the distant vista ~60 % toward the sky (the old cream colour washed the frame out).
    // Warm sun inscatter keeps the low sun behind the gate glowing through the haze.
    fog: { preset: 'haze', density: 0.003, heightFalloff: 0.1, start: 20,
      color: [0.78, 0.86, 0.98], inscatter: 0.2, aerialStrength: 0.7, aerialDistance: 180 },
    // The sun now sits in a gap between the gate pillars, so the wedges read. The additive
    // integral has no extinction: keep it near the 5193 budget (0.12 x 0.02 with a 5.5 sun)
    // instead of the 30x value that washed the last capture white.
    sunShafts: { intensity: 0.25, density: 0.02, maxDistance: 60 },
    // Glossy floor: full-strength SSR over the 0.08-roughness tiles.
    ssr: { intensity: 1.8, maxRoughness: 0.5, thickness: 0.3 },
    grading: { preset: 'neutral', contrast: 1.08,
      saturation: 1.18, vibrance: 0.12 },
  } : world === 'ice' ? {
    // See the ice faces and snowy surround; retain sharp foreground plush and hammer.
    camera: { ...CAMERA, eye: [0.35, 1.45, 2.5], target: [0.35, 1.7, -24], fovYDeg: 56 },
    dof: { fStop: 2, focalLength: 0.085, focus: HERO.focus },
    // Snowy haze stays in the far half (start 24 m) so the near run keeps its saturated
    // blue; the sun scatter is kept at the candy budget so the sky never turns white.
    fog: { preset: 'snow', density: 0.0009, start: 24, heightFalloff: 0.25,
      color: [0.14, 0.46, 0.9], inscatter: 0.05,
      aerialStrength: 0.5, aerialDistance: 200 },
    sunShafts: { intensity: 0.45, density: 0.02, maxDistance: 55 },
    ssr: { intensity: 2, maxRoughness: 0.55, thickness: 0.3 },
  } : world === 'jungle' ? {
    camera: { ...CAMERA, eye: [0.35, 1.45, 2.5], target: [0.35, 1.7, -24], fovYDeg: 56 },
    dof: { fStop: 4, focus: HERO.focus },
    // Blue-green humid aerial haze stays distant; height falloff preserves blue overhead.
    fog: { preset: 'jungle', density: 0.002, start: 20, heightFalloff: 0.3,
      color: [0.12, 0.42, 0.28], inscatter: 0.03,
      aerialStrength: 0.25, aerialDistance: 180 },
    // Shafts integrate density per metre without extinction, independently of fog.
    // The old 0.7 added tens of HDR units and washed out even the near objects.
    sunShafts: { intensity: 0.6, density: 0.008, maxDistance: 45 },
  } : world === 'neon' ? {
    camera: { ...CAMERA, eye: [0.2, 0.75, 3.5], target: [0.2, 0.95, -24], fovYDeg: 56 },
    // Focus on the furry plush so its silhouette and tufts stay sharp; the far run softens.
    dof: { fStop: 4, focus: [PLUSH[0], 0.5, PLUSH[2]] },
    // Far shapes only: no height fog before 28 m, a weak violet fade toward the horizon.
    fog: { preset: 'night', density: 0.0006, heightFalloff: 0.25, start: 28,
      color: [0.6, 0.25, 1], inscatter: 0,
      aerialStrength: 0.1, aerialDistance: 240 },
    ssr: { intensity: 2, maxRoughness: 0.45, thickness: 0.25 },
    // Keep the additive shaft integral subtle and violet against the dark sky.
    sunShafts: { intensity: 0.15, density: 0.002, maxDistance: 48 },
  } : {}) },
}));

// On/off pairs (Q15): all in the candy world; each pair shares its settings and the -off
// variant only disables the feature's pass. `settings` overrides are applied after the
// world: camera, sun, dof, fog (merged over the world's fog preset), sunShafts.
const PLUSH_FOCUS = [PLUSH[0], PLUSH[1] + 0.35, PLUSH[2]];
const PAIRS = {
  // Camera low over the glossy floor, so the run mirrors the obstacles.
  ssr: { pass: 'ssr', text: 'screen-space reflections of the obstacles in the glossy floor',
    settings: { camera: { ...CAMERA, eye: [0, 0.3, 3.2], target: [0, 0.6, -20] } } },
  // f/1.4 focused on the near plush ball: hammer foreground and far run blur, round bokeh.
  dof: { pass: 'dof', text: 'shallow depth of field focused on the near plush ball with round bokeh',
    settings: { dof: { fStop: 1.4, focus: PLUSH_FOCUS } } },
  fog: { pass: 'fog', text: 'height fog and strong aerial perspective fading the distance',
    settings: { fog: { preset: 'haze', aerialStrength: 2 } } },
  // Candy's low sun sits behind the pillar gate; denser fog makes the shafts read.
  sunshafts: { pass: 'sunshafts', text: 'sun shafts through the gaps of the pillar gate',
    settings: { sun: WORLDS.candy.sun, fog: { preset: 'haze', density: 0.04 }, sunShafts: { intensity: 1 } } },
  grading: { pass: 'grading', text: 'candy colour grading', settings: {} },
};
for (const [feature, pair] of Object.entries(PAIRS)) {
  for (const on of [true, false]) {
    VARIANTS.push({
      id: `${feature}-${on ? 'on' : 'off'}`,
      description: `${VISTA_TEXT} Pair ${feature}: ${on ? 'with' : 'without'} ${pair.text}.`,
      references: ['ref-1'],
      judge: false,
      ...(on ? {} : { passes: { disable: [pair.pass] } }),
      settings: { world: 'candy', ...pair.settings },
    });
  }
}

export default {
  id: 'm14-camera-effects',
  milestone: 'm14',
  title: 'Camera effects: SSR, depth of field, fog and aerial perspective, sun shafts, grading',
  engine: { msaa: 4, toneMapper: 'agx' },
  variants: VARIANTS,

  async build(engine, lib, variant) {
    const name = variant.settings?.world;
    const w = variant.id === 'world-candy' ? CANDY_LOOK : WORLDS[name];
    if (!w) throw new Error(`m14-camera-effects: no world for variant '${variant.id}'`);
    engine.applyEnvironment({
      sky: w.sky,
      sun: { dir: w.sun.dir, color: w.sun.color, intensity: w.sun.intensity },
      fog: w.fog,
      post: w.post,
    });
    // Candy's clear-coat sheen: a brighter IBL specular so the glossy inflatables, pads and
    // slime mirror the bright blue sky (the old capture read matte). The judged world only,
    // so the on/off pairs and the accepted worlds keep their lighting.
    if (w.candy) engine.setEnvironmentLighting({ diffuse: 1.1, specular: 2.6 });
    // The night sky stays dark in the background; raise only the IBL so the surfaces keep
    // a dim cool fill. This does not brighten the sky image.
    if (w.night) engine.setEnvironmentLighting({ diffuse: 3.5, specular: 3 });
    buildRun(engine, lib, w);
    buildObstacles(engine, lib, w);
    buildStrip(engine, lib, w);
    buildBackdrop(engine, lib, w);
    engine.addEmitter({ maxParticles: 8192, seed: 61, ...w.particles });
    if (w.icy) engine.addEmitter({ ...w.particles, seed: 97, maxParticles: 8192,
      position: [0, 3.2, -12], volume: [5, 2.6, 6], rate: 1400, size: [0.09, 0.15] });
    const o = variant.settings;
    if (o.grading) engine.setGrading(o.grading);
    if (o.sun) engine.applyEnvironment({ sun: o.sun });
    if (o.fog) engine.setFog(o.fog);
    if (o.sunShafts) engine.setSunShafts(o.sunShafts);
    if (o.ssr) engine.setSsr(o.ssr);
    const camera = o.camera ?? CAMERA;
    engine.setCamera(camera);
    const dof = o.dof ?? { fStop: F_STOP, focus: HERO.focus };
    engine.setDof({ fStop: dof.fStop, autofocus: screenUv(dof.focus, camera),
      ...(dof.focalLength ? { focalLength: dof.focalLength } : {}) });
  },
};
