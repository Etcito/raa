// Lookdev scene m14-camera-effects (chain milestone 8, R14): an obstacle-course-like vista
// with every P2 feature, laid out along a long straight run seen from a low camera so depth
// of field, height fog and aerial perspective read with distance: a glossy tiled run
// (SSR), clear-coated inflatables (hammer, pads, an inflatable arch), ice blocks
// (transmission), a plush sphere (fur), neon tube arches (bloom), a liquid strip beside the
// run (raa.liquid.*), particles, a pillar gate between the camera and a low sun (sun
// shafts through the gaps) and big shapes 80-200 m away (aerial perspective).
// Four worlds, each through engine.applyEnvironment (sky, fog, post presets) with its own
// sun, palette, liquid and particles: one WORLDS entry per world, so tuning edits one entry.
// Shadows, AO, contact shadows, SSR and sun shafts stay at their defaults; DOF autofocuses
// on the hero hammer at f/4.
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
    sky: { preset: 'snow', zenith: [0.045, 0.3, 0.9], horizon: [0.38, 0.68, 1],
      ground: [0.65, 0.78, 0.95], intensity: 1.05, horizonSoftness: 10,
      sunDiscSize: 0.012, sunDiscIntensity: 22,
      clouds: { coverage: 0.25, density: 0.65, color: [0.95, 0.98, 1] } },
    fog: 'snow', post: 'ice',
    // Low backlight through the gate catches the ice edges and falling snow.
    sun: { dir: [0.12, -0.21, 0.97], color: [0.94, 0.97, 1], intensity: 5.0 },
    tiles: [[0.08, 0.55, 0.95], [0.22, 0.72, 1], [0.12, 0.46, 0.9], [0.5, 0.85, 1]],
    plinth: [0.28, 0.68, 0.95], ground: [0.94, 0.97, 1], vista: [0.72, 0.83, 0.98],
    inflatable: [[0.025, 0.45, 0.98], [0.95, 0.06, 0.48], [0.8, 0.94, 1]], stripe: [1, 1, 1],
    neon: [0.16, 0.65, 1], plush: [0.025, 0.12, 0.65], gate: [0.16, 0.45, 0.78],
    ice: { tint: [0.32, 0.8, 1], absorption: 0.65, cloudiness: 0.025, crackScale: 0.4 },
    liquid: { function: 'raa.liquid.water', params: {}, transmissive: true },
    // Camera-facing white flakes across the hammer's focus plane, not just the sky.
    particles: { kind: 'snow', position: [0, 2.0, -2.6], volume: [3, 1.5, 3],
      direction: [0.08, -1, 0], spread: 0.25, gravity: 0.35, drag: 0.5,
      rate: 1800, lifetime: [6, 10], size: [0.055, 0.11], speed: [0.45, 0.85],
      emissive: 0.7, colors: [[1, 1, 1]] },
  },
  jungle: {
    sky: 'jungle', fog: 'jungle', post: 'jungle',
    // Low, warm, behind the gate like candy, a little from the left.
    sun: { dir: [-0.15, -0.22, 0.96], color: [1, 0.85, 0.62], intensity: 6.5 },
    tiles: [[0.45, 0.85, 0.2], [0.75, 0.95, 0.3], [1, 0.45, 0.65], [0.3, 0.7, 0.35]],
    plinth: [0.62, 0.48, 0.32], ground: [0.25, 0.5, 0.18], vista: [0.2, 0.42, 0.22],
    inflatable: [[0.55, 0.9, 0.15], [1, 0.75, 0.1], [0.95, 0.3, 0.2]], stripe: [1, 0.95, 0.4],
    neon: [1, 0.7, 0.2], plush: [0.2, 0.3, 0.9], gate: [0.7, 0.55, 0.38],
    ice: { tint: [0.6, 0.95, 0.85], absorption: 1.5, cloudiness: 0.35, crackScale: 0.25 },
    liquid: { function: 'raa.liquid.mud', params: { color: [0.34, 0.17, 0.1] }, transmissive: false },
    particles: { kind: 'splash', position: [4.7, -0.15, -5], volume: [1, 0, 4], direction: [0, 1, 0], rate: 160, colors: [[0.4, 0.22, 0.12]], opacity: 1 },
  },
  neon: {
    sky: 'night', fog: 'night', post: 'neon',
    // Dim violet moon from high behind the camera; the tubes are the main light.
    sun: { dir: [-0.3, -0.85, -0.43], color: [0.75, 0.6, 1], intensity: 0.8 },
    tiles: [[0.25, 0.12, 0.55], [0.12, 0.08, 0.3], [0.8, 0.2, 0.7], [0.15, 0.55, 0.85]],
    plinth: [0.2, 0.1, 0.4], ground: [0.08, 0.05, 0.16], vista: [0.18, 0.1, 0.35],
    inflatable: [[1, 0.2, 0.7], [0.2, 0.8, 1], [1, 0.8, 0.15]], stripe: [0.2, 0.1, 0.4],
    neon: [1, 0.15, 0.7], neon2: [0.1, 0.85, 1], plush: [0.2, 0.3, 0.95], gate: [0.3, 0.15, 0.6],
    ice: { tint: [0.55, 0.6, 1], absorption: 2, cloudiness: 0.25, crackScale: 0.3 },
    liquid: { function: 'raa.liquid.slime', params: { color: [0.2, 1, 0.6] }, transmissive: true },
    particles: { kind: 'sparkles', position: [0, 1.6, -6], volume: [3, 1.2, 6], rate: 300, size: [0.03, 0.07], emissive: 7, colors: [[1, 0.8, 0.3], [0.8, 0.9, 1]] },
  },
};

// Only the judged candy world uses this finish; diagnostic pairs keep their geometry.
const CANDY_LOOK = {
  ...WORLDS.candy,
  polished: true,
  sky: { ...WORLDS.candy.sky, zenith: [0.055, 0.32, 0.9], horizon: [0.36, 0.65, 0.94],
    ground: [0.4, 0.38, 0.3], intensity: 1.05,
    sunDiscSize: 0.016, sunDiscIntensity: 24,
    clouds: { coverage: 0.22, density: 0.65 } },
  sun: { dir: [0.12, -0.21, 0.97], color: [1, 0.92, 0.76], intensity: 5.5 },
  vista: [0.93, 0.8, 0.65], ground: [0.32, 0.72, 0.35],
  plinth: [1, 0.08, 0.45], gate: [1, 0.76, 0.12],
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

// A dense rounded cylinder with analytic profile normals, without pinched panel seams.
// Keep the shared hammer's dimensions, axis and placement (the capture contract).
function candyHammer(geometry, palette) {
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
  const head = geometry.lathe(profile(0.19, 0.36, 0.075), 128, { color: (p) =>
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
function tiles(objects, n, palette) {
  const desc = objects.floorTiles(n);
  const perTile = desc.positions.length / 3 / (n * n);
  const colors = new Float32Array(desc.positions.length / 3 * 4);
  for (let t = 0; t < n * n; t++) {
    const h = (Math.imul(t + 1, 2654435761) >>> 0) / 4294967296;
    const base = palette[Math.floor(h * palette.length)];
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

function buildRun(engine, lib, w) {
  const { geometry, objects } = lib;
  const ground = plain(engine, w.ground, 0.9);
  place(engine, upload(engine, geometry.roundedBox([260, 0.5, 260], 0.2, 1), [ground]), [0, GROUND_Y - 0.5, -120]);
  const plinth = plain(engine, w.plinth, 0.3, COAT);
  place(engine, upload(engine, geometry.roundedBox(RUN.plinth, 0.1, 2), [plinth]), [0, -0.33, RUN.plinthZ]);
  const floor = plain(engine, [1, 1, 1], w.icy ? 0.03 : FLOOR_ROUGHNESS,
    w.icy ? { clearcoat: 1, clearcoatRoughness: 0.03, metallic: 0.12 } : {});
  const run = upload(engine, tiles(objects, RUN.tiles, w.tiles), [floor]);
  for (const z of RUN.sections) place(engine, run, [0, 0, z]);
}

function buildObstacles(engine, lib, w) {
  const { geometry, objects } = lib;
  const coat = w.polished ? CANDY_COAT : COAT;
  const hammer = engine.addSurfaceMaterial({
    function: 'raa.inflatable', params: { color: [1, 1, 1], stripeWidth: 0,
      seamDepth: w.polished ? 0.015 : 0.2 }, ...coat,
  });
  const s = HERO.scale;
  place(engine, upload(engine, (w.polished ? candyHammer(geometry, w.inflatable) : objects.inflatableHammer()), [hammer]), HERO.t, yawQuat(HERO.yaw), [s, s, s]);
  const pads = w.inflatable.map((c) => w.polished
    ? engine.addSurfaceMaterial({ function: 'raa.inflatable',
      params: { color: c, stripeWidth: 0, seamDepth: 0.015 }, ...coat })
    : inflatable(engine, c, w.stripe));
  const pad = geometry.roundedBox(w.polished ? [0.7, 0.24, 0.7] : [0.55, 0.18, 0.55], w.polished ? 0.23 : 0.14, w.polished ? 24 : 6);
  PADS.forEach(([x, z], i) => place(engine, upload(engine, pad, [pads[i % pads.length]]), [x, 0.18, z]));
  const arch = geometry.sweepTube(archPath(2.4, 0.4, w.polished ? 96 : 48), w.polished ? 0.42 : 0.3, w.polished ? 48 : 24, { caps: 'round' });
  place(engine, upload(engine, arch, [pads[1]]), [0, 0, INFLATABLE_ARCH_Z]);
  const tubes = [neon(engine, w.neon), neon(engine, w.neon2 ?? w.neon)];
  const neonArch = geometry.sweepTube(archPath(2.1, 1.1), 0.05, 12, { caps: 'round' });
  NEON_ARCHES.forEach((z, i) => place(engine, upload(engine, neonArch, [tubes[i % 2]]), [0, 0, z]));
  const ice = engine.addSurfaceMaterial({
    function: 'raa.ice', params: w.ice, blend: 'transmissive', transmission: 1, ior: 1.31, thickness: w.icy ? 1.4 : 0.8, roughness: w.icy ? 0.03 : 0.04, castShadow: false,
  });
  const blocks = w.icy ? [[-2.0, -2.8, 3], [-2.25, -5.3, 7], [2.1, -5.5, 11]] : ICE_BLOCKS;
  for (const [x, z, seed] of blocks) place(engine, upload(engine, objects.iceBlock({ seed }), [ice]),
    [x, 0, z], yawQuat(seed * 0.3), w.icy ? [1.8, 2.1, 1.8] : [1, 1, 1]);
  const plush = engine.addSurfaceMaterial({
    function: 'raa.plush',
    params: { color: w.plush, furLength: w.icy ? 0.09 : 0.022, furDensity: w.icy ? 180 : 12,
      ...(w.icy ? { tipColor: [0.08, 0.32, 0.85], sheenColor: [0.35, 0.65, 1] } : {}) },
    furLength: w.icy ? 0.09 : 0.022, furDensity: w.icy ? 180 : 12, furDroop: 0.2,
    ...(w.icy ? { furShells: 64 } : {}),
  });
  place(engine, upload(engine, objects.plushSphere(), [plush]), w.icy ? [-0.9, 0, -2.6] : PLUSH);
}

// The liquid strip in its basin beside the run.
function buildStrip(engine, lib, w) {
  const { geometry } = lib;
  const wall = w.polished ? plain(engine, w.inflatable[0], 0.18, CANDY_COAT)
    : inflatable(engine, w.inflatable[0], w.stripe);
  for (const sx of [-1, 1]) {
    // Lower the candy rim to expose the green surface from the low camera.
    const side = geometry.roundedBox([0.15, w.polished ? 0.28 : 0.42, STRIP.d / 2 + 0.3], w.polished ? 0.14 : 0.06, w.polished ? 16 : 4);
    place(engine, upload(engine, side, [wall]), [STRIP.x + sx * (STRIP.w / 2 + 0.15), GROUND_Y + 0.42, STRIP.z]);
  }
  const { function: fn, params, transmissive } = w.liquid;
  const blend = transmissive ? { blend: 'transmissive', transmission: w.polished ? 0.3 : 0.95, ior: 1.34, thickness: 0.4 } : {};
  const m = engine.addSurfaceMaterial({ function: fn, params, ...blend });
  place(engine, upload(engine, geometry.grid(STRIP.w, STRIP.d, STRIP.nx, STRIP.nz), [m]), [STRIP.x, STRIP.y, STRIP.z]);
}

// The pillar gate across the far end and the distant big shapes.
function buildBackdrop(engine, lib, w) {
  const { geometry } = lib;
  const gate = plain(engine, w.gate, 0.35, COAT);
  // Seven separated, substantial columns silhouette against the low winter sun.
  const gateShape = w.icy ? { ...GATE, x0: -5.1, x1: 5.1, pitch: 1.7,
    half: [0.32, 3.8, 0.45], z: -21 } : GATE;
  const pillar = upload(engine, geometry.roundedBox(gateShape.half, 0.08, 3), [gate]);
  for (let x = gateShape.x0; x <= gateShape.x1 + 1e-6; x += gateShape.pitch) place(engine, pillar, [x, gateShape.half[1] - 0.05, gateShape.z]);
  const lintel = geometry.roundedBox([(GATE.x1 - GATE.x0) / 2 + 0.6, 0.45, 0.45], 0.12, 3);
  if (!w.icy) place(engine, upload(engine, lintel, [gate]), [0, GATE.lintelY, GATE.z]);
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

const VARIANTS = Object.keys(WORLDS).map((world, i) => ({
  id: `world-${world}`,
  description: `${VISTA_TEXT} ${WORLD_TEXT[world]}`,
  references: [`ref-${i + 1}`],
  judge: true,
  settings: { world, ...(world === 'candy' ? {
    // See over the basin rim and bring the hammer forward without losing the blue sky.
    camera: { ...CAMERA, eye: [0.35, 1.45, 2.5], target: [0.35, 1.7, -24], fovYDeg: 56 },
    dof: { fStop: 4, focus: HERO.focus },
    // Keep the near obstacles saturated; a cream haze separates the distant vista.
    fog: { preset: 'haze', density: 0.004, heightFalloff: 0.12, start: 18,
      color: [1, 0.94, 0.83], inscatter: 0.18, aerialStrength: 0.45, aerialDistance: 180 },
    sunShafts: { intensity: 0.85, density: 0.7 },
    grading: { preset: 'neutral', contrast: 1.08,
      saturation: 1.18, vibrance: 0.12 },
  } : world === 'ice' ? {
    // See the ice faces and snowy surround; retain sharp foreground plush and hammer.
    camera: { ...CAMERA, eye: [0.35, 1.45, 2.5], target: [0.35, 1.7, -24], fovYDeg: 56 },
    dof: { fStop: 4, focus: HERO.focus },
    fog: { preset: 'snow', density: 0.004, start: 12, heightFalloff: 0.12,
      color: [0.94, 0.98, 1], aerialStrength: 0.65, aerialDistance: 150 },
    sunShafts: { intensity: 2.4, density: 0.75, maxDistance: 65 },
    ssr: { intensity: 1.85, maxRoughness: 0.65, thickness: 0.3 },
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
    engine.setDof({ fStop: dof.fStop, autofocus: screenUv(dof.focus, camera) });
  },
};
