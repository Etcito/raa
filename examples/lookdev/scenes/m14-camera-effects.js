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
    sky: 'day', fog: 'haze', post: 'candy',
    // Low, behind the gate: the sun travels towards the camera, 12 deg up, so from the
    // camera it sits behind the pillars (about 7.3 m up at the gate), below the lintel.
    sun: { dir: [0.12, -0.21, 0.97], color: [1, 0.92, 0.8], intensity: 7 },
    tiles: [[1, 0.45, 0.72], [0.35, 0.7, 1], [1, 0.86, 0.3], [0.62, 0.5, 1]],
    plinth: [0.98, 0.55, 0.78], ground: [0.55, 0.85, 0.55], vista: [0.72, 0.62, 0.95],
    inflatable: [[1, 0.25, 0.6], [1, 0.82, 0.15], [0.25, 0.55, 1]], stripe: [1, 1, 1],
    neon: [1, 0.3, 0.75], plush: [0.2, 0.3, 0.95], gate: [1, 0.85, 0.3],
    ice: { tint: [0.6, 0.9, 1], absorption: 1.8, cloudiness: 0.2, crackScale: 0.25 },
    liquid: { function: 'raa.liquid.slime', params: { color: [0.34, 0.95, 0.03] }, transmissive: true },
    particles: { kind: 'confetti', position: [0, 3.5, -7], volume: [2.5, 0.5, 4], direction: [0, 1, 0], spread: 0.6, rate: 300, speed: [1.5, 3], groundY: 0, colors: [[1, 0.15, 0.45], [1, 0.8, 0.1], [0.2, 0.55, 1], [0.3, 0.9, 0.3]] },
  },
  ice: {
    sky: 'snow', fog: 'snow', post: 'ice',
    // Higher sun from behind the camera's left: the ice faces and snow catch light.
    sun: { dir: [0.35, -0.55, -0.76], color: [1, 0.96, 0.9], intensity: 6.5 },
    tiles: [[0.55, 0.82, 1], [0.85, 0.95, 1], [0.35, 0.62, 0.95], [1, 0.45, 0.75]],
    plinth: [0.75, 0.9, 1], ground: [0.94, 0.97, 1], vista: [0.88, 0.93, 1],
    inflatable: [[0.3, 0.6, 1], [1, 0.35, 0.65], [1, 1, 1]], stripe: [1, 1, 1],
    neon: [0.35, 0.85, 1], plush: [0.15, 0.25, 0.9], gate: [0.95, 0.4, 0.7],
    ice: { tint: [0.42, 0.82, 1], absorption: 2.4, cloudiness: 0.3, crackScale: 0.3 },
    liquid: { function: 'raa.liquid.water', params: {}, transmissive: true },
    particles: { kind: 'snow', position: [0, 4, -9], volume: [8, 3.5, 12], groundY: 0, rate: 2400, colors: [[0.97, 0.98, 1]] },
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

const identityQuat = [0, 0, 0, 1];
const yawQuat = (a) => [0, Math.sin(a / 2), 0, Math.cos(a / 2)];
const srgbToLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const sub = (a, b) => a.map((x, i) => x - b[i]);
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a) => a.map((x) => x / Math.hypot(...a));

// Screen [u, v] (0..1, v down) of world point p seen by CAMERA: the DOF autofocus point.
function screenUv(p) {
  const fwd = unit(sub(CAMERA.target, CAMERA.eye));
  const right = unit(cross(fwd, [0, 1, 0]));
  const up = cross(right, fwd);
  const d = sub(p, CAMERA.eye);
  const t = Math.tan((CAMERA.fovYDeg * Math.PI) / 360);
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
  const floor = plain(engine, [1, 1, 1], FLOOR_ROUGHNESS);
  const run = upload(engine, tiles(objects, RUN.tiles, w.tiles), [floor]);
  for (const z of RUN.sections) place(engine, run, [0, 0, z]);
}

function buildObstacles(engine, lib, w) {
  const { geometry, objects } = lib;
  const hammer = engine.addSurfaceMaterial({
    function: 'raa.inflatable', params: { color: [1, 1, 1], stripeWidth: 0, seamDepth: 0.2 }, ...COAT,
  });
  const s = HERO.scale;
  place(engine, upload(engine, objects.inflatableHammer(), [hammer]), HERO.t, yawQuat(HERO.yaw), [s, s, s]);
  const pads = w.inflatable.map((c) => inflatable(engine, c, w.stripe));
  const pad = geometry.roundedBox([0.55, 0.18, 0.55], 0.14, 6);
  PADS.forEach(([x, z], i) => place(engine, upload(engine, pad, [pads[i % pads.length]]), [x, 0.18, z]));
  const arch = geometry.sweepTube(archPath(2.4, 0.4), 0.3, 24, { caps: 'round' });
  place(engine, upload(engine, arch, [pads[1]]), [0, 0, INFLATABLE_ARCH_Z]);
  const tubes = [neon(engine, w.neon), neon(engine, w.neon2 ?? w.neon)];
  const neonArch = geometry.sweepTube(archPath(2.1, 1.1), 0.05, 12, { caps: 'round' });
  NEON_ARCHES.forEach((z, i) => place(engine, upload(engine, neonArch, [tubes[i % 2]]), [0, 0, z]));
  const ice = engine.addSurfaceMaterial({
    function: 'raa.ice', params: w.ice, blend: 'transmissive', transmission: 1, ior: 1.31, thickness: 0.8, roughness: 0.04, castShadow: false,
  });
  for (const [x, z, seed] of ICE_BLOCKS) place(engine, upload(engine, objects.iceBlock({ seed }), [ice]), [x, 0, z], yawQuat(seed * 0.3));
  const plush = engine.addSurfaceMaterial({
    function: 'raa.plush',
    params: { color: w.plush, furLength: 0.022, furDensity: 12 },
    furLength: 0.022, furDensity: 12, furDroop: 0.2,
  });
  place(engine, upload(engine, objects.plushSphere(), [plush]), PLUSH);
}

// The liquid strip in its basin beside the run.
function buildStrip(engine, lib, w) {
  const { geometry } = lib;
  const wall = inflatable(engine, w.inflatable[0], w.stripe);
  for (const sx of [-1, 1]) {
    const side = geometry.roundedBox([0.15, 0.42, STRIP.d / 2 + 0.3], 0.06, 4);
    place(engine, upload(engine, side, [wall]), [STRIP.x + sx * (STRIP.w / 2 + 0.15), GROUND_Y + 0.42, STRIP.z]);
  }
  const { function: fn, params, transmissive } = w.liquid;
  const blend = transmissive ? { blend: 'transmissive', transmission: 0.95, ior: 1.34, thickness: 0.4 } : {};
  const m = engine.addSurfaceMaterial({ function: fn, params, ...blend });
  place(engine, upload(engine, geometry.grid(STRIP.w, STRIP.d, STRIP.nx, STRIP.nz), [m]), [STRIP.x, STRIP.y, STRIP.z]);
}

// The pillar gate across the far end and the distant big shapes.
function buildBackdrop(engine, lib, w) {
  const { geometry } = lib;
  const gate = plain(engine, w.gate, 0.35, COAT);
  const pillar = upload(engine, geometry.roundedBox(GATE.half, 0.08, 3), [gate]);
  for (let x = GATE.x0; x <= GATE.x1 + 1e-6; x += GATE.pitch) place(engine, pillar, [x, GATE.half[1] - 0.05, GATE.z]);
  const lintel = geometry.roundedBox([(GATE.x1 - GATE.x0) / 2 + 0.6, 0.45, 0.45], 0.12, 3);
  place(engine, upload(engine, lintel, [gate]), [0, GATE.lintelY, GATE.z]);
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
  settings: { world },
}));

export default {
  id: 'm14-camera-effects',
  milestone: 'm14',
  title: 'Camera effects: SSR, depth of field, fog and aerial perspective, sun shafts, grading',
  engine: { msaa: 4, toneMapper: 'agx' },
  variants: VARIANTS,

  async build(engine, lib, variant) {
    const name = variant.settings?.world;
    const w = WORLDS[name];
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
    engine.setCamera(CAMERA);
    engine.setDof({ fStop: F_STOP, autofocus: screenUv(HERO.focus) });
  },
};
