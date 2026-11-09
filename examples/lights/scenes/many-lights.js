import { captureEvidence } from '../../../web/common/lights-evidence.js';
// The 64-light preset (S20, RAA-LIGHTS-PAGE, QB8): a fixed camera over an eight-by-eight
// field of instances, each with its own bounded point light. The on variant adds all 64
// distinct lights; the off variant keeps the same geometry under sun and ambient light.
// The numeric run is configured here: 1920x1080 at devicePixelRatio 1, 60 warm-up frames
// and 300 measured frames, with fpsMean >= 60 required for the on run. The runner builds
// each scene once, so the device and shader setup stay resident through every measured
// frame; no per-frame allocation or timer fallback contributes to the frame rate.
const camera = { eye: [0, 9, 13], target: [0, 0.4, 0], fovYDeg: 45, near: 0.1, far: 100 };
const SIDE = 8;
const SPACING = 1.4;
const COUNT = SIDE * SIDE;
const half = (v) => v - (SIDE - 1) / 2;
const positions = Array.from({ length: COUNT }, (_, i) => [
  half(i % SIDE) * SPACING, 0.35, half(Math.floor(i / SIDE)) * SPACING,
]);
const tints = [
  [1, 0.3, 0.2], [1, 0.62, 0.2], [1, 1, 0.25], [0.4, 1, 0.4],
  [0.25, 1, 0.85], [0.3, 0.6, 1], [0.7, 0.4, 1], [1, 0.35, 0.8],
];
const colorAt = (i) => {
  const scale = 0.55 + (0.45 * Math.floor(i / SIDE)) / (SIDE - 1);
  return tints[i % SIDE].map((c) => c * scale);
};
const measure = {
  warmup: 60,
  frames: 300,
  width: 1920,
  height: 1080,
  devicePixelRatio: 1,
  minFpsMean: 60,
  measurementFor: 'Measured in Chrome on Windows through the managed GPU client, at the '
    + "page's fixed camera and 300 measured frames after warm-up: the many-lights preset "
    + 'must hold at least 60 FPS at 1920x1080 with 64 lights on.',
};
const approximation = 'Local lights cast no shadow yet. Tile lighting uses prepass '
  + 'coverage: transparent and displaced fur fragments use the nearest opaque or fur-base '
  + 'tile bounds; tiles without prepass coverage receive no local light.';
const selection = {
  camera,
  subjects: [
    { min: [-11, 0, -11], max: [11, 0, 11] },
    ...positions.map(([x, , z]) => ({
      min: [x - 0.35, 0, z - 0.35], max: [x + 0.35, 0.7, z + 0.35],
    })),
  ],
};

export default {
  id: 'many-lights',
  milestone: 'm28',
  engine: { msaa: 4, toneMapper: 'agx' },
  legend: '64 distinct bounded point lights over a 64-instance field on a fixed camera; '
    + 'measured at 1920x1080 with 60 warm-up and 300 measured frames. ' + approximation,
  measure,
  variants: ['on', 'off'].map((id) => ({
    id,
    description: (id === 'on'
      ? 'Many lights on: a fixed camera looks across an eight-by-eight field of 64 solid '
        + 'grey boxes on a broad grey floor, and each box is lit by its own distinct warm, '
        + 'cool or coloured point light. The 64 bounded pools dim with distance and end at '
        + 'each light range, covering the field in a recognisable spread of coloured '
        + 'illumination. All 64 lights are active for the 1920x1080 measurement. '
      : 'Many lights off: the same fixed camera and eight-by-eight field of 64 solid grey '
        + 'boxes on a broad grey floor under dim sun and ambient light only, with no local '
        + 'lights and no coloured pools. ') + approximation,
    evidence: captureEvidence('many-lights', id),
    judge: true,
    timeMs: 1000,
    selection,
  })),
  /**
   * Handoff: `default.build(engine, lib, variant)` exported by
   * `examples/lights/scenes/many-lights.js` attaches through `loadScenes()` in
   * `examples/lights/main.js`: build-lights.sh discovers it into scenes.json and
   * loadScenes() registers Map<scene.id, default>, with scene.id='many-lights' and
   * variant.id='on'|'off'. The fresh engine and frozen lib carry no retained state; the
   * on variant adds 64 point lights through the public Engine.addLight API.
   */
  async build(engine, lib, variant) {
    engine.setCamera(variant.selection.camera);
    engine.setBackground([0.02, 0.02, 0.02]);
    engine.setLight({ direction: [-0.5, -1, -0.5], color: [1, 1, 1], intensity: 0.2 });
    engine.setAmbient({ color: [1, 1, 1], intensity: 0.12 });
    const material = lib.pbr(engine, [0.55, 0.55, 0.55], { roughness: 0.8 });
    const mesh = (shape) => engine.addMesh({
      ...shape, subMeshes: [{ first: 0, count: shape.indices.length, material }],
    });
    const add = (meshId, position) => {
      const node = engine.addNode(null, position, [0, 0, 0, 1], [1, 1, 1]);
      engine.addMeshInstance(node, meshId);
    };
    add(mesh(lib.geometry.grid(20, 20, 1, 1)), [0, 0, 0]);
    const box = mesh(lib.geometry.roundedBox([0.35, 0.35, 0.35], 0.04, 4));
    for (const position of positions) add(box, position);
    if (variant.id !== 'on') return;
    positions.forEach(([x, , z], i) => {
      engine.addLight({
        kind: 'point', position: [x, 1.55, z], color: colorAt(i), intensity: 3, range: 3.2,
      });
    });
  },
};
