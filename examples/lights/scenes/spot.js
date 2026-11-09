import { captureEvidence } from '../../../web/common/lights-evidence.js';
const camera = { eye: [0, 6, 9], target: [0, 0.8, 0], fovYDeg: 45, near: 0.1, far: 100 };
const sources = [
  { position: [-1.9, 2.6, 0], color: [1, 0.72, 0.4], innerAngleDeg: 24 },
  { position: [1.9, 2.6, 0], color: [0.4, 0.7, 1] },
];
const selection = {
  camera,
  subjects: [
    { min: [-6, 0, -4], max: [6, 0, 4] },
    ...sources.map(({ position }) => ({
      min: position.map((v) => v - 0.12),
      max: position.map((v) => v + 0.12),
    })),
    ...sources.map(({ position: [x] }) => ({ min: [x - 1.51, 0, -1.51], max: [x + 1.51, 0, 1.51] })),
  ],
};
const approximation = 'Local lights cast no shadow yet. Tile selection uses prepass coverage: '
  + 'transparent and displaced fur fragments use the nearest opaque or fur-base surface bounds; '
  + 'tiles without prepass coverage receive no local light.';

export default {
  id: 'spot',
  milestone: 'm28',
  engine: { msaa: 4, toneMapper: 'agx' },
  legend: 'Left warm spot: 24-degree inner / 60-degree outer full cone. '
    + 'Right cool spot: omitted inner angle defaults to 80% (48 degrees). ' + approximation,
  variants: ['on', 'off'].map((id) => ({
    id,
    description: (id === 'on'
      ? 'Two separate cone footprints on a broad matte grey floor below two small source markers: '
        + 'warm orange on the left and cool blue on the right. Both have a bright inner region '
        + 'and a smooth inner-to-outer transition fading into the dim floor. The left spot has '
        + 'an authored 24-degree inner and 60-degree outer full cone; the right omits the inner '
        + 'angle, giving the 80-percent default (48 degrees) and a narrower soft rim. '
      : 'The same broad matte grey floor and two small warm-left and cool-right source markers '
        + 'under dim sun and ambient light only. Local spots are off, with no coloured cone footprints. ')
      + approximation,
    evidence: captureEvidence('spot', id),
    judge: true,
    timeMs: 1000,
    selection,
  })),
  /**
   * Handoff: `default.build(engine, lib, variant)` exported by
   * `examples/lights/scenes/spot.js` follows the landed loadScenes() handoff in
   * examples/lights/main.js: build-lights.sh discovers it in scenes.json and loadScenes()
   * attaches it to Map<scene.id, default>, using scene.id='spot' and variant.id='on'|'off'.
   * The fresh engine and frozen lib carry no shared scene state; addLight(desc) returns
   * engine-owned handles. No page wiring or additional state/key registry is needed.
   */
  async build(engine, lib, variant) {
    engine.setCamera(variant.selection.camera);
    engine.setBackground([0.025, 0.025, 0.025]);
    engine.setLight({ direction: [-0.5, -1, -0.5], color: [1, 1, 1], intensity: 0.15 });
    engine.setAmbient({ color: [1, 1, 1], intensity: 0.08 });
    const floor = lib.pbr(engine, [0.55, 0.55, 0.55], { roughness: 0.85 });
    const add = (shape, position, material) => {
      const mesh = engine.addMesh({
        ...shape, subMeshes: [{ first: 0, count: shape.indices.length, material }],
      });
      const node = engine.addNode(null, position, [0, 0, 0, 1], [1, 1, 1]);
      engine.addMeshInstance(node, mesh);
    };
    add(lib.geometry.grid(12, 8, 1, 1), [0, 0, 0], floor);
    for (const source of sources) {
      add(lib.geometry.uvSphere(0.12, 16, 12), source.position, lib.flat(engine, source.color));
      if (variant.id === 'on') {
        engine.addLight({
          kind: 'spot', direction: [0, -1, 0], intensity: 24, range: 6, angleDeg: 60, ...source,
        });
      }
    }
  },
};
