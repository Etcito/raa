// Bounded point preset (S20, RAA-LIGHTS-PAGE): three solid boxes at increasing distances
// from a warm key light and a cool fill light. Every local light has a finite range, so its
// pool of illumination dims with distance and reaches zero at the range limit; geometry
// outside the pools keeps only sun and ambient light. The off variant builds the same
// geometry under the same sun/ambient lighting and adds no local light.
//
// Handoff: `default.build(engine, lib, variant)` (and the optional
// `default.frame(engine, timeMs, variant)`) exported by `examples/lights/scenes/point.js`;
// `loadScenes()` in `examples/lights/main.js` registers Map<scene.id, default> from the
// built scenes.json and variant.id selects on/off. `build` receives a fresh engine and the
// frozen lookdev `lib`, so the scene keeps no cross-run state; the on variant adds bounded
// point lights through the public Engine.addLight/updateLight API and off adds none.
const camera = { eye: [4, 3, 6], target: [0, 0.6, 0], fovYDeg: 45, near: 0.1, far: 100 };
const boxes = [[-2, 0.45, 0], [0, 0.45, 0], [2, 0.45, 0]];
const selection = {
  camera,
  subjects: [
    { min: [-6, 0, -6], max: [6, 0, 6] },
    ...boxes.map(([x, y, z]) => ({ min: [x - 0.45, y - 0.45, z - 0.45], max: [x + 0.45, y + 0.45, z + 0.45] })),
  ],
};

export default {
  id: 'point',
  milestone: 'm28',
  engine: { msaa: 4, toneMapper: 'agx' },
  legend: 'Point lights: finite range; illumination fades to zero at the range limit. '
    + 'Local lights cast no shadow yet.',
  variants: ['on', 'off'].map((id) => ({
    id,
    description: id === 'on'
      ? 'Neutral bounded point lighting on: three solid grey boxes stand at increasing distances '
        + 'from a warm key and a cool fill point light on a broad grey floor; each light has a '
        + 'finite range, so its pool of illumination dims with distance and ends cleanly at the '
        + 'range limit, while the rest of the floor and boxes stay under sun and ambient light; '
        + 'local lights cast no shadow yet.'
      : 'Neutral bounded point lighting off: the same three solid grey boxes and broad grey floor '
        + 'under sun and ambient light only, with no local point lights; local lights cast no '
        + 'shadow yet.',
    judge: true,
    timeMs: 1000,
    selection,
  })),
  async build(engine, lib, variant) {
    engine.setCamera(variant.selection.camera);
    engine.setBackground([0.04, 0.04, 0.04]);
    engine.setLight({ direction: [-0.5, -1, -0.5], color: [1, 1, 1], intensity: 2 });
    engine.setAmbient({ color: [1, 1, 1], intensity: 0.25 });
    const material = lib.pbr(engine, [0.55, 0.55, 0.55], { roughness: 0.7 });
    const add = (shape, position) => {
      const mesh = engine.addMesh({
        ...shape, subMeshes: [{ first: 0, count: shape.indices.length, material }],
      });
      const node = engine.addNode(null, position, [0, 0, 0, 1], [1, 1, 1]);
      engine.addMeshInstance(node, mesh);
    };
    add(lib.geometry.grid(12, 12, 1, 1), [0, 0, 0]);
    for (const position of boxes) add(lib.geometry.roundedBox([0.45, 0.45, 0.45], 0.05, 4), position);
    if (variant.id !== 'on') return;
    // A legal update merges atomically onto the stored descriptor; the tests exercise the
    // refusal and capacity sides through the same public method.
    const key = engine.addLight({
      kind: 'point', position: [1.4, 1.5, 2.4], color: [1, 0.75, 0.5], intensity: 6, range: 4.5,
    });
    engine.updateLight(key, { intensity: 8 });
    engine.addLight({
      kind: 'point', position: [-2.6, 1, 0.4], color: [0.45, 0.7, 1], intensity: 5, range: 3,
    });
  },
};
