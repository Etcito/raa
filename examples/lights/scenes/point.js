const camera = { eye: [4, 3, 6], target: [0, 0.5, 0], fovYDeg: 45, near: 0.1, far: 100 };
const selection = {
  camera,
  subjects: [
    { min: [-4, 0, -4], max: [4, 0, 4] },
    { min: [-0.6, 0, -0.6], max: [0.6, 1.2, 0.6] },
  ],
};

export default {
  id: 'point',
  milestone: 'm28',
  engine: { msaa: 4, toneMapper: 'agx' },
  variants: ['on', 'off'].map((id) => ({
    id,
    description: `Neutral ${id} preparation scene: one solid grey rounded box on a broad grey `
      + 'floor against a dark neutral background, both clearly visible under sun and ambient light. '
      + 'Local point lighting is not enabled yet; local lights cast no shadow yet.',
    judge: true,
    timeMs: 1000,
    selection,
  })),
  async build(engine, lib, variant) {
    engine.setCamera(variant.selection.camera);
    engine.setBackground([0.04, 0.04, 0.04]);
    engine.setLight({ direction: [-0.5, -1, -0.5], color: [1, 1, 1], intensity: 2 });
    engine.setAmbient({ color: [1, 1, 1], intensity: 0.25 });
    const material = lib.pbr(engine, [0.6, 0.6, 0.6], { roughness: 0.65 });
    const shapes = [lib.geometry.grid(8, 8, 1, 1), lib.geometry.roundedBox([0.6, 0.6, 0.6], 0.06, 4)];
    for (const [i, shape] of shapes.entries()) {
      const mesh = engine.addMesh({ ...shape, subMeshes: [{ first: 0, count: shape.indices.length, material }] });
      const node = engine.addNode(null, [0, i * 0.6, 0], [0, 0, 0, 1], [1, 1, 1]);
      engine.addMeshInstance(node, mesh);
    }
  },
};
