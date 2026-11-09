const camera = { eye: [7, 9, 13], target: [0, 0, 0], fovYDeg: 45, near: 0.1, far: 100 };
const boxes = [[-3.2, 0.4, 1.5], [3.2, 0.4, 1.5]];
const selection = {
  camera,
  subjects: [
    { min: [-7, 0, -4], max: [7, 0, 4] },
    ...boxes.map(([x, y, z]) => ({ min: [x - 0.4, y - 0.4, z - 0.4], max: [x + 0.4, y + 0.4, z + 0.4] })),
  ],
};
const limitations = ' Local lights cast no shadow yet. Tile lighting uses prepass coverage: '
  + 'transparent and displaced fur fragments use the nearest opaque or fur-base tile bounds; '
  + 'tiles without prepass coverage receive no local light.';

export default {
  id: 'area',
  milestone: 'm28',
  engine: { msaa: 4, toneMapper: 'agx' },
  legend: 'Area representative points: left 0.5 × 0.5 m; right 1.2 × 3.6 m full extents.'
    + limitations,
  variants: ['on', 'off'].map((id) => ({
    id,
    description: (id === 'on'
      ? 'Rectangular area lighting on: a broad grey floor with two matching grey boxes against '
        + 'a dark background. Two downward warm lights at equal height and intensity show '
        + 'extent-dependent representative-point illumination: a compact pool on the left from '
        + 'a 0.5 by 0.5 metre rectangle and a visibly wider pool along world X on the right from '
        + 'a 1.2 by 3.6 metre rectangle. These are full extents and an emitter approximation.'
      : 'Rectangular area lighting off: the same two grey boxes and broad grey floor under '
        + 'dim sun and ambient light only, without the two warm area-light pools.')
      + limitations,
    judge: true,
    timeMs: 1000,
    selection,
  })),
  /**
   * Handoff: `default.build(engine, lib, variant)` exported by
   * `examples/lights/scenes/area.js` attaches through `loadScenes()` in
   * `examples/lights/main.js`: Map<scene.id, default>, variant.id selects on/off.
   * The page passes a fresh engine and frozen lookdev lib; no scene state is retained.
   */
  async build(engine, lib, variant) {
    engine.setCamera(variant.selection.camera);
    engine.setBackground([0.025, 0.025, 0.025]);
    engine.setLight({ direction: [-0.5, -1, -0.5], color: [1, 1, 1], intensity: 0.1 });
    engine.setAmbient({ color: [1, 1, 1], intensity: 0.08 });
    const material = lib.pbr(engine, [0.55, 0.55, 0.55], { roughness: 0.85 });
    const add = (shape, position) => {
      const mesh = engine.addMesh({
        ...shape, subMeshes: [{ first: 0, count: shape.indices.length, material }],
      });
      const node = engine.addNode(null, position, [0, 0, 0, 1], [1, 1, 1]);
      engine.addMeshInstance(node, mesh);
    };
    add(lib.geometry.grid(14, 8, 1, 1), [0, 0, 0]);
    for (const position of boxes) add(lib.geometry.roundedBox([0.4, 0.4, 0.4], 0.05, 4), position);
    if (variant.id !== 'on') return;
    // For d = -Y, the deterministic basis is tangent -Z, bitangent +X; no authored roll.
    for (const [x, size] of [[-3.2, [0.5, 0.5]], [3.2, [1.2, 3.6]]]) {
      engine.addLight({
        kind: 'area', position: [x, 1.4, -0.8], direction: [0, -1, 0], size,
        color: [1, 0.8, 0.55], intensity: 5, range: 4.5, shadows: false,
      });
    }
  },
};
