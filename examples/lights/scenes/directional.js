const camera = { eye: [7, 5, 10], target: [0, 0.6, 0], fovYDeg: 45, near: 0.1, far: 100 };
const boxes = [[-3, 0.5, 0], [0, 0.5, 0], [3, 0.5, 0]];
const selection = {
  camera,
  subjects: [
    { min: [-6, 0, -4], max: [6, 0, 4] },
    ...boxes.map(([x, y, z]) => ({ min: [x - 0.5, y - 0.5, z - 0.5], max: [x + 0.5, y + 0.5, z + 0.5] })),
  ],
};
const limitations = ' Local lights cast no shadow yet. Tile lighting uses prepass coverage: '
  + 'transparent and displaced fur fragments use the nearest opaque or fur-base tile bounds; '
  + 'tiles without prepass coverage receive no local light.';

export default {
  id: 'directional',
  milestone: 'm28',
  engine: { msaa: 4, toneMapper: 'agx' },
  legend: 'Directional: matching faces receive the same incident direction without distance dimming.'
    + limitations,
  variants: ['on', 'off'].map((id) => ({
    id,
    description: (id === 'on'
      ? 'Directional lighting on: three identical grey boxes across a broad grey floor receive '
        + 'warm light from one consistent direction. Matching top and front faces remain similarly '
        + 'lit across the row without distance dimming, despite increasing distance from the '
        + 'light record position. The background is dark.'
      : 'Directional lighting off: the same three grey boxes and broad grey floor under dim '
        + 'sun and ambient light, with no local directional light, against a dark background.')
      + limitations,
    judge: true,
    timeMs: 1000,
    selection,
  })),
  /**
   * Handoff: `default.build(engine, lib, variant)` exported by
   * `examples/lights/scenes/directional.js` attaches through `loadScenes()` in
   * `examples/lights/main.js`: Map<scene.id, default>, variant.id selects on/off.
   * The page passes a fresh engine and frozen lookdev lib; no scene state is retained.
   */
  async build(engine, lib, variant) {
    engine.setCamera(variant.selection.camera);
    engine.setBackground([0.025, 0.025, 0.025]);
    engine.setLight({ direction: [-0.5, -1, -0.5], color: [1, 1, 1], intensity: 0.15 });
    engine.setAmbient({ color: [1, 1, 1], intensity: 0.12 });
    const material = lib.pbr(engine, [0.55, 0.55, 0.55], { roughness: 0.85 });
    const add = (shape, position) => {
      const mesh = engine.addMesh({
        ...shape, subMeshes: [{ first: 0, count: shape.indices.length, material }],
      });
      const node = engine.addNode(null, position, [0, 0, 0, 1], [1, 1, 1]);
      engine.addMeshInstance(node, mesh);
    };
    add(lib.geometry.grid(12, 8, 1, 1), [0, 0, 0]);
    for (const position of boxes) add(lib.geometry.roundedBox([0.5, 0.5, 0.5], 0.05, 4), position);
    if (variant.id !== 'on') return;
    // Position and the deliberately short range do not affect directional evaluation.
    engine.addLight({
      kind: 'directional', position: [-4, 3, 2], direction: [0.4, -1, -0.5],
      color: [1, 0.8, 0.55], intensity: 2.5, range: 1, shadows: false,
    });
  },
};
