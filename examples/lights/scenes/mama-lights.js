const camera = { eye: [0, 6, 9], target: [0, 0.8, 0], fovYDeg: 45, near: 0.1, far: 100 };
const selection = { camera, subjects: [{ min: [-6, 0, -4], max: [6, 0, 4] }] };
const evidence = [
  {
    filename: 'mama-lights-on-soft-spot-illumination.png',
    label: 'A visible pool of spot illumination with a soft cone edge on a matte floor, produced by the loaded MAMA scene. Local lights cast no shadows yet.',
  },
  {
    filename: 'mama-lights-off-valid-floor-without-spot.png',
    label: 'The same valid matte-floor scene under its ambient and sun lighting, without the demonstrated spot illumination. Local lights cast no shadows yet.',
  },
];

export default {
  id: 'mama-lights',
  milestone: 'm03',
  engine: { msaa: 4, toneMapper: 'agx' },
  variants: ['on', 'off'].map((id, i) => ({
    id, description: evidence[i].label, judge: true, timeMs: 1000, selection, evidence: evidence[i],
  })),
  /**
   * M3_LIGHT_SCENE browser-build handoff: build-lights.sh discovers this metadata before
   * stageMamaLights supplies the scene and module trees. The browser build must resolve
   * these exact relative URLs (and MAMA's npm imports). No evaluator or registry is passed
   * into the page; loadProceduralScene creates its own owned light registry.
   */
  async build(engine, lib, variant) {
    engine.setCamera(variant.selection.camera);
    engine.setBackground([0.025, 0.025, 0.025]);
    engine.setLight({ direction: [-0.5, -1, -0.5], color: [1, 1, 1], intensity: 0.15 });
    engine.setAmbient({ color: [1, 1, 1], intensity: 0.08 });
    const material = lib.pbr(engine, [0.55, 0.55, 0.55], { roughness: 0.85 });
    const floor = lib.geometry.grid(12, 8, 1, 1);
    const mesh = engine.addMesh({
      ...floor, subMeshes: [{ first: 0, count: floor.indices.length, material }],
    });
    const node = engine.addNode(null, [0, 0, 0], [0, 0, 0, 1], [1, 1, 1]);
    engine.addMeshInstance(node, mesh);
    const [{ parseProceduralScene }, { loadProceduralScene }, response] = await Promise.all([
      import('../mama/procgen/scene-file.js'),
      import('../raa-sdfm/src/procedural.js'),
      fetch(new URL('../mama-lights.scene.json', import.meta.url)),
    ]);
    if (!response.ok) throw new Error(`mama-lights.scene.json: HTTP ${response.status}`);
    const scene = await loadProceduralScene(engine, null, await response.text(), {
      mama: async () => ({ parseProceduralScene }),
    });
    scene.setVisible(variant.id === 'on');
  },
};
