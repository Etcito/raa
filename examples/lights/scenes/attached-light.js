const states = new WeakMap();
const camera = { eye: [4, 4, 7], target: [0, 0.5, 0], fovYDeg: 45, near: 0.1, far: 100 };
const identity = [0, 0, 0, 1];
const scale = [1, 1, 1];
const offset = [0.8, 0.8, 0];
const desc = { kind: 'point', direction: [1, -1, 0], color: [1, 0.65, 0.25], intensity: 12, range: 4 };

export default {
  id: 'attached-light',
  milestone: 'm28',
  engine: { msaa: 4, toneMapper: 'agx' },
  variants: ['on', 'off'].map((id) => ({
    id,
    description: `Attached light ${id}: a grey parent block on a broad grey floor, with a small `
      + 'orange source beside the central block after deterministic translation and rotation. '
      + (id === 'on' ? 'A warm bounded pool illuminates the floor around the source. '
        : 'The source marker remains visible but local illumination is disabled. ')
      + 'Lifecycle PNGs show movement, reattachment, world-preserving detach and parent deletion, '
      + 'then light removal. Local lights cast no shadow yet. Transparent and displaced fur '
      + 'use the nearest opaque or fur-base prepass tile bounds; tiles without prepass coverage receive no local light.',
    judge: true,
    timeMs: 1000,
    selection: {
      camera,
      subjects: [
        { min: [-5, 0, -4], max: [5, 0, 4] },
        { min: [-1.8, 0, -0.3], max: [1.8, 0.7, 1.3] },
        { min: [-1.65, 1, -0.95], max: [1.65, 1.3, 1.15] },
      ],
    },
  })),
  async build(engine, lib, variant) {
    engine.setCamera(camera);
    engine.setBackground([0.025, 0.025, 0.025]);
    engine.setLight({ direction: [-0.5, -1, -0.5], color: [1, 1, 1], intensity: 0.3 });
    engine.setAmbient({ color: [1, 1, 1], intensity: 0.1 });
    const material = lib.pbr(engine, [0.55, 0.55, 0.55], { roughness: 0.7 });
    const source = lib.pbr(engine, [1, 0.65, 0.25], { emissive: [1, 0.65, 0.25] });
    const add = (shape, position, material) => {
      const mesh = engine.addMesh({
        ...shape, subMeshes: [{ first: 0, count: shape.indices.length, material }],
      });
      const node = engine.addNode(null, position, identity, scale);
      return { node, instance: engine.addMeshInstance(node, mesh) };
    };
    add(lib.geometry.grid(10, 8, 1, 1), [0, 0, 0], material);
    const a = add(lib.geometry.roundedBox([0.3, 0.35, 0.3], 0.04, 4), [-1.5, 0.35, 0], material);
    const b = add(lib.geometry.roundedBox([0.3, 0.35, 0.3], 0.04, 4), [1.5, 0.35, 0], material);
    const marker = add(lib.geometry.uvSphere(0.12, 16, 12), [-0.7, 1.15, 0], source);
    const light = variant.id === 'on' ? engine.addLight(desc) : null;
    if (light !== null) engine.attachLight(light, a.instance, offset);
    states.set(engine, { a, b, marker, light });
  },
  frame(engine, timeMs) {
    const { a, marker } = states.get(engine);
    const t = Math.max(0, Math.min(1, timeMs / 1000));
    const angle = t * Math.PI / 2;
    const x = -1.5 + 1.5 * t;
    engine.setTransform(a.node, [x, 0.35, 0], [0, Math.sin(angle / 2), 0, Math.cos(angle / 2)], scale);
    engine.setTransform(marker.node, [x + 0.8 * Math.cos(angle), 1.15, -0.8 * Math.sin(angle)], identity, scale);
  },
  /**
   * Handoff: `default.prepareCapture(engine, variant, { capture })` exported by
   * `examples/lights/scenes/attached-light.js` extends the existing main.js hook.
   * loadScenes() supplies Map<scene.id, default> and variant.id; build/frame share
   * WeakMap<engine, {a, b, marker, light}>. main.js passes capture(name), retains
   * its PNG files in the render envelope, and screenshots the final submitted frame.
   */
  async prepareCapture(engine, variant, { capture }) {
    const state = states.get(engine);
    const on = variant.id === 'on';
    const frames = [];
    const snapshot = (name, attached) => {
      const stats = engine.lightStats();
      const active = on && name !== 'light-removed' ? 1 : 0;
      if (stats.activeLights !== active || stats.attachedLights !== (on && attached ? 1 : 0)) {
        throw new Error(`${name}: unexpected light counts ${JSON.stringify(stats)}`);
      }
      engine.render(variant.timeMs);
      capture(`${name}.png`);
      frames.push({ name, stats });
    };
    const placeMarker = (position) => engine.setTransform(state.marker.node, position, identity, scale);
    this.frame(engine, 0);
    snapshot('initial', true);
    engine.setTransform(state.a.node, [0, 0.35, 0], identity, scale);
    placeMarker([0.8, 1.15, 0]);
    snapshot('translated', true);
    this.frame(engine, variant.timeMs);
    snapshot('rotated', true);
    if (on) engine.attachLight(state.light, state.b.instance, [0, 0.8, 0]);
    placeMarker([1.5, 1.15, 0]);
    snapshot('reattached', true);
    if (on) engine.detachLight(state.light);
    engine.setTransform(state.b.node, [-1.5, 0.35, 1], identity, scale);
    snapshot('detached', false);
    if (on) engine.attachLight(state.light, state.b.instance, [0, 0.8, 0]);
    placeMarker([-1.5, 1.15, 1]);
    snapshot('reattached-before-delete', true);
    engine.removeNode(state.b.node);
    snapshot('parent-deleted', false);
    if (on) engine.removeLight(state.light);
    engine.setVisible(state.marker.instance, false);
    snapshot('light-removed', false);
    state.light = on ? engine.addLight(desc) : null;
    if (on) engine.attachLight(state.light, state.a.instance, offset);
    engine.setVisible(state.marker.instance, true);
    this.frame(engine, variant.timeMs);
    snapshot('attached-final', true);
    return { frames };
  },
};
