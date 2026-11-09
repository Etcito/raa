import { captureEvidence } from '../../../web/common/lights-evidence.js';
const states = new WeakMap();
const camera = { eye: [0, 2, 9], target: [0, 2, 0], fovYDeg: 45, near: 0.1, far: 100 };
const bases = [[-2, 2.7, 0], [2, 2.7, 0], [-2, 1.1, 0], [2, 1.1, 0]];
const directions = [[1, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 1, 0]];
const full = [{ x: 0, y: 0, width: 1, height: 1 }];
const legend = '+X right: red; -X left: cyan; +Y down: green; -Y up: magenta. '
  + 'Zero motion: dark grey (0.1); full colour at 0.05 UV/frame.';

export default {
  id: 'motion',
  milestone: 'm28',
  engine: { msaa: 4, toneMapper: 'agx' },
  legend,
  variants: ['on', 'off'].map((id) => ({
    id,
    description: (id === 'on'
      ? 'Signed motion display: four rigid boxes move right, left, down and up, respectively '
        + 'red, cyan, green and magenta. Stationary floor, newly introduced central sphere '
        + 'on its first captured frame, and clear sky/background all have zero motion: dark grey. '
        + legend
      : 'Ordinarily shaded grey scene: four solid boxes and a central sphere above a broad '
        + 'stationary floor against a dark background. The motion diagnostic is disabled. ')
      + ' Local lights cast no shadow yet. Transparent and displaced fur use the nearest opaque '
      + 'or fur-base prepass tile bounds; tiles without prepass coverage receive no local light.',
    evidence: captureEvidence('motion', id),
    judge: true,
    timeMs: 128,
    passes: { disable: id === 'off' ? ['motion.display'] : [] },
    selection: {
      camera,
      subjects: [
        { min: [-5, 0, -3], max: [5, 0, 3] },
        ...bases.map((p, i) => ({
          min: p.map((v, k) => v + directions[i][k] * 0.6 - 0.35),
          max: p.map((v, k) => v + directions[i][k] * 0.6 + 0.35),
        })),
        { min: [-0.4, 0.7, -0.4], max: [0.4, 1.5, 0.4] },
      ],
    },
  })),
  async build(engine, lib, variant) {
    engine.setCamera(camera);
    engine.setBackground([0.04, 0.04, 0.04]);
    engine.setLight({ direction: [-0.5, -1, -0.5], color: [1, 1, 1], intensity: 2 });
    engine.setAmbient({ color: [1, 1, 1], intensity: 0.25 });
    engine.setPassEnabled('motion.display', !variant.passes.disable.includes('motion.display'));
    const material = lib.pbr(engine, [0.6, 0.6, 0.6], { roughness: 0.65 });
    const mesh = (shape) => engine.addMesh({
      ...shape, subMeshes: [{ first: 0, count: shape.indices.length, material }],
    });
    const add = (meshId, position) => {
      const node = engine.addNode(null, position, [0, 0, 0, 1], [1, 1, 1]);
      engine.addMeshInstance(node, meshId);
      return node;
    };
    add(mesh(lib.geometry.grid(10, 6, 1, 1)), [0, 0, 0]);
    const box = mesh(lib.geometry.roundedBox([0.35, 0.35, 0.35], 0.04, 4));
    const nodes = bases.map((position) => add(box, position));
    const sphere = mesh(lib.geometry.uvSphere(0.4, 24, 16));
    states.set(engine, { nodes, add, sphere, spawned: null });
  },
  frame(engine, timeMs) {
    const amount = 0.6 * Math.max(-1, Math.min(1, (timeMs - 112) / 16));
    states.get(engine).nodes.forEach((node, i) => {
      const position = bases[i].map((v, k) => v + directions[i][k] * amount);
      engine.setTransform(node, position, [0, 0, 0, 1], [1, 1, 1]);
    });
  },
  /**
   * Handoff: `default.prepareCapture(engine, variant, { width, height })` in
   * `examples/lights/scenes/motion.js` attaches after runScene in examples/lights/main.js.
   * Existing loadScenes() passes the scene.id/variant.id registry; build/frame share
   * WeakMap<engine, {nodes, add, sphere, spawned}>. The hook submits the final frame;
   * main.js retains it for screenshot without another render/probe advancing history.
   */
  async prepareCapture(engine, variant, { width, height }) {
    const state = states.get(engine);
    if (state.spawned !== null) engine.removeNode(state.spawned);
    state.spawned = null;
    const frames = [];
    const submit = (timeMs, views) => {
      this.frame(engine, timeMs, variant);
      engine.render(timeMs);
      frames.push({ timeMs, views, spawned: state.spawned !== null });
    };
    engine.resize(Math.max(1, Math.floor(width / 2)), Math.max(1, Math.floor(height / 2)));
    submit(0, 1);
    submit(16, 1);
    engine.setViewports([{ ...full[0], width: 0.5 }, { ...full[0], x: 0.5, width: 0.5 }]);
    engine.setViewportCamera(1, { ...camera, eye: [1, 2, 9] });
    submit(32, 2);
    submit(48, 2);
    engine.resize(width, height);
    submit(64, 2);
    submit(80, 2);
    engine.setViewports(full);
    submit(96, 1);
    submit(112, 1);
    await new Promise((resolve) => setTimeout(resolve, 0));
    state.spawned = state.add(state.sphere, [0, 1.1, 0]);
    submit(variant.timeMs, 1);
    return { frames, size: [width, height] };
  },
};
