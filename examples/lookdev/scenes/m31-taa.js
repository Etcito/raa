// S14: thin geometry under a swaying orbit camera, with TAA toggled by the variant.
import { roundedBox, uvSphere } from '../lib/geometry.js';

const statesByEngine = new WeakMap();
const DARK = [0.05, 0.05, 0.06];
const Q = [0, 0, 0, 1];
const ONE = [1, 1, 1];

const variants = [
  {
    id: 'taa-off',
    taa: false,
    judge: true,
    references: [],
    description: 'Temporal anti-aliasing is off with no multisampling: the thin dark fence posts and rails against the bright sky and ground, the slanted white plank and the red ball show jagged stair-stepped pixel edges along their outlines.',
  },
  {
    id: 'taa-on',
    taa: true,
    judge: true,
    references: [],
    description: 'Temporal anti-aliasing is on: the thin dark fence posts and rails, the slanted white plank and the red ball have smooth anti-aliased outlines without stair steps; the thin posts stay continuous with no broken segments; the red ball is crisp with no smeared trail or doubled ghost copy behind it; the image is not blurred overall.',
  },
];

function place(engine, lib, mesh, color, t, r = Q) {
  const material = lib.pbr(engine, color, { roughness: 0.6, metallic: 0 });
  const id = engine.addMesh({ ...mesh, subMeshes: [{ first: 0, count: mesh.indices.length, material }] });
  const node = engine.addNode(null, t, r, ONE);
  engine.addMeshInstance(node, id);
  return node;
}

async function build(engine, lib, variant) {
  engine.setSky({ preset: 'day' });
  engine.setLight({ direction: [0.4, -0.8, -0.45], color: [1, 0.95, 0.88], intensity: 3 });
  engine.setAmbient({ color: [0.7, 0.8, 1], intensity: 0.2 });
  const box = (half, radius) => roundedBox(half, radius, 2);
  place(engine, lib, box([12, 0.1, 12], 0.05), [0.55, 0.55, 0.55], [0, -0.1, 0]);
  for (let i = 0; i < 11; i++) {
    place(engine, lib, box([0.03, 0.8, 0.03], 0.01), DARK, [-2.5 + 0.5 * i, 0.8, 0]);
  }
  for (const y of [0.5, 1.25]) {
    place(engine, lib, box([2.6, 0.025, 0.025], 0.01), DARK, [0, y, 0]);
  }
  const tilt = [0, 0, Math.sin(0.175), Math.cos(0.175)];
  place(engine, lib, box([1.2, 0.04, 0.3], 0.02), [0.92, 0.92, 0.92], [0, 0.35, 1.2], tilt);
  const ballNode = place(engine, lib, uvSphere(0.3, 48, 24), [0.85, 0.05, 0.04], [0, 0.3, 2.0]);
  statesByEngine.set(engine, { frame: 0, ballNode });
  engine.setCamera({ eye: [0, 1.6, 7], target: [0, 1.0, 0], fovYDeg: 40, near: 0.1, far: 200 });
  engine.setOrbit([0, 1.0, 0], Math.hypot(7, 0.6), 0, Math.atan2(0.6, 7));
  engine.setPassEnabled('taa', variant.taa);
}

// Motion follows the per-engine frame counter, never the time.
function frame(engine) {
  const state = statesByEngine.get(engine);
  if (!state) return;
  const { frame: n, ballNode } = state;
  const sign = Math.floor((n + 5) / 10) % 2 === 0 ? 1 : -1;
  engine.orbit(sign * 0.003, 0, 1);
  const x = -1.5 + 3.0 * (1 - Math.abs(((n / 60) % 2) - 1));
  engine.setTransforms(new Uint32Array([ballNode]), new Float32Array([x, 0.3, 2.0, 0, 0, 0, 1, 1, 1, 1]));
  state.frame = n + 1;
}

export default {
  id: 'm31-taa',
  milestone: 'm31',
  title: 'Temporal anti-aliasing over a swaying camera',
  engine: { msaa: 1, toneMapper: 'agx' },
  variants,
  build,
  frame,
};
