// S14: pure metadata and procedural builders; animation is added on the next card.
import { uvSphere, lathe, sweepTube, roundedBox, merge, withTangents } from '../lib/geometry.js';

const Q = [0, 0, 0, 1];
const ONE = [1, 1, 1];
const HEAD_Y = 1.8;
const BLUE = [0.035, 0.20, 0.72];
const CREAM = [1, 0.96, 0.87];
const yaw = a => [0, Math.sin(a / 2), 0, Math.cos(a / 2)];
const roll = a => [0, 0, Math.sin(a / 2), Math.cos(a / 2)];
const linear = c => c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;

function reshape(desc, scale, offset) {
  const { tangents, ...mesh } = desc;
  mesh.positions = desc.positions.map((v, i) => v * scale[i % 3] + offset[i % 3]);
  mesh.normals = desc.normals.slice();
  for (let i = 0; i < mesh.normals.length; i += 3) {
    const n = [0, 1, 2].map(k => mesh.normals[i + k] / scale[k]);
    mesh.normals.set(n.map(v => v / Math.hypot(...n)), i);
  }
  return withTangents(mesh);
}
const egg = (r, t = [0, 0, 0]) => reshape(uvSphere(1, 64, 40), r, t);

function select(desc, predicate) {
  const indices = [];
  for (let i = 0; i < desc.indices.length; i += 3) {
    const tri = Array.from(desc.indices.subarray(i, i + 3));
    const center = [0, 1, 2].map(k => tri.reduce((s, v) => s + desc.positions[v * 3 + k], 0) / 3);
    if (predicate(center)) indices.push(...tri);
  }
  return { ...desc, indices: new Uint32Array(indices) };
}

function upload(engine, mesh, materials) {
  const subMeshes = (mesh.subMeshes ?? [{ first: 0, count: mesh.indices.length }])
    .map((s, i) => ({ ...s, material: materials[Math.min(i, materials.length - 1)] }));
  return engine.addMesh({ ...mesh, subMeshes });
}

function plush(engine, color, length = 0.018) {
  return engine.addSurfaceMaterial({ function: 'raa.plush',
    params: { color, tipColor: color.map(c => c + (1 - c) * 0.08),
      sheenColor: color.map(c => c + (1 - c) * 0.25), furLength: length, furDensity: 12 },
    furLength: length, furDensity: 12, furDroop: 0.18, furShells: 32 });
}
function inflatable(engine, color) {
  return engine.addSurfaceMaterial({ function: 'raa.inflatable',
    params: { color, stripeWidth: 0, seamDepth: 0.12 },
    clearcoat: 1, clearcoatRoughness: 0.07 });
}

function characterMeshes(engine) {
  const blue = plush(engine, BLUE), neutral = plush(engine, [0.92, 0.92, 0.92]);
  const cream = plush(engine, CREAM, 0.012);
  const orange = inflatable(engine, [1, 0.49, 0.025]);
  const pink = inflatable(engine, [1, 0.025, 0.38]);
  const sole = inflatable(engine, [0.95, 0.93, 1]);
  const eye = engine.addSurfaceMaterial({ color: [0.012, 0.018, 0.035], roughness: 0.09,
    clearcoat: 1, clearcoatRoughness: 0.035 });
  // One pear silhouette. HEAD_Y is an actual profile ring; both parts retain its
  // original normals, so the rest pose has no neck pinch or stacked sphere seam.
  const profile = Array.from({ length: 61 }, (_, i) => {
    const y = 2.7 * i / 60;
    return [i === 0 || i === 60 ? 0 : 1.02 * Math.sin(Math.PI * i / 60) ** 0.65 * (1 - 0.14 * y), y];
  });
  const mass = reshape(lathe(profile, 96), [1, 1, 0.82], [0, 0, 0]);
  const lower = select(mass, p => p[1] < HEAD_Y);
  const upper = reshape(select(mass, p => p[1] >= HEAD_Y), ONE, [0, -HEAD_Y, 0]);
  // Belly is part of the body upload, never a separate instance. Its low cream
  // patch wraps around the underside, like the plush in both references.
  const belly = egg([0.69, 0.80, 0.27], [0, 0.86, 0.48]);
  const body = merge([lower, belly]);
  const flipper = egg([0.22, 0.57, 0.19], [0, -0.43, 0]);
  const foot = merge([egg([0.24, 0.13, 0.37], [0, -0.02, 0.16]),
    egg([0.255, 0.065, 0.385], [0, -0.105, 0.16])]);
  // The reference's six-panel beanie, with a small orange propeller.
  const palette = [[0.03, 0.72, 0.69], [1, 0.72, 0.025], [1, 0.035, 0.38]];
  const cap = select(uvSphere(1, 96, 32, { color: p => {
    const panel = Math.floor((Math.atan2(p[2], p[0]) + Math.PI) * 3 / Math.PI) % 3;
    return [...palette[panel].map(linear), 1];
  } }), p => p[1] >= 0);
  const hat = merge([reshape(cap, [0.69, 0.48, 0.60], [0, 0.57, 0]),
    egg([0.065, 0.11, 0.065], [0, 1.10, 0]),
    egg([0.54, 0.025, 0.075], [0, 1.20, 0])]);
  const scarfPath = Array.from({ length: 97 }, (_, i) => {
    const a = i / 96 * 2 * Math.PI;
    return [0.80 * Math.cos(a), 0.02 + 0.045 * Math.sin(a), 0.65 * Math.sin(a)];
  });
  const scarf = merge([sweepTube(scarfPath, 0.14, 16, { closed: true }),
    reshape(roundedBox([0.16, 0.38, 0.045], 0.04, 6), ONE, [-0.49, -0.32, 0.67])]);
  const scarfMat = plush(engine, [0.40, 0.88, 0.86], 0.005);
  return {
    body: [upload(engine, body, [blue, cream]), upload(engine, body, [neutral, cream])],
    head: [upload(engine, upper, [blue]), upload(engine, upper, [neutral])],
    flipper: [upload(engine, flipper, [blue]), upload(engine, flipper, [neutral])],
    foot: upload(engine, foot, [pink, sole]),
    beak: upload(engine, egg([0.20, 0.105, 0.25], [0, 0, 0.12]), [orange]),
    eye: upload(engine, egg([0.065, 0.085, 0.055]), [eye]),
    hat: upload(engine, hat, [inflatable(engine, [1, 1, 1]), orange, orange]),
    scarf: upload(engine, scarf, [scarfMat]),
  };
}

function buildPenguin(engine, meshes, position, tint, angle = 0, scale = 1) {
  const nodes = [], transforms = [];
  const node = (parent, t, r = Q, s = ONE) => {
    const handle = engine.addNode(parent, [0, 0, 0], Q, ONE);
    nodes.push(handle);
    transforms.push(...t, ...r, ...s);
    return handle;
  };
  const part = (parent, mesh, t, r = Q) => {
    const n = node(parent, t, r);
    return { node: n, instance: engine.addMeshInstance(n, mesh) };
  };
  const costume = tint ? 1 : 0;
  const root = node(null, position, yaw(angle), [scale, scale, scale]);
  // Body origin is the ground contact point. Shoulder and ankle offsets are
  // entirely in the child transforms; their meshes extend away from each joint.
  const body = part(root, meshes.body[costume], [0, 0, 0]);
  const head = part(body.node, meshes.head[costume], [0, HEAD_Y, 0]);
  const flippers = [-1, 1].map(side => part(body.node, meshes.flipper[costume],
    [side * 0.72, 1.53, 0], roll(side * 0.95)));
  const feet = [-1, 1].map(side => part(body.node, meshes.foot,
    [side * 0.35, 0.17, 0.19], yaw(side * 0.13)));
  part(head.node, meshes.beak, [0, 0.22, 0.55]);
  for (const side of [-1, 1]) part(head.node, meshes.eye,
    [side * 0.24, 0.43, 0.48], yaw(side * 0.24));
  part(head.node, meshes.hat, [0, 0, 0]);
  part(head.node, meshes.scarf, [0, 0, 0]);
  if (tint) for (const p of [body, head, ...flippers]) engine.setInstanceTint(p.instance, tint);
  engine.setTransforms(new Uint32Array(nodes), new Float32Array(transforms));
  // Explicit rig handles and rest transforms for the follow-up frame implementation.
  return { root, body, head, flippers, feet, nodes, transforms };
}

export default {
  id: 'm15-waddle',
  milestone: 'm15',
  title: 'Costumed plush penguins: rigid parts and costume tints',
  engine: { msaa: 4, toneMapper: 'agx' },
  variants: [
    ...Array.from({ length: 12 }, (_, n) => ({
      id: `f${String(n).padStart(2, '0')}`, timeMs: n * 100,
      description: `costumed plush penguin character look, waddle frame ${n}`,
      references: ['ref-1', 'ref-2'], judge: n === 0 || n === 6,
    })),
    { id: 'deform-on', timeMs: 300, judge: false, references: [],
      description: 'Penguin deformation enabled; rest pose until the animation task.' },
    { id: 'deform-off', timeMs: 300, judge: false, references: [],
      description: 'Penguin deformation disabled; rest pose until the animation task.' },
    { id: 'tints', timeMs: 0, judge: false, references: [],
      description: 'Blue hero and pink, blue and yellow costume tints, all four in view.' },
  ],
  async build(engine, lib, variant) {
    engine.setSky({ preset: 'day', zenith: [0.055, 0.28, 0.85],
      horizon: [0.4, 0.65, 0.95], ground: [0.32, 0.4, 0.55], intensity: 1,
      clouds: { coverage: 0.22, density: 0.65 } });
    engine.setLight({ direction: [-0.5, -0.8, -0.6], color: [1, 0.94, 0.85], intensity: 3.5 });
    engine.setAmbient({ color: [0.65, 0.79, 1], intensity: 0.45 });
    engine.setEnvironmentLighting({ diffuse: 0.7, specular: 1 });
    engine.setToneMapping({ exposureEv: -0.25, look: 'punchy' });
    engine.setBloom({ strength: 0.025, radius: 0.5 });
    const floorMat = inflatable(engine, [0.025, 0.48, 0.95]);
    const floor = upload(engine, roundedBox([9, 0.16, 10], 0.14, 6), [floorMat]);
    engine.addMeshInstance(engine.addNode(null, [0, -0.16, -4], Q, ONE), floor);
    // Sparse pink/yellow course edging echoes the references without competing
    // with the character silhouette or occluding the tint comparison.
    for (const side of [-1, 1]) {
      const m = inflatable(engine, side < 0 ? [1, 0.035, 0.42] : [1, 0.72, 0.025]);
      const rail = upload(engine, roundedBox([0.24, 0.24, 8], 0.22, 6), [m]);
      engine.addMeshInstance(engine.addNode(null, [side * 6.7, 0.12, -4], Q, ONE), rail);
    }
    const meshes = characterMeshes(engine);
    const all = variant.id === 'tints';
    const placements = all
      ? [[-4.35, 0, 0], [-1.45, 0, 0], [1.45, 0, 0], [4.35, 0, 0]]
      : [[0, 0, 0], [-2.7, 0, -3.4], [0, 0, -5.3], [2.7, 0, -3.4]];
    const tints = [null, [1, 0.20, 0.50, 1], [0.18, 0.57, 1, 1], [1, 0.80, 0.13, 1]];
    const rigs = placements.map((p, i) => buildPenguin(engine, meshes, p, tints[i],
      all ? -0.12 : [-0.16, 0.18, -0.1, -0.22][i]));
    engine.setCamera(all
      ? { eye: [0, 3.5, 12.8], target: [0, 1.35, 0], fovYDeg: 43, near: 0.1, far: 100 }
      : { eye: [0.5, 2.7, 7.3], target: [0, 1.48, -0.35], fovYDeg: 31, near: 0.1, far: 100 });
    return { rigs };
  },
};
