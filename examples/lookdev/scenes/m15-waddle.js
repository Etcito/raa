// S14: pure metadata and procedural builders; animation is added on the next card.
import { uvSphere, lathe, sweepTube, roundedBox, merge, withTangents } from '../lib/geometry.js';

const Q = [0, 0, 0, 1];
const ONE = [1, 1, 1];
const HEAD_Y = 1.8;
const BLUE = [0.018, 0.09, 0.48];
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

function plush(engine, color, length = 0.025) {
  return engine.addSurfaceMaterial({ function: 'raa.plush',
    params: { color, tipColor: color.map(c => c + (1 - c) * 0.025),
      sheenColor: color.map(c => c + (1 - c) * 0.10), furLength: length, furDensity: 28 },
    furLength: length, furDensity: 28, furDroop: 0.12, furShells: 32 });
}
function inflatable(engine, color) {
  return engine.addSurfaceMaterial({ function: 'raa.inflatable',
    params: { color, stripeWidth: 0, seamDepth: 0.025, seamCount: 6 },
    clearcoat: 1, clearcoatRoughness: 0.12 });
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
  const underside = p => p[1] < 0.60 + 0.10 * Math.cos(p[0] * 7);
  const body = merge([select(lower, p => !underside(p)), select(lower, underside), belly]);
  const flipper = egg([0.22, 0.57, 0.19], [0, -0.43, 0]);
  const foot = merge([egg([0.24, 0.13, 0.37], [0, -0.02, 0.16]),
    egg([0.255, 0.065, 0.385], [0, -0.105, 0.16])]);
  // The reference's six-panel beanie, with a small orange propeller.
  const palette = [[0.03, 0.72, 0.69], [1, 0.80, 0.025], [1, 0.035, 0.38]];
  const cap = select(uvSphere(1, 96, 32, { color: p => {
    const panel = Math.floor((Math.atan2(p[2], p[0]) + Math.PI) * 3 / Math.PI) % 3;
    return [...palette[panel].map(linear), 1];
  } }), p => p[1] >= 0);
  const hat = merge([reshape(cap, [0.69, 0.48, 0.60], [0, 0.57, 0]),
    egg([0.065, 0.11, 0.065], [0, 1.10, 0]),
    egg([0.66, 0.025, 0.085], [0, 1.20, 0])]);
  const scarfPath = Array.from({ length: 96 }, (_, i) => {
    const a = i / 96 * 2 * Math.PI;
    return [0.80 * Math.cos(a), 0.02 + 0.045 * Math.sin(a), 0.65 * Math.sin(a)];
  });
  const scarfPrint = p => {
    const x = p[0] * 13 + p[1] * 7, y = p[2] * 17 + p[1] * 11;
    const dx = x - Math.floor(x) - 0.5, dy = y - Math.floor(y) - 0.5;
    const fish = dx * dx / 0.14 + dy * dy / 0.045 < 1
      || (dx > 0.25 && Math.abs(dy) < (dx - 0.20) * 0.7);
    return [...(fish ? palette[Math.abs(Math.floor(x) + Math.floor(y)) % 3]
      : [0.63, 0.92, 0.91]).map(linear), 1];
  };
  const scarf = merge([sweepTube(scarfPath, 0.145, 20, { closed: true, color: scarfPrint }),
    reshape(roundedBox([0.17, 0.40, 0.045], 0.04, 16, { color: scarfPrint }),
      ONE, [-0.48, -0.34, -0.64]),
    reshape(roundedBox([0.13, 0.30, 0.04], 0.035, 12, { color: scarfPrint }),
      ONE, [-0.70, -0.23, -0.53])]);
  const scarfMat = plush(engine, [1, 1, 1], 0.003);
  return {
    body: [upload(engine, body, [blue, cream, cream]), upload(engine, body, [neutral, cream, cream])],
    head: [upload(engine, upper, [blue]), upload(engine, upper, [neutral])],
    flipper: [upload(engine, flipper, [blue]), upload(engine, flipper, [neutral])],
    foot: upload(engine, foot, [pink, sole]),
    beak: upload(engine, egg([0.20, 0.105, 0.25], [0, 0, 0.12]), [orange]),
    eye: upload(engine, egg([0.065, 0.085, 0.055]), [eye]),
    hat: upload(engine, hat, [plush(engine, [1, 1, 1], 0.003), orange, orange]),
    scarf: upload(engine, scarf, [scarfMat]),
  };
}

// Each outfit uses the same animated head/body attachment points as the hero.
function costumeMeshes(engine) {
  const cloth = c => plush(engine, c, 0.008);
  const gold = inflatable(engine, [1, 0.64, 0.015]);
  const green = cloth([0.22, 0.65, 0.025]);
  const pink = cloth([1, 0.035, 0.36]);
  const white = cloth(CREAM);
  const steel = engine.addSurfaceMaterial({ color: [0.27, 0.21, 0.20],
    metallic: 0.65, roughness: 0.32, clearcoat: 0.3 });
  const leather = engine.addSurfaceMaterial({ color: [0.17, 0.065, 0.025], roughness: 0.6 });
  const ring = (rx, rz, y, radius) => sweepTube(Array.from({ length: 64 }, (_, i) => {
    const a = i * Math.PI / 32;
    return [rx * Math.cos(a), y, rz * Math.sin(a)];
  }), radius, 16, { closed: true });
  const bow = merge([egg([0.28, 0.23, 0.12], [-0.24, 0.71, -0.35]),
    egg([0.28, 0.23, 0.12], [0.24, 0.71, -0.35]),
    egg([0.12, 0.13, 0.14], [0, 0.71, -0.39])]);
  const skirt = lathe([[0.70, 0.63], [1.02, 0.78], [0.87, 0.97], [0.72, 1.04]], 96);
  // Scalloped plush hem makes the skirt read as fabric, not another body sphere.
  const hem = Array.from({ length: 18 }, (_, i) => {
    const a = i * Math.PI / 9;
    return egg([0.15, 0.11, 0.15], [0.94 * Math.cos(a), 0.80, 0.78 * Math.sin(a)]);
  });
  const helmet = reshape(select(uvSphere(1, 64, 32), p => p[1] >= 0),
    [0.68, 0.49, 0.60], [0, 0.52, 0]);
  const horns = [-1, 1].map(side => sweepTube([
    [side * 0.55, 0.64, 0], [side * 0.78, 0.68, 0],
    [side * 0.91, 0.87, 0], [side * 0.92, 1.08, 0],
  ], 0.10, 16));
  const collar = Array.from({ length: 18 }, (_, i) => {
    const a = i * Math.PI / 9;
    return egg([0.19, 0.16, 0.16], [0.72 * Math.cos(a), 1.78, 0.59 * Math.sin(a)]);
  });
  const crown = Array.from({ length: 7 }, (_, i) => {
    const a = i * Math.PI * 2 / 7;
    return egg([0.09, 0.32, 0.08], [0.24 * Math.cos(a), 1.00 + (i % 2) * 0.12, 0.22 * Math.sin(a)]);
  });
  const pineapple = reshape(select(uvSphere(1, 64, 40, { color: p => {
    const a = Math.atan2(p[2], p[0]) * 5;
    const seam = Math.abs(Math.sin(a + p[1] * 16)) < 0.16
      || Math.abs(Math.sin(a - p[1] * 16)) < 0.16;
    return seam ? [0.65, 0.25, 0.005, 1] : [1, 0.80, 0.08, 1];
  } }), p => p[1] > -0.38), [0.73, 0.97, 0.63], [0, 0.01, 0]);
  return [
    { head: upload(engine, bow, [pink]), body: upload(engine,
      merge([reshape(skirt, [1, 1, 0.83], [0, 0, 0]), ...hem]), [pink]) },
    { head: upload(engine, merge([helmet, ...horns, ring(0.68, 0.60, 0.52, 0.045)]),
      [steel, white, white, steel]), body: upload(engine,
      merge([...collar, ring(0.83, 0.67, 1.14, 0.085),
        reshape(roundedBox([0.13, 0.11, 0.04], 0.025, 4), ONE, [0, 1.14, -0.76])]),
      [...collar.map(() => white), leather, gold]) },
    { head: upload(engine, merge([pineapple, ...crown]),
      [cloth([1, 1, 1]), ...crown.map(() => green)]) },
  ];
}

function buildPenguin(engine, meshes, position, tint, angle = 0, scale = 1, outfit = null) {
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
  if (outfit) {
    part(head.node, outfit.head, [0, 0, 0]);
    if (outfit.body) part(body.node, outfit.body, [0, 0, 0]);
  } else {
    part(head.node, meshes.hat, [0, 0, 0]);
    part(head.node, meshes.scarf, [0, 0, 0]);
  }
  if (tint) for (const p of [body, head, ...flippers]) engine.setInstanceTint(p.instance, tint);
  engine.setTransforms(new Uint32Array(nodes), new Float32Array(transforms));
  // Explicit rig handles and rest transforms for the follow-up frame implementation.
  return { root, body, head, flippers, feet, nodes, transforms };
}

function buildCourse(engine) {
  const blue = inflatable(engine, [0.015, 0.43, 0.92]);
  const pink = inflatable(engine, [0.96, 0.015, 0.29]);
  const yellow = inflatable(engine, [1, 0.65, 0.012]);
  const white = inflatable(engine, [0.92, 0.97, 1]);
  const mint = inflatable(engine, [0.035, 0.90, 0.40]);
  const place = (mesh, t, r = Q, s = ONE) =>
    engine.addMeshInstance(engine.addNode(null, t, r, s), mesh);
  const box = (half, radius, mat) => upload(engine, roundedBox(half, radius, 6), [mat]);
  const tile = box([2.05, 0.18, 1.48], 0.17, blue);
  const pad = box([2.05, 0.22, 1.48], 0.20, pink);
  const sea = box([13, 0.20, 19], 0.18, mint);
  place(sea, [0, -0.66, 10]);
  for (let row = 0; row < 10; row++) {
    const z = -3 + row * 3;
    for (let col = -1; col <= 1; col++) {
      place(row > 2 && row % 3 === 0 ? pad : tile, [col * 4.14, -0.18, z]);
    }
  }
  const rail = box([0.19, 0.18, 1.5], 0.17, yellow);
  const post = box([0.28, 0.55, 0.28], 0.25, yellow);
  for (const side of [-1, 1]) for (let i = 0; i < 9; i++) {
    place(rail, [side * 6.25, 0.65, -1.5 + i * 3]);
    place(post, [side * 6.25, 0.40, -3 + i * 3]);
  }
  // Inflated stepping discs beyond the foreground platform.
  const disc = upload(engine, egg([1.15, 0.24, 1.15]), [yellow]);
  const inset = upload(engine, egg([0.92, 0.12, 0.92]), [pink]);
  for (const side of [-1, 1]) for (let i = 0; i < 5; i++) {
    place(disc, [side * 8, -0.15, 3 + i * 4]);
    place(inset, [side * 8, 0.05, 3 + i * 4]);
  }
  const pillar = box([0.52, 2.0, 0.52], 0.48, blue);
  const hammer = upload(engine, merge([
    egg([1.05, 1.45, 1.05]), egg([1.13, 0.16, 1.13], [0, -1.30, 0]),
    egg([1.13, 0.16, 1.13], [0, 1.30, 0]),
    egg([1.065, 0.25, 1.065], [0, 0, 0]),
  ]), [pink, yellow, yellow, white]);
  const axle = box([1.8, 0.13, 0.13], 0.12, yellow);
  for (const side of [-1, 1]) for (let i = 0; i < 2; i++) {
    const z = 9 + i * 8;
    place(pillar, [side * 5.6, 1.8, z]);
    place(axle, [side * 4.5, 3.0, z]);
    place(hammer, [side * 3.3, 2.9, z], roll(side * 0.78));
  }
  const archPath = Array.from({ length: 49 }, (_, i) => {
    const a = i / 48 * Math.PI;
    return [5.5 * Math.cos(a), 2.2 + 5.5 * Math.sin(a), 25];
  });
  place(upload(engine, sweepTube(archPath, 0.36, 24), [yellow]), [0, 0, 0]);
  const crownBase = box([1.15, 0.16, 0.25], 0.12, yellow);
  place(crownBase, [0, 8.0, 25]);
  const point = upload(engine, lathe([[0, 0], [0.43, 0], [0.06, 1.1], [0, 1.16]], 24), [yellow]);
  for (const x of [-0.8, 0, 0.8]) place(point, [x, 8.1, 25], Q, [1, x === 0 ? 1.35 : 1, 1]);
  const banner = box([0.55, 1.3, 0.08], 0.06, pink);
  for (const side of [-1, 1]) {
    place(banner, [side * 6.5, 5.0, 24]);
    place(upload(engine, sweepTube([[0, 0, 0], [0, 7, 0]], 0.10, 12), [yellow]), [side * 7.1, 0, 24]);
  }
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
    engine.setSky({ preset: 'day', zenith: [0.10, 0.42, 0.95],
      horizon: [0.56, 0.77, 1], ground: [0.25, 0.34, 0.49], intensity: 0.85,
      clouds: { coverage: 0.36, density: 0.65 } });
    engine.setLight({ direction: [-0.45, -0.8, 0.35], color: [1, 0.96, 0.90], intensity: 2.7 });
    engine.setAmbient({ color: [0.72, 0.84, 1], intensity: 0.32 });
    engine.setEnvironmentLighting({ diffuse: 0.55, specular: 0.85 });
    engine.setToneMapping({ exposureEv: 0.10, look: 'punchy' });
    engine.setBloom({ strength: 0.065, radius: 0.65 });
    buildCourse(engine);
    const meshes = characterMeshes(engine);
    const outfits = costumeMeshes(engine);
    const all = variant.id === 'tints';
    const placements = all
      ? [[-4.35, 0, 0], [-1.45, 0, 0], [1.45, 0, 0], [4.35, 0, 0]]
      : [[0, 0, 0], [-3.1, 0, 5.7], [2.6, 0, 3.6], [-3.0, 0, 1.9]];
    const tints = [null, [1, 0.20, 0.50, 1], [0.18, 0.57, 1, 1], [1, 0.80, 0.13, 1]];
    const rigs = placements.map((p, i) => buildPenguin(engine, meshes, p, tints[i],
      all ? -0.12 : [-0.12, 0.18, -0.1, -0.22][i], 1, i ? outfits[i - 1] : null));
    engine.setCamera(all
      ? { eye: [0, 3.5, 12.8], target: [0, 1.35, 0], fovYDeg: 43, near: 0.1, far: 100 }
      : { eye: [1.2, 3.6, -7.4], target: [0, 2.0, 3.2], fovYDeg: 43, near: 0.1, far: 100 });
    return { rigs };
  },
};
