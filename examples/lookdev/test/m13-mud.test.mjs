// node --test "examples/lookdev/test/*.test.mjs": the m13 mud set builds against a recording
// fake engine, adds white spray emitters and glossy mesh crowns at every pad, and its bank
// reaches the horizon. Proves counts and parameters only, not the look (GPU card judges it).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as geometry from '../lib/geometry.js';
import * as objects from '../lib/objects.js';
import scene, { mudSet } from '../scenes/m13-liquids-particles.js';

function fakeEngine() {
  const rec = { emitters: [], materials: [], meshes: [], instances: [], lights: [], orbit: null };
  const engine = new Proxy({}, {
    get(_, name) {
      if (name === 'addEmitter') return (e) => rec.emitters.push(e) - 1;
      if (name === 'addSurfaceMaterial') return (m) => rec.materials.push(m) - 1;
      if (name === 'addMesh') return (m) => rec.meshes.push(m) - 1;
      if (name === 'addMeshInstance') return (node, mesh) => rec.instances.push(mesh) - 1;
      if (name === 'addNode') return () => 0;
      if (name === 'setLight') return (l) => rec.lights.push(l);
      if (name === 'setOrbit') return (...a) => { rec.orbit = a; };
      return () => {};
    },
  });
  return { engine, rec };
}

const mud = scene.variants.find((v) => v.id === 'mud');

test('mud set spawns white spray at every pad', async () => {
  const { engine, rec } = fakeEngine();
  await scene.build(engine, { geometry, objects }, mud);
  const spray = rec.emitters.filter((e) => e.kind === 'splash');
  assert.ok(spray.length >= mudSet.pads.length * 2, `spray emitters: ${spray.length}`);
  for (const e of spray) {
    assert.ok(e.rate > 0 && e.maxParticles > 0);
    assert.ok(e.opacity > 0.5, `spray opacity ${e.opacity}`);
    for (const c of e.colors) assert.ok(Math.min(...c) > 0.85, `spray colour ${c}`);
  }
  for (const [x, z, r] of mudSet.pads) {
    const near = spray.filter((e) => Math.hypot(e.position[0] - x, e.position[2] - z) < r + 0.2);
    assert.ok(near.length > 0, `no spray at pad ${x}, ${z}`);
  }
});

test('mud set builds a glossy chocolate crown mesh at every pad', async () => {
  const { engine, rec } = fakeEngine();
  await scene.build(engine, { geometry, objects }, mud);
  const crown = rec.materials.findIndex((m) => m.color === mudSet.crownStyle.color);
  assert.ok(crown >= 0, 'no crown material');
  assert.ok(rec.materials[crown].roughness < 0.2);
  const crowns = rec.meshes.filter((m) => m.subMeshes[0].material === crown);
  assert.equal(crowns.length, mudSet.pads.length);
  for (const m of crowns) {
    let top = -Infinity;
    for (let i = 1; i < m.positions.length; i += 3) top = Math.max(top, m.positions[i]);
    assert.ok(top > 0.15, `crown top ${top}`);
    assert.ok(m.positions.every(Number.isFinite) && m.normals.every(Number.isFinite));
  }
});

test('mud bank reaches the horizon', async () => {
  const { engine, rec } = fakeEngine();
  await scene.build(engine, { geometry, objects }, mud);
  const span = Math.max(...rec.meshes.map((m) => Math.max(...m.positions.filter((_, i) => i % 3 === 0))));
  assert.ok(span >= 30, `bank half width ${span}`);
});

test('water and slime sets still build', async () => {
  for (const id of ['water', 'slime', 'splash']) {
    const { engine, rec } = fakeEngine();
    await scene.build(engine, { geometry, objects }, scene.variants.find((v) => v.id === id));
    assert.ok(rec.meshes.length > 0);
  }
});

test('mud sun lights the camera-facing side of the crowns', async () => {
  const { engine, rec } = fakeEngine();
  await scene.build(engine, { geometry, objects }, mud);
  // The last light set wins; the eye sits at target + (cos p sin y, sin p, cos p cos y).
  const d = rec.lights.at(-1).direction;
  const [, , yaw, pitch] = rec.orbit;
  const eye = [Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch), Math.cos(pitch) * Math.cos(yaw)];
  // Light travels along d, so a crown side facing the eye is lit when dot(eye, -d) > 0
  // (a backlit crown rendered grey-black, card 5064).
  const toSun = d.map((v) => -v);
  const facing = eye[0] * toSun[0] + eye[2] * toSun[2];
  assert.ok(facing > 0.2, `sun lights the camera side: ${facing}`);
  assert.ok(toSun[1] > 0.5, `sun elevation: ${toSun[1]}`);
});
