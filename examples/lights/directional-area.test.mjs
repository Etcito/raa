import assert from 'node:assert/strict';
import test from 'node:test';
import { inView, variantProblems } from '../../web/common/lights-job.mjs';
import * as geometry from '../lookdev/lib/geometry.js';
import directional from './scenes/directional.js';
import area from './scenes/area.js';

// Record only public scene-building API calls; use the real geometry library so
// selection boxes are checked against the actual receiver vertices, without a GPU.
async function build(scene, id) {
  const calls = { meshes: [], nodes: [], instances: [], lights: [], materials: [] };
  const engine = {
    setCamera(value) { calls.camera = value; },
    setBackground(value) { calls.background = value; },
    setLight(value) { calls.sun = value; },
    setAmbient(value) { calls.ambient = value; },
    addMaterial(value) { calls.materials.push(value); return calls.materials.length - 1; },
    addMesh(value) { calls.meshes.push(value); return calls.meshes.length - 1; },
    addNode(parent, position, rotation, scale) {
      calls.nodes.push({ parent, position, rotation, scale });
      return calls.nodes.length - 1;
    },
    addMeshInstance(node, mesh) { calls.instances.push({ node, mesh }); },
    addLight(value) { calls.lights.push(value); return calls.lights.length; },
  };
  const lib = {
    geometry,
    pbr: (engine, color, opts) => engine.addMaterial({ ...opts, color, model: 'pbr' }),
  };
  await scene.build(engine, lib, scene.variants.find((variant) => variant.id === id));
  return calls;
}

for (const scene of [directional, area]) {
  test(`${scene.id} metadata supplies valid judged on/off selections and limitations`, () => {
    assert.equal(scene.id, scene === directional ? 'directional' : 'area');
    assert.equal(scene.milestone, 'm28');
    assert.deepEqual(scene.variants.map((variant) => variant.id), ['on', 'off']);
    for (const variant of scene.variants) {
      assert.deepEqual(variantProblems(variant), []);
      assert.match(variant.description, /Local lights cast no shadow yet/);
      assert.match(variant.description, /prepass coverage.*transparent and displaced fur/);
      assert.match(variant.description, /tiles without prepass coverage receive no local light/);
    }
    assert.match(scene.variants[0].description, scene === directional
      ? /consistent direction.*without distance dimming/ : /extent-dependent representative-point/);
  });

  test(`${scene.id} subjects bound all actual receivers and overlap the camera`, async () => {
    const calls = await build(scene, 'on');
    const { camera, subjects } = scene.variants[0].selection;
    assert.deepEqual(calls.camera, camera);
    assert.equal(calls.instances.length, subjects.length);
    for (const [i, { node, mesh }] of calls.instances.entries()) {
      const { position, rotation, scale } = calls.nodes[node];
      assert.deepEqual(rotation, [0, 0, 0, 1]);
      assert.deepEqual(scale, [1, 1, 1]);
      const { positions } = calls.meshes[mesh];
      const min = [Infinity, Infinity, Infinity];
      const max = [-Infinity, -Infinity, -Infinity];
      positions.forEach((value, j) => {
        const axis = j % 3;
        min[axis] = Math.min(min[axis], value + position[axis]);
        max[axis] = Math.max(max[axis], value + position[axis]);
      });
      for (let axis = 0; axis < 3; axis++) {
        assert.ok(Math.abs(min[axis] - subjects[i].min[axis]) < 1e-6);
        assert.ok(Math.abs(max[axis] - subjects[i].max[axis]) < 1e-6);
      }
      assert.ok(inView(camera, subjects[i]));
    }
  });

  test(`${scene.id} off preserves the scene and repeated builds are deterministic`, async () => {
    const on = await build(scene, 'on');
    const off = await build(scene, 'off');
    assert.equal(off.lights.length, 0);
    assert.ok(on.lights.length > 0);
    assert.deepEqual({ ...on, lights: [] }, off);
    assert.deepEqual(await build(scene, 'on'), on);
    for (const light of on.lights) {
      assert.equal(light.kind, scene.id);
      assert.equal(light.shadows, false);
      assert.ok(light.range > 0 && Number.isFinite(light.range));
      assert.ok(Math.hypot(...light.direction) > 0 && light.direction.every(Number.isFinite));
    }
  });
}

test('directional receivers have equal orientation at increasing distances beyond range', async () => {
  const calls = await build(directional, 'on');
  assert.equal(calls.lights.length, 1);
  const [light] = calls.lights;
  assert.deepEqual(light.direction, [0.4, -1, -0.5]);
  const distances = calls.nodes.slice(1).map(({ position }) =>
    Math.hypot(...position.map((value, i) => value - light.position[i])));
  assert.ok(distances.every((distance) => distance > light.range));
  assert.ok(distances[0] < distances[1] && distances[1] < distances[2]);
  assert.deepEqual(calls.meshes[1], calls.meshes[2]);
  assert.deepEqual(calls.meshes[2], calls.meshes[3]);
});

test('area comparison changes positive full extents with matching emission and fixed orientation', async () => {
  const { lights } = await build(area, 'on');
  assert.equal(lights.length, 2);
  const [small, wide] = lights;
  assert.deepEqual(lights.map((light) => light.size), [[0.5, 0.5], [1.2, 3.6]]);
  for (const light of lights) {
    assert.ok(light.size.every((extent) => Number.isFinite(extent) && extent > 0));
    assert.deepEqual(light.direction, [0, -1, 0]);
    assert.ok(light.position[1] > 0, 'the emitting +direction faces the floor');
  }
  assert.deepEqual({ ...small, position: wide.position, size: wide.size }, wide);
  assert.equal(small.position[1], wide.position[1]);
  assert.equal(small.position[2], wide.position[2]);
  assert.ok(small.position[0] < wide.position[0]);
});
