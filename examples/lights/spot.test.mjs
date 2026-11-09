import assert from 'node:assert/strict';
import test from 'node:test';
import { inView, variantProblems } from '../../web/common/lights-job.mjs';
import * as geometry from '../lookdev/lib/geometry.js';
import spot from './scenes/spot.js';

// Same public-API CPU fixture convention as point.test.mjs. This checks scene calls
// and the S17 angle/update contract, not GPU shading or a CPU/GPU comparison.
function lightEngine() {
  const lights = new Map();
  const meshes = [];
  const nodes = [];
  const instances = [];
  const settings = {};
  const validate = (light) => {
    assert.equal(light.kind, 'spot');
    const angles = light.innerAngleDeg === undefined ? [light.angleDeg] : [light.angleDeg, light.innerAngleDeg];
    if (angles.some((v) => !Number.isFinite(v) || v < 1 || v > 179) || angles[1] > angles[0]) {
      throw Object.assign(new Error('light.invalid_desc: invalid cone angles'), { code: 'light.invalid_desc' });
    }
    return structuredClone(light);
  };
  return {
    meshes, nodes, instances, settings,
    localLights: () => [...lights.values()].map((light) => structuredClone(light)),
    addLight(desc) {
      const light = validate(desc);
      const handle = lights.size + 1;
      lights.set(handle, light);
      return handle;
    },
    updateLight(handle, changes) {
      assert.ok(lights.has(handle));
      lights.set(handle, validate({ ...lights.get(handle), ...changes }));
    },
    setCamera(value) { settings.camera = value; },
    setBackground(value) { settings.background = value; },
    setLight(value) { settings.sun = value; },
    setAmbient(value) { settings.ambient = value; },
    addMaterial: (desc) => desc,
    addMesh(desc) { meshes.push(desc); return meshes.length - 1; },
    addNode(parent, position, rotation, scale) {
      nodes.push({ parent, position, rotation, scale });
      return nodes.length - 1;
    },
    addMeshInstance(node, mesh) { instances.push({ node, mesh }); },
  };
}

const lib = {
  geometry,
  pbr: (engine, color, opts = {}) => engine.addMaterial({ ...opts, color, model: 'pbr' }),
  flat: (engine, color) => engine.addMaterial({ color, model: 'flat' }),
};
const cosine = (fullAngle) => Math.cos(fullAngle * Math.PI / 360);
const innerAngle = (light) => light.innerAngleDeg ?? light.angleDeg * 0.8;
// Pure S11 cone reference; image acceptance belongs to the managed description judge.
function cone(light, dot) {
  const outer = cosine(light.angleDeg);
  const inner = cosine(innerAngle(light));
  if (inner <= outer) return dot >= outer ? 1 : 0;
  const t = Math.max(0, Math.min(1, (dot - outer) / (inner - outer)));
  return t * t * (3 - 2 * t);
}

async function build(variant = spot.variants[0]) {
  const engine = lightEngine();
  await spot.build(engine, lib, variant);
  return engine;
}

test('spot metadata selects both variants and describes softness and coverage limits', () => {
  assert.equal(spot.id, 'spot');
  assert.equal(spot.milestone, 'm28');
  assert.deepEqual(spot.variants.map((v) => v.id), ['on', 'off']);
  for (const variant of spot.variants) {
    assert.deepEqual(variantProblems(variant), []);
    assert.match(variant.description, /Local lights cast no shadow yet/);
    assert.match(variant.description, /prepass coverage.*transparent and displaced fur.*nearest opaque or fur-base/);
    assert.match(variant.description, /without prepass coverage receive no local light/);
  }
  assert.match(spot.variants[0].description, /smooth inner-to-outer transition/);
  assert.match(spot.variants[0].description, /authored 24-degree.*60-degree/);
  assert.match(spot.variants[0].description, /80-percent default \(48 degrees\)/);
  assert.match(spot.variants[1].description, /no coloured cone footprints/);
});

test('on authors two distinct spots; off preserves source/surface geometry and baseline light', async () => {
  const on = await build();
  const off = await build(spot.variants[1]);
  assert.deepEqual(on.settings, off.settings);
  assert.deepEqual(on.meshes, off.meshes);
  assert.deepEqual(on.nodes, off.nodes);
  assert.deepEqual(on.instances, off.instances);
  assert.equal(on.instances.length, 3);
  assert.equal(off.localLights().length, 0);
  const [authored, defaults] = on.localLights();
  assert.equal(on.localLights().length, 2);
  assert.equal(authored.innerAngleDeg, 24);
  assert.equal(Object.hasOwn(defaults, 'innerAngleDeg'), false, 'omission reaches addLight unchanged');
  assert.equal(innerAngle(defaults), 48);
  for (const light of [authored, defaults]) {
    assert.deepEqual(light.direction, [0, -1, 0]);
    assert.equal(light.angleDeg, 60);
    const radius = light.position[1] * Math.tan(light.angleDeg * Math.PI / 360);
    assert.ok(radius < 1.51);
    assert.ok(Math.hypot(radius, light.position[1]) < light.range, 'range includes the outer footprint');
  }
});

test('authored and omitted-inner cones have smooth finite transitions', async () => {
  for (const light of (await build()).localLights()) {
    const outer = cosine(light.angleDeg);
    const inner = cosine(innerAngle(light));
    const samples = [0, 0.25, 0.5, 0.75, 1].map((t) => cone(light, outer + t * (inner - outer)));
    samples.forEach((value, i) => assert.ok(Math.abs(value - [0, 0.15625, 0.5, 0.84375, 1][i]) < 1e-12));
    assert.equal(cone(light, 1), 1);
    assert.equal(cone(light, outer - 0.01), 0);
  }
});

test('equal angles accept a finite hard boundary; clearing the inner angle restores 80 percent', async () => {
  const engine = await build();
  engine.updateLight(1, { innerAngleDeg: 60 });
  const light = engine.localLights()[0];
  const boundary = cosine(60);
  assert.deepEqual([boundary - 0.001, boundary, boundary + 0.001].map((dot) => cone(light, dot)), [0, 1, 1]);
  engine.updateLight(1, { angleDeg: 40, innerAngleDeg: undefined });
  assert.equal(innerAngle(engine.localLights()[0]), 32);
});

test('invalid merged angle updates are atomic, including an explicit retained inner angle', async () => {
  const engine = await build();
  const before = engine.localLights();
  for (const changes of [
    { angleDeg: 20, intensity: 99 },
    { innerAngleDeg: 61, position: [9, 9, 9] },
    { innerAngleDeg: 0 }, { innerAngleDeg: NaN }, { angleDeg: Infinity }, { angleDeg: 180 },
  ]) {
    assert.throws(() => engine.updateLight(1, changes), { code: 'light.invalid_desc' });
    assert.deepEqual(engine.localLights(), before);
  }
  engine.updateLight(1, { angleDeg: 70 });
  assert.equal(engine.localLights()[0].innerAngleDeg, 24);
});

test('every source, receiving surface and footprint has matching in-view bounds', async () => {
  for (const variant of spot.variants) {
    const engine = await build(variant);
    const { camera, subjects } = variant.selection;
    assert.deepEqual(engine.settings.camera, camera);
    assert.equal(subjects.length, 5);
    engine.instances.forEach(({ node, mesh }, i) => {
      const { position, rotation, scale } = engine.nodes[node];
      assert.deepEqual(rotation, [0, 0, 0, 1]);
      assert.deepEqual(scale, [1, 1, 1]);
      const coords = [0, 1, 2].map((axis) => Array.from(engine.meshes[mesh].positions)
        .filter((_, k) => k % 3 === axis).map((v) => v + position[axis]));
      const box = { min: coords.map((v) => Math.min(...v)), max: coords.map((v) => Math.max(...v)) };
      assert.ok(inView(camera, box));
      box.min.forEach((v, axis) => assert.ok(Math.abs(v - subjects[i].min[axis]) < 1e-6));
      box.max.forEach((v, axis) => assert.ok(Math.abs(v - subjects[i].max[axis]) < 1e-6));
    });
    subjects.forEach((box) => assert.ok(inView(camera, box)));
    const invalid = structuredClone(variant);
    invalid.selection.subjects[1] = { min: [1000, 0, 0], max: [1001, 1, 1] };
    assert.match(variantProblems(invalid).join('\n'), /subjects\[1\] is outside/);
    assert.ok(inView(camera, { min: [-1000, 0, -1000], max: [1000, 0, 1000] }), 'enclosing floor overlaps');
  }
});
