import { test } from 'node:test';
import assert from 'node:assert/strict';
import scene from '../scenes/m14-camera-effects.js';
import * as geometry from '../lib/geometry.js';
import { m14WorldJudgeCheck } from '../../../proofs/m14-world-judge.mjs';

async function build() {
  const record = { materials: [], meshes: [], nodes: [], instances: [], env: null };
  const engine = new Proxy({}, { get(_, name) {
    if (name === 'addSurfaceMaterial') return m => record.materials.push(m) - 1;
    if (name === 'addMesh') return m => record.meshes.push(m) - 1;
    if (name === 'addNode') return (_, t, r, s) => record.nodes.push({ t, r, s }) - 1;
    if (name === 'addMeshInstance') return (node, mesh) => record.instances.push({ node, mesh });
    if (name === 'applyEnvironment') return e => { record.env = e; };
    return () => {};
  } });
  const variant = scene.variants.find(v => v.id === 'world-neon');
  // No daytime object library: accidentally re-entering shared obstacles must fail.
  await scene.build(engine, { geometry }, variant);
  return { ...record, settings: variant.settings };
}

test('night radiance and log-space grading preserve dark violet shadows', async () => {
  const { env, settings } = await build();
  const luma = c => c.reduce((v, x, i) => v + x * [0.2126, 0.7152, 0.0722][i], 0);
  assert.ok(luma(env.sky.zenith) * env.sky.intensity < 0.012);
  assert.ok(luma(env.sky.horizon) * env.sky.intensity < 0.035);
  assert.ok(luma(settings.fog.color) < 0.035);
  assert.ok(settings.fog.start >= 24 && settings.fog.density <= 0.002);
  assert.equal(env.sun.intensity, 0, 'solar HG glow must not brighten the night');
  // The renderer soft-lights log-encoded values: values above 0.5 brighten blacks.
  // Test a shadow probe against the grade, not just the palette's RGB literals.
  const grade = env.post.grading;
  const shadow = 0.01;
  const logStops = 16; // grading.wgsl LOG_STOPS
  const encoded = Math.log2(shadow / 0.18) * grade.contrast / logStops + 0.5;
  for (const tint of grade.shadowsTint ?? [0.5, 0.5, 0.5]) {
    const soft = (1 - 2 * tint) * encoded ** 2 + 2 * tint * encoded;
    const decoded = 0.18 * 2 ** ((soft - 0.5) * logStops);
    assert.ok(decoded <= shadow, `grade lifts a dark surface to ${decoded}`);
  }
  assert.ok(env.post.bloom.strength >= 0.15);
  assert.ok(settings.ssr.intensity >= 1);
});

test('night architecture has valid checker submeshes and reflective luminous tiles', async () => {
  const { meshes, materials, nodes, instances } = await build();
  assert.ok(materials.some(m => m.roughness <= 0.15 && m.metallic >= 0.35));
  assert.ok(materials.filter(m => Math.max(...(m.emissive ?? [0])) > 1).length >= 2);
  assert.ok(materials.every(m => !m.function && !m.furLength && !m.transmission));
  assert.ok(meshes.some(m => m.subMeshes.length === 2), 'checker material boundaries');
  assert.ok(nodes.some(n => n.t[1] >= 3.4 && n.t[2] > -10), 'overhead foreground pipe');
  for (const mesh of meshes) {
    const count = mesh.positions.length / 3;
    assert.equal(mesh.normals.length, mesh.positions.length);
    assert.ok(mesh.positions.every(Number.isFinite));
    assert.ok(mesh.normals.every(Number.isFinite));
    assert.ok(mesh.indices.every(i => i < count));
    let end = 0;
    for (const sub of mesh.subMeshes) {
      assert.equal(sub.first, end);
      assert.equal(sub.count % 3, 0);
      assert.ok(materials[sub.material]);
      end += sub.count;
    }
    assert.equal(end, mesh.indices.length);
  }
  for (const instance of instances) {
    assert.ok(meshes[instance.mesh]);
    assert.ok(nodes[instance.node]);
  }
});

test('owner exceptions preserve the neon judge gate and require recorded judgments', () => {
  const read = neon => path => ({ judge: path.endsWith('--world-neon.json')
    ? neon : { exit: 1, score: 4, minScore: 6 } });
  assert.equal(m14WorldJudgeCheck(read({ exit: 0, score: 6, minScore: 6 })).ok, true);
  assert.equal(m14WorldJudgeCheck(read({ exit: 1, score: 4, minScore: 6 })).ok, false);
  assert.equal(m14WorldJudgeCheck(read({ exit: 0, score: 4, minScore: 4 })).ok, false);
  assert.equal(m14WorldJudgeCheck(read(null)).ok, false);
  assert.equal(m14WorldJudgeCheck(() => { throw new Error('missing'); }).ok, false);
});
