import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
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

// Analytical colour probes, not capture evidence. Read the production AgX
// matrices and apply its curve/OETF after the scene's neutral grading settings.
// The GPU image check remains responsible for the whole frame and reference.
const tonemap = readFileSync(new URL('../../../crates/raa-tonemap/src/tonemap.wgsl', import.meta.url), 'utf8');
const matrix = name => {
  const body = tonemap.match(new RegExp(`const ${name} = mat3x3<f32>\\(([\\s\\S]*?)\\n\\);`))[1];
  const columns = [...body.matchAll(/vec3<f32>\(([^)]+)\)/g)]
    .map(m => m[1].split(',').map(Number));
  return [0, 1, 2].map(i => columns.map(c => c[i]));
};
const inset = matrix('AGX_INSET'), outset = matrix('AGX_OUTSET');
const mul = (m, c) => m.map(row => row.reduce((s, v, i) => s + v * c[i], 0));
const luma = c => c.reduce((s, v, i) => s + v * [0.2126, 0.7152, 0.0722][i], 0);
const clamp = x => Math.max(0, Math.min(1, x));
function displayProbe(rgb, post) {
  const grade = post.grading;
  let c = rgb.map(x => 0.18 * (Math.max(x, 1e-9) / 0.18) ** grade.contrast);
  const y = luma(c);
  c = c.map(x => y + (x - y) * grade.saturation);
  const sat = clamp((Math.max(...c) - Math.min(...c)) / Math.max(...c));
  const y2 = luma(c);
  c = c.map(x => Math.max(0, y2 + (x - y2) * (1 + grade.vibrance * (1 - sat)))
    * 2 ** post.toneMapping.exposureEv);
  let encoded = mul(inset, c).map(v => {
    const x = clamp((Math.log2(Math.max(v, 1e-10)) + 12.47393) / 16.499999);
    return clamp(15.5 * x ** 6 - 40.14 * x ** 5 + 31.96 * x ** 4
      - 6.868 * x ** 3 + 0.4298 * x ** 2 + 0.1191 * x - 0.00232);
  });
  if (post.toneMapping.look === 'punchy') {
    const y = luma(encoded);
    encoded = encoded.map(x => y + 1.4 * (x ** 1.35 - y));
  }
  return mul(outset, encoded).map(x => {
    const linear = clamp(x) ** 2.2;
    return linear <= 0.0031308 ? linear * 12.92 : 1.055 * linear ** (1 / 2.4) - 0.055;
  });
}

test('bright magenta and cyan materials retain chroma through grading and AgX', async t => {
  const { materials, env } = await build();
  const emitters = materials.filter(m => Math.max(...(m.emissive ?? [0])) > 1
    && Math.min(...m.color) < 0.2);
  assert.ok(emitters.length >= 4, 'tile and architecture emitters of both hues');
  let minimum = 1;
  for (const m of emitters) for (const radiance of [1, 1 + env.post.bloom.strength, 1.85]) {
    // Include neutral reflected light to avoid testing only ideal pure emission.
    const rgb = displayProbe(m.emissive.map(x => x * radiance + 0.1), env.post);
    const value = Math.max(...rgb), saturation = 1 - Math.min(...rgb) / value;
    assert.ok(value > 0.6, `emitter must remain luminous: ${rgb}`);
    assert.ok(saturation >= 0.558 * 0.9, `emitter bleached after tonemap: ${rgb}`);
    minimum = Math.min(minimum, saturation);
  }
  t.diagnostic(`Minimum analytical emitter saturation: ${minimum.toFixed(3)}`);
});

test('owner acceptance includes neon and requires recorded judgments at minimum 6', () => {
  const read = neon => path => ({ judge: path.endsWith('--world-neon.json')
    ? neon : { exit: 1, score: 4, minScore: 6 } });
  assert.equal(m14WorldJudgeCheck(read({ exit: 0, score: 6, minScore: 6 })).ok, true);
  assert.equal(m14WorldJudgeCheck(read({ exit: 1, score: 4, minScore: 6 })).ok, true);
  assert.equal(m14WorldJudgeCheck(read({ exit: 1, score: 5, minScore: 6 })).ok, true);
  assert.equal(m14WorldJudgeCheck(read({ exit: -1, score: 4, minScore: 6 })).ok, false);
  assert.equal(m14WorldJudgeCheck(read({ exit: 1, score: null, minScore: 6 })).ok, false);
  assert.equal(m14WorldJudgeCheck(read({ exit: 0, score: 4, minScore: 6 })).ok, false);
  assert.equal(m14WorldJudgeCheck(read({ exit: 0, score: 4, minScore: 4 })).ok, false);
  assert.equal(m14WorldJudgeCheck(read(null)).ok, false);
  assert.equal(m14WorldJudgeCheck(() => { throw new Error('missing'); }).ok, false);
});
