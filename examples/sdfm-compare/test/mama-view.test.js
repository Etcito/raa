import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  bakeReferenceMesh, cameraMatrices, createMamaView, linearToSrgb, NO_MATERIAL_COLOR, PANE_LABEL,
} from '../lib/mama-view.js';

// One part translated by (1, 2, 3) with a 90 deg turn about Y; two sub-meshes, one without material.
const frame = Float64Array.of(0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 2, 3, 1);
const tri = (material) => ({
  material,
  positions: Float32Array.of(0, 0, 0, 1, 0, 0, 0, 1, 0),
  normals: Float32Array.of(1, 0, 0, 1, 0, 0, 1, 0, 0),
  uvs: null,
  indices: Uint32Array.of(0, 1, 2),
});
const meshed = { groups: [{ group: 'g', parts: [{ partId: 'p', name: 'p', frame, meshes: [tri('paint'), tri(null)] }] }], materials: [] };
const doc = { materials: { paint: { name: 'Paint' } } };
const fakeCpu = (def) => ({ eval: (p) => ({ baseColor: def.name === 'Paint' ? [0.8, 0.1, p[1] / 10] : [0, 0, 0] }) });

test('the pane label names MAMA 864ebc3 and says it is a reference viewport', () => {
  assert.equal(PANE_LABEL, 'MAMA 864ebc3 mesher (reference viewport)');
});

test('bakeReferenceMesh places vertices by the part frame and colours them with MAMA CPU base colours', () => {
  const b = bakeReferenceMesh(meshed, doc, fakeCpu);
  assert.equal(b.triangles, 2);
  assert.deepEqual([...b.indices], [0, 1, 2, 3, 4, 5]);
  const v = (i) => [...b.vertices.slice(i * 9, i * 9 + 9)].map((x) => Math.round(x * 1e6) / 1e6);
  // local (1, 0, 0) -> world (1, 2, 2); local normal +X -> world -Z
  assert.deepEqual(v(1).slice(0, 6), [1, 2, 2, 0, 0, -1]);
  assert.deepEqual(v(2).slice(6), [0.8, 0.1, 0.3]);
  assert.deepEqual(v(4).slice(6), NO_MATERIAL_COLOR);
});

test('bakeReferenceMesh refuses a material the document does not have', () => {
  const bad = { groups: [{ group: 'g', parts: [{ partId: 'p', name: 'p', frame, meshes: [tri('wood')] }] }] };
  assert.throws(() => bakeReferenceMesh(bad, doc, fakeCpu), (e) => e.code === 'mama_view.material' && /wood/.test(e.message));
});

test('cameraMatrices puts the target at the clip centre with depth in (0, 1)', () => {
  const { viewProj } = cameraMatrices({ position: [1.3, 1.3, 2.4], target: [0, 0.5, 0], fovYDeg: 45, near: 0.01, far: 200 }, 640 / 720);
  const p = [0, 0.5, 0, 1];
  const clip = [0, 1, 2, 3].map((r) => [0, 1, 2, 3].reduce((s, c) => s + viewProj[c * 4 + r] * p[c], 0));
  assert.ok(Math.abs(clip[0] / clip[3]) < 1e-6 && Math.abs(clip[1] / clip[3]) < 1e-6);
  const z = clip[2] / clip[3];
  assert.ok(z > 0 && z < 1, `depth ${z}`);
});

test('linearToSrgb follows the sRGB OETF', () => {
  assert.equal(linearToSrgb(0), 0);
  assert.ok(Math.abs(linearToSrgb(1) - 1) < 1e-12);
  assert.ok(Math.abs(linearToSrgb(0.5) - 0.735357) < 1e-5);
});

test('createMamaView refuses without mamaUrl or without WebGPU', async () => {
  await assert.rejects(createMamaView({}, {}), (e) => e.code === 'mama_view.options');
  await assert.rejects(createMamaView({}, { mamaUrl: 'mama/index.js', gpu: null }), (e) => e.code === 'mama_view.no_webgpu');
  const noAdapter = { requestAdapter: async () => null };
  await assert.rejects(createMamaView({}, { mamaUrl: 'mama/index.js', gpu: noAdapter }), (e) => e.code === 'mama_view.no_webgpu');
});
