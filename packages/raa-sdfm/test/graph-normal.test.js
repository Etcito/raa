// The shading normal of a MAMA graph (MAMA slots.ts graphShadingNormal: the bump of its height output, its authored
// normal map) in the field interpreter: fieldPartShading with a material's `normal`.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fieldPartShading } from '../src/field-shading.js';

const FIELD = `struct PmfSample { d: f32, g: vec3f, la: u32, lb: u32, t: f32 }
fn pmf_eval(base: u32, p: vec3f) -> PmfSample { return PmfSample(length(p) - pmf_f(base + 3u), p, 0u, 1u, 0.5); }`;
const ROWS = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];
const FIELDS = ['base_color', 'metallic', 'roughness', 'emissive', 'height'];
const BODY = '  return MaterialFull(vec3f(bitcast<f32>(extra_words[b + 0u])), 0.0, 0.5, vec3f(0.0), 0.0);\n';

const bumped = {
  name: 'rock', key: 'b'.repeat(64), fields: FIELDS, constants: [0x3f000000, 0x3b800000], helpers: [], body: BODY,
  normal: `fn mat_hfn(b: u32, p: vec3f, pn: vec3f, n: vec3f, mk: MatMasks) -> f32 {\n  return sin(pn.x * 40.0);\n}\n
fn mat_graph_normal(b: u32, p: vec3f, pn: vec3f, j: mat3x3f, n: vec3f, tn: vec3f) -> vec3f {
  let e = bitcast<f32>(extra_words[b + 1u]);
  return normalize(n - vec3f(mat_hfn(b, p + j * vec3f(e, 0.0, 0.0), pn, n, MatMasks(0.5, 1.0, 1.0, 0.0, 1.0))));
}\n`,
};
const plain = { name: 'clay', key: 'c'.repeat(64), fields: FIELDS, constants: [0x3f000000], helpers: [], body: BODY };
const shading = (materials) => ({ program: [1, 0, 8, 0.25, 0, 0, 0, 0], single: false, look: { roughness: 0.4, metallic: 0 },
  operands: materials.map((material) => ({ color: [1, 0, 0], material, toNode: ROWS })) });

test('fieldPartShading: a graph with a shading normal bends the fragment normal, a graph without one keeps the field normal', () => {
  const r = fieldPartShading(shading(['rock', 'clay']), [bumped, plain], { field: FIELD });
  const k = r.wgsl.match(/fn sdfm_normal_(\d+)\(/)?.[1];
  assert.ok(k !== undefined, 'the normal function is renamed per graph');
  assert.ok(r.wgsl.includes(`fn sdfm_hfn_${k}(`) && !/\bmat_hfn\b|\bmat_graph_normal\b/.test(r.wgsl));
  assert.ok(r.wgsl.includes(`case ${k}u: { sdff_bent = true; return sdfm_normal_${k}(b, p, pn, j, n, tn); }`));
  assert.equal((r.wgsl.match(/sdfm_normal_\d+\(b, p/g) ?? []).length, 1, 'only the bumped graph has a case');
  assert.ok(r.wgsl.includes('sdff_w(b + 1u)'), 'its words are material data reads after the constants');
  assert.ok(r.wgsl.includes('if (sdff_bent) { o.normal = sdfm_posed_normal(in, sdff_rest_n); }'));
  // No authored prelude: the label passes no authored normal.
  assert.ok(r.wgsl.includes('vec3<f32>(0.0, 0.0, 1.0));\n      return m.m;') && !r.wgsl.includes('mat_authored_reset();'));
  // Without a normal no graph-normal dispatch is emitted.
  assert.ok(!fieldPartShading(shading(['clay']), [plain], { field: FIELD }).wgsl.includes('sdff_graph_normal'));
});

test('fieldPartShading: an authored normal resets the authored outputs before the graph and reads mat_out_normal', () => {
  const prelude = { id: 'mat_authored_prelude', wgsl: 'var<private> mat_out_normal: vec3f = vec3f(0.0, 0.0, 1.0);\nfn mat_authored_reset() { mat_out_normal = vec3f(0.0, 0.0, 1.0); }' };
  const authored = { ...bumped, helpers: [prelude], normal: 'fn mat_graph_normal(b: u32, p: vec3f, pn: vec3f, j: mat3x3f, n: vec3f, tn: vec3f) -> vec3f { return normalize(n + tn); }' };
  const r = fieldPartShading(shading(['rock']), [authored], { field: FIELD });
  assert.ok(r.wgsl.includes('mat_authored_reset();\n    let pn = '));
  assert.ok(r.wgsl.includes('pn, j, np, mat_out_normal);'));
});
