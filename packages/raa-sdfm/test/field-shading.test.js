import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { SdfmError } from '../src/errors.js';
import {
  FIELD_FRAME_WGSL, REST_TANGENT_AXIS, checkPartShading, fieldPartMaterialWgsl, fieldSurfaceAttributes, restTangent,
} from '../src/field-shading.js';
import { volumeSurfaceAttributes } from '../src/normal-volume.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url), 'utf8'));
/** The shape of a MAMA field (field-wgsl.ts): a PmfSample struct and pmf_field. */
const FIELD = `struct PmfSample { d: f32, g: vec3f, la: u32, lb: u32, t: f32 }
fn pmf_field(p: vec3f) -> PmfSample { return PmfSample(length(p) - 1.0, p, 0u, 1u, 0.5); }`;
const ROWS = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];

function shading(over = {}) {
  return {
    field: FIELD, single: false, look: { roughness: 0.4, metallic: 0 },
    operands: [{ color: [1, 0, 0], material: 'speckled', toNode: ROWS }, { color: [0, 0, 1], material: null, toNode: [2, 0, 0, 1, 0, 2, 0, 0, 0, 0, 2, 0] }],
    ...over,
  };
}

test('fieldPartMaterialWgsl: field normal, operand graphs at p and pn, colours elsewhere, mixed by the field weight', () => {
  const m = fixture('compiled-unmappable');
  const { wgsl, tintMode, missing } = fieldPartMaterialWgsl(shading(), [m]);
  assert.equal(wgsl.match(/\bfn\s+material_main\s*\(/g).length, 1);
  assert.ok(wgsl.includes('fn pmf_field('));
  assert.ok(wgsl.includes('let s = pmf_field(in.color.xyz);'), 'the field is read at the rest point');
  assert.ok(wgsl.includes('o.normal = sdff_normal(in, s);'));
  assert.ok(wgsl.includes('fn sdfm_graph_0('), 'one graph per used material');
  assert.ok(wgsl.includes(`const SDFM_K0 = array<u32, ${m.constants.length}>(`));
  assert.ok(!/extra_words/.test(wgsl), 'constants are inlined');
  assert.ok(wgsl.includes('case 0: { return sdfm_graph_0(0u, p, pn, n, SdfmMatMasks(0.5, 1.0, 1.0, 0.0, 1.0)); }'));
  assert.ok(wgsl.includes('vec4f(1.0,0.0,0.0,0.0)'), 'operand 0: red, material 0');
  assert.ok(wgsl.includes('vec4f(0.0,0.0,1.0,-1.0)'), 'operand 1: blue, no material');
  assert.ok(wgsl.includes('vec4f(2.0,0.0,0.0,1.0)'), 'operand 1 node rows (the graph reads pn)');
  assert.ok(wgsl.includes('o.roughness = m.roughness;'), 'per-face look unless single');
  assert.ok(wgsl.includes('fn sdff_program_point(q: vec3<f32>) -> vec3<f32> {\n  return q;'), 'identity frame');
  assert.equal(tintMode, 'multiply');
  assert.deepEqual(missing, []);
  assert.ok(!/\bfn\s+raa_/.test(wgsl) && !/@(group|binding)|var<(storage|uniform)/.test(wgsl), 'S11');
});

test('fieldPartMaterialWgsl: single look, frame, missing material and debug', () => {
  const frame = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1];
  const { wgsl, missing } = fieldPartMaterialWgsl(shading({ single: true, frame, operands: [{ color: [1, 1, 0], material: 'absent', toNode: ROWS }] }), [], { debug: true });
  assert.deepEqual(missing, ['absent']);
  assert.ok(!wgsl.includes('fn sdfm_graph_'), 'a missing material wears its colour');
  assert.ok(!wgsl.includes('o.roughness = m.roughness;'), 'single: roughness from the record');
  assert.ok(wgsl.includes('0.0 * q.z + 2.0'), 'the frame maps the rest point to program space');
  assert.ok(wgsl.includes('o.emissive = o.normal * 0.5 + vec3<f32>(0.5);'));
});

test('fieldPartMaterialWgsl: a player-colour graph puts the material in graph tint mode', () => {
  const m = fixture('compiled-basic');
  const tinted = { ...m, name: 'skin', body: m.body.replace(/return MaterialFull\(/, 'let tint = mat_instance_tint;\n  return MaterialFull(') };
  assert.equal(fieldPartMaterialWgsl(shading({ operands: [{ color: [1, 1, 1], material: 'skin', toNode: ROWS }] }), [tinted]).tintMode, 'graph');
});

test('checkPartShading refuses malformed records; the WGSL size cap refuses', () => {
  const schema = (s, what) => assert.throws(() => checkPartShading(s), (e) => e instanceof SdfmError && e.code === 'sdfm.schema', what);
  schema(null, 'null');
  schema(shading({ field: 'fn other() {}' }), 'no pmf_field');
  schema(shading({ operands: [{ color: [1, 0], material: null, toNode: ROWS }] }), 'colour');
  schema(shading({ operands: [{ color: [1, 0, 0], material: null, toNode: [1] }] }), 'rows');
  schema(shading({ frame: [1, 2] }), 'frame');
  schema(shading({ single: 1 }), 'single');
  assert.throws(() => fieldPartMaterialWgsl(shading({ field: FIELD + '\n//' + 'x'.repeat(70000) }), [fixture('compiled-unmappable')]),
    (e) => e instanceof SdfmError && e.code === 'sdfm.material_unmappable' && /RAA limit 65536/.test(e.message));
});

test('the rest frame is linear in the rest normal and the WGSL axis is REST_TANGENT_AXIS', () => {
  const n = [0.6, 0, 0.8];
  const [ax, ay, az] = REST_TANGENT_AXIS;
  assert.deepEqual(restTangent(n), [ay * 0.8, az * 0.6 - ax * 0.8, -ay * 0.6]);
  assert.ok(FIELD_FRAME_WGSL.includes(`vec3f(${REST_TANGENT_AXIS.map((v) => Math.fround(v).toPrecision(9)).join(', ')})`));
  const positions = new Float32Array([1, 2, 3]), normals = new Float32Array([0, 0, 2]);
  const s = fieldSurfaceAttributes(positions, normals);
  assert.deepEqual(Array.from(s.colors), [1, 2, 3, 1]);
  assert.deepEqual(Array.from(s.uvs), [0, 0]);
  assert.deepEqual(Array.from(s.tangents), [...restTangent([0, 0, 1]).map(Math.fround), 1]);
  assert.deepEqual(volumeSurfaceAttributes(positions, normals), s, 'the volume streams are the same rest frame');
});
