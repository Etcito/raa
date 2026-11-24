import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { SdfmError } from '../src/errors.js';
import {
  FIELD_DATA_HEADER, FIELD_DATA_VERSION, FIELD_FLAG_DEBUG, FIELD_FLAG_RELIEF, FIELD_FLAG_SINGLE, FIELD_FRAME_WGSL, FIELD_OPERAND_WORDS,
  INTERIOR_NO_TEXTURE, INTERIOR_WORDS, MATERIAL_FUNCTION_MAX_BYTES, RELIEF_MODULE_WORDS, REST_TANGENT_AXIS, checkPartShading, fieldPartShading, fieldSurfaceAttributes,
  restTangent,
} from '../src/field-shading.js';
import { forbiddenToken } from '../src/map-material.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url), 'utf8'));
/** The shape of MAMA's evaluator (field-program-wgsl.ts FIELD_PROGRAM_WGSL): PmfSample and pmf_eval over pmf_f. */
const FIELD = `struct PmfSample { d: f32, g: vec3f, la: u32, lb: u32, t: f32 }
fn pmf_eval(base: u32, p: vec3f) -> PmfSample { return PmfSample(length(p) - pmf_f(base + 3u), p, 0u, 1u, 0.5); }`;
const ROWS = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];
const PROGRAM = [1, 0, 8, 0.25, 0, 0, 0, 0];
const CONSTANT = { color: [0.25, 0.5, 0.75], metallic: 1, roughness: 0.2, emissive: [0, 0, 0.5] };

function shading(over = {}) {
  return {
    program: PROGRAM, single: false, look: { roughness: 0.4, metallic: 0 },
    operands: [{ color: [1, 0, 0], material: 'speckled', toNode: ROWS }, { color: [0, 0, 1], material: null, toNode: [2, 0, 0, 1, 0, 2, 0, 0, 0, 0, 2, 0] }],
    ...over,
  };
}
const words = (r) => ({ u: r.words, f: new Float32Array(r.words.buffer) });

test('fieldPartShading: one fixed interpreter, the part is data words', () => {
  const m = fixture('compiled-unmappable');
  const r = fieldPartShading(shading(), [m], { field: FIELD });
  const { wgsl, tintMode, missing } = r;
  assert.equal(wgsl.match(/\bfn\s+material_main\s*\(/g).length, 1);
  assert.ok(wgsl.includes('fn pmf_f(i: u32) -> f32 { return raa_material_f32(i); }'), 'the evaluator reads the material data words');
  assert.ok(wgsl.includes('let base = raa_material_data_base(in.record);'));
  assert.ok(wgsl.includes('let q0 = in.color.xyz;') && wgsl.includes('var pm = q0;') && wgsl.includes('s = pmf_eval(prog, pm);'), 'the field is read at the rest point');
  assert.equal(wgsl.match(/pmf_eval\(/g).length, 2, 'one call site (and its definition): the relief march through a module reuses it');
  assert.ok(wgsl.includes('fn sdfm_graph_0('), 'one graph per distinct body');
  assert.ok(!/extra_words|SDFM_K/.test(wgsl), 'constants are data, not text');
  assert.ok(/sdff_w\(b \+ \d+u\)/.test(wgsl), 'the graph reads its constants from the block');
  assert.ok(r.name.startsWith('sdff.') && r.name.length === 21);
  assert.equal(tintMode, 'multiply');
  assert.deepEqual(missing, []);
  assert.ok(!/\bfn\s+raa_/.test(wgsl) && !/@(group|binding)|var<(storage|uniform)/.test(wgsl), 'S11');
  const { u, f } = words(r);
  assert.deepEqual([u[0], u[1], u[2], u[3], u[4]], [1, 0, FIELD_DATA_HEADER, FIELD_DATA_HEADER + PROGRAM.length, 2]);
  assert.deepEqual(Array.from(f.subarray(8, 20)), ROWS, 'identity frame rows');
  assert.deepEqual(Array.from(f.subarray(20, 28)), PROGRAM, 'the program follows the header');
  const op = (i) => u[3] + FIELD_OPERAND_WORDS * i;
  assert.deepEqual([u[op(0) + 3], u[op(0) + 4]], [2, 0], 'operand 0: graph 0');
  assert.deepEqual(Array.from(u.subarray(u[op(0) + 5], u[op(0) + 5] + m.constants.length)), m.constants.map((w) => w >>> 0), 'its constants');
  assert.deepEqual([...f.subarray(op(1), op(1) + 3), u[op(1) + 3]], [0, 0, 1, 0], 'operand 1: blue colour');
  assert.deepEqual(Array.from(f.subarray(op(1) + 12, op(1) + 24)), [2, 0, 0, 1, 0, 2, 0, 0, 0, 0, 2, 0], 'its node rows');
});

test('fieldPartShading: constant materials share one text whatever the part; single, frame and debug are data', () => {
  const a = fieldPartShading(shading({ operands: [{ color: [1, 0, 0], material: 'paint', toNode: ROWS, constant: CONSTANT }] }), [], { field: FIELD });
  const frame = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1];
  const b = fieldPartShading(shading({ single: true, frame, program: [1, 0, 8, 1, 0, 0, 0, 0, 5, 6], operands: [
    { color: [0, 1, 0], material: 'wood', toNode: ROWS, constant: CONSTANT }, { color: [1, 1, 0], material: null, toNode: ROWS }] }), [], { field: FIELD, debug: true });
  assert.equal(a.wgsl, b.wgsl, 'the same text');
  assert.equal(a.name, b.name);
  assert.ok(!a.wgsl.includes('fn sdfm_graph_'), 'no graph code for constant materials');
  const { u, f } = words(b);
  assert.equal(u[1], FIELD_FLAG_SINGLE | FIELD_FLAG_DEBUG);
  assert.equal(f[8 + 7], 2, 'the frame translation y in row 1');
  const op = u[3];
  assert.deepEqual([...f.subarray(op, op + 3), u[op + 3], f[op + 6], f[op + 7], ...f.subarray(op + 8, op + 11)],
    [0.25, 0.5, 0.75, 1, 1, Math.fround(0.2), 0, 0, 0.5], 'operand 0: its constant values');
  assert.equal(u[op + FIELD_OPERAND_WORDS + 3], 0, 'operand 1: colour');
  assert.ok(a.wgsl.includes(`(flags & ${FIELD_FLAG_DEBUG}u) != 0u`) && a.wgsl.includes('o.emissive = o.normal * 0.5 + vec3<f32>(0.5);'));
});

test('fieldPartShading: graph functions follow the set of bodies; a missing material wears its colour', () => {
  const m = fixture('compiled-unmappable');
  const twin = { ...m, name: 'twin', constants: m.constants.map(() => 7) };
  const one = fieldPartShading(shading(), [m], { field: FIELD });
  const two = fieldPartShading(shading({ operands: [{ color: [1, 0, 0], material: 'twin', toNode: ROWS }, { color: [1, 0, 0], material: 'speckled', toNode: ROWS }] }),
    [m, twin], { field: FIELD });
  assert.equal(one.wgsl, two.wgsl, 'two materials with the same body share one function');
  const missing = fieldPartShading(shading({ operands: [{ color: [1, 1, 0], material: 'absent', toNode: ROWS }] }), [], { field: FIELD });
  assert.deepEqual(missing.missing, ['absent']);
  assert.equal(missing.words[missing.words[3] + 3], 0, 'colour mode');
});

test('fieldPartShading: indexed constant reads (a ramp stop) read the data words too', () => {
  const m = fixture('compiled-basic');
  const ramp = { ...m, name: 'ramp', body: m.body.replace(/return MaterialFull\(/, 'let o0 = b + min(extra_words[b + 1u], 2u);\n  let x = bitcast<f32>(extra_words[o0 + extra_words[b]]);\n  return MaterialFull(') };
  const { wgsl } = fieldPartShading(shading({ operands: [{ color: [1, 1, 1], material: 'ramp', toNode: ROWS }] }), [ramp], { field: FIELD });
  assert.ok(wgsl.includes('let o0 = b + min(sdff_w(b + 1u), 2u);') && wgsl.includes('bitcast<f32>(sdff_w(o0 + sdff_w(b)))'));
  assert.ok(!/extra_words/.test(wgsl));
});

test('fieldPartShading: a player-colour graph puts the material in graph tint mode', () => {
  const m = fixture('compiled-basic');
  const tinted = { ...m, name: 'skin', body: m.body.replace(/return MaterialFull\(/, 'let tint = mat_instance_tint;\n  return MaterialFull(') };
  assert.equal(fieldPartShading(shading({ operands: [{ color: [1, 1, 1], material: 'skin', toNode: ROWS }] }), [tinted], { field: FIELD }).tintMode, 'graph');
});

test('checkPartShading refuses malformed records; the evaluator and the WGSL size cap refuse', () => {
  const schema = (s, what) => assert.throws(() => checkPartShading(s), (e) => e instanceof SdfmError && e.code === 'sdfm.schema', what);
  schema(null, 'null');
  schema({ ...shading(), program: undefined, field: 'fn pmf_field() {}' }, 'a format 2 record');
  schema(shading({ program: [2, 0, 8, 0, 0, 0, 0, 0] }), 'program version');
  schema(shading({ operands: [{ color: [1, 0], material: null, toNode: ROWS }] }), 'colour');
  schema(shading({ operands: [{ color: [1, 0, 0], material: null, toNode: [1] }] }), 'rows');
  schema(shading({ operands: [{ color: [1, 0, 0], material: 'a', toNode: ROWS, constant: { color: [1, 1, 1] } }] }), 'constant');
  schema(shading({ frame: [1, 2] }), 'frame');
  schema(shading({ single: 1 }), 'single');
  schema(shading({ deformed: false }), 'deformed');
  assert.throws(() => fieldPartShading(shading(), [], { field: 'fn other() {}' }), (e) => e instanceof SdfmError && e.code === 'sdfm.schema');
  assert.throws(() => fieldPartShading(shading(), [fixture('compiled-unmappable')], { field: FIELD + '\n//' + 'x'.repeat(70000) }),
    (e) => e instanceof SdfmError && e.code === 'sdfm.material_unmappable' && /RAA limit 65536/.test(e.message));
});

const MODULE = { program: [1, 0, 8, 0.1, 0, 0, 0, 0], rows: [0.5, 0, 0, 1, 0, 0.5, 0, 2, 0, 0, 0.5, 3], count: 2, depth: 0.25, box: [-1, -1, -0.5, 1, 1, 0.5] };
const RELIEF = { depth: 0.25, modules: [MODULE], copies: [0, 0, 0, 0, -1, -1] };
const THREE = [{ color: [1, 0, 0], material: 'paint', toNode: ROWS, constant: CONSTANT }, { color: [0, 1, 0], material: null, toNode: ROWS },
  { color: [0, 0, 1], material: null, toNode: ROWS }];

test('relief: the flag, the depth, the per-operand module table and the module records are data', () => {
  const plain = fieldPartShading(shading({ operands: THREE }), [], { field: FIELD });
  const relief = fieldPartShading(shading({ relief: RELIEF, operands: THREE }), [], { field: FIELD });
  const u = relief.words, f = new Float32Array(u.buffer);
  const list = u[6];
  assert.ok(list >= FIELD_DATA_HEADER + PROGRAM.length + 3 * FIELD_OPERAND_WORDS, 'the copy list follows the operands');
  const at = u[list + 1];
  assert.deepEqual([u[list], u[list + 2]], [1, 0], 'one copy (operands 0 and 1), starting at operand 0');
  assert.deepEqual(Array.from(f.subarray(at + 16, at + 22)), MODULE.box, 'the module box');
  assert.equal(u[at], at + RELIEF_MODULE_WORDS, 'the module program follows its record');
  assert.deepEqual([u[at + 1], f[at + 2]], [2, 0.25], 'operand count and depth');
  assert.deepEqual(Array.from(f.subarray(at + 4, at + 16)), MODULE.rows, 'node -> module rows');
  assert.deepEqual(Array.from(f.subarray(u[at], u[at] + MODULE.program.length)), MODULE.program.map(Math.fround), 'the module program');
  assert.equal(u.length, u[at] + MODULE.program.length, 'the block ends with the last module');
  assert.throws(() => checkPartShading(shading({ relief: { ...RELIEF, copies: [0, 0] }, operands: THREE })), (e) => e instanceof SdfmError && e.code === 'sdfm.schema');
  assert.throws(() => checkPartShading(shading({ relief: { ...RELIEF, modules: [{ ...MODULE, rows: [1] }] }, operands: THREE })), (e) => e instanceof SdfmError && e.code === 'sdfm.schema');
  assert.equal(plain.wgsl, relief.wgsl, 'the same text: relief is a flag');
  assert.equal(plain.words[0], 1, 'version 1 block');
  assert.equal(plain.words[1] & FIELD_FLAG_RELIEF, 0);
  assert.equal(new Float32Array(plain.words.buffer)[5], 0);
  assert.equal(plain.words.length, FIELD_DATA_HEADER + PROGRAM.length + 3 * FIELD_OPERAND_WORDS, 'no extra words');
  assert.equal(plain.words[6], 0, 'no relief copies');
  assert.deepEqual(plain.interiors, []);
  assert.equal(relief.words[0], FIELD_DATA_VERSION);
  assert.equal(relief.words[1], FIELD_FLAG_RELIEF);
  assert.equal(new Float32Array(relief.words.buffer)[5], 0.25, 'the relief depth in header word 5');
  assert.ok(relief.wgsl.includes(`(flags & ${FIELD_FLAG_RELIEF}u) != 0u`) && relief.wgsl.includes('fn sdfm_rest_dir('));
  assert.throws(() => checkPartShading(shading({ relief: { depth: 0 } })), (e) => e instanceof SdfmError && e.code === 'sdfm.schema');
});

test('interiors: a pane operand gets its record, the tile id slots are listed, an unknown set shows glass only', () => {
  const pane = { color: [0.02, 0.03, 0.04], material: 'glass', toNode: ROWS, constant: CONSTANT,
    interior: { set: 'shop', depth: 0, width: 0, height: 0, glass: 0.4, half: [1.475, 2.1, 0.07] } };
  const lost = { ...pane, interior: { ...pane.interior, set: 'hotel' } };
  const interiors = { sets: { shop: { tiles: 3, depth: 7 }, office: { tiles: 4, depth: 5 } }, lightsOn: 0.5, blinds: 0.25, unlitLevel: 0.1 };
  const r = fieldPartShading(shading({ operands: [{ color: [1, 1, 1], material: null, toNode: ROWS }, pane, lost] }), [], { field: FIELD, interiors });
  const { u, f } = words(r);
  assert.equal(u[0], FIELD_DATA_VERSION);
  const op = (i) => u[3] + FIELD_OPERAND_WORDS * i;
  assert.equal(u[op(0) + 11], 0, 'operand 0 is not a pane');
  const at = u[op(1) + 11];
  assert.ok(at >= op(3), 'the record follows the operand table');
  assert.deepEqual(Array.from(f.subarray(at, at + 3)), [1.475, 2.1, 0.07].map(Math.fround), 'the pane half size');
  assert.deepEqual([f[at + 3], f[at + 4], f[at + 5], f[at + 6], u[at + 7]], [0, 0, 7, Math.fround(0.4), 3], 'cell size, the set depth, glass, 3 tiles');
  assert.deepEqual([f[at + 8], f[at + 9], f[at + 10], f[at + 11]], [0.5, 0.25, Math.fround(0.1), 1], 'lights on, blinds, unlit level, emissive');
  assert.deepEqual(Array.from(u.subarray(at + INTERIOR_WORDS, at + INTERIOR_WORDS + 9)).map((w, k) => (k % 3 === 2 ? f[at + INTERIOR_WORDS + k] : w)),
    [0, 1, 2].flatMap(() => [INTERIOR_NO_TEXTURE, INTERIOR_NO_TEXTURE, 0.5]), 'no picture yet, back fraction 0.5');
  const lostAt = u[op(2) + 11];
  assert.equal(u[lostAt + 7], 0, 'an unknown set: no tile, glass only');
  assert.deepEqual(r.interiors, [{ operand: 1, set: 'shop', at: at + INTERIOR_WORDS, tiles: 3 }, { operand: 2, set: 'hotel', at: lostAt + INTERIOR_WORDS, tiles: 0 }]);
  assert.equal(r.words.length, lostAt + INTERIOR_WORDS, 'the block ends with the last record');
  const noPanes = fieldPartShading(shading({ operands: [{ color: [1, 1, 1], material: null, toNode: ROWS }] }), [], { field: FIELD });
  assert.equal(r.wgsl, noPanes.wgsl, 'panes are data: the same text');
  assert.ok(r.wgsl.includes('fn sdff_interior(') && r.wgsl.includes('raa_texture(id, uv)'));
  assert.throws(() => checkPartShading(shading({ operands: [{ ...pane, interior: { set: 'shop', half: [1, 1] } }] })), (e) => e instanceof SdfmError && e.code === 'sdfm.schema');
});

test('the interpreter text holds no S11-forbidden token and stays under the RAA size limit with the real evaluator size', () => {
  const r = fieldPartShading(shading({ relief: { ...RELIEF, copies: [0, 0, 0, 0] } }), [fixture('compiled-unmappable')], { field: FIELD + '\n//' + 'x'.repeat(30000) });
  assert.equal(forbiddenToken(r.wgsl), null);
  const bytes = new TextEncoder().encode(r.wgsl).length;
  assert.ok(bytes < MATERIAL_FUNCTION_MAX_BYTES, `${bytes} bytes`);
});

test('field gradients are checked against the mesh before the silhouette correction', () => {
  const { wgsl } = fieldPartShading(shading(), [], { field: FIELD });
  const guard = wgsl.indexOf('if (validate_field && !(dot(m, n) >= 0.5)) { return n; }');
  const facing = wgsl.indexOf('let d = dot(m, in.view_dir);');
  assert.ok(guard >= 0 && guard < facing, 'a reversed field gradient must fall back before it can face the camera');
  assert.match(wgsl, /sdfm_pose_normal\(in, s\.g \/ len, true\)/);
  assert.match(wgsl, /fn sdfm_posed_normal[^{}]+\{\s*return sdfm_pose_normal\(in, g, false\);/);
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
});
