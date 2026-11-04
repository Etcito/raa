// Per-pixel SDF shading of a mesh part, computed in the fragment shader from the part's own field (owner 2026-11-04: the
// normal AND the materials of a game mesh come from the SDF in the shader, nothing is baked). The ONE material code
// path of SDF Modeler's Game view and of the game (royal-fiesta sdfPort.ts): both hand the same PartShading record
// (MAMA src/core/paramesh/field-program.ts PartShading, stored with a saved game mesh part as `shading`) and the same
// compiled materials (SdfmCompiledMaterial) to fieldPartShading. Pure text and array work, no engine call, no MAMA
// import (the field evaluator text, MAMA field-program-wgsl.ts FIELD_PROGRAM_WGSL, is handed in by the caller).
//
// ONE interpreter, not one shader per part (owner, same day: "UN seul shader qui fasse un calcul avec le SDF en
// entrée"): the material function text is fixed; everything about the part (its field program, its operands' colours,
// materials and node frames, its frame, single look and debug view) is DATA, RAA material data words (m25) the
// function reads from its material record's base. Parts whose materials are all constant (MAMA single-material.ts
// constantMaterial folds them) share one text, hence one pipeline whatever the scene; a part whose operands run a
// material graph (a pattern, the player colour) adds that graph's function, named by the set of graph bodies, so parts
// with the same graphs still share one text.
//
// Per fragment: the part's field (pmf_eval) at the rest point gives the shading normal (its gradient) and the two
// heaviest operands with their weight; each operand is shaded as SDF Modeler's Local fields shades its label (its
// constant values, its material graph at the program-space point p and its node-space point pn, else its colour) and
// the two are mixed by the field's weight. Material boundaries therefore follow the SDF at pixel precision.
//
// Data a fragment needs, and where it travels (fieldSurfaceAttributes):
//   - the rest position in the part's mesh space (the field's space): the vertex colour xyz, unclamped f32, passed
//     through RAA's skinning untouched, so a posed skinned part still reads the field at its rest point;
//   - the rest frame: (uv, colour w) = the rest normal as plain xyz, tangent = restTangent = cross(REST_TANGENT_AXIS,
//     rest normal) (handedness +1); RAA moves tangent and normal with the instance and the skin, and the rest-space
//     gradient is re-expressed in that posed frame. Both streams are LINEAR in the rest normal, so a fragment rebuilds
//     exactly the interpolated rest frame RAA interpolated in posed space (an octahedral normal and a Frisvad tangent do
//     not interpolate: across the octahedral fold and the z = 0 tangent seam the rebuilt frame turned up to 160
//     degrees, dark streaks on the -z half of a sphere, SDF Modeler file 34, 2026-11-04);
//   - the part: its data block (FIELD_DATA_* layout below), the `data` base of its surface material record.
// A fragment where the field has no gradient, or whose field normal is more than 60 degrees off the mesh normal (a thin
// wall, a coarse approximation), keeps the mesh normal.
import { SdfmError } from './errors.js';
import { checkShape, forbiddenToken, tokens } from './map-material.js';

/**
 * The fixed axis of the rest tangent cross(axis, n): (1, 2, 3) / sqrt(14), parallel to no axis-aligned or 45-degree
 * face, so only isolated points of curved surfaces meet its singularity.
 */
export const REST_TANGENT_AXIS = Object.freeze([1 / Math.sqrt(14), 2 / Math.sqrt(14), 3 / Math.sqrt(14)]);

/** S11 user functions are at most this many bytes (raa-shader). */
export const MATERIAL_FUNCTION_MAX_BYTES = 65536;

/** MAMA field program layout version this interpreter reads (MAMA field-program.ts FIELD_PROGRAM_VERSION). */
export const FIELD_PROGRAM_VERSION = 1;

/**
 * Layout of a part's data block (u32 words from the block base; f32 values stored as their bits):
 *   [0] FIELD_DATA_VERSION, [1] flags (FIELD_FLAG_*), [2] program offset, [3] operand table offset, [4] operand count,
 *   [5..7] 0, [8..19] the part -> program frame as 3 rows of 4 f32 (identity when absent), then the field program, the
 *   operand table (FIELD_OPERAND_WORDS per operand) and the graph constants.
 * An operand: [0..2] colour f32, [3] mode (0 colour, 1 constant, 2 graph), [4] graph index, [5] constants offset,
 *   [6] metallic, [7] roughness, [8..10] emissive, [11] 0, [12..23] the program -> node rows (3 x 4 f32).
 */
export const FIELD_DATA_VERSION = 1;
export const FIELD_DATA_HEADER = 20;
export const FIELD_OPERAND_WORDS = 24;
/** Flags of word 1: one material per object (roughness and metallic from the record), the normal debug view. */
export const FIELD_FLAG_SINGLE = 1;
export const FIELD_FLAG_DEBUG = 2;

const MODE = { colour: 0, constant: 1, graph: 2 };

/**
 * `sdfm_posed_normal(in, g)`: a rest-space normal `g` carried onto the drawn (posed, world) surface of the fragment
 * through the rest frame. The interpolated rest normal and its tangent cross(axis, normal) are rebuilt unnormalised,
 * i.e. exactly the vectors RAA interpolated in posed space, then both frames are orthonormalised the same way.
 */
export const FIELD_FRAME_WGSL = /* wgsl */ `
const SDFM_TANGENT_AXIS = vec3f(${REST_TANGENT_AXIS.map((v) => Math.fround(v).toPrecision(9)).join(', ')});

fn sdfm_posed_normal(in: MaterialInput, g: vec3f) -> vec3f {
  let n = normalize(in.normal);
  if (dot(g, g) == 0.0) { return n; }
  let nr0 = vec3f(in.uv, in.color.w);
  if (dot(nr0, nr0) < 1e-12) { return n; }
  let nr = normalize(nr0);
  let tr0 = cross(SDFM_TANGENT_AXIS, nr0);
  let trp = tr0 - nr * dot(nr, tr0);
  let tw = in.tangent.xyz - n * dot(n, in.tangent.xyz);
  // At the axis singularity (one point of a curved surface) both tangents vanish together: the mesh normal there.
  if (dot(trp, trp) < 1e-8 || dot(tw, tw) < 1e-12) { return n; }
  let tr = normalize(trp);
  let br = cross(nr, tr);
  let t = normalize(tw);
  let bw = cross(n, t) * select(-1.0, 1.0, in.tangent.w >= 0.0);
  var m = normalize(dot(g, tr) * t + dot(g, br) * bw + dot(g, nr) * n);
  // At the silhouette the SDF normal of a facet the camera sees can lean past the view: keep it visible (no dark rim).
  let d = dot(m, in.view_dir);
  if (d < 0.02 && dot(n, in.view_dir) > 0.0) { m = normalize(m + in.view_dir * (0.02 - d)); }
  return m;
}
`;

/** `sdff_normal(in, s)`: the field's normal at the fragment, on the posed surface (needs FIELD_FRAME_WGSL and the field). */
export const FIELD_NORMAL_WGSL = /* wgsl */ `
fn sdff_normal(in: MaterialInput, s: PmfSample) -> vec3f {
  let mesh = normalize(in.normal);
  let len = length(s.g);
  if (!(len > 0.0) || len > 3.0e38) { return mesh; }
  let m = sdfm_posed_normal(in, s.g / len);
  // A field normal past 60 degrees from the mesh's (a thin wall, a coarse approximation) keeps the mesh normal.
  return select(mesh, m, dot(m, mesh) >= 0.5);
}
`;

/** The interpreter's fixed part: data readers, the operand shading and material_main (after the field and the graphs). */
const INTERPRETER_WGSL = /* wgsl */ `
fn pmf_f(i: u32) -> f32 { return raa_material_f32(i); }
fn sdff_w(i: u32) -> u32 { return raa_material_word(i); }
fn sdff_v3(i: u32) -> vec3<f32> { return vec3<f32>(raa_material_f32(i), raa_material_f32(i + 1u), raa_material_f32(i + 2u)); }
fn sdff_v4(i: u32) -> vec4<f32> { return vec4<f32>(sdff_v3(i), raa_material_f32(i + 3u)); }
fn sdff_affine(i: u32, q: vec3<f32>) -> vec3<f32> {
  let r0 = sdff_v4(i);
  let r1 = sdff_v4(i + 4u);
  let r2 = sdff_v4(i + 8u);
  return vec3<f32>(dot(r0.xyz, q) + r0.w, dot(r1.xyz, q) + r1.w, dot(r2.xyz, q) + r2.w);
}

// Local fields' evalLabel for field operand i of the block at base: its constant values, its material graph at p and
// its node-space point, else its colour.
fn sdff_label(base: u32, i: u32, p: vec3<f32>, n: vec3<f32>) -> SdfmMaterialFull {
  if (i >= sdff_w(base + 4u)) { return SdfmMaterialFull(vec3<f32>(0.5), 0.0, 0.5, vec3<f32>(0.0), 0.0); }
  let o = base + sdff_w(base + 3u) + ${FIELD_OPERAND_WORDS}u * i;
  let mode = sdff_w(o + 3u);
  let c = sdff_v3(o);
  if (mode == ${MODE.constant}u) { return SdfmMaterialFull(c, pmf_f(o + 6u), pmf_f(o + 7u), sdff_v3(o + 8u), 0.0); }
  if (mode == ${MODE.graph}u) {
    let m = sdff_graph(sdff_w(o + 4u), base + sdff_w(o + 5u), p, sdff_affine(o + 12u, p), n);
    if (m.ok) { return m.m; }
  }
  return SdfmMaterialFull(c, 0.0, 0.5, vec3<f32>(0.0), 0.0);
}

fn material_main(in: MaterialInput) -> MaterialOutput {
  var o = raa_default_output(in);
  o.opacity = 1.0; // the vertex colour's w is the rest normal's z, not an alpha
  let base = raa_material_data_base(in.record);
  let flags = sdff_w(base + 1u);
  let s = pmf_eval(base + sdff_w(base + 2u), in.color.xyz);
  o.normal = sdff_normal(in, s);
  let p = sdff_affine(base + 8u, in.color.xyz);
  var m = sdff_label(base, s.la, p, o.normal);
  if (s.lb != s.la && s.t > 0.0) {
    let b = sdff_label(base, s.lb, p, o.normal);
    m = SdfmMaterialFull(mix(m.base_color, b.base_color, s.t), mix(m.metallic, b.metallic, s.t),
      mix(m.roughness, b.roughness, s.t), mix(m.emissive, b.emissive, s.t), mix(m.height, b.height, s.t));
  }
  o.base_color = m.base_color;
  o.emissive = m.emissive;
  if ((flags & ${FIELD_FLAG_SINGLE}u) == 0u) {
    o.metallic = m.metallic;
    o.roughness = m.roughness;
  }
  if ((flags & ${FIELD_FLAG_DEBUG}u) != 0u) {
    o.base_color = vec3<f32>(0.0);
    o.metallic = 0.0;
    o.roughness = 1.0;
    o.emissive = o.normal * 0.5 + vec3<f32>(0.5);
  }
  return o;
}
`;

/** MAMA's MaterialFull fields in declaration order (core/material-graph/wgsl-parts.ts); a graph must declare exactly these. */
const MATERIAL_FULL_FIELDS = Object.freeze(['base_color', 'metallic', 'roughness', 'emissive', 'height']);
const CONSTANT_READ = /extra_words\s*\[\s*b\s*\+\s*(\d+)u?\s*\]/g;

const finite = (a, n) => Array.isArray(a) && a.length === n && a.every(Number.isFinite);

/**
 * `text` with every `extra_words[i]` (MAMA's binding-7 constant reads, `b` the graph's constant base and indices
 * derived from it, such as a ramp's stop offsets) read from the material data words instead: `sdff_w(i)`. The graph's
 * constants sit at its operand's constant offset in the part's block, which the interpreter passes as `b`.
 */
function dataReads(text) {
  let out = '', at = 0;
  for (const m of text.matchAll(/\bextra_words\s*\[/g)) {
    if (m.index < at) continue;
    let depth = 1, i = m.index + m[0].length;
    for (; i < text.length && depth > 0; i++) depth += text[i] === '[' ? 1 : text[i] === ']' ? -1 : 0;
    if (depth > 0) break;
    out += `${text.slice(at, m.index)}sdff_w(${dataReads(text.slice(m.index + m[0].length, i - 1))})`;
    at = i;
  }
  return out + text.slice(at);
}

/** Checks a PartShading record; returns it, or throws `sdfm.schema` naming the first bad field. */
export function checkPartShading(s) {
  const bad = (what) => { throw new SdfmError('sdfm.schema', `part shading: ${what}`); };
  if (s === null || typeof s !== 'object') bad('not an object');
  if (!Array.isArray(s.program) || s.program.length < 8 || !s.program.every(Number.isFinite)) {
    bad('program is not a field program (MAMA field-program.ts; a format 2 record carries WGSL instead: re-save the mesh)');
  }
  if (s.program[0] !== FIELD_PROGRAM_VERSION) bad(`program version ${s.program[0]}; this interpreter reads ${FIELD_PROGRAM_VERSION}`);
  if (!Array.isArray(s.operands)) bad('operands is not an array');
  s.operands.forEach((o, i) => {
    if (o === null || typeof o !== 'object') bad(`operands[${i}] is not an object`);
    if (!finite(o.color, 3)) bad(`operands[${i}].color is not 3 finite numbers`);
    if (o.material !== null && (typeof o.material !== 'string' || o.material === '')) bad(`operands[${i}].material is not a name or null`);
    if (!finite(o.toNode, 12)) bad(`operands[${i}].toNode is not 12 finite numbers`);
    const k = o.constant;
    if (k !== undefined && (k === null || typeof k !== 'object' || !finite(k.color, 3) || !Number.isFinite(k.metallic) || !Number.isFinite(k.roughness) || !finite(k.emissive, 3))) {
      bad(`operands[${i}].constant is not { color, metallic, roughness, emissive }`);
    }
  });
  if (s.frame !== undefined && !finite(Array.from(s.frame ?? []), 16)) bad('frame is not 16 finite numbers');
  if (typeof s.single !== 'boolean') bad('single is not a boolean');
  if (s.look === null || typeof s.look !== 'object' || !Number.isFinite(s.look.roughness) || !Number.isFinite(s.look.metallic)) bad('look is not { roughness, metallic }');
  if (s.deformed !== undefined && s.deformed !== true) bad('deformed is not true or absent');
  return s;
}

/** 16 hex digits of two independent 32-bit passes (FNV-1a, a rotating odd multiply) over the UTF-16 units of `text` (a name, not a security hash). */
function textHash(text) {
  let a = 0x811c9dc5, b = 0x9e3779b9;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    a = Math.imul(a ^ c, 0x01000193) >>> 0;
    b = Math.imul(((b ^ c) << 13) | ((b ^ c) >>> 19), 0x5bd1e995) >>> 0;
  }
  return a.toString(16).padStart(8, '0') + b.toString(16).padStart(8, '0');
}

/**
 * The per-pixel shading of a field part: ONE interpreter material function (fixed text; parts whose operand graphs
 * are the same share it: `name` is `'sdff.'` + a hash of the text) and the part's data block (`words`, for
 * `Engine.addMaterialData`; its base is the surface record's `data`). Each operand wears its constant material, its
 * material graph (`materials`: the asset's compiled materials, by name), else its colour; a graph material missing from
 * `materials` wears its colour and is named in `missing`. `single` leaves roughness and metallic to the surface record
 * (`look`). `debug` shows the shading normal as an emissive colour (0.5 + n / 2) on black (a data flag: same text).
 *
 * @param {object} shading PartShading (MAMA field-program.ts)
 * @param {ReadonlyMap<string, object> | readonly object[]} materials SdfmCompiledMaterial by name, or a list of them
 * @param {{ field: string, debug?: boolean }} options `field`: MAMA FIELD_PROGRAM_WGSL (the evaluator, pmf_eval)
 * @returns {{ name: string, wgsl: string, words: Uint32Array, tintMode: 'graph' | 'multiply', missing: string[] }}
 * @throws {SdfmError} `sdfm.schema` for a malformed record or evaluator; `sdfm.material_unmappable` for a material that
 *   cannot be inlined, a forbidden token, or more than MATERIAL_FUNCTION_MAX_BYTES bytes
 */
export function fieldPartShading(shading, materials, options) {
  checkPartShading(shading);
  const field = options?.field;
  if (typeof field !== 'string' || !/\bfn\s+pmf_eval\s*\(/.test(field) || !/\bstruct\s+PmfSample\b/.test(field)) {
    throw new SdfmError('sdfm.schema', 'options.field is not the field evaluator (MAMA field-program-wgsl.ts FIELD_PROGRAM_WGSL)');
  }
  const byName = materials instanceof Map ? materials : new Map(Array.from(materials, (m) => [m?.name, m]));
  const fail = (what) => { throw new SdfmError('sdfm.material_unmappable', what); };
  const structs = (text) => text.replace(/\bMaterialFull\b/g, 'SdfmMaterialFull').replace(/\bMatMasks\b/g, 'SdfmMatMasks');
  // The graph materials (non-constant, compiled): each becomes its body text (constants read from the data block).
  const graphOf = new Map(), missing = [], helpers = new Map();
  let tintMode = 'multiply';
  for (const o of shading.operands) {
    const name = o.material;
    if (name === null || o.constant !== undefined || graphOf.has(name) || missing.includes(name)) continue;
    const m = byName.get(name);
    if (!m) { missing.push(name); continue; }
    checkShape(m);
    if (m.fields.join() !== MATERIAL_FULL_FIELDS.join()) fail(`material '${name}': MaterialFull fields ${m.fields.join(', ')}; expected ${MATERIAL_FULL_FIELDS.join(', ')}`);
    const n = m.constants.length;
    for (const [read, i] of m.body.matchAll(CONSTANT_READ)) if (Number(i) >= n) fail(`material '${name}': '${read}' reads past the ${n} constant words`);
    const body = structs(dataReads(m.body)).replace(/\s+$/, '');
    for (const h of m.helpers) {
      const text = structs(dataReads(h.wgsl)).trim();
      const known = helpers.get(h.id);
      if (known !== undefined && known !== text) fail(`material '${name}': helper '${h.id}' differs from another material's`);
      helpers.set(h.id, text);
    }
    for (const text of [body, ...m.helpers.map((h) => dataReads(h.wgsl))]) {
      const toks = tokens(text);
      if (toks.includes('mat_instance_tint')) tintMode = 'graph';
      if (toks.includes('extra_words')) fail(`material '${name}': 'extra_words' is not read as extra_words[...]`);
      const raa = toks.find((t) => /^raa_/.test(t));
      if (raa) fail(`material '${name}': '${raa}' (identifiers starting with raa_ are reserved)`);
    }
    graphOf.set(name, { body, constants: m.constants });
  }
  // One function per distinct body, in text order: the same graphs give the same function text whatever the part.
  const bodies = [...new Set([...graphOf.values()].map((g) => g.body))].sort();
  const graphs = bodies.map((body, k) => [
    `fn sdfm_graph_${k}(b: u32, p: vec3<f32>, pn: vec3<f32>, n: vec3<f32>, mk: SdfmMatMasks) -> SdfmMaterialFull {`, body, '}'].join('\n'));
  const masks = 'SdfmMatMasks(0.5, 1.0, 1.0, 0.0, 1.0)';
  const dispatch = `struct SdffGraph { ok: bool, m: SdfmMaterialFull }

fn sdff_graph(k: u32, b: u32, p: vec3<f32>, pn: vec3<f32>, n: vec3<f32>) -> SdffGraph {
  var out = SdffGraph(false, SdfmMaterialFull(vec3<f32>(0.0), 0.0, 0.0, vec3<f32>(0.0), 0.0));
${bodies.length ? `  switch k {
${bodies.map((_, k) => `    case ${k}u: { out = SdffGraph(true, sdfm_graph_${k}(b, p, pn, n, ${masks})); }`).join('\n')}
    default: { }
  }
` : ''}  return out;
}
`;
  const wgsl = [
    `// sdff: the per-pixel SDF shading interpreter, ${bodies.length} material graphs (@etcito/raa-sdfm field-shading.js)`,
    `struct SdfmMaterialFull { ${MATERIAL_FULL_FIELDS.map((x) => `${x}: ${x === 'base_color' || x === 'emissive' ? 'vec3<f32>' : 'f32'}`).join(', ')} }`,
    'struct SdfmMatMasks { curvature: f32, cavity: f32, ao: f32, height: f32, proximity: f32 }',
    ...[...helpers].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)).map(([id, text]) => `// MAMA helper ${id}\n${text}\n`),
    ...graphs,
    dispatch,
    field.trim(),
    FIELD_FRAME_WGSL,
    FIELD_NORMAL_WGSL,
    INTERPRETER_WGSL,
  ].join('\n');
  const forbidden = forbiddenToken(wgsl);
  if (forbidden) fail(`'${forbidden}' is forbidden in a material function (S11)`);
  const bytes = new TextEncoder().encode(wgsl).length;
  if (bytes > MATERIAL_FUNCTION_MAX_BYTES) fail(`the per-pixel SDF shading needs ${bytes} bytes of WGSL (RAA limit ${MATERIAL_FUNCTION_MAX_BYTES})`);
  return { name: `sdff.${textHash(wgsl)}`, wgsl, words: fieldShadingWords(shading, graphOf, bodies, options.debug === true), tintMode, missing };
}

/** The data block of a part (FIELD_DATA_* layout). */
function fieldShadingWords(shading, graphOf, bodies, debug) {
  const ops = shading.operands, program = shading.program;
  const programAt = FIELD_DATA_HEADER, table = programAt + program.length;
  const constantsAt = new Map();
  let size = table + FIELD_OPERAND_WORDS * ops.length;
  for (const [name, g] of graphOf) { constantsAt.set(name, size); size += g.constants.length; }
  const words = new Uint32Array(size), floats = new Float32Array(words.buffer);
  words[0] = FIELD_DATA_VERSION;
  words[1] = (shading.single ? FIELD_FLAG_SINGLE : 0) | (debug ? FIELD_FLAG_DEBUG : 0);
  words[2] = programAt; words[3] = table; words[4] = ops.length;
  const f = shading.frame === undefined ? [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] : Array.from(shading.frame);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) floats[8 + 4 * r + c] = f[4 * c + r];
  floats.set(program, programAt);
  ops.forEach((o, i) => {
    const at = table + FIELD_OPERAND_WORDS * i, g = o.material === null ? undefined : graphOf.get(o.material);
    const k = o.constant;
    floats.set(k ? k.color : o.color, at);
    if (k) {
      words[at + 3] = MODE.constant;
      floats[at + 6] = k.metallic; floats[at + 7] = k.roughness; floats.set(k.emissive, at + 8);
    } else if (g) {
      words[at + 3] = MODE.graph; words[at + 4] = bodies.indexOf(g.body); words[at + 5] = constantsAt.get(o.material);
    } else {
      words[at + 3] = MODE.colour; floats[at + 7] = 0.5;
    }
    floats.set(o.toNode, at + 12);
  });
  for (const [name, g] of graphOf) words.set(g.constants.map((w) => w >>> 0), constantsAt.get(name));
  return words;
}

/**
 * The rest tangent of a rest normal `n`: cross(REST_TANGENT_AXIS, n), not normalised (linear in `n`, so its
 * interpolation across a triangle is the tangent of the interpolated normal; sdfm_posed_normal rebuilds it).
 */
export function restTangent(n) {
  const [ax, ay, az] = REST_TANGENT_AXIS;
  return [ay * n[2] - az * n[1], az * n[0] - ax * n[2], ax * n[1] - ay * n[0]];
}

/**
 * The per-vertex streams of a field-shaded mesh (positions and normals in the part's mesh space, at rest): colours =
 * the rest position with w = the unit rest normal's z, uvs = its x and y, tangents = restTangent of it.
 * @param {Float32Array} positions
 * @param {Float32Array} normals
 * @returns {{ colors: Float32Array, uvs: Float32Array, tangents: Float32Array }}
 */
export function fieldSurfaceAttributes(positions, normals) {
  const count = positions.length / 3;
  const colors = new Float32Array(count * 4), uvs = new Float32Array(count * 2), tangents = new Float32Array(count * 4);
  for (let v = 0; v < count; v++) {
    colors[v * 4] = positions[v * 3]; colors[v * 4 + 1] = positions[v * 3 + 1]; colors[v * 4 + 2] = positions[v * 3 + 2];
    let n = [normals[v * 3], normals[v * 3 + 1], normals[v * 3 + 2]];
    const l = Math.hypot(n[0], n[1], n[2]);
    n = l > 1e-12 ? n.map((c) => c / l) : [0, 0, 1];
    colors[v * 4 + 3] = n[2];
    uvs[v * 2] = n[0]; uvs[v * 2 + 1] = n[1];
    const t = restTangent(n);
    tangents[v * 4] = t[0]; tangents[v * 4 + 1] = t[1]; tangents[v * 4 + 2] = t[2]; tangents[v * 4 + 3] = 1;
  }
  return { colors, uvs, tangents };
}
