// Per-pixel SDF shading of a mesh part, computed in the fragment shader from the part's own field (owner 2026-11-04: the
// normal AND the materials of a game mesh come from the SDF in the shader, nothing is baked). The ONE material code
// path of SDF Modeler's Game view (app/paramesh/sdf-material.ts fieldMaterialWgsl) and of the game (royal-fiesta
// sdfPort.ts): both hand the same PartShading record (MAMA src/core/paramesh/field-wgsl.ts PartShading, stored with a
// saved game mesh part as `shading`) and the same compiled materials (SdfmCompiledMaterial) to fieldPartMaterialWgsl.
// Pure text and array work, no engine call, no MAMA import.
//
// Per fragment: the part's field (`pmf_field`, MAMA field-wgsl.ts) at the rest point gives the shading normal (its
// gradient) and the two heaviest operands with their weight; each operand is shaded as SDF Modeler's Local fields shades
// its label (its material graph at the program-space point p and its node-space point pn, else its colour) and the two
// are mixed by the field's weight. Material boundaries therefore follow the SDF at pixel precision, whatever the mesh.
//
// Data a fragment needs, and where it travels (fieldSurfaceAttributes):
//   - the rest position in the part's mesh space (the field's space): the vertex colour xyz, unclamped f32, passed
//     through RAA's skinning untouched, so a posed skinned part still reads the field at its rest point;
//   - the rest frame: (uv, colour w) = the rest normal as plain xyz, tangent = restTangent = cross(REST_TANGENT_AXIS,
//     rest normal) (handedness +1); RAA moves tangent and normal with the instance and the skin, and the rest-space
//     gradient is re-expressed in that posed frame. Both streams are LINEAR in the rest normal, so a fragment rebuilds
//     exactly the interpolated rest frame RAA interpolated in posed space (an octahedral normal and a Frisvad tangent do
//     not interpolate: across the octahedral fold and the z = 0 tangent seam the rebuilt frame turned up to 160
//     degrees, dark streaks on the -z half of a sphere, SDF Modeler file 34, 2026-11-04).
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

/** MAMA's MaterialFull fields in declaration order (core/material-graph/wgsl-parts.ts); a graph must declare exactly these. */
const MATERIAL_FULL_FIELDS = Object.freeze(['base_color', 'metallic', 'roughness', 'emissive', 'height']);
const CONSTANT_READ = /extra_words\s*\[\s*b\s*\+\s*(\d+)u?\s*\]/g;

/** WGSL f32 literal that reads back as exactly `x` once rounded to f32. */
function f32(x) {
  const v = Math.fround(x);
  if (!Number.isFinite(v)) return v > 0 ? '3.4e38' : v < 0 ? '-3.4e38' : '0.0';
  if (v === 0) return '0.0';
  const [mantissa, exponent] = v.toPrecision(9).split('e');
  const m = mantissa.includes('.') ? mantissa.replace(/0+$/, '').replace(/\.$/, '.0') : `${mantissa}.0`;
  return exponent ? `${m}e${exponent}` : m;
}

const finite = (a, n) => Array.isArray(a) && a.length === n && a.every(Number.isFinite);

/** Checks a PartShading record; returns it, or throws `sdfm.schema` naming the first bad field. */
export function checkPartShading(s) {
  const bad = (what) => { throw new SdfmError('sdfm.schema', `part shading: ${what}`); };
  if (s === null || typeof s !== 'object') bad('not an object');
  if (typeof s.field !== 'string' || !/\bfn\s+pmf_field\s*\(/.test(s.field)) bad('field is not WGSL defining pmf_field (MAMA field-wgsl.ts)');
  if (!Array.isArray(s.operands)) bad('operands is not an array');
  s.operands.forEach((o, i) => {
    if (o === null || typeof o !== 'object') bad(`operands[${i}] is not an object`);
    if (!finite(o.color, 3)) bad(`operands[${i}].color is not 3 finite numbers`);
    if (o.material !== null && (typeof o.material !== 'string' || o.material === '')) bad(`operands[${i}].material is not a name or null`);
    if (!finite(o.toNode, 12)) bad(`operands[${i}].toNode is not 12 finite numbers`);
  });
  if (s.frame !== undefined && !finite(Array.from(s.frame ?? []), 16)) bad('frame is not 16 finite numbers');
  if (typeof s.single !== 'boolean') bad('single is not a boolean');
  if (s.look === null || typeof s.look !== 'object' || !Number.isFinite(s.look.roughness) || !Number.isFinite(s.look.metallic)) bad('look is not { roughness, metallic }');
  return s;
}

/**
 * The material function of a field part: its field normal, and its operands' materials (`materials`: the asset's
 * compiled materials, by name) mixed per pixel by the field's weights. An operand material missing from `materials`
 * wears its colour and is named in `missing`. `single` (one material per object) leaves roughness and metallic to the
 * surface record (`look`). `debug` shows the shading normal as an emissive colour (0.5 + n / 2) on black.
 *
 * @param {object} shading PartShading (MAMA field-wgsl.ts)
 * @param {ReadonlyMap<string, object> | readonly object[]} materials SdfmCompiledMaterial by name, or a list of them
 * @param {{ debug?: boolean }} [options]
 * @returns {{ wgsl: string, tintMode: 'graph' | 'multiply', missing: string[] }}
 * @throws {SdfmError} `sdfm.schema` for a malformed record; `sdfm.material_unmappable` for a material that cannot be
 *   inlined, a forbidden token, or more than MATERIAL_FUNCTION_MAX_BYTES bytes
 */
export function fieldPartMaterialWgsl(shading, materials, options = {}) {
  checkPartShading(shading);
  const byName = materials instanceof Map ? materials : new Map(Array.from(materials, (m) => [m?.name, m]));
  const fail = (what) => { throw new SdfmError('sdfm.material_unmappable', what); };
  const used = [], missing = [];
  for (const o of shading.operands) {
    if (o.material === null || used.includes(o.material) || missing.includes(o.material)) continue;
    (byName.has(o.material) ? used : missing).push(o.material);
  }
  const helpers = new Map();
  let tintMode = 'multiply';
  const graphs = used.map((name, k) => {
    const m = byName.get(name);
    checkShape(m);
    if (m.fields.join() !== MATERIAL_FULL_FIELDS.join()) fail(`material '${name}': MaterialFull fields ${m.fields.join(', ')}; expected ${MATERIAL_FULL_FIELDS.join(', ')}`);
    const n = m.constants.length;
    const structs = (text) => text.replace(/\bMaterialFull\b/g, 'SdfmMaterialFull').replace(/\bMatMasks\b/g, 'SdfmMatMasks');
    const body = structs(m.body.replace(CONSTANT_READ, (read, i) => {
      if (Number(i) >= n) fail(`material '${name}': '${read}' reads past the ${n} constant words`);
      return `SDFM_K${k}[${i}u]`;
    }));
    for (const h of m.helpers) {
      const text = structs(h.wgsl).trim();
      const known = helpers.get(h.id);
      if (known !== undefined && known !== text) fail(`material '${name}': helper '${h.id}' differs from another material's`);
      helpers.set(h.id, text);
    }
    for (const text of [body, ...m.helpers.map((h) => h.wgsl)]) {
      const toks = tokens(text);
      if (toks.includes('mat_instance_tint')) tintMode = 'graph';
      if (toks.includes('extra_words')) fail(`material '${name}': 'extra_words' (a binding-7 read that is not a constant of this graph)`);
      const raa = toks.find((t) => /^raa_/.test(t));
      if (raa) fail(`material '${name}': '${raa}' (identifiers starting with raa_ are reserved)`);
    }
    return [n ? `const SDFM_K${k} = array<u32, ${n}>(${m.constants.map((w) => `${w >>> 0}u`).join(', ')});` : '',
      `fn sdfm_graph_${k}(b: u32, p: vec3<f32>, pn: vec3<f32>, n: vec3<f32>, mk: SdfmMatMasks) -> SdfmMaterialFull {`,
      body.replace(/\s+$/, ''), '}'].filter(Boolean).join('\n');
  });
  const count = Math.max(1, shading.operands.length);
  const ops = shading.operands.length ? shading.operands.flatMap((o) => {
    const k = o.material === null ? -1 : used.indexOf(o.material);
    const r = o.toNode.map(f32);
    return [`vec4<f32>(${o.color.map(f32).join(', ')}, ${f32(k)})`, `vec4<f32>(${r.slice(0, 4).join(', ')})`,
      `vec4<f32>(${r.slice(4, 8).join(', ')})`, `vec4<f32>(${r.slice(8, 12).join(', ')})`];
  }) : ['vec4<f32>(0.5, 0.5, 0.5, -1.0)', 'vec4<f32>(1.0, 0.0, 0.0, 0.0)', 'vec4<f32>(0.0, 1.0, 0.0, 0.0)', 'vec4<f32>(0.0, 0.0, 1.0, 0.0)'];
  const f = shading.frame === undefined ? null : Array.from(shading.frame);
  const toProgram = f === null ? 'return q;'
    : `return vec3<f32>(${f32(f[0])} * q.x + ${f32(f[4])} * q.y + ${f32(f[8])} * q.z + ${f32(f[12])}, ${f32(f[1])} * q.x + ${f32(f[5])} * q.y + ${f32(f[9])} * q.z + ${f32(f[13])}, ${f32(f[2])} * q.x + ${f32(f[6])} * q.y + ${f32(f[10])} * q.z + ${f32(f[14])});`;
  const cases = used.map((_, k) => `    case ${k}: { return sdfm_graph_${k}(0u, p, pn, n, SdfmMatMasks(0.5, 1.0, 1.0, 0.0, 1.0)); }`);
  const debug = options.debug ? `
  o.base_color = vec3<f32>(0.0);
  o.metallic = 0.0;
  o.roughness = 1.0;
  o.emissive = o.normal * 0.5 + vec3<f32>(0.5);` : '';
  const wgsl = [
    `// sdff: per-pixel SDF shading, ${shading.operands.length} field operands, ${used.length} material graphs (@etcito/raa-sdfm field-shading.js)`,
    `struct SdfmMaterialFull { ${MATERIAL_FULL_FIELDS.map((x) => `${x}: ${x === 'base_color' || x === 'emissive' ? 'vec3<f32>' : 'f32'}`).join(', ')} }`,
    'struct SdfmMatMasks { curvature: f32, cavity: f32, ao: f32, height: f32, proximity: f32 }',
    ...[...helpers].map(([id, text]) => `// MAMA helper ${id}\n${text}\n`),
    ...graphs,
    shading.field,
    FIELD_FRAME_WGSL,
    FIELD_NORMAL_WGSL,
    `var<private> sdff_ops: array<vec4<f32>, ${count * 4}> = array<vec4<f32>, ${count * 4}>(${ops.join(', ')});

// Local fields' evalLabel for field operand i: its material graph at p and its node-space point, else its colour.
fn sdff_label(i: u32, p: vec3<f32>, n: vec3<f32>) -> SdfmMaterialFull {
  if (i >= ${count}u) { return SdfmMaterialFull(vec3<f32>(0.5), 0.0, 0.5, vec3<f32>(0.0), 0.0); }
  let o = i * 4u;
  let h = sdff_ops[o];
${used.length ? `  let r0 = sdff_ops[o + 1u];
  let r1 = sdff_ops[o + 2u];
  let r2 = sdff_ops[o + 3u];
  let pn = vec3<f32>(dot(r0.xyz, p) + r0.w, dot(r1.xyz, p) + r1.w, dot(r2.xyz, p) + r2.w);
  switch i32(h.w) {
${cases.join('\n')}
    default: { }
  }
` : ''}  return SdfmMaterialFull(h.xyz, 0.0, 0.5, vec3<f32>(0.0), 0.0);
}

fn sdff_program_point(q: vec3<f32>) -> vec3<f32> {
  ${toProgram}
}

fn material_main(in: MaterialInput) -> MaterialOutput {
  var o = raa_default_output(in);
  o.opacity = 1.0; // the vertex colour's w is the rest normal's z, not an alpha
  let s = pmf_field(in.color.xyz);
  o.normal = sdff_normal(in, s);
  let p = sdff_program_point(in.color.xyz);
  var m = sdff_label(s.la, p, o.normal);
  if (s.lb != s.la && s.t > 0.0) {
    let b = sdff_label(s.lb, p, o.normal);
    m = SdfmMaterialFull(mix(m.base_color, b.base_color, s.t), mix(m.metallic, b.metallic, s.t),
      mix(m.roughness, b.roughness, s.t), mix(m.emissive, b.emissive, s.t), mix(m.height, b.height, s.t));
  }
  o.base_color = m.base_color;
  o.emissive = m.emissive;
${shading.single ? '' : '  o.metallic = m.metallic;\n  o.roughness = m.roughness;\n'}${debug}
  return o;
}
`].join('\n');
  const forbidden = forbiddenToken(wgsl);
  if (forbidden) fail(`'${forbidden}' is forbidden in a material function (S11)`);
  const bytes = new TextEncoder().encode(wgsl).length;
  if (bytes > MATERIAL_FUNCTION_MAX_BYTES) fail(`the per-pixel SDF shading needs ${bytes} bytes of WGSL (RAA limit ${MATERIAL_FUNCTION_MAX_BYTES})`);
  return { wgsl, tintMode, missing };
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
