// Per-pixel SDF shading of a mesh part, computed in the fragment shader from the part's own field (owner 2026-11-04: the
// normal AND the materials of a game mesh come from the SDF in the shader, nothing is baked). The ONE material code
// path of SDF Modeler's Game view and of the host game: both hand the same PartShading record
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
//
// Two effects of the Game view's facades (owner 2026-11-17), both computed here from data, the text stays fixed:
//   - parallax on recesses (FIELD_FLAG_RELIEF, MAMA view 'relief'): the mesh carries no geometry for a relief shape (a
//     window reveal), the field does. The operand at the face point names its copy of the relief module (the group of
//     the shape, MAMA ReliefModule); the fragment marches the eye ray through THAT module's field (a handful of
//     operands, not the part's hundreds) from the face to the recess (at most RELIEF_STEPS evaluations, bounded by the
//     module's depth) and shades the hit: its normal is the module field's gradient, its material the part operand of
//     the same copy. Nothing baked.
//   - interior mapping behind window panes (an operand with an interior record, MAMA material.output surface
//     'interior'): the eye ray through the pane is intersected with a room box behind it (closed form, five faces) and
//     the exit point is projected to the room picture, a pre-projected tile rendered from the interior scene (SDF Modeler
//     src/app/interiors/interior-bake.ts) and loaded into RAA's material atlas; the tile, a mirror, lights on or off,
//     blinds and a brightness spread follow a hash of the operand, the room index along the pane, the pane's rest-space centre and the
//     instance's cell seed (INSTANCE_PARAM_SLOTS.seed). The room is emitted behind the glass; the glass keeps its own look. Until the
//     caller patches the tile ids (`interiors` of fieldPartShading) the pane shows its glass only.
import { SdfmError } from './errors.js';
import { FOOTPRINT_WGSL, checkShape, declaresFootprint, forbiddenToken, materialPages, pagesBaseOf, textHash, tokens, wordReads } from './map-material.js';

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
 *   [5] the relief depth f32 (FIELD_FLAG_RELIEF; 0 otherwise), [6] the relief table offset (0 without), [7] 0,
 *   [8..19] the part -> program frame as 3 rows of 4 f32 (identity when absent), then the field program, the operand
 *   table (FIELD_OPERAND_WORDS per operand), the graph constants, the interior records and the relief data.
 * The relief copies: [0] their count, then 2 words per copy: its module record offset and the index of its first operand.
 *   A module record (RELIEF_MODULE_WORDS, then its program): [0] program offset, [1] operand count, [2] depth f32,
 *   [3] 0, [4..15] the first operand's node space -> the module's program space (3 rows of 4 f32), [16..21] the
 *   module's box in its program space (min xyz, max xyz), [22..23] 0.
 * An operand: [0..2] colour f32, [3] mode (0 colour, 1 constant, 2 graph), [4] graph index, [5] constants offset,
 *   [6] metallic, [7] roughness, [8..10] emissive, [11] its interior record offset (0 = not a pane), [12..23] the
 *   program -> node rows (3 x 4 f32).
 * An interior record (INTERIOR_WORDS + 2 per tile): [0..2] the pane's half size in its node space (the pane faces +z,
 *   the room goes to -z from its front face), [3] room width, [4] room height (0 = the instance's cell, else the pane),
 *   [5] room depth, [6] glass, [7] tile count (0 = the set is unknown: glass only), [8] lights on, [9] blinds,
 *   [10] unlit level, [11] emissive scale, then per tile (INTERIOR_TILE_WORDS) the texture ids of its full and its small
 *   picture (INTERIOR_NO_TEXTURE until loaded) and its back-wall fraction f32 (the share of the picture its back wall
 *   fills: 0.5 for SDF Modeler's bake, the camera one room depth in front; a reference picture states its own).
 * Version 2 blocks carry relief or interiors; a part with neither packs exactly the version 1 block.
 */
export const FIELD_DATA_VERSION = 2;
export const FIELD_DATA_HEADER = 20;
export const FIELD_OPERAND_WORDS = 24;
/** Flags of word 1: one material per object (roughness and metallic from the record), the normal debug view, parallax. */
export const FIELD_FLAG_SINGLE = 1;
export const FIELD_FLAG_DEBUG = 2;
export const FIELD_FLAG_RELIEF = 4;
/** Words of an interior record before its tile ids. */
export const INTERIOR_WORDS = 12;
/** Texture id of a room picture not loaded yet: the pane shows its glass. */
export const INTERIOR_NO_TEXTURE = 0xffffffff;
/** Words per tile of an interior record: full id, small id, back-wall fraction. */
export const INTERIOR_TILE_WORDS = 3;
/** Sphere-tracing steps of the relief march. */
export const RELIEF_STEPS = 32;
/** A relief march step is at most reach / RELIEF_SPAN (the program distances are estimates, see the march). */
export const RELIEF_SPAN = 12;
/** Words of a relief module record before its program. */
export const RELIEF_MODULE_WORDS = 24;
/** Interior defaults (MAMA interiors.ts INTERIORS_DEFAULTS) when the caller gives no interiors. */
const INTERIOR_DEFAULTS = Object.freeze({ lightsOn: 0.6, blinds: 0.3, unlitLevel: 0.15, emissive: 1 });

const MODE = { colour: 0, constant: 1, graph: 2 };

/**
 * `sdfm_posed_normal(in, g)`: a rest-space normal `g` carried onto the drawn (posed, world) surface of the fragment
 * through the rest frame. The interpolated rest normal and its tangent cross(axis, normal) are rebuilt unnormalised,
 * i.e. exactly the vectors RAA interpolated in posed space, then both frames are orthonormalised the same way.
 */
export const FIELD_FRAME_WGSL = /* wgsl */ `
const SDFM_TANGENT_AXIS = vec3f(${REST_TANGENT_AXIS.map((v) => Math.fround(v).toPrecision(9)).join(', ')});

fn sdfm_pose_normal(in: MaterialInput, g: vec3f, validate_field: bool) -> vec3f {
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
  // A field gradient can belong to another surface on a thin wall or a coarse mesh. Test it before facing it to the view.
  if (validate_field && !(dot(m, n) >= 0.5)) { return n; }
  // At the silhouette the SDF normal of a facet the camera sees can lean past the view: keep it visible (no dark rim).
  let d = dot(m, in.view_dir);
  if (d < 0.02 && dot(n, in.view_dir) > 0.0) { m = normalize(m + in.view_dir * (0.02 - d)); }
  return m;
}

fn sdfm_posed_normal(in: MaterialInput, g: vec3f) -> vec3f {
  return sdfm_pose_normal(in, g, false);
}

// The inverse: a world (posed) direction v re-expressed in the rest frame, the space of the field. A rotation only: with
// a non-uniformly scaled instance the direction is approximate (owner 2026-11-04: "une approximation suffisante").
fn sdfm_rest_dir(in: MaterialInput, v: vec3f) -> vec3f {
  let n = normalize(in.normal);
  let nr0 = vec3f(in.uv, in.color.w);
  if (dot(nr0, nr0) < 1e-12) { return v; }
  let nr = normalize(nr0);
  let tr0 = cross(SDFM_TANGENT_AXIS, nr0);
  let trp = tr0 - nr * dot(nr, tr0);
  let tw = in.tangent.xyz - n * dot(n, in.tangent.xyz);
  if (dot(trp, trp) < 1e-8 || dot(tw, tw) < 1e-12) { return v; }
  let tr = normalize(trp);
  let br = cross(nr, tr);
  let t = normalize(tw);
  let bw = cross(n, t) * select(-1.0, 1.0, in.tangent.w >= 0.0);
  return normalize(dot(v, t) * tr + dot(v, bw) * br + dot(v, n) * nr);
}

// The eye ray into the surface in rest space, EXACT on a face: the rest point q0 is an affine function of the world
// point on a face (one Paramesh face, one copy), so the screen derivatives give that map on the face's plane (px, py ->
// qx, qy) and its scale along the normal (the area ratio, a similarity). sdfm_rest_dir rotates by the interpolated mesh
// frame instead, and that frame turns across a large planar face whose vertices are shared with faces of another
// orientation (measured on the New Surugadai shop pane, 2026-11-18: its tangent swept from pink to cyan to yellow and
// its rest normal left the face normal over one 4.2 m pane, so the rooms behind it bent). Where the derivatives
// straddle two faces (the two screen axes scale differently) or vanish, the frame rotation is kept. Called at the top of
// material_main (uniform flow).
fn sdff_rest_ray(in: MaterialInput, q0: vec3f) -> vec3f {
  let rot = sdfm_rest_dir(in, -in.view_dir);
  let px = dpdx(in.position);
  let py = dpdy(in.position);
  let qx = dpdx(q0);
  let qy = dpdy(q0);
  let nw0 = cross(px, py);
  let nq0 = cross(qx, qy);
  let aw = length(nw0);
  let aq = length(nq0);
  if (!(aw > 1e-14) || !(aq > 1e-14)) { return rot; }
  let nw = nw0 / aw;
  let nq = nq0 / aq;
  // A similarity scales both screen axes alike; a 2x2 quad straddling two faces (a resolve pass) does not.
  let sx = dot(qx, qx) / max(dot(px, px), 1e-30);
  let sy = dot(qy, qy) / max(dot(py, py), 1e-30);
  if (abs(sx - sy) > 0.04 * max(sx, sy)) { return rot; }
  let w = -in.view_dir;
  let c = dot(w, nw);
  let tw = w - c * nw;
  let g11 = dot(px, px);
  let g12 = dot(px, py);
  let g22 = dot(py, py);
  let det = g11 * g22 - g12 * g12;
  if (!(det > 1e-28)) { return rot; }
  let r1 = dot(tw, px);
  let r2 = dot(tw, py);
  let a = (r1 * g22 - r2 * g12) / det;
  let b = (r2 * g11 - r1 * g12) / det;
  let d = a * qx + b * qy + c * sqrt(aq / aw) * nq;
  if (!(dot(d, d) > 1e-20)) { return rot; }
  return normalize(d);
}
`;

/** `sdff_normal(in, s)`: the field's normal at the fragment, on the posed surface (needs FIELD_FRAME_WGSL and the field). */
export const FIELD_NORMAL_WGSL = /* wgsl */ `
fn sdff_normal(in: MaterialInput, s: PmfSample) -> vec3f {
  let mesh = normalize(in.normal);
  let len = length(s.g);
  if (!(len > 0.0) || len > 3.0e38) { return mesh; }
  let m = sdfm_pose_normal(in, s.g / len, true);
  // Keep the corrected normal within 60 degrees of the mesh at the silhouette too.
  return select(mesh, m, dot(m, mesh) >= 0.5);
}
`;

/**
 * The interpreter's fixed core: data readers and the operand shading (after the field and the graphs). The per-pixel
 * material (INTERPRETER_MAIN_WGSL) and the UV-space bake (field-bake.js) both run it. The field sample `s` that
 * material_main already computed is the only field read: the normal and the material both consume it, and the material
 * reads the two labels of that sample through one call site (the shader compiler inlines every call, so a second
 * sdff_label site would inline the whole graph dispatch and its material bodies again).
 */
const INTERPRETER_CORE_WGSL = /* wgsl */ `
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
fn sdff_rows_linear(i: u32, v: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(dot(sdff_v3(i), v), dot(sdff_v3(i + 4u), v), dot(sdff_v3(i + 8u), v));
}
// The transpose of the rows' linear part: a gradient back through the map.
fn sdff_rows_transposed(i: u32, g: vec3<f32>) -> vec3<f32> {
  return sdff_v3(i) * g.x + sdff_v3(i + 4u) * g.y + sdff_v3(i + 8u) * g.z;
}

// The rest-space shading normal sdff_material leaves when a graph bent it (bump or authored normal, sdff_bent), else the
// caller keeps its field normal; sdff_label_n is the normal of the last label read (program space).
var<private> sdff_bent: bool = false;
var<private> sdff_rest_n: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);
var<private> sdff_label_n: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);

// Local fields' evalLabel for field operand i of the block at base: its constant values, its material graph at p and
// its node-space point, else its colour. table, count and p are read once by sdff_material, shared by both labels; the
// colour is read only when the operand has no graph or its graph fails, the node point only when a graph uses pn.
// np = the field normal in program space (the frame of p), which a graph's bump or authored normal bends (sdff_label_n).
fn sdff_label(base: u32, table: u32, count: u32, i: u32, p: vec3<f32>, n: vec3<f32>, np: vec3<f32>) -> SdfmMaterialFull {
  sdff_label_n = np;
  if (i >= count) { return SdfmMaterialFull(vec3<f32>(0.5), 0.0, 0.5, vec3<f32>(0.0), 0.0); }
  let o = base + table + ${FIELD_OPERAND_WORDS}u * i;
  let mode = sdff_w(o + 3u);
  if (mode == ${MODE.constant}u) { return SdfmMaterialFull(sdff_v3(o), pmf_f(o + 6u), pmf_f(o + 7u), sdff_v3(o + 8u), 0.0); }
  if (mode == ${MODE.graph}u) {
__SDFF_GRAPH_CALL__
  }
  return SdfmMaterialFull(sdff_v3(o), 0.0, 0.5, vec3<f32>(0.0), 0.0);
}

fn sdff_mix(a: SdfmMaterialFull, b: SdfmMaterialFull, t: f32) -> SdfmMaterialFull {
  return SdfmMaterialFull(mix(a.base_color, b.base_color, t), mix(a.metallic, b.metallic, t),
    mix(a.roughness, b.roughness, t), mix(a.emissive, b.emissive, t), mix(a.height, b.height, t));
}

// The material of the field at rest point q: the heaviest operand of the sample s, mixed with the second by the field's
// weight (both through ONE sdff_label site); the second label is read only when it differs and its weight is above 0.
// A graph's bump or authored normal bends the label normals (mixed like the materials): sdff_bent and sdff_rest_n.
fn sdff_material(base: u32, s: PmfSample, q: vec3<f32>, n: vec3<f32>) -> SdfmMaterialFull {
  let p = sdff_affine(base + 8u, q);
  let table = sdff_w(base + 3u);
  let count = sdff_w(base + 4u);
  let labels = select(1u, 2u, s.lb != s.la && s.t > 0.0);
  sdff_bent = false;
  let gl = length(s.g);
  let field = gl > 0.0 && gl < 3.0e38;
  // The frame of p is a similarity of the rest frame: a normal maps by its linear part, and back by its transpose.
  let np = select(vec3<f32>(0.0, 0.0, 1.0), normalize(sdff_rows_linear(base + 8u, s.g / max(gl, 1e-30))), field);
  var nm = np;
  var m = SdfmMaterialFull(vec3<f32>(0.0), 0.0, 0.0, vec3<f32>(0.0), 0.0);
  for (var k = 0u; k < labels; k = k + 1u) {
    let label = sdff_label(base, table, count, select(s.lb, s.la, k == 0u), p, n, np);
    if (k == 0u) { m = label; nm = sdff_label_n; } else { m = sdff_mix(m, label, s.t); nm = mix(nm, sdff_label_n, s.t); }
  }
  sdff_bent = sdff_bent && field && dot(nm, nm) > 1e-20;
  if (sdff_bent) { sdff_rest_n = normalize(sdff_rows_transposed(base + 8u, normalize(nm))); }
  return m;
}
`;

/**
 * `sdff_interior(in, base, i, q, rd)`: the room seen through the pane of operand i at rest point q along the rest-space
 * eye ray rd (into the surface): rgb = its emission (before the glass), a = the glass share; a = -1 when operand i is
 * not a pane, the ray leaves the pane, or no picture is loaded. The room box is centred on the pane (or on its room
 * along a long pane) in the pane's node space; the picture is the room seen by a camera on the room axis one room depth
 * in front of the opening, the opening filling the frame (interior-bake.ts renders the same camera).
 */
export const INTERIOR_WGSL = /* wgsl */ `
fn sdff_linear(i: u32, v: vec3<f32>) -> vec3<f32> { return sdff_rows_linear(i, v); }

fn sdff_hash(a: u32) -> u32 {
  var x = a * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (x >> 22u) ^ x;
}

fn sdff_unit(h: u32) -> f32 { return f32(h & 0xffffu) / 65535.0; }

fn sdff_interior(in: MaterialInput, base: u32, i: u32, q: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
  let none = vec4<f32>(0.0, 0.0, 0.0, -1.0);
  if (i >= sdff_w(base + 4u)) { return none; }
  let o = base + sdff_w(base + 3u) + ${FIELD_OPERAND_WORDS}u * i;
  let at = sdff_w(o + 11u);
  if (at == 0u) { return none; }
  let r = base + at;
  let tiles = sdff_w(r + 7u);
  if (tiles == 0u) { return none; }
  let half = sdff_v3(r);
  let pn = sdff_affine(o + 12u, sdff_affine(base + 8u, q));
  let dn0 = sdff_linear(o + 12u, sdff_linear(base + 8u, rd));
  if (!(dot(dn0, dn0) > 0.0)) { return none; }
  let dn = normalize(dn0);
  if (dn.z > -1e-3) { return none; }
  // The eye ray on the pane's front plane; a fragment on the pane's edges shows no room.
  let e0 = pn + dn * ((half.z - pn.z) / dn.z);
  if (abs(e0.x) > half.x * 1.001 || abs(e0.y) > half.y * 1.001) { return none; }
  let cell = vec2<f32>(in.instance_params[1].w, in.instance_params[2].x);
  let wanted = vec2<f32>(pmf_f(r + 3u), pmf_f(r + 4u));
  var size = select(select(2.0 * half.xy, cell, cell > vec2<f32>(0.0)), wanted, wanted > vec2<f32>(0.0));
  // A pane wider than 1.5 rooms (a long shop window) holds whole rooms side by side.
  let rooms = select(1.0, max(1.0, round(2.0 * half.x / size.x)), 2.0 * half.x > 1.5 * size.x);
  if (rooms > 1.0) { size.x = 2.0 * half.x / rooms; }
  let k = clamp(floor((e0.x + half.x) / size.x), 0.0, rooms - 1.0);
  let cx = select(0.0, -half.x + (k + 0.5) * size.x, rooms > 1.0);
  let depth = max(pmf_f(r + 5u), 0.1);
  let lo = vec3<f32>(cx - 0.5 * size.x, -0.5 * size.y, half.z - depth);
  let hi = vec3<f32>(cx + 0.5 * size.x, 0.5 * size.y, half.z);
  let tx = (select(lo.x, hi.x, dn.x > 0.0) - e0.x) / select(dn.x, 1e-6, abs(dn.x) < 1e-6);
  let ty = (select(lo.y, hi.y, dn.y > 0.0) - e0.y) / select(dn.y, 1e-6, abs(dn.y) < 1e-6);
  let tz = (lo.z - e0.z) / dn.z;
  let e = e0 + dn * max(min(min(abs(tx), abs(ty)), tz), 0.0);
  // The pane's centre in rest space tells two copies of one pane apart when they share an operand and an instance (a
  // building merged into one draw carries no cell seed): q minus the pane-space offset mapped back, on a 0.5 m grid.
  let c0 = sdff_linear(o + 12u, sdff_linear(base + 8u, vec3<f32>(1.0, 0.0, 0.0)));
  let c1 = sdff_linear(o + 12u, sdff_linear(base + 8u, vec3<f32>(0.0, 1.0, 0.0)));
  let c2 = sdff_linear(o + 12u, sdff_linear(base + 8u, vec3<f32>(0.0, 0.0, 1.0)));
  let det = dot(c0, cross(c1, c2));
  var cellKey = 0u;
  if (abs(det) > 1e-12) {
    let off = pn - vec3<f32>(cx, 0.0, 0.0);
    let qc = vec3<i32>(round((q - vec3<f32>(dot(cross(c1, c2), off), dot(cross(c2, c0), off), dot(cross(c0, c1), off)) / det) * 2.0));
    cellKey = sdff_hash(bitcast<u32>(qc.x) ^ sdff_hash(bitcast<u32>(qc.y) ^ sdff_hash(bitcast<u32>(qc.z))));
  }
  let h = sdff_hash(i * 7919u + u32(k) * 104729u + (sdff_hash(u32(max(in.instance_params[1].z, 0.0))) ^ cellKey));
  let tile = h % tiles;
  let tw = r + ${INTERIOR_WORDS}u + ${INTERIOR_TILE_WORDS}u * tile;
  // The pre-projection camera: on the room axis, at distance c in front of the opening, the opening filling the
  // picture; the back wall then fills f = c / (c + depth) of it, so c = depth * f / (1 - f) (f = 0.5: c = depth).
  let f = clamp(pmf_f(tw + 2u), 0.05, 0.95);
  let cam = depth * f / (1.0 - f);
  let back = cam + (half.z - e.z);
  let ndc = (e.xy - vec2<f32>(cx, 0.0)) * cam / (back * 0.5 * size);
  var uv = vec2<f32>(0.5 + 0.5 * ndc.x, 0.5 - 0.5 * ndc.y);
  if ((h & 0x100u) != 0u) { uv.x = 1.0 - uv.x; }
  // Far panes read the small picture (no mipmaps in the atlas): a pixel larger than two texels of the full picture.
  let dist = length(raa_frame.camera_pos.xyz - in.position);
  let pixel = dist * 2.0 * raa_frame.viewport.w / max(raa_frame.proj[1][1], 1e-4);
  let full = sdff_w(tw);
  let small = sdff_w(tw + 1u);
  var id = select(full, small, pixel > 2.0 * size.y / 512.0 && small != ${INTERIOR_NO_TEXTURE}u);
  if (id == ${INTERIOR_NO_TEXTURE}u) { id = small; }
  if (id == ${INTERIOR_NO_TEXTURE}u) { return none; }
  var c = raa_srgb_to_linear(raa_texture(id, uv).rgb);
  let h2 = sdff_hash(h ^ 0x9e3779b9u);
  let lit = sdff_unit(h2) < pmf_f(r + 8u);
  var level = select(pmf_f(r + 10u), 1.0, lit) * (0.85 + 0.3 * sdff_unit(h2 >> 16u));
  // Blinds: a band from the top of the pane, slats every 4 cm, lit by the room.
  let h3 = sdff_hash(h2 + 0x632be5abu);
  if (sdff_unit(h3) < pmf_f(r + 9u)) {
    let bottom = half.y - (0.15 + 0.65 * sdff_unit(h3 >> 16u)) * 2.0 * half.y;
    if (e0.y > bottom) {
      let slat = 0.8 + 0.2 * smoothstep(0.0, 0.3, fract(e0.y * 25.0));
      c = vec3<f32>(0.62, 0.6, 0.55) * slat * select(0.35, 0.8, lit);
      level = 1.0;
    }
  }
  return vec4<f32>(c * level * pmf_f(r + 11u), pmf_f(r + 6u));
}
`;

/** The per-pixel material function's entry point (after FIELD_FRAME_WGSL, FIELD_NORMAL_WGSL, the core and INTERIOR_WGSL). */
const INTERPRETER_MAIN_WGSL = /* wgsl */ `
fn material_main(in: MaterialInput) -> MaterialOutput {
  var o = raa_default_output(in);
  o.opacity = 1.0; // the vertex colour's w is the rest normal's z, not an alpha
  let base = raa_material_data_base(in.record);
  let flags = sdff_w(base + 1u);
  let program = base + sdff_w(base + 2u);
  let q0 = in.color.xyz;
  // The eye ray in rest space, into the surface (the relief march and the rooms behind panes follow it).
  let rd = sdff_rest_ray(in, q0);
  let relief = (flags & ${FIELD_FLAG_RELIEF}u) != 0u;
  // ONE pmf_eval site. Sample 0: the part's field at the face point (THE sample without relief). With relief, the
  // operand there names its copy of a relief module (the relief table); the next samples march that module's field.
  // The program distances are first-order estimates (implicit value / gradient length, field-program-wgsl.ts): exact at
  // the surface, too long away from it (measured on the New Surugadai bay: 0.5 m at a pane 0.23 m behind the face). So
  // a step is at most 1/RELIEF_SPAN of the reach, and the first sample inside the solid starts a bisection between it
  // and the last sample outside: the hit never lands behind a pane or a mullion it stepped over.
  let nr0 = vec3<f32>(in.uv, in.color.w);
  let into = max(-dot(rd, normalize(select(vec3<f32>(0.0, 0.0, 1.0), nr0, dot(nr0, nr0) > 1e-12))), 0.05);
  var prog = program;
  var pm = q0;
  var pm0 = q0;
  var dm = rd;
  var dscale = 1.0;
  var mrec = 0u;
  var start = 0u;
  var depth = 0.0;
  var eps = 1e-4;
  var reach = 0.0;
  var cap = 0.0;
  var t = 0.0;
  var lo = 0.0;
  var hi = 0.0;
  var bisect = false;
  var lost = false;
  var s: PmfSample;
  var s0: PmfSample;
  let steps = select(1u, ${RELIEF_STEPS}u, relief);
  for (var k = 0u; k < steps; k = k + 1u) {
    s = pmf_eval(prog, pm);
    if (k == 0u) {
      s0 = s;
      let list = sdff_w(base + 6u);
      if (!relief || s.d < 1e-4 || list == 0u) { break; }
      // The copy whose module box holds the face point (part -> module: the frame, the copy's first operand's toNode,
      // the module record's rows).
      let pf = sdff_affine(base + 8u, q0);
      let copies = sdff_w(base + list);
      for (var c = 0u; c < copies; c = c + 1u) {
        let rc = sdff_w(base + list + 1u + 2u * c);
        let sc = sdff_w(base + list + 2u + 2u * c);
        let p = sdff_affine(base + rc + 4u, sdff_affine(base + sdff_w(base + 3u) + ${FIELD_OPERAND_WORDS}u * sc + 12u, pf));
        if (all(p >= sdff_v3(base + rc + 16u) - vec3<f32>(1e-3)) && all(p <= sdff_v3(base + rc + 19u) + vec3<f32>(1e-3))) {
          mrec = rc; start = sc; pm0 = p; break;
        }
      }
      if (mrec == 0u) { break; }
      let o0 = base + sdff_w(base + 3u) + ${FIELD_OPERAND_WORDS}u * start;
      let r = base + mrec;
      dm = sdff_rows_linear(r + 4u, sdff_rows_linear(o0 + 12u, sdff_rows_linear(base + 8u, rd)));
      dscale = max(length(dm), 1e-6);
      prog = base + sdff_w(r);
      depth = pmf_f(r + 2u);
      eps = max(depth * 0.004, 1e-4);
      reach = depth / into;
      cap = reach / ${RELIEF_SPAN}.0;
      pm = pm0;
      continue;
    }
    // The module's distance in part units along the ray (its map scales by dscale along dm).
    let d = s.d / dscale;
    if (bisect) {
      if (abs(d) < eps) { break; }
      if (d < 0.0) { hi = t; } else { lo = t; }
      t = 0.5 * (lo + hi);
    } else if (d < 0.0) {
      if (k == 1u) { lost = true; break; } // the face point is inside the module's solid: no recess here
      bisect = true; hi = t; t = 0.5 * (lo + hi);
    } else if (d < eps) {
      if (k == 1u) { lost = true; } // the face itself: no recess on this ray
      break;
    } else {
      lo = t;
      t = t + min(d, cap);
      if (t > reach) { lost = true; break; }
    }
    pm = pm0 + dm * t;
  }
  // No module, past the depth bound or at the face: the face sample. Out of steps: the last module sample (bisecting,
  // within a few millimetres of the surface; marching, a ray grazing a reveal, already close to the side it nears).
  let moduleHit = mrec != 0u && !lost;
  if (!moduleHit) { s = s0; t = 0.0; }
  else {
    // The module's sample read as the part's: its labels are the copy's operands, its gradient back to part space.
    let o0 = base + sdff_w(base + 3u) + ${FIELD_OPERAND_WORDS}u * start;
    let g = sdff_rows_transposed(base + 8u, sdff_rows_transposed(o0 + 12u, sdff_rows_transposed(base + mrec + 4u, s.g)));
    let count = sdff_w(base + mrec + 1u);
    s = PmfSample(s.d / dscale, g, start + min(s.la, count - 1u), start + min(s.lb, count - 1u), s.t);
    t = length(pm - pm0) / dscale;
  }
  let q = q0 + rd * t;
  let moved = t > 0.0;
  if (moved) {
    // The recess surface: its own field normal (no 60-degree rule, a reveal's side is 90 degrees off the face) and an
    // ambient occlusion that deepens with the depth behind the face (ambient only, RAA's AO rule).
    let len = length(s.g);
    if (len > 0.0 && len < 3.0e38) { o.normal = sdfm_posed_normal(in, s.g / len); }
    o.ao = 1.0 - 0.55 * clamp(t * into / max(depth, 1e-4), 0.0, 1.0);
  } else {
    o.normal = sdff_normal(in, s);
  }
  if ((flags & ${FIELD_FLAG_DEBUG}u) != 0u) {
    // The normal debug view shows no material: the one sample already gives its normal, so skip the label reads.
    o.base_color = vec3<f32>(0.0);
    o.metallic = 0.0;
    o.roughness = 1.0;
    o.emissive = o.normal * 0.5 + vec3<f32>(0.5);
    return o;
  }
__SDFF_FOOTPRINT__  let m = sdff_material(base, s, q, o.normal);
  if (sdff_bent) { o.normal = sdfm_posed_normal(in, sdff_rest_n); }
  o.base_color = m.base_color;
  o.emissive = m.emissive;
  if (in.instance_params[0].w > 0.5) { o.base_color = o.base_color * in.instance_params[0].xyz; }
  if ((flags & ${FIELD_FLAG_SINGLE}u) == 0u) {
    o.metallic = m.metallic;
    o.roughness = m.roughness;
  }
  // A window pane: the room behind it, emitted through the glass (the glass keeps its colour, roughness and reflection).
  let room = sdff_interior(in, base, s.la, q, rd);
  if (room.w >= 0.0) { o.emissive = o.emissive + room.rgb * (1.0 - room.w); }
  return o;
}
`;

/** MAMA's MaterialFull fields in declaration order (core/material-graph/wgsl-parts.ts); a graph must declare exactly these. */
const MATERIAL_FULL_FIELDS = Object.freeze(['base_color', 'metallic', 'roughness', 'emissive', 'height']);
const CONSTANT_READ = /extra_words\s*\[\s*b\s*\+\s*(\d+)u?\s*\]/g;

const finite = (a, n) => Array.isArray(a) && a.length === n && a.every(Number.isFinite);

/**
 * `text` with every `extra_words[i]` (MAMA's binding-7 reads: the graph's constants at `b` and indices derived from it,
 * such as a ramp's stop offsets, and the image pages its constants point to) read from the material data words instead:
 * `sdff_w(i)`. The graph's constants sit at its operand's constant offset in the part's block, which the interpreter
 * passes as `b`; the pages sit in the host's page block (options.pagesBase).
 */
const dataReads = (text) => wordReads(text, 'sdff_w');

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
  if (s.relief !== undefined) {
    const r = s.relief;
    if (r === null || typeof r !== 'object' || !(r.depth > 0) || !Number.isFinite(r.depth)) bad('relief is not { depth > 0, modules, copies }');
    if (!Array.isArray(r.modules) || !r.modules.length) bad('relief.modules is not a non-empty array');
    r.modules.forEach((m, k) => {
      if (!m || !Array.isArray(m.program) || !m.program.every(Number.isFinite)) bad(`relief.modules[${k}].program is not finite numbers`);
      if (!finite(m.rows, 12)) bad(`relief.modules[${k}].rows is not 12 finite numbers`);
      if (!Number.isInteger(m.count) || m.count < 1 || !(m.depth > 0)) bad(`relief.modules[${k}] needs count >= 1 and depth > 0`);
      if (!finite(m.box, 6)) bad(`relief.modules[${k}].box is not 6 finite numbers`);
    });
    const n = Array.isArray(s.operands) ? s.operands.length : 0;
    if (!Array.isArray(r.copies) || r.copies.length !== 2 * n || !r.copies.every((v, i) => Number.isInteger(v)
      && (i % 2 === 0 ? v >= -1 && v < r.modules.length : v >= -1 && v < n))) bad('relief.copies is not 2 indices per operand');
  }
  s.operands.forEach((o, i) => {
    const n = o.interior;
    if (n === undefined) return;
    if (n === null || typeof n !== 'object' || typeof n.set !== 'string' || !finite(n.half, 3) || ![n.depth, n.width, n.height, n.glass].every(Number.isFinite)) {
      bad(`operands[${i}].interior is not { set, depth, width, height, glass, half: [x, y, z] }`);
    }
  });
  return s;
}

/**
 * @typedef {{ sets: Record<string, { tiles: number, depth: number }>, lightsOn?: number, blinds?: number, unlitLevel?: number,
 *   emissive?: number }} FieldInteriors
 */

export { textHash };

/**
 * The per-pixel shading of a field part: ONE interpreter material function (fixed text; parts whose operand graphs
 * are the same share it: `name` is `'sdff.'` + a hash of the text) and the part's data block (`words`, for
 * `Engine.addMaterialData`; its base is the surface record's `data`). Each operand wears its constant material, its
 * material graph (`materials`: the asset's compiled materials, by name), else its colour; a graph material missing from
 * `materials` wears its colour and is named in `missing`. `single` leaves roughness and metallic to the surface record
 * (`look`). `debug` shows the shading normal as an emissive colour (0.5 + n / 2) on black (a data flag: same text).
 * `relief` (the record's, MAMA view 'relief') marches the part's field from the face to the recess (parallax).
 * An operand with `interior` (a window pane) shows a room of its set; `options.interiors` gives each set's tile count and
 * depth and the room look (lightsOn, blinds, unlitLevel; MAMA doc.interiors). `interiors` lists, per pane operand, its
 * set and `at`, the word of its first tile id in `words`: the caller writes the full and small texture ids of tile k at
 * `at + 2k` and `at + 2k + 1` once the pictures are in the atlas (Engine.addTexture) and calls setMaterialData; until
 * then the pane shows its glass.
 * The per-instance tint is read from MaterialInput.instance_params[0] (INSTANCE_PARAM_SLOTS.tint): when its w is above
 * 0.5, the base colour is multiplied by its xyz.
 * A material with `pages` (it samples image pages: SVG PLUS drawings, bitmaps, surface images; MAMA graphPagesOfDoc)
 * reads them where Local fields reads binding 7: the host places the document's page words once
 * (Engine.addMaterialData) and passes their base as `options.pagesBase`; each `pages` word of the material's constants is
 * written into `words` plus that base. When a graph declares mat_footprint the function sets it per fragment (metres
 * per screen pixel, as the viewport does) before the graphs run.
 *
 * @param {object} shading PartShading (MAMA field-program.ts)
 * @param {ReadonlyMap<string, object> | readonly object[]} materials SdfmCompiledMaterial by name, or a list of them
 * @param {{ field: string, debug?: boolean, interiors?: FieldInteriors, pagesBase?: number }} options `field`: MAMA FIELD_PROGRAM_WGSL (the evaluator, pmf_eval)
 * @returns {{ name: string, wgsl: string, words: Uint32Array, tintMode: 'graph' | 'multiply', missing: string[],
 *   interiors: { operand: number, set: string, at: number, tiles: number }[] }}
 * @throws {SdfmError} `sdfm.schema` for a malformed record or evaluator; `sdfm.material_unmappable` for a material that
 *   cannot be inlined, a forbidden token, more than MATERIAL_FUNCTION_MAX_BYTES bytes, or pages without options.pagesBase
 */
export function fieldPartShading(shading, materials, options) {
  const { core, words, tintMode, missing, interiors, footprint } = fieldShadingCore(shading, materials, options);
  const main = INTERPRETER_MAIN_WGSL.replace('__SDFF_FOOTPRINT__', footprint ? '  mat_footprint = sdfm_footprint(in);\n' : '');
  const wgsl = [core, FIELD_FRAME_WGSL, FIELD_NORMAL_WGSL, INTERIOR_WGSL, ...(footprint ? [FOOTPRINT_WGSL] : []), main].join('\n');
  const fail = (what) => { throw new SdfmError('sdfm.material_unmappable', what); };
  const forbidden = forbiddenToken(wgsl);
  if (forbidden) fail(`'${forbidden}' is forbidden in a material function (S11)`);
  const bytes = new TextEncoder().encode(wgsl).length;
  if (bytes > MATERIAL_FUNCTION_MAX_BYTES) fail(`the per-pixel SDF shading needs ${bytes} bytes of WGSL (RAA limit ${MATERIAL_FUNCTION_MAX_BYTES})`);
  return { name: `sdff.${textHash(wgsl)}`, wgsl, words, tintMode, missing, interiors };
}

/**
 * The interpreter's text without an entry point, and the part's data block: the structs, the graph helpers and
 * functions, the dispatch, the field evaluator (`options.field`), the data readers and the operand shading
 * (sdff_label, sdff_material). It reads its block through raa_material_word / raa_material_f32 and, for graphs with
 * the player colour, mat_instance_tint: RAA's prelude defines them for fieldPartShading, field-bake.js for the bake.
 * Same checks and errors as fieldPartShading (except the size cap and the forbidden tokens, which apply to RAA
 * material functions only).
 *
 * @param {object} shading PartShading (MAMA field-program.ts)
 * @param {ReadonlyMap<string, object> | readonly object[]} materials SdfmCompiledMaterial by name, or a list of them
 * @param {{ field: string, debug?: boolean, interiors?: FieldInteriors, pagesBase?: number }} options (pagesBase: see fieldPartShading)
 * @returns {{ core: string, words: Uint32Array, tintMode: 'graph' | 'multiply', missing: string[],
 *   interiors: { operand: number, set: string, at: number, tiles: number }[], footprint: boolean }} `footprint`: a graph declares mat_footprint (the caller sets it)
 */
export function fieldShadingCore(shading, materials, options) {
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
    const pages = materialPages(m, (what) => fail(`material '${name}': ${what}`));
    // The graph's shading normal (MAMA graphShadingNormal: its bump, its authored normal), part of its key.
    let normal = null;
    if (m.normal !== undefined) {
      if (typeof m.normal !== 'string' || !/\bfn\s+mat_graph_normal\s*\(/.test(m.normal)) fail(`material '${name}': 'normal' does not define mat_graph_normal`);
      normal = structs(dataReads(m.normal)).trim();
      const toks = tokens(normal);
      if (toks.includes('extra_words')) fail(`material '${name}': 'extra_words' is not read as extra_words[...]`);
      const raa = toks.find((t) => /^raa_/.test(t));
      if (raa) fail(`material '${name}': '${raa}' (identifiers starting with raa_ are reserved)`);
    }
    graphOf.set(name, { body: normal === null ? body : `${body}\n// normal\n${normal}`, code: body, normal, constants: m.constants, pages });
  }
  // Graphs that sample image pages: their pointer words are relocated to the host's page block (options.pagesBase).
  const pagesBase = [...graphOf].some(([, g]) => g.pages.length > 0)
    ? pagesBaseOf(options, (what) => fail(`material '${[...graphOf].find(([, g]) => g.pages.length > 0)[0]}': ${what}`)) : 0;
  // One function per distinct body, in text order: the same graphs give the same function text whatever the part.
  const bodies = [...new Set([...graphOf.values()].map((g) => g.body))].sort();
  // A graph that never mentions its node-space point pn needs no program -> node transform: sdff_label then passes p
  // and skips the 12 material words and the 3x3 multiply per label, per fragment (the text stays part-independent).
  const nodePoint = bodies.some((body) => tokens(body).includes('pn')) ? 'sdff_affine(o + 12u, p)' : 'p';
  const parts = new Map([...graphOf.values()].map((g) => [g.body, g]));
  const graphs = bodies.map((key, k) => {
    const g = parts.get(key);
    return [`fn sdfm_graph_${k}(b: u32, p: vec3<f32>, pn: vec3<f32>, n: vec3<f32>, mk: SdfmMatMasks) -> SdfmMaterialFull {`, g.code, '}',
      ...(g.normal === null ? [] : [g.normal.replace(/\bmat_hfn\b/g, `sdfm_hfn_${k}`).replace(/\bmat_graph_normal\b/g, `sdfm_normal_${k}`)]),
    ].join('\n');
  });
  const bent = bodies.map((key, k) => [key, k]).filter(([key]) => parts.get(key).normal !== null).map(([, k]) => k);
  // The label normal: a graph without one keeps np; an authored normal is what its body left in mat_out_normal.
  const authored = helpers.has('mat_authored_prelude');
  const graphCall = bent.length === 0
    ? `    let m = sdff_graph(sdff_w(o + 4u), base + sdff_w(o + 5u), p, ${nodePoint}, n);
    if (m.ok) { return m.m; }`
    : `${authored ? '    mat_authored_reset();\n' : ''}    let pn = ${nodePoint};
    let m = sdff_graph(sdff_w(o + 4u), base + sdff_w(o + 5u), p, pn, n);
    if (m.ok) {
      let j = ${nodePoint === 'p' ? 'mat3x3<f32>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)' : 'transpose(mat3x3<f32>(sdff_v3(o + 12u), sdff_v3(o + 16u), sdff_v3(o + 20u)))'};
      sdff_label_n = sdff_graph_normal(sdff_w(o + 4u), base + sdff_w(o + 5u), p, pn, j, np, ${authored ? 'mat_out_normal' : 'vec3<f32>(0.0, 0.0, 1.0)'});
      return m.m;
    }`;
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
${bent.length ? `
fn sdff_graph_normal(k: u32, b: u32, p: vec3<f32>, pn: vec3<f32>, j: mat3x3<f32>, n: vec3<f32>, tn: vec3<f32>) -> vec3<f32> {
  switch k {
${bent.map((k) => `    case ${k}u: { sdff_bent = true; return sdfm_normal_${k}(b, p, pn, j, n, tn); }`).join('\n')}
    default: { }
  }
  return n;
}
` : ''}`;
  const core = [
    `// sdff: the SDF shading interpreter, ${bodies.length} material graphs (@etcito/raa-sdfm field-shading.js)`,
    `struct SdfmMaterialFull { ${MATERIAL_FULL_FIELDS.map((x) => `${x}: ${x === 'base_color' || x === 'emissive' ? 'vec3<f32>' : 'f32'}`).join(', ')} }`,
    'struct SdfmMatMasks { curvature: f32, cavity: f32, ao: f32, height: f32, proximity: f32 }',
    ...[...helpers].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)).map(([id, text]) => `// MAMA helper ${id}\n${text}\n`),
    ...graphs,
    dispatch,
    field.trim(),
    INTERPRETER_CORE_WGSL.replace('__SDFF_GRAPH_CALL__', graphCall),
  ].join('\n');
  const { words, interiors } = fieldShadingWords(shading, graphOf, bodies, options.debug === true, options.interiors);
  // The pointer words of each graph's image pages, relocated to the host's page block.
  for (const g of graphOf.values()) for (const i of g.pages) words[g.at + i] = (words[g.at + i] + pagesBase) >>> 0;
  const footprint = declaresFootprint([...helpers.values()]);
  return { core, words, interiors, tintMode, missing, footprint };
}

/** The data block of a part (FIELD_DATA_* layout) and where its panes' tile ids sit. */
function fieldShadingWords(shading, graphOf, bodies, debug, interiorsDef) {
  const ops = shading.operands, program = shading.program;
  const programAt = FIELD_DATA_HEADER, table = programAt + program.length;
  const constantsAt = new Map();
  let size = table + FIELD_OPERAND_WORDS * ops.length;
  for (const [name, g] of graphOf) { constantsAt.set(name, size); g.at = size; size += g.constants.length; }
  // Interior records after the constants: a pane of an unknown set gets a record with no tile (glass only).
  const look = { ...INTERIOR_DEFAULTS, ...Object.fromEntries(Object.entries(interiorsDef ?? {}).filter(([k, v]) => k !== 'sets' && Number.isFinite(v))) };
  const records = [];
  ops.forEach((o, i) => {
    if (!o.interior) return;
    const set = interiorsDef?.sets?.[o.interior.set];
    const tiles = set ? Math.max(0, Math.min(8, Math.floor(set.tiles))) : 0;
    records.push({ operand: i, at: size, tiles, set, interior: o.interior });
    size += INTERIOR_WORDS + INTERIOR_TILE_WORDS * tiles;
  });
  const relief = shading.relief !== undefined;
  // Relief: the table (2 words per operand), then each module record and its program.
  const reliefTable = relief ? size : 0;
  const moduleAt = [], copyList = [];
  if (relief) {
    const c = shading.relief.copies;
    for (let i = 0; i < ops.length; i++) if (c[2 * i] >= 0 && c[2 * i + 1] === i) copyList.push([c[2 * i], i]);
    size += 1 + 2 * copyList.length;
    for (const m of shading.relief.modules) { moduleAt.push(size); size += RELIEF_MODULE_WORDS + m.program.length; }
  }
  const words = new Uint32Array(size), floats = new Float32Array(words.buffer);
  words[0] = relief || records.length ? FIELD_DATA_VERSION : 1;
  words[1] = (shading.single ? FIELD_FLAG_SINGLE : 0) | (debug ? FIELD_FLAG_DEBUG : 0) | (relief ? FIELD_FLAG_RELIEF : 0);
  words[2] = programAt; words[3] = table; words[4] = ops.length;
  if (relief) {
    floats[5] = shading.relief.depth;
    words[6] = reliefTable;
    shading.relief.modules.forEach((m, k) => {
      const at = moduleAt[k];
      words[at] = at + RELIEF_MODULE_WORDS; words[at + 1] = m.count; floats[at + 2] = m.depth;
      floats.set(m.rows, at + 4);
      floats.set(m.box, at + 16);
      floats.set(m.program, at + RELIEF_MODULE_WORDS);
    });
    words[reliefTable] = copyList.length;
    copyList.forEach(([k, start], j) => { words[reliefTable + 1 + 2 * j] = moduleAt[k]; words[reliefTable + 2 + 2 * j] = start; });
  }
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
  const interiors = records.map(({ operand, at, tiles, set, interior }) => {
    words[table + FIELD_OPERAND_WORDS * operand + 11] = at;
    floats.set(interior.half, at);
    floats[at + 3] = interior.width; floats[at + 4] = interior.height;
    floats[at + 5] = interior.depth > 0 ? interior.depth : set?.depth ?? 5;
    floats[at + 6] = interior.glass;
    words[at + 7] = tiles;
    floats[at + 8] = look.lightsOn; floats[at + 9] = look.blinds; floats[at + 10] = look.unlitLevel; floats[at + 11] = look.emissive;
    for (let k = 0; k < tiles; k++) {
      const w = at + INTERIOR_WORDS + INTERIOR_TILE_WORDS * k;
      words[w] = INTERIOR_NO_TEXTURE; words[w + 1] = INTERIOR_NO_TEXTURE; floats[w + 2] = 0.5;
    }
    return { operand, set: interior.set, at: at + INTERIOR_WORDS, tiles };
  });
  return { words, interiors };
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
