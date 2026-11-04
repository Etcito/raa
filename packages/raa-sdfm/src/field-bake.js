// The 'baked' shading of a field part (MAMA core/game/render-options.ts, shading 'baked'; owner 2026-11-04: "baked
// texture (normal/material)"): at load time, on the client's GPU, the part's normal and material are baked once into 2D
// textures laid out by its UV atlas (MAMA core/game/part-atlas.ts; never a 3D volume, no texture is downloaded), then
// every baked part is drawn by ONE shared material function that samples its textures (BAKED_FUNCTION).
//
// The bake runs the same interpreter as the per-pixel shading (field-shading.js fieldShadingCore: the part's field
// program and operands as data words, MAMA's FIELD_PROGRAM_WGSL evaluator), so the bake pipeline is one pipeline for
// every part whose operands share the same material graphs (all constant-material parts share one). Each part's
// triangles are rasterised in UV space; per texel the rest point and rest frame are interpolated, the field gives the
// normal (its gradient; the mesh normal where it has none or leans past 60 degrees, as the per-pixel shading does) and
// the material (sdff_material). Then every chart is dilated into its gutter so bilinear sampling never reads an empty
// texel, and the textures are read back for Engine.addTexture (RAA keeps textures in its atlas; it shares no
// GPUTexture with the page, so the bytes travel through the CPU once).
//
// Texture layout (RGBA8, rows top first, row r = uv.y (r + 0.5) / size):
//   normal:   xyz = the field normal in the tangent frame of the mesh (tangent = partTangents, its w the handedness,
//             bitangent = cross(n, t) * w), encoded 0.5 + n / 2; w = metallic. Covered texels have z >= 0.75.
//   color:    rgb = base colour, sRGB-encoded; w = roughness.
//   emissive: rgb = e / (1 + e) per channel (0..inf in 8 bits); only uploaded when some texel is not black.
// The tangent frame moves with the instance and the skin, so a posed or skinned part shades right
// (raa_perturb_normal(in.normal, in.tangent, ts) in BAKED_MATERIAL_WGSL inverts the bake exactly).
import { SdfmError } from './errors.js';
import { fieldShadingCore, textHash } from './field-shading.js';

/** The one material function every baked part draws with (its data words name its textures). */
export const BAKED_FUNCTION = 'sdfb.baked';
/** Data words of a baked part's surface material: [0] version, [1] normal, [2] color, [3] emissive texture ids. */
export const BAKED_DATA_VERSION = 1;
export const BAKED_DATA_WORDS = 4;
/** Emissive texture id of a part that has no emissive. */
export const BAKED_NO_TEXTURE = 0xffffffff;
/** Dilation passes after the raster: covers the atlas gutter at any bake size up to 8x the default density. */
export const BAKE_DILATE_PASSES = 8;

/** The shared material function of baked parts (RAA S11 user function; reads the m25 material data words). */
export const BAKED_MATERIAL_WGSL = /* wgsl */ `
// sdfb: a field part baked into textures (@etcito/raa-sdfm field-bake.js)
fn material_main(in: MaterialInput) -> MaterialOutput {
  var o = raa_default_output(in);
  o.opacity = 1.0;
  let base = raa_material_data_base(in.record);
  let nt = raa_texture(raa_material_word(base + 1u), in.uv);
  let ct = raa_texture(raa_material_word(base + 2u), in.uv);
  let n = normalize(in.normal);
  let ts = nt.xyz * 2.0 - vec3<f32>(1.0);
  // A baked normal leans at most 60 degrees (z >= 0.5, stored >= 0.75); a texel no chart covered (a sliver below one
  // texel) reads lower and keeps the mesh normal instead of a bent one.
  if (dot(in.tangent.xyz, in.tangent.xyz) > 1e-12 && nt.z >= 0.7) {
    o.normal = raa_perturb_normal(n, in.tangent, normalize(ts));
  }
  o.base_color = raa_srgb_to_linear(ct.rgb);
  o.roughness = ct.a;
  o.metallic = nt.a;
  let eid = raa_material_word(base + 3u);
  if (eid != ${BAKED_NO_TEXTURE}u) {
    let e = raa_texture(eid, in.uv).rgb;
    o.emissive = e / max(vec3<f32>(1.0) - e, vec3<f32>(1.0 / 255.0));
  }
  return o;
}
`;

/** The data words of a baked part's surface material (Engine.addMaterialData / setMaterialData). */
export function bakedMaterialWords(normal, color, emissive = BAKED_NO_TEXTURE) {
  for (const [what, id] of [['normal', normal], ['color', color], ['emissive', emissive]]) {
    if (!Number.isInteger(id) || id < 0 || id > BAKED_NO_TEXTURE) throw new SdfmError('sdfm.schema', `baked material: ${what} texture id ${id} is not a u32`);
  }
  return Uint32Array.of(BAKED_DATA_VERSION, normal, color, emissive);
}

/**
 * Per-vertex tangents of a UV-mapped mesh (xyz + handedness w, the RAA tangent stream): the UV gradient of each
 * triangle accumulated per vertex (area weighted), made orthogonal to the vertex normal. A vertex no triangle gives a
 * direction gets an arbitrary perpendicular. The bake and the drawn mesh use the same array.
 * @returns {Float32Array}
 */
export function partTangents(positions, normals, uvs, indices) {
  const n = positions.length / 3;
  if (normals.length !== n * 3 || uvs.length !== n * 2 || indices.length % 3 !== 0) {
    throw new SdfmError('sdfm.schema', `partTangents: ${n} positions, ${normals.length / 3} normals, ${uvs.length / 2} uvs, ${indices.length} indices`);
  }
  const t = new Float64Array(n * 3), b = new Float64Array(n * 3);
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i], c = indices[i + 1], d = indices[i + 2];
    const e1 = [0, 1, 2].map((k) => positions[c * 3 + k] - positions[a * 3 + k]);
    const e2 = [0, 1, 2].map((k) => positions[d * 3 + k] - positions[a * 3 + k]);
    const du1 = uvs[c * 2] - uvs[a * 2], dv1 = uvs[c * 2 + 1] - uvs[a * 2 + 1];
    const du2 = uvs[d * 2] - uvs[a * 2], dv2 = uvs[d * 2 + 1] - uvs[a * 2 + 1];
    const det = du1 * dv2 - du2 * dv1;
    if (!(Math.abs(det) > 1e-20)) continue;
    // Area weighting: the unnormalised gradient times |det| keeps the triangle's 3D size.
    const s = Math.sign(det);
    for (const v of [a, c, d]) for (let k = 0; k < 3; k++) {
      t[v * 3 + k] += s * (e1[k] * dv2 - e2[k] * dv1);
      b[v * 3 + k] += s * (e2[k] * du1 - e1[k] * du2);
    }
  }
  const out = new Float32Array(n * 4);
  for (let v = 0; v < n; v++) {
    let nx = normals[v * 3], ny = normals[v * 3 + 1], nz = normals[v * 3 + 2];
    const nl = Math.hypot(nx, ny, nz) || 1;
    nx /= nl; ny /= nl; nz /= nl;
    let tx = t[v * 3], ty = t[v * 3 + 1], tz = t[v * 3 + 2];
    const d = tx * nx + ty * ny + tz * nz;
    tx -= d * nx; ty -= d * ny; tz -= d * nz;
    let tl = Math.hypot(tx, ty, tz);
    if (!(tl > 1e-12)) {
      // Any perpendicular: the axis least aligned with n, made orthogonal.
      const ax = Math.abs(nx) < 0.9 ? [1, 0, 0] : [0, 1, 0];
      const e = ax[0] * nx + ax[1] * ny + ax[2] * nz;
      tx = ax[0] - e * nx; ty = ax[1] - e * ny; tz = ax[2] - e * nz;
      tl = Math.hypot(tx, ty, tz);
    }
    tx /= tl; ty /= tl; tz /= tl;
    // Handedness: +1 when cross(n, t) points along the accumulated bitangent.
    const cx = ny * tz - nz * ty, cy = nz * tx - nx * tz, cz = nx * ty - ny * tx;
    const w = cx * b[v * 3] + cy * b[v * 3 + 1] + cz * b[v * 3 + 2] < 0 ? -1 : 1;
    out[v * 4] = tx; out[v * 4 + 1] = ty; out[v * 4 + 2] = tz; out[v * 4 + 3] = w;
  }
  return out;
}

/**
 * UV units per metre of a part's atlas: sqrt(uv area / surface area) over its triangles (MAMA part-atlas.ts packs every
 * chart at one scale, so this is that scale). The bake side is MAMA render-options.ts bakeTextureSize(scale, options).
 * 0 when the part has no area or no uv area.
 */
export function atlasScale(positions, uvs, indices) {
  let a3 = 0, a2 = 0;
  for (let i = 0; i + 2 < indices.length; i += 3) {
    const a = indices[i], b = indices[i + 1], c = indices[i + 2];
    const e1x = positions[b * 3] - positions[a * 3], e1y = positions[b * 3 + 1] - positions[a * 3 + 1], e1z = positions[b * 3 + 2] - positions[a * 3 + 2];
    const e2x = positions[c * 3] - positions[a * 3], e2y = positions[c * 3 + 1] - positions[a * 3 + 1], e2z = positions[c * 3 + 2] - positions[a * 3 + 2];
    a3 += Math.hypot(e1y * e2z - e1z * e2y, e1z * e2x - e1x * e2z, e1x * e2y - e1y * e2x) / 2;
    a2 += Math.abs((uvs[b * 2] - uvs[a * 2]) * (uvs[c * 2 + 1] - uvs[a * 2 + 1]) - (uvs[c * 2] - uvs[a * 2]) * (uvs[b * 2 + 1] - uvs[a * 2 + 1])) / 2;
  }
  return a3 > 0 && a2 > 0 ? Math.sqrt(a2 / a3) : 0;
}

/**
 * Texels the bakes of one engine may take: RAA keeps every texture in one 4096 x 4096 atlas (shelf packing, m22); power
 * of two squares added largest first fill it nearly whole, the rest is left to other textures.
 */
export const BAKE_ATLAS_BUDGET = Math.floor(4096 * 4096 * 0.9);
/** The smallest side fitBakeSizes goes down to. */
export const BAKE_MIN_SIZE = 32;

/**
 * Bake sides that fit `budget` texels: `parts[i].size` is the side the density asks for (MAMA bakeTextureSize),
 * `parts[i].textures` how many textures the part takes (2, or 3 with emissive) and `parts[i].scale` its atlas scale
 * (atlasScale; uv units per metre). While the sum is over the budget the part with the highest texel density
 * (size x scale; the largest side when no scale is given; the first part on a tie) is halved, down to BAKE_MIN_SIZE: the
 * densest parts lose density first, so the density stays as even as the budget allows and every part keeps a texture.
 * Deterministic.
 * @param {{ size: number, textures: number, scale?: number }[]} parts
 * @returns {number[]}
 */
export function fitBakeSizes(parts, budget = BAKE_ATLAS_BUDGET) {
  const sizes = parts.map((p) => p.size);
  const density = (i) => sizes[i] * (parts[i].scale > 0 ? parts[i].scale : 1);
  let total = 0;
  for (let i = 0; i < parts.length; i++) total += sizes[i] * sizes[i] * parts[i].textures;
  // Parts in falling density, kept sorted: the head is halved, then moved back to its place (insertion).
  const order = sizes.map((_, i) => i).filter((i) => sizes[i] > BAKE_MIN_SIZE);
  const cmp = (a, b) => density(b) - density(a) || a - b;
  order.sort(cmp);
  while (total > budget && order.length > 0) {
    const i = order.shift();
    const next = Math.max(BAKE_MIN_SIZE, sizes[i] >> 1);
    total -= (sizes[i] * sizes[i] - next * next) * parts[i].textures;
    sizes[i] = next;
    if (next <= BAKE_MIN_SIZE) continue;
    let k = 0;
    while (k < order.length && cmp(order[k], i) < 0) k++;
    order.splice(k, 0, i);
  }
  return sizes;
}

/** Stand-ins for the RAA prelude the interpreter core reads: its data words and the player tint (baked white). */
const BAKE_SHIM_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> sdfb_words: array<u32>;
struct SdfbLook { roughness: f32, metallic: f32, single: f32, pad: f32 }
@group(0) @binding(1) var<uniform> sdfb_look: SdfbLook;
fn raa_material_word(i: u32) -> u32 { return sdfb_words[i]; }
fn raa_material_f32(i: u32) -> f32 { return bitcast<f32>(sdfb_words[i]); }
var<private> mat_instance_tint: vec4<f32> = vec4<f32>(1.0);
var<private> mat_instance_tint_set: f32 = 0.0;
`;

/** The bake entry points (after the shim and the interpreter core). */
const BAKE_ENTRY_WGSL = /* wgsl */ `
struct SdfbVertex {
  @location(0) uv: vec2<f32>,
  @location(1) rest: vec3<f32>,
  @location(2) normal: vec3<f32>,
  @location(3) tangent: vec4<f32>,
}
struct SdfbVarying {
  @builtin(position) clip: vec4<f32>,
  @location(0) rest: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) tangent: vec4<f32>,
}
@vertex fn sdfb_vertex(v: SdfbVertex) -> SdfbVarying {
  var o: SdfbVarying;
  // Texel row 0 (the top row RAA uploads first) is uv.y = 0.
  o.clip = vec4<f32>(v.uv.x * 2.0 - 1.0, 1.0 - v.uv.y * 2.0, 0.0, 1.0);
  o.rest = v.rest;
  o.normal = v.normal;
  o.tangent = v.tangent;
  return o;
}
struct SdfbTargets {
  @location(0) normal: vec4<f32>,
  @location(1) color: vec4<f32>,
  @location(2) emissive: vec4<f32>,
}
fn sdfb_srgb(c: vec3<f32>) -> vec3<f32> {
  let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
  return select(1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055), x * 12.92, x <= vec3<f32>(0.0031308));
}
@fragment fn sdfb_fragment(f: SdfbVarying) -> SdfbTargets {
  let base = 0u;
  var n = f.normal;
  if (dot(n, n) < 1e-20) { n = vec3<f32>(0.0, 0.0, 1.0); }
  n = normalize(n);
  let s = pmf_eval(base + sdff_w(base + 2u), f.rest);
  var g = n;
  let len = length(s.g);
  if (len > 0.0 && len < 3.0e38) {
    let m = s.g / len;
    // A field normal past 60 degrees from the mesh's (a thin wall, a coarse approximation) keeps the mesh normal.
    if (dot(m, n) >= 0.5) { g = m; }
  }
  var t = f.tangent.xyz - n * dot(n, f.tangent.xyz);
  if (dot(t, t) < 1e-20) { t = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9); t = t - n * dot(n, t); }
  t = normalize(t);
  let b = cross(n, t) * select(-1.0, 1.0, f.tangent.w >= 0.0);
  let ts = vec3<f32>(dot(g, t), dot(g, b), dot(g, n));
  let m = sdff_material(base, s, f.rest, g);
  var metallic = m.metallic;
  var roughness = m.roughness;
  if (sdfb_look.single > 0.5) { metallic = sdfb_look.metallic; roughness = sdfb_look.roughness; }
  let e = max(m.emissive, vec3<f32>(0.0));
  var o: SdfbTargets;
  o.normal = vec4<f32>(ts * 0.5 + vec3<f32>(0.5), clamp(metallic, 0.0, 1.0));
  o.color = vec4<f32>(sdfb_srgb(m.base_color), clamp(roughness, 0.0, 1.0));
  o.emissive = vec4<f32>(e / (vec3<f32>(1.0) + e), 1.0);
  return o;
}
`;

/** Dilation: an empty texel (normal z = 0) takes the mean of its covered 8-neighbours; covered texels are kept. */
export const BAKE_DILATE_WGSL = /* wgsl */ `
@group(0) @binding(0) var sdfb_n: texture_2d<f32>;
@group(0) @binding(1) var sdfb_c: texture_2d<f32>;
@group(0) @binding(2) var sdfb_e: texture_2d<f32>;
@vertex fn sdfb_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
}
struct SdfbTargets {
  @location(0) normal: vec4<f32>,
  @location(1) color: vec4<f32>,
  @location(2) emissive: vec4<f32>,
}
@fragment fn sdfb_dilate(@builtin(position) at: vec4<f32>) -> SdfbTargets {
  let size = vec2<i32>(textureDimensions(sdfb_n));
  let p = vec2<i32>(at.xy);
  var o: SdfbTargets;
  o.normal = textureLoad(sdfb_n, p, 0);
  o.color = textureLoad(sdfb_c, p, 0);
  o.emissive = textureLoad(sdfb_e, p, 0);
  if (o.normal.z > 0.0) { return o; }
  var sn = vec4<f32>(0.0);
  var sc = vec4<f32>(0.0);
  var se = vec4<f32>(0.0);
  var k = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let q = p + vec2<i32>(dx, dy);
      if ((dx == 0 && dy == 0) || q.x < 0 || q.y < 0 || q.x >= size.x || q.y >= size.y) { continue; }
      let nq = textureLoad(sdfb_n, q, 0);
      if (nq.z <= 0.0) { continue; }
      sn += nq;
      sc += textureLoad(sdfb_c, q, 0);
      se += textureLoad(sdfb_e, q, 0);
      k += 1.0;
    }
  }
  if (k > 0.0) {
    o.normal = sn / k;
    o.color = sc / k;
    o.emissive = se / k;
  }
  return o;
}
`;

/** The bake shader of a part's interpreter core (fieldShadingCore(...).core): one text per set of material graphs. */
export function fieldBakeWgsl(core) {
  if (typeof core !== 'string' || !/\bfn\s+sdff_material\s*\(/.test(core)) {
    throw new SdfmError('sdfm.schema', 'fieldBakeWgsl: core is not the interpreter core (field-shading.js fieldShadingCore)');
  }
  return [BAKE_SHIM_WGSL, core, BAKE_ENTRY_WGSL].join('\n');
}

// WebGPU flags as numbers: this module also loads where the GPU* globals are absent (Node tests).
const TEX = { COPY_SRC: 0x01, TEXTURE_BINDING: 0x04, RENDER_ATTACHMENT: 0x10 };
const BUF = { MAP_READ: 0x0001, COPY_DST: 0x0008, INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080 };
const FORMAT = 'rgba8unorm';
const TARGETS = [{ format: FORMAT }, { format: FORMAT }, { format: FORMAT }];
const VERTEX_FLOATS = 12; // uv 2, rest 3, normal 3, tangent 4

/**
 * A baker on a GPUDevice (the one RAA renders with, so no second adapter is opened). `bake` returns the part's
 * textures; pipelines are cached by shader text, so parts that share their material graphs share one pipeline.
 * @param {GPUDevice} device
 */
export function createFieldBaker(device) {
  if (!device || typeof device.createRenderPipelineAsync !== 'function') throw new SdfmError('sdfm.schema', 'createFieldBaker: pass a GPUDevice');
  const pipelines = new Map();
  let dilate = null;
  const bakePipeline = (wgsl) => {
    const key = textHash(wgsl);
    let p = pipelines.get(key);
    if (p) return p;
    const module = device.createShaderModule({ label: `sdfb.${key}`, code: wgsl });
    p = device.createRenderPipelineAsync({
      label: `sdfb.${key}`,
      layout: 'auto',
      vertex: {
        module, entryPoint: 'sdfb_vertex',
        buffers: [{ arrayStride: VERTEX_FLOATS * 4, attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x2' },
          { shaderLocation: 1, offset: 8, format: 'float32x3' },
          { shaderLocation: 2, offset: 20, format: 'float32x3' },
          { shaderLocation: 3, offset: 32, format: 'float32x4' },
        ] }],
      },
      fragment: { module, entryPoint: 'sdfb_fragment', targets: TARGETS },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });
    pipelines.set(key, p);
    return p;
  };
  const dilatePipeline = () => {
    if (dilate) return dilate;
    const module = device.createShaderModule({ label: 'sdfb.dilate', code: BAKE_DILATE_WGSL });
    dilate = device.createRenderPipelineAsync({
      label: 'sdfb.dilate', layout: 'auto',
      vertex: { module, entryPoint: 'sdfb_full' },
      fragment: { module, entryPoint: 'sdfb_dilate', targets: TARGETS },
      primitive: { topology: 'triangle-list' },
    });
    return dilate;
  };
  const buffer = (data, usage) => {
    const b = device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 4) * 4), usage: usage | BUF.COPY_DST });
    device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
    return b;
  };
  const textures = (size) => [0, 1, 2].map(() => device.createTexture({
    size: [size, size], format: FORMAT, usage: TEX.RENDER_ATTACHMENT | TEX.TEXTURE_BINDING | TEX.COPY_SRC,
  }));
  const attachments = (set) => set.map((t) => ({ view: t.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }));

  return {
    /**
     * Bakes one part. `mesh` is the part in its rest (mesh) space: positions, normals, the atlas uvs, triangle indices
     * (all sub-meshes of the part, which share one atlas) and `tangents` (partTangents; drawn with the same array).
     * `shading` and `materials` are what fieldPartShading takes; `field` is MAMA's FIELD_PROGRAM_WGSL.
     * @returns {Promise<{ size: number, normal: Uint8Array, color: Uint8Array, emissive: Uint8Array | null, missing: string[], pipeline: string }>}
     */
    async bake({ shading, materials, field, positions, normals, uvs, tangents, indices, size }) {
      if (!Number.isInteger(size) || size < 16 || size > 4096) throw new SdfmError('sdfm.schema', `bake: size ${size} is not an integer in 16..4096`);
      const nv = positions.length / 3;
      if (normals.length !== nv * 3 || uvs.length !== nv * 2 || tangents.length !== nv * 4 || indices.length % 3 !== 0 || indices.length === 0) {
        throw new SdfmError('sdfm.schema', `bake: ${nv} positions, ${normals.length / 3} normals, ${uvs.length / 2} uvs, ${tangents.length / 4} tangents, ${indices.length} indices`);
      }
      const { core, words, missing } = fieldShadingCore(shading, materials, { field });
      const wgsl = fieldBakeWgsl(core);
      const [pipeline, flat] = await Promise.all([bakePipeline(wgsl), dilatePipeline()]);
      const vertices = new Float32Array(nv * VERTEX_FLOATS);
      for (let v = 0; v < nv; v++) {
        const o = v * VERTEX_FLOATS;
        vertices[o] = uvs[v * 2]; vertices[o + 1] = uvs[v * 2 + 1];
        vertices.set(positions.subarray(v * 3, v * 3 + 3), o + 2);
        vertices.set(normals.subarray(v * 3, v * 3 + 3), o + 5);
        vertices.set(tangents.subarray(v * 4, v * 4 + 4), o + 8);
      }
      const look = Float32Array.of(shading.look.roughness, shading.look.metallic, shading.single ? 1 : 0, 0);
      const vbuf = buffer(vertices, BUF.VERTEX), ibuf = buffer(Uint32Array.from(indices), BUF.INDEX);
      const wbuf = buffer(words, BUF.STORAGE), ubuf = buffer(look, BUF.UNIFORM);
      const sets = [textures(size), textures(size)];
      const rowBytes = Math.ceil(size * 4 / 256) * 256;
      const reads = [0, 1, 2].map(() => device.createBuffer({ size: rowBytes * size, usage: BUF.MAP_READ | BUF.COPY_DST }));
      try {
        const enc = device.createCommandEncoder({ label: 'sdfb.bake' });
        const raster = enc.beginRenderPass({ colorAttachments: attachments(sets[0]) });
        raster.setPipeline(pipeline);
        raster.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: wbuf } }, { binding: 1, resource: { buffer: ubuf } }] }));
        raster.setVertexBuffer(0, vbuf);
        raster.setIndexBuffer(ibuf, 'uint32');
        raster.drawIndexed(indices.length);
        raster.end();
        let from = 0;
        for (let k = 0; k < BAKE_DILATE_PASSES; k++) {
          const pass = enc.beginRenderPass({ colorAttachments: attachments(sets[1 - from]) });
          pass.setPipeline(flat);
          pass.setBindGroup(0, device.createBindGroup({ layout: flat.getBindGroupLayout(0),
            entries: sets[from].map((t, i) => ({ binding: i, resource: t.createView() })) }));
          pass.draw(3);
          pass.end();
          from = 1 - from;
        }
        sets[from].forEach((t, i) => enc.copyTextureToBuffer({ texture: t }, { buffer: reads[i], bytesPerRow: rowBytes, rowsPerImage: size }, [size, size]));
        device.queue.submit([enc.finish()]);
        const out = await Promise.all(reads.map(async (r) => {
          await r.mapAsync(1);
          const src = new Uint8Array(r.getMappedRange()), dst = new Uint8Array(size * size * 4);
          for (let y = 0; y < size; y++) dst.set(src.subarray(y * rowBytes, y * rowBytes + size * 4), y * size * 4);
          r.unmap();
          return dst;
        }));
        let lit = false;
        for (let i = 0; i < out[2].length && !lit; i += 4) lit = out[2][i] > 0 || out[2][i + 1] > 0 || out[2][i + 2] > 0;
        return { size, normal: out[0], color: out[1], emissive: lit ? out[2] : null, missing, pipeline: `sdfb.${textHash(wgsl)}` };
      } finally {
        for (const b of [vbuf, ibuf, wbuf, ubuf, ...reads]) b.destroy();
        for (const t of sets.flat()) t.destroy();
      }
    },
    /** Pipelines this baker compiled (one per distinct bake shader). */
    get pipelines() { return pipelines.size; },
  };
}
