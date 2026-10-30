// Per-pixel SDF normals from a MAMA normal volume (MAMA src/core/game/normal-volume.ts): the field's unit gradient
// stored around a part mesh in one rgba8 image, read per fragment with raa_texture. Pure text and array work, no engine
// call: the caller uploads the image (Engine.addTexture), registers the wrapped material functions and sets the
// instance params of each mesh instance.
//
// Data a fragment needs, and where it travels:
//   - the rest position in the part's mesh space: the vertex colour xyz (w = 1), unclamped f32, passed through RAA's
//     skinning untouched, so a posed skinned part still reads the field at its rest point;
//   - the rest frame: uv = the octahedral rest normal, tangent = orthoTangent of the decoded normal (handedness +1);
//     RAA moves tangent and normal with the instance and the skin, and the rest-space gradient is re-expressed in that
//     posed frame (the same construction as SDF Modeler's per-pixel 'sdf' material, app/paramesh/sdf-material.ts);
//   - the volume: instance params (Engine.setInstanceParams, 16 floats) p0 = origin xyz, spacing; p1 = brick cells
//     xyz, texture id; p2 = image width, height, tiles per row, index rows; p3.x = 1 when the volume is set.
// An instance without the params (p3.x = 0), a point outside the stored bricks, a node without gradient or a sampled
// normal more than 60 degrees off the mesh normal keeps the mesh normal: the wrapped function is a drop-in for its base.
import { SdfmError } from './errors.js';

export const NORMAL_VOLUME_FORMAT = 'sdfm.normal-volume';
/** The volume versions this module reads. */
export const NORMAL_VOLUME_VERSIONS = Object.freeze([1]);
/** MAMA normal-volume.ts: cells per brick side and the brick tile size. */
const BRICK = 4;
const TILE_W = (BRICK + 1) * (BRICK + 1);
const TILE_H = 2 * (BRICK + 1);

/** Prefix of the wrapped function names; the rest of the name is the base function's. */
export const VOLUME_FUNCTION_PREFIX = 'sdfv.';
/** The wrapped standard surface (no MAMA material): raa_default_output with the volume normal. */
export const VOLUME_STANDARD_FUNCTION = 'sdfv.standard';

const VOLUME_WGSL = /* wgsl */ `
// sdfv: per-pixel normal from a MAMA normal volume (@etcito/raa-sdfm normal-volume.js).
fn sdfv_q16(hi: f32, lo: f32) -> f32 {
  return (hi * 65280.0 + lo * 255.0) / 65535.0 * 2.0 - 1.0;
}

// The gradient of one brick slice: two bilinear taps (rows xy, then z) at texel coordinate t of the image.
fn sdfv_slice(tex: u32, size: vec2<f32>, t: vec2<f32>) -> vec3<f32> {
  let a = raa_texture(tex, t / size);
  let b = raa_texture(tex, (t + vec2<f32>(0.0, ${BRICK + 1}.0)) / size);
  return vec3<f32>(sdfv_q16(a.x, a.y), sdfv_q16(a.z, a.w), sdfv_q16(b.x, b.y));
}

// The interpolated (not normalised) rest-space gradient at the fragment's rest point; zero where there is none.
fn sdfv_volume_gradient(in: MaterialInput) -> vec3<f32> {
  let q = in.instance_params;
  if (q[3].x < 0.5 || !(q[0].w > 0.0)) { return vec3<f32>(0.0); }
  let c = (in.color.xyz - q[0].xyz) / q[0].w;
  let bc = floor(c / ${BRICK}.0);
  if (any(bc < vec3<f32>(0.0)) || any(bc >= q[1].xyz)) { return vec3<f32>(0.0); }
  let tex = u32(q[1].w + 0.5);
  let size = q[2].xy;
  let width = u32(q[2].x + 0.5);
  let dims = vec3<u32>(q[1].xyz + vec3<f32>(0.5));
  let cell = u32(bc.x) + dims.x * (u32(bc.y) + dims.y * u32(bc.z));
  let e = raa_texture(tex, (vec2<f32>(f32(cell % width), f32(cell / width)) + vec2<f32>(0.5)) / size);
  let id = u32(round(e.x * 255.0)) * 65536u + u32(round(e.y * 255.0)) * 256u + u32(round(e.z * 255.0));
  if (id == 0u) { return vec3<f32>(0.0); }
  let tiles = u32(q[2].z + 0.5);
  let b = id - 1u;
  let o = vec2<f32>(f32((b % tiles) * ${TILE_W}u), q[2].w + f32((b / tiles) * ${TILE_H}u));
  let l = clamp(c - bc * ${BRICK}.0, vec3<f32>(0.0), vec3<f32>(${BRICK}.0));
  let z0 = min(floor(l.z), ${BRICK - 1}.0);
  let s0 = sdfv_slice(tex, size, o + vec2<f32>(z0 * ${BRICK + 1}.0 + l.x + 0.5, l.y + 0.5));
  let s1 = sdfv_slice(tex, size, o + vec2<f32>((z0 + 1.0) * ${BRICK + 1}.0 + l.x + 0.5, l.y + 0.5));
  return mix(s0, s1, l.z - z0);
}

fn sdfv_oct_decode(e: vec2<f32>) -> vec3<f32> {
  var n = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
  if (n.z < 0.0) { n = vec3<f32>((1.0 - abs(n.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xy >= vec2<f32>(0.0)), n.z); }
  return normalize(n);
}

// orthoTangent of normal-volume.js: a unit tangent across n (Frisvad / Duff et al.).
fn sdfv_ortho_tangent(n: vec3<f32>) -> vec3<f32> {
  let s = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (s + n.z);
  let b = n.x * n.y * a;
  return vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
}

// The unit shading normal of the fragment: the rest-space volume gradient carried onto the drawn (posed, world)
// surface through the rest frame (uv, tangent); the mesh normal where the volume has nothing to say.
fn sdfv_shading_normal(in: MaterialInput) -> vec3<f32> {
  let n = normalize(in.normal);
  let g = sdfv_volume_gradient(in);
  let len = length(g);
  // Trilinear unit vectors keep a length near 1; a short one mixes opposite sides of a thin wall or empty nodes.
  if (!(len > 0.5)) { return n; }
  let gr = g / len;
  let nr = sdfv_oct_decode(in.uv);
  let tr = sdfv_ortho_tangent(nr);
  let br = cross(nr, tr);
  let tw = in.tangent.xyz - n * dot(n, in.tangent.xyz);
  if (dot(tw, tw) < 1e-12) { return n; }
  let t = normalize(tw);
  let bw = cross(n, t) * select(-1.0, 1.0, in.tangent.w >= 0.0);
  var m = normalize(dot(gr, tr) * t + dot(gr, br) * bw + dot(gr, nr) * n);
  if (dot(m, n) < 0.5) { return n; }
  // At the silhouette a normal the camera sees can lean past the view: keep it visible (no dark rim).
  let d = dot(m, in.view_dir);
  if (d < 0.02 && dot(n, in.view_dir) > 0.0) { m = normalize(m + in.view_dir * (0.02 - d)); }
  return m;
}
`;

const MAIN = /\bfn\s+material_main\s*\(/g;

/**
 * Wraps an S11 material function so its fragments shade with the volume normal: the base `material_main` becomes
 * `sdfv_base_main` and runs with the volume normal as `in.normal` and a white vertex colour (the colour stream carries
 * the rest position), then the output normal is the volume normal. Without volume params it is the base function.
 *
 * @param {string} functionName the base function's registered name
 * @param {string} wgsl the base function's WGSL (it defines material_main exactly once)
 * @returns {{ functionName: string, wgsl: string }}
 * @throws {SdfmError} `sdfm.material_unmappable` when the base does not define material_main exactly once or already
 *   uses the sdfv_ names
 */
export function volumeMaterial(functionName, wgsl) {
  if (typeof functionName !== 'string' || functionName === '' || typeof wgsl !== 'string') {
    throw new SdfmError('sdfm.material_unmappable', 'volumeMaterial: pass the base function name and its WGSL');
  }
  const mains = wgsl.match(MAIN) ?? [];
  if (mains.length !== 1) {
    throw new SdfmError('sdfm.material_unmappable', `material function '${functionName}' defines material_main ${mains.length} times; expected once`);
  }
  if (/\bsdfv_/.test(wgsl)) {
    throw new SdfmError('sdfm.material_unmappable', `material function '${functionName}' already uses sdfv_ names`);
  }
  const base = wgsl.replace(MAIN, 'fn sdfv_base_main(');
  return { functionName: VOLUME_FUNCTION_PREFIX + functionName, wgsl: `${base}\n${VOLUME_WGSL}\n${WRAPPED_MAIN}` };
}

const WRAPPED_MAIN = /* wgsl */ `
fn material_main(in: MaterialInput) -> MaterialOutput {
  let n = sdfv_shading_normal(in);
  var b = in;
  b.normal = n;
  b.color = vec4<f32>(1.0);
  var o = sdfv_base_main(b);
  o.normal = n;
  // Debug view (volumeInstanceParams debug): the shading normal as an emissive colour on black.
  if (in.instance_params[3].y > 0.5) {
    o.base_color = vec3<f32>(0.0);
    o.metallic = 0.0;
    o.roughness = 1.0;
    o.emissive = n * 0.5 + vec3<f32>(0.5);
  }
  return o;
}
`;

/** The wrapped standard surface: the material record's own values (raa_default_output), shaded with the volume normal. */
export function volumeStandardMaterial() {
  return volumeMaterial('standard', 'fn material_main(in: MaterialInput) -> MaterialOutput {\n  return raa_default_output(in);\n}\n');
}

/**
 * Checks a normal volume (the MAMA payload field) and returns it, or throws.
 * @throws {SdfmError} `sdfm.schema` naming the first bad field
 */
export function checkNormalVolume(v) {
  const bad = (what) => { throw new SdfmError('sdfm.schema', `normal volume: ${what}`); };
  if (v === null || typeof v !== 'object') bad('not an object');
  if (v.format !== NORMAL_VOLUME_FORMAT) bad(`format is not '${NORMAL_VOLUME_FORMAT}'`);
  if (!NORMAL_VOLUME_VERSIONS.includes(v.version)) bad(`version ${v.version} is not one of ${NORMAL_VOLUME_VERSIONS.join(', ')}`);
  const vec3 = (a) => Array.isArray(a) && a.length === 3 && a.every(Number.isFinite);
  if (!vec3(v.origin)) bad('origin is not 3 finite numbers');
  if (!(v.spacing > 0) || !Number.isFinite(v.spacing)) bad('spacing is not a positive number');
  if (!vec3(v.bricks) || v.bricks.some((n) => !Number.isInteger(n) || n < 1)) bad('bricks is not 3 positive integers');
  for (const k of ['count', 'width', 'height', 'tilesPerRow', 'indexRows']) {
    if (!Number.isInteger(v[k]) || v[k] < 1) bad(`${k} is not a positive integer`);
  }
  if (v.width !== v.tilesPerRow * TILE_W) bad('width is not tilesPerRow tiles');
  if (v.indexRows * v.width < v.bricks[0] * v.bricks[1] * v.bricks[2]) bad('the index rows do not hold every brick cell');
  if (v.height < v.indexRows + Math.ceil(v.count / v.tilesPerRow) * TILE_H) bad('the image does not hold every brick');
  if (!(v.texels instanceof Uint8Array) || v.texels.length !== v.width * v.height * 4) bad('texels is not width x height rgba8');
  return v;
}

/** The Engine.addTexture argument of a checked volume. */
export function volumeTexture(v) {
  return { width: v.width, height: v.height, data: v.texels };
}

/**
 * The 16 Engine.setInstanceParams floats of a volume uploaded as texture `textureId`. `debug` shows the shading normal
 * as an emissive colour (0.5 + n / 2) instead of the material.
 */
export function volumeInstanceParams(v, textureId, options = {}) {
  return [
    v.origin[0], v.origin[1], v.origin[2], v.spacing,
    v.bricks[0], v.bricks[1], v.bricks[2], textureId,
    v.width, v.height, v.tilesPerRow, v.indexRows,
    1, options.debug ? 1 : 0, 0, 0,
  ];
}

/** A unit tangent across the unit normal `n` (Frisvad, Duff et al. 2017); sdfv_ortho_tangent is the same in WGSL. */
export function orthoTangent(n) {
  const s = n[2] >= 0 ? 1 : -1, a = -1 / (s + n[2]), b = n[0] * n[1] * a;
  return [1 + s * n[0] * n[0] * a, s * b, -s * n[0]];
}

function octEncode(n) {
  const l1 = Math.abs(n[0]) + Math.abs(n[1]) + Math.abs(n[2]) || 1;
  let x = n[0] / l1, y = n[1] / l1;
  if (n[2] < 0) [x, y] = [(1 - Math.abs(y)) * (x >= 0 ? 1 : -1), (1 - Math.abs(x)) * (y >= 0 ? 1 : -1)];
  return [x, y];
}

function octDecode(e) {
  let n = [e[0], e[1], 1 - Math.abs(e[0]) - Math.abs(e[1])];
  if (n[2] < 0) n = [(1 - Math.abs(n[1])) * (n[0] >= 0 ? 1 : -1), (1 - Math.abs(n[0])) * (n[1] >= 0 ? 1 : -1), n[2]];
  const l = Math.hypot(n[0], n[1], n[2]) || 1;
  return n.map((c) => c / l);
}

/**
 * The per-vertex streams of a volume-shaded mesh (positions and normals in the part's mesh space, at rest): colours =
 * the rest position (w = 1), uvs = the octahedral rest normal, tangents = orthoTangent of the decoded normal.
 * @param {Float32Array} positions
 * @param {Float32Array} normals
 * @returns {{ colors: Float32Array, uvs: Float32Array, tangents: Float32Array }}
 */
export function volumeSurfaceAttributes(positions, normals) {
  const count = positions.length / 3;
  const colors = new Float32Array(count * 4), uvs = new Float32Array(count * 2), tangents = new Float32Array(count * 4);
  for (let v = 0; v < count; v++) {
    colors[v * 4] = positions[v * 3]; colors[v * 4 + 1] = positions[v * 3 + 1]; colors[v * 4 + 2] = positions[v * 3 + 2];
    colors[v * 4 + 3] = 1;
    let n = [normals[v * 3], normals[v * 3 + 1], normals[v * 3 + 2]];
    const l = Math.hypot(n[0], n[1], n[2]);
    n = l > 1e-12 ? n.map((c) => c / l) : [0, 0, 1];
    const e = octEncode(n);
    uvs[v * 2] = e[0]; uvs[v * 2 + 1] = e[1];
    // The tangent of the DECODED normal, so the WGSL rebuilds the very same rest frame.
    const t = orthoTangent(octDecode(e));
    tangents[v * 4] = t[0]; tangents[v * 4 + 1] = t[1]; tangents[v * 4 + 2] = t[2]; tangents[v * 4 + 3] = 1;
  }
  return { colors, uvs, tangents };
}
