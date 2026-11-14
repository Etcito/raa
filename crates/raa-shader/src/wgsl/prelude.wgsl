// RAA prelude (S10 structs and bindings of groups 0-1, S11 material structs).

// S10: four 16-byte lanes; stable handles and authoring metadata remain on the CPU.
struct LightGpu {
    position_range: vec4<f32>,       // 0: world xyz, range in metres
    color_inv_range_sq: vec4<f32>,   // 16: linear RGB * intensity, inverse range squared
    direction_cos_outer: vec4<f32>,  // 32: unit travel xyz, outer half-angle cosine
    cos_inner: f32,                 // 48
    kind: u32,                      // 52: point 0, spot 1, directional 2, area 3
    width: f32,                     // 56: full extent in metres
    height: f32,                    // 60: full extent in metres
}

struct FrameGpu {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    camera_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    ambient: vec4<f32>,
    viewport: vec4<f32>,
    clip: vec4<f32>,
    flags: vec4<u32>,
    exposure: vec4<f32>,
    reserved: array<vec4<f32>, 4>,
    previous_view_proj: mat4x4<f32>,
    unjittered_view_proj: mat4x4<f32>,
    // Current NDC xy followed by previous NDC xy; zero when no consumer requests jitter.
    camera_jitter: vec4<f32>,
    // Active light count, tile columns, tile rows and tile stride in words.
    light_info: vec4<u32>,
    // Frame parity, accumulation enabled (0/1), history valid (0/1), reserved zero.
    temporal: vec4<u32>,
    reserved2: array<vec4<f32>, 5>,
}

struct SurfaceMaterialGpu {
    base: vec4<f32>,
    emissive: vec4<f32>,
    surface: vec4<f32>,
    sheen: vec4<f32>,
    transmission: vec4<f32>,
    attenuation: vec4<f32>,
    header: vec4<u32>,
    p: array<vec4<f32>, 8>,
    fur: vec4<f32>,
}

struct InstanceExtraGpu {
    tint: vec4<f32>,
    squash: vec4<f32>,
    jiggle: vec4<f32>,
    params: array<vec4<f32>, 4>,
    reserved: vec4<f32>,
}

struct ShadowGpu {
    cascade_view_proj: array<mat4x4<f32>, 4>,
    splits: vec4<f32>,
    params: vec4<f32>,
    params2: vec4<f32>,
    reserved: vec4<f32>,
}

struct EnvSh {
    sh: array<vec4<f32>, 9>,
    info: vec4<f32>,
}

struct EnvGpu {
    sun_disc: vec4<f32>,
    zenith: vec4<f32>,
    horizon: vec4<f32>,
    ground: vec4<f32>,
    sky_params: vec4<f32>,
    background: vec4<f32>,
    clouds: vec4<f32>,
    clouds2: vec4<f32>,
    cloud_color: vec4<f32>,
    fog: vec4<f32>,
    fog_color: vec4<f32>,
    aerial: vec4<f32>,
    // reserved[0]: cel toon enabled, softness, specular strength, unused.
    reserved: array<vec4<f32>, 4>,
}

// Persistent instance record (S1 Instance, 128 B): model columns @0, normal matrix columns
// @64, material index @112, tag @116, skin palette base @120. Indexed by the stable scene
// instance slot carried by the compacted id stream.
struct RaaInstanceGpu {
    model0: vec4<f32>,
    model1: vec4<f32>,
    model2: vec4<f32>,
    model3: vec4<f32>,
    normal0: vec4<f32>,
    normal1: vec4<f32>,
    normal2: vec4<f32>,
    material_index: u32,
    tag: u32,
    palette_base: u32,
    reserved: u32,
}

// S10 affine ROWS preserve translation; validity belongs to the same scene slot.
struct RaaPreviousInstanceGpu {
    row0: vec4<f32>,
    row1: vec4<f32>,
    row2: vec4<f32>,
    previous_valid: u32,
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
}

// Vertex-only binding. Pass history/validity through varyings; never read in fragments.
@group(0) @binding(8) var<storage, read> raa_previous_instances: array<RaaPreviousInstanceGpu>;

fn raa_previous_model(r: RaaPreviousInstanceGpu) -> mat4x4<f32> {
    return mat4x4<f32>(
        vec4<f32>(r.row0.x, r.row1.x, r.row2.x, 0.0),
        vec4<f32>(r.row0.y, r.row1.y, r.row2.y, 0.0),
        vec4<f32>(r.row0.z, r.row1.z, r.row2.z, 0.0),
        vec4<f32>(r.row0.w, r.row1.w, r.row2.w, 1.0),
    );
}

@group(0) @binding(0) var<uniform> raa_frame: FrameGpu;
@group(0) @binding(1) var<storage, read> raa_materials: array<SurfaceMaterialGpu>;
@group(0) @binding(2) var<storage, read> raa_extras: array<InstanceExtraGpu>;
// Material textures (raa-lit atlas): rgba8unorm atlas, its linear clamp sampler and the rect
// of each texture id, (x, y, width, height) in atlas uv. Id 0 and removed ids are zero rects.
@group(0) @binding(3) var raa_texture_atlas: texture_2d<f32>;
@group(0) @binding(4) var raa_texture_sampler: sampler;
@group(0) @binding(5) var<storage, read> raa_texture_rects: array<vec4<f32>>;
// Material data words (raa-lit add_material_data, JS addMaterialData): read with
// raa_material_word / raa_material_f32 from the base of the material record
// (raa_material_data_base). Words 0..4 are reserved zeros.
@group(0) @binding(6) var<storage, read> raa_material_data: array<u32>;
// Persistent instance records, indexed by the stable scene instance slot (the value of the
// compacted id stream).
@group(0) @binding(7) var<storage, read> raa_instances: array<RaaInstanceGpu>;

@group(1) @binding(0) var raa_shadow_map: texture_depth_2d_array;
@group(1) @binding(1) var raa_shadow_cmp: sampler_comparison;
@group(1) @binding(2) var<uniform> raa_shadow: ShadowGpu;
@group(1) @binding(3) var raa_env_specular: texture_cube<f32>;
@group(1) @binding(4) var<uniform> raa_env_sh: EnvSh;
// 5: the BRDF/LTC atlas (Rgba16Float, 128 x 192 texels): BRDF split-sum at x 0..127,
// y 0..127, LTC1 at x 0..63, y 128..191 and LTC2 at x 64..127, y 128..191. Sampled with
// raa_linear_clamp; raa_brdf_lut_uv and raa_ltc_uv (below) select its texels.
@group(1) @binding(5) var raa_brdf_lut: texture_2d<f32>;
@group(1) @binding(6) var raa_linear_clamp: sampler;
@group(1) @binding(7) var raa_ao_tex: texture_2d<f32>;
@group(1) @binding(8) var raa_contact_tex: texture_2d<f32>;
@group(1) @binding(9) var raa_opaque_color: texture_2d<f32>;
@group(1) @binding(10) var raa_opaque_depth: texture_2d<f32>;
@group(1) @binding(11) var<uniform> raa_env: EnvGpu;
// Fragment-only shared resources; compute tiling uses its own layout and equivalent bytes.
@group(1) @binding(12) var<uniform> raa_local_lights: array<LightGpu, MAX_LOCAL_LIGHTS>;
@group(1) @binding(13) var raa_light_tiles: texture_2d_array<u32>;
// S38 GI declarations for fragment consumers; the GI compute passes bind their own group 1.
struct RaaGiVolumeParams { levels: array<vec4<f32>, 5>, resolution: f32, intensity: f32, level_count: u32, flags: u32 }
const RAA_GI_MAX_LEVELS: u32 = 5u;
const RAA_GI_FLAG_VALID: u32 = 1u;
const RAA_GI_FLAG_REFLECTIONS: u32 = 2u;
const RAA_GI_FLAG_LOCAL_SHADOWS: u32 = 4u;
// 14: screen irradiance and occlusion.
@group(1) @binding(14) var raa_gi_screen: texture_2d<f32>;
// 17: the volume atlas (rgb radiance, a distance in metres).
@group(1) @binding(17) var raa_gi_volume: texture_3d<f32>;
// 18: the 96-B params, level_count 0 = no volume.
@group(1) @binding(18) var<uniform> raa_gi_volume_params: RaaGiVolumeParams;
// The generated prelude appends weather_lighting.wgsl
// exactly once for the fragment-only texture/uniform bindings 15/16; no added storage.

// SurfaceMaterialGpu.header.x models and header.z flags.
const RAA_MODEL_FLAT: u32 = 0u;
const RAA_MODEL_STANDARD: u32 = 1u;
const RAA_MODEL_FUNCTION: u32 = 2u;
const RAA_FLAG_TRANSPARENT: u32 = 1u;
const RAA_FLAG_TRANSMISSIVE: u32 = 2u;
const RAA_FLAG_FUR: u32 = 4u;
const RAA_FLAG_DOUBLE_SIDED: u32 = 8u;
const RAA_FLAG_NO_SHADOW: u32 = 16u;
const RAA_FLAG_VERTEX_FN: u32 = 32u;
const RAA_FLAG_UNLIT: u32 = 64u;
const RAA_FLAG_CEL: u32 = 128u;

// Vertex stream 0 (S1 Vertex, 24 B).
struct RaaVertex0 {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

// One instance as the lit passes use it: the S1 instance record's matrices, its material
// index and its skin palette base, read by the stable slot in the compacted id stream.
struct RaaInstance {
    model0: vec4<f32>,
    model1: vec4<f32>,
    model2: vec4<f32>,
    model3: vec4<f32>,
    normal0: vec4<f32>,
    normal1: vec4<f32>,
    normal2: vec4<f32>,
    material_index: u32,
    palette_base: u32,
}

// The instance of compacted id `id` (a stable scene instance slot).
fn raa_instance(id: u32) -> RaaInstance {
    let r = raa_instances[id];
    return RaaInstance(
        r.model0,
        r.model1,
        r.model2,
        r.model3,
        r.normal0,
        r.normal1,
        r.normal2,
        r.material_index,
        r.palette_base,
    );
}

// Vertex stream 1 (S10, 40 B).
struct RaaVertex1 {
    @location(10) uv: vec2<f32>,
    @location(11) tangent: vec4<f32>,
    @location(12) color: vec4<f32>,
}

struct MaterialInput {
    position: vec3<f32>,
    local_position: vec3<f32>,
    normal: vec3<f32>,
    tangent: vec4<f32>,
    uv: vec2<f32>,
    color: vec4<f32>,
    view_dir: vec3<f32>,
    time: f32,
    front_facing: bool,
    frag_coord: vec4<f32>,
    instance_params: array<vec4<f32>, 4>,
    record: SurfaceMaterialGpu,
}

struct MaterialOutput {
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    emissive: vec3<f32>,
    normal: vec3<f32>,
    ao: f32,
    opacity: f32,
    clearcoat: f32,
    clearcoat_roughness: f32,
    sheen_color: vec3<f32>,
    sheen_roughness: f32,
    transmission: f32,
    ior: f32,
    thickness: f32,
    attenuation_color: vec3<f32>,
    attenuation_distance: f32,
}

struct VertexMaterialInput {
    local_position: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
    color: vec4<f32>,
    time: f32,
    instance_params: array<vec4<f32>, 4>,
    record: SurfaceMaterialGpu,
}

struct VertexMaterialOutput {
    local_position: vec3<f32>,
    normal: vec3<f32>,
}

struct ShadeCtx {
    material_flags: u32,
    position: vec3<f32>,
    view_dir: vec3<f32>,
    frag: vec4<f32>,
    n_dot_v: f32,
    ao: f32,
    shadow: f32,
    f0: vec3<f32>,
}

fn raa_default_output(in: MaterialInput) -> MaterialOutput {
    let r = in.record;
    var o: MaterialOutput;
    o.base_color = r.base.rgb * in.color.rgb;
    o.metallic = r.surface.y;
    o.roughness = r.surface.x;
    o.emissive = r.emissive.rgb;
    o.normal = in.normal;
    o.ao = 1.0;
    o.opacity = r.base.a * in.color.a;
    o.clearcoat = r.surface.z;
    o.clearcoat_roughness = r.surface.w;
    o.sheen_color = r.sheen.rgb;
    o.sheen_roughness = r.sheen.w;
    o.transmission = r.transmission.x;
    o.ior = r.transmission.y;
    o.thickness = r.transmission.z;
    o.attenuation_color = r.attenuation.rgb;
    o.attenuation_distance = r.attenuation.w;
    return o;
}

// Instance tint of the shaded instance, readable by material graph code (S23). Each pass
// sets both from InstanceExtraGpu with raa_set_instance_tint before material_main.
// `mat_instance_tint_set` is 1 after setInstanceTint and 0 after clearInstanceTint; it is
// the spare lane InstanceExtraGpu.reserved.x (the record layout is unchanged).
var<private> mat_instance_tint: vec4<f32>;
var<private> mat_instance_tint_set: f32;

fn raa_set_instance_tint(x: InstanceExtraGpu) {
    mat_instance_tint = x.tint;
    mat_instance_tint_set = select(0.0, 1.0, x.reserved.x > 0.5);
}

// Instance tint (S10): rgb multiplies the base colour, a multiplies the opacity.
fn raa_apply_tint(s: MaterialOutput, x: InstanceExtraGpu) -> MaterialOutput {
    var o = s;
    o.base_color = o.base_color * x.tint.rgb;
    o.opacity = o.opacity * x.tint.a;
    return o;
}

// Tint mode `graph` (S23): the material graph used mat_instance_tint itself, so only the
// opacity multiply remains.
fn raa_apply_tint_opacity(s: MaterialOutput, x: InstanceExtraGpu) -> MaterialOutput {
    var o = s;
    o.opacity = o.opacity * x.tint.a;
    return o;
}

// Base word of the material's data block (SurfaceMaterialGpu.header.w; 0 = no data, the
// reserved zero words). Pass `in.record` of MaterialInput or VertexMaterialInput.
fn raa_material_data_base(record: SurfaceMaterialGpu) -> u32 {
    return record.header.w;
}

// Material data word `i` (absolute: base + offset); 0 past the end of the buffer.
fn raa_material_word(i: u32) -> u32 {
    let n = arrayLength(&raa_material_data);
    return select(0u, raa_material_data[min(i, n - 1u)], i < n);
}

// Material data word `i` read as the f32 it holds (bitcast).
fn raa_material_f32(i: u32) -> f32 {
    return bitcast<f32>(raa_material_word(i));
}

// Material texture `id` (Engine.addTexture) at `uv`, (0, 0) the first texel row's left edge,
// clamped to [0, 1] like a clamp-to-edge texture and kept half a texel inside the texture's
// atlas rect, so linear filtering never reads a neighbour. An unknown id gives (0, 0, 0, 0).
// Level 0 explicitly, so material code may call it in non-uniform control flow.
fn raa_texture(id: u32, uv: vec2<f32>) -> vec4<f32> {
    let r = raa_texture_rects[min(id, arrayLength(&raa_texture_rects) - 1u)];
    let half_texel = 0.5 / vec2<f32>(textureDimensions(raa_texture_atlas));
    let p = r.xy + clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * r.zw;
    let q = clamp(p, r.xy + half_texel, max(r.xy + r.zw - half_texel, r.xy + half_texel));
    let c = textureSampleLevel(raa_texture_atlas, raa_texture_sampler, q, 0.0);
    return select(vec4<f32>(0.0), c, r.z > 0.0);
}

// S11 area lights: shared rectangle form-factor maths and the BRDF/LTC atlas uv helpers.
// Pure functions: no binding access, no texture sampling and no bitwise operator, so the
// forward passes and every GI builder that starts with PRELUDE_WGSL can call them.
// Adapted from three.js r180 src/renderers/shaders/ShaderChunk/lights_physical_pars_fragment.glsl.js (LTC_EdgeVectorFormFactor, LTC_ClippedSphereFormFactor, LTC_Uv), MIT licence, Copyright (c) 2010-2025 three.js authors; Heitz, Dupuy, Hill, Neubelt, SIGGRAPH 2016.

// S11 rectangle basis: forward is the light's travel direction d; up0 is Y unless d is
// within 0.99 of Y, then X; tangent = normalize(cross(up0, d)); bitangent = cross(d, tangent).
// The rectangle is centred on the light and spans width along tangent, height along bitangent.
fn raa_area_basis(travel: vec3<f32>) -> mat3x3<f32> {
    let up0 = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
        abs(dot(travel, vec3<f32>(0.0, 1.0, 0.0))) < 0.99);
    let tangent = normalize(cross(up0, travel));
    return mat3x3<f32>(tangent, cross(travel, tangent), travel);
}

// The four rectangle corners around `origin`, relative to it: [p - T - B, p - T + B,
// p + T + B, p + T - B] with p = light position - origin, T and B the half extents along the
// basis tangent and bitangent. Counter-clockwise seen from the emitting side, so the vector
// form factor at a receiver in front of the light points towards the light.
fn raa_area_corners(light: LightGpu, origin: vec3<f32>) -> array<vec3<f32>, 4> {
    let m = raa_area_basis(light.direction_cos_outer.xyz);
    let t = m[0] * max(light.width, 0.0) * 0.5;
    let b = m[1] * max(light.height, 0.0) * 0.5;
    let p = light.position_range.xyz - origin;
    return array<vec3<f32>, 4>(p - t - b, p - t + b, p + t + b, p + t - b);
}

// three.js r180 LTC_EdgeVectorFormFactor: one edge's contribution, theta / sin(theta) / 2PI
// as the rational polynomial v, with the 0.5 / sin(theta) branch when the edge spans the
// horizon (x <= 0).
fn raa_area_edge(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let x = dot(a, b);
    let y = abs(x);
    let v = (0.8543985 + (0.4965155 + 0.0145206 * y) * y)
        / (3.4175940 + (4.1616724 + y) * y);
    let theta_sintheta = select(0.5 * inverseSqrt(max(1.0 - x * x, 1e-7)) - v, v, x > 0.0);
    return cross(a, b) * theta_sintheta;
}

// Sum of the four edge form factors over the unit directions of the corners transformed by
// `m`: the polygon's vector form factor. `m` is the identity for the diffuse evaluation.
fn raa_area_vector_form_factor(corners: array<vec3<f32>, 4>, m: mat3x3<f32>) -> vec3<f32> {
    var u: array<vec3<f32>, 4>;
    for (var i = 0; i < 4; i = i + 1) {
        u[i] = normalize(m * corners[i]);
    }
    var result = vec3<f32>(0.0);
    for (var i = 0; i < 4; i = i + 1) {
        result += raa_area_edge(u[i], u[(i + 1) % 4]);
    }
    return result;
}

// three.js r180 LTC_ClippedSphereFormFactor: the horizon-clipped sphere form factor of the
// vector form factor f, with `z` = dot(f, normal). Kept in [0, 1].
fn raa_area_clipped(length_f: f32, z: f32) -> f32 {
    return max((length_f * length_f + z) / (length_f + 1.0), 0.0);
}

// Handoff: card "raa-gi: voxel lighting of area lights by the diffuse form factor" takes
// `raa_area_form_factor(light: LightGpu, origin: vec3<f32>, normal: vec3<f32>) -> f32` from
// this file. It returns 0 behind the emitter; otherwise the horizon-clipped sphere form
// factor of the rectangle in [0, 1] (irradiance = pi x radiance x form factor).
fn raa_area_form_factor(light: LightGpu, origin: vec3<f32>, normal: vec3<f32>) -> f32 {
    if dot(origin - light.position_range.xyz, light.direction_cos_outer.xyz) <= 0.0 {
        return 0.0;
    }
    let f = raa_area_vector_form_factor(
        raa_area_corners(light, origin),
        mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0)),
    );
    return raa_area_clipped(length(f), dot(f, normal));
}

// S17 area-light radiance: the intensity carried by color_inv_range_sq.rgb spread over the
// rectangle's area (intensity / (width x height)), at least (1e-8, 1e-8).
fn raa_area_radiance(light: LightGpu) -> vec3<f32> {
    return light.color_inv_range_sq.rgb / max(max(light.width, 0.0) * max(light.height, 0.0), 1e-8);
}

// Range window without an inverse square: 0 at or beyond `range`, else
// (clamp(1 - (distance / range)^4, 0, 1))^2.
fn raa_area_window(distance: f32, range: f32) -> f32 {
    if distance >= range { return 0.0; }
    let q = distance / range;
    let window = clamp(1.0 - q * q * q * q, 0.0, 1.0);
    return window * window;
}

// The atlas region sizes of group 1 binding 5 in texels, (width, height).
const RAA_BRDF_ATLAS_SIZE: vec2<f32> = vec2<f32>(128.0, 192.0);

// Handoff: card "raa-ibl and raa-ssr: BRDF LUT written into and read from the atlas region"
// takes `raa_brdf_lut_uv(n_dot_v: f32, roughness: f32) -> vec2<f32>` from this file. It maps
// a BRDF split-sum lookup (x along n_dot_v, y along roughness) to the atlas region at
// x 0..127, y 0..127, kept half a texel inside.
fn raa_brdf_lut_uv(n_dot_v: f32, roughness: f32) -> vec2<f32> {
    return clamp(vec2<f32>(n_dot_v, roughness) * 128.0, vec2<f32>(0.5), vec2<f32>(127.5))
        / vec2<f32>(128.0, 192.0);
}

// Handoff: card "raa-shader: LTC area light shading in local_lighting.wgsl" takes
// `raa_ltc_uv(table: u32, roughness: f32, n_dot_v: f32) -> vec2<f32>` from this file; it also
// uses `raa_area_corners`, `raa_area_vector_form_factor`, `raa_area_clipped`,
// `raa_area_form_factor`, `raa_area_radiance` and `raa_area_window`, all defined here today.
// Table 0 is LTC1 at x 0..63 and table 1 LTC2 at x 64..127; roughness runs along x,
// sqrt(1 - n_dot_v) along y (three.js r180 LTC_Uv, half a texel inside).
fn raa_ltc_uv(table: u32, roughness: f32, n_dot_v: f32) -> vec2<f32> {
    return (vec2<f32>(64.0 * f32(table), 128.0)
        + vec2<f32>(clamp(roughness, 0.0, 1.0), sqrt(1.0 - clamp(n_dot_v, 0.0, 1.0))) * 63.0
        + vec2<f32>(0.5)) / vec2<f32>(128.0, 192.0);
}
// End of the S11 area form-factor section.
