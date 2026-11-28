// RAA prelude (S10 structs and bindings of groups 0-1, S11 material structs).

// S10: four 16-byte lanes; stable handles and authoring metadata remain on the CPU.
struct LightGpu {
    position_range: vec4<f32>,       // 0: world xyz, range in metres
    color_inv_range_sq: vec4<f32>,   // 16: linear RGB * intensity, inverse range squared
    direction_cos_outer: vec4<f32>,  // 32: unit travel xyz, outer half-angle cosine
    cos_inner: f32,                 // 48
    kind: u32,                      // 52: point 0, spot 1, directional 2, area 3
    width: f32,                     // 56: full extent in metres
    height: f32,                    // 60: full extent in metres (area); point/spot: traced shadow slot + 1, 0 = none, -layer = own map in raa_shadow_map
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
    // Frame parity, accumulation enabled (0/1), history valid (0/1), forward "layers-interlaced"
    // word (raa-pipeline LightingInterlace::word: bit 0 mode, bit 1 R frame, bits 2-3 phase
    // px/py, bits 4-5 lighting frame % 4; 0 with the mode off).
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
    gi: vec4<f32>,
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

// Clamp to the neutral 1x1 texture when the screen-space AO pass is disabled.
fn raa_gtao_at(frag: vec4<f32>) -> f32 {
    let size = vec2<i32>(textureDimensions(raa_ao_tex));
    let pixel = clamp(vec2<i32>(frag.xy), vec2<i32>(0), size - vec2<i32>(1));
    return textureLoad(raa_ao_tex, pixel, 0).r;
}
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
// S44 area-light stochastic shadows: the 80-B mid-tier slot table of the shadow mask.
struct RaaAreaStochasticGpu { lights: array<vec4<u32>, 2>, weights: array<vec4<f32>, 2>, params: vec4<u32> }
// 19: the slot table (packed light index or 0xFFFFFFFF, w_stoch per slot; params.x limit).
@group(1) @binding(19) var<uniform> raa_area_stochastic: RaaAreaStochasticGpu;
// 20: the Rgba16Float viewport mask; slot s is layer s / 4, channel s % 4.
@group(1) @binding(20) var raa_area_shadow_mask: texture_2d_array<f32>;
// 22: cached local-light cube shadows (R32Float): layer 6 x slot + face (+X, -X, +Y, -Y, +Z, -Z),
// texel = distance from the light to the first occluder, 0 = not baked (read as unoccluded).
@group(1) @binding(22) var raa_local_shadow_map: texture_2d_array<f32>;
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
const RAA_FLAG_TRACE_MIRROR: u32 = 256u;
const RAA_FLAG_DIFFUSE_TRANSMISSION: u32 = 512u;

fn raa_reflection_class(s: MaterialOutput, record: SurfaceMaterialGpu) -> f32 {
    let flags = record.header.z;
    if (flags & RAA_FLAG_TRACE_MIRROR) != 0u { return 1.0; }
    let non_pbr = RAA_FLAG_UNLIT | RAA_FLAG_TRANSPARENT | RAA_FLAG_TRANSMISSIVE | RAA_FLAG_FUR | RAA_FLAG_CEL;
    // Stock SSR reads these values back in the exact G-buffer formats.
    let metallic = round(clamp(s.metallic, 0.0, 1.0) * 255.0) / 255.0;
    let roughness = quantizeToF16(clamp(s.roughness, 0.0, 1.0));
    return select(0.0, 2.0, record.header.x != RAA_MODEL_FLAT
        && (flags & non_pbr) == 0u && raa_near_mirror(metallic, roughness));
}

// Vertex stream 0 (S1 Vertex, 24 B).
struct RaaVertex0 {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

// The reflected camera has its own geometry and depth, with no main-view screen inputs.
const RAA_PLANAR_CAPTURE_FEATURE: u32 = 40u;

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
    spherical_diffuse: vec4<f32>,
    // Screen reflection of a transmissive fragment (transparent pass, opaque colour copy hit):
    // rgb incoming radiance along the reflected ray, a its weight. Zero elsewhere.
    screen_reflection: vec4<f32>,
}

fn raa_spherical_blend_diffuse(legacy: vec3<f32>, s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    if c.spherical_diffuse.w <= 0.0 { return legacy; }
    let spherical = c.spherical_diffuse.rgb * s.base_color * (1.0 - clamp(s.metallic, 0.0, 1.0)) * s.ao;
    return mix(legacy, spherical, c.spherical_diffuse.w);
}

// Environment specular under DDGI: the sky cube is unoccluded, so inside a room every surface
// would reflect the sky (a closed room lit only through its door reflected more sky than it
// received light). The probes know how much of the environment reaches the point: the ratio of
// their irradiance to the unoccluded sky irradiance (`sky_diffuse`, the ambient diffuse factor
// that the probe term replaces, same units as `spherical_diffuse.rgb`), at most 1, scales the
// environment term. Outdoors the probes (sky plus bounced sun) exceed the sky and nothing changes.
// While raa_probe_sky_unoccluded is set (the forward layer split, opaque_layers.wgsl, around its
// ambient measurements) the term returns env with the sky fully visible and records the
// denominator max(luma(sky_diffuse), 1e-6) in raa_probe_sky_luma (unchanged where the term does
// not apply), so that the layers carry vis as a separate factor.
var<private> raa_probe_sky_unoccluded: bool;
var<private> raa_probe_sky_luma: f32;
// Set by blended (OIT) fragments (pass_transparent.wgsl) before their shadow lookups: the sun
// shadow provider may answer with a cheaper kernel (the virtual shadow map skips its blocker
// search, crates/raa-shadow/src/wgsl/vsm.wgsl). Overdraw of transparent layers makes the full
// PCSS there cost more than it shows.
var<private> raa_shadow_lite: bool;

fn raa_probe_sky_environment(env: vec3<f32>, sky_diffuse: vec3<f32>, c: ShadeCtx) -> vec3<f32> {
    if !raa_feature_enabled(RAA_DDGI_GI_FEATURE) || c.spherical_diffuse.w <= 0.0 { return env; }
    let luma = vec3<f32>(0.2126, 0.7152, 0.0722);
    let sky_luma = max(dot(sky_diffuse, luma), 1e-6);
    if raa_probe_sky_unoccluded {
        raa_probe_sky_luma = sky_luma;
        return env;
    }
    let visible = clamp(dot(c.spherical_diffuse.rgb, luma) / sky_luma, 0.0, 1.0);
    return env * mix(1.0, visible, c.spherical_diffuse.w);
}

// Sun disc and glow in the environment seen by a transmissive fragment (the transparent pass
// sets RAA_FLAG_TRANSMISSIVE in ShadeCtx.material_flags): the radiance above 16x the sky's
// average (luminance of `sky_diffuse`) is the sun, which the fragment sees only where the sun
// reaches it. That excess is scaled by the sun visibility (ShadeCtx.shadow) and capped at 4x
// the threshold, so a rippled water surface in shade shows no sun glints and a sunlit one no
// single-pixel fireflies (its direct term already carries the sun highlight).
fn raa_sky_sun_peak(env: vec3<f32>, sky_diffuse: vec3<f32>, c: ShadeCtx) -> vec3<f32> {
    if (c.material_flags & RAA_FLAG_TRANSMISSIVE) == 0u { return env; }
    let luma = vec3<f32>(0.2126, 0.7152, 0.0722);
    let lum = dot(env, luma);
    let cap = 16.0 * max(dot(sky_diffuse, luma), 1e-4);
    if lum <= cap { return env; }
    let keep = cap + min(lum - cap, 4.0 * cap) * clamp(c.shadow, 0.0, 1.0);
    return env * (keep / lum);
}

// The incoming specular radiance of the ambient providers: the environment occluded as above,
// replaced by the screen reflection (ShadeCtx.screen_reflection) by its weight. Opaque passes
// leave that weight 0, so the result is the occluded environment unchanged.
fn raa_probe_sky_specular(env: vec3<f32>, sky_diffuse: vec3<f32>, c: ShadeCtx) -> vec3<f32> {
    let occluded = raa_probe_sky_environment(raa_sky_sun_peak(env, sky_diffuse, c), sky_diffuse, c);
    let w = clamp(c.screen_reflection.a, 0.0, 1.0);
    if w <= 0.0 { return occluded; }
    return mix(occluded, c.screen_reflection.rgb, w);
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

// Material atlas mip levels (S48): levels 0..4 of the raa-lit atlas.
const RAA_ATLAS_MIP_LEVELS: f32 = 5.0;

// Material textures (Engine.addTexture), S48. Rect encoding: binding 5 holds (x, y, w, h) /
// atlas size per texture id; a negative w marks a texture added with `mips: false`, sampled
// at level 0 only; the extent is abs(r.zw). An unknown or removed id is all zero and samples
// (0, 0, 0, 0). uv (0, 0) is the first texel row's left edge, clamped to [0, 1] like a
// clamp-to-edge texture and kept inside the texture's atlas rect, so filtering never reads
// a neighbour.
// Derivative rule: call the WGSL built-ins dpdx and dpdy on the uv once at the top of
// `material_main`, in uniform control flow, from the unwrapped uv (never `fract(uv)`), then
// pass them down. raa_texture_grad takes no derivative itself, so it may be called in
// branches; a caller that cannot take derivatives keeps raa_texture.
// raa_texture: level 0 explicitly, half a texel inset, callable in non-uniform control flow.
fn raa_texture(id: u32, uv: vec2<f32>) -> vec4<f32> {
    let r = raa_texture_rects[min(id, arrayLength(&raa_texture_rects) - 1u)];
    let e = abs(r.zw);
    let half_texel = 0.5 / vec2<f32>(textureDimensions(raa_texture_atlas));
    let p = r.xy + clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * e;
    let q = clamp(p, r.xy + half_texel, max(r.xy + e - half_texel, r.xy + half_texel));
    let c = textureSampleLevel(raa_texture_atlas, raa_texture_sampler, q, 0.0);
    return select(vec4<f32>(0.0), c, r.z != 0.0);
}

// Material texture `id` at `uv` with an explicit lod from the caller's uv derivatives
// (S48; CPU twins texture_levels, texture_lod, level_inset in raa-lit textures.rs).
// Handoff: raa_texture_grad samples texture `id` at uv clamped to [0, 1] with
// lod = clamp(0.5 * log2(rho2), 0, levels - 1) and an inset of 0.5 * exp2(ceil(lod)) / atlas;
// "raa-sdfm: baked material samples raa_texture_grad; colour bake added with srgb" and
// "post page: mips-ground scene and its test" call raa_texture_grad(id, uv, duv_dx, duv_dy)
// from their material WGSL with derivatives taken at the top of `material_main`.
// Largest footprint stretch the 4 taps of raa_texture_grad cover before the lod rises.
const RAA_ATLAS_MAX_ANISO: f32 = 4.0;

fn raa_texture_grad(id: u32, uv: vec2<f32>, duv_dx: vec2<f32>, duv_dy: vec2<f32>) -> vec4<f32> {
    let r = raa_texture_rects[min(id, arrayLength(&raa_texture_rects) - 1u)];
    let e = abs(r.zw);
    let atlas = vec2<f32>(textureDimensions(raa_texture_atlas, 0));
    let size = e * atlas;
    let levels = select(
        min(RAA_ATLAS_MIP_LEVELS, 1.0 + floor(log2(max(min(size.x, size.y), 1.0)))),
        1.0,
        r.z < 0.0,
    );
    let gx = duv_dx * size;
    let gy = duv_dy * size;
    // Anisotropic footprint: the lod follows the minor axis (or the major axis / 4 when the
    // footprint is stretched more than 4:1) and 4 taps spread along the major axis cover the
    // rest, so a grazing near ground stays at level 0 instead of the major axis' coarse level.
    let lx = dot(gx, gx);
    let ly = dot(gy, gy);
    let major2 = max(lx, ly);
    let minor2 = min(lx, ly);
    let rho2 = max(minor2, major2 / (RAA_ATLAS_MAX_ANISO * RAA_ATLAS_MAX_ANISO));
    // reserved2[0].x: negative mip bias while a jittered frame renders below the display size
    // (temporal upscaling), so material textures keep the display resolution's detail.
    let lod = clamp(0.5 * log2(max(rho2, 1e-12)) + raa_frame.reserved2[0].x, 0.0, levels - 1.0);
    let axis = select(duv_dy, duv_dx, lx >= ly) * e;
    let inset = 0.5 * exp2(ceil(lod)) / atlas;
    let lo = r.xy + inset;
    let hi = max(r.xy + e - inset, lo);
    let p = r.xy + clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * e;
    var c = vec4<f32>(0.0);
    for (var i = 0u; i < 4u; i = i + 1u) {
        let t = (f32(i) - 1.5) * 0.25;
        let q = clamp(p + axis * t, lo, hi);
        c = c + 0.25 * textureSampleLevel(raa_texture_atlas, raa_texture_sampler, q, lod);
    }
    return select(vec4<f32>(0.0), c, r.z != 0.0);
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

// Shadow-map PCSS kernel shared by the csm Shadow provider and the hero area-light maps.

// 32 fixed Poisson-disk positions in the unit disk; the blocker search uses the first 16.
const RAA_CSM_POISSON: array<vec2<f32>, 32> = array<vec2<f32>, 32>(
    vec2<f32>(0.000000, 0.000000),
    vec2<f32>(0.245572, 0.025625),
    vec2<f32>(0.395874, -0.069721),
    vec2<f32>(0.233324, -0.288666),
    vec2<f32>(-0.072749, -0.222033),
    vec2<f32>(0.343213, 0.169892),
    vec2<f32>(0.282429, 0.319118),
    vec2<f32>(0.391256, -0.326816),
    vec2<f32>(0.214275, 0.519212),
    vec2<f32>(-0.041023, 0.441851),
    vec2<f32>(0.149747, 0.189052),
    vec2<f32>(0.528000, -0.535515),
    vec2<f32>(0.579855, -0.197497),
    vec2<f32>(0.590327, 0.103002),
    vec2<f32>(-0.066529, 0.613064),
    vec2<f32>(0.126337, -0.108697),
    vec2<f32>(0.067175, -0.364676),
    vec2<f32>(0.848151, -0.185183),
    vec2<f32>(-0.239579, -0.077709),
    vec2<f32>(0.081520, -0.668538),
    vec2<f32>(-0.213674, 0.357494),
    vec2<f32>(-0.193341, -0.509093),
    vec2<f32>(0.513411, 0.310819),
    vec2<f32>(-0.217035, 0.149110),
    vec2<f32>(-0.009268, 0.246234),
    vec2<f32>(0.295802, -0.527740),
    vec2<f32>(0.775168, 0.373917),
    vec2<f32>(0.104490, 0.636529),
    vec2<f32>(0.642104, 0.522812),
    vec2<f32>(-0.404827, 0.019434),
    vec2<f32>(0.841166, 0.019880),
    vec2<f32>(-0.418969, 0.395065)
);

// Interleaved gradient noise (Jimenez 2014) in [0, 2 pi): the per-pixel PCF rotation.
fn raa_csm_rotation(p: vec2<f32>) -> f32 {
    let g = fract(dot(p, vec2<f32>(0.06711056, 0.00583715)));
    return fract(52.9829189 * g) * 6.28318530718;
}

fn raa_csm_rotate(v: vec2<f32>, a: f32) -> vec2<f32> {
    let s = sin(a);
    let c = cos(a);
    return vec2<f32>(c * v.x - s * v.y, s * v.x + c * v.y);
}

// Handoff: card "raa-shader: area-light visibility tiers in local_lighting.wgsl (hero PCSS,
// cone, far fade)" calls `raa_shadow_blockers(layer: i32, uv: vec2<f32>, depth: f32,
// search_texels: f32, res: f32) -> vec2<f32>` from this file, together with `raa_shadow_pcf`
// and `raa_csm_rotation`, all defined here today. It averages the raw depths of the 16
// Poisson taps in front of `depth` into `.x` and counts them in `.y`, or returns (0, 0)
// without a tap. Sizes are shadow-map texels; `res` is the layer width (1 / raa_shadow.params.z).
fn raa_shadow_blockers(layer: i32, uv: vec2<f32>, depth: f32, search_texels: f32, res: f32) -> vec2<f32> {
    var blocker_sum = 0.0;
    var blocker_count = 0.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let offset = RAA_CSM_POISSON[i] * search_texels;
        let coord = vec2<i32>(
            clamp(uv * res + offset, vec2<f32>(0.0), vec2<f32>(res - 1.0))
        );
        let d = textureLoad(raa_shadow_map, coord, layer, 0);
        if (d < depth) {
            blocker_sum = blocker_sum + d;
            blocker_count = blocker_count + 1.0;
        }
    }
    if (blocker_count < 0.5) {
        return vec2<f32>(0.0, 0.0);
    }
    return vec2<f32>(blocker_sum / blocker_count, blocker_count);
}

// Probe taps of raa_shadow_pcf: the centre and eight outer Poisson positions, one per 45 degree
// sector (indices 0, 30, 28, 14, 31, 29, 21, 19, 11), as a bit mask over RAA_CSM_POISSON.
const RAA_PCF_PROBE_COUNT: f32 = 9.0;
const RAA_PCF_PROBE_MASK: u32 = (1u << 0u) | (1u << 30u) | (1u << 28u) | (1u << 14u) | (1u << 31u)
    | (1u << 29u) | (1u << 21u) | (1u << 19u) | (1u << 11u);

fn raa_shadow_pcf_tap(layer: i32, uv: vec2<f32>, depth: f32, radius_texels: f32, res: f32,
    rotation: f32, i: u32) -> f32 {
    let offset = raa_csm_rotate(RAA_CSM_POISSON[i], rotation) * radius_texels;
    let coord = clamp(uv + offset / res, vec2<f32>(0.0), vec2<f32>(1.0));
    return textureSampleCompareLevel(raa_shadow_map, raa_shadow_cmp, coord, layer, depth);
}

// Mean of the rotated 32-tap Poisson PCF comparisons of `uv` at `depth`, in [0, 1].
// The nine probe taps run first: when they all agree (the whole kernel lit, or the whole
// kernel in umbra) their common value is returned and the other 23 taps are skipped;
// otherwise the remaining taps complete the same 32-tap mean.
fn raa_shadow_pcf(layer: i32, uv: vec2<f32>, depth: f32, radius_texels: f32, res: f32, rotation: f32) -> f32 {
    var probe = 0.0;
    for (var i = 0u; i < 32u; i = i + 1u) {
        if ((RAA_PCF_PROBE_MASK >> i) & 1u) != 0u {
            probe = probe + raa_shadow_pcf_tap(layer, uv, depth, radius_texels, res, rotation, i);
        }
    }
    if (probe == 0.0 || probe == RAA_PCF_PROBE_COUNT) {
        return probe / RAA_PCF_PROBE_COUNT;
    }
    var pcf = probe;
    for (var i = 0u; i < 32u; i = i + 1u) {
        if ((RAA_PCF_PROBE_MASK >> i) & 1u) == 0u {
            pcf = pcf + raa_shadow_pcf_tap(layer, uv, depth, radius_texels, res, rotation, i);
        }
    }
    return pcf / 32.0;
}
// End of the shadow-map PCSS kernel section.
