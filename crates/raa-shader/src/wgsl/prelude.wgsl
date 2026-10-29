// RAA prelude (S10 structs and bindings of groups 0-1, S11 material structs).

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
    reserved: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<uniform> raa_frame: FrameGpu;
@group(0) @binding(1) var<storage, read> raa_materials: array<SurfaceMaterialGpu>;
@group(0) @binding(2) var<storage, read> raa_extras: array<InstanceExtraGpu>;

@group(1) @binding(0) var raa_shadow_map: texture_depth_2d_array;
@group(1) @binding(1) var raa_shadow_cmp: sampler_comparison;
@group(1) @binding(2) var<uniform> raa_shadow: ShadowGpu;
@group(1) @binding(3) var raa_env_specular: texture_cube<f32>;
@group(1) @binding(4) var<uniform> raa_env_sh: EnvSh;
@group(1) @binding(5) var raa_brdf_lut: texture_2d<f32>;
@group(1) @binding(6) var raa_linear_clamp: sampler;
@group(1) @binding(7) var raa_ao_tex: texture_2d<f32>;
@group(1) @binding(8) var raa_contact_tex: texture_2d<f32>;
@group(1) @binding(9) var raa_opaque_color: texture_2d<f32>;
@group(1) @binding(10) var raa_opaque_depth: texture_2d<f32>;
@group(1) @binding(11) var<uniform> raa_env: EnvGpu;

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

// Vertex stream 0 (S1 Vertex, 24 B).
struct RaaVertex0 {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

// Instance stream (S1 Instance, 128 B).
struct RaaInstance {
    @location(2) model0: vec4<f32>,
    @location(3) model1: vec4<f32>,
    @location(4) model2: vec4<f32>,
    @location(5) model3: vec4<f32>,
    @location(6) normal0: vec4<f32>,
    @location(7) normal1: vec4<f32>,
    @location(8) normal2: vec4<f32>,
    @location(9) material_index: u32,
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
