// M7 relightable foliage material (contracts m7-material-wgsl and m7-material-data).
// `Composer::compose_foliage` assembles the named parts below: the group-2 declarations and
// the parallax helpers for modes 0, 1, 4 and 5 (with foliage_parallax.wgsl), the offset
// helpers for modes 2 and 3 (foliage_offset.wgsl declares its own group 2), then the
// material part and the prepass or forward entry part. `RAA_FOLIAGE_MODE` is composed
// ahead of them. No other pass reads this file or the M7 material-data envelope.

// raa foliage part: group2
// Group 2 of m7-foliage-resources with the names foliage_parallax.wgsl reads.
struct RaaFoliageParams {
    min: vec3<f32>,
    resolution: u32,
    max: vec3<f32>,
    mode: u32,
    translucency: f32,
    generation: u32,
    cutoff: f32,
    view_count: u32,
    texture_resolution: u32,
    zero0: u32,
    zero1: u32,
    zero2: u32,
}

struct RaaFoliageView {
    clip: mat4x4<f32>,
    near_far: vec4<f32>,
}

struct RaaFoliageViewTable {
    views: array<RaaFoliageView, 128>,
}

@group(2) @binding(0) var<uniform> raa_foliage_params: RaaFoliageParams;
@group(2) @binding(1) var raa_foliage_volume: texture_3d<f32>;
@group(2) @binding(2) var raa_foliage_sampler: sampler;
@group(2) @binding(3) var raa_foliage_albedo: texture_2d_array<f32>;
@group(2) @binding(4) var raa_foliage_normal_depth: texture_2d_array<f32>;
@group(2) @binding(5) var<uniform> raa_foliage_views: RaaFoliageViewTable;

// raa foliage part: parallax sample
fn raa_foliage_settings() -> RaaFoliageSettings {
    let p = raa_foliage_params;
    return RaaFoliageSettings(p.min, p.max, p.translucency, p.cutoff);
}

fn raa_foliage_impostor(position: vec3<f32>, direction: vec3<f32>) -> FoliageHit {
    return foliage_parallax_hit(position, direction, RAA_FOLIAGE_MODE);
}

// raa foliage part: offset sample
fn raa_foliage_settings() -> RaaFoliageSettings {
    let p = raa_foliage;
    return RaaFoliageSettings(p.min, p.max, p.translucency, p.cutoff);
}

fn raa_foliage_impostor(position: vec3<f32>, direction: vec3<f32>) -> FoliageHit {
    return foliage_offset_hit(position, direction, RAA_FOLIAGE_MODE);
}

// raa foliage part: material
// Relative words of the M7 material-data envelope (shared section 97).
const RAA_FOLIAGE_DATA_MAGIC: u32 = 0x4d37464cu;
const RAA_FOLIAGE_DATA_VERSION: u32 = 1u;
const RAA_FOLIAGE_MASK_BASE_COLOR: u32 = 1u;
const RAA_FOLIAGE_MASK_NORMAL: u32 = 2u;
const RAA_FOLIAGE_MASK_AO: u32 = 4u;
const RAA_FOLIAGE_MASK_OPACITY: u32 = 8u;
const RAA_FOLIAGE_MASK_ALL: u32 = 15u;
// First distilled mode rank (stochastic 4, differentiable 5).
const RAA_FOLIAGE_MODE_DISTILLED: u32 = 4u;
// Draw-class bits a foliage fragment never reports: translucency stays opaque.
const RAA_FOLIAGE_TRANSLUCENT_CLASSES: u32 = 3u;

struct RaaFoliageSettings {
    min: vec3<f32>,
    max: vec3<f32>,
    translucency: f32,
    cutoff: f32,
}

// The authored-channel mask of header words 0..2; a wrong magic or version, or a reserved
// bit, reads as 0 (every channel baked), the defaults native preparation also refuses.
fn raa_foliage_header_mask(magic: u32, version: u32, mask: u32) -> u32 {
    if (magic != RAA_FOLIAGE_DATA_MAGIC || version != RAA_FOLIAGE_DATA_VERSION) {
        return 0u;
    }
    return select(0u, mask, (mask & ~RAA_FOLIAGE_MASK_ALL) == 0u);
}

fn raa_foliage_mask(record: SurfaceMaterialGpu) -> u32 {
    let base = raa_material_data_base(record);
    return raa_foliage_header_mask(raa_material_word(base), raa_material_word(base + 1u), raa_material_word(base + 2u));
}

// Native volume texel at a source-root position: RGB unlit albedo, A canopy occlusion.
fn raa_foliage_volume_sample(local_position: vec3<f32>, settings: RaaFoliageSettings) -> vec4<f32> {
    let coord = clamp((local_position - settings.min) / (settings.max - settings.min), vec3<f32>(0.0), vec3<f32>(1.0));
    return textureSampleLevel(raa_foliage_volume, raa_foliage_sampler, coord, 0.0);
}

// Backlit translucency of an opaque fragment: light arriving from behind the surface.
fn raa_foliage_backlight(albedo: vec3<f32>, translucency: f32, normal: vec3<f32>, to_light: vec3<f32>, radiance: vec3<f32>, visibility: f32) -> vec3<f32> {
    return albedo * translucency * max(dot(-normal, to_light), 0.0) * radiance * visibility;
}

struct RaaFoliageSurface {
    albedo: vec3<f32>,
    normal: vec3<f32>,
    ao: f32,
    coverage: f32,
}

// Mask selection of section 97: a set bit takes the authored LOD1 result (albedo modulates
// the baked albedo, the authored normal already perturbs the source normal, authored AO
// multiplies the canopy factor, opacity modulates impostor coverage); a clear bit keeps the
// baked value. Distilled triangles stay fully opaque.
fn raa_foliage_surface(hit: FoliageHit, s: MaterialOutput, mask: u32, occlusion: f32, distilled: bool) -> RaaFoliageSurface {
    let canopy = 1.0 - clamp(occlusion, 0.0, 1.0);
    var out: RaaFoliageSurface;
    out.albedo = select(hit.albedo, hit.albedo * s.base_color, (mask & RAA_FOLIAGE_MASK_BASE_COLOR) != 0u);
    out.normal = select(hit.normal, s.normal, (mask & RAA_FOLIAGE_MASK_NORMAL) != 0u);
    out.ao = select(canopy, canopy * s.ao, (mask & RAA_FOLIAGE_MASK_AO) != 0u);
    let coverage = select(hit.coverage, hit.coverage * s.opacity, (mask & RAA_FOLIAGE_MASK_OPACITY) != 0u);
    out.coverage = select(coverage, 1.0, distilled);
    return out;
}

// An impostor miss, or authored opacity modulated below the cutoff, is cut.
fn raa_foliage_cut(coverage: f32, mask: u32, cutoff: f32) -> bool {
    return coverage <= 0.0 || ((mask & RAA_FOLIAGE_MASK_OPACITY) != 0u && coverage < cutoff);
}

struct RaaFoliageVaryings {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) local_position: vec3<f32>,
    @location(2) @interpolate(flat, either) normal: vec3<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) uv: vec2<f32>,
    @location(5) @interpolate(flat, either) material_index: u32,
    @location(6) @interpolate(flat, either) instance_index: u32,
    @location(7) motion_previous_world: vec3<f32>,
    @location(8) @interpolate(flat, either) motion_valid: u32,
}

// The forward vertex path without the colour stream; the normal is triangle-constant.
@vertex
fn vs_main(v0: RaaVertex0, v1: RaaVertex1, @location(2) raa_instance_id: u32) -> RaaFoliageVaryings {
    let inst = raa_instance(raa_instance_id);
    let x = raa_extras[raa_instance_id];
    let time = raa_frame.camera_pos.w;
    var vin: VertexMaterialInput;
    vin.local_position = v0.position;
    vin.normal = v0.normal;
    vin.uv = v1.uv;
    vin.color = vec4<f32>(1.0);
    vin.time = time;
    vin.instance_params = x.params;
    vin.record = raa_materials[inst.material_index];
    let mv = material_vertex(vin);
    let d = raa_slot_deform(mv.local_position, mv.normal, x, time);
    let model = mat4x4<f32>(inst.model0, inst.model1, inst.model2, inst.model3);
    let normal_matrix = mat3x3<f32>(inst.normal0.xyz, inst.normal1.xyz, inst.normal2.xyz);
    let world = model * vec4<f32>(d.local_position, 1.0);
    let previous = raa_previous_instances[raa_instance_id];
    let has_vertex_fn = (vin.record.header.z & RAA_FLAG_VERTEX_FN) != 0u;
    let motion_valid = previous.previous_valid != 0u && !has_vertex_fn;
    let previous_world = raa_previous_model(previous) * vec4<f32>(d.local_position, 1.0);
    var out: RaaFoliageVaryings;
    out.clip = raa_frame.view_proj * world;
    out.world_position = world.xyz;
    out.local_position = mv.local_position;
    out.normal = normal_matrix * d.normal;
    out.tangent = vec4<f32>((model * vec4<f32>(v1.tangent.xyz, 0.0)).xyz, v1.tangent.w);
    out.uv = v1.uv;
    out.material_index = inst.material_index;
    out.instance_index = raa_instance_id;
    out.motion_previous_world = select(world.xyz, previous_world.xyz, motion_valid);
    out.motion_valid = select(0u, 1u, motion_valid);
    return out;
}

struct RaaFoliageFragment {
    in: MaterialInput,
    s: MaterialOutput,
    settings: RaaFoliageSettings,
    depth: f32,
    previous_world: vec3<f32>,
}

// The shared fragment of both passes: the distilled volume sample on the triangle, or the
// impostor hit along the source-root view ray; then the LOD1 material and the mask. Cut
// fragments are discarded, so coverage and depth agree in prepass and forward.
fn raa_foliage_fragment(v: RaaFoliageVaryings, front_facing: bool) -> RaaFoliageFragment {
    let inst = raa_instance(v.instance_index);
    let model = mat4x4<f32>(inst.model0, inst.model1, inst.model2, inst.model3);
    let normal_matrix = mat3x3<f32>(inst.normal0.xyz, inst.normal1.xyz, inst.normal2.xyz);
    let record = raa_materials[v.material_index];
    let settings = raa_foliage_settings();
    let distilled = RAA_FOLIAGE_MODE >= RAA_FOLIAGE_MODE_DISTILLED;
    // transpose(normal matrix) is the inverse model 3x3: a world ray in source-root space.
    let ray = normalize(transpose(normal_matrix) * (v.world_position - raa_frame.camera_pos.xyz));
    var hit = raa_foliage_impostor(v.local_position, ray);
    if (distilled) {
        hit = FoliageHit(v.local_position, vec3<f32>(0.0), vec3<f32>(0.0), 1.0, v.clip.z);
    }
    let volume = raa_foliage_volume_sample(hit.position, settings);
    let world = select((model * vec4<f32>(hit.position, 1.0)).xyz, v.world_position, distilled);
    let normal = normalize(select(normal_matrix * hit.normal, v.normal, distilled));
    let clip = raa_frame.view_proj * vec4<f32>(world, 1.0);
    hit.albedo = select(hit.albedo, volume.rgb, distilled);
    hit.normal = normal;
    var in: MaterialInput;
    in.position = world;
    in.local_position = hit.position;
    in.normal = normal;
    in.tangent = v.tangent;
    in.uv = v.uv;
    in.color = vec4<f32>(1.0);
    in.view_dir = normalize(raa_frame.camera_pos.xyz - world);
    in.time = raa_frame.camera_pos.w;
    in.front_facing = front_facing;
    in.frag_coord = vec4<f32>(v.clip.xy, select(clip.z / clip.w, v.clip.z, distilled), v.clip.w);
    in.instance_params = raa_extras[v.instance_index].params;
    in.record = record;
    let x = raa_extras[v.instance_index];
    raa_set_instance_tint(x);
    var s = raa_apply_tint(material_main(in), x);
    let mask = raa_foliage_mask(record);
    let surface = raa_foliage_surface(hit, s, mask, volume.a, distilled);
    if (raa_foliage_cut(surface.coverage, mask, settings.cutoff)) {
        discard;
    }
    s.base_color = surface.albedo;
    s.normal = surface.normal;
    s.ao = surface.ao;
    s.opacity = 1.0;
    s.transmission = 0.0;
    let previous = world + (v.motion_previous_world - v.world_position);
    return RaaFoliageFragment(in, s, settings, in.frag_coord.z, previous);
}

fn raa_foliage_f0(s: MaterialOutput) -> vec3<f32> {
    let ior = clamp(s.ior, 1.0, 3.0);
    let r = (ior - 1.0) / (ior + 1.0);
    return mix(vec3<f32>(r * r), s.base_color, clamp(s.metallic, 0.0, 1.0));
}

// raa foliage part: prepass
fn raa_foliage_finite(v: f32) -> bool {
    return (v - v) == 0.0;
}

// S12 motion of the resolved surface point (rigid instance history, as the prepass).
fn raa_foliage_motion(world: vec3<f32>, previous_world: vec3<f32>, valid: u32) -> vec4<f32> {
    let current_clip = raa_frame.unjittered_view_proj * vec4<f32>(world, 1.0);
    let previous_clip = raa_frame.previous_view_proj * vec4<f32>(previous_world, 1.0);
    if (raa_frame.temporal.z == 0u || current_clip.w <= 0.0 || previous_clip.w <= 0.0) {
        return vec4<f32>(0.0, 0.0, 1.0, 0.0);
    }
    let current_ndc = current_clip.xy / current_clip.w;
    let previous_ndc = previous_clip.xy / previous_clip.w - raa_frame.camera_jitter.zw;
    if (!raa_foliage_finite(current_ndc.x) || !raa_foliage_finite(current_ndc.y)
        || !raa_foliage_finite(previous_ndc.x) || !raa_foliage_finite(previous_ndc.y)) {
        return vec4<f32>(0.0, 0.0, 1.0, 0.0);
    }
    let reject = select(1.0, 0.0, valid != 0u);
    return vec4<f32>((current_ndc.x - previous_ndc.x) * 0.5, (previous_ndc.y - current_ndc.y) * 0.5, reject, 0.0);
}

fn raa_foliage_encode_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let lo = x * 12.92;
    let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}

struct RaaFoliagePrepassTargets {
    @location(0) normal: vec4<f32>,
    @location(1) material: vec4<f32>,
    @location(2) motion: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(v: RaaFoliageVaryings, @builtin(front_facing) front_facing: bool) -> RaaFoliagePrepassTargets {
    let f = raa_foliage_fragment(v, front_facing);
    let n = normalize((raa_frame.view * vec4<f32>(f.s.normal, 0.0)).xyz);
    var out: RaaFoliagePrepassTargets;
    out.normal = vec4<f32>(n, clamp(f.s.roughness, 0.0, 1.0));
    out.material = vec4<f32>(raa_foliage_encode_srgb(raa_foliage_f0(f.s)), clamp(f.s.metallic, 0.0, 1.0));
    out.motion = raa_foliage_motion(f.in.position, f.previous_world, v.motion_valid);
    out.depth = f.depth;
    return out;
}

// raa foliage part: forward
struct RaaFoliageForwardTargets {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

// Emissive, sun, ambient/environment (the only AO consumer, once), indirect, the shared
// tiled local lights and the backlit translucency; the alpha stays 1 (opaque draw class).
@fragment
fn fs_main(v: RaaFoliageVaryings, @builtin(front_facing) front_facing: bool) -> RaaFoliageForwardTargets {
    let f = raa_foliage_fragment(v, front_facing);
    let s = f.s;
    let shadow = raa_slot_shadow(f.in.position, s.normal, f.in.frag_coord);
    let ao = raa_slot_ao(f.in.frag_coord);
    var c: ShadeCtx;
    c.material_flags = f.in.record.header.z & ~RAA_FOLIAGE_TRANSLUCENT_CLASSES;
    c.position = f.in.position;
    c.view_dir = f.in.view_dir;
    c.frag = f.in.frag_coord;
    c.n_dot_v = max(dot(s.normal, f.in.view_dir), 1e-4);
    c.ao = min(s.ao, ao);
    c.shadow = shadow;
    c.f0 = raa_foliage_f0(s);
    let to_light = -raa_frame.sun_dir.xyz;
    let radiance = raa_directional_radiance(raa_frame.sun_color.rgb);
    let lit = s.emissive
        + raa_slot_direct(s, c, to_light, radiance * shadow)
        + raa_slot_ambient(s, c)
        + raa_slot_indirect(s, c)
        + raa_local_lighting(s, c)
        + raa_foliage_backlight(s.base_color, clamp(f.settings.translucency, 0.0, 1.0), s.normal, to_light, radiance, shadow);
    let unlit = f.in.record.header.x == RAA_MODEL_FLAT || (f.in.record.header.z & RAA_FLAG_UNLIT) != 0u;
    var out: RaaFoliageForwardTargets;
    out.color = vec4<f32>(select(lit, s.base_color + s.emissive, unlit), 1.0);
    out.depth = f.depth;
    return out;
}
