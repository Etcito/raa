// RAA prepass (S11, S12): depth plus the normal, material and motion targets.

struct RaaVaryings {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) local_position: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) uv: vec2<f32>,
    @location(5) color: vec4<f32>,
    @location(6) @interpolate(flat, either) material_index: u32,
    @location(7) @interpolate(flat, either) instance_index: u32,
    // S12 motion: previous world position (same local geometry, previous rigid model).
    @location(8) motion_previous_world: vec3<f32>,
    // 1 when the instance has submitted history and no unsupported deformation.
    @location(9) @interpolate(flat, either) motion_valid: u32,
}

// material_vertex, then the deform slot (local space), then the model and normal matrices.
// MaterialInput.local_position is the material_vertex output (before deform).
@vertex
fn vs_main(
    v0: RaaVertex0,
    v1: RaaVertex1,
    @location(2) raa_instance_id: u32,
) -> RaaVaryings {
    let inst = raa_instance(raa_instance_id);
    let x = raa_extras[raa_instance_id];
    let time = raa_frame.camera_pos.w;
    var vin: VertexMaterialInput;
    vin.local_position = v0.position;
    vin.normal = v0.normal;
    vin.uv = v1.uv;
    vin.color = v1.color;
    vin.time = time;
    vin.instance_params = x.params;
    vin.record = raa_materials[inst.material_index];
    let mv = material_vertex(vin);
    let d = raa_slot_deform(mv.local_position, mv.normal, x, time);
    let model = mat4x4<f32>(inst.model0, inst.model1, inst.model2, inst.model3);
    let normal_matrix = mat3x3<f32>(inst.normal0.xyz, inst.normal1.xyz, inst.normal2.xyz);
    let world = model * vec4<f32>(d.local_position, 1.0);
    var out: RaaVaryings;
    out.clip = raa_frame.view_proj * world;
    out.world_position = world.xyz;
    out.local_position = mv.local_position;
    out.normal = normal_matrix * d.normal;
    out.tangent = vec4<f32>((model * vec4<f32>(v1.tangent.xyz, 0.0)).xyz, v1.tangent.w);
    out.uv = v1.uv;
    out.color = v1.color;
    out.material_index = inst.material_index;
    out.instance_index = raa_instance_id;
    // S12 motion only carries rigid previous models: a vertex function deforms geometry in
    // a way this record cannot reconstruct, and a slot without submitted history is new,
    // re-used or posed, so both write zero in the fragment stage.
    let previous = raa_previous_instances[raa_instance_id];
    let has_vertex_fn = (raa_materials[inst.material_index].header.z & RAA_FLAG_VERTEX_FN) != 0u;
    let motion_valid = previous.previous_valid != 0u && !has_vertex_fn;
    let previous_world = raa_previous_model(previous) * vec4<f32>(d.local_position, 1.0);
    out.motion_previous_world = select(world.xyz, previous_world.xyz, motion_valid);
    out.motion_valid = select(0u, 1u, motion_valid);
    return out;
}

// MaterialInput of a fragment; the geometric normal is flipped on back faces of
// DOUBLE_SIDED materials.
fn raa_pass_material_input(v: RaaVaryings, front_facing: bool) -> MaterialInput {
    let record = raa_materials[v.material_index];
    var n = normalize(v.normal);
    if (!front_facing && (record.header.z & RAA_FLAG_DOUBLE_SIDED) != 0u) {
        n = -n;
    }
    var in: MaterialInput;
    in.position = v.world_position;
    in.local_position = v.local_position;
    in.normal = n;
    in.tangent = v.tangent;
    in.uv = v.uv;
    in.color = v.color;
    in.view_dir = normalize(raa_frame.camera_pos.xyz - v.world_position);
    in.time = raa_frame.camera_pos.w;
    in.front_facing = front_facing;
    in.frag_coord = v.clip;
    in.instance_params = raa_extras[v.instance_index].params;
    in.record = record;
    return in;
}

// material_main followed by the instance tint (S10). mat_instance_tint and
// mat_instance_tint_set are set first so material graph code can read them (S23).
fn raa_pass_material(in: MaterialInput, instance: u32) -> MaterialOutput {
    let x = raa_extras[instance];
    raa_set_instance_tint(x);
    return raa_apply_tint(material_main(in), x);
}

// Normal-incidence reflectance: dielectric from the ior (0.04 at ior 1.5), base colour for
// metals.
fn raa_pass_f0(s: MaterialOutput) -> vec3<f32> {
    let ior = clamp(s.ior, 1.0, 3.0);
    let r = (ior - 1.0) / (ior + 1.0);
    return mix(vec3<f32>(r * r), s.base_color, clamp(s.metallic, 0.0, 1.0));
}

// True for a finite f32 (naga's WGSL front-end has no isNan/isInf builtin; `v - v` is NaN
// for both NaN and infinity, and every comparison against NaN is false).
fn raa_pass_finite(v: f32) -> bool {
    return (v - v) == 0.0;
}

// S12 motion: current unjittered NDC minus previous unjittered NDC as a UV displacement
// (positive X right, positive Y down). Both jitters are removed: current by using the
// unjittered view-projection, previous by subtracting camera_jitter.zw from its NDC.
// Invalid history or projection, new/re-used instances and unsupported deformation zero.
fn raa_pass_motion(v: RaaVaryings) -> vec2<f32> {
    if (v.motion_valid == 0u || raa_frame.temporal.z == 0u) {
        return vec2<f32>(0.0);
    }
    let current_clip = raa_frame.unjittered_view_proj * vec4<f32>(v.world_position, 1.0);
    let previous_clip = raa_frame.previous_view_proj * vec4<f32>(v.motion_previous_world, 1.0);
    if (current_clip.w <= 0.0 || previous_clip.w <= 0.0) {
        return vec2<f32>(0.0);
    }
    let current_ndc = current_clip.xy / current_clip.w;
    let previous_ndc = previous_clip.xy / previous_clip.w - raa_frame.camera_jitter.zw;
    if (!raa_pass_finite(current_ndc.x) || !raa_pass_finite(current_ndc.y)
        || !raa_pass_finite(previous_ndc.x) || !raa_pass_finite(previous_ndc.y)) {
        return vec2<f32>(0.0);
    }
    return vec2<f32>(
        (current_ndc.x - previous_ndc.x) * 0.5,
        (previous_ndc.y - current_ndc.y) * 0.5,
    );
}

struct RaaPrepassTargets {
    // S12 `normal` (Rgba16Float): view-space shading normal xyz, roughness.
    @location(0) normal: vec4<f32>,
    // S12 `material` (Rgba8Unorm): F0 rgb sRGB-encoded, metallic.
    @location(1) material: vec4<f32>,
    // S12 `motion` (Rg16Float): signed UV displacement, positive X right and Y down.
    @location(2) motion: vec2<f32>,
}

// Piecewise sRGB OETF of a value in [0, 1].
fn raa_pass_encode_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let lo = x * 12.92;
    let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}

@fragment
fn fs_main(v: RaaVaryings, @builtin(front_facing) front_facing: bool) -> RaaPrepassTargets {
    let in = raa_pass_material_input(v, front_facing);
    let s = raa_pass_material(in, v.instance_index);
    let n = normalize((raa_frame.view * vec4<f32>(s.normal, 0.0)).xyz);
    var out: RaaPrepassTargets;
    out.normal = vec4<f32>(n, clamp(s.roughness, 0.0, 1.0));
    out.material = vec4<f32>(raa_pass_encode_srgb(raa_pass_f0(s)), clamp(s.metallic, 0.0, 1.0));
    out.motion = raa_pass_motion(v);
    return out;
}
