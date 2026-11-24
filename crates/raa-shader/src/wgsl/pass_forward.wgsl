// RAA forward pass (S11): opaque shading to the HDR target, MSAA.

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

// Flat model or UNLIT flag: the fragment is base + emissive.
fn raa_pass_unlit(record: SurfaceMaterialGpu) -> bool {
    return record.header.x == RAA_MODEL_FLAT || (record.header.z & RAA_FLAG_UNLIT) != 0u;
}

// Normal-incidence reflectance: dielectric from the ior (0.04 at ior 1.5), base colour for
// metals.
fn raa_pass_f0(s: MaterialOutput) -> vec3<f32> {
    let ior = clamp(s.ior, 1.0, 3.0);
    let r = (ior - 1.0) / (ior + 1.0);
    return mix(vec3<f32>(r * r), s.base_color, clamp(s.metallic, 0.0, 1.0));
}

fn raa_pass_shade_ctx(s: MaterialOutput, in: MaterialInput, ao: f32, shadow: f32) -> ShadeCtx {
    var c: ShadeCtx;
    c.material_flags = in.record.header.z;
    c.position = in.position;
    c.view_dir = in.view_dir;
    c.frag = in.frag_coord;
    c.n_dot_v = max(dot(s.normal, in.view_dir), 1e-4);
    c.ao = min(s.ao, ao);
    c.shadow = shadow;
    c.f0 = raa_pass_f0(s);
    return c;
}

// S11 final colour: emissive, sun, ambient/environment, indirect and the shared tiled local
// lights. Every slot and helper runs before the unlit select so the calls stay in uniform
// control flow. The Composer attaches LOCAL_LIGHTING_WGSL ahead of this pass.
@fragment
fn fs_main(v: RaaVaryings, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    let in = raa_pass_material_input(v, front_facing);
    let s = raa_pass_material(in, v.instance_index);
    let shadow = raa_slot_shadow(in.position, in.normal, in.frag_coord);
    var ao = 1.0;
    let capture = raa_feature_enabled(RAA_PLANAR_CAPTURE_FEATURE);
    if !capture {
        ao = raa_slot_ao(in.frag_coord);
    }
    var c = raa_pass_shade_ctx(s, in, ao, shadow);
    let unlit = raa_pass_unlit(in.record);
    var spherical_debug = vec4<f32>(0.0);
    if !capture && !unlit && raa_feature_enabled(RAA_SPHERICAL_GI_FEATURE)
        && !raa_feature_enabled(RAA_DIRECT_ONLY_FEATURE) {
        let spherical = raa_spherical_evaluate(c.position, s.normal);
        spherical_debug = spherical.debug;
        c.spherical_diffuse = spherical.diffuse;
    }
    let direct = raa_slot_direct(s, c, -raa_frame.sun_dir.xyz,
            raa_directional_radiance(raa_frame.sun_color.rgb) * shadow);
    let ambient = raa_slot_ambient(s, c);
    var indirect = vec3<f32>(0.0);
    if !capture {
        indirect = raa_slot_indirect(s, c);
        if c.spherical_diffuse.w > 0.0 {
            indirect *= 1.0 - c.spherical_diffuse.w;
        }
    }
    let local = raa_local_lighting(s, c);
    if spherical_debug.a > 0.0 { return spherical_debug; }
    if raa_feature_enabled(RAA_DIRECT_ONLY_FEATURE) {
        return vec4<f32>(select(s.emissive + direct + local, s.base_color + s.emissive, unlit), 1.0);
    }
    if raa_feature_enabled(RAA_INDIRECT_ONLY_FEATURE) {
        return vec4<f32>(select(ambient + indirect, vec3<f32>(0.0), unlit), 1.0);
    }
    let lit = s.emissive + direct + ambient + indirect + local;
    let color = select(lit, s.base_color + s.emissive, unlit);
    return vec4<f32>(color, 1.0);
}
