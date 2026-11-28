// RAA forward pass (S11): opaque shading to the HDR target, MSAA.

struct RaaVaryings {
    @invariant @builtin(position) clip: vec4<f32>,
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

@fragment
fn fs_main(v: RaaVaryings, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    let in = raa_pass_material_input(v, front_facing);
    let s = raa_pass_material(in, v.instance_index);
    return raa_opaque_lighting(in, s);
}
