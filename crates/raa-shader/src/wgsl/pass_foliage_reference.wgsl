// Unlit foliage reference pass. Writes the world normal and coverage, the standard-Z depth
// and the linear unlit albedo; the pipeline clears depth to 1 (Less) and every colour
// target to zero. Vertex evaluation follows the billboard bake pass.
struct RaaReferenceVaryings {
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

@vertex
fn vs_main(v0: RaaVertex0, v1: RaaVertex1, @location(2) instance_id: u32) -> RaaReferenceVaryings {
    let inst = raa_instance(instance_id);
    let x = raa_extras[instance_id];
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
    var out: RaaReferenceVaryings;
    out.clip = raa_frame.view_proj * world;
    out.world_position = world.xyz;
    out.local_position = mv.local_position;
    out.normal = normal_matrix * d.normal;
    out.tangent = vec4<f32>((model * vec4<f32>(v1.tangent.xyz, 0.0)).xyz, v1.tangent.w);
    out.uv = v1.uv;
    out.color = v1.color;
    out.material_index = inst.material_index;
    out.instance_index = instance_id;
    return out;
}

// Location 0: Rgba16Float world normal XYZ and coverage W. Location 1: Rgba32Float
// standard-Z depth X and zero YZW. Location 2: Rgba8Unorm linear unlit albedo RGB and
// coverage A. Coverage is the material opacity of the source, never a lit intensity.
struct RaaReferenceTargets {
    @location(0) normal_coverage: vec4<f32>,
    @location(1) depth: vec4<f32>,
    @location(2) albedo_coverage: vec4<f32>,
    @builtin(frag_depth) frag_depth: f32,
}

@fragment
fn fs_main(v: RaaReferenceVaryings, @builtin(front_facing) front_facing: bool) -> RaaReferenceTargets {
    let record = raa_materials[v.material_index];
    var n = normalize(v.normal);
    if (!front_facing && (record.header.z & RAA_FLAG_DOUBLE_SIDED) != 0u) {
        n = -n;
    }
    var input: MaterialInput;
    input.position = v.world_position;
    input.local_position = v.local_position;
    input.normal = n;
    input.tangent = v.tangent;
    input.uv = v.uv;
    input.color = v.color;
    input.view_dir = normalize(raa_frame.camera_pos.xyz - v.world_position);
    input.time = raa_frame.camera_pos.w;
    input.front_facing = front_facing;
    input.frag_coord = v.clip;
    input.instance_params = raa_extras[v.instance_index].params;
    input.record = record;
    let x = raa_extras[v.instance_index];
    raa_set_instance_tint(x);
    let surface = raa_apply_tint(material_main(input), x);
    let coverage = clamp(surface.opacity, 0.0, 1.0);
    if (coverage <= 0.0) {
        discard;
    }
    let depth = v.clip.z / v.clip.w;
    var out: RaaReferenceTargets;
    out.normal_coverage = vec4<f32>(normalize(surface.normal), coverage);
    out.depth = vec4<f32>(depth, 0.0, 0.0, 0.0);
    out.albedo_coverage = vec4<f32>(surface.base_color, coverage);
    out.frag_depth = depth;
    return out;
}
