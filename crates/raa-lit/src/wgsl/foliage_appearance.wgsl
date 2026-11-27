// Unlit source-material albedo evaluation for native foliage baking (M7, m7-appearance).
// Appended after the material prelude, LIB and the source material's `material_main`.

struct FoliageSampleGpu {
    position: vec4<f32>,     // 0: source-root xyz, w zero
    normal: vec4<f32>,       // 16: source shading normal xyz, w zero
    uv: vec4<f32>,           // 32: barycentric source UV, zw zero
    barycentric: vec4<f32>,  // 48: source triangle weights, w zero
    ids: vec4<u32>,          // 64: source triangle, material, zero, zero
}

@group(1) @binding(0) var<storage, read> raa_foliage_samples: array<FoliageSampleGpu>;
@group(1) @binding(1) var<storage, read_write> raa_foliage_albedo: array<vec4<f32>>;

// One linear unlit albedo value per sample: the material output's base colour evaluated at
// the sample's source-root position, source normal and source UV. Lighting, GI, exposure and
// shadow terms are absent; the instance tint is neutral.
@compute @workgroup_size(64)
fn sample_albedo(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&raa_foliage_samples)) {
        return;
    }
    let sample = raa_foliage_samples[id.x];
    var tint: InstanceExtraGpu;
    tint.tint = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    tint.reserved = vec4<f32>(0.0);
    raa_set_instance_tint(tint);
    var input: MaterialInput;
    input.position = sample.position.xyz;
    input.local_position = sample.position.xyz;
    input.normal = sample.normal.xyz;
    input.tangent = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    input.uv = sample.uv.xy;
    input.color = vec4<f32>(1.0);
    input.view_dir = sample.normal.xyz;
    input.time = 0.0;
    input.front_facing = true;
    input.frag_coord = vec4<f32>(0.0);
    input.instance_params = array<vec4<f32>, 4>(
        vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    input.record = raa_materials[sample.ids.y];
    let surface = material_main(input);
    raa_foliage_albedo[id.x] = vec4<f32>(surface.base_color, 0.0);
}
