// raa-render forward pass. raa_material::WGSL_SHADING (MaterialGpu, FrameUniform,
// raa_shade, raa_encode_srgb) is prepended to this file by FORWARD_WGSL.

@group(0) @binding(0) var<uniform> frame: FrameUniform;
@group(0) @binding(1) var<storage, read> materials: array<MaterialGpu>;

// S1 vertex (locations 0-1) and instance (locations 2-9) attributes.
struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) model_0: vec4<f32>,
    @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>,
    @location(5) model_3: vec4<f32>,
    @location(6) normal_0: vec4<f32>,
    @location(7) normal_1: vec4<f32>,
    @location(8) normal_2: vec4<f32>,
    @location(9) material_index: u32,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) @interpolate(flat, either) material_index: u32,
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    let normal_matrix = mat3x3<f32>(in.normal_0.xyz, in.normal_1.xyz, in.normal_2.xyz);
    let world = model * vec4<f32>(in.position, 1.0);
    var out: VsOut;
    out.clip = frame.view_proj * world;
    out.world_pos = world.xyz;
    out.world_normal = normal_matrix * in.normal;
    out.material_index = in.material_index;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let v = normalize(frame.camera_pos.xyz - in.world_pos);
    var c = raa_shade(materials[in.material_index], n, v, frame);
    if (frame.flags.x == 1u) {
        c = raa_encode_srgb(c);
    }
    return vec4<f32>(c, 1.0);
}
