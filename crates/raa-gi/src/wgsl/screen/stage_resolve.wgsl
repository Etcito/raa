@group(0) @binding(0) var irradiance: texture_2d<f32>;
@group(0) @binding(1) var occlusion: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn raa_gi_stage_resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id.xy >= textureDimensions(output))) { return; }
    let pixel = vec2<i32>(id.xy);
    let occlusion_pixel = vec2<i32>(min(id.xy, textureDimensions(occlusion) - vec2<u32>(1)));
    textureStore(output, pixel, vec4<f32>(textureLoad(irradiance, pixel, 0).rgb, textureLoad(occlusion, occlusion_pixel, 0).a));
}
