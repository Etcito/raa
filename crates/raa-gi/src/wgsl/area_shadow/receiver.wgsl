// Compare positions on both tangent planes. A sloped receiver can have a large
// view-depth gradient, while even a shallow step between parallel faces can leak light.
fn raa_area_receiver_weight(
    delta: vec3<f32>, normal: vec3<f32>, tap_normal: vec3<f32>, depth: f32,
) -> f32 {
    let plane_distance = max(abs(dot(delta, normal)), abs(dot(delta, tap_normal)));
    let tolerance = 0.001 * depth + 0.001;
    return 1.0 - smoothstep(tolerance, 3.0 * tolerance, plane_distance);
}

// Unproject the pixel centre at a given positive linear depth using the lit
// frame's jittered inverse projection. clip.w is 1 for orthographic projection.
fn raa_area_receiver_position(pixel: vec2<i32>, depth: f32, frame: RaaTemporalFrame) -> vec3<f32> {
    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) * frame.viewport.zw;
    let ndc = uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    let clip_zw = frame.proj[3].zw - depth * frame.proj[2].zw;
    return (frame.inv_proj * vec4<f32>(ndc * clip_zw.y, clip_zw)).xyz;
}
