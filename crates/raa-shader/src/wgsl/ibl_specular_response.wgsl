// Dominant direction of the GGX specular lobe (Lagarde and de Rousiers 2014): the
// reflection vector bent towards the normal as the roughness grows.
fn raa_ibl_dominant_dir(n: vec3<f32>, r: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let f = (1.0 - a) * (sqrt(1.0 - a) + a);
    return normalize(mix(n, r, f));
}

// Specular occlusion from ambient occlusion (Lagarde and de Rousiers 2014).
fn raa_specular_occlusion(n_dot_v: f32, ao: f32, roughness: f32) -> f32 {
    return clamp(pow(max(n_dot_v + ao, 0.0), exp2(-16.0 * roughness - 1.0)) - 1.0 + ao, 0.0, 1.0);
}

fn raa_ibl_specular_response(
    f0: vec3<f32>, dfg: vec2<f32>, n_dot_v: f32, ao: f32, roughness: f32,
) -> vec3<f32> {
    let energy = vec3<f32>(1.0) + f0 * (1.0 / max(dfg.x + dfg.y, 1e-4) - 1.0);
    return (f0 * dfg.x + vec3<f32>(dfg.y)) * energy * raa_specular_occlusion(n_dot_v, ao, roughness);
}
