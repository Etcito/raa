// IBL_LIB_WGSL: image-based lighting helpers over the S10 group-1 environment
// (raa_env_specular, raa_env_sh, raa_brdf_lut, raa_linear_clamp). They read only prelude
// globals, so other providers (raa-brdf) reuse them; private helpers are prefixed raa_ibl_.

// Diffuse radiance multiplier at the unit normal n: the 9 SH coefficients of raa_env_sh
// (cosine-convolved and divided by pi, real SH order 2 in the basis of src/sh.rs), so a
// constant-radiance environment c gives c.
fn raa_ibl_irradiance(n: vec3<f32>) -> vec3<f32> {
    let sh = raa_env_sh.sh;
    var e = sh[0].rgb * 0.282095;
    e = e + sh[1].rgb * (0.488603 * n.y);
    e = e + sh[2].rgb * (0.488603 * n.z);
    e = e + sh[3].rgb * (0.488603 * n.x);
    e = e + sh[4].rgb * (1.092548 * n.x * n.y);
    e = e + sh[5].rgb * (1.092548 * n.y * n.z);
    e = e + sh[6].rgb * (0.315392 * (3.0 * n.z * n.z - 1.0));
    e = e + sh[7].rgb * (1.092548 * n.x * n.z);
    e = e + sh[8].rgb * (0.546274 * (n.x * n.x - n.y * n.y));
    return max(e, vec3<f32>(0.0));
}

// Prefiltered specular radiance along r: GGX-filtered cube mip roughness x max mip.
fn raa_ibl_specular(r: vec3<f32>, roughness: f32) -> vec3<f32> {
    let lod = clamp(roughness, 0.0, 1.0) * raa_env_sh.info.z;
    return textureSampleLevel(raa_env_specular, raa_linear_clamp, r, lod).rgb;
}

// Split-sum DFG terms (A = scale, B = bias of F0) from the BRDF LUT at (n_dot_v, roughness).
fn raa_ibl_dfg(n_dot_v: f32, roughness: f32) -> vec2<f32> {
    let uv = vec2<f32>(clamp(n_dot_v, 0.0, 1.0), clamp(roughness, 0.0, 1.0));
    return textureSampleLevel(raa_brdf_lut, raa_linear_clamp, uv, 0.0).rg;
}

// Specular occlusion from ambient occlusion (Lagarde and de Rousiers 2014).
fn raa_specular_occlusion(n_dot_v: f32, ao: f32, roughness: f32) -> f32 {
    return saturate(pow(max(n_dot_v + ao, 0.0), exp2(-16.0 * roughness - 1.0)) - 1.0 + ao);
}
