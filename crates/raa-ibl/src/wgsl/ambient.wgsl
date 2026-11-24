// Ambient slot of provider `ibl` (follows IBL_LIB_WGSL in the provider source).

// Multi-bounce AO (Jimenez et al. 2016, cubic fit over the albedo).
fn raa_ibl_multi_bounce_ao(ao: f32, albedo: vec3<f32>) -> vec3<f32> {
    let a = 2.0404 * albedo - vec3<f32>(0.3324);
    let b = -4.7951 * albedo + vec3<f32>(0.6417);
    let c = 2.7552 * albedo + vec3<f32>(0.6903);
    return max(vec3<f32>(ao), ((ao * a + b) * ao + c) * ao);
}

// Dominant direction of the GGX specular lobe (Lagarde and de Rousiers 2014): the
// reflection vector bent towards the normal as the roughness grows.
fn raa_ibl_dominant_dir(n: vec3<f32>, r: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let f = (1.0 - a) * (sqrt(1.0 - a) + a);
    return normalize(mix(n, r, f));
}

const RAA_IBL_PI: f32 = 3.14159265;

// The GI screen irradiance E at the fragment (`raa_gi_screen`, binding 14, read with the
// pixel clamped to its size) x the GI intensity; the 1x1 dummy is black while GI is inactive.
fn raa_ibl_gi_irradiance(frag: vec4<f32>) -> vec3<f32> {
    if raa_feature_enabled(RAA_PLANAR_CAPTURE_FEATURE) { return vec3<f32>(0.0); }
    let size = vec2<i32>(textureDimensions(raa_gi_screen));
    let pixel = clamp(vec2<i32>(frag.xy), vec2<i32>(0), size - vec2<i32>(1));
    return textureLoad(raa_gi_screen, pixel, 0).rgb * raa_gi_volume_params.intensity;
}

// Without an environment (info.w < 0.5): the S1 PbrLite ambient term, occluded once by
// `c.ao`. Otherwise SH diffuse with multi-bounce AO plus prefiltered specular with the
// split-sum DFG, Fdez-Aguera multiscatter energy compensation and specular occlusion. The
// incoming specular is mix(sky x info.x, GI irradiance / pi, EnvSh.gi.x) (S45; solstice@7e465b5
// crates/solstice-render/shaders/lighting/deferred.wgsl:189), so an active GI frame stops
// lighting interiors with the unshadowed sky.
fn raa_slot_ambient(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    if raa_env_sh.info.w < 0.5 {
        return raa_frame.ambient.rgb * s.base_color * (1.0 - 0.5 * s.metallic) * c.ao;
    }
    let n = s.normal;
    let v = c.view_dir;
    let roughness = clamp(s.roughness, 0.0, 1.0);
    let metallic = clamp(s.metallic, 0.0, 1.0);
    let f0 = c.f0;
    let dfg = raa_ibl_dfg(c.n_dot_v, roughness);
    let energy = vec3<f32>(1.0) + f0 * (1.0 / max(dfg.x + dfg.y, 1e-4) - 1.0);
    let diffuse = raa_ibl_irradiance(n) * s.base_color * (1.0 - metallic) * energy
        * raa_ibl_multi_bounce_ao(c.ao, s.base_color) * raa_env_sh.info.y;
    let r = raa_ibl_dominant_dir(n, reflect(-v, n), roughness);
    var w_gi = 0.0;
    if !raa_feature_enabled(RAA_PLANAR_CAPTURE_FEATURE) {
        w_gi = clamp(raa_env_sh.gi.x, 0.0, 1.0);
    }
    let incoming = mix(raa_ibl_specular(r, roughness) * raa_env_sh.info.x,
        raa_ibl_gi_irradiance(c.frag) / RAA_IBL_PI, w_gi);
    let specular = incoming * (f0 * dfg.x + vec3<f32>(dfg.y)) * energy
        * raa_specular_occlusion(c.n_dot_v, c.ao, roughness);
    return diffuse + specular;
}
