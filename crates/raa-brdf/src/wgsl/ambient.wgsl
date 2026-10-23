// Ambient slot of provider `layered` (raa-brdf), appended after raa_ibl::IBL_LIB_WGSL in the
// provider source. Its helpers are prefixed raa_layered_ambient_ so that this source and the
// layered Direct source (wgsl/direct.wgsl) can both win in one module; the sheen fit is
// therefore a copy of direct.wgsl's (cpu.rs sheen_albedo_fit, the same table, checked by the
// unit tests of src/shaders.rs).

// cpu.rs MIN_SHEEN_ROUGHNESS.
const RAA_LAYERED_AMBIENT_MIN_SHEEN_ROUGHNESS: f32 = 0.07;
// cpu.rs MIN_N_DOT_V.
const RAA_LAYERED_AMBIENT_MIN_N_DOT_V: f32 = 1e-4;
// cpu.rs CLEARCOAT_F0.
const RAA_LAYERED_AMBIENT_CLEARCOAT_F0: f32 = 0.04;

// cpu.rs SHEEN_FIT: [i][j] multiplies r^i y^j.
const RAA_LAYERED_AMBIENT_SHEEN_FIT: array<array<f32, 5>, 4> = array(
    array(-4.1473703, -1.816077, 2.3879853e-1, 4.692581e-2, -3.0085763e-2),
    array(7.652171, 3.2228845e-1, 2.4963927, -2.5018659, 5.7769537e-1),
    array(-3.3230572, 6.867474e-1, -8.552088, 9.326004, -2.0739326),
    array(-2.581875e-1, -1.9488754, 9.092524, -9.094541, 1.3882859),
);

// cpu.rs SHEEN_SV_FIT.
const RAA_LAYERED_AMBIENT_SHEEN_SV_FIT: array<f32, 3> = array(3.228903, -7.618936, 4.0530457);

// cpu.rs poly2 for a 4 x 5 table (as raa_layered_poly_4x5).
fn raa_layered_ambient_poly_4x5(table: array<array<f32, 5>, 4>, x: f32, y: f32) -> f32 {
    var c = table;
    var acc = 0.0;
    for (var i = 3; i >= 0; i--) {
        var ry = 0.0;
        for (var j = 4; j >= 0; j--) {
            ry = ry * y + c[i][j];
        }
        acc = acc * x + ry;
    }
    return acc;
}

// cpu.rs sheen_albedo_fit (as raa_layered_sheen_albedo_fit).
fn raa_layered_ambient_sheen_albedo_fit(n_dot_v: f32, sheen_roughness: f32) -> f32 {
    let mu = clamp(n_dot_v, RAA_LAYERED_AMBIENT_MIN_N_DOT_V, 1.0);
    let r = clamp(sheen_roughness, RAA_LAYERED_AMBIENT_MIN_SHEEN_ROUGHNESS, 1.0);
    let y = min(mu / r, 3.0);
    let sv = sqrt(max(1.0 - mu * mu, 0.0));
    let c = RAA_LAYERED_AMBIENT_SHEEN_SV_FIT;
    return exp(raa_layered_ambient_poly_4x5(RAA_LAYERED_AMBIENT_SHEEN_FIT, r, y) + sv * (c[0] + r * (c[1] + r * c[2]))) / r;
}

// cpu.rs schlick.
fn raa_layered_ambient_schlick(f0: f32, cos: f32) -> f32 {
    let x = 1.0 - cos;
    let x2 = x * x;
    return f0 + (1.0 - f0) * (x2 * x2 * x);
}

// Multi-bounce AO (Jimenez et al. 2016, cubic fit over the albedo), as raa-ibl's ambient.
fn raa_layered_ambient_multi_bounce_ao(ao: f32, albedo: vec3<f32>) -> vec3<f32> {
    let a = 2.0404 * albedo - vec3<f32>(0.3324);
    let b = -4.7951 * albedo + vec3<f32>(0.6417);
    let c = 2.7552 * albedo + vec3<f32>(0.6903);
    return max(vec3<f32>(ao), ((ao * a + b) * ao + c) * ao);
}

// Dominant direction of the GGX specular lobe (Lagarde and de Rousiers 2014), as raa-ibl's
// ambient: the reflection vector bent towards the normal as the roughness grows.
fn raa_layered_ambient_dominant_dir(n: vec3<f32>, r: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let f = (1.0 - a) * (sqrt(1.0 - a) + a);
    return normalize(mix(n, r, f));
}

// Without an environment (info.w < 0.5): the S1 PbrLite ambient term, as raa-ibl. Otherwise
// raa-ibl's base terms (SH diffuse with multi-bounce AO, prefiltered specular with the
// split-sum DFG, multiscatter compensation and specular occlusion), scaled by the sheen
// albedo left to the base, plus the sheen indirect (SH irradiance x sheen colour x the sheen
// DG term: the albedo min(E_sheen, 1) of the direct sheen lobe after its normalisation), all
// attenuated by 1 - clearcoat Fc(n.v) as in direct, plus the clearcoat reflection
// (prefiltered at the clearcoat roughness along its dominant direction, split-sum DFG with
// F0 0.04, specular occlusion). Like the direct slot, the coat uses the shading normal (the
// slot receives no geometric normal).
fn raa_slot_ambient(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    if raa_env_sh.info.w < 0.5 {
        return raa_frame.ambient.rgb * s.base_color * (1.0 - 0.5 * s.metallic);
    }
    let n = s.normal;
    let v = c.view_dir;
    let roughness = clamp(s.roughness, 0.0, 1.0);
    let metallic = clamp(s.metallic, 0.0, 1.0);
    let f0 = c.f0;
    let dfg = raa_ibl_dfg(c.n_dot_v, roughness);
    let energy = vec3<f32>(1.0) + f0 * (1.0 / max(dfg.x + dfg.y, 1e-4) - 1.0);
    let irradiance = raa_ibl_irradiance(n);
    let diffuse = irradiance * s.base_color * (1.0 - metallic) * energy
        * raa_layered_ambient_multi_bounce_ao(c.ao, s.base_color) * raa_env_sh.info.y;
    let r = raa_layered_ambient_dominant_dir(n, reflect(-v, n), roughness);
    let specular = raa_ibl_specular(r, roughness) * (f0 * dfg.x + vec3<f32>(dfg.y)) * energy
        * raa_specular_occlusion(c.n_dot_v, c.ao, roughness) * raa_env_sh.info.x;

    let sr = clamp(s.sheen_roughness, RAA_LAYERED_AMBIENT_MIN_SHEEN_ROUGHNESS, 1.0);
    let e_sheen = min(raa_layered_ambient_sheen_albedo_fit(c.n_dot_v, sr), 1.0);
    let s_max = max(max(max(s.sheen_color.r, s.sheen_color.g), s.sheen_color.b), 0.0);
    let scale = max(1.0 - s_max * e_sheen, 0.0);
    let sheen = irradiance * max(s.sheen_color, vec3<f32>(0.0)) * e_sheen * raa_env_sh.info.y;

    let n_c = s.normal;
    let n_dot_v_c = max(dot(n_c, v), RAA_LAYERED_AMBIENT_MIN_N_DOT_V);
    let k = clamp(s.clearcoat, 0.0, 1.0);
    let cr = clamp(s.clearcoat_roughness, 0.0, 1.0);
    let atten = 1.0 - k * raa_layered_ambient_schlick(RAA_LAYERED_AMBIENT_CLEARCOAT_F0, n_dot_v_c);
    let dfg_c = raa_ibl_dfg(n_dot_v_c, cr);
    let r_c = raa_layered_ambient_dominant_dir(n_c, reflect(-v, n_c), cr);
    let coat = k * raa_ibl_specular(r_c, cr) * (RAA_LAYERED_AMBIENT_CLEARCOAT_F0 * dfg_c.x + dfg_c.y)
        * raa_specular_occlusion(n_dot_v_c, c.ao, cr) * raa_env_sh.info.x;
    return ((diffuse + specular) * scale + sheen) * atten + coat;
}
