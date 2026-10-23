// Direct slot of provider `layered` (raa-brdf): the layered BRDF of src/cpu.rs term by term,
// with the same operations, guards, constants and fit coefficients, so the CPU white furnace
// (tests/energy.rs) speaks for this shader. Each item names the CPU item it mirrors; the
// unit tests of src/shaders.rs check that the constants and fit tables are the CPU ones.
// Base and sheen use the shading normal; the clearcoat takes its own normal (see
// raa_slot_direct). With both normals equal, raa_layered_direct is eval_direct.

// cpu.rs MIN_ROUGHNESS: smallest base and clearcoat roughness.
const RAA_LAYERED_MIN_ROUGHNESS: f32 = 0.03;
// cpu.rs MIN_SHEEN_ROUGHNESS: smallest sheen roughness (the sheen fit starts there).
const RAA_LAYERED_MIN_SHEEN_ROUGHNESS: f32 = 0.07;
// cpu.rs MIN_N_DOT_V: smallest n.v, the clamp of the shading context.
const RAA_LAYERED_MIN_N_DOT_V: f32 = 1e-4;
// cpu.rs CLEARCOAT_F0: reflectance at normal incidence of the clearcoat (IOR 1.5).
const RAA_LAYERED_CLEARCOAT_F0: f32 = 0.04;
// std::f32::consts::PI and FRAC_1_PI.
const RAA_LAYERED_PI: f32 = 3.14159265358979;
const RAA_LAYERED_FRAC_1_PI: f32 = 0.318309886183791;

// cpu.rs GGX_E_FIT: [i][j] multiplies r^i s^j.
const RAA_LAYERED_GGX_E_FIT: array<array<f32, 6>, 6> = array(
    array(9.8261607e-1, 2.568458e-1, -2.6844068, 4.714729, -1.9344544, -3.1908864e-1),
    array(2.751118e-1, -4.1380787, 2.1080915e1, -4.091027e1, 2.798998e1, -4.9574423),
    array(-1.13299, 1.3636595e1, -4.7906364e1, 6.020594, 1.4083081e2, -1.01894196e2),
    array(1.975456, -1.3389243e1, -3.1948303e1, 4.4584875e2, -9.005062e2, 4.4014893e2),
    array(-1.5405954, -5.4301715, 1.45836e2, -7.5445105e2, 1.1558339e3, -4.051355e2),
    array(4.4077283e-1, 5.226246, -6.617924e1, 2.7440497e2, -3.078345e2, -5.4808893),
);

// cpu.rs GGX_B_FIT: [i][j] multiplies sqrt(r)^i s^j.
const RAA_LAYERED_GGX_B_FIT: array<array<f32, 5>, 5> = array(
    array(6.1409366e-1, 1.4151262, -5.111445, 8.190093, -4.0225077),
    array(4.102667, -1.0679793e1, 1.7009315e1, -2.5127844e1, 1.3386653e1),
    array(-1.5363979e1, 2.8778572e1, -1.1299996e1, 2.8101679e1, -2.2526014e1),
    array(1.6285337e1, -3.173869e1, -7.739733, -2.9208614e1, 3.2184086e1),
    array(-5.587544, 1.2111293e1, 8.391763, 1.3928249e1, -5.957267),
);

// cpu.rs SHEEN_FIT: [i][j] multiplies r^i y^j.
const RAA_LAYERED_SHEEN_FIT: array<array<f32, 5>, 4> = array(
    array(-4.1473703, -1.816077, 2.3879853e-1, 4.692581e-2, -3.0085763e-2),
    array(7.652171, 3.2228845e-1, 2.4963927, -2.5018659, 5.7769537e-1),
    array(-3.3230572, 6.867474e-1, -8.552088, 9.326004, -2.0739326),
    array(-2.581875e-1, -1.9488754, 9.092524, -9.094541, 1.3882859),
);

// cpu.rs SHEEN_SV_FIT: coefficients of sqrt(1 - mu^2) r^k in the sheen fit.
const RAA_LAYERED_SHEEN_SV_FIT: array<f32, 3> = array(3.228903, -7.618936, 4.0530457);

// cpu.rs poly2 for a 6 x 6 table: sum c[i][j] x^i y^j, Horner in y then in x. The table is
// copied into a variable because WGSL indexes arrays dynamically only through memory.
fn raa_layered_poly_6x6(table: array<array<f32, 6>, 6>, x: f32, y: f32) -> f32 {
    var c = table;
    var acc = 0.0;
    for (var i = 5; i >= 0; i--) {
        var ry = 0.0;
        for (var j = 5; j >= 0; j--) {
            ry = ry * y + c[i][j];
        }
        acc = acc * x + ry;
    }
    return acc;
}

// cpu.rs poly2 for a 5 x 5 table.
fn raa_layered_poly_5x5(table: array<array<f32, 5>, 5>, x: f32, y: f32) -> f32 {
    var c = table;
    var acc = 0.0;
    for (var i = 4; i >= 0; i--) {
        var ry = 0.0;
        for (var j = 4; j >= 0; j--) {
            ry = ry * y + c[i][j];
        }
        acc = acc * x + ry;
    }
    return acc;
}

// cpu.rs poly2 for a 4 x 5 table.
fn raa_layered_poly_4x5(table: array<array<f32, 5>, 4>, x: f32, y: f32) -> f32 {
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

// cpu.rs pow5.
fn raa_layered_pow5(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

// cpu.rs len2.
fn raa_layered_len2(a: vec3<f32>) -> f32 {
    return dot(a, a);
}

// cpu.rs normalize: guarded against a zero vector (v = -l), unlike the WGSL built-in.
fn raa_layered_normalize(a: vec3<f32>) -> vec3<f32> {
    let k = 1.0 / sqrt(max(dot(a, a), 1e-12));
    return a * k;
}

// cpu.rs ggx_d: GGX distribution for alpha = roughness^2, with sin2_h = |n x h|^2
// (n.h^2 (a2 - 1) + 1 written as sin2_h + a2 n.h^2).
fn raa_layered_ggx_d(n_dot_h: f32, sin2_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let t = sin2_h + a2 * n_dot_h * n_dot_h;
    return a2 / (RAA_LAYERED_PI * max(t * t, 1e-20));
}

// cpu.rs smith_correlated_v: height-correlated Smith visibility G2 / (4 n.l n.v).
fn raa_layered_smith_correlated_v(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let gv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let gl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-12);
}

// cpu.rs charlie_d: Charlie sheen distribution for alpha = sheen_roughness^2. pow(0, y) is
// not defined in WGSL, so sin2_h is kept above 1e-20 (the result then stays below 1e-9
// where the CPU gives 0).
fn raa_layered_charlie_d(sin2_h: f32, alpha: f32) -> f32 {
    return (2.0 + 1.0 / alpha) * pow(max(sin2_h, 1e-20), 0.5 / alpha) / (2.0 * RAA_LAYERED_PI);
}

// cpu.rs neubelt_v: Neubelt and Pettineo cloth visibility.
fn raa_layered_neubelt_v(n_dot_v: f32, n_dot_l: f32) -> f32 {
    return 1.0 / (4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v));
}

// cpu.rs kelemen_v: Kelemen and Szirmay-Kalos visibility 1 / (4 (l.h)^2), with l.h = v.h.
fn raa_layered_kelemen_v(v_dot_h: f32) -> f32 {
    return 0.25 / max(v_dot_h * v_dot_h, 1e-4);
}

// cpu.rs schlick.
fn raa_layered_schlick(f0: f32, cos: f32) -> f32 {
    return f0 + (1.0 - f0) * raa_layered_pow5(1.0 - cos);
}

// cpu.rs ggx_dfg_fit: split-sum terms (A, B) of single-scatter GGX with height-correlated
// Smith visibility; E = A + B is the directional albedo at F0 = 1 (over-estimated).
fn raa_layered_ggx_dfg_fit(n_dot_v: f32, roughness: f32) -> vec2<f32> {
    let mu = clamp(n_dot_v, RAA_LAYERED_MIN_N_DOT_V, 1.0);
    let r = clamp(roughness, RAA_LAYERED_MIN_ROUGHNESS, 1.0);
    let s = mu / (mu + r * r);
    let e = clamp(raa_layered_poly_6x6(RAA_LAYERED_GGX_E_FIT, r, s), 0.25, 1.0);
    let b = clamp(
        raa_layered_pow5(1.0 - mu) * raa_layered_poly_5x5(RAA_LAYERED_GGX_B_FIT, sqrt(r), s) + 0.006,
        0.0,
        e,
    );
    return vec2<f32>(e - b, b);
}

// cpu.rs sheen_albedo_fit: directional albedo E_sheen of the Charlie x Neubelt lobe for a
// white sheen colour (over-estimated; above 1 at grazing view for a smooth sheen).
fn raa_layered_sheen_albedo_fit(n_dot_v: f32, sheen_roughness: f32) -> f32 {
    let mu = clamp(n_dot_v, RAA_LAYERED_MIN_N_DOT_V, 1.0);
    let r = clamp(sheen_roughness, RAA_LAYERED_MIN_SHEEN_ROUGHNESS, 1.0);
    let y = min(mu / r, 3.0);
    let sv = sqrt(max(1.0 - mu * mu, 0.0));
    let c = RAA_LAYERED_SHEEN_SV_FIT;
    return exp(raa_layered_poly_4x5(RAA_LAYERED_SHEEN_FIT, r, y) + sv * (c[0] + r * (c[1] + r * c[2]))) / r;
}

// cpu.rs eval_base: compensated GGX plus the albedo-weighted Lambert lobe, times n.l; 0
// below the horizon of n (the negated test also catches a NaN where the GPU keeps it).
fn raa_layered_base(s: MaterialOutput, f0: vec3<f32>, n: vec3<f32>, v: vec3<f32>, l: vec3<f32>) -> vec3<f32> {
    let n_dot_l = dot(n, l);
    if !(n_dot_l > 0.0) {
        return vec3<f32>(0.0);
    }
    let n_dot_v = max(dot(n, v), RAA_LAYERED_MIN_N_DOT_V);
    let h = raa_layered_normalize(v + l);
    let n_dot_h = max(dot(n, h), 0.0);
    let sin2_h = raa_layered_len2(cross(n, h));
    let v_dot_h = max(dot(v, h), 0.0);
    let r = clamp(s.roughness, RAA_LAYERED_MIN_ROUGHNESS, 1.0);
    let a = r * r;
    let dv = raa_layered_ggx_d(n_dot_h, sin2_h, a) * raa_layered_smith_correlated_v(n_dot_v, n_dot_l, a);
    let fc = raa_layered_pow5(1.0 - v_dot_h);
    let dfg = raa_layered_ggx_dfg_fit(n_dot_v, r);
    let e = dfg.x + dfg.y;
    let diffuse = (1.0 - clamp(s.metallic, 0.0, 1.0)) * RAA_LAYERED_FRAC_1_PI;
    let energy = vec3<f32>(1.0) + f0 * (1.0 / e - 1.0);
    let spec = dv * (f0 + (vec3<f32>(1.0) - f0) * fc) * energy;
    let weight = max(vec3<f32>(1.0) - (f0 * dfg.x + vec3<f32>(dfg.y)) * energy, vec3<f32>(0.0));
    return (s.base_color * diffuse * weight + spec) * n_dot_l;
}

// cpu.rs eval_under_coat: the base scaled by 1 - max(sheen) min(E_sheen, 1) plus the sheen
// lobe divided by max(E_sheen, 1), times n.l.
fn raa_layered_under_coat(s: MaterialOutput, f0: vec3<f32>, n: vec3<f32>, v: vec3<f32>, l: vec3<f32>) -> vec3<f32> {
    let base = raa_layered_base(s, f0, n, v, l);
    let n_dot_l = dot(n, l);
    if !(n_dot_l > 0.0) {
        return vec3<f32>(0.0);
    }
    let n_dot_v = max(dot(n, v), RAA_LAYERED_MIN_N_DOT_V);
    let h = raa_layered_normalize(v + l);
    let sin2_h = raa_layered_len2(cross(n, h));
    let sr = clamp(s.sheen_roughness, RAA_LAYERED_MIN_SHEEN_ROUGHNESS, 1.0);
    let e_sheen = raa_layered_sheen_albedo_fit(n_dot_v, sr);
    let s_max = max(max(max(s.sheen_color.r, s.sheen_color.g), s.sheen_color.b), 0.0);
    let scale = max(1.0 - s_max * min(e_sheen, 1.0), 0.0);
    let lobe = raa_layered_charlie_d(sin2_h, sr * sr) * raa_layered_neubelt_v(n_dot_v, n_dot_l) * n_dot_l
        / max(e_sheen, 1.0);
    return base * scale + max(s.sheen_color, vec3<f32>(0.0)) * lobe;
}

// cpu.rs eval_direct: the under-coat layers on the shading normal n, attenuated by
// 1 - clearcoat Fc(n_c.v), plus the clearcoat lobe (GGX, F0 0.04, Kelemen) on n_c, times
// n_c.l; 0 below the horizon of n_c. With n_c = n this is eval_direct.
fn raa_layered_direct(s: MaterialOutput, f0: vec3<f32>, n: vec3<f32>, n_c: vec3<f32>, v: vec3<f32>, l: vec3<f32>) -> vec3<f32> {
    let under = raa_layered_under_coat(s, f0, n, v, l);
    let n_dot_l = dot(n_c, l);
    if !(n_dot_l > 0.0) {
        return vec3<f32>(0.0);
    }
    let n_dot_v = max(dot(n_c, v), RAA_LAYERED_MIN_N_DOT_V);
    let h = raa_layered_normalize(v + l);
    let n_dot_h = max(dot(n_c, h), 0.0);
    let sin2_h = raa_layered_len2(cross(n_c, h));
    let v_dot_h = max(dot(v, h), 0.0);
    let k = clamp(s.clearcoat, 0.0, 1.0);
    let cr = clamp(s.clearcoat_roughness, RAA_LAYERED_MIN_ROUGHNESS, 1.0);
    let atten = 1.0 - k * raa_layered_schlick(RAA_LAYERED_CLEARCOAT_F0, n_dot_v);
    let coat = k
        * raa_layered_ggx_d(n_dot_h, sin2_h, cr * cr)
        * raa_layered_kelemen_v(v_dot_h)
        * raa_layered_schlick(RAA_LAYERED_CLEARCOAT_F0, v_dot_h)
        * n_dot_l;
    return under * atten + vec3<f32>(coat);
}

// Layered direct light: f(v, l) n.l x radiance, with F0 from the shading context (cpu.rs
// f0_from_ior, as raa_pass_f0). The S11 Direct slot receives no geometric normal, so the
// clearcoat uses the shading normal too: the interpolated geometric normal for every
// material that keeps raa_default_output's normal, and then exactly eval_direct.
fn raa_slot_direct(s: MaterialOutput, c: ShadeCtx, l: vec3<f32>, radiance: vec3<f32>) -> vec3<f32> {
    let n_c = s.normal;
    return raa_layered_direct(s, c.f0, s.normal, n_c, c.view_dir, l) * radiance;
}
