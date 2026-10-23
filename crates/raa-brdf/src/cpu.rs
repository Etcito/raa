//! CPU reference of the layered `Direct` model, written so the WGSL provider mirrors it line
//! by line (same operations, same order, same guards, same fit coefficients).
//!
//! Layers, from the bottom up, for a unit normal `n`, view `v` and light `l`:
//!
//! - **Base**: Lambert x (1 - metallic) + GGX with height-correlated Smith visibility, the
//!   Schlick Fresnel and the multiscatter compensation `1 + F0 (1 / E(mu) - 1)`, where `E` is
//!   the single-scatter GGX directional albedo of [`ggx_dfg_fit`]. The Lambert lobe is weighted
//!   by `1 - (F0 A + B) (1 + F0 (1 / E - 1))`, the albedo of the compensated specular, so the
//!   layer never reflects more than it receives (without it a white dielectric reflects up to
//!   about 1.5 at grazing view).
//! - **Sheen**: Charlie NDF (Estevez and Kulla 2017) x Neubelt visibility. The base is scaled
//!   by `1 - max(sheen_color) min(E_sheen, 1)` with `E_sheen` from [`sheen_albedo_fit`]; where
//!   `E_sheen > 1` (Neubelt over-reflects at grazing view on a smooth sheen) the sheen lobe is
//!   divided by `E_sheen`, so a white sheen reflects at most what it receives.
//! - **Clearcoat**: GGX with F0 = [`CLEARCOAT_F0`] and Kelemen visibility `1 / (4 (l.h)^2)`
//!   on the geometric normal; base and sheen are attenuated by `1 - clearcoat Fc(n.v)` with
//!   `Fc` the coat's Schlick Fresnel. This reference takes one normal; the WGSL provider
//!   passes the geometric normal for the coat and the shading normal for the rest.
//!
//! Every fit is evaluated on clamped inputs and is conservative (it over-estimates the albedo
//! it approximates), which is what keeps the layered albedo at or below 1.

use std::f32::consts::{FRAC_1_PI, PI};

/// Smallest base and clearcoat roughness, the clamp of the material upload.
pub const MIN_ROUGHNESS: f32 = 0.03;
/// Smallest sheen roughness; the sheen albedo fit is made from here to 1.
pub const MIN_SHEEN_ROUGHNESS: f32 = 0.07;
/// Smallest `n.v`, the clamp of the shading context.
pub const MIN_N_DOT_V: f32 = 1e-4;
/// Reflectance at normal incidence of the clearcoat (IOR 1.5).
pub const CLEARCOAT_F0: f32 = 0.04;

/// The material terms the layered `Direct` model reads: the `MaterialOutput` fields of the
/// same names plus `f0`, the shading context's reflectance at normal incidence.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DirectTerms {
    /// Linear base color.
    pub base_color: [f32; 3],
    /// Metalness in [0, 1].
    pub metallic: f32,
    /// Perceptual roughness, clamped to [[`MIN_ROUGHNESS`], 1].
    pub roughness: f32,
    /// Reflectance at normal incidence, see [`f0_from_ior`].
    pub f0: [f32; 3],
    /// Clearcoat weight in [0, 1].
    pub clearcoat: f32,
    /// Perceptual clearcoat roughness, clamped to [[`MIN_ROUGHNESS`], 1].
    pub clearcoat_roughness: f32,
    /// Linear sheen color; black disables the sheen.
    pub sheen_color: [f32; 3],
    /// Perceptual sheen roughness, clamped to [[`MIN_SHEEN_ROUGHNESS`], 1].
    pub sheen_roughness: f32,
}

/// Reflectance at normal incidence, as `raa_pass_f0`: the dielectric value of `ior` (clamped
/// to [1, 3]) mixed toward `base_color` by `metallic` (clamped to [0, 1]).
pub fn f0_from_ior(base_color: [f32; 3], metallic: f32, ior: f32) -> [f32; 3] {
    let ior = ior.clamp(1.0, 3.0);
    let r = (ior - 1.0) / (ior + 1.0);
    let m = metallic.clamp(0.0, 1.0);
    base_color.map(|b| mix(r * r, b, m))
}

/// Reflected radiance per unit irradiance x cos (`f(v, l) n.l`) of the full layered model:
/// base, sheen and clearcoat. Zero when `l` is below the horizon of `n`.
pub fn eval_direct(m: &DirectTerms, n: [f32; 3], v: [f32; 3], l: [f32; 3]) -> [f32; 3] {
    let under = eval_under_coat(m, n, v, l);
    let n_dot_l = dot(n, l);
    if n_dot_l <= 0.0 || n_dot_l.is_nan() {
        return [0.0; 3];
    }
    let n_dot_v = dot(n, v).max(MIN_N_DOT_V);
    let h = normalize(add(v, l));
    let n_dot_h = dot(n, h).max(0.0);
    let sin2_h = len2(cross(n, h));
    let v_dot_h = dot(v, h).max(0.0);
    let k = m.clearcoat.clamp(0.0, 1.0);
    let cr = m.clearcoat_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let atten = 1.0 - k * schlick(CLEARCOAT_F0, n_dot_v);
    let coat = k
        * ggx_d(n_dot_h, sin2_h, cr * cr)
        * kelemen_v(v_dot_h)
        * schlick(CLEARCOAT_F0, v_dot_h)
        * n_dot_l;
    under.map(|u| u * atten + coat)
}

/// [`eval_direct`] without the clearcoat: the base scaled by the sheen albedo plus the sheen
/// lobe. With clearcoat 0, [`eval_direct`] returns exactly this value.
pub fn eval_under_coat(m: &DirectTerms, n: [f32; 3], v: [f32; 3], l: [f32; 3]) -> [f32; 3] {
    let base = eval_base(m, n, v, l);
    let n_dot_l = dot(n, l);
    if n_dot_l <= 0.0 || n_dot_l.is_nan() {
        return [0.0; 3];
    }
    let n_dot_v = dot(n, v).max(MIN_N_DOT_V);
    let h = normalize(add(v, l));
    let sin2_h = len2(cross(n, h));
    let sr = m.sheen_roughness.clamp(MIN_SHEEN_ROUGHNESS, 1.0);
    let e_sheen = sheen_albedo_fit(n_dot_v, sr);
    let s_max = m.sheen_color[0]
        .max(m.sheen_color[1])
        .max(m.sheen_color[2])
        .max(0.0);
    let scale = (1.0 - s_max * e_sheen.min(1.0)).max(0.0);
    let lobe =
        charlie_d(sin2_h, sr * sr) * neubelt_v(n_dot_v, n_dot_l) * n_dot_l / e_sheen.max(1.0);
    std::array::from_fn(|c| base[c] * scale + m.sheen_color[c].max(0.0) * lobe)
}

/// The base layer alone: compensated GGX plus the albedo-weighted Lambert lobe. With sheen
/// and clearcoat 0, [`eval_direct`] returns exactly this value.
pub fn eval_base(m: &DirectTerms, n: [f32; 3], v: [f32; 3], l: [f32; 3]) -> [f32; 3] {
    let n_dot_l = dot(n, l);
    if n_dot_l <= 0.0 || n_dot_l.is_nan() {
        return [0.0; 3];
    }
    let n_dot_v = dot(n, v).max(MIN_N_DOT_V);
    let h = normalize(add(v, l));
    let n_dot_h = dot(n, h).max(0.0);
    let sin2_h = len2(cross(n, h));
    let v_dot_h = dot(v, h).max(0.0);
    let r = m.roughness.clamp(MIN_ROUGHNESS, 1.0);
    let a = r * r;
    let dv = ggx_d(n_dot_h, sin2_h, a) * smith_correlated_v(n_dot_v, n_dot_l, a);
    let fc = pow5(1.0 - v_dot_h);
    let [dfg_a, dfg_b] = ggx_dfg_fit(n_dot_v, r);
    let e = dfg_a + dfg_b;
    let diffuse = (1.0 - m.metallic.clamp(0.0, 1.0)) * FRAC_1_PI;
    std::array::from_fn(|c| {
        let f0 = m.f0[c];
        let energy = 1.0 + f0 * (1.0 / e - 1.0);
        let spec = dv * (f0 + (1.0 - f0) * fc) * energy;
        let weight = (1.0 - (f0 * dfg_a + dfg_b) * energy).max(0.0);
        (m.base_color[c] * diffuse * weight + spec) * n_dot_l
    })
}

/// GGX normal distribution for `alpha = roughness^2`, with `sin2_h = |n x h|^2`:
/// `n.h^2 (a2 - 1) + 1` written as `sin2_h + a2 n.h^2`, which keeps its precision in f32 for
/// a narrow lobe (Filament's form).
fn ggx_d(n_dot_h: f32, sin2_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let t = sin2_h + a2 * n_dot_h * n_dot_h;
    a2 / (PI * (t * t).max(1e-20))
}

/// Height-correlated Smith visibility `G2 / (4 n.l n.v)` (Heitz 2014).
fn smith_correlated_v(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let gv = n_dot_l * (n_dot_v * n_dot_v * (1.0 - a2) + a2).sqrt();
    let gl = n_dot_v * (n_dot_l * n_dot_l * (1.0 - a2) + a2).sqrt();
    0.5 / (gv + gl).max(1e-12)
}

/// Charlie sheen distribution (Estevez and Kulla 2017) for `alpha = sheen_roughness^2`, with
/// `sin2_h = |n x h|^2`.
fn charlie_d(sin2_h: f32, alpha: f32) -> f32 {
    (2.0 + 1.0 / alpha) * sin2_h.powf(0.5 / alpha) / (2.0 * PI)
}

/// Neubelt and Pettineo 2013 cloth visibility.
fn neubelt_v(n_dot_v: f32, n_dot_l: f32) -> f32 {
    1.0 / (4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v))
}

/// Kelemen and Szirmay-Kalos 2001 visibility `1 / (4 (l.h)^2)`, with `l.h = v.h`.
fn kelemen_v(v_dot_h: f32) -> f32 {
    0.25 / (v_dot_h * v_dot_h).max(1e-4)
}

fn schlick(f0: f32, cos: f32) -> f32 {
    f0 + (1.0 - f0) * pow5(1.0 - cos)
}

/// Split-sum terms `[A, B]` of single-scatter GGX with height-correlated Smith visibility for
/// a white Schlick Fresnel: the directional albedo is `F0 A + B`, and `E = A + B` is the albedo
/// at F0 = 1. Same integral as the `raa-ibl` DFG LUT, here as an analytic fit.
///
/// Source: own minimax fit to that integral, computed by Gauss-Legendre quadrature over the
/// exact `n.l > 0` domain (error below 1e-6) on 1189 points of roughness [0.03, 1] x n.v
/// [1e-4, 1], checked on 400 random points. With `s = mu / (mu + r^2)` (the GGX albedo
/// valley sits at `mu ~ r^2`): `E = P(r, s)`, degree 5 x 5, minimax relative error 0.52 %
/// scaled up by 0.6 % so that `E >= E_true` (over-estimate at most 1.1 %), then clamped to 1;
/// `B = (1 - mu)^5 Q(sqrt r, s) + 0.006`, degree 4 x 4, minimax absolute error 0.0048 (0.0054
/// off the fit points), so that `B >= B_true` (over-estimate at most 0.012), then clamped to
/// `E`.
pub fn ggx_dfg_fit(n_dot_v: f32, roughness: f32) -> [f32; 2] {
    let mu = n_dot_v.clamp(MIN_N_DOT_V, 1.0);
    let r = roughness.clamp(MIN_ROUGHNESS, 1.0);
    let s = mu / (mu + r * r);
    let e = poly2(&GGX_E_FIT, r, s).clamp(0.25, 1.0);
    let b = (pow5(1.0 - mu) * poly2(&GGX_B_FIT, r.sqrt(), s) + 0.006).clamp(0.0, e);
    [e - b, b]
}

/// Directional albedo `E_sheen` of the Charlie x Neubelt sheen lobe (white sheen color).
///
/// Source: own minimax fit of `ln(E_sheen r)` to that integral, computed by Gauss-Legendre
/// quadrature over the exact `n.l > 0` domain on 1131 points of sheen roughness [0.07, 1] x
/// n.v [1e-4, 1], checked on 300 random points. For a smooth sheen `E_sheen r` is close to a
/// function of `y = mu / r`, so
/// `ln(E_sheen r) = P(r, min(y, 3)) + sqrt(1 - mu^2) (c0 + c1 r + c2 r^2)`, degree 3 x 4, minimax error 0.0295 in log (3 %), shifted by 0.036 so that
/// `E_sheen >= E_true` (over-estimate at most 6.8 %). `E_sheen` decreases with `mu`, so the
/// clamp of `y` keeps the fit above the true value beyond 3. `E_sheen` exceeds 1 at grazing
/// view for a sheen roughness below about 0.45 (about 5.7 at roughness 0.07, n.v 1e-4).
pub fn sheen_albedo_fit(n_dot_v: f32, sheen_roughness: f32) -> f32 {
    let mu = n_dot_v.clamp(MIN_N_DOT_V, 1.0);
    let r = sheen_roughness.clamp(MIN_SHEEN_ROUGHNESS, 1.0);
    let y = (mu / r).min(3.0);
    let sv = (1.0 - mu * mu).max(0.0).sqrt();
    let [c0, c1, c2] = SHEEN_SV_FIT;
    (poly2(&SHEEN_FIT, r, y) + sv * (c0 + r * (c1 + r * c2))).exp() / r
}

/// `GGX_E_FIT[i][j]` multiplies `r^i s^j`.
#[rustfmt::skip]
const GGX_E_FIT: [[f32; 6]; 6] = [
    [9.8261607e-1, 2.568458e-1, -2.6844068, 4.714729, -1.9344544, -3.1908864e-1],
    [2.751118e-1, -4.1380787, 2.1080915e1, -4.091027e1, 2.798998e1, -4.9574423],
    [-1.13299, 1.3636595e1, -4.7906364e1, 6.020594, 1.4083081e2, -1.01894196e2],
    [1.975456, -1.3389243e1, -3.1948303e1, 4.4584875e2, -9.005062e2, 4.4014893e2],
    [-1.5405954, -5.4301715, 1.45836e2, -7.5445105e2, 1.1558339e3, -4.051355e2],
    [4.4077283e-1, 5.226246, -6.617924e1, 2.7440497e2, -3.078345e2, -5.4808893],
];

/// `GGX_B_FIT[i][j]` multiplies `sqrt(r)^i s^j`.
#[rustfmt::skip]
const GGX_B_FIT: [[f32; 5]; 5] = [
    [6.1409366e-1, 1.4151262, -5.111445, 8.190093, -4.0225077],
    [4.102667, -1.0679793e1, 1.7009315e1, -2.5127844e1, 1.3386653e1],
    [-1.5363979e1, 2.8778572e1, -1.1299996e1, 2.8101679e1, -2.2526014e1],
    [1.6285337e1, -3.173869e1, -7.739733, -2.9208614e1, 3.2184086e1],
    [-5.587544, 1.2111293e1, 8.391763, 1.3928249e1, -5.957267],
];

/// `SHEEN_FIT[i][j]` multiplies `r^i y^j`.
#[rustfmt::skip]
const SHEEN_FIT: [[f32; 5]; 4] = [
    [-4.1473703, -1.816077, 2.3879853e-1, 4.692581e-2, -3.0085763e-2],
    [7.652171, 3.2228845e-1, 2.4963927, -2.5018659, 5.7769537e-1],
    [-3.3230572, 6.867474e-1, -8.552088, 9.326004, -2.0739326],
    [-2.581875e-1, -1.9488754, 9.092524, -9.094541, 1.3882859],
];

/// Coefficients of `sqrt(1 - mu^2) r^k` in the sheen fit.
const SHEEN_SV_FIT: [f32; 3] = [3.228903, -7.618936, 4.0530457];

/// `sum c[i][j] x^i y^j`, Horner in `y` then in `x`.
fn poly2<const I: usize, const J: usize>(c: &[[f32; J]; I], x: f32, y: f32) -> f32 {
    let mut acc = 0.0;
    for row in c.iter().rev() {
        let mut ry = 0.0;
        for &k in row.iter().rev() {
            ry = ry * y + k;
        }
        acc = acc * x + ry;
    }
    acc
}

fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    x2 * x2 * x
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn len2(a: [f32; 3]) -> f32 {
    dot(a, a)
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn normalize(a: [f32; 3]) -> [f32; 3] {
    let k = 1.0 / dot(a, a).max(1e-12).sqrt();
    a.map(|x| x * k)
}

#[cfg(test)]
mod tests {
    use super::*;

    // (n.v, roughness, E, B) from the reference quadrature of the fits.
    const GGX_REF: [(f32, f32, f32, f32); 6] = [
        (0.1, 0.25, 0.896770, 0.391622),
        (0.5, 0.5, 0.857263, 0.022347),
        (1.0, 1.0, 0.306853, 0.000034),
        (0.01, 0.1, 0.891987, 0.767624),
        (1e-4, 0.03, 0.977483, 0.954071),
        (1.0, 0.05, 0.999994, 0.0),
    ];
    // (n.v, sheen roughness, E_sheen).
    const SHEEN_REF: [(f32, f32, f32); 5] = [
        (0.1, 0.25, 0.854087),
        (0.5, 0.5, 0.229553),
        (1.0, 1.0, 0.141421),
        (0.02, 0.07, 3.341755),
        (0.3, 0.15, 0.162102),
    ];

    #[test]
    fn ggx_fit_over_estimates_within_bounds() {
        for (mu, r, e, b) in GGX_REF {
            let [fa, fb] = ggx_dfg_fit(mu, r);
            let fe = fa + fb;
            assert!(
                fe >= e && fe <= (e * 1.012).max(1.0),
                "E({mu}, {r}) = {fe} vs {e}"
            );
            assert!(fb >= b && fb <= b + 0.012, "B({mu}, {r}) = {fb} vs {b}");
        }
    }

    #[test]
    fn sheen_fit_over_estimates_within_bounds() {
        for (mu, r, e) in SHEEN_REF {
            let f = sheen_albedo_fit(mu, r);
            assert!(f >= e && f <= e * 1.07, "E_sheen({mu}, {r}) = {f} vs {e}");
        }
    }

    #[test]
    fn f0_matches_the_pass() {
        let r: f32 = 0.5 / 2.5;
        assert_eq!(f0_from_ior([0.2, 0.5, 0.9], 0.0, 1.5), [r * r; 3]);
        assert_eq!(f0_from_ior([0.2, 0.5, 0.9], 1.0, 1.5), [0.2, 0.5, 0.9]);
    }

    #[test]
    fn below_horizon_and_degenerate_inputs_stay_finite() {
        let m = DirectTerms {
            base_color: [1.0; 3],
            metallic: 0.0,
            roughness: 0.0,
            f0: [0.04; 3],
            clearcoat: 1.0,
            clearcoat_roughness: 0.0,
            sheen_color: [1.0; 3],
            sheen_roughness: 0.0,
        };
        let n = [0.0, 0.0, 1.0];
        assert_eq!(eval_direct(&m, n, n, [0.0, 0.0, -1.0]), [0.0; 3]);
        for l in [[0.0, 0.0, 1.0], [1.0, 0.0, 1e-6], [-1.0, 0.0, 1e-3]] {
            for v in [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 1e-6]] {
                let f = eval_direct(&m, n, normalize(v), normalize(l));
                assert!(
                    f.iter().all(|x| x.is_finite() && *x >= 0.0),
                    "{v:?} {l:?} {f:?}"
                );
            }
        }
    }
}
