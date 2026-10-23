//! CPU white furnace of the layered `Direct` model.
//!
//! `eval_direct` is integrated over the hemisphere of `n = +z` for a white base on the grid
//! roughness x n.v x metallic x clearcoat x sheen (the roughness is also the clearcoat and the
//! sheen roughness; sheen 1 is a white sheen color). The integral is a deterministic
//! stratified quadrature in the half-vector domain (see [`integrate`]), the integrand in f32
//! and the sums in f64. On this grid it is within 1e-4 of the same quadrature with 4x finer
//! strata, and it integrates `n.l / pi` to 1 within 2e-4 (checked below).
//! The layered albedo must stay at or below `1 + 1e-3`, clearcoat 0 must give exactly the
//! under-coat result (and, with sheen 0, exactly the base result), and the base alone must
//! keep most of the energy (the multiscatter compensation works). The cases are split over
//! `NMC_CPUS` threads when it is above 1.

use raa_brdf::{DirectTerms, MIN_ROUGHNESS, eval_base, eval_direct, eval_under_coat, f0_from_ior};
use std::f64::consts::PI;

const ROUGHNESS: [f32; 5] = [0.05, 0.25, 0.5, 0.75, 1.0];
const N_DOT_V: [f32; 3] = [0.1, 0.5, 1.0];
/// Azimuth strata over [0, pi).
const PHI_STRATA: usize = 32;
/// Polar strata per graded interval.
const THETA_STRATA: usize = 24;

/// Signature of `eval_direct`, `eval_under_coat` and `eval_base`.
type Eval = fn(&DirectTerms, [f32; 3], [f32; 3], [f32; 3]) -> [f32; 3];

#[derive(Clone, Copy, Debug)]
struct Case {
    roughness: f32,
    n_dot_v: f32,
    metallic: f32,
    clearcoat: f32,
    sheen: f32,
}

#[derive(Debug)]
struct Albedo {
    case: Case,
    direct: [f64; 3],
    /// `eval_under_coat` (clearcoat 0) or `eval_base` (clearcoat and sheen 0) on the same samples.
    reference: Option<[f64; 3]>,
}

fn terms(c: Case) -> DirectTerms {
    let white = [1.0; 3];
    DirectTerms {
        base_color: white,
        metallic: c.metallic,
        roughness: c.roughness,
        f0: f0_from_ior(white, c.metallic, 1.5),
        clearcoat: c.clearcoat,
        clearcoat_roughness: c.roughness,
        sheen_color: [c.sheen; 3],
        sheen_roughness: c.roughness,
    }
}

/// Midpoint stratified quadrature over the hemisphere of `n = +z`, for `v` in the xz plane,
/// in the half-vector domain: `l = reflect(-v, h)` and `dw_l = 4 (v.h) dw_h`. The azimuth
/// strata cover [0, pi) and count twice (the integrand is mirror-symmetric in y). The polar
/// range is the exact `l.z > 0` domain `theta_h < theta_max(phi)`, cut into intervals that
/// double from `alpha / 16` away from the lobe peak (`theta_h = 0`) and from the horizon, and
/// each interval into `THETA_STRATA` strata. `visit` gets each direction and its weight.
fn integrate(v: [f64; 3], alpha: f64, mut visit: impl FnMut([f32; 3], f64)) {
    let mut edges = Vec::new();
    for jp in 0..PHI_STRATA {
        let phi = PI * (jp as f64 + 0.5) / PHI_STRATA as f64;
        let (sin_p, cos_p) = phi.sin_cos();
        let theta_max = ((v[0] * cos_p).atan2(v[2]) + PI / 2.0) / 2.0;
        edges.clear();
        edges.extend([0.0, theta_max]);
        let mut d = alpha / 16.0;
        while d < theta_max / 2.0 {
            edges.extend([d, theta_max - d]);
            d *= 2.0;
        }
        edges.sort_by(f64::total_cmp);
        for w in edges.windows(2) {
            let step = (w[1] - w[0]) / THETA_STRATA as f64;
            for k in 0..THETA_STRATA {
                let (sin_t, cos_t) = (w[0] + step * (k as f64 + 0.5)).sin_cos();
                let h = [sin_t * cos_p, sin_t * sin_p, cos_t];
                let v_dot_h = v[0] * h[0] + v[2] * h[2];
                let l = [
                    2.0 * v_dot_h * h[0] - v[0],
                    2.0 * v_dot_h * h[1],
                    2.0 * v_dot_h * h[2] - v[2],
                ];
                if l[2] > 0.0 {
                    let weight = 4.0 * v_dot_h * sin_t * step * 2.0 * PI / PHI_STRATA as f64;
                    visit(l.map(|x| x as f32), weight);
                }
            }
        }
    }
}

fn view(n_dot_v: f32) -> [f64; 3] {
    let mu = f64::from(n_dot_v);
    [(1.0 - mu * mu).max(0.0).sqrt(), 0.0, mu]
}

fn alpha(roughness: f32) -> f64 {
    f64::from(roughness.max(MIN_ROUGHNESS)).powi(2)
}

fn albedo(case: Case) -> Albedo {
    let m = terms(case);
    let v = view(case.n_dot_v);
    let vf = v.map(|x| x as f32);
    let n = [0.0, 0.0, 1.0];
    let reference: Option<Eval> = match (case.clearcoat == 0.0, case.sheen == 0.0) {
        (true, true) => Some(eval_base),
        (true, false) => Some(eval_under_coat),
        _ => None,
    };
    let (mut direct, mut refsum) = ([0.0f64; 3], [0.0f64; 3]);
    integrate(v, alpha(case.roughness), |l, w| {
        let f = eval_direct(&m, n, vf, l);
        let r = reference.map_or(f, |g| g(&m, n, vf, l));
        for (acc, x) in direct.iter_mut().zip(f) {
            *acc += f64::from(x) * w;
        }
        for (acc, x) in refsum.iter_mut().zip(r) {
            *acc += f64::from(x) * w;
        }
    });
    Albedo {
        case,
        direct,
        reference: reference.map(|_| refsum),
    }
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for roughness in ROUGHNESS {
        for n_dot_v in N_DOT_V {
            for metallic in [0.0, 1.0] {
                for clearcoat in [0.0, 1.0] {
                    for sheen in [0.0, 1.0] {
                        out.push(Case {
                            roughness,
                            n_dot_v,
                            metallic,
                            clearcoat,
                            sheen,
                        });
                    }
                }
            }
        }
    }
    out
}

fn threads() -> usize {
    std::env::var("NMC_CPUS")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(1)
        .max(1)
}

#[test]
fn layered_albedo_is_at_most_one() {
    let cases = cases();
    let workers = threads().min(cases.len());
    let results: Vec<Albedo> = if workers > 1 {
        let chunk = cases.len().div_ceil(workers);
        std::thread::scope(|s| {
            let handles: Vec<_> = cases
                .chunks(chunk)
                .map(|part| s.spawn(move || part.iter().map(|&c| albedo(c)).collect::<Vec<_>>()))
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("worker panicked"))
                .collect()
        })
    } else {
        cases.iter().map(|&c| albedo(c)).collect()
    };
    assert_eq!(results.len(), 120);
    let mut worst = (0.0f64, None);
    for a in &results {
        let max = a.direct.iter().copied().fold(f64::MIN, f64::max);
        if max > worst.0 {
            worst = (max, Some(a.case));
        }
        assert!(
            max <= 1.0 + 1e-3,
            "albedo {:?} > 1 + 1e-3 for {:?}",
            a.direct,
            a.case
        );
        if let Some(r) = a.reference {
            assert_eq!(
                a.direct, r,
                "clearcoat 0 differs from the layers below for {:?}",
                a.case
            );
        }
        if a.case.clearcoat == 0.0 && a.case.sheen == 0.0 {
            let min = a.direct.iter().copied().fold(f64::MAX, f64::min);
            assert!(
                min >= 0.97,
                "base albedo {:?} lost energy for {:?}",
                a.direct,
                a.case
            );
        }
    }
    eprintln!("max layered albedo {:.6} at {:?}", worst.0, worst.1);
}

#[test]
fn quadrature_integrates_lambert_to_one() {
    for roughness in ROUGHNESS {
        for n_dot_v in N_DOT_V {
            let mut sum = 0.0;
            integrate(view(n_dot_v), alpha(roughness), |l, w| {
                sum += f64::from(l[2]) / PI * w
            });
            assert!(
                (sum - 1.0).abs() < 2e-4,
                "cosine integral {sum} at {roughness}, {n_dot_v}"
            );
        }
    }
}
