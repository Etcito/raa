//! CPU twins of the ReSTIR GI maths of the screen part's `restir` mode
//! (`wgsl/screen/restir_gi_common.wgsl`, `restir_gi_main.wgsl`, `restir_gi_spatial.wgsl`), after
//! Ouyang et al., "ReSTIR GI: Path Resampling for Real-Time Path Tracing", HPG 2021. Each function
//! carries the WGSL name minus the `raa_rgi_` prefix; the reservoir helpers it builds on (WRS,
//! GRIS weight, confidence clamp) are the area-light ReSTIR DI twins of
//! [`crate::parts::area_shadow::restir_cpu`], shared with the GPU through `restir_core.wgsl`.
//! The tests check every ported formula against a value derived independently in the test (a
//! closed form or a numerical solid angle), never against the ported code itself.

use crate::parts::area_shadow::restir_cpu::{contribution_weight, luminance, wrs_accept};

/// Spatial neighbours merged per pixel (Ouyang et al. use 3-5 with a visibility test each).
pub const RESTIR_GI_SPATIAL_SAMPLES: u32 = 4;
/// Spatial-reuse disc radius in pixels.
pub const RESTIR_GI_SPATIAL_RADIUS: f32 = 24.0;
/// Temporal confidence cap: the history's M is clamped to this many times the current M.
pub const RESTIR_GI_M_CAP: f32 = 20.0;
/// Boiling filter factor over the workgroup mean of target x W.
pub const RESTIR_GI_BOILING_FACTOR: f32 = 20.0;
/// Smallest normal cosine between a pixel and a reused reservoir's owner.
pub const RESTIR_GI_NORMAL_COS: f32 = 0.9;
/// Largest shift Jacobian a spatial neighbour may carry (and its inverse the smallest).
pub const RESTIR_GI_JACOBIAN_MAX: f32 = 10.0;

type V3 = [f32; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The target p_hat of a sample (x_s, n_s, radiance) at the receiver (x_v, n_v): luminance of the
/// radiance times max(n_v . w, 0), w the unit direction to x_s; 0 when the receiver is behind the
/// sample surface (n_s . -w <= 0).
pub fn target(x_v: V3, n_v: V3, x_s: V3, n_s: V3, radiance: V3) -> f32 {
    let d = sub(x_s, x_v);
    let len = dot(d, d).sqrt();
    if len <= 0.0 {
        return 0.0;
    }
    let w = [d[0] / len, d[1] / len, d[2] / len];
    let cos_v = dot(n_v, w);
    if cos_v <= 0.0 || -dot(n_s, w) <= 0.0 {
        return 0.0;
    }
    luminance(radiance) * cos_v
}

/// The reconnection-shift Jacobian |d w_r / d w_q| of a sample x_s (normal n_s) moved from the
/// visible point x_q to x_r: (|cos phi_r| / |cos phi_q|) (|x_q - x_s|^2 / |x_r - x_s|^2), phi the
/// angle at x_s between n_s and the direction to the visible point. 0 when degenerate.
pub fn jacobian(x_r: V3, x_q: V3, x_s: V3, n_s: V3) -> f32 {
    let to_r = sub(x_r, x_s);
    let to_q = sub(x_q, x_s);
    let d2_r = dot(to_r, to_r);
    let d2_q = dot(to_q, to_q);
    if d2_r <= 0.0 || d2_q <= 0.0 {
        return 0.0;
    }
    let cos_r = dot(n_s, to_r).abs() / d2_r.sqrt();
    let cos_q = dot(n_s, to_q).abs() / d2_q.sqrt();
    if cos_q <= 0.0 {
        return 0.0;
    }
    (cos_r / cos_q) * (d2_q / d2_r)
}

/// The solid-angle pdf of the cosine-weighted direction with cosine `cos_theta`: cos / pi.
pub fn cosine_pdf(cos_theta: f32) -> f32 {
    cos_theta.max(0.0) / std::f32::consts::PI
}

/// The initial one-candidate W: w = p_hat / pdf, W = w / p_hat (GRIS), so 1 / pdf when p_hat > 0.
pub fn initial_weight(p_hat: f32, pdf: f32) -> f32 {
    if pdf <= 0.0 {
        return 0.0;
    }
    contribution_weight(p_hat / pdf, p_hat)
}

/// The temporal M clamp of `raa_rgi_stage1`: `restir_gi_main.wgsl` applies
/// `raa_restir_clamp_confidence(t.confidence, c.confidence, raa_rgi_params.m_cap)`, whose body in
/// `restir_core.wgsl` is `max(min(previous, cap * current), 0.0)`: the history's M is clamped to
/// `m_cap` times the current M and never below zero.
///
/// Handoff: raa-gi CPU tests: ReSTIR GI reservoir temporal and spatial updates. The test calls
/// `temporal_m_clamp(previous_m: f32, current_m: f32, m_cap: f32) -> f32`, exported as
/// `raa_gi::parts::screen::restir_gi_cpu::temporal_m_clamp`; it passes no state.
pub fn temporal_m_clamp(previous_m: f32, current_m: f32, m_cap: f32) -> f32 {
    previous_m.min(m_cap * current_m).max(0.0)
}

/// One reservoir as the merge sees it: its sample's target at the receiver (already times the
/// shift Jacobian for a neighbour), its W and its M.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Candidate {
    pub target: f32,
    pub weight: f32,
    pub confidence: f32,
}

/// Streams `candidates` through weighted reservoir sampling with weights target x W x M and the
/// uniform numbers `xi`; returns (kept index, w_sum, summed M). The kept index is `None` when
/// every weight is 0.
pub fn merge(candidates: &[Candidate], xi: &[f32]) -> (Option<usize>, f32, f32) {
    let mut kept = None;
    let mut w_sum = 0.0;
    let mut m = 0.0;
    for (i, c) in candidates.iter().enumerate() {
        let w = c.target * c.weight * c.confidence;
        let w = if w.is_finite() && w >= 0.0 { w } else { 0.0 };
        w_sum += w;
        m += c.confidence;
        if wrs_accept(w_sum, w, xi[i]) {
            kept = Some(i);
        }
    }
    (kept, w_sum, m)
}

/// The merged W: w_sum / (Z p_hat(z)), Z the confidence normaliser and `target_z` the kept
/// sample's unshifted target at the receiver.
pub fn merged_weight(w_sum: f32, z: f32, target_z: f32) -> f32 {
    if z <= 0.0 {
        return 0.0;
    }
    contribution_weight(w_sum / z, target_z)
}

/// The irradiance estimate of a reservoir at its receiver: L cos theta W (per channel).
pub fn irradiance(radiance: V3, cos_theta: f32, weight: f32) -> V3 {
    let k = cos_theta.max(0.0) * weight;
    [radiance[0] * k, radiance[1] * k, radiance[2] * k]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn close(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    /// Solid angle of the triangle (a, b, c) seen from the origin, Van Oosterom and Strackee 1983:
    /// tan(Omega / 2) = |a . (b x c)| / (|a||b||c| + (a.b)|c| + (a.c)|b| + (b.c)|a|). Independent
    /// of the Jacobian formula under test.
    fn triangle_solid_angle(a: V3, b: V3, c: V3) -> f64 {
        let f = |v: V3| [v[0] as f64, v[1] as f64, v[2] as f64];
        let (a, b, c) = (f(a), f(b), f(c));
        let dot = |x: [f64; 3], y: [f64; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
        let cross = |x: [f64; 3], y: [f64; 3]| {
            [
                x[1] * y[2] - x[2] * y[1],
                x[2] * y[0] - x[0] * y[2],
                x[0] * y[1] - x[1] * y[0],
            ]
        };
        let len = |x: [f64; 3]| dot(x, x).sqrt();
        let num = dot(a, cross(b, c)).abs();
        let den =
            len(a) * len(b) * len(c) + dot(a, b) * len(c) + dot(a, c) * len(b) + dot(b, c) * len(a);
        2.0 * num.atan2(den)
    }

    /// Solid angle of a small square (side `h`, centre `centre`, in the plane spanned by `u`, `v`)
    /// seen from `eye`, as two triangles.
    fn square_solid_angle(eye: V3, centre: V3, u: V3, v: V3, h: f32) -> f64 {
        let p = |su: f32, sv: f32| {
            [
                centre[0] + (u[0] * su + v[0] * sv) * h * 0.5 - eye[0],
                centre[1] + (u[1] * su + v[1] * sv) * h * 0.5 - eye[1],
                centre[2] + (u[2] * su + v[2] * sv) * h * 0.5 - eye[2],
            ]
        };
        triangle_solid_angle(p(-1.0, -1.0), p(1.0, -1.0), p(1.0, 1.0))
            + triangle_solid_angle(p(-1.0, -1.0), p(1.0, 1.0), p(-1.0, 1.0))
    }

    #[test]
    fn jacobian_matches_a_hand_derived_case() {
        // x_s at the origin facing +z, x_q straight above at distance 1, x_r at (1, 0, 1):
        // cos_q = 1, cos_r = 1/sqrt(2), d_q^2 = 1, d_r^2 = 2, so J = (1/sqrt(2)) / 1 * 1 / 2.
        let j = jacobian([1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0; 3], [0.0, 0.0, 1.0]);
        close(j, 0.353_553_4, 1e-6);
    }

    #[test]
    fn jacobian_is_the_ratio_of_the_solid_angles_of_a_small_patch() {
        // A 1 mm patch at x_s tilted off the axes, seen from two visible points: the shift maps
        // the patch's solid angle at x_q onto its solid angle at x_r, so J = Omega_r / Omega_q.
        let x_s = [0.3, -0.2, 0.5];
        let n = {
            let raw = [0.2f32, 0.4, 0.89];
            let l = dot(raw, raw).sqrt();
            [raw[0] / l, raw[1] / l, raw[2] / l]
        };
        // A tangent basis of the patch plane.
        let u = {
            let raw = [n[2], 0.0, -n[0]];
            let l = dot(raw, raw).sqrt();
            [raw[0] / l, raw[1] / l, raw[2] / l]
        };
        let v = [
            n[1] * u[2] - n[2] * u[1],
            n[2] * u[0] - n[0] * u[2],
            n[0] * u[1] - n[1] * u[0],
        ];
        let x_q = [1.1, 0.4, 2.0];
        let x_r = [-0.7, 1.3, 1.6];
        let omega_q = square_solid_angle(x_q, x_s, u, v, 1e-3);
        let omega_r = square_solid_angle(x_r, x_s, u, v, 1e-3);
        close(jacobian(x_r, x_q, x_s, n), (omega_r / omega_q) as f32, 1e-3);
    }

    #[test]
    fn initial_sample_under_a_uniform_sky_estimates_pi_l() {
        // E = integral of L cos over the hemisphere = pi L for a uniform radiance L; a
        // cosine-sampled one-candidate reservoir estimates L cos W = L cos pi / cos = pi L for any
        // direction, so its variance is zero.
        let radiance = [0.5, 1.0, 2.0];
        for cos_theta in [0.05f32, 0.3, 0.71, 1.0] {
            let n_v = [0.0, 0.0, 1.0];
            let sin = (1.0 - cos_theta * cos_theta).sqrt();
            let x_s = [sin * 50.0, 0.0, cos_theta * 50.0];
            let n_s = [-sin, 0.0, -cos_theta];
            let p_hat = target([0.0; 3], n_v, x_s, n_s, radiance);
            let w = initial_weight(p_hat, cosine_pdf(cos_theta));
            let e = irradiance(radiance, cos_theta, w);
            for (got, l) in e.iter().zip(radiance) {
                close(*got, PI * l, 1e-5);
            }
        }
    }

    #[test]
    fn temporal_merge_of_two_sky_reservoirs_still_estimates_pi_l() {
        // Two one-candidate reservoirs (cosines 0.2 and 0.9) of the same uniform sky L = 1: each
        // weight is p_hat W M = cos (pi / cos) = pi, w_sum = 2 pi, M = 2, and whichever sample is
        // kept W = 2 pi / (2 cos_k), so L cos_k W = pi exactly.
        let cosines = [0.2f32, 0.9];
        let candidates: Vec<Candidate> = cosines
            .iter()
            .map(|&c| Candidate {
                target: c,
                weight: PI / c,
                confidence: 1.0,
            })
            .collect();
        for xi in [[0.0, 0.0], [0.0, 0.99], [0.7, 0.2]] {
            let (kept, w_sum, m) = merge(&candidates, &xi);
            let k = kept.expect("a positive weight is kept");
            close(w_sum, 2.0 * PI, 1e-6);
            assert_eq!(m, 2.0);
            let w = merged_weight(w_sum, m, cosines[k]);
            close(irradiance([1.0; 3], cosines[k], w)[0], PI, 1e-6);
        }
    }

    #[test]
    fn spatial_reuse_of_a_small_emitter_gives_its_closed_form_irradiance() {
        // A small emitter of area A and radiance L at x_s (normal n_s). At x_q a sample of it drawn
        // uniformly over its area has the solid-angle pdf d_q^2 / (A cos_sq), so W_q = A cos_sq /
        // d_q^2. Reused at x_r through the shift, W_r = W_q J, and the estimate L cos_r W_r must be
        // the small-source irradiance L A cos_r cos_sr / d_r^2 computed directly here.
        let area = 1e-2f32;
        let l = 3.0f32;
        let x_s = [0.0, 2.0, 0.0];
        let n_s = [0.0, -1.0, 0.0];
        let x_q = [0.5, 0.0, 0.0];
        let x_r = [-1.0, 0.0, 0.4];
        let n_r = [0.0, 1.0, 0.0];
        let d2 = |a: V3| dot(sub(a, x_s), sub(a, x_s));
        let cos_s = |a: V3| dot(n_s, sub(a, x_s)) / d2(a).sqrt();
        let w_q = area * cos_s(x_q) / d2(x_q);
        let w_r = w_q * jacobian(x_r, x_q, x_s, n_s);
        let to_s = sub(x_s, x_r);
        let cos_r = dot(n_r, to_s) / d2(x_r).sqrt();
        let estimate = irradiance([l; 3], cos_r, w_r)[0];
        // Hand form: d_r^2 = 1 + 4 + 0.16 = 5.16, cos_sr = 2 / sqrt(5.16), cos_r = 2 / sqrt(5.16).
        let closed = l * area * (2.0 / 5.16f32.sqrt()) * (2.0 / 5.16f32.sqrt()) / 5.16;
        close(estimate, closed, 1e-5);
    }

    #[test]
    fn target_is_zero_behind_either_surface() {
        let up = [0.0, 0.0, 1.0];
        // Sample below the receiver's horizon.
        assert_eq!(target([0.0; 3], up, [0.0, 0.0, -1.0], up, [1.0; 3]), 0.0);
        // Receiver behind the sample surface (n_s faces away from the receiver).
        assert_eq!(target([0.0; 3], up, [0.0, 0.0, 1.0], up, [1.0; 3]), 0.0);
        // Facing each other at 1 m: luminance of white (1) times cos 1.
        close(
            target([0.0; 3], up, [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0; 3]),
            1.0,
            1e-6,
        );
    }
}
