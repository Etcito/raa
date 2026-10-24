//! Analytic optical depth of exponential height fog along a straight ray.
//!
//! Density follows `rho(y) = density * exp(-falloff * (y - base))` (S10 `fog` fields) and a
//! ray is `y(t) = origin_y + dir_y * t` for a distance `t` in metres. Integrating rho over
//! `[t0, t1]` gives the closed form
//!
//! ```text
//! density * exp(-falloff * (origin_y - base))
//!     * (exp(-falloff * dir_y * t0) - exp(-falloff * dir_y * t1)) / (falloff * dir_y)
//! ```
//!
//! where `t0 = max(start_m, desc.start_m, 0)` and `t1 = end_m`; the integral is 0 when
//! `end_m <= t0`. For near-horizontal rays (`|falloff * dir_y * (t1 - t0)| < 1e-4`) the
//! quotient is replaced by its series limit, so `dir_y = 0` never divides by zero.
//!
//! All arithmetic is `f64`; the result is returned as `f32`. The WGSL Fog slot provider
//! (`fog_slot.wgsl`, added with the pass task) mirrors this function so the CPU transmittance
//! and the shader agree. [`optical_depth_numeric`] is the composite-Simpson reference used by
//! `tests/integral.rs`, and [`transmittance`] is `exp(-optical_depth)`.

use crate::FogDesc;

/// Threshold on `|falloff * dir_y * (t1 - t0)|` below which the series limit is used.
const NEAR_HORIZONTAL: f64 = 1e-4;

/// Analytic optical depth of `desc`'s fog along a ray from `start_m` to `end_m` metres.
///
/// `origin_y` is the ray origin height and `dir_y` its (unit or not) vertical direction.
/// The integration starts at `max(start_m, desc.start_m, 0)` so fog never accumulates before
/// its configured start; the result is 0 when `end_m` is not past that start.
pub fn optical_depth(origin_y: f32, dir_y: f32, start_m: f32, end_m: f32, desc: &FogDesc) -> f32 {
    let (t0, t1) = limits(start_m, end_m, desc);
    if t1 <= t0 {
        return 0.0;
    }
    let density = desc.density as f64;
    if density == 0.0 {
        return 0.0;
    }

    let falloff = desc.height_falloff as f64;
    let k = falloff * dir_y as f64;
    let span = t1 - t0;
    let scale = density * (-falloff * (origin_y as f64 - desc.base_height_m as f64)).exp();

    let integral = if (k * span).abs() < NEAR_HORIZONTAL {
        // exp(-k t0) * integral_0^span exp(-k s) ds, with the exponential expanded:
        // span * (1 - k span / 2 + (k span)^2 / 6 - ...), truncated where the tail is < 1e-12.
        let kd = k * span;
        scale * (-k * t0).exp() * span * (1.0 - 0.5 * kd + kd * kd / 6.0)
    } else {
        scale * ((-k * t0).exp() - (-k * t1).exp()) / k
    };

    // The integrand is non-negative; tiny negatives are f64 cancellation only.
    integral.max(0.0) as f32
}

/// Composite Simpson reference for [`optical_depth`], using `steps` intervals.
///
/// `steps` is raised to at least 2 and rounded up to an even number (Simpson needs an even
/// count); the result is computed in `f64` like the analytic function.
pub fn optical_depth_numeric(
    origin_y: f32,
    dir_y: f32,
    start_m: f32,
    end_m: f32,
    desc: &FogDesc,
    steps: usize,
) -> f32 {
    let (t0, t1) = limits(start_m, end_m, desc);
    if t1 <= t0 {
        return 0.0;
    }

    let mut n = steps.max(2);
    if n % 2 == 1 {
        n += 1;
    }
    let h = (t1 - t0) / n as f64;
    let mut sum = density_at(origin_y, dir_y, t0, desc) + density_at(origin_y, dir_y, t1, desc);
    for i in 1..n {
        let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
        sum += weight * density_at(origin_y, dir_y, t0 + i as f64 * h, desc);
    }
    (sum * h / 3.0).max(0.0) as f32
}

/// Transmittance through an optical depth: `exp(-optical_depth)`.
pub fn transmittance(optical_depth: f32) -> f32 {
    (-optical_depth as f64).exp() as f32
}

/// `rho(y(t))` at distance `t`, in `f64`.
fn density_at(origin_y: f32, dir_y: f32, t: f64, desc: &FogDesc) -> f64 {
    let y = origin_y as f64 + dir_y as f64 * t;
    desc.density as f64 * (-(desc.height_falloff as f64) * (y - desc.base_height_m as f64)).exp()
}

/// Integration limits: `t0 = max(start_m, desc.start_m, 0)`, `t1 = end_m`.
fn limits(start_m: f32, end_m: f32, desc: &FogDesc) -> (f64, f64) {
    let t0 = (start_m as f64).max(desc.start_m as f64).max(0.0);
    (t0, end_m as f64)
}
