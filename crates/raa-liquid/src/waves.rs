//! Gerstner wave sum shared by the liquid vertex stages, and its CPU twin.
//!
//! # Parameter packing
//!
//! Every WGSL file of this crate reads the same packing from the material record's `p`
//! array (S10 `SurfaceMaterialGpu.p`):
//!
//! - `p[i]`, for `i` in `0..4`, is wave `i`: `(amplitude A in m, wavelength L in m,
//!   direction angle theta in rad measured in XZ from +X towards +Z, steepness Q in
//!   [0, 1])`.
//! - `p[4]` is `(speed of wave 0, speed of wave 1, speed of wave 2, speed of wave 3)` in
//!   m/s.
//! - `p[5..8]` are function-specific (foam, ripple and mud parameters); the twin does not
//!   read them.
//!
//! # One wave
//!
//! With `k = 2 pi / L` and `D = (cos theta, sin theta)`, the phase is
//!
//! `phase = k (D.x x + D.y z - speed t)`.
//!
//! The rest-position height is `A sin(phase)`. [`wave_height`] returns the sum of that
//! height over the [`WAVE_COUNT`] waves. A wave with `L <= 0`, `A == 0`, or a non-finite
//! amplitude, wavelength, direction angle, steepness or speed contributes 0; non-finite
//! `x`, `z` or `t` also return 0. The result is therefore never NaN.
//!
//! # Horizontal Gerstner displacement and analytic normal
//!
//! The vertex stage additionally displaces the rest position horizontally. With
//! `A > 0` and
//!
//! `Qi = clamp(Q / (k A 4), 0, 1)` (`4` is [`WAVE_COUNT`])
//!
//! (the division guard: `A == 0` or `L <= 0` contributes no wave at all), the horizontal
//! displacement is
//!
//! `dxz = D Qi A cos(phase)`.
//!
//! The displaced position is
//!
//! `P = (x + sum dxz.x, sum A sin(phase), z + sum dxz.y)`.
//!
//! Writing `s = sin(phase)` and `c = cos(phase)`, its partial derivatives (GPU Gems 1,
//! chapter 1) are
//!
//! ```text
//! dP/dx = (1 - sum Qi k A D.x^2 s,  sum k A D.x c,  -sum Qi k A D.x D.y s)
//! dP/dz = (   -sum Qi k A D.x D.y s, sum k A D.y c, 1 - sum Qi k A D.y^2 s)
//! ```
//!
//! and the outward normal (the rest normal is `+Y`) is
//!
//! `n = normalize(cross(dP/dz, dP/dx))`.
//!
//! Dropping the terms of second order in the wave slope gives the GPU Gems 1 chapter 1
//! closed form, which the WGSL vertex stages use:
//!
//! `n = normalize(-sum k A D.x c, 1 - sum Qi k A s, -sum k A D.y c)`.
//!
//! All sums run over the [`WAVE_COUNT`] waves; a wave that contributes 0 to the height
//! contributes 0 to the displacement and the normal as well.

use std::f32::consts::TAU;

/// Number of Gerstner waves packed in the material record's `p` array.
pub const WAVE_COUNT: usize = 4;

/// Rest-position height of the four Gerstner waves at `(x, z)` and time `t`, in metres.
///
/// The packing of `params` is the crate contract; see the [module documentation](self).
/// Each wave contributes `A sin(phase)` and a degenerate wave (non-positive wavelength,
/// zero amplitude or any non-finite field) contributes 0. Non-finite inputs return 0.
pub fn wave_height(params: &[[f32; 4]; 8], x: f32, z: f32, t: f32) -> f32 {
    if !x.is_finite() || !z.is_finite() || !t.is_finite() {
        return 0.0;
    }
    let mut height = 0.0;
    for (i, wave) in params.iter().take(WAVE_COUNT).enumerate() {
        let [amplitude, wavelength, theta, steepness] = *wave;
        let speed = params[4][i];
        if !amplitude.is_finite()
            || !wavelength.is_finite()
            || !theta.is_finite()
            || !steepness.is_finite()
            || !speed.is_finite()
        {
            continue;
        }
        if wavelength <= 0.0 || amplitude == 0.0 {
            continue;
        }
        let k = TAU / wavelength;
        if !k.is_finite() {
            continue;
        }
        let (sin_theta, cos_theta) = theta.sin_cos();
        let phase = k * (cos_theta * x + sin_theta * z - speed * t);
        if !phase.is_finite() {
            continue;
        }
        let term = amplitude * phase.sin();
        if term.is_finite() {
            height += term;
        }
    }
    height
}
