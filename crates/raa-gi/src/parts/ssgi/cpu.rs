//! CPU twins of the screen-space indirect light of `gi.ssgi` (contract `ssgi-cpu`): the visibility
//! bitmask of Therrien, Levesque and Gilet, "Screen Space Indirect Lighting with Visibility
//! Bitmask" (2023), in its cosine-weighted-sector form. The GPU kernels implement these formulas
//! exactly.
//!
//! Angles are radians in the slice plane spanned by the unit view vector `V` (toward the camera)
//! and the slice direction `omega`. `theta` is measured from `V` and is positive toward `+omega`;
//! `gamma` is the projected-normal angle, `atan2(dot(N, omega), dot(N, V))`. Nothing here panics:
//! the masks use u32 operations and `resolve` guards the empty slice list.

use std::f32::consts::{FRAC_PI_2, PI};

/// Sectors per slice. Each sector carries 1/32 of the slice's cosine weight, so a mask is one u32.
pub const SECTORS: u32 = 32;

/// The near field in metres. Samples farther from the pixel than this neither occlude nor light:
/// they belong to the volume and the probes.
pub const RADIUS_METRES: f32 = 2.0;

/// Maximum reconstructed thickness of a sample in metres. Near the receiver, the trace
/// further bounds the estimated thickness by half the sample distance.
pub const THICKNESS_METRES: f32 = 0.5;

/// Maps an angle `theta` from the view vector, measured against the projected-normal angle `gamma`,
/// to a sector coordinate: `(sin(clamp(theta - gamma, -PI/2, PI/2)) + 1) / 2`. Sectors are uniform
/// in `u`, so each of the [`SECTORS`] carries 1/32 of the slice's cosine weight.
pub fn sector_u(theta: f32, gamma: f32) -> f32 {
    let angle = (theta - gamma).clamp(-FRAC_PI_2, FRAC_PI_2);
    (angle.sin() + 1.0) / 2.0
}

/// The bitmask of the sectors covering `[min_u, max_u]`: `start = min(floor(min_u x 32), 31)` and
/// `count = min(ceil((max_u - min_u) x 32), 32 - start)`. It is 0 when `count` is 0, otherwise the
/// `count` bits from bit `start`, `(u32::MAX >> (32 - count)) << start`.
pub fn update_sectors(min_u: f32, max_u: f32) -> u32 {
    let scale = SECTORS as f32;
    let start = ((min_u * scale).floor() as u32).min(SECTORS - 1);
    let count = (((max_u - min_u) * scale).ceil() as u32).min(SECTORS - start);
    if count == 0 {
        return 0;
    }
    (u32::MAX >> (SECTORS - count)) << start
}

/// The sectors one sample occludes in one slice. `delta` is the sample's view position minus the
/// pixel's and `back = delta - view x min(thickness, length(delta) / 2)` is its back face. Each face gives an angle from
/// `view`, `side x acos(clamp(dot(normalize(p), view), -1, 1))`, and a sector coordinate through
/// [`sector_u`]; the mask is the sectors between the two ([`update_sectors`]). A sample at the
/// pixel has no direction and gives 0.
pub fn sample_bits(delta: [f32; 3], view: [f32; 3], thickness: f32, side: f32, gamma: f32) -> u32 {
    let back = sub_scaled(delta, view, thickness.min(0.5 * dot(delta, delta).sqrt()));
    let u_front = sector_u(view_angle(delta, view, side), gamma);
    let u_back = sector_u(view_angle(back, view, side), gamma);
    update_sectors(u_front.min(u_back), u_front.max(u_back))
}

/// Accumulates one sample into one slice. `new = bits & !mask` are the sectors this sample opens
/// for the first time. When `facing`, `irradiance += radiance x popcount(new) / 32`, so a sector
/// contributes once, at the first sample that opens it. It returns the updated mask, `mask | bits`.
pub fn accumulate(
    mask: u32,
    bits: u32,
    radiance: [f32; 3],
    facing: bool,
    irradiance: &mut [f32; 3],
) -> u32 {
    let new = bits & !mask;
    if facing {
        let sectors = new.count_ones() as f32;
        for (out, value) in irradiance.iter_mut().zip(radiance) {
            *out += value * sectors / SECTORS as f32;
        }
    }
    mask | bits
}

/// Updates one slice's lit mask for one sample: `lit | (bits & !mask)` when `lit_sample` (the
/// sample faces the pixel and its radiance is available), otherwise `lit` unchanged. `mask` is the
/// slice's occlusion mask before this sample, so a sector a nearer sample hides is never claimed.
/// Only lit sectors leave the probe term: a sector hidden by a back face or a sample without
/// radiance stays in `V` and keeps the probe irradiance.
pub fn lit_mask(lit: u32, mask: u32, bits: u32, lit_sample: bool) -> u32 {
    if lit_sample {
        lit | (bits & !mask)
    } else {
        lit
    }
}

/// Resolves the slices of one pixel. Each entry is a slice's accumulated irradiance sum and its
/// final lit mask ([`lit_mask`], not the occlusion mask). It returns `E_ss = PI x mean(sum)` and
/// the visibility `V = mean(1 - popcount(lit) / 32)`, both over the slices. With no slices
/// nothing is occluded: `([0, 0, 0], 1)`.
pub fn resolve(slices: &[([f32; 3], u32)]) -> ([f32; 3], f32) {
    if slices.is_empty() {
        return ([0.0; 3], 1.0);
    }
    let count = slices.len() as f32;
    let mut sum = [0.0_f32; 3];
    let mut open = 0.0_f32;
    for (irradiance, mask) in slices {
        for (total, value) in sum.iter_mut().zip(irradiance) {
            *total += value;
        }
        open += 1.0 - mask.count_ones() as f32 / SECTORS as f32;
    }
    (sum.map(|total| PI * (total / count)), open / count)
}

/// Composes screen irradiance with the part of the probe that still has weight `V`.
/// The probe's ambient occlusion is independent of the split between the irradiance sources.
pub fn compose(screen: [f32; 4], probe: [f32; 4]) -> [f32; 4] {
    let [er, eg, eb, v] = screen;
    let [pr, pg, pb, pa] = probe;
    let v = v.clamp(0.0, 1.0);
    [er + v * pr, eg + v * pg, eb + v * pb, pa]
}

/// The weight the new frame takes in the history blend of a pixel whose reprojected history is
/// usable: 0.1, whatever the pixel moved (`_moved_px`). A weight that rose with the motion kept
/// the screen-space GI at two or three frames of history under a moving camera, so it never
/// converged while the camera moved; rejecting unusable history is the reprojection's job.
pub fn temporal_alpha(_moved_px: f32) -> f32 {
    0.1
}

/// `RAA_SSGI_CHANGE_SIGMA` of ssgi/resolve.wgsl: the width of the lighting change test.
pub const CHANGE_SIGMA: f32 = 3.0;

/// The history weight after the change test (`raa_ssgi_change_alpha`): [`temporal_alpha`] (0.1)
/// while |history - mean| / (sqrt(variance) + 0.05 max(|history|, |mean|) + 1e-4) <= CHANGE_SIGMA,
/// rising linearly to 1 at twice that.
pub fn change_alpha(history: f32, mean: f32, variance: f32) -> f32 {
    let slack = 0.05 * history.abs().max(mean.abs()) + 1.0e-4;
    let z = (history - mean).abs() / (variance.max(0.0).sqrt() + slack);
    let cut = (z / CHANGE_SIGMA - 1.0).clamp(0.0, 1.0);
    0.1 + (1.0 - 0.1) * cut
}

/// `side x acos(clamp(dot(normalize(p), view), -1, 1))`. A zero `p` has no direction: its angle is
/// NaN, and `sample_bits` then yields no bits, because `min` and `max` keep only the other face.
fn view_angle(p: [f32; 3], view: [f32; 3], side: f32) -> f32 {
    let cosine = dot(p, view) / dot(p, p).sqrt();
    side * cosine.clamp(-1.0, 1.0).acos()
}

/// Dot product of two 3-vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    let [ax, ay, az] = a;
    let [bx, by, bz] = b;
    ax * bx + ay * by + az * bz
}

/// `a - b x s`, component-wise.
fn sub_scaled(a: [f32; 3], b: [f32; 3], s: f32) -> [f32; 3] {
    let [ax, ay, az] = a;
    let [bx, by, bz] = b;
    [ax - bx * s, ay - by * s, az - bz * s]
}

#[cfg(test)]
mod change_alpha_tests {
    use super::change_alpha;

    #[test]
    fn noise_keeps_the_blend_and_a_light_change_replaces_the_history() {
        // History 1.0, taps mean 1.2 with variance 0.04 (sigma 0.2): slack 0.06 + 1e-4, z = 0.2 /
        // 0.2601 = 0.77 < 3: the fixed 0.1.
        assert!((change_alpha(1.0, 1.2, 0.04) - 0.1).abs() < 1e-6);
        // The light went out: history 1.0, mean 0, sigma 0.05: z = 1 / 0.1001 = 9.99 > 6: weight 1.
        assert_eq!(change_alpha(1.0, 0.0, 0.0025), 1.0);
        // Between: history 1.0, mean 0.5, sigma 0.1: z = 0.5 / 0.1501 = 3.331, cut = 0.110,
        // weight 0.1 + 0.9 x 0.110 = 0.199.
        let z = 0.5f32 / 0.1501;
        assert!((change_alpha(1.0, 0.5, 0.01) - (0.1 + 0.9 * (z / 3.0 - 1.0))).abs() < 1e-5);
    }
}
