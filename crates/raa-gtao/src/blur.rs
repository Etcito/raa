//! CPU form of the `ao.blur` depth-aware bilateral filter (`denoise.wgsl:23-56`).
//!
//! The `ao.blur` stage runs the unchanged `denoise.wgsl` kernels; these functions give the
//! moved filter weights a numeric form so their maths can be checked on the CPU. Tap `i`
//! sits at offset `i - 3` along the filter axis, matching `raa_gtao_denoise`.

/// Number of taps of the separable filter (7).
pub const BLUR_TAPS: usize = 7;

/// The `GAUSS` constant of `denoise.wgsl:25-27`: `[1, 3.5, 7, 9, 7, 3.5, 1] / 32`.
pub fn blur_weights() -> [f32; BLUR_TAPS] {
    [
        1.0 / 32.0,
        3.5 / 32.0,
        7.0 / 32.0,
        9.0 / 32.0,
        7.0 / 32.0,
        3.5 / 32.0,
        1.0 / 32.0,
    ]
}

/// Depth weight `exp(-|tap_z - center_z| / (0.02 * max(center_z, 1e-4)))`
/// (`denoise.wgsl:51`).
pub fn depth_weight(center_z: f32, tap_z: f32) -> f32 {
    let scale = 0.02 * center_z.max(1e-4);
    (-(tap_z - center_z).abs() / scale).exp()
}

/// Bilateral 7-tap filter of `values` at `center_z` (`denoise.wgsl:46-55`):
/// `sum(v_i * w_i) / max(sum(w_i), 1e-6)` with
/// `w_i = blur_weights()[i] * depth_weight(center_z, depths[i])`.
pub fn bilateral(values: &[f32; BLUR_TAPS], depths: &[f32; BLUR_TAPS], center_z: f32) -> f32 {
    let weights = blur_weights();
    let mut sum = 0.0f32;
    let mut wsum = 0.0f32;
    for i in 0..BLUR_TAPS {
        let w = weights[i] * depth_weight(center_z, depths[i]);
        sum += values[i] * w;
        wsum += w;
    }
    sum / wsum.max(1e-6)
}

#[cfg(test)]
mod tests {
    use crate::shaders::DENOISE_WGSL;

    #[test]
    fn shader_gauss_holds_the_cpu_weights_in_order() {
        let literals = [
            "0.03125", "0.109375", "0.21875", "0.28125", "0.21875", "0.109375", "0.03125",
        ];
        let mut rest = DENOISE_WGSL;
        for literal in literals {
            let at = rest
                .find(literal)
                .unwrap_or_else(|| panic!("denoise.wgsl lacks {literal} in order"));
            rest = &rest[at + literal.len()..];
        }
    }
}
