//! CPU forms of the bloom shader's downsample, upsample, and composite filters.

/// Linear RGB coefficients used by `bloom_luma` in the bloom shader.
pub const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Weights for the inner box and four corner boxes in `fs_down`.
pub const BOX_WEIGHTS: [f32; 5] = [0.5, 0.125, 0.125, 0.125, 0.125];

/// Normalized 3x3 tent weights, in row-major tap order a through i.
pub const TENT: [f32; 9] = [
    1.0 / 16.0,
    2.0 / 16.0,
    1.0 / 16.0,
    2.0 / 16.0,
    4.0 / 16.0,
    2.0 / 16.0,
    1.0 / 16.0,
    2.0 / 16.0,
    1.0 / 16.0,
];

/// Computes linear luminance from RGB.
pub fn luma(rgb: [f32; 3]) -> f32 {
    rgb[0] * LUMA[0] + rgb[1] * LUMA[1] + rgb[2] * LUMA[2]
}

/// Scales a box's base weight by the Karis luminance factor.
pub fn karis_weight(weight: f32, box_average: [f32; 3]) -> f32 {
    weight / (1.0 + luma(box_average))
}

/// Combines the five box averages with the plain downsample weights.
pub fn down(boxes: &[[f32; 3]; 5]) -> [f32; 3] {
    let mut result = [0.0; 3];
    for (box_average, weight) in boxes.iter().zip(BOX_WEIGHTS) {
        for (channel, value) in result.iter_mut().zip(box_average) {
            *channel += weight * value;
        }
    }
    result
}

/// Combines the first-level boxes with Karis weights normalized by their sum.
pub fn down_first(boxes: &[[f32; 3]; 5]) -> [f32; 3] {
    let weights: [f32; 5] = std::array::from_fn(|i| karis_weight(BOX_WEIGHTS[i], boxes[i]));
    let total_weight: f32 = weights.iter().sum();
    let mut result = [0.0; 3];
    for (box_average, weight) in boxes.iter().zip(weights) {
        for (channel, value) in result.iter_mut().zip(box_average) {
            *channel += weight * value;
        }
    }
    result.map(|value| value / total_weight)
}

/// Applies the normalized 3x3 tent filter.
pub fn up(taps: &[[f32; 3]; 9]) -> [f32; 3] {
    let mut result = [0.0; 3];
    for (tap, weight) in taps.iter().zip(TENT) {
        for (channel, value) in result.iter_mut().zip(tap) {
            *channel += weight * value;
        }
    }
    result
}

/// Mixes HDR colour toward bloom by `strength`, matching WGSL `mix`.
pub fn composite(hdr: [f32; 3], bloom: [f32; 3], strength: f32) -> [f32; 3] {
    std::array::from_fn(|i| hdr[i] + (bloom[i] - hdr[i]) * strength)
}

/// Handoff: `raa-bloom stage passes 1/2: bloom.down and bloom.up levels` calls
/// `half_size` when preparing the `level` output; it halves each dimension with a minimum of one.
pub fn half_size(src: (u32, u32)) -> (u32, u32) {
    ((src.0 / 2).max(1), (src.1 / 2).max(1))
}

#[cfg(test)]
mod tests {
    use crate::shaders::BLOOM_WGSL;

    #[test]
    fn shader_source_pins_filter_constants() {
        assert!(BLOOM_WGSL.contains("0.2126, 0.7152, 0.0722"));
        assert!(BLOOM_WGSL.contains("0.5"));
        assert!(BLOOM_WGSL.contains("0.125"));
        assert!(BLOOM_WGSL.contains("1.0 / 16.0"));
    }
}
