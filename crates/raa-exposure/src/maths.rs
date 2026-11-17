//! CPU twins of the S46 exposure maths; the WGSL computes the same.
//! Adapted from solstice@7e465b5 crates/solstice-render/shaders/post/exposure.wgsl, MIT licence, Copyright (c) 2026 SamG-Coder
//! Solstice clamps the average; RAA clamps the exposure (S46).

use crate::{
    AutoExposureDesc, HISTOGRAM_BINS, LOG_LUMINANCE_RANGE, LUMINANCE_FLOOR, MIDDLE_GREY,
    MIN_LOG_LUMINANCE, SETTLED_EV,
};

/// Luminance of linear HDR colour.
pub fn luminance(rgb: [f32; 3]) -> f32 {
    rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
}

/// Histogram bin of a luminance, None below `LUMINANCE_FLOOR`.
pub fn bin_index(luminance: f32) -> Option<u32> {
    if luminance.is_nan() || luminance < LUMINANCE_FLOOR {
        return None;
    }
    let position = (luminance.log2() - MIN_LOG_LUMINANCE) / LOG_LUMINANCE_RANGE;
    Some((position.clamp(0.0, 1.0) * (HISTOGRAM_BINS - 1) as f32) as u32)
}

/// Average luminance over the percentile band, None for an empty histogram.
pub fn band_average(histogram: &[u32; 128], low_percent: f32, high_percent: f32) -> Option<f32> {
    let total: u64 = histogram.iter().map(|&c| u64::from(c)).sum();
    if total == 0 {
        return None;
    }
    let low_cut = total as f32 * low_percent;
    let high_cut = total as f32 * high_percent;
    let mut running = 0.0f32;
    let mut sum = 0.0f32;
    let mut weight = 0.0f32;
    for (i, &count) in histogram.iter().enumerate() {
        let next = running + count as f32;
        let used = (next.min(high_cut) - running.max(low_cut)).max(0.0);
        let log_l =
            MIN_LOG_LUMINANCE + (i as f32 + 0.5) / HISTOGRAM_BINS as f32 * LOG_LUMINANCE_RANGE;
        sum += log_l * used;
        weight += used;
        running = next;
    }
    Some((sum / weight.max(1.0)).exp2())
}

/// Exponential adaptation of the average luminance, faster towards a brighter scene.
pub fn adapt(
    previous: Option<f32>,
    measured: Option<f32>,
    dt_s: f32,
    speed_up: f32,
    speed_down: f32,
) -> f32 {
    let Some(previous) = previous else {
        return measured.unwrap_or(1.0);
    };
    let Some(measured) = measured else {
        return previous;
    };
    let rate = if measured > previous {
        speed_up
    } else {
        speed_down
    };
    let keep = (-dt_s * rate).exp();
    measured + (previous - measured) * keep
}

/// Linear exposure mapping the average to middle grey, clamped to `[2^min_ev, 2^max_ev]`.
pub fn exposure_from_average(average: f32, desc: &AutoExposureDesc) -> f32 {
    let exposure = desc.compensation_ev.exp2() * MIDDLE_GREY / average.max(1e-6);
    exposure.clamp(desc.min_ev.exp2(), desc.max_ev.exp2())
}

/// True when two exposures (EV) differ by less than `SETTLED_EV`.
pub fn settled(previous_ev: f32, current_ev: f32) -> bool {
    (current_ev - previous_ev).abs() < SETTLED_EV
}
