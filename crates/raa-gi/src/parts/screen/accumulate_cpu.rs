//! CPU twins of the screen accumulation (`wgsl/screen/accumulate.wgsl`) and of the
//! variance-guided a-trous passes (`wgsl/screen/atrous.wgsl`): the 1 / n blend weight, the
//! moments, the disocclusion test, the luminance clamp, the variance of the mean, the B3-spline
//! kernel and the propagation of the variance through the squared a-trous weights (Schied et al.,
//! "Spatiotemporal Variance-Guided Filtering", HPG 2017). Each function has the name of its WGSL
//! twin minus the `raa_gi_acc_` / `raa_gi_atrous_` prefix and the same constants. Pure functions;
//! nothing panics.

/// `RAA_GI_ACC_DEPTH_TOLERANCE`: the relative depth a kept history tap may differ by.
pub const DEPTH_TOLERANCE: f32 = 0.04;
/// `RAA_GI_ACC_NORMAL_COS`: the cosine a kept history tap's normal must reach.
pub const NORMAL_COS: f32 = 0.9;
/// `RAA_GI_ACC_CHANGE_SLACK`: the relative slack of the change test.
pub const CHANGE_SLACK: f32 = 0.05;
/// `RAA_GI_ACC_CHANGE_MIN`: the shortest history the change test applies to.
pub const CHANGE_MIN: f32 = 4.0;
/// `RAA_GI_ATROUS_KERNEL`: the 1D B3-spline (1, 4, 6, 4, 1) / 16.
pub const ATROUS_KERNEL: [f32; 5] = [0.0625, 0.25, 0.375, 0.25, 0.0625];
/// `RAA_GI_ATROUS_SIGMA_LUMINANCE`.
pub const ATROUS_SIGMA_LUMINANCE: f32 = 4.0;
/// `RAA_GI_ATROUS_NORMAL_POWER`.
pub const ATROUS_NORMAL_POWER: f32 = 32.0;
/// `RAA_GI_ATROUS_PLANE`.
pub const ATROUS_PLANE: f32 = 0.01;

/// The blend weight once the history holds `n` frames: 1 / n, never below 1 / n_max.
pub fn alpha(n: f32, n_max: f32) -> f32 {
    1.0 / n.min(n_max).max(1.0)
}

/// The next history length: one more frame, capped at n_max.
pub fn next_length(n: f32, n_max: f32) -> f32 {
    (n + 1.0).min(n_max.max(1.0))
}

/// One frame of the accumulation of a scalar signal `x` with its first two moments: the history
/// length, value and moments after the frame (accumulate.wgsl `raa_gi_acc_resolve`, matched
/// pixel with kept history).
pub fn accumulate(state: (f32, f32, f32, f32), x: f32, n_max: f32) -> (f32, f32, f32, f32) {
    let (n, value, m1, m2) = state;
    let n = next_length(n, n_max);
    let a = alpha(n, n_max);
    (
        n,
        value + (x - value) * a,
        m1 + (x - m1) * a,
        m2 + (x * x - m2) * a,
    )
}

/// The variance of the moments, max(m2 - m1^2, 0).
pub fn variance(m1: f32, m2: f32) -> f32 {
    (m2 - m1 * m1).max(0.0)
}

/// Whether a history tap describes the pixel's surface (`raa_gi_acc_tap_matches`).
pub fn tap_matches(
    stored_depth: f32,
    expected_depth: f32,
    current_depth: f32,
    normal_cos: f32,
) -> bool {
    if stored_depth.is_nan() || stored_depth <= 0.0 {
        return false;
    }
    let moved = (stored_depth - expected_depth).abs() <= DEPTH_TOLERANCE * expected_depth;
    let still = (stored_depth - current_depth).abs() <= DEPTH_TOLERANCE * current_depth;
    (moved || still) && normal_cos >= NORMAL_COS
}

/// The share of the history length the lighting change test keeps (`raa_gi_acc_change_keep`):
/// z = |mean - history| / (sigma + slack), sigma^2 = max(spatial_var, temporal_var), slack =
/// CHANGE_SLACK max(|history|, |mean|) + 1e-4; 1 up to z = gamma, 0 from 2 gamma, linear between.
pub fn change_keep(history: f32, mean: f32, spatial_var: f32, temporal_var: f32, gamma: f32) -> f32 {
    let sigma = spatial_var.max(temporal_var).max(0.0).sqrt();
    let slack = CHANGE_SLACK * history.abs().max(mean.abs()) + 1.0e-4;
    let z = (mean - history).abs() / (sigma + slack);
    (2.0 - z / gamma.max(1.0e-3)).clamp(0.0, 1.0)
}

/// The moments after a cut by the change test (accumulate.wgsl): this frame's neighbourhood mean
/// and its spatial variance, (mean, mean^2 + spatial_var).
pub fn cut_moments(mean: f32, spatial_var: f32) -> (f32, f32) {
    (mean, mean * mean + spatial_var.max(0.0))
}

/// The change test's action (`raa_gi_acc_change_action`): (cut, mark). Inside the band (keep 1)
/// neither; a certain change (keep 0) cuts; between, a pixel marked last frame is cut and an
/// unmarked one is marked.
pub fn change_action(keep: f32, marked: bool) -> (bool, bool) {
    if keep >= 1.0 {
        (false, false)
    } else if keep <= 0.0 {
        (true, false)
    } else {
        (marked, !marked)
    }
}

/// The history length after a cut: at most history_min - 1 frames.
pub fn cut_length(n: f32, history_min: f32) -> f32 {
    n.min((history_min - 1.0).max(0.0))
}

/// The variance the a-trous passes steer by: var / n from four frames on, else the larger of the
/// spatial and temporal variances.
pub fn filter_variance(temporal_var: f32, spatial_var: f32, n: f32) -> f32 {
    if n >= 4.0 {
        temporal_var.max(0.0) / n
    } else {
        spatial_var.max(temporal_var).max(0.0)
    }
}

/// The weight of one a-trous tap (`raa_gi_atrous_weight`).
pub fn atrous_weight(
    kernel: f32,
    plane_distance: f32,
    depth: f32,
    normal_cos: f32,
    luminance_delta: f32,
    sigma: f32,
) -> f32 {
    let w_plane = (-plane_distance.abs() / (ATROUS_PLANE * depth + 1.0e-3)).exp();
    let w_normal = normal_cos.max(0.0).powf(ATROUS_NORMAL_POWER);
    let w_luminance = (-luminance_delta.abs() / (ATROUS_SIGMA_LUMINANCE * sigma + 1.0e-4)).exp();
    kernel * w_plane * w_normal * w_luminance
}

/// One a-trous output from its taps (value, variance, weight): the normalised weighted value and
/// the variance filtered by the squared normalised weights, sum(w^2 var) / (sum w)^2.
pub fn atrous_filter(taps: &[(f32, f32, f32)]) -> (f32, f32) {
    let total: f32 = taps.iter().map(|t| t.2).sum();
    if total <= 0.0 {
        return (0.0, 0.0);
    }
    let value = taps.iter().map(|t| t.0 * t.2).sum::<f32>() / total;
    let var = taps.iter().map(|t| t.1 * t.2 * t.2).sum::<f32>() / (total * total);
    (value, var)
}
