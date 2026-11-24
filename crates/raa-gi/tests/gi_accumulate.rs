//! The screen accumulation and a-trous maths (accumulate.wgsl, atrous.wgsl) through their CPU
//! twins, each against a value derived by hand in the test: the 1 / n blend is the arithmetic
//! mean, the capped blend an exponential average with a closed form, the moments give the
//! population variance, the disocclusion tolerances, the clamp box, the variance of the mean, and
//! the SVGF variance propagation sum(w^2 var) / (sum w)^2 of the B3-spline kernel.

use raa_gi::parts::screen::accumulate_cpu::{
    ATROUS_KERNEL, accumulate, alpha, atrous_filter, atrous_weight, change_action, change_keep,
    cut_length, cut_moments, filter_variance, tap_matches, variance,
};

fn close(got: f32, want: f32, tol: f32) {
    assert!((got - want).abs() <= tol, "got {got}, want {want}");
}

#[test]
fn one_over_n_blend_is_the_arithmetic_mean_with_its_moments() {
    // Inputs 1, 2, 3, 4 from an empty history: mean (1 + 2 + 3 + 4) / 4 = 2.5, second moment
    // (1 + 4 + 9 + 16) / 4 = 7.5, population variance 7.5 - 2.5^2 = 1.25.
    let mut state = (0.0, 0.0, 0.0, 0.0);
    for x in [1.0, 2.0, 3.0, 4.0] {
        state = accumulate(state, x, 64.0);
    }
    let (n, value, m1, m2) = state;
    assert_eq!(n, 4.0);
    close(value, 2.5, 1e-6);
    close(m1, 2.5, 1e-6);
    close(m2, 7.5, 1e-6);
    close(variance(m1, m2), 1.25, 1e-6);
}

#[test]
fn capped_blend_is_an_exponential_average() {
    // n_max = 4: alpha never drops below 1 / 4. After a long run at 0 the input steps to 1;
    // k frames later the value is 1 - (3 / 4)^k: k = 3 gives 1 - 27 / 64 = 0.578125.
    let mut state = (0.0, 0.0, 0.0, 0.0);
    for _ in 0..20 {
        state = accumulate(state, 0.0, 4.0);
    }
    assert_eq!(state.0, 4.0);
    for _ in 0..3 {
        state = accumulate(state, 1.0, 4.0);
    }
    close(state.1, 0.578125, 1e-6);
    // The weight itself: 1 / n below the cap, 1 / n_max at and above it.
    assert_eq!(alpha(2.0, 64.0), 0.5);
    assert_eq!(alpha(100.0, 64.0), 1.0 / 64.0);
    assert_eq!(alpha(0.0, 64.0), 1.0);
}

#[test]
fn disocclusion_test_keeps_the_same_surface_only() {
    // 3 % off the expected depth: kept (tolerance 4 %).
    assert!(tap_matches(10.0, 10.3, 12.0, 1.0));
    // 5 % off the expected depth and 6 % off the current one: another surface.
    assert!(!tap_matches(10.0, 10.5, 10.6, 1.0));
    // A still camera over a moving object: the expected depth is far off but the current one
    // matches within 2 %.
    assert!(tap_matches(10.0, 20.0, 10.2, 1.0));
    // A normal turned by 31.8 degrees (cos 0.85 < 0.9): rejected.
    assert!(!tap_matches(10.0, 10.0, 10.0, 0.85));
    // No surface stored.
    assert!(!tap_matches(0.0, 10.0, 10.0, 1.0));
}

#[test]
fn change_test_keeps_noise_and_cuts_a_lighting_change() {
    // History 10, current mean 11, spatial variance 1, temporal 4: sigma = 2, slack = 0.05 * 11 +
    // 1e-4 = 0.5501, z = 1 / 2.5501 = 0.392 < gamma 2.5: kept whole.
    assert_eq!(change_keep(10.0, 11.0, 1.0, 4.0, 2.5), 1.0);
    // The light went out: history 10, mean 0, sigma 2, slack 0.5001, z = 10 / 2.5001 = 4.0
    // (between 2.5 and 5): keep = 2 - 4.0 / 2.5 = 0.4 (to 1e-3).
    assert!((change_keep(10.0, 0.0, 1.0, 4.0, 2.5) - (2.0 - 10.0 / 2.5001 / 2.5)).abs() < 1e-5);
    // A larger jump (history 10, mean 0, noise sigma 1): z = 10 / 1.5001 = 6.67 > 5: nothing kept.
    assert_eq!(change_keep(10.0, 0.0, 1.0, 0.0, 2.5), 0.0);
    assert_eq!(change_action(1.0, true, true), (false, false));
    assert_eq!(change_action(0.0, false, false), (true, false));
    assert_eq!(change_action(0.0, false, true), (false, true));
    assert_eq!(change_action(0.0, true, true), (true, false));
    assert_eq!(change_action(0.4, false, false), (false, true));
    assert_eq!(change_action(0.4, true, false), (true, false));
    // A cut keeps historyMin - 1 frames: 0 (the next frame, n = 1, is the new light alone), or 3.
    assert_eq!(cut_length(64.0, 1.0), 0.0);
    assert_eq!(cut_length(64.0, 4.0), 3.0);
}

#[test]
fn a_still_noisy_signal_converges_without_resets() {
    // A still pixel whose frames alternate 9 and 11 (mean 10, per-frame sigma 1): with the change
    // test at gamma 2.5 no frame from the fourth on is ever cut, the history length grows to n_max = 32, and once the
    // history holds an even number of frames it is exactly their mean, 10.
    let (mut n, mut value, mut m1, mut m2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for frame in 0..32 {
        let x = if frame % 2 == 0 { 9.0 } else { 11.0 };
        if n >= 4.0 {
            let keep = change_keep(value, x, 0.0, (m2 - m1 * m1).max(0.0), 2.5);
            assert_eq!(keep, 1.0, "frame {frame} was cut");
        }
        (n, value, m1, m2) = accumulate((n, value, m1, m2), x, 32.0);
    }
    assert_eq!(n, 32.0);
    assert!((value - 10.0).abs() < 1e-4);
}

#[test]
fn filter_variance_is_the_variance_of_the_mean() {
    // Eight frames of variance 0.8: the mean's variance is 0.8 / 8 = 0.1.
    close(filter_variance(0.8, 0.3, 8.0), 0.1, 1e-7);
    // Before four frames the larger of the spatial and temporal variances.
    close(filter_variance(0.8, 0.3, 2.0), 0.8, 1e-7);
    close(filter_variance(0.1, 0.3, 1.0), 0.3, 1e-7);
}

#[test]
fn atrous_variance_propagates_through_squared_weights() {
    // The 1D B3 kernel sums to 1; its squares sum to (1 + 16 + 36 + 16 + 1) / 256 = 70 / 256.
    // With every edge term 1, the 25 weights are h_i h_j, so a uniform variance v becomes
    // v (70 / 256)^2 = 0.07476806640625 v.
    let sum: f32 = ATROUS_KERNEL.iter().sum();
    close(sum, 1.0, 1e-7);
    let mut taps = Vec::new();
    for hy in ATROUS_KERNEL {
        for hx in ATROUS_KERNEL {
            taps.push((3.0, 1.0, hx * hy));
        }
    }
    let (value, var) = atrous_filter(&taps);
    close(value, 3.0, 1e-6);
    close(var, 0.074_768_07, 1e-7);
}

#[test]
fn atrous_weight_edge_terms() {
    let kernel = 0.375 * 0.375;
    // Same plane, same normal, same luminance: the kernel weight 0.140625.
    close(
        atrous_weight(kernel, 0.0, 10.0, 1.0, 0.0, 0.5),
        0.140625,
        1e-7,
    );
    // A luminance step of sigma_l x sigma = 4 x 0.5 = 2: weight kernel x e^-1 (the 1e-4 guard
    // changes the exponent by 5e-5).
    close(
        atrous_weight(kernel, 0.0, 10.0, 1.0, 2.0, 0.5),
        0.140625 * (-1.0f32).exp(),
        1e-5,
    );
    // A tap 0.1 m off the tangent plane at 10 m (scale 0.01 x 10 + 0.001 = 0.101): e^(-0.1 /
    // 0.101) = e^-0.990099 = e^-0.99 x e^-0.000099 = 0.371577 x 0.999901 = 0.371540.
    close(
        atrous_weight(1.0, 0.1, 10.0, 1.0, 0.0, 0.5),
        0.371_540,
        1e-5,
    );
    // A normal at cos 0.9: 0.9^32 = 0.034337.
    close(
        atrous_weight(1.0, 0.0, 10.0, 0.9, 0.0, 0.5),
        0.034_337,
        1e-5,
    );
}

/// One accumulation frame with the change test (accumulate.wgsl raa_gi_acc_resolve, matched
/// pixel, kept history): state (n, value, m1, m2), this frame's value x and spatial variance.
fn frame_with_test(
    state: (f32, f32, f32, f32, i8),
    x: f32,
    spatial_var: f32,
) -> (f32, f32, f32, f32, i8) {
    let (mut n, value, mut m1, mut m2, marked) = state;
    let keep = change_keep(value, x, spatial_var, (m2 - m1 * m1).max(0.0), 2.5);
    let direction = if x > value { 1 } else { -1 };
    let (cut, mark) = if n >= 4.0 {
        change_action(keep, marked == direction, direction > 0)
    } else {
        (false, false)
    };
    if mark && keep <= 0.0 {
        return (n, value, m1, m2, direction);
    }
    if cut {
        n = cut_length(n, 1.0);
        (m1, m2) = cut_moments(x, spatial_var);
    }
    let (n, value, m1, m2) = accumulate((n, value, m1, m2), x, 64.0);
    (n, value, m1, m2, if mark { direction } else { 0 })
}

#[test]
fn a_light_that_goes_out_leaves_the_history_at_once() {
    // Converged at 10 with per-frame noise sigma 1 (m2 = 101), 64 frames. The light goes out:
    // this frame reads 0 with spatial sigma 1. z = 10 / (1 + 0.5001) = 6.67 > 2 x 2.5: nothing
    // kept, so the next history is this frame alone, exactly 0.
    let state = frame_with_test((64.0, 10.0, 10.0, 101.0, 0), 0.0, 1.0);
    assert_eq!((state.0, state.1), (1.0, 0.0));
    // A partial change, 10 -> 5.5 (z = 4.5 / 1.5001 = 3.0, in the band): the first frame only
    // marks the pixel (history 10 - 4.5 / 64 = 9.9296875), the second, out again (z = 2.7),
    // cuts it: the history is the new light, 5.5, after two frames.
    let first = frame_with_test((63.0, 10.0, 10.0, 101.0, 0), 5.5, 1.0);
    assert_eq!((first.0, first.4), (64.0, -1));
    assert!((first.1 - 9.929_687_5).abs() < 1e-5);
    let second = frame_with_test(first, 5.5, 1.0);
    assert_eq!((second.0, second.1), (1.0, 5.5));
    // One noisy frame in the band (5.5) between in-band frames (10): marked, then cleared, kept.
    let noisy = frame_with_test((63.0, 10.0, 10.0, 101.0, 0), 5.5, 1.0);
    let back = frame_with_test(noisy, 10.0, 1.0);
    assert_eq!((back.0, back.4), (64.0, 0));
}

#[test]
fn a_bright_probe_sample_needs_confirmation_before_replacing_history() {
    let converged = (64.0, 10.0, 10.0, 101.0, 0);
    let spike = frame_with_test(converged, 40.0, 1.0);
    assert_eq!(spike, (64.0, 10.0, 10.0, 101.0, 1));

    // A single hit on a bright emitter preserves the converged result.
    let recovered = frame_with_test(spike, 10.0, 1.0);
    assert_eq!((recovered.0, recovered.1, recovered.4), (64.0, 10.0, 0));

    // A sustained change is visible on its second frame.
    let sustained = frame_with_test(spike, 40.0, 1.0);
    assert_eq!((sustained.0, sustained.1, sustained.4), (1.0, 40.0, 0));

    // A preceding dark departure cannot confirm a bright one.
    let dark = frame_with_test(converged, 5.5, 1.0);
    assert_eq!(dark.4, -1);
    let opposite = frame_with_test(dark, 40.0, 1.0);
    assert_eq!((opposite.0, opposite.1, opposite.4), (64.0, dark.1, 1));
}
