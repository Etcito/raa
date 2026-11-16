//! The screen accumulation and a-trous maths (accumulate.wgsl, atrous.wgsl) through their CPU
//! twins, each against a value derived by hand in the test: the 1 / n blend is the arithmetic
//! mean, the capped blend an exponential average with a closed form, the moments give the
//! population variance, the disocclusion tolerances, the clamp box, the variance of the mean, and
//! the SVGF variance propagation sum(w^2 var) / (sum w)^2 of the B3-spline kernel.

use raa_gi::parts::screen::accumulate_cpu::{
    ATROUS_KERNEL, accumulate, alpha, atrous_filter, atrous_weight, clamp_luminance,
    filter_variance, tap_matches, variance,
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
fn clamp_box_is_mean_plus_gamma_combined_sigma() {
    // sigma = sqrt(0.25 + 0.75) = 1, gamma 2: the box is [0, 4].
    assert_eq!(clamp_luminance(5.0, 2.0, 0.25, 0.75, 2.0), 4.0);
    assert_eq!(clamp_luminance(-1.0, 2.0, 0.25, 0.75, 2.0), 0.0);
    assert_eq!(clamp_luminance(3.0, 2.0, 0.25, 0.75, 2.0), 3.0);
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
    close(var, 0.074_768_066, 1e-7);
}

#[test]
fn atrous_weight_edge_terms() {
    let kernel = 0.375 * 0.375;
    // Same plane, same normal, same luminance: the kernel weight 0.140625.
    close(atrous_weight(kernel, 0.0, 10.0, 1.0, 0.0, 0.5), 0.140625, 1e-7);
    // A luminance step of sigma_l x sigma = 4 x 0.5 = 2: weight kernel x e^-1 (the 1e-4 guard
    // changes the exponent by 5e-5).
    close(
        atrous_weight(kernel, 0.0, 10.0, 1.0, 2.0, 0.5),
        0.140625 * (-1.0f32).exp(),
        1e-5,
    );
    // A tap 0.1 m off the tangent plane at 10 m (scale 0.01 x 10 + 0.001 = 0.101): e^(-0.1 /
    // 0.101) = e^-0.990099 = e^-0.99 x e^-0.000099 = 0.371577 x 0.999901 = 0.371540.
    close(atrous_weight(1.0, 0.1, 10.0, 1.0, 0.0, 0.5), 0.371_540, 1e-5);
    // A normal at cos 0.9: 0.9^32 = 0.034337.
    close(atrous_weight(1.0, 0.0, 10.0, 0.9, 0.0, 0.5), 0.034_337, 1e-5);
}
