//! Hand-derived numeric tests of the `ao.blur` bilateral filter (`denoise.wgsl:23-56`).
//!
//! Every expected value is written here from the fractions or the formula, never by
//! calling a second function under test.

use raa_gtao::blur::{BLUR_TAPS, bilateral, blur_weights, depth_weight};

#[test]
fn weights_are_the_normalised_gaussian() {
    let k = [1.0f32, 3.5, 7.0, 9.0, 7.0, 3.5, 1.0];
    let weights = blur_weights();
    assert_eq!(BLUR_TAPS, 7);
    let mut sum = 0.0f32;
    for i in 0..BLUR_TAPS {
        assert!(
            (weights[i] - k[i] / 32.0).abs() <= 1e-7,
            "weight {i}: {} vs {}",
            weights[i],
            k[i] / 32.0
        );
        sum += weights[i];
    }
    assert!((sum - 1.0).abs() <= 1e-7, "sum {sum} is not 1");
}

#[test]
fn depth_weight_matches_the_formula() {
    assert_eq!(depth_weight(10.0, 10.0), 1.0);
    // 0.02 * max(10, 1e-4) = 0.2, |10.2 - 10| = 0.2, exp(-0.2 / 0.2) = e^-1.
    let e_inv = 0.36787944f32;
    let got = depth_weight(10.0, 10.2);
    assert!((got - e_inv).abs() <= 1e-6, "got {got}, want {e_inv}");
}

#[test]
fn equal_depths_reduce_to_the_gaussian() {
    let depths = [10.0f32; BLUR_TAPS];
    // Tap 3 (offset 0) has weight 9/32; every other value is 0.
    let centre_only = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let got = bilateral(&centre_only, &depths, 10.0);
    assert!((got - 0.28125).abs() <= 1e-7, "got {got}, want 0.28125");

    // Centre 0 leaves (32 - 9) / 32 = 23 / 32.
    let all_but_centre = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
    let got = bilateral(&all_but_centre, &depths, 10.0);
    assert!((got - 0.71875).abs() <= 1e-7, "got {got}, want 0.71875");
}

#[test]
fn depth_edge_downweights_the_far_taps() {
    let depths = [10.2f32, 10.2, 10.2, 10.0, 10.0, 10.0, 10.0];
    let values = [1.0f32, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    // Far taps are 1 + 3.5 + 7 = 11.5 over 32, each times e^-1; near taps are
    // 9 + 7 + 3.5 + 1 = 20.5 over 32 with weight 1.
    let e_inv = std::f32::consts::E.powf(-1.0);
    let numerator = 11.5f32 / 32.0 * e_inv;
    let denominator = numerator + 20.5f32 / 32.0;
    let expected = numerator / denominator;
    let got = bilateral(&values, &depths, 10.0);
    assert!((got - expected).abs() <= 1e-5, "got {got}, want {expected}");
    assert!(
        (expected - 0.171068).abs() <= 1e-5,
        "hand value {expected} differs from 0.171068"
    );
}
