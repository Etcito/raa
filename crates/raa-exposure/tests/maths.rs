//! Hand-derived values of the S46 maths twins.

use raa_exposure::{
    AutoExposureDesc, adapt, band_average, bin_index, exposure_from_average, luminance, settled,
};

fn close(actual: f32, expected: f32) {
    let tol = 1e-5 * expected.abs();
    assert!((actual - expected).abs() <= tol, "{actual} != {expected}");
}

#[test]
fn luminance_weights() {
    close(luminance([1.0, 1.0, 1.0]), 1.0);
    close(luminance([0.0, 1.0, 0.0]), 0.7152);
}

#[test]
fn bin_index_floor_and_range() {
    assert_eq!(bin_index(9e-6), None);
    assert_eq!(bin_index(1e-5), Some(0));
    assert_eq!(bin_index(1.0), Some(56));
    assert_eq!(bin_index(1024.0), Some(127));
    assert_eq!(bin_index(1e6), Some(127));
}

#[test]
fn band_average_two_bins() {
    let mut hist = [0u32; 128];
    hist[0] = 10;
    hist[127] = 10;
    let avg = band_average(&hist, 0.2, 0.7).expect("non-empty");
    close(avg, (-0.7859375f32).exp2());
}

#[test]
fn band_average_cuts_bins_in_half() {
    let mut hist = [0u32; 128];
    hist[10] = 2;
    hist[20] = 2;
    let avg = band_average(&hist, 0.25, 0.75).expect("non-empty");
    close(avg, (-5.8203125f32).exp2());
}

#[test]
fn band_average_empty() {
    assert_eq!(band_average(&[0u32; 128], 0.2, 0.7), None);
}

#[test]
fn adapt_missing_values() {
    assert_eq!(adapt(None, Some(0.5), 0.1, 2.4, 1.1), 0.5);
    assert_eq!(adapt(None, None, 0.1, 2.4, 1.1), 1.0);
    assert_eq!(adapt(Some(0.3), None, 0.1, 2.4, 1.1), 0.3);
}

#[test]
fn adapt_time_limits() {
    close(adapt(Some(0.3), Some(0.9), 0.0, 2.4, 1.1), 0.3);
    assert!((adapt(Some(0.3), Some(0.9), 100.0, 2.4, 1.1) - 0.9).abs() <= 1e-6);
}

#[test]
fn adapt_rates() {
    close(
        adapt(Some(1.0), Some(2.0), 0.5, 2.4, 1.1),
        2.0 - (-1.2f32).exp(),
    );
    close(
        adapt(Some(2.0), Some(1.0), 0.5, 2.4, 1.1),
        1.0 + (-0.55f32).exp(),
    );
}

#[test]
fn exposure_clamps() {
    let desc = AutoExposureDesc::default();
    close(exposure_from_average(1e-6, &desc), 64.0);
    close(exposure_from_average(1e4, &desc), 0.015625);
    let zero = AutoExposureDesc {
        compensation_ev: 0.0,
        ..desc
    };
    close(exposure_from_average(0.18, &zero), 1.0);
}

#[test]
fn settled_threshold() {
    assert!(settled(0.0, 0.009));
    assert!(!settled(0.0, 0.011));
}
