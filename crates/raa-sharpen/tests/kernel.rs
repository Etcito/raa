//! Hand-derived values of the S47 kernel twins and the `SharpenDesc` validation.

use raa_core::RaaError;
use raa_sharpen::{SharpenDesc, compress, expand, sharpen_texel};

fn rel(actual: f32, expected: f32, tol: f32) {
    assert!(
        (actual - expected).abs() <= tol * expected.abs().max(1.0),
        "{actual} vs {expected}"
    );
}

const EDGE_C: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const EDGE_N: [f32; 3] = [0.25, 0.0, 0.0];
const EDGE_W: [f32; 3] = [3.0, 0.0, 0.0];

#[test]
fn flat_neighbourhood_returns_the_centre() {
    let t = [2.0, 1.0, 0.5];
    let out = sharpen_texel([2.0, 1.0, 0.5, 0.25], t, t, t, t, 0.9);
    for (a, b) in out.iter().zip([2.0, 1.0, 0.5, 0.25]) {
        rel(*a, b, 1e-5);
    }
}

#[test]
fn step_edge_gives_the_hand_derived_weight_and_stays_in_range() {
    let out = sharpen_texel(EDGE_C, EDGE_N, EDGE_N, EDGE_W, EDGE_W, 1.0);
    assert!((out[0] - 1.07296).abs() < 1e-4, "{}", out[0]);
    assert_eq!((out[1], out[2], out[3]), (0.0, 0.0, 1.0));
    assert!(out[0] > expand([0.2, 0.0, 0.0])[0] && out[0] < expand([0.75, 0.0, 0.0])[0]);
    assert!(out[0] > 0.25 && out[0] < 3.0);
}

#[test]
fn zero_sharpness_returns_the_centre() {
    let out = sharpen_texel(EDGE_C, EDGE_N, EDGE_N, EDGE_W, EDGE_W, 0.0);
    for (a, b) in out.iter().zip(EDGE_C) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }
}

#[test]
fn a_value_above_one_survives_the_round_trip() {
    let c = compress([4.0, 2.0, 0.5]);
    for (a, b) in c.iter().zip([0.8, 0.4, 0.1]) {
        rel(*a, b, 1e-5);
    }
    for (a, b) in expand(c).iter().zip([4.0, 2.0, 0.5]) {
        rel(*a, b, 1e-5);
    }
}

#[test]
fn validate_refuses_out_of_range_sharpness() {
    for v in [-0.1, 1.1, f32::NAN] {
        let desc = SharpenDesc {
            sharpness: v,
            ..SharpenDesc::default()
        };
        let err = desc.validate().unwrap_err();
        assert_eq!(err.code(), "post.invalid");
        match err {
            RaaError::FrameGraph { message, .. } => {
                assert!(
                    message.contains("`sharpness` must be in [0, 1]"),
                    "{message}"
                )
            }
            other => panic!("unexpected {other:?}"),
        }
    }
    let d = SharpenDesc::default();
    assert_eq!((d.enabled, d.sharpness), (false, 0.9));
    assert!(d.validate().is_ok());
}
