//! CPU numeric tests of the maths the GI stage nodes moved out of the whole-pass screen path:
//! the a-trous step uniform and dilated tap grid, the B3 kernel product the a-trous weight starts
//! from, the accumulation blend from an invalid (empty) history and the resolve composition.
//! Every expected value is derived by hand in the comment above its assertion; nothing calls the
//! code under test to produce an expectation.
//!
//! Upstream handoff: raa-gi stages 4/11: gi.atrous and gi.resolve stages with the resolve kernel,
//! `crates/raa-gi/src/parts/screen/accumulate_cpu.rs:148 atrous_stage_record` (`atrous_tap_offsets`
//! and `resolve_compose` sit beside it).

use raa_gi::parts::screen::accumulate_cpu::{
    accumulate, alpha, atrous_stage_record, atrous_tap_offsets, atrous_weight, resolve_compose,
};

/// The absolute tolerance of every float comparison; the derived values are exact in f32 but a
/// tolerance keeps the tests about the maths rather than the last bit.
const TOLERANCE: f32 = 1e-6;

fn close(got: f32, want: f32) {
    assert!(
        (got - want).abs() <= TOLERANCE,
        "got {got}, want {want} (tolerance {TOLERANCE})"
    );
}

#[test]
fn atrous_stage_record_packs_the_step_as_an_i32_in_the_first_four_bytes() {
    // The stage uniform is one i32 step in little-endian bytes 0..4 and twelve zero bytes after
    // it. Step 4: 4 = 0x0000_0004 -> [4, 0, 0, 0]; nothing else is written.
    let record = atrous_stage_record(4);
    assert_eq!(&record[..4], &[4, 0, 0, 0]);
    assert_eq!(&record[4..], &[0u8; 12]);

    // Step 0 is clamped to the first a-trous step 1: 1 = 0x0000_0001 -> [1, 0, 0, 0].
    let first = atrous_stage_record(0);
    assert_eq!(&first[..4], &[1, 0, 0, 0]);
    assert_eq!(&first[4..], &[0u8; 12]);
}

#[test]
fn atrous_tap_offsets_are_the_dilated_five_by_five_b3_grid() {
    // The 25 taps are a 5x5 grid indexed row-major, tap i at column i % 5 - 2 and row i / 5 - 2,
    // each axis multiplied by the step. With step 4:
    //   entry 0  = (-2, -2) * 4 = (-8, -8);
    //   entry 12 = (0, 0) * 4    = (0, 0);
    //   entry 24 = (2, 2) * 4    = (8, 8);
    //   entry 13 = (1, 0) * 4    = (4, 0).
    let step4 = atrous_tap_offsets(4);
    assert_eq!(step4.len(), 25);
    assert_eq!(step4[0], [-8, -8]);
    assert_eq!(step4[12], [0, 0]);
    assert_eq!(step4[24], [8, 8]);
    assert_eq!(step4[13], [4, 0]);

    // With step 2 the same grid halves: entry 7 = (0, -1) * 2 = (0, -2).
    let step2 = atrous_tap_offsets(2);
    assert_eq!(step2[7], [0, -2]);
}

#[test]
fn atrous_weight_with_neutral_edges_is_the_two_dimensional_kernel_product() {
    // The 1D B3-spline is (1, 4, 6, 4, 1) / 16 = (0.0625, 0.25, 0.375, 0.25, 0.0625); a 2D tap
    // weight is the product of its two axis weights.
    // Centre tap: 0.375 * 0.375 = 0.140625.
    // Tap (dx, dy) = (2, -1): 0.0625 * 0.25 = 0.015625.
    let centre = 0.375_f32 * 0.375_f32;
    let corner = 0.0625_f32 * 0.25_f32;
    assert_eq!(centre, 0.140625);
    assert_eq!(corner, 0.015625);

    // A neutral edge term is exp(0) = 1 for both the plane and the luminance term and 1^32 = 1 for
    // the normal term, so atrous_weight returns the kernel product it was given. Plane distance 0,
    // normal cosine 1, luminance delta 0; depth 0 and sigma 1 make the neutral arguments concrete.
    close(atrous_weight(centre, 0.0, 0.0, 1.0, 0.0, 1.0), 0.140625);
    close(atrous_weight(corner, 0.0, 0.0, 1.0, 0.0, 1.0), 0.015625);
}

#[test]
fn accumulate_from_an_invalid_empty_history_starts_at_the_current_sample() {
    // The executor reports an invalid history as the empty state (n, value, m1, m2) = (0, 0, 0, 0).
    // The next length is min(0 + 1, 128) = 1, the blend weight alpha = 1 / 1 = 1, so the current
    // sample alone is kept: value 0 + (0.7 - 0) * 1 = 0.7 with n = 1.
    let state = accumulate((0.0, 0.0, 0.0, 0.0), 0.7, 128.0);
    assert_eq!(state.0, 1.0);
    close(state.1, 0.7);
    assert_eq!(alpha(1.0, 128.0), 1.0);

    // A second sample 0.3: n = min(1 + 1, 128) = 2, alpha = 1 / 2 = 0.5, so the value is
    // 0.7 + (0.3 - 0.7) * 0.5 = 0.5, the arithmetic mean of 0.7 and 0.3.
    let state = accumulate(state, 0.3, 128.0);
    assert_eq!(state.0, 2.0);
    close(state.1, 0.5);
    assert_eq!(alpha(2.0, 128.0), 0.5);
}

#[test]
fn resolve_compose_takes_the_occlusion_alpha_and_defaults_it_to_one() {
    // The resolve writes irradiance rgb through unchanged and replaces alpha with the occlusion
    // alpha (the fourth component) when an occlusion image is present: alpha 0.25 replaces 9.0.
    assert_eq!(
        resolve_compose([0.2, 0.4, 0.6, 9.0], Some([0.0, 0.0, 0.0, 0.25])),
        [0.2, 0.4, 0.6, 0.25]
    );

    // Without occlusion the alpha is 1.0, the irradiance rgb unchanged.
    assert_eq!(
        resolve_compose([0.2, 0.4, 0.6, 9.0], None),
        [0.2, 0.4, 0.6, 1.0]
    );
}
