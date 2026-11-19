//! The ReSTIR GI reservoir updates of the `gi.restir.temporal` and `gi.restir.spatial` stages
//! through their CPU twins in `parts::screen::restir_gi_cpu`: the temporal M clamp, the merge of
//! an initial candidate with a previous reservoir, the merged W, the spatial merge over three
//! neighbours, the irradiance estimate and the merge over an initial candidate alone. Every
//! expected value is derived by hand in the test's comments, never by running the code under test.
//!
//! `merge` keeps candidate i when `u * w_sum_i < w_i` (`wrs_accept`, a streaming reservoir), where
//! `u` is one uniform number per candidate. The card's inverse-CDF numbers r (keep the candidate
//! whose cumulative weight first exceeds r * w_sum) select the same index as u = 1 - r, so the
//! tests pass `1 - r` and say so where they do.

use std::f32::consts::PI;

use raa_gi::parts::screen::restir_gi_cpu::{
    Candidate, cosine_pdf, initial_weight, irradiance, merge, merged_weight, target,
    temporal_m_clamp,
};

fn close(got: f32, want: f32, tol: f32) {
    assert!(
        (got - want).abs() <= tol * want.abs().max(1.0),
        "got {got}, want {want}"
    );
}

/// A candidate with target 1, M 1 and the given W, so its merge weight is W.
fn cand(weight: f32) -> Candidate {
    Candidate {
        target: 1.0,
        weight,
        confidence: 1.0,
    }
}

#[test]
fn temporal_m_clamp_caps_the_history_at_m_cap_times_the_current_m() {
    // min(previous, m_cap * current), never below 0.
    close(temporal_m_clamp(50.0, 1.0, 20.0), 20.0, 1e-6); // min(50, 20 * 1)
    close(temporal_m_clamp(5.0, 1.0, 20.0), 5.0, 1e-6); // min(5, 20 * 1)
    close(temporal_m_clamp(50.0, 2.0, 20.0), 40.0, 1e-6); // min(50, 20 * 2)
}

#[test]
fn temporal_merge_of_initial_and_previous_candidates_selects_by_weight_one_and_three() {
    // Candidate 0 (the initial sample) has W = 1, candidate 1 (the previous reservoir) W = 3, both
    // with target 1 and M 1: merge weights 1 and 3, so w_sum = 4 and M = 2.
    // Candidate 0 is kept when u * 1 < 1 (always for u < 1). Candidate 1 is kept when u * 4 < 3,
    // i.e. u < 3/4, and it then replaces candidate 0.
    // r = 0.1 (below 1/4, the first): u = 0.9, 0.9 * 4 = 3.6 is not below 3, so candidate 0 stays.
    let candidates = [cand(1.0), cand(3.0)];
    let (kept, w_sum, m) = merge(&candidates, &[0.9, 0.9]);
    assert_eq!(kept, Some(0));
    close(w_sum, 4.0, 1e-6);
    close(m, 2.0, 1e-6);
    // r = 0.6 (above 1/4, the second): u = 0.4, 0.4 * 4 = 1.6 < 3, so candidate 1 is kept.
    let (kept, w_sum, _) = merge(&candidates, &[0.4, 0.4]);
    assert_eq!(kept, Some(1));
    close(w_sum, 4.0, 1e-6);
}

#[test]
fn merged_weight_is_w_sum_over_z_times_the_kept_target() {
    // Doc comment: W = w_sum / (Z p_hat(z)), w_sum = 4, Z = 2, p_hat(z) = 0.5.
    // Hand value: 4 / (2 * 0.5) = 4.
    close(merged_weight(4.0, 2.0, 0.5), 4.0, 1e-6);
}

#[test]
fn spatial_merge_over_three_neighbours_keeps_the_third() {
    // Neighbours with W 2, 2, 4 (target 1, M 1): w_sum = 2 + 2 + 4 = 8, cumulative 2, 4, 8.
    // The card's r = 0.6 gives r * 8 = 4.8 > 4, so the third neighbour (cumulative 8) is selected.
    // In merge's form the last candidate is kept for u * 8 < 4, i.e. u < 1/2; u = 1 - r = 0.4.
    // Candidate 0: 0.4 * 2 = 0.8 < 2, kept. Candidate 1: 0.4 * 4 = 1.6 < 2, kept.
    // Candidate 2: 0.4 * 8 = 3.2 < 4, kept. So the kept index is 2.
    let candidates = [cand(2.0), cand(2.0), cand(4.0)];
    let (kept, w_sum, m) = merge(&candidates, &[0.4, 0.4, 0.4]);
    assert_eq!(kept, Some(2));
    close(w_sum, 8.0, 1e-6);
    close(m, 3.0, 1e-6);
}

#[test]
fn temporal_irradiance_is_radiance_times_cos_times_weight() {
    // Doc comment: L cos theta W per channel. For L = 1, cos theta = 0.5, W = 2: 1 * 0.5 * 2 = 1.
    for c in irradiance([1.0; 3], 0.5, 2.0) {
        close(c, 1.0, 1e-6);
    }
    // Per channel with L = [0.5, 1, 2]: [0.5 * 0.5 * 2, 1 * 0.5 * 2, 2 * 0.5 * 2] = [0.5, 1, 2].
    let e = irradiance([0.5, 1.0, 2.0], 0.5, 2.0);
    close(e[0], 0.5, 1e-6);
    close(e[1], 1.0, 1e-6);
    close(e[2], 2.0, 1e-6);
}

#[test]
fn merge_over_the_initial_candidate_alone_keeps_it() {
    // Receiver normal up at the origin; the sample at (0, 0, 1) faces down with white radiance.
    // target = luminance(1, 1, 1) * cos 1 = 0.2126 + 0.7152 + 0.0722 = 1.
    // cosine_pdf(0.5) = 0.5 / pi, so initial_weight = (p_hat / pdf) / p_hat = 1 / pdf = 2 pi.
    let up = [0.0, 0.0, 1.0];
    let p_hat = target([0.0; 3], up, [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0; 3]);
    close(p_hat, 1.0, 1e-6);
    let w = initial_weight(p_hat, cosine_pdf(0.5));
    close(w, 2.0 * PI, 1e-5);
    let initial = Candidate {
        target: p_hat,
        weight: w,
        confidence: 1.0,
    };
    // Alone: candidate 0 is kept (0.5 * 2 pi < 2 pi), w_sum = 1 * 2 pi * 1 = 2 pi, M = 1.
    let (kept, w_sum, m) = merge(&[initial], &[0.5]);
    assert_eq!(kept, Some(0));
    close(w_sum, 2.0 * PI, 1e-5);
    close(m, 1.0, 1e-6);
    // An invalid history (M = 0) has w = 1 * 3 * 0 = 0, adds nothing, and the initial stands.
    let invalid = Candidate {
        target: 1.0,
        weight: 3.0,
        confidence: 0.0,
    };
    let (kept, w_sum, _) = merge(&[initial, invalid], &[0.5, 0.5]);
    assert_eq!(kept, Some(0));
    close(w_sum, 2.0 * PI, 1e-5);
}
