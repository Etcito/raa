//! CPU numeric tests for SSGI temporal history weighting and resolve composition.
//! Expected values are hand-derived from the documented equations.

use raa_gi::parts::ssgi::cpu::{change_alpha, compose};

const TOLERANCE: f32 = 1e-3;

fn close(got: f32, expected: f32) {
    assert!(
        (got - expected).abs() <= TOLERANCE,
        "got {got}, expected {expected} (tolerance {TOLERANCE})"
    );
}

#[test]
fn change_alpha_keeps_history_within_noise_and_rejects_large_changes() {
    // Equal history and mean give z = 0, within CHANGE_SIGMA, so the usable-history
    // weight is temporal_alpha = 0.1.
    assert_eq!(change_alpha(0.5, 0.5, 0.0), 0.1);

    // With history 10 and mean 0: slack = 0.05 * 10 + 1e-4 = 0.5001;
    // z = 10 / 0.5001 ~= 19.996, so z / 3 - 1 ~= 5.67 clamps to 1.
    // Therefore alpha = 0.1 + 0.9 * 1 = 1: the current sample stands alone.
    assert_eq!(change_alpha(10.0, 0.0, 0.0), 1.0);

    // With variance 1, slack = 0.05 * 4.5 + 1e-4 = 0.2251 and denominator =
    // 1 + 0.2251 = 1.2251. Thus z ~= 3.67317, cut ~= 0.22439,
    // and alpha ~= 0.1 + 0.9 * 0.22439 = 0.30195.
    close(change_alpha(4.5, 0.0, 1.0), 0.30195);
}

#[test]
fn compose_adds_probe_irradiance_through_screen_visibility() {
    // E_ss = 0.2 and V = 0.5, so each output channel is 0.2 + 0.5 * 0.4 = 0.4.
    // Alpha is min(V, probe alpha) = min(0.5, 0.8) = 0.5.
    assert_eq!(
        compose([0.2, 0.2, 0.2, 0.5], [0.4, 0.4, 0.4, 0.8]),
        [0.4, 0.4, 0.4, 0.5]
    );
}
