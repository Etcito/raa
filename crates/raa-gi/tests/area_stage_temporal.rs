//! CPU numeric tests for the area temporal blend and ReSTIR confidence clamp.
//! Expected values below are derived from the documented formulas, not from the functions under test.

use raa_gi::parts::area_shadow::cpu::{TEMPORAL_ALPHA, temporal_blend};
use raa_gi::parts::area_shadow::restir_cpu::clamp_confidence;

#[test]
fn temporal_blend_uses_history_only_when_valid() {
    // TEMPORAL_ALPHA is 0.1, so a usable history of 1 blended toward current 0 is
    // (1 - TEMPORAL_ALPHA) * 1 + TEMPORAL_ALPHA * 0 = 0.9.
    assert_eq!(temporal_blend(1.0, 0.0, true), 0.9);

    // An invalid history takes the current value directly.
    assert_eq!(temporal_blend(1.0, 0.0, false), 0.0);

    // Three successive usable frames at current 0 multiply the retained history by 0.9 each time.
    let after_three: f32 = (1.0 - TEMPORAL_ALPHA).powi(3);
    assert!((after_three - 0.729).abs() < 1e-6);
    let mut value = 1.0;
    for _ in 0..3 {
        value = temporal_blend(value, 0.0, true);
    }
    assert!((value - after_three).abs() < 1e-6);
}

#[test]
fn clamp_confidence_caps_previous_count_by_current_count() {
    // Formula: max(min(previous, cap * current), 0). With (50, 1, 20),
    // cap * current = 20, min(50, 20) = 20, and max(20, 0) = 20.
    assert_eq!(clamp_confidence(50.0, 1.0, 20.0), 20.0);
}
