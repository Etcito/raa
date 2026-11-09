//! Engine invocation boundaries (M2-FRAME-LIFECYCLE × M2-SUBMISSION-OBSERVATIONS).

#![cfg(not(target_arch = "wasm32"))]

#[allow(dead_code)]
#[path = "../src/convert.rs"]
mod convert;
#[allow(dead_code)]
#[path = "../src/feature_map.rs"]
mod feature_map;
#[allow(dead_code)]
#[path = "../src/state.rs"]
mod state;

use raa_pipeline::SubmissionState;
use state::SceneInvocation;

fn observed(submitted: u8, pending: u8, record_failed: bool) -> SubmissionState {
    SubmissionState {
        submitted_mask: submitted,
        pending_mask: pending,
        record_failed,
    }
}

#[test]
fn once_per_invocation_advancement_and_skipped_rendering() {
    let mut invocation = SceneInvocation::default();
    assert_eq!(invocation.begin(), 0);
    for view in [0, 1, 0, 2, 3] {
        invocation.prepare(view);
    }
    let end = invocation.finish(true, observed(0b0111, 0b1000, false));
    assert_eq!(end, (0b1111, true), "every really submitted view");
    assert_eq!(invocation.begin(), 1, "one advance per invocation");
    // The runtime failed before recording: no scene frame, so keep index and history.
    assert_eq!(
        invocation.finish(false, SubmissionState::default()),
        (0, true)
    );
    assert_eq!(invocation.begin(), 1);
    assert_eq!(invocation.finish(false, observed(0, 0, true)), (0, false));
}

#[test]
fn multi_view_failure_acknowledges_only_submitted_views() {
    let mut invocation = SceneInvocation::default();
    invocation.begin();
    invocation.prepare(0);
    invocation.prepare(1);
    let end = invocation.finish(false, observed(0b0001, 0b0010, true));
    assert_eq!(end, (0b0001, false), "view 1 never reached a submit");
    assert_eq!(invocation.begin(), 1, "the submitted image advances once");
}

#[test]
fn discontinuity_marks_are_one_shot_per_view() {
    let mut invocation = SceneInvocation::default();
    invocation.begin();
    invocation.discontinuity_all();
    assert!(invocation.prepare(0) && invocation.prepare(1));
    assert!(!invocation.prepare(0) && !invocation.prepare(1));
    invocation.discontinuity_all();
    invocation.prepare(0);
    invocation.finish(true, observed(0b0001, 0, false));
    invocation.begin();
    assert!(!invocation.prepare(0));
    assert!(invocation.prepare(2), "unprepared views keep their mark");
}
