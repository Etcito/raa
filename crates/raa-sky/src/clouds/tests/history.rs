//! Lifecycle rules of `CloudHistory` and the backend's seam behaviour.

use raa_math::Mat4;

use super::super::history::CloudHistory;
use super::test_frame;
use crate::cloud_backend::CloudInvalidation;

const REASONS: [CloudInvalidation; 6] = [
    CloudInvalidation::Resize,
    CloudInvalidation::CameraDiscontinuity,
    CloudInvalidation::Weather,
    CloudInvalidation::Model,
    CloudInvalidation::Enable,
    CloudInvalidation::ViewRemoved,
];

const SIZE: (u32, u32) = (1280, 720);

fn matrix() -> Mat4 {
    test_frame(0, 1, SIZE).camera.view_proj()
}

/// A view with a committed screen march and panorama validity at invocation 1.
fn committed(view: u32) -> CloudHistory {
    let mut h = CloudHistory::new();
    h.prepare(view, SIZE, 1, true, true);
    h.record_screen(view, 1, Some(matrix()));
    h.submitted(view, 1);
    assert_eq!(h.flags(view), (true, true));
    h
}

#[test]
fn panorama_valid_follows_marchable_prepare_and_every_invalidation() {
    let mut h = CloudHistory::new();
    h.prepare(0, SIZE, 1, true, false);
    assert_eq!(h.flags(0), (false, false));
    h.prepare(0, SIZE, 2, true, true);
    assert_eq!(h.flags(0), (true, false));
    for reason in REASONS {
        for target in [Some(0), None] {
            let mut h = committed(0);
            h.prepare(1, SIZE, 1, true, true);
            h.invalidate(target, reason);
            assert_eq!(h.flags(0), (false, false), "{reason:?} {target:?}");
            assert_eq!(h.history(0, true), None, "{reason:?} {target:?}");
            let other = target.is_some();
            assert_eq!(h.flags(1).0, other, "{reason:?} {target:?}");
            h.prepare(0, SIZE, 2, true, true);
            assert_eq!(h.flags(0), (true, false), "{reason:?} {target:?}");
        }
    }
}

#[test]
fn screen_valid_needs_an_acknowledged_march() {
    let mut h = CloudHistory::new();
    h.prepare(0, SIZE, 1, true, false);
    h.record_screen(0, 1, Some(matrix()));
    assert_eq!(h.flags(0), (false, false));
    h.submitted(0, 1);
    assert_eq!(h.flags(0), (false, true));
    h.prepare(0, SIZE, 2, true, false);
    h.submitted(0, 2);
    assert_eq!(h.flags(0), (false, false));
    h.prepare(0, SIZE, 3, true, false);
    h.record_screen(0, 3, Some(matrix()));
    h.submitted(0, 3);
    h.prepare(0, SIZE, 4, true, false);
    h.record_screen(0, 4, None);
    h.submitted(0, 4);
    assert_eq!(h.flags(0), (false, false));
    // Notes and acknowledgements of other invocations are ignored.
    h.record_screen(0, 9, Some(matrix()));
    h.submitted(0, 9);
    assert_eq!(h.flags(0), (false, false));
}

#[test]
fn resize_and_non_marchable_prepare_drop_validity() {
    let mut h = committed(0);
    h.prepare(0, (640, 360), 2, true, true);
    assert_eq!(h.flags(0), (true, false));
    assert_eq!(h.history(0, true), None);
    let mut h = committed(0);
    h.prepare(0, SIZE, 2, false, true);
    assert_eq!(h.flags(0), (false, false));
    assert_eq!(h.history(0, true), None);
    // A pending note does not survive a resize before submission.
    let mut h = CloudHistory::new();
    h.prepare(0, SIZE, 1, true, false);
    h.record_screen(0, 1, Some(matrix()));
    h.prepare(0, (640, 360), 1, true, false);
    h.submitted(0, 1);
    assert_eq!(h.flags(0), (false, false));
}

#[test]
fn history_is_unusable_after_invalidation_or_frame_history_false() {
    let h = committed(0);
    assert_eq!(h.history(0, true), Some(matrix()));
    assert_eq!(h.history(0, false), None);
    let mut moved = committed(0);
    moved.prepare(0, SIZE, 2, true, true);
    assert_eq!(moved.history(0, true), Some(matrix()));
    for reason in REASONS {
        let mut h = committed(0);
        h.invalidate(Some(0), reason);
        assert_eq!(h.history(0, true), None, "{reason:?}");
    }
}

#[test]
fn marchable_is_true_only_for_the_prepared_marchable_invocation() {
    let mut h = CloudHistory::new();
    assert!(!h.marchable(0, 1));
    h.prepare(0, SIZE, 1, true, false);
    assert!(h.marchable(0, 1));
    assert!(!h.marchable(0, 2));
    assert!(!h.marchable(1, 1));
    h.prepare(0, SIZE, 2, false, false);
    assert!(!h.marchable(0, 2));
    h.prepare(0, SIZE, 3, true, false);
    h.invalidate(None, CloudInvalidation::ViewRemoved);
    assert!(!h.marchable(0, 3));
}

#[test]
fn unknown_views_are_neutral() {
    let mut h = CloudHistory::new();
    for view in [0, 3, 4, u32::MAX] {
        h.prepare(view, SIZE, 1, true, true);
        h.record_screen(view, 1, Some(matrix()));
        h.submitted(view, 1);
        h.invalidate(Some(view), CloudInvalidation::Weather);
    }
    for view in [4, u32::MAX] {
        assert_eq!(h.flags(view), (false, false));
        assert_eq!(h.history(view, true), None);
        assert!(!h.marchable(view, 1));
    }
    let h = CloudHistory::new();
    assert_eq!(h.flags(1), (false, false));
    assert_eq!(h.history(1, true), None);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn m2_backend_prepares_and_rejects_like_the_seam_requires() {
    let _gpu = crate::gpu_test_guard();
    use super::super::create;
    use crate::atmosphere::AtmosphereResources;
    use crate::cloud_backend::CloudViews;
    use raa_pipeline::LightingResources;

    let Some(gpu) = raa_core::testing::software_gpu() else {
        raa_core::testing::skip(
            "m2_backend_prepares_and_rejects_like_the_seam_requires",
            "software_gpu returned None; see the software_gpu line above",
        );
        return;
    };
    // No pass under test reads depth contents here: the neutral screen view only
    // stands in for the depth binding, as the seam test does.
    let atmosphere = AtmosphereResources::new(&gpu).expect("atmosphere resources");
    let mut lighting = LightingResources::new(&gpu);
    let mut backend = create(&gpu).expect("m2 backend");
    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let depth = CloudViews::neutral(&gpu).expect("views").screen;
    let valid = test_frame(0, 1, SIZE);
    for _ in 0..2 {
        backend
            .prepare(&gpu, &valid, &atmosphere, &mut lighting)
            .expect("valid prepare");
        backend
            .record_lighting(&gpu, &mut encoder, &valid, &atmosphere)
            .expect("valid lighting record");
        backend
            .record_screen(&gpu, &mut encoder, &valid, &depth, &atmosphere)
            .expect("valid screen record");
    }
    let views = backend.views(0);
    assert!(!views.panorama_valid && !views.screen_valid);
    for (bad, name) in [
        (test_frame(4, 1, SIZE), "out of range"),
        (test_frame(0, 1, (0, 720)), "zero size"),
    ] {
        let error = backend
            .prepare(&gpu, &bad, &atmosphere, &mut lighting)
            .expect_err(name);
        assert_eq!(error.code(), "sky.invalid", "{name}");
        let error = backend
            .record_lighting(&gpu, &mut encoder, &bad, &atmosphere)
            .expect_err(name);
        assert_eq!(error.code(), "sky.invalid", "{name}");
        let error = backend
            .record_screen(&gpu, &mut encoder, &bad, &depth, &atmosphere)
            .expect_err(name);
        assert_eq!(error.code(), "sky.invalid", "{name}");
    }
}

#[test]
fn committed_matrix_is_the_marched_camera_not_the_frame_previous() {
    use super::super::resources::CloudGpu;
    use raa_math::Vec3;

    let mut h = CloudHistory::new();
    let first = test_frame(0, 1, SIZE);
    let mut moved = test_frame(0, 2, SIZE);
    moved.camera.eye = Vec3::new(3.0, 2.0, 6.0);
    // The feature updates previous_view_proj only when the camera changes.
    moved.previous_view_proj = first.camera.view_proj();
    let stopped = test_frame(0, 3, SIZE);
    let stopped = super::super::super::CloudFrame {
        camera: moved.camera,
        previous_view_proj: moved.previous_view_proj,
        ..stopped
    };
    for frame in [first, moved] {
        h.prepare(0, SIZE, frame.invocation, true, true);
        h.record_screen(0, frame.invocation, Some(frame.camera.view_proj()));
        h.submitted(0, frame.invocation);
    }
    h.prepare(0, SIZE, stopped.invocation, true, true);
    let history = h
        .history(0, stopped.history_valid)
        .expect("committed history");
    assert_eq!(history, moved.camera.view_proj(), "the last marched camera");
    assert_ne!(
        history, stopped.previous_view_proj,
        "not the stale frame matrix"
    );
    let record = CloudGpu::from_frame(&stopped, Some(history));
    assert_eq!(record.previous_view_proj, moved.camera.view_proj().cols);
    assert_eq!(record.forward[3], 1.0, "historyUsable");
    assert_eq!(h.history(0, false), None, "the frame refuses history");
}
