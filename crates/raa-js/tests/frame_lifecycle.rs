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

/// The public renderer and pipeline wired as the Engine wires them, on a software GPU.
/// Every viewport selects itself and prepares through `LitRenderer`, then the real pipeline
/// records the views: a skipped view contributes no bit, a later view's record failure keeps
/// only the view that already reached an intermediate submit, and `begin` clears the
/// observation for the next invocation.
#[test]
fn software_gpu_public_renderer_pipeline_tracks_only_submitted_views() {
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    use raa_core::testing::{skip, software_gpu};
    use raa_core::{FrameGraph, GpuContext, OffscreenTarget, RaaError, wgpu};
    use raa_lit::{LitOptions, LitRenderer};
    use raa_pipeline::{FeaturePass, Pipeline, RecordCtx};
    use raa_scene::{RenderList, Scene};

    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    const SIZE: (u32, u32) = (64, 16);

    /// Fails one view's record: the invocation's final encoder is never submitted.
    struct FailOnRecord {
        calls: Rc<Cell<u32>>,
        fail_at: u32,
    }

    impl FeaturePass for FailOnRecord {
        fn name(&self) -> &str {
            "submission.failing"
        }
        fn order(&self) -> u32 {
            61
        }
        fn record(&mut self, _ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
            let calls = self.calls.get() + 1;
            self.calls.set(calls);
            if calls == self.fail_at {
                return Err(RaaError::Gpu("mock record failure".into()));
            }
            Ok(())
        }
    }

    fn execute(pipeline: &Rc<RefCell<Pipeline>>, gpu: &GpuContext) -> Result<(), RaaError> {
        let target = OffscreenTarget::new(gpu, SIZE.0, SIZE.1, FORMAT);
        let mut graph = FrameGraph::new(FORMAT);
        Pipeline::install(pipeline, &mut graph);
        graph.compile()?;
        let commands = graph.execute(gpu, target.view(), target.size())?;
        gpu.queue().submit([commands]);
        Ok(())
    }

    let Some(gpu) = software_gpu() else {
        skip(
            "public_renderer_pipeline_views",
            "software_gpu returned None",
        );
        return;
    };
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, FORMAT, false)));
    let lit = Rc::new(RefCell::new(
        LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit renderer"),
    ));
    {
        let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, FORMAT, false)));
        let lit = RefCell::new(LitRenderer::new(&gpu, &pipeline, LitOptions::default()).unwrap());
        // The same sky handle receives one runtime delta and every view observes that snapshot.
        let ibl = raa_ibl::IblFeature::install(&gpu, &pipeline, &mut lit.borrow_mut()).unwrap();
        let sky =
            raa_sky::SkyFeature::install(&gpu, &pipeline, &mut lit.borrow_mut(), ibl).unwrap();
        let mut settings = sky.time_of_day().settings;
        settings.enabled = true;
        settings.playing = true;
        settings.speed = 3600.0;
        sky.set_time_of_day(settings).unwrap();
        sky.advance_frame(1, 0.0).unwrap();
        let light = sky.advance_frame(2, 0.5).unwrap();
        let snapshot = sky.time_of_day();
        assert!((snapshot.current.hours - settings.time.hours - 0.5).abs() < 1e-9);
        let mut camera = raa_scene::Camera::default();
        for view in 0..2 {
            camera.eye.y = 10.0 + view as f32 * 100.0;
            sky.set_view(view, &camera, true).unwrap();
            assert_eq!(sky.time_of_day(), snapshot);
            assert_eq!(sky.advance_frame(2, 99.0).unwrap(), light);
        }
        assert_eq!(
            sky.advance_frame(3, f64::NAN).unwrap_err().code(),
            "sky.invalid"
        );
        assert_eq!(sky.time_of_day(), snapshot);
        sky.advance_frame(3, -1.0).unwrap();
        assert_eq!(sky.time_of_day(), snapshot);
        sky.observe_submission(3, 0, true);
        assert_eq!(sky.diagnostics().submitted_invocation, None);
        sky.observe_submission(3, 1, false);
        let diagnostics = sky.diagnostics();
        assert_eq!(diagnostics.submitted_invocation, Some(3));
        assert_eq!(diagnostics.views[0].unwrap().submitted_invocation, Some(3));
        assert_eq!(diagnostics.views[1].unwrap().submitted_invocation, None);
        assert!(!diagnostics.views[0].unwrap().history_valid);
        sky.remove_view(1).unwrap();
        assert!(sky.diagnostics().views[1].is_none());
    }
    let calls = Rc::new(Cell::new(0));
    pipeline
        .borrow_mut()
        .add(Box::new(FailOnRecord {
            calls: Rc::clone(&calls),
            fail_at: 4,
        }))
        .expect("failing pass");
    {
        let lit = Rc::clone(&lit);
        let scene = Rc::new(RefCell::new(Scene::new()));
        let list = Rc::new(RefCell::new(RenderList::default()));
        let tags = Rc::new(RefCell::new(Vec::new()));
        let slots = Rc::new(RefCell::new(Vec::new()));
        pipeline
            .borrow_mut()
            .set_view_prepare(Box::new(move |gpu, view, size| {
                let mut lit = lit.borrow_mut();
                let mut scene = scene.borrow_mut();
                scene.camera.set_viewport(size.0, size.1);
                lit.prepare_scene_view(view, false);
                lit.prepare(
                    gpu,
                    &mut scene,
                    &mut list.borrow_mut(),
                    &mut tags.borrow_mut(),
                    &mut slots.borrow_mut(),
                )
                .map(|_| ())
            }));
    }

    // Invocation 0: view 0 is submitted intermediately, view 1 is skipped (zero area) and
    // view 2 is recorded into the final encoder, pending the runtime's own submit.
    pipeline
        .borrow_mut()
        .set_view_rects(&[[0, 0, 32, 16], [0, 0, 0, 16], [32, 0, 32, 16]]);
    lit.borrow_mut().begin_scene_frame(0);
    pipeline.borrow_mut().begin_scene_submission();
    assert_eq!(
        pipeline.borrow().scene_submission(),
        SubmissionState::default(),
        "begin opens a cleared observation"
    );
    execute(&pipeline, &gpu).expect("first invocation records");
    let first = pipeline.borrow().scene_submission();
    assert_eq!(
        (
            first.submitted_mask,
            first.pending_mask,
            first.record_failed
        ),
        (0b001, 0b100, false),
        "view 1 was skipped and never submitted"
    );
    let acknowledged = first.submitted_mask | first.pending_mask;
    for view in (0..4).filter(|v| acknowledged & (1 << v) != 0) {
        lit.borrow_mut().submitted_scene_view(view);
    }
    lit.borrow_mut().finish_scene_frame(true);

    // Invocation 1: view 0 reaches the intermediate submit, then view 1's record fails, so
    // the observation keeps only view 0 and marks the invocation incomplete.
    pipeline
        .borrow_mut()
        .set_view_rects(&[[0, 0, 32, 16], [32, 0, 32, 16]]);
    lit.borrow_mut().begin_scene_frame(1);
    pipeline.borrow_mut().begin_scene_submission();
    assert_eq!(
        pipeline.borrow().scene_submission(),
        SubmissionState::default()
    );
    assert!(
        execute(&pipeline, &gpu).is_err(),
        "the failing pass aborts the final encoder"
    );
    let second = pipeline.borrow().scene_submission();
    assert_eq!(
        (
            second.submitted_mask,
            second.pending_mask,
            second.record_failed
        ),
        (0b01, 0, true),
        "view 1 never reached a submit"
    );
    lit.borrow_mut().submitted_scene_view(0);
    lit.borrow_mut().finish_scene_frame(false);

    // Invocation 2: every view is skipped. The invocation records nothing, submits nothing
    // and still closes (skipped is a complete invocation for history).
    pipeline.borrow_mut().set_view_rects(&[[0, 0, 0, 0]]);
    lit.borrow_mut().begin_scene_frame(2);
    pipeline.borrow_mut().begin_scene_submission();
    execute(&pipeline, &gpu).expect("skipped invocation records");
    assert_eq!(
        pipeline.borrow().scene_submission(),
        SubmissionState::default(),
        "no skipped view contributes a bit"
    );
    lit.borrow_mut().finish_scene_frame(true);
}

#[test]
fn sky_identity_survives_skipped_failed_and_partial_submissions() {
    let mut id = 0;
    let mut history = SceneInvocation::default();
    for (expected, ok, submission) in [
        (1, true, observed(0, 0, false)),
        (2, false, observed(0, 0, true)),
        (3, false, observed(1, 2, true)),
        (4, true, observed(0, 3, false)),
    ] {
        state::next_sky_invocation(&mut id).unwrap();
        assert_eq!(id, expected);
        history.begin();
        history.finish(ok, submission);
    }
    assert_eq!(history.begin(), 2, "submission history remains independent");
    id = u64::MAX;
    assert_eq!(
        state::next_sky_invocation(&mut id).unwrap_err().code(),
        "sky.invalid"
    );
    assert_eq!(id, u64::MAX);
}

#[test]
fn sky_inspection_does_not_consume_lit_discontinuity() {
    let mut invocation = SceneInvocation::default();
    invocation.discontinuity(2);
    assert!(invocation.is_discontinuous(2));
    assert!(invocation.is_discontinuous(2));
    assert!(!invocation.is_discontinuous(1));
    assert!(invocation.prepare(2));
    assert!(!invocation.is_discontinuous(2));
}
