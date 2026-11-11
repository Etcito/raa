//! Sky pass installation, typed diagnostics and submitted-lifecycle integration tests.
//!
//! The GPU tests use the repository's software adapter and skip (with an explicit line) when
//! it is unavailable. They never read pixels: the assertions are the records the renderer
//! publishes about its own passes, allocations and submissions.

use raa_core::testing::{block_on, skip, software_gpu};
use raa_core::{FrameGraph, OffscreenTarget};
use raa_ibl::IblFeature;
use raa_lit::{LitOptions, LitRenderer};
use raa_pipeline::{PassRole, Pipeline};
use raa_scene::Camera;
use raa_sky::{
    CloudResetReasons, SkyDiagnostics, SkyFeature, SkyModel, SkyPreset, SkyReadinessStage,
    SkyReadinessStatus, SkyTransaction, Weather,
};

const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SIZE: (u32, u32) = (32, 32);

#[test]
fn diagnostics_defaults_are_pending_install_records() {
    let d = SkyDiagnostics::default();
    assert_eq!(d.invocation, None);
    assert_eq!(d.submitted_invocation, None);
    assert_eq!(d.views, [None; 4]);
    assert_eq!(d.readiness.status, SkyReadinessStatus::Pending);
    assert_eq!(d.readiness.stage, SkyReadinessStage::Install);
    assert_eq!(d.environment.published_generation, None);
    assert_eq!(CloudResetReasons::VIEW_REMOVED.names(), vec!["ViewRemoved"]);
}

#[test]
fn installed_passes_track_invalidations_and_submitted_masks() {
    let test = "installed_passes_track_invalidations_and_submitted_masks";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline = std::rc::Rc::new(std::cell::RefCell::new(Pipeline::new(&gpu, SURFACE, false)));
    let mut lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit");
    let ibl = IblFeature::install(&gpu, &pipeline, &mut lit).expect("ibl");
    let sky = SkyFeature::install(&gpu, &pipeline, &mut lit, ibl.clone()).expect("sky");

    {
        let passes = pipeline.borrow().passes();
        let position = |name: &str| {
            passes
                .iter()
                .position(|p| p.name == name)
                .unwrap_or_else(|| panic!("{name} is installed"))
        };
        let atmosphere = position("sky.atmosphere");
        let lighting = position("sky.cloud-lighting");
        let env = position("sky.env");
        let screen = position("sky.cloud-screen");
        assert!(atmosphere < lighting && lighting < env, "S12 sky ordering");
        assert_eq!(passes[atmosphere].order, 5);
        assert_eq!(passes[lighting].order, 8);
        assert_eq!(passes[env].order, 10);
        assert_eq!(passes[screen].order, 55);
        assert_eq!(passes[screen].role, PassRole::Pass);
        for name in [
            "sky.atmosphere",
            "sky.cloud-lighting",
            "sky.env",
            "sky.cloud-screen",
        ] {
            let pass = passes.iter().find(|p| p.name == name).expect(name);
            assert!(!pass.required, "{name} stays individually controllable");
        }
    }

    let d = sky.diagnostics();
    assert_eq!(d.readiness.stage, SkyReadinessStage::Install);
    assert_eq!(d.views, [None; 4]);

    let mut camera = Camera::default();
    sky.set_view(0, &camera, true).expect("view 0");
    let view = sky.diagnostics().views[0].expect("view 0 record");
    assert_eq!((view.view, view.width, view.height), (0, 0, 0));
    assert_eq!(view.invocation, None);
    assert_eq!(view.submitted_invocation, None);
    assert_eq!(view.sky_view_generation, 0);
    assert_eq!(
        view.reset_reasons,
        CloudResetReasons::NONE,
        "creation is not an invalidation"
    );
    assert_eq!(
        sky.set_view(4, &camera, true).expect_err("index").code(),
        "sky.invalid"
    );

    camera.eye.x = 1.0;
    sky.set_view(0, &camera, false).expect("camera cut");
    assert!(
        sky.diagnostics().views[0]
            .expect("view 0")
            .reset_reasons
            .contains(CloudResetReasons::CAMERA_DISCONTINUITY),
        "an actual camera cut records the discontinuity"
    );
    sky.set_view(0, &camera, true).expect("camera move");

    let mut legacy = sky.get();
    legacy.model = SkyModel::Legacy;
    sky.set(legacy).expect("model switch");
    let mut weather = sky.weather().settings;
    weather.target = Weather::by_name("storm").expect("storm");
    sky.set_weather(weather).expect("weather change");
    pipeline
        .borrow_mut()
        .set_enabled("sky.cloud-lighting", false)
        .expect("disable clouds");
    pipeline
        .borrow_mut()
        .set_enabled("sky.cloud-lighting", true)
        .expect("enable clouds");

    let d = sky.diagnostics();
    let reasons = d.views[0].expect("view 0").reset_reasons;
    assert!(reasons.contains(CloudResetReasons::MODEL));
    assert!(reasons.contains(CloudResetReasons::WEATHER));
    assert!(reasons.contains(CloudResetReasons::ENABLE));
    assert_eq!(d.weather_generation, 1);

    sky.observe_submission(1, 0b0001, false);
    let d = sky.diagnostics();
    assert_eq!(d.submitted_invocation, Some(1));
    assert_eq!(d.readiness.status, SkyReadinessStatus::Pending);
    assert_eq!(d.readiness.stage, SkyReadinessStage::Submission);
    assert!(!d.views[0].expect("view 0").history_valid);

    sky.observe_submission(2, 0b0001, true);
    assert_eq!(
        sky.diagnostics().readiness.status,
        SkyReadinessStatus::Ready,
        "a complete submission of a prepared view becomes ready"
    );

    sky.remove_view(0).expect("remove view");
    assert_eq!(sky.diagnostics().views[0], None);
    sky.remove_view(0)
        .expect("removing an absent view is a no-op");
    assert_eq!(sky.remove_view(4).expect_err("index").code(), "sky.invalid");

    let error = block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");
    std::mem::forget(ibl);
    std::mem::forget(lit);
    std::mem::forget(sky);
    std::mem::forget(pipeline);
    std::mem::forget(gpu);
}

#[test]
fn a_real_frame_acknowledges_its_bake_and_lut_work() {
    let test = "a_real_frame_acknowledges_its_bake_and_lut_work";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline = std::rc::Rc::new(std::cell::RefCell::new(Pipeline::new(&gpu, SURFACE, false)));
    let mut lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit");
    let ibl = IblFeature::install(&gpu, &pipeline, &mut lit).expect("ibl");
    let sky = SkyFeature::install(&gpu, &pipeline, &mut lit, ibl).expect("sky");
    sky.set_transaction(SkyTransaction {
        desc: raa_sky::SkyDesc::preset(SkyPreset::Day.as_str()).expect("day"),
        preset: Some(SkyPreset::Day),
        cloud_fields: raa_sky::LegacyCloudFields::NONE,
    })
    .expect("day preset");
    sky.set_view(0, &Camera::default(), false).expect("view 0");

    let mut graph = FrameGraph::new(SURFACE);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().expect("compile");
    let backbuffer = OffscreenTarget::new(&gpu, SIZE.0, SIZE.1, SURFACE);
    let commands = graph
        .execute(&gpu, backbuffer.view(), SIZE)
        .expect("execute");
    gpu.queue().submit([commands]);
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");

    let pending = sky.diagnostics();
    assert_eq!(
        pending.readiness.status,
        SkyReadinessStatus::Pending,
        "recording alone is not ready"
    );
    assert_eq!(pending.atmosphere_generation, 0);
    assert_eq!(pending.views[0].expect("view 0").sky_view_generation, 0);

    sky.observe_submission(0, 0b0001, true);
    let ready = sky.diagnostics();
    assert_eq!(ready.submitted_invocation, Some(0));
    assert_eq!(
        ready.views[0].expect("view 0").submitted_invocation,
        Some(0)
    );
    assert_eq!(ready.atmosphere_generation, 1, "the LUT work was accepted");
    assert_eq!(ready.views[0].expect("view 0").sky_view_generation, 1);
    assert_eq!(ready.views[0].expect("view 0").width, SIZE.0);
    assert_eq!(ready.views[0].expect("view 0").height, SIZE.1);
    assert_eq!(ready.readiness.status, SkyReadinessStatus::Ready);

    // A resize keeps the view's history invalid until the new size is submitted.
    let resized = (SIZE.0 / 2, SIZE.1 / 2);
    let commands = graph
        .execute(&gpu, backbuffer.view(), resized)
        .expect("resized execute");
    gpu.queue().submit([commands]);
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    let pending = sky.diagnostics();
    let view = pending.views[0].expect("view 0");
    assert!(view.reset_reasons.contains(CloudResetReasons::RESIZE));
    assert_eq!((view.width, view.height), resized);
    assert_eq!(pending.readiness.status, SkyReadinessStatus::Pending);
    sky.observe_submission(1, 0b0001, true);
    let ready = sky.diagnostics();
    assert_eq!(ready.views[0].expect("view 0").sky_view_generation, 2);
    assert_eq!(ready.readiness.status, SkyReadinessStatus::Ready);

    let error = block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");
    std::mem::forget(graph);
    std::mem::forget(backbuffer);
    std::mem::forget(lit);
    std::mem::forget(sky);
    std::mem::forget(pipeline);
    std::mem::forget(gpu);
}
