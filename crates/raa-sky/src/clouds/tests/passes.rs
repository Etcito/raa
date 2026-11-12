//! Ground cloud shadow: publication decisions and the recorded shadow passes.

use raa_math::Vec3;

use super::super::super::CloudFrame;
use super::super::resources::CloudTargets;
use super::super::{Publication, marchable, needs_publication};
use super::test_frame;

#[test]
fn publication_changes_only_when_map_light_or_view_changes() {
    let frame = test_frame(0, 1, (32, 32));
    let shadowed = Publication::for_frame(&frame, true);
    assert!(shadowed.cloud);
    assert_eq!(shadowed.map[2..], [12_000.0, 1.0]);
    assert_eq!(shadowed.light[3], 0.0);

    assert!(needs_publication(None, Publication::NEUTRAL));
    assert!(needs_publication(None, shadowed));
    assert!(!needs_publication(Some(shadowed), shadowed), "steady frame");
    let steady = test_frame(0, 2, (64, 64));
    assert!(!needs_publication(
        Some(shadowed),
        Publication::for_frame(&steady, true)
    ));

    let mut moved = frame;
    moved.camera.eye = Vec3::new(500.0, 2.0, 6.0);
    assert!(needs_publication(
        Some(shadowed),
        Publication::for_frame(&moved, true)
    ));
    let mut turned = frame;
    turned.dominant.direction = Vec3::new(0.3, -1.0, 0.2);
    assert!(needs_publication(
        Some(shadowed),
        Publication::for_frame(&turned, true)
    ));
    let neutral_view = Publication {
        cloud: false,
        ..shadowed
    };
    assert!(
        needs_publication(Some(shadowed), neutral_view),
        "view identity"
    );
    assert!(needs_publication(Some(shadowed), Publication::NEUTRAL));
    assert!(!needs_publication(
        Some(Publication::NEUTRAL),
        Publication::NEUTRAL
    ));
}

#[test]
fn shadows_off_weather_off_and_legacy_publish_the_neutral_map() {
    let on = test_frame(0, 1, (32, 32));
    let frames = [
        CloudFrame {
            cloud_shadows: false,
            cloud_quality: 1.0,
            ..on
        },
        CloudFrame {
            weather_enabled: false,
            ..on
        },
        CloudFrame {
            physical: false,
            ..on
        },
    ];
    for frame in frames {
        let publication = Publication::for_frame(&frame, marchable(&frame, true));
        assert_eq!(publication, Publication::NEUTRAL);
        assert_eq!(publication.map, [0.0, 0.0, 1.0, 0.0]);
    }
    assert_eq!(
        Publication::for_frame(&on, marchable(&on, false)),
        Publication::NEUTRAL,
        "no sky view"
    );
    assert!(Publication::for_frame(&on, marchable(&on, true)).cloud);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn ground_shadow_records_without_validation_errors() {
    let _gpu = crate::gpu_test_guard();
    use crate::atmosphere::{AtmosphereResources, PhysicalSkyGpu};
    use raa_pipeline::LightingResources;

    let Some(gpu) = raa_core::testing::software_gpu() else {
        raa_core::testing::skip(
            "ground_shadow_records_without_validation_errors",
            "software_gpu returned None; see the software_gpu line above",
        );
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    // Shared inputs: the atmosphere LUTs of view 0 are allocated by prepare_view and
    // filled by their own record below; the noise is generated earlier in the same
    // encoder by the backend's record_lighting. No depth or HDR target is read. The
    // camera (eye (0, 2, 6)) lies inside the 12000 m map, centred by the texel snap.
    let mut atmosphere = AtmosphereResources::new(&gpu).expect("atmosphere");
    atmosphere
        .prepare_view(&gpu, 0, (32, 32), &PhysicalSkyGpu::default())
        .expect("sky view");
    let mut lighting = LightingResources::new(&gpu);
    let mut backend = super::super::create(&gpu).expect("cloud backend");
    let frame = test_frame(0, 1, (32, 32));
    backend
        .prepare(&gpu, &frame, &atmosphere, &mut lighting)
        .expect("prepare");
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    atmosphere
        .record(
            &gpu,
            &mut encoder,
            0,
            lighting.atmospheric_transmission_buffer(),
        )
        .expect("atmosphere record");
    backend
        .record_lighting(&gpu, &mut encoder, &frame, &atmosphere)
        .expect("record");
    gpu.queue().submit([encoder.finish()]);
    backend.submitted(0, 1);
    let error = raa_core::testing::block_on(errors.pop());
    assert!(error.is_none(), "validation error: {error:?}");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn published_transmittance_is_a_two_layer_rgba16float_array() {
    let _gpu = crate::gpu_test_guard();
    use raa_pipeline::LightingResources;

    let Some(gpu) = raa_core::testing::software_gpu() else {
        raa_core::testing::skip(
            "published_transmittance_is_a_two_layer_rgba16float_array",
            "software_gpu returned None; see the software_gpu line above",
        );
        return;
    };
    let targets = CloudTargets::new(&gpu).expect("cloud targets");
    let texture = targets.transmittance().texture();
    assert_eq!(texture.format(), wgpu::TextureFormat::Rgba16Float);
    assert_eq!(texture.dimension(), wgpu::TextureDimension::D2);
    assert_eq!(texture.depth_or_array_layers(), 2);
    assert_eq!((texture.width(), texture.height()), (256, 256));
    assert!(texture.usage().contains(
        wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
    ));
    assert!(*targets.shadow_map().texture() == *texture);
    assert!(*targets.shadow_mean().texture() == *texture);
    LightingResources::new(&gpu)
        .set_cloud_transmittance(
            targets.transmittance().clone(),
            [-6000.0, -6000.0, 12_000.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
        )
        .expect("the lighting resources accept the published view");
}

/// A backend with atmosphere view 0 prepared at (32, 32); `None` without a software GPU.
#[cfg(not(target_arch = "wasm32"))]
fn panorama_setup(
    name: &str,
) -> Option<(
    raa_core::GpuContext,
    crate::atmosphere::AtmosphereResources,
    raa_pipeline::LightingResources,
    Box<dyn crate::cloud_backend::CloudRenderer>,
)> {
    use crate::atmosphere::{AtmosphereResources, PhysicalSkyGpu};
    let Some(gpu) = raa_core::testing::software_gpu() else {
        raa_core::testing::skip(
            name,
            "software_gpu returned None; see the software_gpu line above",
        );
        return None;
    };
    // Shared inputs: the atmosphere LUTs of view 0 come from prepare_view and their own
    // record; noise is generated by record_lighting earlier in the same encoder. No depth,
    // HDR or G-buffer target is read; the panorama covers every sky direction.
    let mut atmosphere = AtmosphereResources::new(&gpu).expect("atmosphere");
    atmosphere
        .prepare_view(&gpu, 0, (32, 32), &PhysicalSkyGpu::default())
        .expect("sky view");
    let lighting = raa_pipeline::LightingResources::new(&gpu);
    let backend = super::super::create(&gpu).expect("cloud backend");
    Some((gpu, atmosphere, lighting, backend))
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn panorama_is_valid_from_a_marchable_prepare_and_neutral_otherwise() {
    let _gpu = crate::gpu_test_guard();
    use crate::cloud_backend::CloudInvalidation;

    let Some((gpu, atmosphere, mut lighting, mut backend)) =
        panorama_setup("panorama_is_valid_from_a_marchable_prepare_and_neutral_otherwise")
    else {
        return;
    };
    let size = |backend: &dyn crate::cloud_backend::CloudRenderer| {
        let texture = backend.views(0).panorama.texture().clone();
        (texture.width(), texture.height())
    };
    let on = test_frame(0, 1, (32, 32));
    backend
        .prepare(&gpu, &on, &atmosphere, &mut lighting)
        .expect("prepare");
    assert!(backend.views(0).panorama_valid);
    assert_eq!(size(&*backend), (192, 108));
    for (invocation, frame) in [
        CloudFrame {
            physical: false,
            ..test_frame(0, 2, (32, 32))
        },
        CloudFrame {
            weather_enabled: false,
            ..test_frame(0, 3, (32, 32))
        },
    ]
    .into_iter()
    .enumerate()
    {
        backend
            .prepare(&gpu, &frame, &atmosphere, &mut lighting)
            .expect("prepare");
        assert!(!backend.views(0).panorama_valid, "frame {invocation}");
        assert_eq!(size(&*backend), (1, 1));
    }
    let reasons = [
        CloudInvalidation::Resize,
        CloudInvalidation::CameraDiscontinuity,
        CloudInvalidation::Weather,
        CloudInvalidation::Model,
        CloudInvalidation::Enable,
        CloudInvalidation::ViewRemoved,
    ];
    for (i, reason) in reasons.into_iter().enumerate() {
        let frame = test_frame(0, 10 + i as u64, (32, 32));
        backend
            .prepare(&gpu, &frame, &atmosphere, &mut lighting)
            .expect("prepare");
        assert!(backend.views(0).panorama_valid, "{reason:?}");
        backend.invalidate(Some(0), reason);
        assert!(!backend.views(0).panorama_valid, "{reason:?}");
        assert_eq!(size(&*backend), (1, 1));
    }
    assert!(!backend.views(7).panorama_valid);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn lighting_records_without_validation_errors_and_keeps_one_panorama_texture() {
    let _gpu = crate::gpu_test_guard();
    let Some((gpu, mut atmosphere, mut lighting, mut backend)) =
        panorama_setup("lighting_records_without_validation_errors_and_keeps_one_panorama_texture")
    else {
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let mut textures = Vec::new();
    for invocation in [1, 2] {
        let frame = test_frame(0, invocation, (32, 32));
        backend
            .prepare(&gpu, &frame, &atmosphere, &mut lighting)
            .expect("prepare");
        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        atmosphere
            .record(
                &gpu,
                &mut encoder,
                0,
                lighting.atmospheric_transmission_buffer(),
            )
            .expect("atmosphere record");
        backend
            .record_lighting(&gpu, &mut encoder, &frame, &atmosphere)
            .expect("record");
        gpu.queue().submit([encoder.finish()]);
        backend.submitted(0, invocation);
        let views = backend.views(0);
        assert!(views.panorama_valid);
        textures.push(views.panorama.texture().clone());
    }
    assert!(textures[0] == textures[1], "one panorama texture");
    let error = raa_core::testing::block_on(errors.pop());
    assert!(error.is_none(), "validation error: {error:?}");
}

/// A Depth32Float target cleared to 1.0 (far, every texel sky) by an empty render pass:
/// no mesh pass writes depth in these tests.
#[cfg(not(target_arch = "wasm32"))]
fn cleared_depth(gpu: &raa_core::GpuContext, size: (u32, u32), samples: u32) -> wgpu::TextureView {
    let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("test depth"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: samples,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("clear test depth"),
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    gpu.queue().submit([encoder.finish()]);
    view
}

/// Inputs of the screen tests: a software GPU, atmosphere views 0 and 1 prepared at
/// (32, 32) (their LUTs are recorded with each frame), lighting resources and a backend.
/// The camera (eye (0, 2, 6)) lies under the cloud layer and inside the 12000 m map.
#[cfg(not(target_arch = "wasm32"))]
struct ScreenSetup {
    gpu: raa_core::GpuContext,
    atmosphere: crate::atmosphere::AtmosphereResources,
    lighting: raa_pipeline::LightingResources,
    backend: super::super::VolumetricClouds,
}

#[cfg(not(target_arch = "wasm32"))]
impl ScreenSetup {
    fn new(name: &str) -> Option<ScreenSetup> {
        use crate::atmosphere::{AtmosphereResources, PhysicalSkyGpu};
        let Some(gpu) = raa_core::testing::software_gpu() else {
            raa_core::testing::skip(
                name,
                "software_gpu returned None; see the software_gpu line above",
            );
            return None;
        };
        let mut atmosphere = AtmosphereResources::new(&gpu).expect("atmosphere");
        for view in [0, 1] {
            atmosphere
                .prepare_view(&gpu, view, (32, 32), &PhysicalSkyGpu::default())
                .expect("sky view");
        }
        let lighting = raa_pipeline::LightingResources::new(&gpu);
        let backend = super::super::VolumetricClouds::new(&gpu).expect("cloud backend");
        Some(ScreenSetup {
            gpu,
            atmosphere,
            lighting,
            backend,
        })
    }

    /// Prepares `frame`, records the atmosphere and cloud lighting, then the screen march
    /// over `depth` when given, and submits; the acknowledgement is left to the caller.
    fn frame(&mut self, frame: &CloudFrame, depth: Option<&wgpu::TextureView>) {
        use crate::cloud_backend::CloudRenderer;
        let gpu = &self.gpu;
        self.backend
            .prepare(gpu, frame, &self.atmosphere, &mut self.lighting)
            .expect("prepare");
        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        self.atmosphere
            .record(
                gpu,
                &mut encoder,
                frame.view,
                self.lighting.atmospheric_transmission_buffer(),
            )
            .expect("atmosphere record");
        self.backend
            .record_lighting(gpu, &mut encoder, frame, &self.atmosphere)
            .expect("record lighting");
        if let Some(depth) = depth {
            self.backend
                .record_screen(gpu, &mut encoder, frame, depth, &self.atmosphere)
                .expect("record screen");
        }
        gpu.queue().submit([encoder.finish()]);
    }

    fn screen(&self, view: u32) -> (bool, (u32, u32)) {
        use crate::cloud_backend::CloudRenderer;
        let views = self.backend.views(view);
        let texture = views.screen.texture();
        (views.screen_valid, (texture.width(), texture.height()))
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn screen_march_becomes_valid_only_after_submitted() {
    let _gpu = crate::gpu_test_guard();
    use crate::cloud_backend::CloudRenderer;
    let Some(mut s) = ScreenSetup::new("screen_march_becomes_valid_only_after_submitted") else {
        return;
    };
    let errors = s
        .gpu
        .device()
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let depth = cleared_depth(&s.gpu, (32, 32), 1);
    let frame = test_frame(0, 1, (32, 32));
    s.frame(&frame, Some(&depth));
    assert!(!s.screen(0).0, "recorded, not acknowledged");
    s.backend.submitted(0, 1);
    assert_eq!(s.screen(0), (true, (16, 16)));
    let error = raa_core::testing::block_on(errors.pop());
    assert!(error.is_none(), "validation error: {error:?}");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn every_invalidation_and_resize_resets_screen_validity_and_history() {
    let _gpu = crate::gpu_test_guard();
    use crate::cloud_backend::{CloudInvalidation, CloudRenderer};
    let Some(mut s) =
        ScreenSetup::new("every_invalidation_and_resize_resets_screen_validity_and_history")
    else {
        return;
    };
    let errors = s
        .gpu
        .device()
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let depth = cleared_depth(&s.gpu, (32, 32), 1);
    let reasons = [
        CloudInvalidation::Resize,
        CloudInvalidation::CameraDiscontinuity,
        CloudInvalidation::Weather,
        CloudInvalidation::Model,
        CloudInvalidation::Enable,
        CloudInvalidation::ViewRemoved,
    ];
    for (i, reason) in reasons.into_iter().enumerate() {
        let invocation = 1 + i as u64;
        s.frame(&test_frame(0, invocation, (32, 32)), Some(&depth));
        s.backend.submitted(0, invocation);
        assert!(s.screen(0).0, "{reason:?}");
        assert!(s.backend.history.history(0, true).is_some(), "{reason:?}");
        s.backend.invalidate(Some(0), reason);
        assert!(!s.screen(0).0, "{reason:?}");
        assert_eq!(s.backend.history.history(0, true), None, "{reason:?}");
    }
    assert_eq!(
        s.screen(0).1,
        (1, 1),
        "a removed view hands out the neutral view"
    );

    s.frame(&test_frame(0, 10, (32, 32)), Some(&depth));
    s.backend.submitted(0, 10);
    let before = s.backend.views(0).screen.texture().clone();
    assert_eq!(s.screen(0), (true, (16, 16)));
    let small = cleared_depth(&s.gpu, (16, 16), 1);
    let resized = test_frame(0, 11, (16, 16));
    s.backend
        .prepare(&s.gpu, &resized, &s.atmosphere, &mut s.lighting)
        .expect("prepare");
    assert_eq!(s.screen(0), (false, (8, 8)), "recreated at the new size");
    assert_eq!(s.backend.history.history(0, true), None);
    assert!(*s.backend.views(0).screen.texture() != before);
    s.frame(&resized, Some(&small));
    s.backend.submitted(0, 11);
    assert_eq!(s.screen(0), (true, (8, 8)));
    let error = raa_core::testing::block_on(errors.pop());
    assert!(error.is_none(), "validation error: {error:?}");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn screen_pass_disabled_frame_drops_screen_validity() {
    let _gpu = crate::gpu_test_guard();
    use crate::cloud_backend::CloudRenderer;
    let Some(mut s) = ScreenSetup::new("screen_pass_disabled_frame_drops_screen_validity") else {
        return;
    };
    let depth = cleared_depth(&s.gpu, (32, 32), 1);
    s.frame(&test_frame(0, 1, (32, 32)), Some(&depth));
    s.backend.submitted(0, 1);
    assert!(s.screen(0).0);
    // Prepare and lighting only: the screen pass is disabled for invocation 2.
    s.frame(&test_frame(0, 2, (32, 32)), None);
    s.backend.submitted(0, 2);
    assert!(!s.screen(0).0);
    assert_eq!(s.backend.history.history(0, true), None);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn multisampled_depth_uses_the_multisampled_screen_pipeline() {
    let _gpu = crate::gpu_test_guard();
    use crate::cloud_backend::CloudRenderer;
    let Some(mut s) = ScreenSetup::new("multisampled_depth_uses_the_multisampled_screen_pipeline")
    else {
        return;
    };
    let errors = s
        .gpu
        .device()
        .push_error_scope(wgpu::ErrorFilter::Validation);
    // The page's default MSAA 4: binding 12 must be texture_depth_multisampled_2d, which
    // only the multisampled layout and pipeline accept without a validation error.
    let depth = cleared_depth(&s.gpu, (32, 32), 4);
    assert_eq!(depth.texture().sample_count(), 4);
    s.frame(&test_frame(0, 1, (32, 32)), Some(&depth));
    s.backend.submitted(0, 1);
    assert!(s.screen(0).0);
    let error = raa_core::testing::block_on(errors.pop());
    assert!(error.is_none(), "validation error: {error:?}");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn two_views_keep_separate_histories() {
    let _gpu = crate::gpu_test_guard();
    use crate::cloud_backend::{CloudInvalidation, CloudRenderer};
    use raa_math::Vec3;
    let Some(mut s) = ScreenSetup::new("two_views_keep_separate_histories") else {
        return;
    };
    let errors = s
        .gpu
        .device()
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let depth = cleared_depth(&s.gpu, (32, 32), 1);
    let first = test_frame(0, 1, (32, 32));
    let mut second = test_frame(1, 1, (32, 32));
    second.camera.eye = Vec3::new(-4.0, 3.0, 5.0);
    s.frame(&first, Some(&depth));
    s.frame(&second, Some(&depth));
    s.backend.submitted(0, 1);
    assert!(
        s.screen(0).0 && !s.screen(1).0,
        "only view 0 was acknowledged"
    );
    assert_eq!(s.backend.history.history(1, true), None);
    s.backend.submitted(1, 1);
    assert!(s.screen(1).0);
    let screen = |view| s.backend.views(view).screen.texture().clone();
    assert!(screen(0) != screen(1), "one target pair per view");
    assert_eq!(
        s.backend.history.history(0, true),
        Some(first.camera.view_proj())
    );
    assert_eq!(
        s.backend.history.history(1, true),
        Some(second.camera.view_proj())
    );
    s.backend
        .invalidate(Some(0), CloudInvalidation::CameraDiscontinuity);
    assert!(!s.screen(0).0 && s.screen(1).0);
    assert_eq!(s.backend.history.history(0, true), None);
    assert!(s.backend.history.history(1, true).is_some());
    let error = raa_core::testing::block_on(errors.pop());
    assert!(error.is_none(), "validation error: {error:?}");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn installed_sky_feature_publishes_cloud_validity_and_drops_it_when_weather_is_off() {
    let _gpu = crate::gpu_test_guard();
    use crate::{
        LegacyCloudFields, SkyDesc, SkyFeature, SkyPreset, SkyReadinessStatus, SkyTransaction,
        WeatherSettings,
    };
    use raa_core::testing::{block_on, skip, software_gpu};
    use raa_core::{FrameGraph, OffscreenTarget};
    use raa_ibl::IblFeature;
    use raa_lit::{LitOptions, LitRenderer};
    use raa_pipeline::Pipeline;
    use raa_scene::Camera;

    let test = "installed_sky_feature_publishes_cloud_validity_and_drops_it_when_weather_is_off";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline = std::rc::Rc::new(std::cell::RefCell::new(Pipeline::new(&gpu, SURFACE, false)));
    let mut lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit");
    let ibl = IblFeature::install(&gpu, &pipeline, &mut lit).expect("ibl");
    let sky = SkyFeature::install(&gpu, &pipeline, &mut lit, ibl).expect("sky");
    sky.set_transaction(SkyTransaction {
        desc: SkyDesc::preset(SkyPreset::Day.as_str()).expect("day"),
        preset: Some(SkyPreset::Day),
        cloud_fields: LegacyCloudFields::NONE,
    })
    .expect("day preset");
    sky.set_view(0, &Camera::default(), true).expect("view 0");
    // The lit prepass clears the standard-Z depth to 1 (no mesh is drawn), so
    // every screen texel is sky; the camera lies under the day preset's cloud layer.
    let mut graph = FrameGraph::new(SURFACE);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().expect("compile");
    let backbuffer = OffscreenTarget::new(&gpu, 32, 32, SURFACE);
    let mut run = |invocation: u64| {
        let commands = graph
            .execute(&gpu, backbuffer.view(), (32, 32))
            .expect("execute");
        gpu.queue().submit([commands]);
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        sky.observe_submission(invocation, 0b0001, true);
        sky.diagnostics()
    };
    run(0);
    let ready = run(1);
    let view = ready.views[0].expect("view 0");
    assert!(view.cloud_panorama_valid, "panorama");
    assert!(view.cloud_screen_valid, "screen");
    assert_eq!(ready.readiness.status, SkyReadinessStatus::Ready);

    let settings = sky.weather().settings;
    sky.set_weather(WeatherSettings {
        enabled: false,
        ..settings
    })
    .expect("weather off");
    let off = run(2);
    let view = off.views[0].expect("view 0");
    assert!(!view.cloud_panorama_valid && !view.cloud_screen_valid);

    let error = block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");
    std::mem::forget(graph);
    std::mem::forget(backbuffer);
    std::mem::forget(lit);
    std::mem::forget(sky);
    std::mem::forget(pipeline);
    std::mem::forget(gpu);
}
