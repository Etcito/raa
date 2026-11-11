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
