//! The M1 cloud backend registry and the neutral cloud views (S25).
//!
//! These tests read the public seam only: the build-time M2 discovery decision, the
//! registry names for both decisions, the copied frame snapshot, every invalidation
//! reason and the neutral view descriptors. They never write an M2 source file.

use raa_math::Mat4;
use raa_scene::{Camera, DirectionalLight};
use raa_sky::{
    CLOUD_SCREEN_UNKNOWN_ALPHA, CloudFrame, CloudInvalidation, CloudViews, M2_CLOUD_SOURCE_PRESENT,
    TimeOfDayState, Weather, cloud_backend_names, registered_cloud_backend_names,
};

/// The declared M2 producer file, relative to this package.
const M2_SOURCE: &str = "src/clouds/mod.rs";

#[test]
fn registry_names_follow_the_m2_discovery_decision() {
    assert_eq!(cloud_backend_names(false), vec!["neutral"]);
    assert_eq!(cloud_backend_names(true), vec!["neutral", "m2"]);
    assert_eq!(
        registered_cloud_backend_names(),
        cloud_backend_names(M2_CLOUD_SOURCE_PRESENT),
        "the registered set follows the build-time discovery"
    );
}

#[test]
fn declared_m2_presence_matches_the_manifest() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(M2_SOURCE);
    assert_eq!(
        M2_CLOUD_SOURCE_PRESENT,
        path.is_file(),
        "build.rs must register the declared M2 source exactly when it exists"
    );
}

#[test]
fn cloud_frame_carries_the_copied_snapshot() {
    let frame = CloudFrame {
        invocation: 9,
        view: 2,
        size: (1920, 1080),
        camera: Camera::default(),
        previous_view_proj: Mat4::IDENTITY,
        history_valid: false,
        weather: Weather::default(),
        sky: TimeOfDayState::default().sky,
        dominant: DirectionalLight::default(),
        animation_seconds: 12.5,
        wind_offset: [0.0; 2],
        wind_velocity: [0.0; 2],
        evolution: 0.0,
        weather_enabled: true,
        cloud_shadows: true,
        cloud_samples: 56,
        cloud_scale: 0.5,
        cloud_dither: true,
        cloud_noise_size: 1,
        cloud_dome: false,
        cloud_dome_size: 1024,
        animated: false,
        signature: 0,
        physical: true,
    };
    assert_eq!(frame.invocation, 9);
    assert_eq!(frame.view, 2);
    assert_eq!(frame.size, (1920, 1080));
    assert!(!frame.history_valid);
    assert_eq!(frame.animation_seconds, 12.5);
    assert_eq!(frame.dominant, DirectionalLight::default());
    assert_eq!(frame.wind_offset, [0.0; 2]);
    assert_eq!(frame.wind_velocity, [0.0; 2]);
    assert_eq!(frame.evolution, 0.0);
    assert!(frame.weather_enabled && frame.cloud_shadows && frame.physical);
    // `CloudFrame` is a copied snapshot, so a copy leaves the original intact.
    let copy = frame;
    assert_eq!(copy, frame);
}

#[test]
fn cloud_invalidation_names_every_reason() {
    let reasons = [
        CloudInvalidation::Resize,
        CloudInvalidation::CameraDiscontinuity,
        CloudInvalidation::Weather,
        CloudInvalidation::Model,
        CloudInvalidation::Enable,
        CloudInvalidation::ViewRemoved,
    ];
    for (index, reason) in reasons.iter().enumerate() {
        assert!(
            !reasons[..index].contains(reason),
            "reason {reason:?} is duplicated"
        );
    }
}

#[test]
fn the_negative_screen_alpha_sentinel_stays_reserved() {
    assert_eq!(CLOUD_SCREEN_UNKNOWN_ALPHA, -1.0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn neutral_views_are_rgba16float_with_false_validity() {
    let Some(gpu) = raa_core::testing::software_gpu() else {
        raa_core::testing::skip(
            "neutral_views_are_rgba16float_with_false_validity",
            "software_gpu returned None; see the software_gpu line above",
        );
        return;
    };
    let views = CloudViews::neutral(&gpu).expect("neutral cloud views");
    assert!(!views.screen_valid, "neutral screen validity is false");
    assert!(!views.panorama_valid, "neutral panorama validity is false");
    for (label, view) in [("panorama", &views.panorama), ("screen", &views.screen)] {
        let texture = view.texture();
        assert_eq!(
            texture.format(),
            wgpu::TextureFormat::Rgba16Float,
            "{label}"
        );
        assert_eq!((texture.width(), texture.height()), (1, 1), "{label}");
        assert_eq!(texture.dimension(), wgpu::TextureDimension::D2, "{label}");
        assert!(
            texture
                .usage()
                .contains(wgpu::TextureUsages::TEXTURE_BINDING),
            "{label} is bindable"
        );
    }
    let transmittance = views.transmittance.texture();
    assert_eq!(transmittance.format(), wgpu::TextureFormat::Rgba16Float);
    assert_eq!(
        transmittance.depth_or_array_layers(),
        2,
        "the S10 transmittance array has exactly two layers"
    );
}

#[path = "../build.rs"]
mod registry_build;
const _: fn() = registry_build::main;

#[test]
fn generated_registry_uses_only_present_producers_and_rejects_duplicates() {
    let source = std::path::Path::new("/fixture/src/clouds/mod.rs");
    let absent = registry_build::registry_for_presence(source, false);
    assert!(absent.contains("NeutralCloudRenderer::create"));
    assert!(!absent.contains("mod clouds"));
    assert!(!absent.contains("clouds::create"));
    assert!(!absent.contains("/fixture"));
    let present = registry_build::registry_for_presence(source, true);
    assert!(present.contains("#[path = \"/fixture/src/clouds/mod.rs\"]"));
    assert!(present.contains("registry.register(M2_BACKEND, clouds::create)?"));
    for candidates in [["neutral", "neutral"], ["m2", "m2"]] {
        assert!(registry_build::generate_registry(source, &candidates).is_err());
    }
}
