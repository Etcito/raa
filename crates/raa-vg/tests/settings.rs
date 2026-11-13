//! Settings, refusal codes and path eligibility of raa-vg (S31): the defaults of the S31
//! table, patch merging with atomic refusal, the debug-view codes and the eligibility rank table.

use raa_core::RaaError;
use raa_lit::{DrawClass, InstanceShape};
use raa_vg::eligibility::refusal;
use raa_vg::error::{DEFORMED_UNSUPPORTED, SKINNED_UNSUPPORTED, TRANSPARENT_MATERIAL};
use raa_vg::{DebugView, VgSettings, VgSettingsPatch, refusal_message};

/// The `(code, message)` of a refusal, or a panic when the error is not a frame-graph refusal.
fn refused(err: RaaError) -> (&'static str, String) {
    match err {
        RaaError::FrameGraph { code, message } => (code, message),
        other => panic!("expected a FrameGraph refusal, got {other:?}"),
    }
}

fn opaque() -> InstanceShape {
    InstanceShape {
        skinned: false,
        deformed: false,
        class: DrawClass::Opaque,
        vertex_fn: false,
        has_colors: false,
    }
}

#[test]
fn default_equals_the_s31_table() {
    let d = VgSettings::default();
    assert_eq!(
        d,
        VgSettings {
            enabled: false,
            error_pixels: 1.0,
            software_raster: true,
            occlusion: true,
            shadows: true,
            runtime_bake: true,
            instance_default: true,
            max_visible_clusters: 262_144,
            debug_view: DebugView::None,
        }
    );
    assert!(d.validate().is_ok());
}

#[test]
fn patch_with_only_error_pixels_keeps_every_other_field() {
    let base = VgSettings {
        enabled: true,
        software_raster: false,
        occlusion: false,
        shadows: false,
        runtime_bake: false,
        instance_default: false,
        max_visible_clusters: 4096,
        debug_view: DebugView::Lod,
        ..VgSettings::default()
    };
    let patch = VgSettingsPatch {
        error_pixels: Some(4.0),
        ..VgSettingsPatch::default()
    };
    let merged = patch.apply(&base).expect("valid patch");
    assert_eq!(merged.error_pixels, 4.0);
    assert_eq!(
        VgSettings {
            error_pixels: base.error_pixels,
            ..merged
        },
        base
    );
}

#[test]
fn invalid_fields_refuse_with_their_camel_case_name_and_leave_the_base() {
    let base = VgSettings::default();
    let cases: [(VgSettingsPatch, &str); 6] = [
        (
            VgSettingsPatch {
                error_pixels: Some(0.2),
                ..Default::default()
            },
            "errorPixels",
        ),
        (
            VgSettingsPatch {
                error_pixels: Some(64.5),
                ..Default::default()
            },
            "errorPixels",
        ),
        (
            VgSettingsPatch {
                error_pixels: Some(f32::NAN),
                ..Default::default()
            },
            "errorPixels",
        ),
        (
            VgSettingsPatch {
                error_pixels: Some(f32::INFINITY),
                ..Default::default()
            },
            "errorPixels",
        ),
        (
            VgSettingsPatch {
                max_visible_clusters: Some(1023),
                ..Default::default()
            },
            "maxVisibleClusters",
        ),
        (
            VgSettingsPatch {
                max_visible_clusters: Some(4_194_305),
                ..Default::default()
            },
            "maxVisibleClusters",
        ),
    ];
    for (patch, field) in cases {
        let err = patch.apply(&base).expect_err("invalid patch must refuse");
        let (code, message) = refused(err);
        assert_eq!(code, "vg.invalid");
        assert!(
            message.starts_with(&format!("{field}: ")),
            "message {message:?} must name {field}"
        );
        assert_eq!(base, VgSettings::default(), "the base must stay unchanged");
    }
}

#[test]
fn range_bounds_are_accepted() {
    let base = VgSettings::default();
    let low = VgSettingsPatch {
        error_pixels: Some(0.25),
        max_visible_clusters: Some(1024),
        ..Default::default()
    };
    let high = VgSettingsPatch {
        error_pixels: Some(64.0),
        max_visible_clusters: Some(4_194_304),
        ..Default::default()
    };
    assert!(low.apply(&base).is_ok());
    assert!(high.apply(&base).is_ok());
}

#[test]
fn debug_view_codes_and_names_round_trip() {
    let table = [
        (DebugView::None, 0, "none"),
        (DebugView::Clusters, 1, "clusters"),
        (DebugView::Lod, 2, "lod"),
        (DebugView::Raster, 3, "raster"),
        (DebugView::Occlusion, 4, "occlusion"),
    ];
    for (view, code, name) in table {
        assert_eq!(view.code(), code);
        assert_eq!(view.name(), name);
        assert_eq!(DebugView::from_name(name), Some(view));
    }
    assert_eq!(DebugView::from_name("x"), None);
}

#[test]
fn eligibility_rank_table() {
    let skinned_deformed_transparent = InstanceShape {
        skinned: true,
        deformed: true,
        class: DrawClass::Transparent,
        ..opaque()
    };
    assert_eq!(
        refusal(&skinned_deformed_transparent),
        Some(SKINNED_UNSUPPORTED)
    );

    let deformed_transparent = InstanceShape {
        deformed: true,
        class: DrawClass::Transparent,
        ..opaque()
    };
    assert_eq!(refusal(&deformed_transparent), Some(DEFORMED_UNSUPPORTED));

    let vertex_fn = InstanceShape {
        vertex_fn: true,
        ..opaque()
    };
    assert_eq!(refusal(&vertex_fn), Some(DEFORMED_UNSUPPORTED));

    let fur = InstanceShape {
        class: DrawClass::Fur,
        ..opaque()
    };
    assert_eq!(refusal(&fur), Some(TRANSPARENT_MATERIAL));
    let fur_message = refusal_message(&fur).expect("fur is refused");
    assert!(fur_message.contains("fur"), "message {fur_message:?}");

    let transparent = InstanceShape {
        class: DrawClass::Transparent,
        ..opaque()
    };
    let transparent_message = refusal_message(&transparent).expect("transparent is refused");
    assert!(transparent_message.contains("transparent"));

    let plain = opaque();
    assert_eq!(refusal(&plain), None);
    assert_eq!(refusal_message(&plain), None);
}
