//! The public settings surface of S36: defaults, the [`DebugView`] table, the atomic
//! [`GiSettingsPatch`] and [`GiQualityPatch`] merges and their `gi.invalid` refusals.

use raa_core::RaaError;
use raa_gi::error::{GI_INVALID, invalid};
use raa_gi::{
    DebugView, FINEST_VOXEL_METRES, GiQuality, GiQualityPatch, GiSettings, GiSettingsPatch,
};

const INTENSITY_MESSAGE: &str = "intensity: must be finite and in [0, 16]";
const ENVIRONMENT_DIFFUSE_MESSAGE: &str = "environmentDiffuse: must be finite and in [0, 1]";

fn refused(err: &RaaError, message: &str) {
    assert_eq!(err.code(), GI_INVALID, "code of {err:?}");
    assert_eq!(err.message(), message);
}

#[test]
fn settings_default_is_the_s36_table() {
    let settings = GiSettings::default();
    assert!(!settings.enabled);
    assert_eq!(settings.intensity, 1.0);
    assert_eq!(settings.environment_diffuse, 0.0);
    assert!(settings.local_light_shadows);
    assert!(settings.reflections);
    assert!(settings.screen_traces);
    assert_eq!(settings.debug_view, DebugView::None);
}

#[test]
fn quality_default_is_96_5_8() {
    let quality = GiQuality::default();
    assert_eq!(quality.resolution, 96);
    assert_eq!(quality.levels, 5);
    assert_eq!(quality.probe_rays, 8);
}

#[test]
fn finest_voxel_is_twenty_centimetres() {
    assert_eq!(FINEST_VOXEL_METRES, 0.2);
}

#[test]
fn debug_view_table_round_trips() {
    let views = [
        (DebugView::None, "none", 0),
        (DebugView::Voxels, "voxels", 1),
        (DebugView::Radiance, "radiance", 2),
        (DebugView::Distance, "distance", 3),
        (DebugView::Levels, "levels", 4),
    ];
    for (view, name, code) in views {
        assert_eq!(view.as_str(), name);
        assert_eq!(view.code(), code);
        assert_eq!(DebugView::parse(name), Some(view));
    }
    assert_eq!(DebugView::parse("x"), None);
    assert_eq!(DebugView::default(), DebugView::None);
}

#[test]
fn settings_patch_merges_omitted_fields() {
    let current = GiSettings {
        enabled: true,
        intensity: 2.0,
        environment_diffuse: 0.5,
        local_light_shadows: false,
        reflections: false,
        screen_traces: false,
        debug_view: DebugView::Distance,
    };
    let merged = GiSettingsPatch {
        reflections: Some(true),
        debug_view: Some(DebugView::Levels),
        ..GiSettingsPatch::default()
    }
    .apply(&current)
    .expect("valid patch");
    assert!(merged.enabled);
    assert_eq!(merged.intensity, 2.0);
    assert_eq!(merged.environment_diffuse, 0.5);
    assert!(!merged.local_light_shadows);
    assert!(merged.reflections);
    assert!(!merged.screen_traces);
    assert_eq!(merged.debug_view, DebugView::Levels);
}

#[test]
fn intensity_refusals_name_their_key() {
    for value in [-0.1, 16.5, f64::NAN, f64::INFINITY] {
        let patch = GiSettingsPatch {
            intensity: Some(value),
            ..GiSettingsPatch::default()
        };
        let err = patch.apply(&GiSettings::default()).unwrap_err();
        refused(&err, INTENSITY_MESSAGE);
    }
    let domain = GiSettingsPatch {
        intensity: Some(16.0),
        ..GiSettingsPatch::default()
    };
    assert_eq!(
        domain.apply(&GiSettings::default()).unwrap().intensity,
        16.0
    );
}

#[test]
fn environment_diffuse_refusal_names_its_key() {
    let patch = GiSettingsPatch {
        environment_diffuse: Some(1.5),
        ..GiSettingsPatch::default()
    };
    let err = patch.apply(&GiSettings::default()).unwrap_err();
    refused(&err, ENVIRONMENT_DIFFUSE_MESSAGE);
}

#[test]
fn first_bad_key_in_key_order_is_reported() {
    let patch = GiSettingsPatch {
        environment_diffuse: Some(1.5),
        intensity: Some(16.5),
        ..GiSettingsPatch::default()
    };
    let err = patch.apply(&GiSettings::default()).unwrap_err();
    refused(&err, INTENSITY_MESSAGE);
}

#[test]
fn refused_apply_leaves_the_input_unchanged() {
    let current = GiSettings {
        enabled: true,
        intensity: 3.0,
        environment_diffuse: 0.25,
        local_light_shadows: true,
        reflections: false,
        screen_traces: true,
        debug_view: DebugView::Radiance,
    };
    let before = current;
    let patch = GiSettingsPatch {
        enabled: Some(false),
        intensity: Some(-0.1),
        reflections: Some(true),
        ..GiSettingsPatch::default()
    };
    assert!(patch.apply(&current).is_err());
    assert_eq!(current, before);
}

#[test]
fn quality_accepts_the_listed_values() {
    for resolution in [64.0, 96.0] {
        let patch = GiQualityPatch {
            resolution: Some(resolution),
            ..GiQualityPatch::default()
        };
        let quality = patch
            .apply(&GiQuality::default())
            .expect("valid resolution");
        assert_eq!(f64::from(quality.resolution), resolution);
    }
    for (resolution, levels, probe_rays) in [(64, 3, 6), (96, 4, 8), (96, 5, 8)] {
        let quality = GiQualityPatch {
            resolution: Some(f64::from(resolution)),
            levels: Some(f64::from(levels)),
            probe_rays: Some(f64::from(probe_rays)),
        }
        .apply(&GiQuality::default())
        .expect("valid quality");
        assert_eq!(quality.resolution, resolution);
        assert_eq!(quality.levels, levels);
        assert_eq!(quality.probe_rays, probe_rays);
    }
}

#[test]
fn quality_patch_merges_omitted_fields() {
    let current = GiQuality {
        resolution: 64,
        levels: 4,
        probe_rays: 6,
    };
    let merged = GiQualityPatch {
        levels: Some(5.0),
        ..GiQualityPatch::default()
    }
    .apply(&current)
    .expect("valid patch");
    assert_eq!(merged.resolution, 64);
    assert_eq!(merged.levels, 5);
    assert_eq!(merged.probe_rays, 6);
}

#[test]
fn quality_refusals_name_their_key() {
    type Case = (Option<f64>, Option<f64>, Option<f64>, &'static str);
    let cases: [Case; 4] = [
        (Some(80.0), None, None, "resolution: must be one of 64, 96"),
        (Some(96.5), None, None, "resolution: must be one of 64, 96"),
        (None, Some(2.0), None, "levels: must be one of 3, 4, 5"),
        (None, None, Some(7.0), "probeRays: must be one of 6, 8"),
    ];
    for (resolution, levels, probe_rays, message) in cases {
        let patch = GiQualityPatch {
            resolution,
            levels,
            probe_rays,
        };
        let err = patch.apply(&GiQuality::default()).unwrap_err();
        refused(&err, message);
    }
}

#[test]
fn invalid_helper_formats_key_then_reason() {
    let err = invalid("k", "r");
    assert_eq!(err.code(), GI_INVALID);
    assert!(err.to_string().contains("k: r"), "display was {err}");
}
