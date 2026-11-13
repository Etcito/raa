//! m29 virtual geometry, share 1 (S31): the pure helpers run natively. The wasm32 runtime
//! (`install`, `frame`) is not reachable here; the patch parser, its refusals, the settings and
//! instance entries and the stats entries are checked against the S31 table.

#![cfg(not(target_arch = "wasm32"))]

#[path = "../src/api/m29_virtual_geometry.rs"]
mod m29vg;

use m29vg::{
    SETTING_KEYS, WireValue, instance_entries, parse_settings_patch, settings_entries,
    stats_entries,
};
use raa_core::RaaError;
use raa_vg::{DebugView, InstancePath, VgSettings, VgSettingsPatch, VgStats};

/// The entries of a patch as the wasm boundary builds them.
fn entries(pairs: &[(&str, WireValue)]) -> Vec<(String, WireValue)> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect()
}

/// Parses and applies a patch over the defaults.
fn applied(pairs: &[(&str, WireValue)]) -> Result<VgSettings, RaaError> {
    parse_settings_patch(&entries(pairs))?.apply(&VgSettings::default())
}

/// The code and the message of a refusal; any other outcome fails the test.
fn refusal<T: std::fmt::Debug>(result: Result<T, RaaError>) -> (&'static str, String) {
    match result {
        Err(RaaError::FrameGraph { code, message }) => (code, message),
        other => panic!("expected a FrameGraph refusal, got {other:?}"),
    }
}

#[test]
fn an_empty_patch_keeps_the_defaults() {
    let patch = parse_settings_patch(&[]).expect("an empty patch parses");
    assert_eq!(patch, VgSettingsPatch::default());
    let settings = patch
        .apply(&VgSettings::default())
        .expect("the empty patch applies");
    assert_eq!(settings, VgSettings::default());
}

#[test]
fn each_key_with_a_valid_value_lands_in_its_field() {
    let patch = parse_settings_patch(&entries(&[
        ("enabled", WireValue::Bool(true)),
        ("errorPixels", WireValue::Number(4.0)),
        ("softwareRaster", WireValue::Bool(false)),
        ("occlusion", WireValue::Bool(false)),
        ("shadows", WireValue::Bool(false)),
        ("runtimeBake", WireValue::Bool(false)),
        ("instanceDefault", WireValue::Bool(false)),
        ("maxVisibleClusters", WireValue::Number(2048.0)),
        ("debugView", WireValue::Text("lod".to_string())),
    ]))
    .expect("a valid patch parses");
    assert_eq!(
        patch,
        VgSettingsPatch {
            enabled: Some(true),
            error_pixels: Some(4.0),
            software_raster: Some(false),
            occlusion: Some(false),
            shadows: Some(false),
            runtime_bake: Some(false),
            instance_default: Some(false),
            max_visible_clusters: Some(2048),
            debug_view: Some(DebugView::Lod),
        }
    );
    let settings = patch
        .apply(&VgSettings::default())
        .expect("the valid patch applies");
    assert_eq!(settings.error_pixels, 4.0);
    assert_eq!(settings.max_visible_clusters, 2048);
    assert_eq!(settings.debug_view, DebugView::Lod);
}

#[test]
fn undefined_is_omission() {
    let patch = parse_settings_patch(&entries(&[
        ("enabled", WireValue::Undefined),
        ("maxVisibleClusters", WireValue::Undefined),
    ]))
    .expect("undefined values are omitted");
    assert_eq!(patch, VgSettingsPatch::default());
}

#[test]
fn null_a_wrong_type_and_an_unknown_key_refuse_with_the_key_name() {
    let cases: [(&str, WireValue, &str); 7] = [
        ("enabled", WireValue::Null, "enabled: must not be null"),
        (
            "errorPixels",
            WireValue::Null,
            "errorPixels: must not be null",
        ),
        (
            "enabled",
            WireValue::Number(1.0),
            "enabled: must be a boolean",
        ),
        (
            "errorPixels",
            WireValue::Text("1".to_string()),
            "errorPixels: must be a number",
        ),
        (
            "debugView",
            WireValue::Number(1.0),
            "debugView: must be a string",
        ),
        (
            "maxVisibleClusters",
            WireValue::Bool(true),
            "maxVisibleClusters: must be a number",
        ),
        ("gain", WireValue::Undefined, "gain: unknown setting"),
    ];
    for (key, value, expected) in cases {
        let (code, message) = refusal(parse_settings_patch(&entries(&[(key, value)])));
        assert_eq!(code, "vg.invalid", "{key}");
        assert_eq!(message, expected);
    }
}

#[test]
fn a_debug_view_outside_the_five_names_refuses() {
    let (code, message) = refusal(parse_settings_patch(&entries(&[(
        "debugView",
        WireValue::Text("grid".to_string()),
    )])));
    assert_eq!(code, "vg.invalid");
    assert_eq!(
        message,
        "debugView: must be none, clusters, lod, raster or occlusion, got grid"
    );
}

#[test]
fn a_fractional_max_visible_clusters_refuses() {
    let (code, message) = refusal(parse_settings_patch(&entries(&[(
        "maxVisibleClusters",
        WireValue::Number(1024.5),
    )])));
    assert_eq!(code, "vg.invalid");
    assert_eq!(message, "maxVisibleClusters: must be an integer");
}

#[test]
fn out_of_domain_values_refuse_through_apply_with_the_field() {
    let cases: [(&str, f64, &str); 5] = [
        ("errorPixels", 0.1, "errorPixels"),
        ("errorPixels", 65.0, "errorPixels"),
        ("errorPixels", f64::NAN, "errorPixels"),
        ("maxVisibleClusters", 1023.0, "maxVisibleClusters"),
        ("maxVisibleClusters", 4_194_305.0, "maxVisibleClusters"),
    ];
    for (key, value, field) in cases {
        let (code, message) = refusal(applied(&[(key, WireValue::Number(value))]));
        assert_eq!(code, "vg.invalid", "{key}={value}");
        assert!(message.starts_with(&format!("{field}: ")), "{message}");
    }
}

#[test]
fn settings_entries_of_the_default_are_the_s31_table_in_key_order() {
    let entries = settings_entries(&VgSettings::default());
    let keys: Vec<&str> = entries.iter().map(|(key, _)| *key).collect();
    assert_eq!(keys, SETTING_KEYS);
    assert_eq!(
        entries,
        vec![
            ("enabled", WireValue::Bool(false)),
            ("errorPixels", WireValue::Number(1.0)),
            ("softwareRaster", WireValue::Bool(true)),
            ("occlusion", WireValue::Bool(true)),
            ("shadows", WireValue::Bool(true)),
            ("runtimeBake", WireValue::Bool(true)),
            ("instanceDefault", WireValue::Bool(true)),
            ("maxVisibleClusters", WireValue::Number(262_144.0)),
            ("debugView", WireValue::Text("none".to_string())),
        ]
    );
}

#[test]
fn instance_entries_for_the_virtual_path_are_null_where_absent() {
    assert_eq!(
        instance_entries(&InstancePath::Virtual, None),
        vec![
            ("requested", WireValue::Null),
            ("path", WireValue::Text("virtual".to_string())),
            ("pending", WireValue::Bool(false)),
            ("code", WireValue::Null),
            ("message", WireValue::Null),
        ]
    );
}

#[test]
fn instance_entries_for_a_refused_forward_path_carry_the_code() {
    let path = InstancePath::Forward {
        pending: false,
        code: Some("vg.skinned_unsupported"),
        message: Some("skinned instance: posed skins stay on the forward path".to_string()),
    };
    assert_eq!(
        instance_entries(&path, Some(true)),
        vec![
            ("requested", WireValue::Bool(true)),
            ("path", WireValue::Text("forward".to_string())),
            ("pending", WireValue::Bool(false)),
            (
                "code",
                WireValue::Text("vg.skinned_unsupported".to_string()),
            ),
            (
                "message",
                WireValue::Text(
                    "skinned instance: posed skins stay on the forward path".to_string()
                ),
            ),
        ]
    );
}

#[test]
fn instance_entries_of_a_pending_instance_have_no_code() {
    let path = InstancePath::Forward {
        pending: true,
        code: None,
        message: None,
    };
    let entries = instance_entries(&path, Some(true));
    assert_eq!(entries[2], ("pending", WireValue::Bool(true)));
    assert_eq!(entries[3], ("code", WireValue::Null));
    assert_eq!(entries[4], ("message", WireValue::Null));
}

#[test]
fn stats_entries_keep_the_s31_keys_in_order_with_null_counters() {
    let entries = stats_entries(&VgStats::default());
    let keys: Vec<&str> = entries.iter().map(|(key, _)| *key).collect();
    assert_eq!(
        keys,
        [
            "active",
            "inactiveReason",
            "virtualInstances",
            "forwardInstances",
            "refused",
            "bakingMeshes",
            "readyMeshes",
            "drawnClusters",
            "drawnTriangles",
            "hardwareClusters",
            "softwareClusters",
            "shadowClusters",
            "droppedClusters",
            "statsFrameIndex",
            "pending",
        ]
    );
    let value = |key: &str| {
        entries
            .iter()
            .find(|(name, _)| *name == key)
            .map(|(_, value)| value.clone())
    };
    assert_eq!(value("active"), Some(WireValue::Bool(false)));
    assert_eq!(value("inactiveReason"), Some(WireValue::Null));
    assert_eq!(value("refused"), Some(WireValue::Other));
    assert_eq!(value("drawnClusters"), Some(WireValue::Null));
    assert_eq!(value("droppedClusters"), Some(WireValue::Null));
    assert_eq!(value("statsFrameIndex"), Some(WireValue::Null));
    assert_eq!(value("pending"), Some(WireValue::Bool(false)));
}

#[test]
fn stats_entries_of_published_counters_carry_their_values() {
    let stats = VgStats {
        active: false,
        inactive_reason: Some("noInstance"),
        drawn_clusters: Some(12),
        drawn_triangles: Some(1536),
        stats_frame_index: Some(7),
        ..VgStats::default()
    };
    let entries = stats_entries(&stats);
    let value = |key: &str| {
        entries
            .iter()
            .find(|(name, _)| *name == key)
            .map(|(_, value)| value.clone())
    };
    assert_eq!(
        value("inactiveReason"),
        Some(WireValue::Text("noInstance".to_string()))
    );
    assert_eq!(value("drawnClusters"), Some(WireValue::Number(12.0)));
    assert_eq!(value("drawnTriangles"), Some(WireValue::Number(1536.0)));
    assert_eq!(value("statsFrameIndex"), Some(WireValue::Number(7.0)));
    assert_eq!(value("hardwareClusters"), Some(WireValue::Null));
}

#[test]
fn scene_invocation_reaches_vg_observe_submission_with_the_sky_invocation_id() {
    let _: fn(&raa_vg::VgHandle, u64, bool) = raa_vg::VgHandle::observe_submission;
    let engine = include_str!("../src/engine.rs");
    let finish = &engine[engine.find("fn finish_scene_invocation(").unwrap()..];
    let finish = &finish[..finish.find("\n    }\n").unwrap()];
    let vg_call = "vg.observe_submission(s.sky_invocation_id(), complete)";
    assert_eq!(
        finish
            .matches("s.features.get::<raa_vg::VgHandle>()")
            .count(),
        1
    );
    assert_eq!(finish.matches(vg_call).count(), 1);
    let sky_call = "sky.observe_submission(s.sky_invocation_id(), acknowledge, complete)";
    assert!(finish.find(sky_call).unwrap() < finish.find(vg_call).unwrap());
    assert_eq!(engine.matches("vg.observe_submission(").count(), 1);
    let m29 = include_str!("../src/api/m29_virtual_geometry.rs");
    assert!(!m29.contains("observe_submission("));
    assert!(!m29.contains("sky_invocation_id"));
    let begin = &engine[engine.find("fn begin_scene_invocation(").unwrap()..];
    let begin = &begin[..begin.find("\n    }\n").unwrap()];
    assert_eq!(
        begin
            .matches("next_sky_invocation(&mut s.sky_invocation_id)")
            .count(),
        1
    );
}
