//! m30 global illumination, share 1 (S36, contract `m30-wire`): the pure helpers run natively.
//! The defaults, every refusal message, the merge of omitted fields, the range refusals of
//! `apply`, and the instance and stats entry orders are checked against the S36 table. The
//! wasm32 runtime and the Engine forwarding are another card's, checked by its wasm32 build.

#![cfg(not(target_arch = "wasm32"))]

#[path = "../src/api/m30_global_illumination.rs"]
mod m30gi;

use m30gi::{
    GiWire, QUALITY_KEYS, instance_entries, parse_quality_patch, parse_settings_patch,
    quality_entries, settings_entries, stats_entries,
};
use raa_core::RaaError;
use raa_gi::membership::refusal_message;
use raa_gi::{
    DebugView, Driver, FINEST_VOXEL_METRES, GiQuality, GiSettings, GiSettingsPatch, GiStats,
    InstanceGi,
};

/// The wire entries of a patch, as the wasm boundary builds them.
fn owned(pairs: &[(&str, GiWire)]) -> Vec<(String, GiWire)> {
    pairs
        .iter()
        .map(|(key, value)| (key.to_string(), value.clone()))
        .collect()
}

fn text(value: &str) -> GiWire {
    GiWire::Text(value.to_string())
}

/// The `gi.invalid` message of a refusal; any other result fails the test.
fn refusal<T>(result: Result<T, RaaError>) -> String {
    match result {
        Ok(_) => panic!("expected a gi.invalid refusal"),
        Err(error) => {
            assert_eq!(error.code(), "gi.invalid");
            error.message().to_string()
        }
    }
}

/// A settings patch parsed from wire entries and applied over the defaults.
fn applied_settings(pairs: &[(&str, GiWire)]) -> Result<GiSettings, RaaError> {
    parse_settings_patch(&owned(pairs))?.apply(&GiSettings::default())
}

/// A quality patch parsed from wire entries and applied over the defaults.
fn applied_quality(pairs: &[(&str, GiWire)]) -> Result<GiQuality, RaaError> {
    parse_quality_patch(&owned(pairs))?.apply(&GiQuality::default())
}

fn settings_refusal(pairs: &[(&str, GiWire)]) -> String {
    refusal(parse_settings_patch(&owned(pairs)))
}

fn quality_refusal(pairs: &[(&str, GiWire)]) -> String {
    refusal(parse_quality_patch(&owned(pairs)))
}

#[test]
fn settings_defaults_follow_the_s36_key_order() {
    assert_eq!(
        settings_entries(&GiSettings::default()),
        [
            ("enabled", GiWire::Bool(false)),
            ("intensity", GiWire::Number(1.0)),
            ("environmentDiffuse", GiWire::Number(0.0)),
            ("localLightShadows", GiWire::Bool(true)),
            ("reflections", GiWire::Bool(true)),
            ("screenTraces", GiWire::Bool(true)),
            ("debugView", text("none")),
        ]
    );
}

#[test]
fn quality_defaults_are_96_5_8() {
    assert_eq!(
        quality_entries(&GiQuality::default()),
        [
            ("resolution", GiWire::Number(96.0)),
            ("levels", GiWire::Number(5.0)),
            ("probeRays", GiWire::Number(8.0)),
        ]
    );
}

#[test]
fn settings_refusals_name_the_key_and_reason() {
    assert_eq!(
        settings_refusal(&[("foo", GiWire::Bool(true))]),
        "foo: unknown key"
    );
    assert_eq!(
        settings_refusal(&[("intensity", GiWire::Null)]),
        "intensity: must not be null"
    );
    assert_eq!(
        settings_refusal(&[("enabled", GiWire::Number(1.0))]),
        "enabled: expected a boolean"
    );
    assert_eq!(
        settings_refusal(&[("intensity", text("2"))]),
        "intensity: expected a number"
    );
    assert_eq!(
        settings_refusal(&[("debugView", text("x"))]),
        "debugView: expected one of none, voxels, radiance, distance, levels"
    );
}

#[test]
fn quality_refusals_accept_numbers_only() {
    for key in QUALITY_KEYS {
        assert_eq!(
            quality_refusal(&[(key, text("96"))]),
            format!("{key}: expected a number")
        );
        assert_eq!(
            quality_refusal(&[(key, GiWire::Null)]),
            format!("{key}: must not be null")
        );
    }
    assert_eq!(
        quality_refusal(&[("foo", GiWire::Number(1.0))]),
        "foo: unknown key"
    );
}

#[test]
fn the_first_offending_entry_in_order_wins() {
    assert_eq!(
        settings_refusal(&[("intensity", text("2")), ("foo", GiWire::Bool(true))]),
        "intensity: expected a number"
    );
    assert_eq!(
        settings_refusal(&[
            ("foo", GiWire::Bool(true)),
            ("enabled", GiWire::Number(1.0))
        ]),
        "foo: unknown key"
    );
    assert_eq!(
        quality_refusal(&[("probeRays", GiWire::Null), ("resolution", text("96"))]),
        "probeRays: must not be null"
    );
}

#[test]
fn omitted_fields_merge_and_undefined_entries_are_ignored() {
    let patch = parse_settings_patch(&owned(&[
        ("intensity", GiWire::Number(2.0)),
        ("enabled", GiWire::Undefined),
    ]));
    assert_eq!(
        patch.unwrap(),
        GiSettingsPatch {
            intensity: Some(2.0),
            ..GiSettingsPatch::default()
        }
    );
    assert_eq!(
        applied_settings(&[("intensity", GiWire::Number(2.0))]).unwrap(),
        GiSettings {
            intensity: 2.0,
            ..GiSettings::default()
        }
    );
    assert_eq!(
        applied_settings(&[("enabled", GiWire::Undefined)]).unwrap(),
        GiSettings::default()
    );
    assert_eq!(
        applied_settings(&[("debugView", text("levels"))])
            .unwrap()
            .debug_view,
        DebugView::Levels
    );
    assert_eq!(
        applied_quality(&[
            ("levels", GiWire::Number(4.0)),
            ("probeRays", GiWire::Undefined)
        ])
        .unwrap(),
        GiQuality {
            levels: 4,
            ..GiQuality::default()
        }
    );
}

#[test]
fn range_refusals_come_from_apply() {
    let intensity = "intensity: must be finite and in [0, 16]";
    assert_eq!(
        refusal(applied_settings(&[("intensity", GiWire::Number(17.0))])),
        intensity
    );
    assert_eq!(
        refusal(applied_settings(&[("intensity", GiWire::Number(f64::NAN))])),
        intensity
    );
    assert_eq!(
        refusal(applied_settings(&[(
            "environmentDiffuse",
            GiWire::Number(1.5)
        )])),
        "environmentDiffuse: must be finite and in [0, 1]"
    );
    assert_eq!(
        refusal(applied_quality(&[("resolution", GiWire::Number(80.0))])),
        "resolution: must be one of 64, 96"
    );
    assert_eq!(
        refusal(applied_quality(&[("levels", GiWire::Number(2.0))])),
        "levels: must be one of 3, 4, 5"
    );
    assert_eq!(
        refusal(applied_quality(&[("probeRays", GiWire::Number(7.0))])),
        "probeRays: must be one of 6, 8"
    );
}

#[test]
fn instance_entries_of_a_forward_voxelized_group() {
    let instance = InstanceGi {
        requested: None,
        voxelized: true,
        driver: Some(Driver::Forward),
        code: None,
        message: None,
    };
    assert_eq!(
        instance_entries(&instance),
        [
            ("requested", GiWire::Null),
            ("voxelized", GiWire::Bool(true)),
            ("driver", text("forward")),
            ("code", GiWire::Null),
            ("message", GiWire::Null),
        ]
    );
}

#[test]
fn instance_entries_of_a_refused_group_carry_code_and_message() {
    let message = refusal_message("gi.opted_out");
    let instance = InstanceGi {
        requested: Some(false),
        voxelized: false,
        driver: None,
        code: Some("gi.opted_out"),
        message: Some(message),
    };
    assert_eq!(
        instance_entries(&instance),
        [
            ("requested", GiWire::Bool(false)),
            ("voxelized", GiWire::Bool(false)),
            ("driver", GiWire::Null),
            ("code", text("gi.opted_out")),
            ("message", text(message)),
        ]
    );
}

fn stats_at(index: Option<u64>) -> GiStats {
    GiStats {
        inactive_reason: Some("disabled"),
        finest_voxel_metres: FINEST_VOXEL_METRES,
        stats_frame_index: index,
        ..GiStats::default()
    }
}

#[test]
fn stats_entries_follow_the_key_order_with_a_null_frame_index() {
    assert_eq!(
        stats_entries(&stats_at(None)),
        [
            ("active", GiWire::Bool(false)),
            ("inactiveReason", text("disabled")),
            ("quality", GiWire::Other),
            (
                "finestVoxelMetres",
                GiWire::Number(f64::from(FINEST_VOXEL_METRES))
            ),
            ("voxelizedForwardInstances", GiWire::Number(0.0)),
            ("voxelizedVirtualInstances", GiWire::Number(0.0)),
            ("pendingRebuild", GiWire::Bool(false)),
            ("framesSinceRebuild", GiWire::Number(0.0)),
            ("allocatedBytes", GiWire::Number(0.0)),
            ("statsFrameIndex", GiWire::Null),
            ("pending", GiWire::Bool(false)),
        ]
    );
    assert_eq!(
        stats_entries(&stats_at(Some(7)))[9],
        ("statsFrameIndex", GiWire::Number(7.0))
    );
}
