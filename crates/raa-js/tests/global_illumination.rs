//! m30 global illumination, share 1 (S36, contract `m30-wire`): the pure helpers run natively.
//! The defaults, every refusal message, the merge of omitted fields, the range refusals of
//! `apply`, and the instance and stats entry orders are checked against the S36 table. The
//! wasm32 runtime and the Engine forwarding are another card's, checked by its wasm32 build.

#![cfg(not(target_arch = "wasm32"))]

#[path = "../src/api/m30_global_illumination.rs"]
mod m30gi;

use m30gi::{
    GiWire, QUALITY_KEYS, colors_to_rgba, colour_data, forward_colors, instance_entries,
    parse_quality_patch, parse_settings_patch, quality_entries, settings_entries, stats_entries,
};
use raa_core::RaaError;
use raa_gi::membership::refusal_message;
use raa_gi::{
    DebugView, Driver, FINEST_VOXEL_METRES, GiQuality, GiSettings, GiSettingsPatch, GiStats,
    InstanceGi, SsgiQuality,
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
            ("screenSpace", GiWire::Bool(false)),
            ("screenSpaceQuality", text("medium")),
            ("indirect", text("probes")),
            ("historyMax", GiWire::Number(64.0)),
            ("clampGamma", GiWire::Number(2.5)),
            ("spatialPasses", GiWire::Number(3.0)),
            ("restirMCap", GiWire::Number(20.0)),
            ("voxelFullRelight", GiWire::Bool(false)),
            ("historyMin", GiWire::Number(1.0)),
            ("voxelHistoryMax", GiWire::Number(32.0)),
            ("voxelRays", GiWire::Number(3.0)),
        ]
    );
}

#[test]
fn indirect_names_the_irradiance_source() {
    assert_eq!(
        applied_settings(&[("indirect", text("restir"))])
            .unwrap()
            .indirect,
        raa_gi::IndirectMode::Restir
    );
    assert_eq!(
        settings_refusal(&[("indirect", text("voxels"))]),
        "indirect: expected one of probes, restir"
    );
    assert_eq!(
        settings_refusal(&[("indirect", GiWire::Bool(true))]),
        "indirect: expected one of probes, restir"
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
        "debugView: expected one of none, voxels, radiance, distance, levels, probes, hitType, \
         history, historyLength, reservoirs, rawDenoised"
    );
    assert_eq!(
        settings_refusal(&[("screenSpaceQuality", text("ultra"))]),
        "screenSpaceQuality: expected one of low, medium, high"
    );
    assert_eq!(
        settings_refusal(&[("screenSpaceQuality", GiWire::Number(1.0))]),
        "screenSpaceQuality: expected one of low, medium, high"
    );
    assert_eq!(
        settings_refusal(&[("screenSpace", GiWire::Number(1.0))]),
        "screenSpace: expected a boolean"
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
        applied_settings(&[
            ("screenSpace", GiWire::Bool(true)),
            ("screenSpaceQuality", text("high")),
        ])
        .unwrap(),
        GiSettings {
            screen_space: true,
            screen_space_quality: SsgiQuality::High,
            ..GiSettings::default()
        }
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
            ("historyLengthMean", GiWire::Null),
            ("rejectedPercent", GiWire::Null),
            ("clampedPercent", GiWire::Null),
        ]
    );
    assert_eq!(
        stats_entries(&stats_at(Some(7)))[9],
        ("statsFrameIndex", GiWire::Number(7.0))
    );
}

#[test]
fn vertex_colour_buffers_expand_to_rgba_or_refuse() {
    // coloured rgb buffer gains alpha 1; rgba passes through; an omitted buffer is not expanded here.
    assert_eq!(
        colors_to_rgba(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 3).unwrap(),
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    );
    assert_eq!(colors_to_rgba(&[0.5; 8], 4).unwrap(), vec![0.5; 8]);
    assert!(refusal(colors_to_rgba(&[0.5; 4], 3)).starts_with("colors: "));
    assert!(refusal(colors_to_rgba(&[0.5; 4], 2)).starts_with("components: "));
}

#[test]
fn colour_setters_forward_coloured_float_byte_and_absent_buffers() {
    // three vertices, rgb floats: forwarded as rgba with alpha 1 to every sub-mesh id
    let rgb = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0];
    let data = colour_data(Some(&rgb), None, Some(3)).unwrap();
    let mut seen: Vec<(u32, Option<Vec<f32>>)> = Vec::new();
    forward_colors(&[7u32, 9], data.as_deref(), |id, c| {
        seen.push((id, c.map(<[f32]>::to_vec)));
        Ok(())
    })
    .unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].1.as_ref().unwrap().len(), 12);
    assert_eq!(seen[1].0, 9);

    // Uint8Array rgba is divided by 255
    let bytes = [255u8, 0, 0, 255, 0, 255, 0, 255];
    let data = colour_data(None, Some(&bytes), None).unwrap().unwrap();
    assert_eq!(data, vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

    // no buffer: forwarded as no colour (albedo unchanged)
    let none = colour_data(None, None, Some(3)).unwrap();
    assert!(none.is_none());
    let mut calls = 0;
    forward_colors(&[1u32], none.as_deref(), |_, c| {
        assert!(c.is_none());
        calls += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(calls, 1);

    // a bad buffer is refused
    assert!(colour_data(Some(&[0.5; 4]), None, Some(3)).is_err());
}
