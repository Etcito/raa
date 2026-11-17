//! The GI tunable store (contract 'gi-tunables-store'): the `pass.gi.volume` keys and defaults of
//! docs/render-graph/tunables.md, the defaults of `GiTunables::new`, the refusals of
//! `GiTunables::set` and the overrides that `GiTunables::constants` returns.

use raa_gi::{GiTunables, tunables};

/// The `pass.gi.volume` rows of docs/render-graph/tunables.md in table order: key and default.
const VOLUME_ROWS: [(&str, f64); 3] = [("behind", 0.5), ("resetRays", 8.0), ("directChange", 0.05)];

#[test]
fn volume_keys_and_defaults_follow_the_table() {
    let specs = tunables();
    let keys: Vec<&str> = specs.iter().map(|spec| spec.key).collect();
    assert_eq!(keys[..3], ["behind", "resetRays", "directChange"]);
    for (spec, (key, default)) in specs.iter().zip(VOLUME_ROWS) {
        assert_eq!(spec.key, key);
        assert_eq!(spec.default, default, "default of {key}");
        assert_eq!(spec.min, None, "{key} states no range");
        assert_eq!(spec.max, None, "{key} states no range");
        assert_eq!(spec.rebuild, "pipeline", "{key}");
    }
    let integer: Vec<bool> = specs.iter().map(|spec| spec.integer).collect();
    assert_eq!(integer[..3], [false, true, false]);
}

#[test]
fn new_holds_every_default_at_generation_zero() {
    let store = GiTunables::new();
    assert_eq!(store.generation(), 0);
    for (key, default) in VOLUME_ROWS {
        assert_eq!(store.value(key), Some(default), "{key}");
    }
    assert_eq!(store.value("nope"), None);
}

#[test]
fn set_refuses_unknown_key_nan_and_a_fraction_with_gi_invalid_and_changes_nothing() {
    let mut store = GiTunables::new();
    let before = store.clone();
    let unknown = store.set("nope", 1.0).expect_err("an unknown key");
    assert_eq!(unknown.code(), "gi.invalid");
    let nan = store.set("directChange", f64::NAN).expect_err("NaN");
    assert_eq!(nan.code(), "gi.invalid");
    let fraction = store.set("resetRays", 2.5).expect_err("an integer row");
    assert_eq!(fraction.code(), "gi.invalid");
    assert_eq!(store.generation(), 0);
    assert_eq!(store, before);
}

#[test]
fn set_stores_a_change_and_advances_the_generation_once() {
    let mut store = GiTunables::new();
    assert!(store.set("directChange", 0.1).expect("a finite value"));
    assert_eq!(store.value("directChange"), Some(0.1));
    assert_eq!(store.generation(), 1);
    assert!(!store.set("directChange", 0.1).expect("the same value"));
    assert_eq!(store.generation(), 1);
}

#[test]
fn constants_pair_each_override_with_its_current_value() {
    let mut store = GiTunables::new();
    store.set("directChange", 0.2).expect("a finite value");
    assert_eq!(
        store.constants(&[("RAA_GI_DIRECT_CHANGE", "directChange")]),
        vec![("RAA_GI_DIRECT_CHANGE", 0.2)]
    );
    // Input order is kept and a key the store does not list is skipped.
    assert_eq!(
        store.constants(&[
            ("RAA_GI_BEHIND", "behind"),
            ("NOT_AN_OVERRIDE", "nope"),
            ("RAA_GI_RESET_RAYS", "resetRays"),
        ]),
        vec![("RAA_GI_BEHIND", 0.5), ("RAA_GI_RESET_RAYS", 8.0)]
    );
}

/// The `pass.gi.screen` rows of docs/render-graph/tunables.md that card 2/5 adds, in table order:
/// key, default and whether the row is integer.
const SCREEN_ROWS: [(&str, f64, bool); 18] = [
    ("accDepthTolerance", 0.04, false),
    ("accNormalCos", 0.9, false),
    ("accMinWeight", 0.05, false),
    ("accChangeSlack", 0.05, false),
    ("accChangeMin", 4.0, false),
    ("occlusionDirections", 3.0, true),
    ("occlusionSteps", 4.0, true),
    ("occlusionRadius", 0.9, false),
    ("atrousSigmaLuminance", 4.0, false),
    ("atrousNormalPower", 32.0, false),
    ("atrousPlane", 0.01, false),
    ("probeScreenDistance", 8.0, false),
    ("probeScreenSteps", 32.0, true),
    ("probeScreenThickness", 0.15, false),
    ("probeVolumeDistance", 320.0, false),
    ("probeVolumeSteps", 40.0, true),
    ("probeVolumeHitVoxels", 0.85, false),
    ("screenEdgeFade", 0.08, false),
];

#[test]
fn screen_keys_and_defaults_follow_the_table() {
    let specs = tunables();
    assert!(specs.len() >= 3 + SCREEN_ROWS.len());
    for (spec, (key, default, integer)) in specs[3..].iter().zip(SCREEN_ROWS) {
        assert_eq!(spec.key, key);
        assert_eq!(spec.default, default, "default of {key}");
        assert_eq!(spec.integer, integer, "integer of {key}");
        assert_eq!((spec.min, spec.max), (None, None), "{key} states no range");
        assert_eq!(spec.rebuild, "pipeline", "{key}");
    }
    let store = GiTunables::new();
    assert_eq!(store.value("probeScreenDistance"), Some(8.0));
    assert_eq!(store.value("probeScreenSteps"), Some(32.0));
    assert_eq!(store.value("probeScreenThickness"), Some(0.15));
}

#[test]
fn probe_screen_steps_is_integer_and_refuses_a_fraction() {
    let mut store = GiTunables::new();
    let before = store.clone();
    let spec = tunables()
        .iter()
        .find(|spec| spec.key == "probeScreenSteps");
    assert!(spec.is_some_and(|spec| spec.integer));
    let fraction = store
        .set("probeScreenSteps", 2.5)
        .expect_err("an integer row");
    assert_eq!(fraction.code(), "gi.invalid");
    assert_eq!(store, before);
    assert!(store.set("probeScreenSteps", 16.0).expect("an integer"));
    assert_eq!(store.value("probeScreenSteps"), Some(16.0));
}

/// The ReSTIR GI rows of `pass.gi.screen` that card 3/5 adds, in table order: key, default,
/// whether the row is integer and its rebuild.
const RESTIR_ROWS: [(&str, f64, bool, &str); 13] = [
    ("restirJacobianMax", 10.0, false, "pipeline"),
    ("restirVisibilitySteps", 32.0, true, "pipeline"),
    ("restirVisibilityDistance", 320.0, false, "pipeline"),
    ("restirScreenDistance", 8.0, false, "pipeline"),
    ("restirScreenSteps", 32.0, true, "pipeline"),
    ("restirScreenThickness", 0.15, false, "pipeline"),
    ("restirVolumeDistance", 320.0, false, "pipeline"),
    ("restirVolumeSteps", 40.0, true, "pipeline"),
    ("restirVolumeHitVoxels", 0.85, false, "pipeline"),
    ("restirSpatialSamples", 4.0, true, "none"),
    ("restirSpatialRadius", 24.0, false, "none"),
    ("restirBoilingFactor", 20.0, false, "none"),
    ("restirNormalCos", 0.9, false, "none"),
];

#[test]
fn restir_keys_and_defaults_follow_the_table() {
    let specs = tunables();
    let start = 3 + SCREEN_ROWS.len();
    assert!(specs.len() >= start + RESTIR_ROWS.len());
    let store = GiTunables::new();
    for (spec, (key, default, integer, rebuild)) in specs[start..].iter().zip(RESTIR_ROWS) {
        assert_eq!(spec.key, key);
        assert_eq!(spec.default, default, "default of {key}");
        assert_eq!(spec.integer, integer, "integer of {key}");
        assert_eq!((spec.min, spec.max), (None, None), "{key} states no range");
        assert_eq!(spec.rebuild, rebuild, "{key}");
        assert_eq!(store.value(key), Some(default), "{key}");
    }
}

#[test]
fn restir_steps_and_spatial_samples_refuse_a_fraction() {
    let mut store = GiTunables::new();
    let before = store.clone();
    for key in [
        "restirVisibilitySteps",
        "restirScreenSteps",
        "restirVolumeSteps",
        "restirSpatialSamples",
    ] {
        let fraction = store.set(key, 2.5).expect_err("an integer row");
        assert_eq!(fraction.code(), "gi.invalid", "{key}");
        assert_eq!(store, before, "{key}");
    }
    assert!(store.set("restirSpatialSamples", 6.0).expect("an integer"));
    assert_eq!(store.value("restirSpatialSamples"), Some(6.0));
}

#[test]
fn restir_constants_pair_the_overrides_with_the_current_values() {
    let mut store = GiTunables::new();
    assert!(store.set("restirJacobianMax", 5.0).expect("a finite value"));
    let constants = store.constants(&[
        ("RAA_RGI_JACOBIAN_MAX", "restirJacobianMax"),
        ("RAA_RGI_SCREEN_STEPS", "restirScreenSteps"),
    ]);
    assert_eq!(
        constants,
        [
            ("RAA_RGI_JACOBIAN_MAX", 5.0),
            ("RAA_RGI_SCREEN_STEPS", 32.0)
        ]
    );
}

/// The `pass.gi.area-shadow` override rows that card 4/5 adds, in table order: key, default and
/// whether the row is integer. Each rebuilds the area-shadow pipelines.
const AREA_ROWS: [(&str, f64, bool); 5] = [
    ("restirDepthTolerance", 0.1, false),
    ("spatialStride", 2.0, true),
    ("normalPower", 8.0, true),
    ("temporalAlpha", 0.1, false),
    ("depthTolerance", 0.05, false),
];

#[test]
fn area_shadow_keys_and_defaults_follow_the_table() {
    let specs = tunables();
    let start = 3 + SCREEN_ROWS.len() + RESTIR_ROWS.len();
    assert!(specs.len() >= start + AREA_ROWS.len());
    let store = GiTunables::new();
    for (spec, (key, default, integer)) in specs[start..].iter().zip(AREA_ROWS) {
        assert_eq!(spec.key, key);
        assert_eq!(spec.default, default, "default of {key}");
        assert_eq!(spec.integer, integer, "integer of {key}");
        assert_eq!((spec.min, spec.max), (None, None), "{key} states no range");
        assert_eq!(spec.rebuild, "pipeline", "{key}");
        assert_eq!(store.value(key), Some(default), "{key}");
    }
    assert_eq!(store.value("temporalAlpha"), Some(0.1));
}

#[test]
fn area_shadow_temporal_alpha_accepts_one_and_strides_refuse_a_fraction() {
    let mut store = GiTunables::new();
    assert!(
        store
            .set("temporalAlpha", 1.0)
            .expect("the range is unstated")
    );
    assert_eq!(store.value("temporalAlpha"), Some(1.0));
    let before = store.clone();
    for key in ["spatialStride", "normalPower"] {
        let fraction = store.set(key, 2.5).expect_err("an integer row");
        assert_eq!(fraction.code(), "gi.invalid", "{key}");
        assert_eq!(store, before, "{key}");
    }
    let constants = store.constants(&[
        ("RAA_AREA_TEMPORAL_ALPHA", "temporalAlpha"),
        ("RAA_AREA_DEPTH_TOLERANCE", "depthTolerance"),
    ]);
    assert_eq!(
        constants,
        [
            ("RAA_AREA_TEMPORAL_ALPHA", 1.0),
            ("RAA_AREA_DEPTH_TOLERANCE", 0.05)
        ]
    );
}

/// The six `pass.gi.area-shadow` uniform rows (store keys with the `area` prefix), in table
/// order after the override rows: key, default and whether the row is integer.
const AREA_UNIFORM_ROWS: [(&str, f64, bool); 6] = [
    ("areaRestirCandidates", 8.0, true),
    ("areaRestirSpatialSamples", 4.0, true),
    ("areaRestirSpatialRadius", 30.0, false),
    ("areaRestirMCap", 20.0, false),
    ("areaRestirBoilingFactor", 8.0, false),
    ("areaRestirNormalCos", 0.9, false),
];

#[test]
fn area_shadow_uniform_keys_and_defaults_follow_the_table() {
    let specs = tunables();
    let start = 3 + SCREEN_ROWS.len() + RESTIR_ROWS.len() + AREA_ROWS.len();
    assert!(specs.len() >= start + AREA_UNIFORM_ROWS.len());
    for (spec, (key, default, integer)) in specs[start..].iter().zip(AREA_UNIFORM_ROWS) {
        assert_eq!(spec.key, key);
        assert_eq!(spec.default, default, "default of {key}");
        assert_eq!(spec.integer, integer, "integer of {key}");
        assert_eq!((spec.min, spec.max), (None, None), "{key} states no range");
        assert_eq!(spec.rebuild, "none", "{key}");
    }
}

#[test]
fn area_shadow_uniform_rows_are_separate_from_the_restir_gi_rows() {
    let mut store = GiTunables::new();
    let before = store.clone();
    let fraction = store
        .set("areaRestirCandidates", 2.5)
        .expect_err("an integer row");
    assert_eq!(fraction.code(), "gi.invalid");
    assert_eq!(store, before);
    assert!(
        store
            .set("areaRestirSpatialRadius", 12.0)
            .expect("unstated range")
    );
    assert_eq!(store.value("areaRestirSpatialRadius"), Some(12.0));
    assert_eq!(store.value("restirSpatialRadius"), Some(24.0));
}
