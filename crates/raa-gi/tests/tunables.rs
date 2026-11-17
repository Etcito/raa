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
    assert_eq!(keys, ["behind", "resetRays", "directChange"]);
    for (spec, (key, default)) in specs.iter().zip(VOLUME_ROWS) {
        assert_eq!(spec.key, key);
        assert_eq!(spec.default, default, "default of {key}");
        assert_eq!(spec.min, None, "{key} states no range");
        assert_eq!(spec.max, None, "{key} states no range");
        assert_eq!(spec.rebuild, "pipeline", "{key}");
    }
    let integer: Vec<bool> = specs.iter().map(|spec| spec.integer).collect();
    assert_eq!(integer, [false, true, false]);
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
