//! m36 billboard groups: the registry exports the group methods; validation, replacement,
//! removal and the initial not-ready state run natively.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use raa_core::RaaError;

#[allow(dead_code)]
#[path = "../src/api/m36_billboard_groups.rs"]
mod m36;

use m36::{
    BILLBOARD_INVALID, BillboardGroupConfig, BillboardGroupRegistry, BillboardLevel, validate_group,
};

fn config(root: u32, full: &[u32], proxy: u32) -> BillboardGroupConfig {
    BillboardGroupConfig {
        root,
        full: full.to_vec(),
        proxy,
        bake: 7,
        switch_pixels: 100.0,
        bias: 0,
        level: BillboardLevel::Auto,
    }
}

fn live(entries: &[(u32, bool)]) -> BTreeMap<u32, bool> {
    entries.iter().copied().collect()
}

fn handles(entries: &[u32]) -> BTreeSet<u32> {
    entries.iter().copied().collect()
}

fn code_of(e: &RaaError) -> &'static str {
    e.code()
}

#[test]
fn registry_exports_the_group_methods() {
    let path = Path::new(env!("OUT_DIR")).join("api_registry.rs");
    let reg = fs::read_to_string(&path).expect("api_registry.rs");
    assert!(reg.contains("mod m36_billboard_groups;"));
    assert!(reg.contains("m36_billboard_groups::install(ctx)?;"));
    assert!(reg.contains("m36_billboard_groups::frame(st, gpu, input)?;"));
    let src = include_str!("../src/api/m36_billboard_groups.rs");
    for name in ["setBillboardGroup", "billboardGroupState"] {
        assert!(src.contains(&format!("js_name = {name}")), "no {name}");
    }
    for name in [
        "bakeBillboard",
        "removeBillboardBake",
        "createBillboardProxy",
    ] {
        assert!(
            !src.contains(&format!("js_name = {name}")),
            "declaration-only {name} has a binding"
        );
    }
}

#[test]
fn a_new_group_is_not_ready_and_has_no_views() {
    let mut registry = BillboardGroupRegistry::new();
    let instances = live(&[(1, true), (2, false)]);
    assert_eq!(
        registry.set(
            "tree",
            Some(config(10, &[1], 2)),
            &handles(&[10]),
            &instances
        ),
        Ok(Vec::new())
    );
    let record = registry.get("tree").expect("registered");
    assert_eq!(record.status.level, 0);
    assert_eq!(record.status.direction, 0);
    assert!(!record.status.ready);
    assert_eq!(record.status.error, None);
    assert!(record.views.is_empty());
    assert_eq!(record.original_visibility, live(&[(1, true), (2, false)]));
}

#[test]
fn invalid_input_is_refused_without_mutation() {
    let mut registry = BillboardGroupRegistry::new();
    let instances = live(&[(1, true), (2, false), (3, true), (4, false)]);
    let roots = handles(&[10, 11]);
    registry
        .set("a", Some(config(10, &[1], 2)), &roots, &instances)
        .expect("valid first group");
    let before = registry.get("a").cloned().expect("group a");

    let cases = [
        config(99, &[3], 4),
        config(10, &[], 4),
        config(10, &[3, 3], 4),
        config(10, &[4], 4),
        config(10, &[99], 4),
        config(10, &[3], 99),
        config(10, &[2], 4),
        config(10, &[3], 1),
    ];
    for case in cases {
        let error = registry
            .set("b", Some(case), &roots, &instances)
            .expect_err("invalid group must be refused");
        assert_eq!(code_of(&error), BILLBOARD_INVALID);
        assert!(registry.get("b").is_none());
        assert_eq!(registry.get("a"), Some(&before));
    }

    let mut zero = config(10, &[3], 4);
    zero.switch_pixels = 0.0;
    let mut nan = config(10, &[3], 4);
    nan.switch_pixels = f32::NAN;
    let mut high_bias = config(10, &[3], 4);
    high_bias.bias = 4;
    for case in [zero, nan, high_bias] {
        let error = registry
            .set("b", Some(case), &roots, &instances)
            .expect_err("invalid settings must be refused");
        assert_eq!(code_of(&error), BILLBOARD_INVALID);
        assert!(registry.get("b").is_none());
    }

    let error = registry
        .set("", Some(config(10, &[3], 4)), &roots, &instances)
        .expect_err("an empty id is refused");
    assert_eq!(code_of(&error), BILLBOARD_INVALID);
    assert_eq!(registry.get("a"), Some(&before));
}

#[test]
fn membership_is_disjoint_across_groups() {
    let mut registry = BillboardGroupRegistry::new();
    let instances = live(&[(1, true), (2, false), (3, true), (4, false)]);
    let roots = handles(&[10]);
    registry
        .set("a", Some(config(10, &[1], 2)), &roots, &instances)
        .expect("valid first group");

    for case in [
        config(10, &[1], 4),
        config(10, &[3], 1),
        config(10, &[2], 4),
    ] {
        let error = registry
            .set("b", Some(case), &roots, &instances)
            .expect_err("overlapping membership must be refused");
        assert_eq!(code_of(&error), BILLBOARD_INVALID);
    }
    assert!(
        registry
            .set("b", Some(config(10, &[3], 4)), &roots, &instances)
            .is_ok()
    );
}

#[test]
fn replacement_and_removal_return_saved_visibility_in_handle_order() {
    let mut registry = BillboardGroupRegistry::new();
    let instances = live(&[(1, true), (2, false), (3, true), (4, false)]);
    let roots = handles(&[10]);
    assert_eq!(
        registry.set("tree", Some(config(10, &[3, 1], 2)), &roots, &instances),
        Ok(Vec::new())
    );
    assert_eq!(
        registry.set("tree", Some(config(10, &[4], 2)), &roots, &instances),
        Ok(vec![(1, true), (2, false), (3, true)])
    );
    let record = registry.get("tree").expect("replaced group");
    assert_eq!(record.original_visibility, live(&[(2, false), (4, false)]));
    assert!(record.views.is_empty());
    assert!(!record.status.ready);

    assert_eq!(
        registry.set("tree", None, &roots, &instances),
        Ok(vec![(2, false), (4, false)])
    );
    assert!(registry.get("tree").is_none());
    assert_eq!(
        registry.set("tree", None, &roots, &instances),
        Ok(Vec::new())
    );
}

#[test]
fn removal_of_an_absent_id_keeps_other_groups() {
    let mut registry = BillboardGroupRegistry::new();
    let instances = live(&[(1, true), (2, false), (3, true)]);
    let roots = handles(&[10]);
    registry
        .set("a", Some(config(10, &[1], 2)), &roots, &instances)
        .expect("valid group");
    assert_eq!(
        registry.set("missing", None, &roots, &instances),
        Ok(Vec::new())
    );
    assert!(registry.get("a").is_some());
}

#[test]
fn validate_group_checks_each_rule() {
    let instances = live(&[(1, true), (2, false), (3, true)]);
    let roots = handles(&[10]);
    let none = BTreeSet::new();
    assert!(validate_group(&config(10, &[1], 2), &roots, &instances, &none).is_ok());

    let refused = [
        config(11, &[1], 2),
        config(10, &[], 2),
        config(10, &[1, 1], 2),
        config(10, &[2], 2),
        config(10, &[9], 2),
        config(10, &[1], 9),
    ];
    for case in refused {
        let error = validate_group(&case, &roots, &instances, &none).expect_err("invalid group");
        assert_eq!(code_of(&error), BILLBOARD_INVALID);
    }

    let error = validate_group(&config(10, &[1], 2), &roots, &instances, &handles(&[1]))
        .expect_err("taken membership");
    assert_eq!(code_of(&error), BILLBOARD_INVALID);

    let mut zero = config(10, &[1], 2);
    zero.switch_pixels = 0.0;
    let mut infinite = config(10, &[1], 2);
    infinite.switch_pixels = f32::INFINITY;
    let mut low_bias = config(10, &[1], 2);
    low_bias.bias = -4;
    for case in [zero, infinite, low_bias] {
        let error = validate_group(&case, &roots, &instances, &none).expect_err("invalid settings");
        assert_eq!(code_of(&error), BILLBOARD_INVALID);
    }
}
