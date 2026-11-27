//! m36 billboard group preparation: pending and failed gates, per-view hysteresis,
//! camera changes, two views and the removal handoff. The pure preparation core runs
//! natively; the wasm adapter is not compiled here.

use std::collections::{BTreeMap, BTreeSet};

use raa_math::{Aabb, Mat4, Vec3};

#[allow(dead_code)]
#[path = "../src/api/m36_billboard_groups.rs"]
mod m36;

use m36::{
    BILLBOARD_INVALID, BillboardGroupConfig, BillboardGroupRegistry, BillboardGroupStatus,
    BillboardLevel, BillboardPrepareRecipe, fail_billboard_record, prepare_billboard_record,
};

fn config(root: u32, full: &[u32], proxy: u32, level: BillboardLevel) -> BillboardGroupConfig {
    BillboardGroupConfig {
        root,
        full: full.to_vec(),
        proxy,
        bake: 7,
        switch_pixels: 100.0,
        bias: 0,
        level,
    }
}

fn registered(visibility: &[(u32, bool)], level: BillboardLevel) -> BillboardGroupRegistry {
    let mut registry = BillboardGroupRegistry::new();
    let roots: BTreeSet<u32> = [10].into_iter().collect();
    let live: BTreeMap<u32, bool> = visibility.iter().copied().collect();
    registry
        .set("tree", Some(config(10, &[1], 2, level)), &roots, &live)
        .expect("valid group");
    registry
}

fn recipe(eye: [f32; 3], extent: f32) -> BillboardPrepareRecipe {
    let half = extent * 0.5;
    BillboardPrepareRecipe {
        eye: Vec3::from_array(eye),
        target: Vec3::ZERO,
        up: Vec3::Y,
        fov_y_rad: 60.0_f32.to_radians(),
        near: 0.1,
        ortho_height: None,
        viewport: (1000, 1000),
        root_world: Mat4::IDENTITY,
        proxy_world: Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)),
        bounds: Aabb {
            min: Vec3::splat(-half),
            max: Vec3::splat(half),
        },
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Gate {
    Waiting,
    Failed(String),
    Ready,
}

/// A gated action proceeds only when the preparation reports readiness.
fn gate(status: &BillboardGroupStatus) -> Gate {
    if status.ready {
        Gate::Ready
    } else if let Some(error) = &status.error {
        Gate::Failed(error.clone())
    } else {
        Gate::Waiting
    }
}

fn run_action(gate: &Gate, ran: &mut u32) {
    if *gate == Gate::Ready {
        *ran += 1;
    }
}

#[test]
fn pending_bake_keeps_lod0_and_the_gate_waits() {
    let mut registry = registered(&[(1, true), (2, true)], BillboardLevel::Auto);
    let mut ran = 0;
    {
        let record = registry.get_mut_for_test("tree");
        let prepared = prepare_billboard_record(record, 0, &recipe([0.0, 0.0, 5.0], 0.1), false)
            .expect("valid recipe");
        assert_eq!(prepared.level, 0, "pending keeps LOD0");
        assert_eq!(record.status.level, 0);
        assert_eq!(record.status.direction, 0);
        assert!(!record.status.ready, "pending never reports ready");
        assert_eq!(record.status.error, None, "pending is not a failure");
        assert_eq!(record.views[&0].level, 0);
        assert_eq!(record.views[&0].direction, 0);
        let gate = gate(&record.status);
        assert_eq!(gate, Gate::Waiting);
        run_action(&gate, &mut ran);
    }
    assert_eq!(ran, 0, "the gated action waits for the bake");
}

#[test]
fn a_failed_group_reports_the_failure_and_the_gate_never_proceeds() {
    let mut registry = registered(&[(1, true), (2, true)], BillboardLevel::Auto);
    let mut ran = 0;
    {
        let record = registry.get_mut_for_test("tree");
        fail_billboard_record(record, 0, "LOD0 instance 1 is stale").expect("valid view");
        assert_eq!(record.status.level, 0);
        assert!(!record.status.ready);
        assert_eq!(
            record.status.error.as_deref(),
            Some("LOD0 instance 1 is stale")
        );
        assert_eq!(record.views[&0].level, 0);
        let gate = gate(&record.status);
        assert!(matches!(gate, Gate::Failed(_)));
        run_action(&gate, &mut ran);
    }
    assert_eq!(ran, 0, "a failed preparation never lets the action run");
}

#[test]
fn camera_changes_and_two_views_keep_independent_records() {
    let mut registry = registered(&[(1, true), (2, true)], BillboardLevel::Auto);
    {
        let record = registry.get_mut_for_test("tree");
        prepare_billboard_record(record, 0, &recipe([0.0, 0.0, 5.0], 1.0), false).unwrap();
        assert_eq!(record.views[&0].direction, 0);

        prepare_billboard_record(record, 0, &recipe([5.0, 0.0, 0.0], 1.0), false).unwrap();
        assert_eq!(
            record.views[&0].direction, 2,
            "the camera change turns view 0"
        );

        prepare_billboard_record(record, 1, &recipe([0.0, 0.0, -5.0], 1.0), false).unwrap();
        assert_eq!(record.views[&1].direction, 4);
        assert_eq!(record.views[&0].direction, 2, "view 0 keeps its own record");
        assert_eq!(record.status.direction, 4, "status is the last preparation");
    }
}

#[test]
fn per_view_hysteresis_is_independent() {
    let mut registry = registered(&[(1, true), (2, true)], BillboardLevel::Auto);
    {
        let record = registry.get_mut_for_test("tree");
        prepare_billboard_record(record, 0, &recipe([0.0, 0.0, 5.0], 0.1), true).unwrap();
        assert_eq!(record.views[&0].level, 1, "small on screen enters LOD1");

        prepare_billboard_record(record, 1, &recipe([0.0, 0.0, 5.0], 1.0), true).unwrap();
        assert_eq!(record.views[&1].level, 0, "view 1 starts at LOD0");

        prepare_billboard_record(record, 0, &recipe([0.0, 0.0, 5.0], 1.0), true).unwrap();
        assert_eq!(
            record.views[&0].level, 0,
            "view 0 re-enters LOD0 above the hysteresis band"
        );
        assert_eq!(record.views[&1].level, 0, "view 1 is unchanged");
    }
}

#[test]
fn removal_and_replacement_hand_back_the_saved_visibility() {
    let mut registry = registered(&[(1, false), (2, true)], BillboardLevel::Auto);
    let roots: BTreeSet<u32> = [10].into_iter().collect();
    let live: BTreeMap<u32, bool> = [(1, false), (2, true)].into_iter().collect();
    assert_eq!(
        registry
            .set(
                "tree",
                Some(config(10, &[1], 2, BillboardLevel::Auto)),
                &roots,
                &live
            )
            .unwrap(),
        vec![(1, false), (2, true)],
        "replacement restores the prior handles in ascending order"
    );
    assert!(registry.get("tree").expect("registered").views.is_empty());
    assert_eq!(
        registry.set("tree", None, &roots, &live).unwrap(),
        vec![(1, false), (2, true)]
    );
    assert!(registry.get("tree").is_none());
    assert_eq!(
        registry.set("tree", None, &roots, &live).unwrap(),
        Vec::<(u32, bool)>::new(),
        "removal of an absent id succeeds"
    );
}

#[test]
fn an_invalid_view_index_is_refused_before_mutation() {
    let mut registry = registered(&[(1, true), (2, true)], BillboardLevel::Auto);
    let record = registry.get_mut_for_test("tree");
    let before = record.status.clone();
    let error = prepare_billboard_record(record, 4, &recipe([0.0, 0.0, 5.0], 1.0), false)
        .expect_err("view 4 is invalid");
    assert_eq!(error.code(), BILLBOARD_INVALID);
    assert_eq!(record.status, before);
    assert!(record.views.is_empty());
}
