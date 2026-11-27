//! m7 generation-owned foliage assets and whole-tree groups: the pending, failed and
//! ready distinction, stale completion, replacement, removal and source/proxy exclusion.
//! The pure registry and preparation core run natively; the wasm adapter is not compiled.

use std::collections::{BTreeMap, BTreeSet};

use raa_math::{Aabb, Mat4, Vec3};

mod billboard_selection {
    pub use raa_js::billboard_selection::*;
}

#[path = "../src/foliage_state.rs"]
mod foliage_state;

use foliage_state::{
    FOLIAGE_INVALID, FoliageAssets, FoliageGroupConfig, FoliageGroups, FoliageLevel, FoliagePhase,
    FoliagePrepareRecipe, FoliageStateStore, fail_foliage_record, foliage_visibility,
    prepare_foliage_record,
};

const ROOT: u32 = 10;

fn config(asset: u32, level: FoliageLevel) -> FoliageGroupConfig {
    FoliageGroupConfig {
        asset,
        full: vec![1, 2],
        root: ROOT,
        switch_pixels: 100.0,
        level,
    }
}

fn roots() -> BTreeSet<u32> {
    [ROOT].into_iter().collect()
}

fn live() -> BTreeMap<u32, bool> {
    [(1, true), (2, true)].into_iter().collect()
}

fn groups(asset: u32) -> FoliageGroups {
    let mut groups = FoliageGroups::new();
    groups
        .set(
            "tree",
            Some(config(asset, FoliageLevel::Auto)),
            &roots(),
            &live(),
        )
        .expect("valid group");
    groups
}

fn recipe(eye: [f32; 3], extent: f32) -> FoliagePrepareRecipe {
    let half = extent * 0.5;
    FoliagePrepareRecipe {
        eye: Vec3::from_array(eye),
        target: Vec3::ZERO,
        up: Vec3::Y,
        fov_y_rad: 60.0_f32.to_radians(),
        near: 0.1,
        ortho_height: None,
        viewport: (1000, 1000),
        root_world: Mat4::IDENTITY,
        bounds: Aabb {
            min: Vec3::splat(-half),
            max: Vec3::splat(half),
        },
    }
}

fn prepare(
    groups: &mut FoliageGroups,
    assets: &FoliageAssets,
    asset: u32,
    view: u8,
    extent: f32,
) -> u8 {
    let entry = assets.get(asset);
    let record = groups.get_mut("tree").expect("registered group");
    prepare_foliage_record(record, view, &recipe([0.0, 0.0, 5.0], extent), entry)
        .expect("valid preparation")
}

#[test]
fn pending_failed_and_ready_are_distinct_and_keep_sources() {
    let mut assets = FoliageAssets::new();
    assert!(assets.is_empty());
    let id = assets.begin(1, 0, 32, 90);
    assert_eq!(assets.len(), 1);
    assert!(assets.contains(id));
    let mut groups = groups(id);

    assert_eq!(prepare(&mut groups, &assets, id, 0, 0.1), 0);
    let state = &groups.get("tree").expect("registered").state;
    assert_eq!(state.phase, FoliagePhase::Pending);
    assert_eq!(state.level, 0);
    assert_eq!(state.asset, None);
    assert_eq!(state.generation, 1);
    let visibility = foliage_visibility(state.phase, state.level);
    assert!(
        visibility.source && !visibility.proxy,
        "pending keeps sources"
    );

    assert!(assets.fail(id, 1, "bake failed"));
    prepare(&mut groups, &assets, id, 0, 0.1);
    let state = &groups.get("tree").expect("registered").state;
    assert_eq!(state.phase, FoliagePhase::Failed);
    assert_eq!(state.error.as_deref(), Some("bake failed"));
    let visibility = foliage_visibility(state.phase, state.level);
    assert!(
        visibility.source && !visibility.proxy,
        "failed keeps sources"
    );

    assets.replace(id, 2, 0, 32, 90).expect("known asset");
    assert!(assets.complete(id, 2, 40, 41, 300));
    assert_eq!(prepare(&mut groups, &assets, id, 0, 0.1), 1);
    let state = &groups.get("tree").expect("registered").state;
    assert_eq!(state.phase, FoliagePhase::Ready);
    assert_eq!(state.asset, Some(id));
    assert_eq!(state.generation, 2);
    assert_eq!(state.level, 1);
    let visibility = foliage_visibility(state.phase, state.level);
    assert!(
        !visibility.source && visibility.proxy,
        "ready LOD1 shows proxy"
    );

    assert_eq!(prepare(&mut groups, &assets, id, 0, 1.0), 0);
    let state = &groups.get("tree").expect("registered").state;
    assert_eq!(state.level, 0);
    let visibility = foliage_visibility(state.phase, state.level);
    assert!(
        visibility.source && !visibility.proxy,
        "ready LOD0 shows source"
    );
}

#[test]
fn a_stale_completion_is_ignored() {
    let mut assets = FoliageAssets::new();
    let id = assets.begin(1, 0, 32, 90);
    assets.replace(id, 2, 0, 32, 90).expect("known asset");

    assert!(
        !assets.complete(id, 1, 40, 41, 300),
        "generation 1 is stale"
    );
    let entry = assets.get(id).expect("published");
    assert_eq!(entry.phase, FoliagePhase::Pending);
    assert_eq!(entry.generation, 2);

    assert!(assets.complete(id, 2, 40, 41, 300));
    assert_eq!(
        assets.get(id).expect("published").phase,
        FoliagePhase::Ready
    );
    assert!(!assets.complete(99, 2, 0, 0, 0), "unknown ids are ignored");
}

#[test]
fn a_replacement_generation_moves_the_group_state() {
    let mut assets = FoliageAssets::new();
    let id = assets.begin(1, 0, 32, 90);
    assets.complete(id, 1, 40, 41, 300);
    let mut groups = groups(id);
    prepare(&mut groups, &assets, id, 0, 1.0);
    assert_eq!(groups.get("tree").expect("registered").state.generation, 1);

    assets.replace(id, 2, 4, 16, 90).expect("known asset");
    assert_eq!(prepare(&mut groups, &assets, id, 0, 1.0), 0);
    let state = &groups.get("tree").expect("registered").state;
    assert_eq!(state.phase, FoliagePhase::Pending);
    assert_eq!(state.generation, 2);
    assert_eq!(state.mode, 4);
    assert_eq!(state.asset, None);

    assert!(assets.complete(id, 2, 50, 51, 800));
    prepare(&mut groups, &assets, id, 0, 1.0);
    let state = &groups.get("tree").expect("registered").state;
    assert_eq!(state.phase, FoliagePhase::Ready);
    assert_eq!(state.generation, 2);
    assert_eq!(state.mode, 4);
}

#[test]
fn removal_detaches_dependent_groups_and_restores_visibility() {
    let mut store = FoliageStateStore::new();
    let id = store.assets.begin(1, 0, 32, 90);
    store
        .groups
        .set(
            "tree",
            Some(config(id, FoliageLevel::Auto)),
            &roots(),
            &live(),
        )
        .expect("valid group");
    store
        .groups
        .set(
            "other",
            Some(FoliageGroupConfig {
                asset: id + 100,
                ..config(id, FoliageLevel::Auto)
            }),
            &roots(),
            &live(),
        )
        .expect("valid group");

    let (asset, restored) = store.remove_asset(id);
    assert!(asset.is_some());
    assert_eq!(restored, vec![(1, true), (2, true)]);
    assert!(
        store.groups.get("tree").is_none(),
        "dependent group detached"
    );
    assert!(store.groups.get("other").is_some(), "other group untouched");

    let mut groups = store.groups;
    assert_eq!(
        groups
            .set("other", None, &roots(), &live())
            .expect("idempotent removal"),
        vec![(1, true), (2, true)]
    );
}

#[test]
fn a_missing_asset_fails_the_group_and_keeps_sources() {
    let mut assets = FoliageAssets::new();
    let id = assets.begin(1, 0, 32, 90);
    assets.complete(id, 1, 40, 41, 300);
    let mut groups = groups(id);
    prepare(&mut groups, &assets, id, 0, 1.0);

    let removed = assets.remove(id);
    assert!(removed.is_some());
    assert!(!assets.contains(id));
    prepare(&mut groups, &assets, id, 0, 1.0);
    let state = &groups.get("tree").expect("registered").state;
    assert_eq!(state.phase, FoliagePhase::Failed);
    assert_eq!(state.asset, None);
    assert!(
        state
            .error
            .as_deref()
            .is_some_and(|e| e.contains("not published"))
    );
    let visibility = foliage_visibility(state.phase, state.level);
    assert!(visibility.source && !visibility.proxy);
}

#[test]
fn source_and_proxy_are_never_both_visible() {
    for phase in [
        FoliagePhase::Pending,
        FoliagePhase::Ready,
        FoliagePhase::Failed,
    ] {
        for level in [0, 1] {
            let visibility = foliage_visibility(phase, level);
            assert!(
                !(visibility.source && visibility.proxy),
                "{phase:?} level {level} shows both"
            );
            if phase == FoliagePhase::Ready && level == 1 {
                assert!(visibility.proxy, "only a ready LOD1 shows the proxy");
            } else {
                assert!(
                    visibility.source,
                    "{phase:?} level {level} hides the source"
                );
            }
        }
    }
}

#[test]
fn hull_proxy_level_uses_size_not_facing_rotation() {
    let mut assets = FoliageAssets::new();
    let id = assets.begin(1, 0, 32, 90);
    assets.complete(id, 1, 40, 41, 300);
    let mut groups = groups(id);
    let record = groups.get_mut("tree").expect("registered group");
    let front = prepare_foliage_record(record, 0, &recipe([0.0, 0.0, 5.0], 0.1), assets.get(id))
        .expect("valid preparation");
    let side = prepare_foliage_record(record, 1, &recipe([5.0, 0.0, 0.0], 0.1), assets.get(id))
        .expect("valid preparation");
    assert_eq!(front, 1);
    assert_eq!(side, 1, "the level ignores the camera azimuth");
    assert_eq!(record.views[&0], 1);
    assert_eq!(record.views[&1], 1);
}

#[test]
fn an_invalid_group_is_refused_before_mutation() {
    let mut groups = FoliageGroups::new();
    let cases = [
        FoliageGroupConfig {
            asset: 1,
            full: Vec::new(),
            root: ROOT,
            switch_pixels: 100.0,
            level: FoliageLevel::Auto,
        },
        FoliageGroupConfig {
            asset: 1,
            full: vec![1, 1],
            root: ROOT,
            switch_pixels: 100.0,
            level: FoliageLevel::Auto,
        },
        FoliageGroupConfig {
            asset: 1,
            full: vec![1],
            root: 99,
            switch_pixels: 100.0,
            level: FoliageLevel::Auto,
        },
        FoliageGroupConfig {
            asset: 1,
            full: vec![1],
            root: ROOT,
            switch_pixels: 0.0,
            level: FoliageLevel::Auto,
        },
    ];
    for case in cases {
        let error = groups
            .set("tree", Some(case), &roots(), &live())
            .expect_err("invalid group");
        assert_eq!(error.code(), FOLIAGE_INVALID);
        assert!(groups.get("tree").is_none());
    }
}

#[test]
fn a_failed_view_records_the_diagnostic_and_returns_to_source() {
    let mut assets = FoliageAssets::new();
    let id = assets.begin(1, 0, 32, 90);
    assets.complete(id, 1, 40, 41, 300);
    let mut groups = groups(id);
    prepare(&mut groups, &assets, id, 0, 0.1);
    let record = groups.get_mut("tree").expect("registered group");
    fail_foliage_record(record, 1, "source instance 2 is stale").expect("valid view");
    assert_eq!(record.state.phase, FoliagePhase::Failed);
    assert_eq!(
        record.state.error.as_deref(),
        Some("source instance 2 is stale")
    );
    assert_eq!(record.state.level, 0);
    assert_eq!(record.views[&0], 1, "view 0 keeps its own level");
}

#[test]
fn the_store_exposes_its_groups_and_their_state() {
    let mut store = FoliageStateStore::new();
    let id = store.assets.begin(1, 0, 32, 90);
    store
        .groups
        .set(
            "tree",
            Some(config(id, FoliageLevel::Auto)),
            &roots(),
            &live(),
        )
        .expect("valid group");
    assert_eq!(store.groups.iter().count(), 1);
    for (_, record) in store.groups.iter_mut() {
        record.state.generation = 7;
    }
    let state = store.state("tree").expect("registered group");
    assert_eq!(state.generation, 7);
    assert!(store.state("missing").is_none());
}

#[test]
fn a_forced_level_still_waits_for_the_asset() {
    let mut assets = FoliageAssets::new();
    let id = assets.begin(1, 0, 32, 90);
    let mut groups = FoliageGroups::new();
    groups
        .set(
            "tree",
            Some(config(id, FoliageLevel::Proxy)),
            &roots(),
            &live(),
        )
        .expect("valid group");
    let record = groups.get_mut("tree").expect("registered group");
    prepare_foliage_record(record, 0, &recipe([0.0, 0.0, 5.0], 1.0), assets.get(id))
        .expect("valid preparation");
    assert_eq!(record.state.level, 0, "a pending proxy stays on the source");
    assert!(foliage_visibility(record.state.phase, record.state.level).source);

    assets.complete(id, 1, 40, 41, 300);
    groups
        .set(
            "tree",
            Some(config(id, FoliageLevel::Full)),
            &roots(),
            &live(),
        )
        .expect("valid group");
    let record = groups.get_mut("tree").expect("registered group");
    prepare_foliage_record(record, 0, &recipe([0.0, 0.0, 5.0], 0.1), assets.get(id))
        .expect("valid preparation");
    assert_eq!(
        record.state.level, 0,
        "a forced full level ignores the size"
    );
}
