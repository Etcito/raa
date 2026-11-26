//! m38 asynchronous foliage preparation: descriptor validation, mode and settings
//! mapping, the generation-exclusive material lease, disposed/stale completion handling
//! and the removal order that never releases the caller-owned material lease. The pure
//! core of the api file runs natively; no GPU is involved.

use std::fs;
use std::path::Path;

use raa_core::{MaterialId, RaaError};
use raa_lit::foliage_bake::IMPOSTOR_VIEWS;

mod foliage_state {
    pub use raa_js::foliage_state::*;
}

#[allow(dead_code)]
#[path = "../src/api/m38_foliage.rs"]
mod m38;

use m38::{
    CompletionTarget, FOLIAGE_BAKE_FAILED, FOLIAGE_TEXTURE_RESOLUTION, FoliageAssetInfo,
    FoliageFields, FoliageLedger, FoliageMode, FoliageRemovalStep, FoliageSpec, bake_settings,
    completion_target, mode_name, phase_name, plan_removal, source_draw_material, translate_error,
};
use raa_js::foliage_state::{
    FOLIAGE_INVALID, FoliageAssetEntry, FoliageAssets, FoliageGroupConfig, FoliageLevel,
    FoliagePhase, FoliageState, FoliageStateStore,
};

const MATERIAL: u32 = 41;
const ASSET: u32 = 5;

fn fields() -> FoliageFields {
    FoliageFields {
        instances: Some(vec![1, 2]),
        foliage_instances: Some(vec![2]),
        root: Some(10),
        generation: Some(3),
        mode: Some("stochastic".into()),
        resolution: Some(32),
        triangles: Some(1024),
        seed: Some(7),
        views: Some(96),
        iterations: Some(64),
        silhouette_weight: Some(1.0),
        normal_weight: Some(0.5),
        edge_weight: Some(0.25),
        translucency: Some(0.35),
        cutoff: Some(0.5),
        material: Some(MATERIAL),
    }
}

fn missing(field: &str) -> FoliageFields {
    let mut fields = fields();
    match field {
        "instances" => fields.instances = None,
        "foliageInstances" => fields.foliage_instances = None,
        "root" => fields.root = None,
        "generation" => fields.generation = None,
        "mode" => fields.mode = None,
        "resolution" => fields.resolution = None,
        "triangles" => fields.triangles = None,
        "seed" => fields.seed = None,
        "views" => fields.views = None,
        "iterations" => fields.iterations = None,
        "silhouetteWeight" => fields.silhouette_weight = None,
        "normalWeight" => fields.normal_weight = None,
        "edgeWeight" => fields.edge_weight = None,
        "translucency" => fields.translucency = None,
        "cutoff" => fields.cutoff = None,
        "material" => fields.material = None,
        other => panic!("unknown field {other}"),
    }
    fields
}

fn code_of(error: &RaaError) -> &'static str {
    error.code()
}

fn assert_invalid(result: Result<FoliageSpec, RaaError>) {
    assert_eq!(code_of(&result.expect_err("refused")), FOLIAGE_INVALID);
}

fn spec() -> FoliageSpec {
    FoliageSpec::validate(fields()).expect("valid descriptor")
}

fn entry(phase: FoliagePhase) -> FoliageAssetEntry {
    FoliageAssetEntry {
        id: ASSET,
        generation: 3,
        phase,
        mode: 4,
        volume_resolution: 32,
        mesh: 7,
        proxy: 8,
        material: MATERIAL,
        triangle_count: 900,
        error: None,
    }
}

fn store_with_group() -> (FoliageStateStore, u32) {
    let mut store = FoliageStateStore::new();
    let asset = store.assets.begin(3, 4, 32, MATERIAL);
    assert!(store.assets.complete(asset, 3, 7, 8, 900));
    let config = FoliageGroupConfig {
        asset,
        full: vec![1, 2],
        root: 10,
        switch_pixels: 120.0,
        level: FoliageLevel::Auto,
    };
    let roots = [10u32].into_iter().collect();
    let live = [(1u32, true), (2u32, true)].into_iter().collect();
    store
        .groups
        .set("tree", Some(config), &roots, &live)
        .expect("valid group");
    (store, asset)
}

#[test]
fn registry_exports_the_foliage_methods() {
    let path = Path::new(env!("OUT_DIR")).join("api_registry.rs");
    let registry = fs::read_to_string(&path).expect("api_registry.rs");
    assert!(registry.contains("mod m38_foliage;"));
    assert!(registry.contains("m38_foliage::install(ctx)?;"));
    assert!(registry.contains("m38_foliage::frame(st, gpu, input)?;"));
    let source = include_str!("../src/api/m38_foliage.rs");
    for name in [
        "prepareFoliage",
        "removeFoliage",
        "setFoliageGroup",
        "foliageGroupState",
    ] {
        assert!(source.contains(&format!("js_name = {name}")), "no {name}");
    }
    assert!(source.contains("bake_context("));
    assert!(source.contains("completion.await"));
    assert!(source.contains("release_foliage(asset)"));
    assert!(!source.contains("self.state.borrow"));
}

#[test]
fn a_valid_descriptor_becomes_a_spec() {
    let spec = spec();
    assert_eq!(spec.instances, vec![1, 2]);
    assert_eq!(spec.foliage_instances, vec![2]);
    assert_eq!(spec.root, 10);
    assert_eq!(spec.generation, 3);
    assert_eq!(spec.mode, FoliageMode::Stochastic);
    assert_eq!(spec.resolution, 32);
    assert_eq!(spec.triangles, 1024);
    assert_eq!(spec.seed, 7);
    assert_eq!(spec.views, 96);
    assert_eq!(spec.iterations, 64);
    assert_eq!(spec.silhouette_weight, 1.0);
    assert_eq!(spec.normal_weight, 0.5);
    assert_eq!(spec.edge_weight, 0.25);
    assert_eq!(spec.translucency, 0.35);
    assert_eq!(spec.cutoff, 0.5);
    assert_eq!(spec.material, MATERIAL);
}

#[test]
fn every_missing_descriptor_field_refuses() {
    for field in [
        "instances",
        "foliageInstances",
        "root",
        "generation",
        "mode",
        "resolution",
        "triangles",
        "seed",
        "views",
        "iterations",
        "silhouetteWeight",
        "normalWeight",
        "edgeWeight",
        "translucency",
        "cutoff",
        "material",
    ] {
        assert_invalid(FoliageSpec::validate(missing(field)));
    }
}

#[test]
fn empty_duplicate_and_inconsistent_instance_lists_refuse() {
    let mut empty = fields();
    empty.instances = Some(Vec::new());
    assert_invalid(FoliageSpec::validate(empty));

    let mut duplicate = fields();
    duplicate.instances = Some(vec![1, 1]);
    assert_invalid(FoliageSpec::validate(duplicate));

    let mut empty_foliage = fields();
    empty_foliage.foliage_instances = Some(Vec::new());
    assert_invalid(FoliageSpec::validate(empty_foliage));

    let mut duplicate_foliage = fields();
    duplicate_foliage.foliage_instances = Some(vec![2, 2]);
    assert_invalid(FoliageSpec::validate(duplicate_foliage));

    let mut outside = fields();
    outside.foliage_instances = Some(vec![9]);
    assert_invalid(FoliageSpec::validate(outside));
}

#[test]
fn invalid_modes_and_ranges_refuse() {
    let mut mode = fields();
    mode.mode = Some("billboard".into());
    assert_invalid(FoliageSpec::validate(mode));

    let mut resolution = fields();
    resolution.resolution = Some(8);
    assert_invalid(FoliageSpec::validate(resolution));

    let mut triangles = fields();
    triangles.triangles = Some(0);
    assert_invalid(FoliageSpec::validate(triangles));

    for views in [0, 63, 129] {
        let mut views_field = fields();
        views_field.views = Some(views);
        assert_invalid(FoliageSpec::validate(views_field));
    }

    let mut iterations = fields();
    iterations.iterations = Some(0);
    assert_invalid(FoliageSpec::validate(iterations));

    for weight in [-1.0, f32::NAN, f32::INFINITY] {
        let mut bad = fields();
        bad.normal_weight = Some(weight);
        assert_invalid(FoliageSpec::validate(bad));
    }

    let mut zero_weights = fields();
    zero_weights.silhouette_weight = Some(0.0);
    zero_weights.normal_weight = Some(0.0);
    zero_weights.edge_weight = Some(0.0);
    assert_invalid(FoliageSpec::validate(zero_weights));

    for value in [-0.1, 1.1, f32::NAN] {
        let mut translucency = fields();
        translucency.translucency = Some(value);
        assert_invalid(FoliageSpec::validate(translucency));
        let mut cutoff = fields();
        cutoff.cutoff = Some(value);
        assert_invalid(FoliageSpec::validate(cutoff));
    }
}

#[test]
fn modes_round_trip_their_ranks() {
    for (rank, name) in [
        "spom",
        "pom",
        "offset",
        "shell",
        "stochastic",
        "differentiable",
    ]
    .into_iter()
    .enumerate()
    {
        let mode = FoliageMode::parse(name).expect("known mode");
        assert_eq!(mode.rank(), rank as u32);
        assert_eq!(mode.name(), name);
        assert_eq!(FoliageMode::from_rank(rank as u32), Some(mode));
    }
    assert_eq!(FoliageMode::from_rank(6), None);
    assert_eq!(
        code_of(&FoliageMode::parse("billboard").expect_err("refused")),
        FOLIAGE_INVALID
    );
    assert!(FoliageMode::Spom.is_impostor() && !FoliageMode::Stochastic.is_impostor());
    assert_eq!(mode_name(4), "stochastic");
    assert_eq!(mode_name(99), "spom");
    assert_eq!(phase_name(FoliagePhase::Pending), "pending");
    assert_eq!(phase_name(FoliagePhase::Ready), "ready");
    assert_eq!(phase_name(FoliagePhase::Failed), "failed");
}

#[test]
fn settings_carry_the_descriptor_and_the_exact_material() {
    let spec = spec();
    let settings = bake_settings(
        &spec,
        MaterialId(MATERIAL),
        [-1.0, 0.0, -1.0],
        [1.0, 2.0, 1.0],
    );
    assert_eq!(settings.material, MaterialId(MATERIAL));
    assert_eq!(settings.count, 1024);
    assert_eq!(settings.seed, 7);
    assert_eq!(settings.iterations, 64);
    assert_eq!(settings.silhouette_weight, 1.0);
    assert_eq!(settings.normal_weight, 0.5);
    assert_eq!(settings.edge_weight, 0.25);
    assert_eq!(settings.params.min, [-1.0, 0.0, -1.0]);
    assert_eq!(settings.params.max, [1.0, 2.0, 1.0]);
    assert_eq!(settings.params.generation, 3);
    assert_eq!(settings.params.mode, FoliageMode::Stochastic.rank());
    assert_eq!(settings.params.resolution, 32);
    assert_eq!(settings.params.view_count, 96);
    assert_eq!(
        settings.params.texture_resolution,
        FOLIAGE_TEXTURE_RESOLUTION
    );
    assert!(settings.validate().is_ok());

    let mut impostor = fields();
    impostor.mode = Some("spom".into());
    let spec = FoliageSpec::validate(impostor).expect("valid impostor");
    let settings = bake_settings(&spec, MaterialId(MATERIAL), [0.0; 3], [1.0; 3]);
    assert_eq!(settings.params.view_count, IMPOSTOR_VIEWS);
    assert!(settings.validate().is_ok());
}

#[test]
fn asset_info_echoes_the_same_material_and_mesh() {
    let spec = spec();
    let info = FoliageAssetInfo::from_entry(&entry(FoliagePhase::Ready));
    assert_eq!(info.id, ASSET);
    assert_eq!(info.generation, 3);
    assert_eq!(info.mode, FoliageMode::Stochastic);
    assert_eq!(info.triangles, 900);
    assert_eq!(info.volume_resolution, 32);
    assert_eq!(info.mesh, 7);
    assert_eq!(info.material, spec.material);
    assert_eq!(info.material, MATERIAL);
}

#[test]
fn the_source_draw_material_is_never_the_lod1_material() {
    let spec = spec();
    let lod1 = MaterialId(spec.material);
    assert_eq!(
        source_draw_material(None, Some(MaterialId(4))),
        MaterialId(4)
    );
    assert_eq!(
        source_draw_material(Some(MaterialId(3)), Some(MaterialId(4))),
        MaterialId(3)
    );
    assert_eq!(source_draw_material(None, None), MaterialId(0));
    assert_ne!(source_draw_material(None, Some(MaterialId(4))), lod1);
}

#[test]
fn a_material_is_generation_exclusive_until_released() {
    let mut ledger = FoliageLedger::<()>::new();
    assert_eq!(ledger.holder(MATERIAL), None);
    ledger.claim(MATERIAL, 1).expect("first claim");
    assert_eq!(ledger.holder(MATERIAL), Some(1));
    assert_eq!(
        code_of(&ledger.claim(MATERIAL, 2).expect_err("refused")),
        FOLIAGE_INVALID
    );
    assert_eq!(ledger.holder(MATERIAL), Some(1));
    assert!(!ledger.release_material(MATERIAL, 2));
    assert_eq!(ledger.holder(MATERIAL), Some(1));
    assert!(ledger.release_material(MATERIAL, 1));
    ledger.claim(MATERIAL, 2).expect("released material");
}

#[test]
fn disposed_and_late_completions_are_released_never_published() {
    let mut assets = FoliageAssets::new();
    let id = assets.begin(3, 4, 32, MATERIAL);
    assert_eq!(
        completion_target(Some(&assets), id, 3),
        CompletionTarget::Current
    );
    assert_eq!(completion_target(None, id, 3), CompletionTarget::Disposed);
    assert!(assets.complete(id, 3, 7, 8, 900));
    assert_eq!(
        completion_target(Some(&assets), id, 3),
        CompletionTarget::Stale,
        "a ready entry never accepts a second completion"
    );

    let mut ledger = FoliageLedger::new();
    let removed = assets.begin(4, 0, 16, 90);
    ledger.claim(90, removed).expect("claim");
    assert!(assets.remove(removed).is_some());
    assert_eq!(
        completion_target(Some(&assets), removed, 4),
        CompletionTarget::Stale
    );
    assert_eq!(
        ledger.holder(90),
        Some(removed),
        "a removed pending generation keeps its material claim until it settles"
    );
    assert_eq!(ledger.take(removed), None, "no native asset was installed");
    assert!(ledger.release_material(90, removed));
    assert_eq!(ledger.holder(90), None);

    let replaced = assets.begin(5, 0, 16, 91);
    assets.replace(replaced, 6, 0, 16, 91).expect("known asset");
    assert_eq!(
        completion_target(Some(&assets), replaced, 5),
        CompletionTarget::Stale
    );

    ledger.claim(91, replaced).expect("claim");
    ledger.insert(replaced, "prepared native asset");
    assert_eq!(ledger.take(replaced), Some("prepared native asset"));
    assert!(ledger.release_material(91, replaced));
}

#[test]
fn removal_detaches_groups_before_the_proxy_and_never_the_material() {
    let (mut store, asset) = store_with_group();
    let removal = plan_removal(&mut store, asset).expect("published asset removed");
    assert_eq!(removal.entry.material, MATERIAL);
    assert_eq!(removal.entry.mesh, 7);
    assert_eq!(removal.entry.proxy, 8);
    assert_eq!(removal.restored, vec![(1, true), (2, true)]);
    assert!(
        store.groups.get("tree").is_none(),
        "dependent group detached"
    );
    assert_eq!(
        removal.steps(),
        [
            FoliageRemovalStep::DetachGroups,
            FoliageRemovalStep::RemoveProxy(8),
            FoliageRemovalStep::DetachAssociation(MATERIAL),
            FoliageRemovalStep::ReleaseMesh(7),
        ],
        "the material lease and its data block have no removal step"
    );
}

#[test]
fn an_unknown_asset_refuses_before_any_mutation() {
    let (mut store, _asset) = store_with_group();
    assert_eq!(
        code_of(&plan_removal(&mut store, 999).expect_err("refused")),
        FOLIAGE_INVALID
    );
    assert!(
        store.groups.get("tree").is_some(),
        "a refused removal detached nothing"
    );
    assert!(store.assets.contains(_asset));
}

#[test]
fn failures_keep_foliage_invalid_and_translate_the_rest() {
    let kept = translate_error(&m38::invalid("bad descriptor"));
    assert_eq!(kept.code(), FOLIAGE_INVALID);
    assert_eq!(kept.message(), "bad descriptor");
    let translated = translate_error(&RaaError::Gpu("device lost".into()));
    assert_eq!(translated.code(), FOLIAGE_BAKE_FAILED);
    assert_eq!(translated.message(), "device lost");
}

#[test]
fn state_reports_the_prepared_phase_and_mode() {
    let state = FoliageState {
        phase: FoliagePhase::Ready,
        generation: 3,
        asset: Some(ASSET),
        mode: 1,
        level: 1,
        error: None,
    };
    assert_eq!(phase_name(state.phase), "ready");
    assert_eq!(mode_name(state.mode), "pom");
    assert_eq!(state.asset, Some(ASSET));
    assert_eq!(state.level, 1);
}
