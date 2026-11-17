//! The starter graphs (`crates/raa-graph/graphs/*.json`, registered by `build.rs` as
//! `registry::STARTERS`): every starter is canonical text that parses, validates and plans with no
//! warning; Full GI's plan order of the passes it shares with the pinned GI order is fixed; and Full
//! GI is the no-graph export of an engine configured that way.

use raa_graph::catalog::{Catalog, FeatureId, ParamValue};
use raa_graph::exec::{SettingsSnapshot, default_values, export_snapshot};
use raa_graph::format::{parse_graph, write_graph};
use raa_graph::plan::plan_from_doc;
use raa_graph::registry;
use raa_graph::validate::{GraphLimits, ValidateCtx, canonical_doc, validate};

/// `GI_DEFAULT_ORDER` of `crates/raa-pipeline/tests/config.rs:665-678`, copied because a test cannot
/// import another crate's test.
const GI_DEFAULT_ORDER: [&str; 12] = [
    "shadow.csm",
    "vg.resolve",
    "gi.volume",
    "prepass",
    "vg.gbuffer",
    "ao.gtao",
    "gi.screen",
    "forward",
    "gi.history",
    "ssr",
    "bloom",
    "tonemap",
];

/// The pass entries the Full GI engine enables, in S12 default order. The contributor
/// `sky.background` is listed apart by [`full_gi_snapshot`].
const FULL_GI_ORDER: [&str; 20] = [
    "sky.atmosphere",
    "sky.cloud-lighting",
    "sky.env",
    "shadow.csm",
    "gi.volume",
    "prepass",
    "light.tiles",
    "ao.gtao",
    "gi.screen",
    "gi.area-shadow",
    "gi.ssgi",
    "shadow.contact",
    "sky.cloud-screen",
    "forward",
    "gi.history",
    "ssr",
    "fog",
    "taa",
    "bloom",
    "tonemap",
];

/// Runs `run` with the context of `plan_from_json` (S53): every pass installed, one viewport, msaa 4
/// and the device-free limits `GraphLimits::CORE`.
fn with_ctx<T>(run: impl FnOnce(&ValidateCtx<'_>) -> T) -> T {
    let catalog = Catalog::build();
    let installed = |_: &str| true;
    let engine_options = [("msaa", ParamValue::Int(4))];
    let ctx = ValidateCtx {
        catalog: &catalog,
        limits: GraphLimits::CORE,
        installed: &installed,
        viewports: 1,
        engine_options: &engine_options,
    };
    run(&ctx)
}

/// The text of the starter whose file stem is `stem`.
fn starter(stem: &str) -> &'static str {
    registry::STARTERS
        .iter()
        .find(|&&(_, file, _)| file == stem)
        .map(|&(_, _, json)| json)
        .unwrap_or_else(|| panic!("no starter named {stem}"))
}

/// Sets `key` of one feature's snapshot values. The key must already hold a default.
fn set(values: &mut [(String, ParamValue)], key: &str, value: ParamValue) {
    let entry = values
        .iter_mut()
        .find(|(name, _)| name == key)
        .unwrap_or_else(|| panic!("no snapshot key {key}"));
    entry.1 = value;
}

/// The engine configured as Full GI: the default values of every feature the enabled passes read,
/// with the sdfm Quality preset written over the GI features (S61; sdfm@9bd0661
/// `src/core/global-illumination-settings.ts:62`, the high volume preset, and `:164-165`, the
/// Quality panel preset), and the enabled entries.
fn full_gi_snapshot() -> SettingsSnapshot {
    let features = [
        FeatureId::Tonemap,
        FeatureId::Shadow,
        FeatureId::GiQuality,
        FeatureId::Gi,
        FeatureId::AreaShadows,
        FeatureId::AreaShading,
        FeatureId::Gtao,
        FeatureId::Contact,
        FeatureId::Ssr,
        FeatureId::Fog,
        FeatureId::Bloom,
    ];
    let mut values = Vec::new();
    for feature in features {
        let mut entry = default_values(&feature);
        match feature {
            FeatureId::GiQuality => {
                set(&mut entry, "resolution", ParamValue::Int(96));
                set(&mut entry, "levels", ParamValue::Int(5));
                set(&mut entry, "probeRays", ParamValue::Int(8));
            }
            FeatureId::Gi => {
                // The presence keys: gi.volume enables GI, gi.ssgi enables screen space.
                set(&mut entry, "enabled", ParamValue::Bool(true));
                set(&mut entry, "screenSpace", ParamValue::Bool(true));
                set(&mut entry, "historyMax", ParamValue::Int(128));
                set(&mut entry, "clampGamma", ParamValue::Number(3.0));
                set(&mut entry, "spatialPasses", ParamValue::Int(4));
                set(&mut entry, "restirMCap", ParamValue::Number(30.0));
                set(&mut entry, "voxelFullRelight", ParamValue::Bool(true));
                set(&mut entry, "historyMin", ParamValue::Int(1));
                set(&mut entry, "voxelHistoryMax", ParamValue::Int(64));
                set(&mut entry, "voxelRays", ParamValue::Int(3));
                set(&mut entry, "debugView", ParamValue::Enum("none".to_owned()));
            }
            _ => {}
        }
        values.push((feature, entry));
    }
    let order: Vec<String> = FULL_GI_ORDER.iter().map(|name| name.to_string()).collect();
    let mut passes: Vec<(String, bool)> = order.iter().map(|name| (name.clone(), true)).collect();
    passes.push(("sky.background".to_owned(), true));
    SettingsSnapshot {
        passes,
        order,
        values,
    }
}

#[test]
fn every_starter_parses_validates_and_plans_without_a_warning() {
    for &(name, stem, json) in registry::STARTERS {
        let doc = parse_graph(json).unwrap_or_else(|issues| panic!("{stem} parses: {issues:?}"));
        with_ctx(|ctx| {
            let issues = validate(&doc, ctx);
            assert!(issues.is_empty(), "{stem} validates: {issues:?}");
            let (plan, warnings) = plan_from_doc(&doc, ctx)
                .unwrap_or_else(|issues| panic!("{stem} plans: {issues:?}"));
            assert!(
                warnings.is_empty(),
                "{stem} plans with a warning: {warnings:?}"
            );
            assert_eq!(plan.name, name);
        });
    }
}

#[test]
fn every_starter_is_the_canonical_text_of_its_document() {
    for &(_, stem, json) in registry::STARTERS {
        let doc = parse_graph(json).unwrap_or_else(|issues| panic!("{stem} parses: {issues:?}"));
        let canonical = with_ctx(|ctx| write_graph(&canonical_doc(&doc, ctx.catalog)));
        assert_eq!(canonical, json, "{stem} is canonical text");
    }
}

#[test]
fn full_gi_plans_its_shared_passes_in_the_pinned_gi_order() {
    let doc = parse_graph(starter("full-gi")).expect("the Full GI starter parses");
    let (plan, _) = with_ctx(|ctx| plan_from_doc(&doc, ctx)).expect("Full GI plans");
    let mut nodes: Vec<_> = plan.nodes.iter().collect();
    nodes.sort_by_key(|node| node.order_index);
    let shared: Vec<&str> = nodes
        .iter()
        .filter_map(|node| node.type_id.strip_prefix("pass."))
        .filter(|entry| GI_DEFAULT_ORDER.contains(entry))
        .collect();
    assert_eq!(
        shared,
        [
            "shadow.csm",
            "gi.volume",
            "prepass",
            "ao.gtao",
            "gi.screen",
            "forward",
            "gi.history",
            "ssr",
            "bloom",
            "tonemap",
        ]
    );
}

/// Handoff, card 'raa-graph exec 5/5: frame ownership, stats and export': `export_snapshot`
/// (crates/raa-graph/src/exec/export.rs:28) takes the Full GI snapshot. The export is named
/// "Engine default" (contract rg-export-snapshot), the starter "Full GI", so the export is written
/// under the starter's name.
#[test]
fn full_gi_is_the_no_graph_export_of_the_full_gi_engine() {
    let mut doc = export_snapshot(&Catalog::build(), &full_gi_snapshot(), 4);
    doc.name = "Full GI".to_owned();
    assert_eq!(write_graph(&doc), starter("full-gi"));
}

#[test]
fn starters_are_named_from_their_files_in_file_stem_order() {
    let names: Vec<&str> = registry::STARTERS
        .iter()
        .map(|&(name, _, _)| name)
        .collect();
    let stems: Vec<&str> = registry::STARTERS
        .iter()
        .map(|&(_, stem, _)| stem)
        .collect();
    assert_eq!(names, ["Full GI", "Simple forward"]);
    assert_eq!(stems, ["full-gi", "simple-forward"]);
}
