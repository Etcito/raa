//! The `emissive` parameter of the GI pass nodes (contract 'rg-gi-emissive-param'): its catalog
//! entries, the Compute storage buffers of the screen and SSGI nodes, the Full GI starter's plan,
//! its saved text and the catalog JSON. Public items only.

use raa_graph::catalog::{
    Catalog, NodeTypeSpec, ParamBinding, ParamKind, ParamSpec, ParamValue, Rebuild, ShaderStage,
};
use raa_graph::format::{parse_graph, write_graph};
use raa_graph::plan::{GraphPlan, plan_from_doc};
use raa_graph::registry;
use raa_graph::validate::{GraphLimits, ValidateCtx};
use raa_loader::json::JsonValue;

/// Runs `run` with the context of `plan_from_json` (S53): every pass installed, one viewport, msaa 4
/// and `GraphLimits::CORE`.
fn with_ctx<T>(catalog: &Catalog, run: impl FnOnce(&ValidateCtx<'_>) -> T) -> T {
    let installed = |_: &str| true;
    let engine_options = [("msaa", ParamValue::Int(4))];
    let ctx = ValidateCtx {
        catalog,
        limits: GraphLimits::CORE,
        installed: &installed,
        viewports: 1,
        engine_options: &engine_options,
    };
    run(&ctx)
}

/// The text of the Full GI starter.
fn full_gi() -> &'static str {
    registry::STARTERS
        .iter()
        .find(|&&(_, stem, _)| stem == "full-gi")
        .map(|&(_, _, json)| json)
        .expect("the Full GI starter")
}

/// The value `key` of node `id` in `plan`, as the plan carries it.
fn planned(plan: &GraphPlan, id: &str, key: &str) -> Option<ParamValue> {
    let node = plan.nodes.iter().find(|node| node.id == id)?;
    node.params
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.clone())
}

/// The `emissive` parameters of node type `id`.
fn emissive_params<'a>(catalog: &'a Catalog, id: &str) -> Vec<&'a ParamSpec> {
    let node = catalog.node(id).expect("node type");
    node.params.iter().filter(|p| p.key == "emissive").collect()
}

#[test]
fn the_three_gi_pass_nodes_hold_one_emissive_parameter_each() {
    let catalog = Catalog::build();
    for id in ["pass.gi.volume", "pass.gi.screen", "pass.gi.ssgi"] {
        let params = emissive_params(&catalog, id);
        assert_eq!(params.len(), 1, "{id}");
        let p = params[0];
        assert_eq!(p.kind, ParamKind::Bool, "{id}");
        assert_eq!(p.default, ParamValue::Bool(true), "{id}");
        assert_eq!(p.group, "Emissive", "{id}");
        assert_eq!(p.label, "Emissive", "{id}");
        assert_eq!(p.rebuild, Rebuild::Pipeline, "{id}");
        assert_eq!(p.binding, ParamBinding::Node, "{id}");
    }
}

#[test]
fn the_area_shadow_and_history_nodes_hold_no_emissive_parameter() {
    let catalog = Catalog::build();
    for id in ["pass.gi.area-shadow", "pass.gi.history"] {
        assert!(emissive_params(&catalog, id).is_empty(), "{id}");
    }
}

#[test]
fn the_screen_and_ssgi_compute_budgets_count_their_voxel_buffer() {
    let catalog = Catalog::build();
    let storage_buffers = |id: &str| -> u32 {
        let node: &NodeTypeSpec = catalog.node(id).expect("node type");
        node.budgets
            .iter()
            .find(|budget| matches!(budget.stage, ShaderStage::Compute))
            .expect("a Compute budget")
            .storage_buffers
    };
    assert_eq!(storage_buffers("pass.gi.screen"), 3);
    assert_eq!(storage_buffers("pass.gi.ssgi"), 1);
}

#[test]
fn full_gi_plans_emissive_on_for_its_three_gi_nodes() {
    let catalog = Catalog::build();
    let doc = parse_graph(full_gi()).expect("the Full GI starter parses");
    let (plan, _) = with_ctx(&catalog, |ctx| plan_from_doc(&doc, ctx)).expect("Full GI plans");
    for id in ["giVolume", "giScreen", "giSsgi"] {
        assert_eq!(
            planned(&plan, id, "emissive"),
            Some(ParamValue::Bool(true)),
            "{id}"
        );
    }
}

#[test]
fn a_saved_false_survives_the_text_and_reaches_the_plan() {
    let catalog = Catalog::build();
    let mut doc = parse_graph(full_gi()).expect("the Full GI starter parses");
    let node = doc
        .nodes
        .iter_mut()
        .find(|node| node.id == "giScreen")
        .expect("giScreen");
    let (_, value) = node
        .params
        .iter_mut()
        .find(|(key, _)| key == "emissive")
        .expect("giScreen holds emissive");
    *value = JsonValue::Bool(false);

    let reread = parse_graph(&write_graph(&doc)).expect("the saved Full GI parses");
    let node = reread
        .nodes
        .iter()
        .find(|node| node.id == "giScreen")
        .expect("giScreen");
    assert!(
        node.params
            .contains(&("emissive".to_string(), JsonValue::Bool(false)))
    );

    let (plan, _) = with_ctx(&catalog, |ctx| plan_from_doc(&reread, ctx)).expect("Full GI plans");
    assert_eq!(
        planned(&plan, "giScreen", "emissive"),
        Some(ParamValue::Bool(false))
    );
    assert_eq!(
        planned(&plan, "giVolume", "emissive"),
        Some(ParamValue::Bool(true))
    );
}

/// The entries of a JSON object, `None` for any other value.
fn object(value: &JsonValue) -> Option<&[(String, JsonValue)]> {
    match value {
        JsonValue::Object(entries) => Some(entries),
        _ => None,
    }
}

/// The member `key` of a JSON object's entries.
fn member<'a>(entries: &'a [(String, JsonValue)], key: &str) -> Option<&'a JsonValue> {
    entries
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
}

#[test]
fn the_catalog_json_lists_emissive_on_pass_gi_volume() {
    let JsonValue::Object(top) = Catalog::build().to_json(&|_: &NodeTypeSpec| true) else {
        panic!("the catalog JSON is not an object")
    };
    let Some(JsonValue::Array(nodes)) = member(&top, "nodes") else {
        panic!("the catalog JSON has no nodes array")
    };
    let volume = nodes
        .iter()
        .filter_map(object)
        .find(|entries| member(entries, "id") == Some(&JsonValue::String("pass.gi.volume".into())))
        .expect("pass.gi.volume is listed");
    let Some(JsonValue::Array(params)) = member(volume, "params") else {
        panic!("pass.gi.volume has no params array")
    };
    let emissive = params
        .iter()
        .filter_map(object)
        .find(|entries| member(entries, "key") == Some(&JsonValue::String("emissive".into())))
        .expect("emissive is listed on pass.gi.volume");
    assert_eq!(
        member(emissive, "kind"),
        Some(&JsonValue::String("bool".into()))
    );
    assert_eq!(member(emissive, "default"), Some(&JsonValue::Bool(true)));
}
