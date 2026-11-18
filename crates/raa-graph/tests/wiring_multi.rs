//! Device-free checks of the multi-instance post passes: two TAA nodes in one graph (their pipeline
//! entries and link order), and the first-node rule for the values of two bloom nodes. The
//! instance table these functions serve is `WiringExtension.instances`; its Handoff doc comment
//! (`exec_ext/wiring.rs` `WiringExtension`, line 83) names the next card.
#[allow(dead_code)]
#[path = "../src/exec_ext/wiring/plan_fns.rs"]
mod plan_fns;

use raa_graph::Catalog;
use raa_graph::catalog::{FeatureId, ParamValue};
use raa_graph::exec::{SettingsSnapshot, default_values, settings_targets};
use raa_graph::format::{GraphDoc, LinkDoc, NodeDoc, PortRef, parse_graph, write_graph};
use raa_graph::plan::{
    GraphPlan, PlanHistory, PlanNode, PlanPortRef, PlanResource, plan_from_json,
};
use raa_graph::registry;
use raa_loader::json::JsonValue;
use raa_pipeline::{PassInfo, PassRole};

fn full() -> GraphDoc {
    let (_, _, json) = registry::STARTERS
        .iter()
        .find(|(n, _, _)| *n == "Full GI")
        .unwrap();
    parse_graph(json).unwrap()
}
fn node(id: &str, ty: &str) -> NodeDoc {
    NodeDoc {
        id: id.into(),
        type_id: ty.into(),
        version: None,
        params: vec![],
        compute: None,
    }
}
fn port(text: &str) -> PortRef {
    let (node, port) = text.split_once('.').unwrap();
    PortRef {
        node: node.into(),
        port: port.into(),
    }
}
fn link(from: &str, to: &str) -> LinkDoc {
    LinkDoc {
        from: port(from),
        to: port(to),
        history: false,
    }
}
fn plan(doc: &GraphDoc) -> GraphPlan {
    plan_from_json(&write_graph(doc)).unwrap()
}
fn info(names: &[&str]) -> Vec<PassInfo> {
    names
        .iter()
        .enumerate()
        .map(|(i, name)| PassInfo {
            name: (*name).into(),
            role: PassRole::Pass,
            enabled: true,
            order: i as u32,
            required: false,
        })
        .collect()
}

/// Full GI without `bloom`: fog feeds two TAA nodes, and `cmp` compares them into the tonemap.
fn two_taa() -> GraphDoc {
    let mut doc = full();
    doc.nodes.retain(|n| n.id != "bloom");
    doc.links
        .retain(|l| l.from.node != "bloom" && l.to.node != "bloom");
    let mut taa_b = node("taaB", "pass.taa");
    taa_b.params = vec![
        ("blendBase".into(), JsonValue::Number(1.0)),
        ("blendMin".into(), JsonValue::Number(1.0)),
        ("blendMax".into(), JsonValue::Number(1.0)),
    ];
    let mut cmp = node("cmp", "view.compare");
    cmp.params = vec![
        ("mode".into(), JsonValue::String("split".into())),
        ("position".into(), JsonValue::Number(0.5)),
    ];
    doc.nodes.extend([taa_b, cmp]);
    doc.links.extend([
        link("fog.color", "taaB.color"),
        link("taa.color", "cmp.a"),
        link("taaB.color", "cmp.b"),
        link("cmp.color", "tonemap.color"),
    ]);
    doc
}

#[test]
fn two_taa_nodes_get_their_own_entries_and_stay_before_tonemap() {
    let catalog = Catalog::build();
    let plan = plan(&two_taa());
    let entries = plan_fns::node_entries(&plan, &catalog);
    let entry = |id: &str| {
        entries
            .iter()
            .find(|(node, _)| node == id)
            .map(|(_, entry)| entry.as_str())
    };
    assert_eq!(entry("taa"), Some("taa"));
    assert_eq!(entry("taaB"), Some("taa#taaB"));
    assert_eq!(entry("cmp"), Some("view#cmp"));

    let installed = info(&[
        "prepass", "forward", "fog", "taa", "taa#taaB", "view#cmp", "tonemap",
    ]);
    let order = plan_fns::link_order(&plan, &entries, &installed);
    let at = |name: &str| order.iter().position(|n| n == name);
    let (taa, taa_b, cmp, tonemap) = (at("taa"), at("taa#taaB"), at("view#cmp"), at("tonemap"));
    assert!(
        taa < taa_b && taa_b < cmp && cmp < tonemap,
        "order {order:?}"
    );
}

#[test]
fn installed_bloom_takes_the_first_node_value_of_two_bloom_nodes() {
    let catalog = Catalog::build();
    let mut doc = full();
    if let Some(bloom) = doc.nodes.iter_mut().find(|n| n.id == "bloom") {
        bloom.params = vec![("strength".into(), JsonValue::Number(0.2))];
    }
    let mut bloom_b = node("bloomB", "pass.bloom");
    bloom_b.params = vec![("strength".into(), JsonValue::Number(0.9))];
    doc.nodes.push(bloom_b);
    doc.links
        .retain(|l| l.from != port("bloom.color") || l.to != port("tonemap.color"));
    doc.links.extend([
        link("bloom.color", "bloomB.color"),
        link("bloomB.color", "tonemap.color"),
    ]);
    let plan = plan(&doc);
    let current = SettingsSnapshot {
        passes: vec![],
        order: vec![],
        values: vec![(FeatureId::Bloom, default_values(&FeatureId::Bloom))],
    };
    let targets = settings_targets(&plan, &catalog, &current).expect("the targets resolve");
    let (_, bloom) = targets
        .values
        .iter()
        .find(|(feature, _)| *feature == FeatureId::Bloom)
        .expect("bloom values");
    let strength = bloom
        .iter()
        .find(|(key, _)| key == "strength")
        .map(|(_, value)| value.clone());
    // The feature holds f32, so the first node's 0.2 reads back as the f32 value widened to f64.
    assert_eq!(strength, Some(ParamValue::Number(f64::from(0.2_f32))));
}

#[test]
fn instance_plan_installs_keeps_and_removes_by_id_and_type() {
    let pair = |id: &str, ty: &str| (id.to_owned(), ty.to_owned());
    let kept = [pair("taaB", "pass.taa"), pair("bloomB", "pass.bloom")];
    let wanted = [
        pair("taaB", "pass.taa"),
        pair("bloomB", "pass.dof"),
        pair("gradeB", "pass.grading"),
    ];
    let (install, keep, remove) = plan_fns::instance_plan(&kept, &wanted);
    assert_eq!(
        install,
        [pair("bloomB", "pass.dof"), pair("gradeB", "pass.grading")]
    );
    assert_eq!(keep, [pair("taaB", "pass.taa")]);
    assert_eq!(remove, ["bloomB".to_owned()]);
}
