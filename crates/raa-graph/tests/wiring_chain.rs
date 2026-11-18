//! Device-free colour chain and executor ordering checks.
#[allow(dead_code)]
#[path = "../src/exec_ext/wiring/plan_fns.rs"]
mod plan_fns;
use raa_graph::exec::{SettingsSnapshot, adopt_pipeline, pipeline_differs, pipeline_instructions};
use raa_graph::format::{GraphDoc, LinkDoc, NodeDoc, PortRef, parse_graph, write_graph};
use raa_graph::plan::{
    GraphPlan, PlanHistory, PlanNode, PlanPortRef, PlanResource, plan_from_json,
};
use raa_graph::{Catalog, registry};
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
fn reordered() -> GraphDoc {
    let mut doc = full();
    doc.nodes.push(node("dof", "pass.dof"));
    doc.links
        .retain(|l| l.from != port("bloom.color") || l.to != port("tonemap.color"));
    doc.links.extend([
        link("bloom.color", "dof.color"),
        link("dof.color", "tonemap.color"),
    ]);
    doc
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
fn names(values: &[&str]) -> Vec<String> {
    values.iter().map(|s| (*s).into()).collect()
}

#[test]
fn chain_outputs_and_pool() {
    let p = plan(&full());
    let c = Catalog::build();
    let resources = plan_fns::chain_resources(&p, &c);
    assert_eq!(
        resources
            .iter()
            .map(|r| r.producer.node.as_str())
            .collect::<Vec<_>>(),
        ["forward", "giHistory", "ssr", "fog", "taa", "bloom"]
    );
    assert_eq!(
        plan_fns::pool_resources(&resources).into_iter().max(),
        Some(1)
    );
    assert_eq!(plan_fns::bound_view_count(&p, &c), 2);
}
#[test]
fn entries_and_link_order() {
    let mut doc = reordered();
    let c = Catalog::build();
    let p = plan(&doc);
    let entries = plan_fns::node_entries(&p, &c);
    for name in ["bloom", "dof", "tonemap"] {
        assert!(entries.contains(&(name.into(), name.into())));
    }
    assert!(!entries.iter().any(|(id, _)| id == "out"));
    let installed = info(&[
        "prepass", "forward", "ssr", "taa", "dof", "bloom", "tonemap", "view#ov",
    ]);
    assert_eq!(
        plan_fns::link_order(&p, &entries, &installed),
        names(&[
            "prepass", "forward", "ssr", "taa", "bloom", "dof", "tonemap", "view#ov"
        ])
    );
    doc.links
        .retain(|l| l.from != port("bloom.color") || l.to != port("dof.color"));
    doc.nodes.push(node("bloomB", "pass.bloom"));
    doc.links.extend([
        link("bloom.color", "bloomB.color"),
        link("bloomB.color", "dof.color"),
    ]);
    let p = plan(&doc);
    let entries = plan_fns::node_entries(&p, &c);
    assert!(entries.contains(&("bloomB".into(), "bloom#bloomB".into())));
    let installed = info(&["prepass", "forward", "bloom", "bloom#bloomB", "tonemap"]);
    let instructions = pipeline_instructions(&p, &c, &installed);
    assert_eq!(
        instructions.order.iter().filter(|n| *n == "bloom").count(),
        1
    );
    assert_eq!(
        instructions
            .enable
            .iter()
            .filter(|(n, _)| n == "bloom")
            .count(),
        1
    );
    assert!(!instructions.order.iter().any(|n| n.contains('#')));
    assert!(!instructions.enable.iter().any(|(n, _)| n.contains('#')));
}
#[test]
fn external_pipeline_changes_ignore_instances() {
    let base = ["prepass", "forward", "bloom", "dof", "tonemap"];
    let applied = SettingsSnapshot {
        passes: base.iter().map(|n| ((*n).into(), true)).collect(),
        order: names(&base),
        values: vec![],
    };
    let mut observed = info(&[
        "prepass",
        "forward",
        "bloom",
        "bloom#bloomB",
        "dof",
        "tonemap",
        "view#cmp",
    ]);
    assert!(!pipeline_differs(&applied, &observed));
    observed.swap(2, 4);
    assert!(pipeline_differs(&applied, &observed));
    observed[3].enabled = false;
    let mut no_graph = applied.clone();
    adopt_pipeline(&mut no_graph, &applied, &observed);
    assert_eq!(
        no_graph.order,
        names(&["prepass", "forward", "dof", "bloom", "tonemap"])
    );
    assert!(no_graph.passes.iter().all(|(n, _)| !n.contains('#')));
    assert!(no_graph.order.iter().all(|n| !n.contains('#')));
    let mut disabled = info(&base);
    disabled[3].enabled = false;
    assert!(pipeline_differs(&applied, &disabled));
}
#[test]
fn non_chain_tonemap_source_adds_one_view() {
    let mut doc = full();
    doc.nodes.retain(|n| n.id != "bloom");
    doc.links
        .retain(|l| l.from.node != "bloom" && l.to.node != "bloom");
    doc.nodes
        .extend([node("taaB", "pass.taa"), node("cmp", "view.compare")]);
    doc.links.extend([
        link("fog.color", "taaB.color"),
        link("taa.color", "cmp.a"),
        link("taaB.color", "cmp.b"),
        link("cmp.color", "tonemap.color"),
    ]);
    let p = plan(&doc);
    let c = Catalog::build();
    let slots = plan_fns::pool_resources(&plan_fns::chain_resources(&p, &c));
    assert_eq!(
        plan_fns::bound_view_count(&p, &c),
        *slots.iter().max().unwrap() as usize + 2
    );
}
