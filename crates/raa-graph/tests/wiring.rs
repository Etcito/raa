//! Family `wiring` (contract rg-wiring-family): the generic colour chain, the TAA history input and
//! its history outputs, the multi-instance flags of the post nodes, and the planning of a
//! reordered post chain, of two instances and of a history link. Public items only.

use raa_core::RaaError;
use raa_graph::Catalog;
use raa_graph::catalog::{GraphFormat, NodeTypeSpec, PortSpec, PortType};
use raa_graph::error::Issue;
use raa_graph::exec::pipeline_instructions;
use raa_graph::format::{GraphDoc, LinkDoc, NodeDoc, PortRef, parse_graph, write_graph};
use raa_graph::plan::{GraphPlan, plan_from_json};
use raa_graph::registry;
use raa_pipeline::{FeaturePass, PipelineConfig, RecordCtx};

/// The pass nodes whose `color` input and output are generic (the chain of the colour graph).
const CHAIN: [&str; 12] = [
    "pass.gi.history",
    "pass.ssr",
    "pass.fog",
    "pass.sunshafts",
    "pass.opaque.copy",
    "pass.transparent.oit",
    "pass.particles.blend",
    "pass.taa",
    "pass.dof",
    "pass.bloom",
    "pass.grading",
    "pass.cel.resolve",
];

/// The node types that accept several instances.
const MULTI: [&str; 7] = [
    "pass.ssr",
    "pass.dof",
    "pass.sunshafts",
    "pass.bloom",
    "pass.grading",
    "pass.cel.resolve",
    "pass.taa",
];

/// The node types that stay single-instance.
const SINGLE: [&str; 8] = [
    "pass.prepass",
    "pass.forward",
    "pass.tonemap",
    "pass.fog",
    "pass.opaque.copy",
    "pass.transparent.oit",
    "pass.particles.blend",
    "pass.gi.history",
];

fn node_of<'a>(catalog: &'a Catalog, id: &str) -> &'a NodeTypeSpec {
    catalog
        .node(id)
        .unwrap_or_else(|| panic!("no node type `{id}`"))
}

fn input<'a>(catalog: &'a Catalog, id: &str, key: &str) -> &'a PortSpec {
    node_of(catalog, id)
        .inputs
        .iter()
        .find(|port| port.key == key)
        .unwrap_or_else(|| panic!("`{id}` has no input `{key}`"))
}

fn output<'a>(catalog: &'a Catalog, id: &str, key: &str) -> &'a PortSpec {
    node_of(catalog, id)
        .outputs
        .iter()
        .find(|port| port.key == key)
        .unwrap_or_else(|| panic!("`{id}` has no output `{key}`"))
}

#[test]
fn colour_chain_ports_are_generic_and_the_ends_of_the_chain_too() {
    let catalog = Catalog::build();
    for id in CHAIN {
        assert!(input(&catalog, id, "color").generic, "{id} colour input");
        assert!(output(&catalog, id, "color").generic, "{id} colour output");
    }
    assert!(output(&catalog, "pass.forward", "color").generic);
    assert!(input(&catalog, "pass.tonemap", "color").generic);
    assert!(!input(&catalog, "pass.overlay", "color").generic);
    assert!(!output(&catalog, "pass.overlay", "color").generic);
}

#[test]
fn taa_has_a_history_input_and_the_history_producers_are_marked() {
    let catalog = Catalog::build();
    let history = input(&catalog, "pass.taa", "history");
    assert!(history.history && history.generic && history.absent_ok && !history.required);
    assert_eq!(history.ty, PortType::Image2d);
    assert_eq!(history.format, Some(GraphFormat::Rgba16Float));
    assert!(output(&catalog, "pass.taa", "history").history);
    assert!(output(&catalog, "pass.gi.history", "history").history);
}

#[test]
fn post_nodes_are_multi_instance_and_the_others_are_not() {
    let catalog = Catalog::build();
    for id in MULTI {
        assert!(
            node_of(&catalog, id).multi_instance,
            "{id} is multi-instance"
        );
    }
    for id in SINGLE {
        assert!(
            !node_of(&catalog, id).multi_instance,
            "{id} is single-instance"
        );
    }
}

fn node(id: &str, type_id: &str) -> NodeDoc {
    NodeDoc {
        id: id.to_owned(),
        type_id: type_id.to_owned(),
        version: None,
        params: Vec::new(),
        compute: None,
    }
}

fn port(text: &str) -> PortRef {
    let (node, port) = text.split_once('.').expect("a port reference has a dot");
    PortRef {
        node: node.to_owned(),
        port: port.to_owned(),
    }
}

fn link(from: &str, to: &str) -> LinkDoc {
    LinkDoc {
        from: port(from),
        to: port(to),
        history: false,
    }
}

/// The Full GI starter as a document.
fn full_gi() -> GraphDoc {
    let (_, _, json) = registry::STARTERS
        .iter()
        .find(|(name, _, _)| *name == "Full GI")
        .expect("the Full GI starter exists");
    parse_graph(json).expect("the Full GI starter parses")
}

/// Full GI with `dof` between bloom and tonemap.
fn reordered() -> GraphDoc {
    let mut doc = full_gi();
    doc.nodes.push(node("dof", "pass.dof"));
    doc.links
        .retain(|l| l.from != port("bloom.color") || l.to != port("tonemap.color"));
    doc.links.push(link("bloom.color", "dof.color"));
    doc.links.push(link("dof.color", "tonemap.color"));
    doc
}

fn plan_of(doc: &GraphDoc) -> Result<GraphPlan, Vec<Issue>> {
    plan_from_json(&write_graph(doc))
}

fn order_of(plan: &GraphPlan, id: &str) -> u32 {
    plan.nodes
        .iter()
        .find(|node| node.id == id)
        .unwrap_or_else(|| panic!("the plan has no node `{id}`"))
        .order_index
}

#[test]
fn reordered_post_chain_plans_with_bloom_before_dof() {
    let plan = plan_of(&reordered()).unwrap_or_else(|issues| panic!("{issues:?}"));
    assert!(order_of(&plan, "bloom") < order_of(&plan, "dof"));
    assert!(order_of(&plan, "dof") < order_of(&plan, "tonemap"));
}

#[test]
fn two_bloom_nodes_plan_and_two_fog_nodes_are_refused() {
    let mut doc = reordered();
    doc.links
        .retain(|l| l.from != port("bloom.color") || l.to != port("dof.color"));
    doc.nodes.push(node("bloomB", "pass.bloom"));
    doc.links.push(link("bloom.color", "bloomB.color"));
    doc.links.push(link("bloomB.color", "dof.color"));
    let plan = plan_of(&doc).unwrap_or_else(|issues| panic!("{issues:?}"));
    assert!(order_of(&plan, "bloom") < order_of(&plan, "bloomB"));
    assert!(order_of(&plan, "bloomB") < order_of(&plan, "dof"));

    let mut doc = full_gi();
    doc.nodes.push(node("fogB", "pass.fog"));
    let issues = plan_of(&doc).expect_err("a second fog node is refused");
    assert!(
        issues
            .iter()
            .any(|issue| issue.code == "graph.single_instance"
                && issue.node.as_deref() == Some("fogB")),
        "{issues:?}"
    );
}

#[test]
fn a_taa_history_link_plans_one_history_into_the_taa_input() {
    let mut doc = reordered();
    doc.links.push(LinkDoc {
        from: port("taa.history"),
        to: port("taa.history"),
        history: true,
    });
    let plan = plan_of(&doc).unwrap_or_else(|issues| panic!("{issues:?}"));
    assert_eq!(plan.histories.len(), 1);
    assert_eq!(plan.histories[0].consumer.node, "taa");
    assert_eq!(plan.histories[0].consumer.port, "history");
}

/// A pass entry of the test pipeline: its name, catalog default order and required flag.
struct Mock {
    name: &'static str,
    order: u32,
    required: bool,
}

impl FeaturePass for Mock {
    fn name(&self) -> &str {
        self.name
    }

    fn order(&self) -> u32 {
        self.order
    }

    fn required(&self) -> bool {
        self.required
    }

    fn record(&mut self, _ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        Ok(())
    }
}

#[test]
fn the_order_from_links_is_accepted_by_the_pipeline_config() {
    let catalog = Catalog::build();
    let plan = plan_of(&reordered()).unwrap_or_else(|issues| panic!("{issues:?}"));
    let mut config = PipelineConfig::new();
    for planned in &plan.nodes {
        let spec = node_of(&catalog, &planned.type_id);
        let Some(name) = spec.pass.filter(|_| !spec.contributor) else {
            continue;
        };
        let mock = Mock {
            name,
            order: spec.default_order,
            required: spec.required_in_graph,
        };
        config
            .add(Box::new(mock))
            .unwrap_or_else(|error| panic!("{error:?}"));
    }
    let sunshafts = node_of(&catalog, "pass.sunshafts");
    let mock = Mock {
        name: "sunshafts",
        order: sunshafts.default_order,
        required: false,
    };
    config
        .add(Box::new(mock))
        .unwrap_or_else(|error| panic!("{error:?}"));

    let instructions = pipeline_instructions(&plan, &catalog, &config.passes());
    let order: Vec<&str> = instructions.order.iter().map(String::as_str).collect();
    config
        .set_order_links(&order)
        .unwrap_or_else(|error| panic!("{error:?}"));

    let names: Vec<String> = config.passes().into_iter().map(|info| info.name).collect();
    let at = |name: &str| {
        names
            .iter()
            .position(|entry| entry == name)
            .unwrap_or_else(|| panic!("no pass `{name}` in {names:?}"))
    };
    assert!(at("bloom") < at("dof"));
    assert!(at("dof") < at("tonemap"));
    assert_eq!(at("tonemap"), names.len() - 1);
}
