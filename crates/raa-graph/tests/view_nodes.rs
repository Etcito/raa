//! The view nodes (contract 'view-family', `crates/raa-graph/src/nodes/view.rs`): their catalog
//! types, the required passes they mark as side-effect nodes, and a Simple forward graph whose
//! display reads a view.

use raa_graph::catalog::{
    Catalog, Category, GraphFormat, NodeTypeSpec, ParamKind, ParamValue, PortType,
};
use raa_graph::format::{LinkDoc, NodeDoc, PortRef, parse_graph, write_graph};
use raa_graph::plan::{PlanPortRef, plan_from_json};
use raa_graph::registry;

const NONE: &[&str] = &[];
const CHANNELS: &[&str] = &["rgb", "r", "g", "b", "a", "rgba-alpha"];
const DECODE: &[&str] = &["none", "half2lo", "half2hi"];
const MODES: &[&str] = &["split", "sideBySide", "difference"];

fn node<'a>(catalog: &'a Catalog, id: &str) -> &'a NodeTypeSpec {
    catalog
        .node(id)
        .unwrap_or_else(|| panic!("no node type {id}"))
}

/// The `(key, type)` of every input of `spec`, in port order.
fn inputs(spec: &NodeTypeSpec) -> Vec<(&'static str, PortType)> {
    spec.inputs.iter().map(|p| (p.key, p.ty.clone())).collect()
}

fn keys(spec: &NodeTypeSpec) -> Vec<&'static str> {
    spec.params.iter().map(|p| p.key).collect()
}

fn kinds(spec: &NodeTypeSpec) -> Vec<ParamKind> {
    spec.params.iter().map(|p| p.kind.clone()).collect()
}

fn defaults(spec: &NodeTypeSpec) -> Vec<ParamValue> {
    spec.params.iter().map(|p| p.default.clone()).collect()
}

fn options(spec: &NodeTypeSpec) -> Vec<&'static [&'static str]> {
    spec.params.iter().map(|p| p.options).collect()
}

#[test]
fn view_node_types_follow_the_contract() {
    let catalog = Catalog::build();
    for id in ["view.image", "view.compare", "view.grid"] {
        let spec = node(&catalog, id);
        assert_eq!(spec.category, Category::View, "{id}");
        assert_eq!(spec.pass, None, "{id}");
        assert!(spec.multi_instance, "{id}");
        assert!(spec.inputs.iter().all(|p| p.generic), "{id}");
        let [color] = spec.outputs.as_slice() else {
            panic!("{id} has one output");
        };
        assert_eq!(color.key, "color", "{id}");
        assert_eq!(color.ty, PortType::Image2d, "{id}");
        assert_eq!(color.format, Some(GraphFormat::Rgba16Float), "{id}");
    }

    let image = node(&catalog, "view.image");
    assert_eq!(
        inputs(image),
        [
            ("src", PortType::Image2d),
            ("srcDepth", PortType::Depth2d),
            ("srcArray", PortType::Image2dArray),
            ("srcDepthArray", PortType::Depth2dArray),
            ("src3d", PortType::Image3d),
            ("srcCube", PortType::ImageCube),
        ]
    );
    assert!(image.inputs.iter().all(|p| !p.required && p.absent_ok));
    assert_eq!(
        keys(image),
        [
            "channels",
            "rangeMin",
            "rangeMax",
            "exposureEv",
            "linearizeDepth",
            "layer",
            "slice",
            "mip",
            "decode",
            "label",
        ]
    );
    assert_eq!(
        kinds(image),
        [
            ParamKind::Enum,
            ParamKind::Number,
            ParamKind::Number,
            ParamKind::Number,
            ParamKind::Bool,
            ParamKind::Int,
            ParamKind::Int,
            ParamKind::Int,
            ParamKind::Enum,
            ParamKind::String,
        ]
    );
    assert_eq!(
        defaults(image),
        [
            ParamValue::Enum("rgb".into()),
            ParamValue::Number(0.0),
            ParamValue::Number(1.0),
            ParamValue::Number(0.0),
            ParamValue::Bool(false),
            ParamValue::Int(0),
            ParamValue::Int(0),
            ParamValue::Int(0),
            ParamValue::Enum("none".into()),
            ParamValue::String(String::new()),
        ]
    );
    assert_eq!(
        options(image),
        [
            CHANNELS, NONE, NONE, NONE, NONE, NONE, NONE, NONE, DECODE, NONE
        ]
    );

    let compare = node(&catalog, "view.compare");
    assert_eq!(
        inputs(compare),
        [("a", PortType::Image2d), ("b", PortType::Image2d)]
    );
    assert!(compare.inputs.iter().all(|p| p.required && !p.absent_ok));
    assert_eq!(keys(compare), ["mode", "position"]);
    assert_eq!(
        defaults(compare),
        [ParamValue::Enum("split".into()), ParamValue::Number(0.5)]
    );
    assert_eq!(options(compare), [MODES, NONE]);

    let grid = node(&catalog, "view.grid");
    let tiles: Vec<String> = (0..16).map(|i| format!("t{i}")).collect();
    let grid_keys: Vec<String> = grid.inputs.iter().map(|p| p.key.to_string()).collect();
    assert_eq!(grid_keys, tiles);
    assert!(
        grid.inputs
            .iter()
            .all(|p| p.ty == PortType::Image2d && !p.required && p.absent_ok)
    );
    assert_eq!(keys(grid), ["columns", "labels"]);
    assert_eq!(defaults(grid), [ParamValue::Int(4), ParamValue::Bool(true)]);
    assert_eq!(kinds(grid), [ParamKind::Int, ParamKind::Bool]);

    for id in ["pass.prepass", "pass.forward", "pass.tonemap"] {
        assert!(node(&catalog, id).side_effect, "{id}");
    }
}

#[test]
fn a_display_fed_by_a_view_keeps_the_required_passes_planned() {
    let (_, _, json) = registry::STARTERS
        .iter()
        .find(|&&(_, stem, _)| stem == "simple-forward")
        .expect("the Simple forward starter");
    let mut doc = parse_graph(json).unwrap_or_else(|issues| panic!("{issues:?}"));
    doc.nodes.push(NodeDoc {
        id: "show".into(),
        type_id: "view.image".into(),
        version: None,
        params: Vec::new(),
        compute: None,
    });
    doc.links
        .retain(|link| !(link.to.node == "out" && link.to.port == "color"));
    doc.links.push(link("prepass", "depth", "show", "srcDepth"));
    doc.links.push(link("show", "color", "out", "color"));
    let text = write_graph(&doc);

    let plan = plan_from_json(&text).unwrap_or_else(|issues| panic!("{issues:?}"));
    let ids: Vec<&str> = plan.nodes.iter().map(|n| n.id.as_str()).collect();
    for id in ["prepass", "forward", "tonemap", "show"] {
        assert!(ids.contains(&id), "{id} is not planned: {ids:?}");
    }
    let show_color = PlanPortRef {
        node: "show".into(),
        port: "color".into(),
    };
    assert_eq!(plan.output, show_color);
}

/// A document link from `from_node.from_port` to `to_node.to_port`.
fn link(from_node: &str, from_port: &str, to_node: &str, to_port: &str) -> LinkDoc {
    LinkDoc {
        from: PortRef {
            node: from_node.into(),
            port: from_port.into(),
        },
        to: PortRef {
            node: to_node.into(),
            port: to_port.into(),
        },
        history: false,
    }
}
