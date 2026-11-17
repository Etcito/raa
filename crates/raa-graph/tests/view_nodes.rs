//! The view nodes (contract 'view-family', `crates/raa-graph/src/nodes/view.rs`): their catalog
//! types, the required passes they mark as side-effect nodes, and a Simple forward graph whose
//! display reads a view. The Debug starter's plan feeds every view node into its grid, and the
//! taps table `docs/render-graph/taps.md` names every catalog output port of a pass.

use std::collections::BTreeSet;

use raa_graph::catalog::{
    Catalog, Category, GraphFormat, NodeTypeSpec, ParamKind, ParamValue, PortType,
};
use raa_graph::format::{LinkDoc, NodeDoc, PortRef, parse_graph, write_graph};
use raa_graph::plan::{InputSource, PlanPortRef, plan_from_json};
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

#[test]
fn debug_starter_plans_every_view_node_into_the_grid() {
    let (_, _, json) = registry::STARTERS
        .iter()
        .find(|&&(_, stem, _)| stem == "debug")
        .expect("the Debug starter");
    let plan = plan_from_json(json).unwrap_or_else(|issues| panic!("{issues:?}"));

    let ids: Vec<&str> = plan.nodes.iter().map(|n| n.id.as_str()).collect();
    for id in [
        "vDepth",
        "vNormal",
        "vMotion",
        "vAo",
        "vAreaMask",
        "giRawVsDenoised",
        "vHistory",
        "vFinal",
        "grid",
    ] {
        assert!(ids.contains(&id), "{id} is not planned: {ids:?}");
    }
    let grid_color = PlanPortRef {
        node: "grid".into(),
        port: "color".into(),
    };
    assert_eq!(plan.output, grid_color);

    // t0..t7 read the colour of these eight view nodes in order; t8..t15 are unlinked.
    let fed = [
        "vDepth",
        "vNormal",
        "vMotion",
        "vAo",
        "vAreaMask",
        "giRawVsDenoised",
        "vHistory",
        "vFinal",
    ];
    let expected: Vec<(String, InputSource)> = (0..16)
        .map(|index| {
            let source = match fed.get(index) {
                Some(&node) => InputSource::Link {
                    node: node.into(),
                    port: "color".into(),
                    history: false,
                },
                None => InputSource::Absent,
            };
            (format!("t{index}"), source)
        })
        .collect();
    let grid = plan
        .nodes
        .iter()
        .find(|node| node.id == "grid")
        .expect("the grid is planned");
    let inputs: Vec<(String, InputSource)> = grid
        .inputs
        .iter()
        .map(|input| (input.port.clone(), input.source.clone()))
        .collect();
    assert_eq!(inputs, expected);
}

/// The taps table of `docs/render-graph/taps.md`, the text its rows are read from.
const TAPS: &str = include_str!("../../../docs/render-graph/taps.md");

/// The line that opens the taps table; its rows follow it.
const TAPS_HEADER: &str = "| pass | port | field | file |";

/// The `(pass, port)` of every row of the taps table, in file order. A row is a line after the
/// header that starts with `| `; the `|---|` separator starts with `|-` and is not a row.
fn tap_rows() -> Vec<(String, String)> {
    let mut rows = Vec::new();
    let mut in_table = false;
    for line in TAPS.lines() {
        if line == TAPS_HEADER {
            in_table = true;
        } else if in_table && line.starts_with("| ") {
            let cells: Vec<&str> = line.split('|').map(str::trim).collect();
            let [_, pass, port, ..] = cells.as_slice() else {
                panic!("a taps row names no pass and port: {line}");
            };
            rows.push((pass.to_string(), port.to_string()));
        }
    }
    assert!(
        in_table,
        "docs/render-graph/taps.md has no `{TAPS_HEADER}` header"
    );
    rows
}

/// A `(pass, port)` pair as the failure messages name it.
fn pair_label((pass, port): &(String, String)) -> String {
    format!("({pass}, {port})")
}

#[test]
fn every_catalog_output_port_has_a_taps_row() {
    let catalog = Catalog::build();
    let published: BTreeSet<(String, String)> = catalog
        .nodes
        .iter()
        .filter_map(|spec| spec.pass.map(|entry| (entry, spec)))
        .flat_map(|(entry, spec)| {
            spec.outputs
                .iter()
                .map(move |port| (entry.to_string(), port.key.to_string()))
        })
        .collect();
    let rows = tap_rows();
    let listed: BTreeSet<(String, String)> = rows.iter().cloned().collect();

    let missing: Vec<String> = published.difference(&listed).map(pair_label).collect();
    assert!(
        missing.is_empty(),
        "docs/render-graph/taps.md has no row for {}",
        missing.join(", ")
    );
    let stale: Vec<String> = listed.difference(&published).map(pair_label).collect();
    assert!(
        stale.is_empty(),
        "docs/render-graph/taps.md has rows naming no catalog output port: {}",
        stale.join(", ")
    );
    let mut seen = BTreeSet::new();
    let repeated: Vec<String> = rows
        .iter()
        .filter(|row| !seen.insert(*row))
        .map(pair_label)
        .collect();
    assert!(
        repeated.is_empty(),
        "docs/render-graph/taps.md has more than one row for {}",
        repeated.join(", ")
    );
}
