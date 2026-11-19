//! S59 compute preset files: canonical registry entries, valid graph nodes and WGSL modules.

#[allow(dead_code)]
#[path = "../src/nodes/compute/codegen.rs"]
mod codegen;

use codegen::generate_module;
use raa_graph::catalog::{Catalog, GraphFormat, ParamValue, PortType};
use raa_graph::error::{Issue, Severity, Stage};
use raa_graph::format::{GraphDoc, LinkDoc, NodeDoc, PortRef, parse_graph, write_graph};
use raa_graph::registry;
use raa_graph::validate::{GraphLimits, ValidateCtx, validate};
use raa_loader::json::JsonValue;

const CONTRACT: [(&str, &str, &str); 6] = [
    ("atrous-step", "A-trous step", "atrousStep"),
    ("bilateral-blur", "Bilateral blur", "bilateralBlur"),
    ("blend", "Blend", "blend"),
    ("box-blur", "Box blur", "boxBlur"),
    ("gaussian-blur", "Gaussian blur", "gaussianBlur"),
    (
        "temporal-accumulate",
        "Temporal accumulate",
        "temporalAccumulate",
    ),
];

fn link(from_node: &str, from_port: &str, to_node: &str, to_port: &str) -> LinkDoc {
    LinkDoc {
        from: PortRef {
            node: from_node.to_string(),
            port: from_port.to_string(),
        },
        to: PortRef {
            node: to_node.to_string(),
            port: to_port.to_string(),
        },
        history: false,
    }
}

fn starter() -> GraphDoc {
    let text = registry::STARTERS
        .iter()
        .find(|&&(_, stem, _)| stem == "simple-forward")
        .map(|&(_, _, text)| text)
        .expect("Simple forward starter exists");
    parse_graph(text).expect("Simple forward parses")
}

#[test]
fn every_compute_preset_is_canonical_and_valid() {
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

    for stem in ["blend", "gaussian-blur", "box-blur"] {
        assert!(
            registry::PRESETS.iter().any(|entry| entry.1 == stem),
            "missing {stem}"
        );
    }

    for &(name, stem, text) in registry::PRESETS {
        let expected = CONTRACT
            .iter()
            .find(|&&(candidate, _, _)| candidate == stem)
            .expect("registered preset stem is in the contract");
        assert_eq!(expected.1, name, "registry name for {stem}");

        let preset = parse_graph(text).unwrap_or_else(|issues| panic!("{stem}: {issues:?}"));
        assert_eq!(write_graph(&preset), text, "{stem} is canonical");
        assert_eq!(preset.name, name);
        assert!(preset.links.is_empty(), "{stem} has no outer links");
        assert_eq!(preset.nodes.len(), 1, "{stem} has one node");
        let compute = preset.nodes[0].clone();
        assert_eq!(compute.id, expected.2);
        assert_eq!(compute.type_id, "compute.wgsl");

        let mut doc = starter();
        doc.nodes.push(compute.clone());
        for input in &compute
            .compute
            .as_ref()
            .expect("compute declaration")
            .inputs
        {
            let source = if input.ty == PortType::Depth2d {
                ("prepass", "depth")
            } else {
                ("prepass", "normal")
            };
            doc.links
                .push(link(source.0, source.1, &compute.id, &input.key));
        }
        doc.nodes.push(NodeDoc {
            id: "showPreset".to_string(),
            type_id: "view.image".to_string(),
            version: None,
            params: Vec::new(),
            compute: None,
        });
        doc.links
            .retain(|edge| !(edge.to.node == "out" && edge.to.port == "color"));
        doc.links
            .push(link(&compute.id, "dst", "showPreset", "src"));
        doc.links.push(link("showPreset", "color", "out", "color"));

        let issues = validate(&doc, &ctx);
        assert!(
            issues
                .iter()
                .all(|issue| issue.node.as_deref() != Some(&compute.id)),
            "{stem} validation issues: {issues:?}"
        );
        let (module, _) = generate_module(&compute)
            .unwrap_or_else(|issue| panic!("{stem} module generation: {issue:?}"));
        raa_shader::Composer::validate(&module)
            .unwrap_or_else(|error| panic!("{stem} generated module: {error}"));
    }
}
