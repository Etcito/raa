//! The render-graph freeze generator (S60): plans one `raa.render-graph` document and prints the
//! Rust module that reconstructs its `GraphPlan` as literals.
//!
//! `cargo run -p raa-graph --bin raa-graph-freeze -- <graph.json> <name>` writes the module of
//! `crates/raa-graph/src/frozen/<name>.rs` to stdout; the sibling `<name>.json` is the committed
//! source document. The name follows RG_FREEZE_NAME, and a name or input refusal goes to stderr
//! with a nonzero exit before anything reaches stdout.

use std::fmt::Write as _;

use raa_graph::catalog::{GraphFormat, ParamValue, PortType, SizeSpec};
use raa_graph::format::ComputePortDecl;
use raa_graph::plan::{
    GraphPlan, InputSource, PlanCompute, PlanHistory, PlanLink, PlanNode, PlanPortRef, PlanResource,
};

/// RG_FREEZE_NAME: `^[a-z][a-z0-9_]{0,47}$`.
const FREEZE_NAME_MAX: usize = 48;

fn main() {
    match run(std::env::args().skip(1)) {
        Ok(module) => print!("{module}"),
        Err(message) => {
            eprintln!("raa-graph-freeze: {message}");
            std::process::exit(1);
        }
    }
}

/// Validates the CLI arguments, plans the input document and returns the module text. Nothing is
/// printed here: the name and the whole input are checked before `main` writes stdout.
fn run(args: impl IntoIterator<Item = String>) -> Result<String, String> {
    let args: Vec<String> = args.into_iter().collect();
    let [input, name] = args.as_slice() else {
        return Err("usage: raa-graph-freeze <graph.json> <name>".to_string());
    };
    if !valid_freeze_name(name) {
        return Err(format!(
            "invalid freeze name {name:?}: RG_FREEZE_NAME is ^[a-z][a-z0-9_]{{0,47}}$"
        ));
    }
    let text = std::fs::read_to_string(input)
        .map_err(|error| format!("cannot read the render graph {input}: {error}"))?;
    let plan = raa_graph::plan_from_json(&text).map_err(|issues| {
        let mut message = format!("{input} is not a valid render graph:");
        for issue in &issues {
            let _ = write!(message, "\n  {}", diagnostic(issue));
        }
        message
    })?;
    Ok(print_plan_module(&plan, name, &format!("{name}.json")))
}

/// One issue as the stderr diagnostic writes it.
fn diagnostic(issue: &raa_graph::error::Issue) -> String {
    let mut line = format!("{}: {}", issue.code, issue.message);
    if let Some(node) = &issue.node {
        let _ = write!(line, " (node {node})");
    }
    if let Some(port) = &issue.port {
        let _ = write!(line, " (port {port})");
    }
    if let Some(param) = &issue.param {
        let _ = write!(line, " (param {param})");
    }
    if let Some(link) = issue.link {
        let _ = write!(line, " (link {link})");
    }
    let _ = write!(line, "; fix: {}", issue.fix);
    line
}

/// Whether `name` matches RG_FREEZE_NAME.
fn valid_freeze_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    first.is_ascii_lowercase()
        && name.len() <= FREEZE_NAME_MAX
        && bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

/// A Rust string literal for `text`, escaped so any value is compilable source.
fn string_literal(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for character in text.chars() {
        for escaped in character.escape_default() {
            out.push(escaped);
        }
    }
    out.push('"');
    out
}

/// The constant name of a compute node's WGSL: `WGSL_<NODE_ID_UPPER>`.
fn wgsl_constant(node_id: &str) -> String {
    let mut name = String::from("WGSL_");
    for character in node_id.chars() {
        name.push(character.to_ascii_uppercase());
    }
    name
}

fn write_param_value(out: &mut String, value: &ParamValue) {
    match value {
        ParamValue::Number(number) => {
            let _ = write!(out, "crate::catalog::ParamValue::Number({number:?})");
        }
        ParamValue::Int(number) => {
            let _ = write!(out, "crate::catalog::ParamValue::Int({number})");
        }
        ParamValue::Bool(value) => {
            let _ = write!(out, "crate::catalog::ParamValue::Bool({value})");
        }
        ParamValue::Enum(value) => {
            let _ = write!(
                out,
                "crate::catalog::ParamValue::Enum({}.to_string())",
                string_literal(value)
            );
        }
        ParamValue::Vec2([x, y]) => {
            let _ = write!(out, "crate::catalog::ParamValue::Vec2([{x:?}, {y:?}])");
        }
        ParamValue::Vec3([x, y, z]) => {
            let _ = write!(
                out,
                "crate::catalog::ParamValue::Vec3([{x:?}, {y:?}, {z:?}])"
            );
        }
        ParamValue::Vec4([x, y, z, w]) => {
            let _ = write!(
                out,
                "crate::catalog::ParamValue::Vec4([{x:?}, {y:?}, {z:?}, {w:?}])"
            );
        }
        ParamValue::String(value) => {
            let _ = write!(
                out,
                "crate::catalog::ParamValue::String({}.to_string())",
                string_literal(value)
            );
        }
    }
}

fn write_size(out: &mut String, size: &SizeSpec) {
    match size {
        SizeSpec::Scale(scale) => {
            let _ = write!(out, "crate::catalog::SizeSpec::Scale({scale:?})");
        }
        SizeSpec::Fixed { width, height } => {
            let _ = write!(
                out,
                "crate::catalog::SizeSpec::Fixed {{ width: {width}, height: {height} }}"
            );
        }
        SizeSpec::FromParam(param) => {
            let _ = write!(
                out,
                "crate::catalog::SizeSpec::FromParam({})",
                string_literal(param)
            );
        }
        SizeSpec::Input(input) => {
            let _ = write!(
                out,
                "crate::catalog::SizeSpec::Input({})",
                string_literal(input)
            );
        }
    }
}

fn write_port_type(out: &mut String, ty: &PortType) {
    let name = match ty {
        PortType::Image2d => "Image2d",
        PortType::Image2dArray => "Image2dArray",
        PortType::Image3d => "Image3d",
        PortType::ImageCube => "ImageCube",
        PortType::Depth2d => "Depth2d",
        PortType::Depth2dArray => "Depth2dArray",
        PortType::Buffer => "Buffer",
    };
    let _ = write!(out, "crate::catalog::PortType::{name}");
}

fn write_graph_format(out: &mut String, format: &GraphFormat) {
    let name = match format {
        GraphFormat::R8Unorm => "R8Unorm",
        GraphFormat::Rg8Unorm => "Rg8Unorm",
        GraphFormat::Rgba8Unorm => "Rgba8Unorm",
        GraphFormat::Rgba8UnormSrgb => "Rgba8UnormSrgb",
        GraphFormat::Bgra8Unorm => "Bgra8Unorm",
        GraphFormat::R16Float => "R16Float",
        GraphFormat::Rg16Float => "Rg16Float",
        GraphFormat::Rgba16Float => "Rgba16Float",
        GraphFormat::R32Float => "R32Float",
        GraphFormat::Rg32Float => "Rg32Float",
        GraphFormat::Rgba32Float => "Rgba32Float",
        GraphFormat::R32Uint => "R32Uint",
        GraphFormat::Rg32Uint => "Rg32Uint",
        GraphFormat::Rgba32Uint => "Rgba32Uint",
        GraphFormat::Rgb10a2Unorm => "Rgb10a2Unorm",
        GraphFormat::Rg11b10Ufloat => "Rg11b10Ufloat",
        GraphFormat::Depth32Float => "Depth32Float",
        GraphFormat::Surface => "Surface",
    };
    let _ = write!(out, "crate::catalog::GraphFormat::{name}");
}

fn write_format(out: &mut String, format: Option<&GraphFormat>) {
    match format {
        Some(format) => {
            out.push_str("Some(");
            write_graph_format(out, format);
            out.push(')');
        }
        None => out.push_str("None"),
    }
}

fn write_port_ref(out: &mut String, port: &PlanPortRef) {
    let _ = write!(
        out,
        "crate::plan::PlanPortRef {{ node: {}.to_string(), port: {}.to_string() }}",
        string_literal(&port.node),
        string_literal(&port.port)
    );
}

fn write_compute_port(out: &mut String, port: &ComputePortDecl) {
    let _ = write!(
        out,
        "crate::format::ComputePortDecl {{ key: {}.to_string(), ty: ",
        string_literal(&port.key)
    );
    write_port_type(out, &port.ty);
    out.push_str(", format: ");
    write_format(out, port.format.as_ref());
    out.push_str(", size: ");
    match &port.size {
        Some(size) => {
            out.push_str("Some(");
            write_size(out, size);
            out.push(')');
        }
        None => out.push_str("None"),
    }
    out.push_str(" }");
}

fn write_input_source(out: &mut String, source: &InputSource) {
    match source {
        InputSource::Link {
            node,
            port,
            history,
        } => {
            let _ = write!(
                out,
                "crate::plan::InputSource::Link {{ node: {}.to_string(), port: {}.to_string(), history: {history} }}",
                string_literal(node),
                string_literal(port)
            );
        }
        InputSource::Default { node, port } => {
            let _ = write!(
                out,
                "crate::plan::InputSource::Default {{ node: {}.to_string(), port: {}.to_string() }}",
                string_literal(node),
                string_literal(port)
            );
        }
        InputSource::Absent => out.push_str("crate::plan::InputSource::Absent"),
    }
}

fn write_node(out: &mut String, node: &PlanNode) {
    out.push_str("        crate::plan::PlanNode {\n");
    let _ = writeln!(
        out,
        "            id: {}.to_string(),",
        string_literal(&node.id)
    );
    let _ = writeln!(
        out,
        "            type_id: {}.to_string(),",
        string_literal(&node.type_id)
    );
    let _ = writeln!(out, "            type_version: {},", node.type_version);
    let _ = writeln!(out, "            order_index: {},", node.order_index);
    if node.params.is_empty() {
        out.push_str("            params: vec![],\n");
    } else {
        out.push_str("            params: vec![\n");
        for (key, value) in &node.params {
            let _ = write!(
                out,
                "                ({}.to_string(), ",
                string_literal(key)
            );
            write_param_value(out, value);
            out.push_str("),\n");
        }
        out.push_str("            ],\n");
    }
    if node.inputs.is_empty() {
        out.push_str("            inputs: vec![],\n");
    } else {
        out.push_str("            inputs: vec![\n");
        for input in &node.inputs {
            out.push_str("                crate::plan::PlanInput { port: ");
            out.push_str(&string_literal(&input.port));
            out.push_str(".to_string(), source: ");
            write_input_source(out, &input.source);
            out.push_str(" },\n");
        }
        out.push_str("            ],\n");
    }
    match &node.compute {
        None => out.push_str("            compute: None,\n"),
        Some(compute) => write_compute(out, &node.id, compute),
    }
    out.push_str("        },\n");
}

fn write_compute(out: &mut String, node_id: &str, compute: &PlanCompute) {
    out.push_str("            compute: Some(crate::plan::PlanCompute {\n");
    let _ = writeln!(
        out,
        "                wgsl: {}.to_string(),",
        wgsl_constant(node_id)
    );
    if compute.inputs.is_empty() {
        out.push_str("                inputs: vec![],\n");
    } else {
        out.push_str("                inputs: vec![\n");
        for port in &compute.inputs {
            out.push_str("                    ");
            write_compute_port(out, port);
            out.push_str(",\n");
        }
        out.push_str("                ],\n");
    }
    if compute.outputs.is_empty() {
        out.push_str("                outputs: vec![],\n");
    } else {
        out.push_str("                outputs: vec![\n");
        for port in &compute.outputs {
            out.push_str("                    ");
            write_compute_port(out, port);
            out.push_str(",\n");
        }
        out.push_str("                ],\n");
    }
    let _ = writeln!(
        out,
        "                workgroup: [{}, {}],",
        compute.workgroup[0], compute.workgroup[1]
    );
    out.push_str("            }),\n");
}

fn write_link(out: &mut String, link: &PlanLink) {
    out.push_str("        crate::plan::PlanLink { from: ");
    write_port_ref(out, &link.from);
    out.push_str(", to: ");
    write_port_ref(out, &link.to);
    let _ = writeln!(out, ", history: {}, }},", link.history);
}

fn write_resource(out: &mut String, resource: &PlanResource) {
    out.push_str("        crate::plan::PlanResource {\n");
    let _ = writeln!(out, "            id: {},", resource.id);
    out.push_str("            producer: ");
    write_port_ref(out, &resource.producer);
    out.push_str(",\n");
    out.push_str("            ty: ");
    write_port_type(out, &resource.ty);
    out.push_str(",\n            format: ");
    write_format(out, resource.format.as_ref());
    out.push_str(",\n            size: ");
    write_size(out, &resource.size);
    out.push_str(",\n");
    let _ = writeln!(out, "            mips: {},", resource.mips);
    let _ = writeln!(out, "            layers: {},", resource.layers);
    let _ = writeln!(out, "            first_use: {},", resource.first_use);
    let _ = writeln!(out, "            last_use: {},", resource.last_use);
    let _ = writeln!(out, "            history: {},", resource.history);
    out.push_str("        },\n");
}

fn write_history(out: &mut String, history: &PlanHistory) {
    let _ = writeln!(out, "        crate::plan::PlanHistory {{");
    let _ = writeln!(out, "            resource: {},", history.resource);
    out.push_str("            consumer: ");
    write_port_ref(out, &history.consumer);
    out.push_str(",\n        },\n");
}

/// The complete module text of `plan`, as the frozen module `crates/raa-graph/src/frozen/<name>.rs`
/// holds it. `source_file` is the sibling JSON filename the module `include_str!`s; it is written
/// as given, so a caller passes a bare file name and never the input directory.
///
/// Handoff: 'Freeze Simple forward and establish plan equality' consumes `pub fn print_plan_module(plan: &GraphPlan, name: &str, source_file: &str) -> String` from `crates/raa-graph/src/bin/raa-graph-freeze.rs` through its CLI; the CLI attaches the printer to stdout and passes the planned Simple forward document, registry key `simple_forward` and sibling filename `simple_forward.json`.
///
/// Handoff: 'Freeze Full GI' consumes `pub fn print_plan_module(plan: &GraphPlan, name: &str, source_file: &str) -> String` from `crates/raa-graph/src/bin/raa-graph-freeze.rs` through its CLI; the CLI attaches the printer to stdout and passes the planned Full GI document, registry key `full_gi` and sibling filename `full_gi.json`.
///
/// Handoff: 'Freeze Debug and finish the three-starter registry' consumes `pub fn print_plan_module(plan: &GraphPlan, name: &str, source_file: &str) -> String` from `crates/raa-graph/src/bin/raa-graph-freeze.rs` through its CLI; the CLI attaches the printer to stdout and passes the planned Debug document, registry key `debug` and sibling filename `debug.json`.
///
/// no handoff: Document the repeatable freeze procedure takes nothing from this card.
pub fn print_plan_module(plan: &GraphPlan, name: &str, source_file: &str) -> String {
    let mut out = String::new();
    out.push_str(
        "//! A frozen render graph, generated by `raa-graph-freeze`; do not edit by hand.\n",
    );
    let _ = writeln!(out, "//! Registry key: {name:?}.");
    out.push_str(
        "//!\n//! `plan()` equals `plan_from_json(SOURCE_JSON)` of the committed sibling JSON.\n\n",
    );
    let _ = writeln!(
        out,
        "pub const SOURCE_JSON: &str = include_str!({});\n",
        string_literal(source_file)
    );
    for node in &plan.nodes {
        if let Some(compute) = &node.compute {
            let _ = writeln!(
                out,
                "const {}: &str = {};\n",
                wgsl_constant(&node.id),
                string_literal(&compute.wgsl)
            );
        }
    }
    out.push_str("pub fn plan() -> crate::plan::GraphPlan {\n");
    out.push_str("    crate::plan::GraphPlan {\n");
    let _ = writeln!(
        out,
        "        name: {}.to_string(),",
        string_literal(&plan.name)
    );
    write_nodes(&mut out, &plan.nodes);
    write_links(&mut out, &plan.links);
    write_resources(&mut out, &plan.resources);
    write_histories(&mut out, &plan.histories);
    out.push_str("        output: ");
    write_port_ref(&mut out, &plan.output);
    out.push_str(",\n    }\n}\n");
    out
}

fn write_nodes(out: &mut String, nodes: &[PlanNode]) {
    if nodes.is_empty() {
        out.push_str("        nodes: vec![],\n");
        return;
    }
    out.push_str("        nodes: vec![\n");
    for node in nodes {
        write_node(out, node);
    }
    out.push_str("        ],\n");
}

fn write_links(out: &mut String, links: &[PlanLink]) {
    if links.is_empty() {
        out.push_str("        links: vec![],\n");
        return;
    }
    out.push_str("        links: vec![\n");
    for link in links {
        write_link(out, link);
    }
    out.push_str("        ],\n");
}

fn write_resources(out: &mut String, resources: &[PlanResource]) {
    if resources.is_empty() {
        out.push_str("        resources: vec![],\n");
        return;
    }
    out.push_str("        resources: vec![\n");
    for resource in resources {
        write_resource(out, resource);
    }
    out.push_str("        ],\n");
}

fn write_histories(out: &mut String, histories: &[PlanHistory]) {
    if histories.is_empty() {
        out.push_str("        histories: vec![],\n");
        return;
    }
    out.push_str("        histories: vec![\n");
    for history in histories {
        write_history(out, history);
    }
    out.push_str("        ],\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_graph::plan::PlanInput;

    fn port(node: &str, key: &str) -> PlanPortRef {
        PlanPortRef {
            node: node.to_string(),
            port: key.to_string(),
        }
    }

    fn empty_plan() -> GraphPlan {
        GraphPlan {
            name: "Test".to_string(),
            nodes: vec![PlanNode {
                id: "out".to_string(),
                type_id: "output.display".to_string(),
                type_version: 1,
                order_index: 0,
                params: Vec::new(),
                inputs: Vec::new(),
                compute: None,
            }],
            links: Vec::new(),
            resources: Vec::new(),
            histories: Vec::new(),
            output: port("out", "color"),
        }
    }

    #[test]
    fn freeze_name_boundaries() {
        assert!(valid_freeze_name("a"));
        assert!(valid_freeze_name("simple_forward"));
        assert!(valid_freeze_name("a0_b9"));
        assert!(valid_freeze_name(&"a".repeat(48)));
        assert!(!valid_freeze_name(&"a".repeat(49)));
        for bad in ["", "A", "0a", "_a", "a-b", "a b", "a.", "a/b", "á"] {
            assert!(!valid_freeze_name(bad), "{bad:?} is not RG_FREEZE_NAME");
        }
    }

    #[test]
    fn cli_validates_name_and_input_before_stdout() {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let input = format!("{manifest}/graphs/simple-forward.json");
        let module = run([input.clone(), "simple_forward".to_string()]).expect("the starter plans");
        assert!(module.contains(r#"include_str!("simple_forward.json")"#));
        assert!(!module.contains(manifest));
        assert!(run([input.clone(), "Simple".to_string()]).is_err());
        assert!(run([input.clone(), String::new()]).is_err());
        assert!(run(["/missing/graph.json".to_string(), "simple".to_string()]).is_err());
        assert!(run([format!("{manifest}/Cargo.toml"), "simple".to_string()]).is_err());
        assert!(run(Vec::new()).is_err());
        assert!(run(["only-one".to_string()]).is_err());
    }

    #[test]
    fn strings_are_escaped() {
        let mut plan = empty_plan();
        plan.name = "say \"hi\"\n\ttab \\ é".to_string();
        plan.nodes[0].params = vec![(
            "text".to_string(),
            ParamValue::String("a\"b\né".to_string()),
        )];
        let text = print_plan_module(&plan, "escaping", "escaping.json");
        assert!(text.contains(r#"name: "say \"hi\"\n\ttab \\ \u{e9}".to_string()"#));
        assert!(text.contains(r#"crate::catalog::ParamValue::String("a\"b\n\u{e9}".to_string())"#));
        assert!(text.contains(r#"("text".to_string(), "#));
    }

    #[test]
    fn every_param_value_is_a_rust_literal() {
        let mut plan = empty_plan();
        plan.nodes[0].params = vec![
            ("number".to_string(), ParamValue::Number(-1.25)),
            ("int".to_string(), ParamValue::Int(-7)),
            ("bool".to_string(), ParamValue::Bool(true)),
            ("enum".to_string(), ParamValue::Enum("agx".to_string())),
            ("vec2".to_string(), ParamValue::Vec2([0.0, -0.0])),
            ("vec3".to_string(), ParamValue::Vec3([1.0, 2.5, -3.0])),
            ("vec4".to_string(), ParamValue::Vec4([1.0, 2.0, 3.0, 4.0])),
            (
                "string".to_string(),
                ParamValue::String("hello".to_string()),
            ),
        ];
        let text = print_plan_module(&plan, "literals", "literals.json");
        for expected in [
            r#"("number".to_string(), crate::catalog::ParamValue::Number(-1.25))"#,
            r#"("int".to_string(), crate::catalog::ParamValue::Int(-7))"#,
            r#"("bool".to_string(), crate::catalog::ParamValue::Bool(true))"#,
            r#"("enum".to_string(), crate::catalog::ParamValue::Enum("agx".to_string()))"#,
            r#"("vec2".to_string(), crate::catalog::ParamValue::Vec2([0.0, -0.0]))"#,
            r#"("vec3".to_string(), crate::catalog::ParamValue::Vec3([1.0, 2.5, -3.0]))"#,
            r#"("vec4".to_string(), crate::catalog::ParamValue::Vec4([1.0, 2.0, 3.0, 4.0]))"#,
            r#"("string".to_string(), crate::catalog::ParamValue::String("hello".to_string()))"#,
        ] {
            assert!(text.contains(expected), "missing {expected}");
        }
    }

    #[test]
    fn sizes_ports_and_formats_are_literals() {
        let mut plan = empty_plan();
        plan.nodes[0].inputs = vec![
            PlanInput {
                port: "linked".to_string(),
                source: InputSource::Link {
                    node: "a".to_string(),
                    port: "color".to_string(),
                    history: true,
                },
            },
            PlanInput {
                port: "defaulted".to_string(),
                source: InputSource::Default {
                    node: "b".to_string(),
                    port: "depth".to_string(),
                },
            },
            PlanInput {
                port: "absent".to_string(),
                source: InputSource::Absent,
            },
        ];
        plan.resources = vec![
            PlanResource {
                id: 1,
                producer: port("a", "color"),
                ty: PortType::Image2dArray,
                format: Some(GraphFormat::Rgba16Float),
                size: SizeSpec::Scale(0.5),
                mips: 1,
                layers: 2,
                first_use: 0,
                last_use: 1,
                history: false,
            },
            PlanResource {
                id: 2,
                producer: port("a", "depth"),
                ty: PortType::Depth2dArray,
                format: None,
                size: SizeSpec::Fixed {
                    width: 1920,
                    height: 1080,
                },
                mips: 0,
                layers: 4,
                first_use: 0,
                last_use: 1,
                history: false,
            },
            PlanResource {
                id: 3,
                producer: port("b", "buffer"),
                ty: PortType::Buffer,
                format: None,
                size: SizeSpec::FromParam("resolution"),
                mips: 0,
                layers: 0,
                first_use: 0,
                last_use: 1,
                history: false,
            },
            PlanResource {
                id: 4,
                producer: port("b", "out"),
                ty: PortType::Image3d,
                format: Some(GraphFormat::Surface),
                size: SizeSpec::Input("depth"),
                mips: 0,
                layers: 0,
                first_use: 0,
                last_use: 1,
                history: false,
            },
        ];
        let text = print_plan_module(&plan, "sizes", "sizes.json");
        for expected in [
            r#"crate::plan::InputSource::Link { node: "a".to_string(), port: "color".to_string(), history: true }"#,
            r#"crate::plan::InputSource::Default { node: "b".to_string(), port: "depth".to_string() }"#,
            "crate::plan::InputSource::Absent",
            "crate::catalog::SizeSpec::Scale(0.5)",
            "crate::catalog::SizeSpec::Fixed { width: 1920, height: 1080 }",
            r#"crate::catalog::SizeSpec::FromParam("resolution")"#,
            r#"crate::catalog::SizeSpec::Input("depth")"#,
            "crate::catalog::PortType::Image2dArray",
            "crate::catalog::PortType::Depth2dArray",
            "crate::catalog::PortType::Buffer",
            "crate::catalog::PortType::Image3d",
            "Some(crate::catalog::GraphFormat::Rgba16Float)",
            "Some(crate::catalog::GraphFormat::Surface)",
            "format: None",
        ] {
            assert!(text.contains(expected), "missing {expected}");
        }
    }

    #[test]
    fn resources_and_histories_keep_order() {
        let mut plan = empty_plan();
        plan.resources = vec![
            PlanResource {
                id: 1,
                producer: port("producer", "first"),
                ty: PortType::Image2d,
                format: Some(GraphFormat::Rgba16Float),
                size: SizeSpec::Scale(1.0),
                mips: 1,
                layers: 1,
                first_use: 0,
                last_use: 2,
                history: false,
            },
            PlanResource {
                id: 2,
                producer: port("producer", "second"),
                ty: PortType::Image2d,
                format: Some(GraphFormat::Rgba16Float),
                size: SizeSpec::Scale(1.0),
                mips: 1,
                layers: 1,
                first_use: 0,
                last_use: u32::MAX,
                history: true,
            },
        ];
        plan.histories = vec![PlanHistory {
            resource: 2,
            consumer: port("consumer", "history"),
        }];
        let text = print_plan_module(&plan, "order", "order.json");
        assert!(text.find("id: 1,").unwrap() < text.find("id: 2,").unwrap());
        assert!(text.contains("last_use: 4294967295,"));
        assert!(text.contains("crate::plan::PlanHistory {"));
        assert!(text.contains("resource: 2,"));
        assert!(text.contains(r#"crate::plan::PlanPortRef { node: "consumer".to_string(), port: "history".to_string() }"#));
    }

    #[test]
    fn compute_wgsl_is_printed_as_a_constant() {
        let mut plan = empty_plan();
        plan.nodes[0].id = "denoise".to_string();
        plan.nodes[0].type_id = "compute.wgsl".to_string();
        plan.nodes[0].compute = Some(PlanCompute {
            wgsl: "@compute @workgroup_size(8, 8)\nfn rg_main(p: vec2<u32>) {\n  // \"quote\"\n}"
                .to_string(),
            inputs: vec![ComputePortDecl {
                key: "src".to_string(),
                ty: PortType::Image2d,
                format: Some(GraphFormat::Rgba16Float),
                size: Some(SizeSpec::Scale(1.0)),
            }],
            outputs: vec![ComputePortDecl {
                key: "dst".to_string(),
                ty: PortType::Image2d,
                format: Some(GraphFormat::Rgba32Float),
                size: None,
            }],
            workgroup: [8, 8],
        });
        let text = print_plan_module(&plan, "compute", "compute.json");
        assert!(text.contains(
            r#"const WGSL_DENOISE: &str = "@compute @workgroup_size(8, 8)\nfn rg_main(p: vec2<u32>) {\n  // \"quote\"\n}";"#
        ));
        assert!(text.contains("wgsl: WGSL_DENOISE.to_string(),"));
        assert!(text.contains(
            r#"key: "src".to_string(), ty: crate::catalog::PortType::Image2d, format: Some(crate::catalog::GraphFormat::Rgba16Float), size: Some(crate::catalog::SizeSpec::Scale(1.0))"#
        ));
        assert!(text.contains(r#"key: "dst".to_string(), ty: crate::catalog::PortType::Image2d, format: Some(crate::catalog::GraphFormat::Rgba32Float), size: None"#));
        assert!(text.contains("workgroup: [8, 8],"));
    }

    #[test]
    fn every_port_type_and_format_is_a_variant_literal() {
        for (ty, name) in [
            (PortType::Image2d, "Image2d"),
            (PortType::Image2dArray, "Image2dArray"),
            (PortType::Image3d, "Image3d"),
            (PortType::ImageCube, "ImageCube"),
            (PortType::Depth2d, "Depth2d"),
            (PortType::Depth2dArray, "Depth2dArray"),
            (PortType::Buffer, "Buffer"),
        ] {
            let mut out = String::new();
            write_port_type(&mut out, &ty);
            assert_eq!(out, format!("crate::catalog::PortType::{name}"));
        }
        for (format, name) in [
            (GraphFormat::R8Unorm, "R8Unorm"),
            (GraphFormat::Rg8Unorm, "Rg8Unorm"),
            (GraphFormat::Rgba8Unorm, "Rgba8Unorm"),
            (GraphFormat::Rgba8UnormSrgb, "Rgba8UnormSrgb"),
            (GraphFormat::Bgra8Unorm, "Bgra8Unorm"),
            (GraphFormat::R16Float, "R16Float"),
            (GraphFormat::Rg16Float, "Rg16Float"),
            (GraphFormat::Rgba16Float, "Rgba16Float"),
            (GraphFormat::R32Float, "R32Float"),
            (GraphFormat::Rg32Float, "Rg32Float"),
            (GraphFormat::Rgba32Float, "Rgba32Float"),
            (GraphFormat::R32Uint, "R32Uint"),
            (GraphFormat::Rg32Uint, "Rg32Uint"),
            (GraphFormat::Rgba32Uint, "Rgba32Uint"),
            (GraphFormat::Rgb10a2Unorm, "Rgb10a2Unorm"),
            (GraphFormat::Rg11b10Ufloat, "Rg11b10Ufloat"),
            (GraphFormat::Depth32Float, "Depth32Float"),
            (GraphFormat::Surface, "Surface"),
        ] {
            let mut out = String::new();
            write_graph_format(&mut out, &format);
            assert_eq!(out, format!("crate::catalog::GraphFormat::{name}"));
        }
    }

    #[test]
    fn every_repository_graph_plan_is_printed() {
        let base = format!("{}/graphs", env!("CARGO_MANIFEST_DIR"));
        for (file, name) in [
            ("simple-forward.json", "simple_forward"),
            ("full-gi.json", "full_gi"),
            ("debug.json", "debug"),
            ("full-gi-stages.json", "full_gi_stages"),
            ("custom-denoise.json", "custom_denoise"),
        ] {
            let text = std::fs::read_to_string(format!("{base}/{file}")).unwrap();
            let plan = raa_graph::plan_from_json(&text).unwrap();
            let module = print_plan_module(&plan, name, &format!("{name}.json"));
            assert_eq!(
                module.matches("crate::plan::PlanNode {").count(),
                plan.nodes.len(),
                "{file} nodes"
            );
            assert_eq!(
                module.matches("crate::plan::PlanResource {").count(),
                plan.resources.len(),
                "{file} resources"
            );
            assert_eq!(
                module.matches("crate::plan::PlanHistory {").count(),
                plan.histories.len(),
                "{file} histories"
            );
            assert!(module.contains(&format!("name: {}", string_literal(&plan.name))));
            assert!(module.ends_with("}\n"));
        }
    }
}
