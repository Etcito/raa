//! Core validation and planning of declared compute nodes.

use raa_graph::catalog::{Catalog, GraphFormat, ParamValue, PortType, SizeSpec};
use raa_graph::error::Stage;
use raa_graph::format::{
    ComputeDecl, ComputePortDecl, GraphDoc, LinkDoc, NodeDoc, PortRef, parse_graph,
};
use raa_graph::plan::{InputSource, plan_from_doc};
use raa_graph::registry;
use raa_graph::validate::{GraphLimits, ValidateCtx, validate};
use raa_loader::json::JsonValue;

const BODY: &str = "fn rg_main(p: vec2<u32>) {\n  textureStore(rg_out_dst, p, textureLoad(rg_in_src, vec2<i32>(p), 0));\n}";

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

fn base() -> GraphDoc {
    let text = registry::STARTERS
        .iter()
        .find(|&&(_, stem, _)| stem == "simple-forward")
        .unwrap()
        .2;
    parse_graph(text).unwrap()
}

fn port(key: &str, ty: PortType, format: Option<GraphFormat>) -> ComputePortDecl {
    ComputePortDecl {
        key: key.into(),
        ty,
        format,
        size: None,
    }
}

fn compute(id: &str, input_format: GraphFormat) -> NodeDoc {
    NodeDoc {
        id: id.into(),
        type_id: "compute.wgsl".into(),
        version: None,
        params: Vec::new(),
        compute: Some(ComputeDecl {
            inputs: vec![port("src", PortType::Image2d, Some(input_format))],
            outputs: vec![port(
                "dst",
                PortType::Image2d,
                Some(GraphFormat::Rgba16Float),
            )],
            math: JsonValue::Null,
            wgsl: BODY.into(),
        }),
    }
}

fn end(node: &str, port: &str) -> PortRef {
    PortRef {
        node: node.into(),
        port: port.into(),
    }
}
fn link(from: (&str, &str), to: (&str, &str)) -> LinkDoc {
    LinkDoc {
        from: end(from.0, from.1),
        to: end(to.0, to.1),
        history: false,
    }
}

fn graph() -> GraphDoc {
    let mut doc = base();
    doc.links
        .retain(|link| !(link.from.node == "tonemap" && link.to.node == "out"));
    doc.nodes.push(compute("calc", GraphFormat::Rgba16Float));
    doc.nodes.push(NodeDoc {
        id: "show".into(),
        type_id: "view.image".into(),
        version: None,
        params: Vec::new(),
        compute: None,
    });
    doc.links.push(link(("prepass", "normal"), ("calc", "src")));
    doc.links.push(link(("calc", "dst"), ("show", "src")));
    doc.links.push(link(("show", "color"), ("out", "color")));
    doc
}

#[test]
fn declared_ports_validate_and_plan_with_resource_and_order() {
    let doc = graph();
    with_ctx(|ctx| {
        assert!(validate(&doc, ctx).is_empty());
        let (plan, _) = plan_from_doc(&doc, ctx).unwrap();
        let calc = plan.nodes.iter().find(|node| node.id == "calc").unwrap();
        assert_eq!(calc.inputs.len(), 1);
        assert_eq!(calc.inputs[0].port, "src");
        assert_eq!(
            calc.inputs[0].source,
            InputSource::Link {
                node: "prepass".into(),
                port: "normal".into(),
                history: false
            }
        );
        assert!(
            calc.order_index
                > plan
                    .nodes
                    .iter()
                    .find(|node| node.id == "prepass")
                    .unwrap()
                    .order_index
        );
        assert!(
            plan.resources
                .iter()
                .any(|resource| resource.producer.node == "calc"
                    && resource.producer.port == "dst"
                    && resource.ty == PortType::Image2d
                    && resource.format == Some(GraphFormat::Rgba16Float)
                    && resource.size == SizeSpec::Scale(1.0))
        );
    });
}

#[test]
fn declared_input_is_required_and_params_keep_document_order_and_inferred_kinds() {
    let mut doc = graph();
    doc.links
        .retain(|link| !(link.to.node == "calc" && link.to.port == "src"));
    let issues = with_ctx(|ctx| validate(&doc, ctx));
    assert!(
        issues
            .iter()
            .any(|issue| issue.code == "graph.input_missing"
                && issue.node.as_deref() == Some("calc")
                && issue.port.as_deref() == Some("src"))
    );
    let mut doc = graph();
    let calc = doc.nodes.iter_mut().find(|node| node.id == "calc").unwrap();
    calc.params = vec![
        ("gain".into(), JsonValue::Number(0.5)),
        ("flag".into(), JsonValue::Bool(true)),
        (
            "tint".into(),
            JsonValue::Array(vec![
                JsonValue::Number(1.0),
                JsonValue::Number(2.0),
                JsonValue::Number(3.0),
            ]),
        ),
    ];
    with_ctx(|ctx| {
        let (plan, _) = plan_from_doc(&doc, ctx).unwrap();
        let params = &plan
            .nodes
            .iter()
            .find(|node| node.id == "calc")
            .unwrap()
            .params;
        assert_eq!(
            params,
            &[
                ("gain".into(), ParamValue::Number(0.5)),
                ("flag".into(), ParamValue::Bool(true)),
                ("tint".into(), ParamValue::Vec3([1.0, 2.0, 3.0]))
            ]
        );
    });
}

#[test]
fn declaration_budget_and_compute_format_mismatch_are_reported() {
    let mut doc = graph();
    doc.nodes
        .iter_mut()
        .find(|node| node.id == "calc")
        .unwrap()
        .compute
        .as_mut()
        .unwrap()
        .outputs = (0..5)
        .map(|i| {
            port(
                if i == 0 {
                    "dst".to_string()
                } else {
                    format!("dst{i}")
                }
                .as_str(),
                PortType::Image2d,
                Some(GraphFormat::Rgba16Float),
            )
        })
        .collect();
    doc.nodes
        .iter_mut()
        .find(|node| node.id == "calc")
        .unwrap()
        .compute
        .as_mut()
        .unwrap()
        .wgsl = "fn rg_main(p: vec2<u32>) {}".into();
    let issues = with_ctx(|ctx| validate(&doc, ctx));
    let budget = issues
        .iter()
        .find(|issue| issue.code == "graph.budget")
        .unwrap();
    assert_eq!(budget.stage, Stage::Budget);
    assert!(budget.message.contains("5 storage textures") && budget.message.contains("limit is 4"));

    let mut doc = graph();
    doc.nodes.push(compute("calc2", GraphFormat::Rgba8Unorm));
    doc.links.push(link(("calc", "dst"), ("calc2", "src")));
    let issues = with_ctx(|ctx| validate(&doc, ctx));
    assert!(issues.iter().any(|issue| issue.code == "graph.link_type"
        && issue.node.as_deref() == Some("calc2")
        && issue.port.as_deref() == Some("src")));
}
