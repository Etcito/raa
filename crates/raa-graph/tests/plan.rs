//! Planning tests of the render graph (S53): the executed set and its warning, the execution order,
//! the resources and histories, the Engine options and determinism. The catalog is hand-built, as
//! tests/validate.rs builds it.

use raa_graph::catalog::{
    Catalog, Category, NodeTypeSpec, ParamBinding, ParamKind, ParamSpec, ParamValue, PortSpec,
    PortType, Rebuild, SizeSpec,
};
use raa_graph::error::{Issue, Severity, Stage};
use raa_graph::format::parse_graph;
use raa_graph::plan::{
    GraphPlan, InputSource, PlanHistory, PlanNode, PlanPortRef, plan_from_doc, plan_from_json,
};
use raa_graph::validate::{GraphLimits, ValidateCtx};

/// A port of type `ty` with no default source; it takes any link (`generic`).
fn port(key: &'static str, ty: PortType) -> PortSpec {
    PortSpec {
        key,
        label: key,
        ty,
        format: None,
        size: SizeSpec::Scale(1.0),
        mips: 1,
        layers: 1,
        required: false,
        default_source: &[],
        absent_ok: true,
        generic: true,
        history: false,
        slot: None,
    }
}

/// A parameter of `kind` and `default` with no bounds, read by the node itself.
fn param(key: &'static str, kind: ParamKind, default: ParamValue) -> ParamSpec {
    ParamSpec {
        key,
        label: key,
        group: "test",
        kind,
        default,
        min: None,
        max: None,
        step: None,
        options: &[],
        unit: "",
        rebuild: Rebuild::None,
        primary: false,
        unstated_range: false,
        binding: ParamBinding::Node,
        source: "crates/raa-graph/tests/plan.rs",
    }
}

/// An upgrade with nothing to convert: the catalog version is 1.
fn keep(_from: u32, _params: &mut Vec<(String, ParamValue)>) -> Result<(), String> {
    Ok(())
}

/// A multi-instance node type of default order `order` with the given ports.
fn node_type(
    id: &'static str,
    order: u32,
    inputs: Vec<PortSpec>,
    outputs: Vec<PortSpec>,
) -> NodeTypeSpec {
    NodeTypeSpec {
        id,
        version: 1,
        category: Category::Lighting,
        label: id,
        description: id,
        pass: None,
        contributor: false,
        default_order: order,
        multi_instance: true,
        side_effect: false,
        required_in_graph: false,
        inputs,
        outputs,
        params: Vec::new(),
        budgets: Vec::new(),
        upgrade: keep,
        family: "tests",
    }
}

/// The test catalog: the three required passes and the display output, the Engine-bound `msaa`
/// of the forward pass, and the test nodes of each scenario. `fx.hi` and `fx.lo` are contributors;
/// `fx.dead` and `fx.stale` are neither contributors nor side-effect nodes; the other `fx.*` nodes
/// are side-effect nodes.
fn catalog() -> Catalog {
    let prepass = node_type(
        "pass.prepass",
        30,
        Vec::new(),
        vec![
            port("depth", PortType::Depth2d),
            port("normal", PortType::Image2d),
            port("material", PortType::Image2d),
        ],
    );
    let forward = NodeTypeSpec {
        params: vec![ParamSpec {
            binding: ParamBinding::Engine { option: "msaa" },
            options: &["1", "4"],
            ..param("msaa", ParamKind::Int, ParamValue::Int(4))
        }],
        ..node_type(
            "pass.forward",
            60,
            vec![
                PortSpec {
                    default_source: &["pass.prepass.depth"],
                    ..port("depth", PortType::Depth2d)
                },
                port("lightTiles", PortType::Image2dArray),
            ],
            vec![port("color", PortType::Image2d)],
        )
    };
    let tonemap = node_type(
        "pass.tonemap",
        200,
        vec![PortSpec {
            default_source: &["pass.forward.color"],
            ..port("color", PortType::Image2d)
        }],
        vec![port("color", PortType::Image2d)],
    );
    let output = node_type(
        "output.display",
        220,
        vec![PortSpec {
            default_source: &["pass.tonemap.color"],
            ..port("color", PortType::Image2d)
        }],
        Vec::new(),
    );
    let normal = || PortSpec {
        default_source: &["pass.prepass.normal"],
        ..port("normal", PortType::Image2d)
    };
    let side = |id: &'static str, order: u32| NodeTypeSpec {
        side_effect: true,
        ..node_type(id, order, vec![normal()], Vec::new())
    };
    let contributor = |id: &'static str, order: u32| NodeTypeSpec {
        contributor: true,
        ..node_type(id, order, Vec::new(), Vec::new())
    };
    let shade = NodeTypeSpec {
        side_effect: true,
        ..node_type(
            "fx.shade",
            40,
            vec![normal()],
            vec![port("color", PortType::Image2d)],
        )
    };
    let post = NodeTypeSpec {
        side_effect: true,
        ..node_type(
            "fx.post",
            100,
            vec![port("depth", PortType::Depth2d)],
            vec![port("color", PortType::Image2d)],
        )
    };
    let accum = NodeTypeSpec {
        side_effect: true,
        ..node_type(
            "fx.accum",
            50,
            vec![PortSpec {
                history: true,
                ..port("history", PortType::Image2d)
            }],
            vec![port("color", PortType::Image2d)],
        )
    };
    let reader = NodeTypeSpec {
        side_effect: true,
        ..node_type(
            "fx.reader",
            70,
            vec![port("color", PortType::Image2d)],
            Vec::new(),
        )
    };
    Catalog {
        nodes: vec![
            prepass,
            forward,
            tonemap,
            output,
            shade,
            post,
            accum,
            reader,
            side("fx.low", 10),
            side("fx.mid", 50),
            side("fx.high", 90),
            contributor("fx.hi", 900),
            contributor("fx.lo", 450),
            node_type("fx.dead", 50, Vec::new(), Vec::new()),
            node_type(
                "fx.stale",
                50,
                Vec::new(),
                vec![port("color", PortType::Image2d)],
            ),
        ],
        starters: Vec::new(),
        presets: Vec::new(),
        frozen: Vec::new(),
    }
}

/// Plans `text` against the test catalog with the core limits, one viewport and `msaa` as the
/// engine's option.
fn plan_at(text: &str, msaa: i64) -> Result<(GraphPlan, Vec<Issue>), Vec<Issue>> {
    let catalog = catalog();
    let installed = |_: &str| true;
    let engine_options = [("msaa", ParamValue::Int(msaa))];
    let ctx = ValidateCtx {
        catalog: &catalog,
        limits: GraphLimits::CORE,
        installed: &installed,
        viewports: 1,
        engine_options: &engine_options,
    };
    plan_from_doc(&parse_graph(text).expect("the fixture parses"), &ctx)
}

/// The plan of `text` at `msaa` 4, which must plan without an error.
fn planned(text: &str) -> (GraphPlan, Vec<Issue>) {
    plan_at(text, 4).expect("the fixture plans")
}

/// The only issue with `code`; the test fails when there are none or several.
fn one<'a>(issues: &'a [Issue], code: &str) -> &'a Issue {
    let found: Vec<&Issue> = issues.iter().filter(|issue| issue.code == code).collect();
    assert_eq!(found.len(), 1, "expected one `{code}` in {issues:#?}");
    found[0]
}

/// The node ids of a plan in execution order.
fn ids(plan: &GraphPlan) -> Vec<&str> {
    plan.nodes.iter().map(|node| node.id.as_str()).collect()
}

/// The planned node `id`.
fn node_of<'a>(plan: &'a GraphPlan, id: &str) -> &'a PlanNode {
    plan.nodes
        .iter()
        .find(|node| node.id == id)
        .expect("the node is planned")
}

/// A port reference of a plan.
fn at(node: &str, port: &str) -> PlanPortRef {
    PlanPortRef {
        node: node.to_string(),
        port: port.to_string(),
    }
}

/// A node object named `id` of type `type_id`.
fn node(id: &str, type_id: &str) -> String {
    format!(r#"{{"id":"{id}","type":"{type_id}"}}"#)
}

/// A link object from `from` (`node.port`) to `to`.
fn link(from: &str, to: &str) -> String {
    format!(r#"{{"from":"{from}","to":"{to}"}}"#)
}

/// A history link object from `from` to `to`.
fn history(from: &str, to: &str) -> String {
    format!(r#"{{"from":"{from}","to":"{to}","history":true}}"#)
}

/// A `raa.render-graph` document of the given node and link objects.
fn doc(nodes: &[String], links: &[String]) -> String {
    format!(
        r#"{{"format":"raa.render-graph","version":1,"name":"test","nodes":[{}],"links":[{}]}}"#,
        nodes.join(","),
        links.join(",")
    )
}

/// The required passes and the display output, linked from forward to tonemap.
fn base() -> Vec<String> {
    vec![
        node("prepass", "pass.prepass"),
        node("forward", "pass.forward"),
        node("tonemap", "pass.tonemap"),
        node("out", "output.display"),
    ]
}

#[test]
fn the_executed_set_runs_what_reaches_the_output_and_warns_about_the_rest() {
    let mut nodes = base();
    nodes.extend([node("shade", "fx.shade"), node("dead", "fx.dead")]);
    let text = doc(
        &nodes,
        &[
            link("prepass.depth", "forward.depth"),
            link("forward.color", "tonemap.color"),
        ],
    );
    let (plan, warnings) = planned(&text);
    assert_eq!(
        ids(&plan),
        ["prepass", "shade", "forward", "tonemap", "out"]
    );
    let warning = one(&warnings, "graph.unreachable");
    assert_eq!(
        (warning.severity, warning.stage, warning.node.as_deref()),
        (Severity::Warning, Stage::Plan, Some("dead"))
    );
    assert_eq!(warnings.len(), 1);
}

#[test]
fn ready_nodes_run_by_default_order_then_document_index() {
    let mut nodes = base();
    nodes.extend([
        node("midB", "fx.mid"),
        node("midA", "fx.mid"),
        node("low", "fx.low"),
        node("high", "fx.high"),
    ]);
    let text = doc(
        &nodes,
        &[
            link("prepass.depth", "forward.depth"),
            link("forward.color", "tonemap.color"),
        ],
    );
    let (plan, _) = planned(&text);
    // Once prepass is placed, `low` (10) leads; the two order-50 nodes follow in document order;
    // `forward` (60) comes before `high` (90).
    assert_eq!(
        ids(&plan),
        [
            "prepass", "low", "midB", "midA", "forward", "high", "tonemap", "out"
        ]
    );
    let order: Vec<u32> = plan.nodes.iter().map(|node| node.order_index).collect();
    assert_eq!(order, [0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn contributors_run_right_after_forward_in_their_default_order() {
    let mut nodes = base();
    nodes.extend([
        node("hi", "fx.hi"),
        node("lo", "fx.lo"),
        node("post", "fx.post"),
    ]);
    let text = doc(&nodes, &[link("forward.color", "tonemap.color")]);
    let (plan, _) = planned(&text);
    // `lo` and `hi` are executed with forward and placed before `post` (100), which is ready
    // from the start, and before `tonemap`.
    assert_eq!(
        ids(&plan),
        ["prepass", "forward", "lo", "hi", "post", "tonemap", "out"]
    );
}

#[test]
fn resource_ids_follow_first_use_and_span_every_consumer() {
    let mut nodes = base();
    nodes.extend([node("shade", "fx.shade"), node("post", "fx.post")]);
    let text = doc(
        &nodes,
        &[
            link("prepass.depth", "post.depth"),
            link("forward.color", "tonemap.color"),
        ],
    );
    let (plan, _) = planned(&text);
    assert_eq!(
        ids(&plan),
        ["prepass", "shade", "forward", "post", "tonemap", "out"]
    );
    // forward.depth reads prepass.depth by default; the resource ends at post (order 3).
    let resources: Vec<(u32, &str, &str, u32, u32, bool)> = plan
        .resources
        .iter()
        .map(|res| {
            (
                res.id,
                res.producer.node.as_str(),
                res.producer.port.as_str(),
                res.first_use,
                res.last_use,
                res.history,
            )
        })
        .collect();
    assert_eq!(
        resources,
        [
            (1, "prepass", "depth", 0, 3, false),
            (2, "prepass", "normal", 0, 1, false),
            (3, "forward", "color", 2, 4, false),
            (4, "tonemap", "color", 4, 5, false),
        ]
    );
    assert_eq!(
        node_of(&plan, "forward").inputs[0].source,
        InputSource::Default {
            node: "prepass".to_string(),
            port: "depth".to_string(),
        }
    );
    assert_eq!(plan.output, at("tonemap", "color"));
}

#[test]
fn a_history_read_extends_its_resource_to_the_end_of_the_frame() {
    let mut nodes = base();
    nodes.extend([node("accum", "fx.accum"), node("reader", "fx.reader")]);
    let text = doc(
        &nodes,
        &[
            history("accum.color", "accum.history"),
            link("accum.color", "reader.color"),
            link("forward.color", "tonemap.color"),
        ],
    );
    let (plan, _) = planned(&text);
    assert_eq!(
        ids(&plan),
        ["prepass", "accum", "forward", "reader", "tonemap", "out"]
    );
    let color = plan
        .resources
        .iter()
        .find(|res| res.producer == at("accum", "color"))
        .expect("accum.color is read");
    // The reader at order 3 would end it there; the history read holds it to the frame's end.
    assert_eq!(
        (color.first_use, color.last_use, color.history),
        (1, u32::MAX, true)
    );
    assert_eq!(
        plan.histories,
        [PlanHistory {
            resource: color.id,
            consumer: at("accum", "history"),
        }]
    );
    assert_eq!(
        node_of(&plan, "accum").inputs[0].source,
        InputSource::Link {
            node: "accum".to_string(),
            port: "color".to_string(),
            history: true,
        }
    );
    assert!(
        plan.links
            .iter()
            .any(|found| found.history && found.to == at("accum", "history"))
    );
}

#[test]
fn a_written_engine_option_that_differs_from_the_engine_is_refused() {
    let nodes = [
        node("prepass", "pass.prepass"),
        r#"{"id":"forward","type":"pass.forward","params":{"msaa":1}}"#.to_string(),
        node("tonemap", "pass.tonemap"),
        node("out", "output.display"),
    ];
    let text = doc(&nodes, &[link("forward.color", "tonemap.color")]);
    let issues = plan_at(&text, 4).expect_err("msaa 1 differs from the engine's 4");
    let refused = one(&issues, "graph.engine_option");
    assert_eq!(
        (
            issues.len(),
            refused.severity,
            refused.stage,
            refused.node.as_deref(),
            refused.param.as_deref(),
            refused.fix.as_str(),
        ),
        (
            1,
            Severity::Error,
            Stage::Plan,
            Some("forward"),
            Some("msaa"),
            "create the engine with msaa: 1",
        )
    );
}

#[test]
fn an_absent_engine_param_takes_the_engine_value_and_params_follow_the_catalog() {
    let absent = doc(&base(), &[link("forward.color", "tonemap.color")]);
    let (plan, _) = plan_at(&absent, 1).expect("the graph plans with the engine's msaa 1");
    assert_eq!(
        node_of(&plan, "forward").params,
        [("msaa".to_string(), ParamValue::Int(1))]
    );
    let mut nodes = base();
    nodes[1] = r#"{"id":"forward","type":"pass.forward","params":{"msaa":4}}"#.to_string();
    let written = doc(&nodes, &[link("forward.color", "tonemap.color")]);
    let (plan, _) = plan_at(&written, 4).expect("the graph plans when the written msaa matches");
    assert_eq!(
        node_of(&plan, "forward").params,
        [("msaa".to_string(), ParamValue::Int(4))]
    );
}

#[test]
fn a_validation_error_returns_its_issues_and_no_plan() {
    let text = doc(
        &[
            node("prepass", "pass.prepass"),
            node("out", "output.display"),
        ],
        &[],
    );
    let issues = plan_at(&text, 4).expect_err("forward and tonemap are missing");
    let codes: Vec<&str> = issues.iter().map(|issue| issue.code).collect();
    assert_eq!(codes, ["graph.required_node", "graph.required_node"]);
}

#[test]
fn two_plans_of_one_text_are_equal() {
    let mut nodes = base();
    nodes.extend([node("shade", "fx.shade"), node("post", "fx.post")]);
    let text = doc(
        &nodes,
        &[
            link("prepass.depth", "post.depth"),
            link("forward.color", "tonemap.color"),
        ],
    );
    assert_eq!(planned(&text), planned(&text));
}

#[test]
fn a_history_read_of_a_node_left_out_reads_nothing() {
    // `stale` is reached only through a history link, which the walk does not follow.
    let mut nodes = base();
    nodes.extend([node("stale", "fx.stale"), node("accum", "fx.accum")]);
    let text = doc(
        &nodes,
        &[
            history("stale.color", "accum.history"),
            link("forward.color", "tonemap.color"),
        ],
    );
    let (plan, warnings) = planned(&text);
    assert_eq!(
        one(&warnings, "graph.unreachable").node.as_deref(),
        Some("stale")
    );
    assert_eq!(
        ids(&plan),
        ["prepass", "accum", "forward", "tonemap", "out"]
    );
    assert_eq!(
        node_of(&plan, "accum").inputs[0].source,
        InputSource::Absent
    );
    assert!(plan.histories.is_empty());
    assert!(plan.links.iter().all(|link| !link.history));
}

#[test]
fn plan_from_json_refuses_text_that_is_not_json() {
    let issues = plan_from_json("{").expect_err("the text is not JSON");
    assert_eq!(one(&issues, "graph.json").stage, Stage::Parse);
}
