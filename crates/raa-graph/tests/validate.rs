//! Structure-stage tests of the render graph validator (S52) and of `canonical_doc` (S50).

use raa_graph::catalog::{
    Catalog, Category, GraphFormat, NodeTypeSpec, ParamBinding, ParamKind, ParamSpec, ParamValue,
    PortSpec, PortType, Rebuild, SizeSpec,
};
use raa_graph::error::{Issue, Severity, Stage};
use raa_graph::format::{GraphDoc, parse_graph};
use raa_graph::validate::{GraphLimits, ValidateCtx, canonical_doc, validate};

/// A port of type `ty` with no format, history or default source.
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
        generic: false,
        history: false,
        slot: None,
    }
}

/// A parameter of `kind` with no bounds, options or unit.
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
        source: "crates/raa-graph/tests/validate.rs",
    }
}

/// A node type with no params or budgets; callers set the rest with struct update syntax.
fn node_type(
    id: &'static str,
    version: u32,
    inputs: Vec<PortSpec>,
    outputs: Vec<PortSpec>,
) -> NodeTypeSpec {
    NodeTypeSpec {
        id,
        version,
        category: Category::Lighting,
        label: id,
        description: id,
        pass: None,
        contributor: false,
        default_order: 0,
        multi_instance: false,
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

/// An upgrade with nothing to convert.
fn keep(_from: u32, _params: &mut Vec<(String, ParamValue)>) -> Result<(), String> {
    Ok(())
}

/// `fx.blur` v2 converts v1 radii (pixels) to metres by halving them. A negative v1 radius
/// has no conversion, so the step fails.
#[allow(
    clippy::ptr_arg,
    reason = "NodeTypeSpec.upgrade is a fn pointer taking &mut Vec, so the signature must match"
)]
fn blur_upgrade(from: u32, params: &mut Vec<(String, ParamValue)>) -> Result<(), String> {
    if from != 1 {
        return Err(format!("no upgrade from v{from}"));
    }
    for (key, value) in params.iter_mut() {
        if key.as_str() != "radius" {
            continue;
        }
        if let ParamValue::Number(radius) = value {
            if *radius < 0.0 {
                return Err(format!("radius {radius} cannot be converted"));
            }
            *radius *= 0.5;
        }
    }
    Ok(())
}

/// The test catalog: the four stand-ins the graph needs, an upgradable v2 blur with three
/// parameter kinds, a multi-input forward node and a buffer producer and consumer.
fn catalog() -> Catalog {
    let prepass = NodeTypeSpec {
        pass: Some("prepass"),
        required_in_graph: true,
        outputs: vec![
            port("depth", PortType::Depth2d),
            PortSpec {
                format: Some(GraphFormat::Rgba16Float),
                ..port("normal", PortType::Image2d)
            },
            PortSpec {
                format: Some(GraphFormat::Rgba8Unorm),
                ..port("material", PortType::Image2d)
            },
        ],
        ..node_type("pass.prepass", 1, Vec::new(), Vec::new())
    };
    let forward = NodeTypeSpec {
        pass: Some("forward"),
        required_in_graph: true,
        outputs: vec![PortSpec {
            format: Some(GraphFormat::Rgba16Float),
            ..port("color", PortType::Image2d)
        }],
        ..node_type(
            "pass.forward",
            1,
            vec![
                port("depth", PortType::Depth2d),
                port("lightTiles", PortType::Image2dArray),
                port("ao", PortType::Image2d),
            ],
            Vec::new(),
        )
    };
    let tonemap = NodeTypeSpec {
        pass: Some("tonemap"),
        required_in_graph: true,
        ..node_type(
            "pass.tonemap",
            1,
            vec![PortSpec {
                format: Some(GraphFormat::Rgba16Float),
                ..port("color", PortType::Image2d)
            }],
            vec![port("color", PortType::Image2d)],
        )
    };
    let output = node_type(
        "output.display",
        1,
        vec![port("color", PortType::Image2d)],
        Vec::new(),
    );
    let blur = NodeTypeSpec {
        multi_instance: true,
        params: vec![
            ParamSpec {
                min: Some(0.0),
                max: Some(10.0),
                ..param("radius", ParamKind::Number, ParamValue::Number(1.0))
            },
            ParamSpec {
                options: &["fast", "slow"],
                ..param(
                    "mode",
                    ParamKind::Enum,
                    ParamValue::Enum("fast".to_string()),
                )
            },
            ParamSpec {
                options: &["1", "4"],
                ..param("samples", ParamKind::Int, ParamValue::Int(4))
            },
        ],
        upgrade: blur_upgrade,
        ..node_type(
            "fx.blur",
            2,
            vec![PortSpec {
                format: Some(GraphFormat::Rgba16Float),
                ..port("color", PortType::Image2d)
            }],
            vec![port("color", PortType::Image2d)],
        )
    };
    let cull = node_type(
        "fx.cull",
        1,
        Vec::new(),
        vec![port("list", PortType::Buffer)],
    );
    let gather = node_type(
        "fx.gather",
        1,
        vec![PortSpec {
            history: true,
            ..port("list", PortType::Buffer)
        }],
        Vec::new(),
    );
    Catalog {
        nodes: vec![prepass, forward, tonemap, output, blur, cull, gather],
        starters: Vec::new(),
        presets: Vec::new(),
        frozen: Vec::new(),
    }
}

/// Validates `text` against the test catalog with the core limits and one viewport.
fn run(text: &str) -> Vec<Issue> {
    let catalog = catalog();
    let installed = |_: &str| true;
    let ctx = ValidateCtx {
        catalog: &catalog,
        limits: GraphLimits::CORE,
        installed: &installed,
        viewports: 1,
        engine_options: &[],
    };
    validate(&parse_graph(text).expect("the fixture parses"), &ctx)
}

/// Every issue with `code`, in the order `validate` returned them.
fn of<'a>(issues: &'a [Issue], code: &str) -> Vec<&'a Issue> {
    issues.iter().filter(|issue| issue.code == code).collect()
}

/// The only issue with `code`; the test fails when there are none or several.
fn one<'a>(issues: &'a [Issue], code: &str) -> &'a Issue {
    let found = of(issues, code);
    assert_eq!(found.len(), 1, "expected one `{code}` in {issues:#?}");
    found[0]
}

/// A `raa.render-graph` document holding the given node and link objects.
fn doc(nodes: &[&str], links: &[&str]) -> String {
    format!(
        r#"{{"format":"raa.render-graph","version":1,"name":"test","nodes":[{}],"links":[{}]}}"#,
        nodes.join(","),
        links.join(",")
    )
}

const PREPASS: &str = r#"{"id":"prepass","type":"pass.prepass"}"#;
const FORWARD: &str = r#"{"id":"forward","type":"pass.forward"}"#;
const TONEMAP: &str = r#"{"id":"tonemap","type":"pass.tonemap"}"#;
const OUT: &str = r#"{"id":"out","type":"output.display"}"#;
const BLUR: &str = r#"{"id":"blur","type":"fx.blur"}"#;
const DEPTH: &str = r#"{"from":"prepass.depth","to":"forward.depth"}"#;
const COLOR: &str = r#"{"from":"forward.color","to":"tonemap.color"}"#;
const TO_OUT: &str = r#"{"from":"tonemap.color","to":"out.color"}"#;

/// The four required nodes linked the way every graph links them: no issue at all.
fn valid() -> String {
    doc(&[PREPASS, FORWARD, TONEMAP, OUT], &[DEPTH, COLOR, TO_OUT])
}

#[test]
fn a_valid_graph_has_no_structure_issues() {
    let issues = run(&valid());
    assert!(issues.is_empty(), "{issues:#?}");
}

#[test]
fn node_id_refuses_invalid_and_repeated_ids() {
    let text = doc(
        &[
            PREPASS,
            r#"{"id":"2fast","type":"pass.forward"}"#,
            r#"{"id":"prepass","type":"pass.prepass"}"#,
        ],
        &[],
    );
    let issues = run(&text);
    let ids = of(&issues, "graph.node_id");
    let found: Vec<Option<&str>> = ids.iter().map(|issue| issue.node.as_deref()).collect();
    // `sort_issues` ranks a node issue by the first document position of its id, so the
    // repeated `prepass` sorts before `2fast`.
    assert_eq!(found, [Some("prepass"), Some("2fast")]);
    assert!(
        ids[0].message.contains("already used"),
        "{}",
        ids[0].message
    );
    assert!(ids[1].message.contains("2fast"), "{}", ids[1].message);
    assert!(ids.iter().all(|issue| issue.stage == Stage::Structure));
}

#[test]
fn node_type_names_an_unknown_type() {
    let text = doc(
        &[
            PREPASS,
            FORWARD,
            TONEMAP,
            OUT,
            r#"{"id":"ghost","type":"fx.ghost"}"#,
        ],
        &[DEPTH, COLOR, TO_OUT],
    );
    let issues = run(&text);
    let issue = one(&issues, "graph.node_type");
    assert_eq!(issue.node.as_deref(), Some("ghost"));
    assert!(issue.message.contains("fx.ghost"), "{}", issue.message);
    assert_eq!(issue.severity, Severity::Error);
}

#[test]
fn node_version_refuses_a_newer_version_and_a_failed_upgrade() {
    let text = doc(
        &[
            PREPASS,
            FORWARD,
            TONEMAP,
            OUT,
            r#"{"id":"newer","type":"fx.blur","v":3}"#,
            r#"{"id":"old","type":"fx.blur","v":1,"params":{"radius":-1}}"#,
        ],
        &[DEPTH, COLOR, TO_OUT],
    );
    let issues = run(&text);
    let versions = of(&issues, "graph.node_version");
    let found: Vec<Option<&str>> = versions.iter().map(|issue| issue.node.as_deref()).collect();
    assert_eq!(found, [Some("newer"), Some("old")]);
    assert!(
        versions[0].message.contains("v3"),
        "{}",
        versions[0].message
    );
    assert!(
        versions[1].message.contains("-1"),
        "{}",
        versions[1].message
    );
    // A failed upgrade leaves the params unknown, so no range check runs on them.
    assert!(of(&issues, "graph.param_range").is_empty());
}

#[test]
fn node_upgraded_warns_and_range_checks_the_upgraded_values() {
    let text = doc(
        &[
            PREPASS,
            FORWARD,
            TONEMAP,
            OUT,
            r#"{"id":"blur","type":"fx.blur","v":1,"params":{"radius":30}}"#,
        ],
        &[DEPTH, COLOR, TO_OUT],
    );
    let issues = run(&text);
    let upgraded = one(&issues, "graph.node_upgraded");
    assert_eq!(upgraded.severity, Severity::Warning);
    assert_eq!(upgraded.node.as_deref(), Some("blur"));
    // 30 in v1 is 15 in v2, which the [0, 10] range refuses.
    let range = one(&issues, "graph.param_range");
    assert_eq!(range.param.as_deref(), Some("radius"));
    assert!(range.message.contains("15"), "{}", range.message);
}

#[test]
fn param_unknown_names_the_undeclared_key() {
    let text = doc(
        &[
            PREPASS,
            FORWARD,
            TONEMAP,
            OUT,
            r#"{"id":"blur","type":"fx.blur","params":{"sigma":2}}"#,
        ],
        &[DEPTH, COLOR, TO_OUT],
    );
    let issues = run(&text);
    let issue = one(&issues, "graph.param_unknown");
    assert_eq!(issue.node.as_deref(), Some("blur"));
    assert_eq!(issue.param.as_deref(), Some("sigma"));
}

#[test]
fn param_kind_names_the_key_and_the_wrong_value() {
    let text = doc(
        &[
            PREPASS,
            FORWARD,
            TONEMAP,
            OUT,
            r#"{"id":"blur","type":"fx.blur","params":{"radius":"wide"}}"#,
        ],
        &[DEPTH, COLOR, TO_OUT],
    );
    let issues = run(&text);
    let issue = one(&issues, "graph.param_kind");
    assert_eq!(issue.node.as_deref(), Some("blur"));
    assert_eq!(issue.param.as_deref(), Some("radius"));
    assert!(issue.message.contains("wide"), "{}", issue.message);
}

#[test]
fn param_range_refuses_bounds_enum_options_and_int_options() {
    let text = doc(
        &[
            PREPASS,
            FORWARD,
            TONEMAP,
            OUT,
            r#"{"id":"wide","type":"fx.blur","params":{"radius":11}}"#,
            r#"{"id":"mood","type":"fx.blur","params":{"mode":"medium"}}"#,
            r#"{"id":"aa","type":"fx.blur","params":{"samples":2}}"#,
            r#"{"id":"fine","type":"fx.blur","params":{"radius":2,"mode":"slow","samples":4}}"#,
        ],
        &[DEPTH, COLOR, TO_OUT],
    );
    let issues = run(&text);
    let ranges = of(&issues, "graph.param_range");
    let found: Vec<(Option<&str>, Option<&str>)> = ranges
        .iter()
        .map(|issue| (issue.node.as_deref(), issue.param.as_deref()))
        .collect();
    assert_eq!(
        found,
        [
            (Some("wide"), Some("radius")),
            (Some("mood"), Some("mode")),
            (Some("aa"), Some("samples")),
        ]
    );
    assert!(ranges[0].fix.contains("[0, 10]"), "{}", ranges[0].fix);
    assert!(
        ranges[1].message.contains("medium"),
        "{}",
        ranges[1].message
    );
    assert!(ranges[2].fix.contains("one of 1, 4"), "{}", ranges[2].fix);
}

#[test]
fn port_unknown_names_the_node_port_and_link() {
    let text = doc(
        &[PREPASS, FORWARD, TONEMAP, OUT],
        &[
            r#"{"from":"prepass.nothing","to":"forward.depth"}"#,
            r#"{"from":"prepass.depth","to":"ghost.color"}"#,
            r#"{"from":"prepass.depth","to":"forward.Depth"}"#,
            TO_OUT,
        ],
    );
    let issues = run(&text);
    let unknown = of(&issues, "graph.port_unknown");
    let found: Vec<(Option<&str>, Option<&str>, Option<u32>)> = unknown
        .iter()
        .map(|issue| (issue.node.as_deref(), issue.port.as_deref(), issue.link))
        .collect();
    // Sorted by node document index, so the node the document lacks comes last.
    assert_eq!(
        found,
        [
            (Some("prepass"), Some("nothing"), Some(0)),
            (Some("forward"), Some("Depth"), Some(2)),
            (Some("ghost"), Some("color"), Some(1)),
        ]
    );
    assert!(
        unknown[2].message.contains("ghost"),
        "{}",
        unknown[2].message
    );
}

#[test]
fn link_type_is_reported_on_the_target_node() {
    let text = doc(
        &[PREPASS, TONEMAP, BLUR],
        &[
            r#"{"from":"prepass.depth","to":"tonemap.color"}"#,
            r#"{"from":"prepass.material","to":"blur.color"}"#,
        ],
    );
    let issues = run(&text);
    let types = of(&issues, "graph.link_type");
    let found: Vec<(Option<&str>, Option<&str>, Option<u32>)> = types
        .iter()
        .map(|issue| (issue.node.as_deref(), issue.port.as_deref(), issue.link))
        .collect();
    assert_eq!(
        found,
        [
            (Some("tonemap"), Some("color"), Some(0)),
            (Some("blur"), Some("color"), Some(1)),
        ]
    );
    assert!(types[0].message.contains("depth2d"), "{}", types[0].message);
    assert!(types[0].message.contains("image2d"), "{}", types[0].message);
    assert!(
        types[1].message.contains("rgba8unorm"),
        "{}",
        types[1].message
    );
    assert!(
        types[1].message.contains("rgba16float"),
        "{}",
        types[1].message
    );
}

#[test]
fn input_taken_refuses_a_second_link_into_one_input() {
    let text = doc(
        &[PREPASS, FORWARD, TONEMAP, OUT],
        &[DEPTH, DEPTH, COLOR, TO_OUT],
    );
    let issues = run(&text);
    let issue = one(&issues, "graph.input_taken");
    assert_eq!(
        (issue.node.as_deref(), issue.port.as_deref(), issue.link),
        (Some("forward"), Some("depth"), Some(1))
    );
    assert!(issue.message.contains("link 0"), "{}", issue.message);
}

#[test]
fn history_refuses_a_port_without_history_and_a_buffer_source() {
    let text = doc(
        &[
            PREPASS,
            FORWARD,
            TONEMAP,
            OUT,
            r#"{"id":"cull","type":"fx.cull"}"#,
            r#"{"id":"gather","type":"fx.gather"}"#,
        ],
        &[
            r#"{"from":"prepass.depth","to":"forward.depth","history":true}"#,
            r#"{"from":"cull.list","to":"gather.list","history":true}"#,
            COLOR,
            TO_OUT,
        ],
    );
    let issues = run(&text);
    let history = of(&issues, "graph.history");
    let found: Vec<(Option<&str>, Option<u32>)> = history
        .iter()
        .map(|issue| (issue.node.as_deref(), issue.link))
        .collect();
    // A history link names only its link, so these issues carry no node.
    assert_eq!(found, [(None, Some(0)), (None, Some(1))]);
    assert!(
        history[0].message.contains("forward.depth"),
        "{}",
        history[0].message
    );
    assert!(
        history[1].message.contains("cull.list"),
        "{}",
        history[1].message
    );
}

#[test]
fn output_refuses_zero_or_several_display_nodes() {
    let none = run(&doc(&[PREPASS, FORWARD, TONEMAP], &[DEPTH, COLOR]));
    let issue = one(&none, "graph.output");
    assert_eq!(issue.node, None);
    assert!(
        issue.message.contains("0 `output.display`"),
        "{}",
        issue.message
    );

    let second = r#"{"id":"out2","type":"output.display"}"#;
    let several = run(&doc(
        &[PREPASS, FORWARD, TONEMAP, OUT, second],
        &[DEPTH, COLOR, TO_OUT],
    ));
    let issue = one(&several, "graph.output");
    assert!(
        issue.message.contains("2 `output.display`"),
        "{}",
        issue.message
    );
}

#[test]
fn required_node_names_the_missing_type() {
    let issues = run(&doc(&[FORWARD, TONEMAP, OUT], &[COLOR, TO_OUT]));
    let issue = one(&issues, "graph.required_node");
    assert_eq!(issue.node, None);
    assert!(issue.message.contains("pass.prepass"), "{}", issue.message);
}

#[test]
fn single_instance_refuses_a_second_node_of_a_one_per_graph_type() {
    let second = r#"{"id":"prepass2","type":"pass.prepass"}"#;
    let issues = run(&doc(
        &[PREPASS, second, FORWARD, TONEMAP, OUT],
        &[DEPTH, COLOR, TO_OUT],
    ));
    let issue = one(&issues, "graph.single_instance");
    assert_eq!(issue.node.as_deref(), Some("prepass2"));
    assert!(issue.message.contains("pass.prepass"), "{}", issue.message);
}

#[test]
fn canonical_doc_orders_params_by_catalog_and_keeps_the_rest() {
    let text = doc(
        &[
            PREPASS,
            FORWARD,
            TONEMAP,
            OUT,
            r#"{"id":"blur","type":"fx.blur","v":2,"params":{"samples":4,"zeta":true,"mode":"slow","radius":2}}"#,
            r#"{"id":"ghost","type":"fx.ghost","params":{"b":1,"a":2}}"#,
        ],
        &[DEPTH, COLOR, TO_OUT],
    );
    let parsed = parse_graph(&text).expect("the fixture parses");
    let canonical = canonical_doc(&parsed, &catalog());
    let keys = |doc: &GraphDoc, node: usize| -> Vec<String> {
        doc.nodes[node]
            .params
            .iter()
            .map(|(key, _)| key.clone())
            .collect()
    };
    // Declared keys follow the catalog; undeclared keys keep their order after them. An
    // unknown type keeps its document order.
    assert_eq!(keys(&parsed, 4), ["samples", "zeta", "mode", "radius"]);
    assert_eq!(keys(&canonical, 4), ["radius", "mode", "samples", "zeta"]);
    assert_eq!(keys(&canonical, 5), ["b", "a"]);
    // Nothing else changes.
    assert_eq!(canonical.name, parsed.name);
    assert_eq!(canonical.links, parsed.links);
    assert_eq!(canonical.editor, parsed.editor);
    assert_eq!(canonical.nodes.len(), parsed.nodes.len());
    for (node, copy) in parsed.nodes.iter().zip(&canonical.nodes) {
        assert_eq!(
            (&copy.id, &copy.type_id, copy.version),
            (&node.id, &node.type_id, node.version)
        );
        assert_eq!(copy.params.len(), node.params.len());
        for pair in &node.params {
            assert!(copy.params.contains(pair), "missing {pair:?}");
        }
    }
}

#[test]
fn validate_sorts_node_less_issues_first_then_nodes_in_document_order() {
    let text = doc(
        &[
            PREPASS,
            FORWARD,
            TONEMAP,
            OUT,
            r#"{"id":"zeta","type":"fx.blur","params":{"radius":11}}"#,
            r#"{"id":"alpha","type":"fx.ghost"}"#,
        ],
        &[
            r#"{"from":"prepass.depth","to":"forward.depth","history":true}"#,
            COLOR,
            r#"{"from":"prepass.depth","to":"zeta.color"}"#,
            r#"{"from":"prepass.depth","to":"ghost.color"}"#,
            TO_OUT,
        ],
    );
    let issues = run(&text);
    let order: Vec<(&str, Option<&str>, Option<u32>)> = issues
        .iter()
        .map(|issue| (issue.code, issue.node.as_deref(), issue.link))
        .collect();
    // Link-less issues sort first. Known nodes follow in document order (zeta before alpha),
    // each with its link-less issues before its link issues. A node the document lacks
    // sorts after every known node.
    assert_eq!(
        order,
        [
            ("graph.history", None, Some(0)),
            ("graph.param_range", Some("zeta"), None),
            ("graph.link_type", Some("zeta"), Some(2)),
            ("graph.node_type", Some("alpha"), None),
            ("graph.port_unknown", Some("ghost"), Some(3)),
        ]
    );
}
