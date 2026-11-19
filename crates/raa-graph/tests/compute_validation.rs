//! Declaration and WGSL validation for `compute.wgsl` nodes (S59).

use raa_graph::catalog::{Catalog, GraphFormat, ParamValue, PortType};
use raa_graph::error::{Issue, Severity, Stage};
use raa_graph::format::{ComputeDecl, ComputePortDecl, GraphDoc, NodeDoc, parse_graph};
use raa_graph::registry;
use raa_graph::validate::{GraphLimits, ValidateCtx, validate};
use raa_loader::json::JsonValue;

const GOOD: &str = "fn rg_main(p: vec2<u32>) {\n  textureStore(rg_out_dst, p, vec4<f32>(1.0));\n}";

fn base_doc() -> GraphDoc {
    let json = registry::STARTERS
        .iter()
        .find(|&&(_, stem, _)| stem == "simple-forward")
        .map(|&(_, _, json)| json)
        .expect("Simple forward starter exists");
    parse_graph(json).expect("Simple forward starter parses")
}

fn node(wgsl: &str) -> NodeDoc {
    NodeDoc {
        id: "calc".to_string(),
        type_id: "compute.wgsl".to_string(),
        version: None,
        params: Vec::new(),
        compute: Some(ComputeDecl {
            inputs: Vec::new(),
            outputs: vec![output(
                "dst",
                PortType::Image2d,
                Some(GraphFormat::Rgba16Float),
            )],
            math: JsonValue::Null,
            wgsl: wgsl.to_string(),
        }),
    }
}

fn output(key: &str, ty: PortType, format: Option<GraphFormat>) -> ComputePortDecl {
    ComputePortDecl {
        key: key.to_string(),
        ty,
        format,
        size: None,
    }
}

fn input(key: &str, ty: PortType, format: Option<GraphFormat>) -> ComputePortDecl {
    output(key, ty, format)
}

fn doc_with(node: NodeDoc) -> GraphDoc {
    let mut doc = base_doc();
    doc.nodes.push(node);
    doc
}

fn calc_issues(doc: &GraphDoc) -> Vec<Issue> {
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
    validate(doc, &ctx)
        .into_iter()
        .filter(|issue| issue.node.as_deref() == Some("calc"))
        .collect()
}

fn code_issues(doc: &GraphDoc, code: &str) -> Vec<Issue> {
    calc_issues(doc)
        .into_iter()
        .filter(|issue| issue.code == code)
        .collect()
}

fn assert_structure_error(issue: &Issue) {
    assert_eq!(issue.severity, Severity::Error);
    assert_eq!(issue.stage, Stage::Structure);
    assert_eq!(issue.node.as_deref(), Some("calc"));
    assert_eq!(issue.link, None);
}

#[test]
fn valid_compute_body_has_no_issue_on_calc() {
    assert!(calc_issues(&doc_with(node(GOOD))).is_empty());
}

#[test]
fn composer_syntax_error_uses_the_user_line_and_fix() {
    let body = "fn rg_main(p: vec2<u32>) {\n  let a = 1.0;\n  let b = ;\n  textureStore(rg_out_dst, p, vec4<f32>(a));\n}";
    let errors = code_issues(&doc_with(node(body)), "graph.wgsl");
    let issue = errors
        .iter()
        .find(|issue| issue.message.starts_with("WGSL error at line 3, column"))
        .expect("line 3 is reported as a user WGSL line");
    assert_structure_error(issue);
    assert_eq!(issue.port, None);
    assert_eq!(issue.param, None);
    assert_eq!(issue.fix, "Correct line 3 of the WGSL of node `calc`.");
}

#[test]
fn user_wgsl_rules_reject_bindings_reserved_names_and_missing_main() {
    let group = "fn rg_main(p: vec2<u32>) {\n@group(0)\n}";
    let issues = code_issues(&doc_with(node(group)), "graph.wgsl");
    let issue = issues
        .iter()
        .find(|issue| issue.message.contains("@group is not allowed"))
        .expect("user binding is refused");
    assert_structure_error(issue);
    assert!(issue.message.starts_with("WGSL error at line 2, column 1:"));
    assert_eq!(issue.fix, "Correct line 2 of the WGSL of node `calc`.");

    let helper = doc_with(node("fn rg_helper() {}"));
    assert!(
        code_issues(&helper, "graph.wgsl")
            .iter()
            .any(|issue| issue.message.contains("rg_helper is reserved"))
    );

    let missing = doc_with(node("fn helper() {}"));
    assert!(
        code_issues(&missing, "graph.wgsl")
            .iter()
            .any(|issue| issue.message == "WGSL error: the WGSL does not define fn rg_main")
    );
}

#[test]
fn declaration_rules_report_ports_and_parameters() {
    let mut bad_key = node(GOOD);
    bad_key.compute.as_mut().unwrap().outputs[0].key = "Dst".to_string();
    let issue = code_issues(&doc_with(bad_key), "graph.port_unknown")
        .into_iter()
        .next()
        .expect("invalid key is refused");
    assert_structure_error(&issue);
    assert_eq!(issue.port.as_deref(), Some("Dst"));
    assert!(issue.fix.contains("Dst"));
    assert!(issue.fix.contains("[a-z][A-Za-z0-9]"));

    let mut bad_format = node(GOOD);
    bad_format.compute.as_mut().unwrap().outputs[0].format = Some(GraphFormat::R8Unorm);
    let issue = code_issues(&doc_with(bad_format), "graph.link_type")
        .into_iter()
        .next()
        .expect("unsupported storage format is refused");
    assert_structure_error(&issue);
    assert_eq!(issue.port.as_deref(), Some("dst"));
    assert!(issue.fix.contains("rgba16float"));

    let mut no_outputs = node(GOOD);
    no_outputs.compute.as_mut().unwrap().outputs.clear();
    let issue = code_issues(&doc_with(no_outputs), "graph.port_unknown")
        .into_iter()
        .next()
        .expect("an output is required");
    assert_structure_error(&issue);
    assert_eq!(issue.port, None);

    let mut integer_input = node(GOOD);
    integer_input.compute.as_mut().unwrap().inputs.push(input(
        "src",
        PortType::Image2d,
        Some(GraphFormat::R32Uint),
    ));
    let issue = code_issues(&doc_with(integer_input), "graph.link_type")
        .into_iter()
        .next()
        .expect("integer image input is refused");
    assert_structure_error(&issue);
    assert_eq!(issue.port.as_deref(), Some("src"));

    let mut bad_param = node(GOOD);
    bad_param
        .params
        .push(("bad-Key".to_string(), JsonValue::String("x".to_string())));
    let issue = code_issues(&doc_with(bad_param), "graph.param_kind")
        .into_iter()
        .next()
        .expect("invalid compute parameter is refused");
    assert_structure_error(&issue);
    assert_eq!(issue.param.as_deref(), Some("bad-Key"));
    assert!(issue.fix.contains("bad-Key"));
}
