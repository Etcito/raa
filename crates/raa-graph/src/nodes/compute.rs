//! Family `compute`: the `compute.wgsl` node whose WGSL body comes from the math graph or is
//! written by hand (S59). This file declares the catalog type and validates the declaration and
//! user WGSL; the module generator lives in `compute/codegen.rs`.

use crate::catalog::{Category, GraphFormat, NodeFamily, NodeTypeSpec, ParamValue, PortType};
use crate::error::{Issue, Severity, Stage};
use crate::format::{ComputeDecl, NodeDoc};
use raa_loader::json::JsonValue;

#[allow(dead_code)] // the compute pass and validation cards call these; the tests compile the same file
#[path = "compute/codegen.rs"]
pub(crate) mod codegen;

/// The `compute` family, discovered by build.rs.
pub fn family() -> Box<dyn NodeFamily> {
    Box::new(ComputeFamily)
}

struct ComputeFamily;

fn identity_upgrade(_from: u32, _params: &mut Vec<(String, ParamValue)>) -> Result<(), String> {
    Ok(())
}

impl NodeFamily for ComputeFamily {
    fn name(&self) -> &'static str {
        "compute"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        vec![NodeTypeSpec {
            id: "compute.wgsl",
            version: 1,
            category: Category::Compute,
            label: "Compute (WGSL)",
            description: "A compute pass whose WGSL body comes from the math graph or is written by hand",
            pass: None,
            contributor: false,
            default_order: 46,
            multi_instance: true,
            side_effect: false,
            required_in_graph: false,
            inputs: Vec::new(),
            outputs: Vec::new(),
            params: Vec::new(),
            budgets: Vec::new(),
            upgrade: identity_upgrade,
            family: "compute",
        }]
    }

    fn validate_node(
        &self,
        node: &NodeDoc,
        _spec: &NodeTypeSpec,
        _ctx: &crate::validate::ValidateCtx<'_>,
        issues: &mut Vec<Issue>,
    ) {
        if node.type_id != "compute.wgsl" {
            return;
        }
        let Some(compute) = &node.compute else {
            issues.push(issue(
                node,
                "graph.wgsl",
                None,
                None,
                "WGSL error: the node has no compute declaration".to_string(),
                format!("Correct the WGSL of node `{}`.", node.id),
            ));
            return;
        };

        let start = issues.len();
        validate_ports(node, compute, issues);
        validate_params(node, issues);
        validate_user_wgsl(node, &compute.wgsl, issues);
        if issues.len() != start {
            return;
        }

        let (module, offset) = match codegen::generate_module(node) {
            Ok(generated) => generated,
            Err(error) => {
                // Keep the generator's typed refusal if a future declaration rule is added
                // there first.
                issues.push(error);
                return;
            }
        };
        if let Err(error) = raa_shader::Composer::validate(&module) {
            let message = &error.message;
            let (line, column, detail) = parse_location(message);
            let user_lines = compute.wgsl.lines().count();
            let (message, fix) = match (line, column) {
                (Some(module_line), Some(column)) => {
                    let user_line = module_line.checked_sub(offset);
                    if let Some(user_line) =
                        user_line.filter(|line| *line > 0 && *line as usize <= user_lines)
                    {
                        (
                            format!("WGSL error at line {user_line}, column {column}: {detail}"),
                            line_fix(node, user_line),
                        )
                    } else {
                        (
                            format!(
                                "WGSL error in the generated module at line {module_line}, column {column}: {detail}"
                            ),
                            format!("Correct the WGSL of node `{}`.", node.id),
                        )
                    }
                }
                _ => (
                    format!("WGSL error: {message}"),
                    format!("Correct the WGSL of node `{}`.", node.id),
                ),
            };
            issues.push(issue(node, "graph.wgsl", None, None, message, fix));
        }
    }
}

fn issue(
    node: &NodeDoc,
    code: &'static str,
    port: Option<String>,
    param: Option<String>,
    message: String,
    fix: String,
) -> Issue {
    Issue {
        code,
        severity: Severity::Error,
        stage: Stage::Structure,
        node: Some(node.id.clone()),
        port,
        param,
        link: None,
        message,
        fix,
    }
}

fn validate_ports(node: &NodeDoc, compute: &ComputeDecl, issues: &mut Vec<Issue>) {
    let mut keys = Vec::new();
    for port in compute.inputs.iter().chain(&compute.outputs) {
        if !valid_key(&port.key) || keys.iter().any(|key| key == &port.key) {
            issues.push(issue(
                node,
                "graph.port_unknown",
                Some(port.key.clone()),
                None,
                format!("compute port key `{}` is invalid or repeated", port.key),
                format!(
                    "Use a unique port key `{}` matching /^[a-z][A-Za-z0-9]{{0,31}}$/.",
                    port.key
                ),
            ));
        }
        keys.push(port.key.clone());
    }
    if compute.outputs.is_empty() {
        issues.push(issue(
            node,
            "graph.port_unknown",
            None,
            None,
            "compute nodes must declare at least one output".to_string(),
            "Declare at least one output with a unique key matching /^[a-z][A-Za-z0-9]{0,31}$/ and an image2d storage format.".to_string(),
        ));
    }
    for output in &compute.outputs {
        if output.ty != PortType::Image2d
            || output
                .format
                .as_ref()
                .and_then(codegen::storage_format_name)
                .is_none()
        {
            issues.push(issue(
                node,
                "graph.link_type",
                Some(output.key.clone()),
                None,
                format!(
                    "output `{}` must be image2d with a supported storage format",
                    output.key
                ),
                format!(
                    "Declare `{}` as image2d with one of the supported storage formats: {}.",
                    output.key,
                    storage_format_list()
                ),
            ));
        }
    }
    for input in &compute.inputs {
        if input.ty == PortType::Image2d
            && matches!(
                input.format,
                Some(GraphFormat::R32Uint | GraphFormat::Rg32Uint | GraphFormat::Rgba32Uint)
            )
        {
            issues.push(issue(
                node,
                "graph.link_type",
                Some(input.key.clone()),
                None,
                format!(
                    "image2d input `{}` binds as texture_2d<f32> and cannot use integer format {}",
                    input.key,
                    input.format.as_ref().map_or("?", GraphFormat::name)
                ),
                format!(
                    "Use a non-integer image2d format for `{}`, such as rgba16float.",
                    input.key
                ),
            ));
        }
    }
}

fn validate_params(node: &NodeDoc, issues: &mut Vec<Issue>) {
    for (key, value) in &node.params {
        if !valid_key(key) || codegen::infer_param(value).is_none() {
            issues.push(issue(
                node,
                "graph.param_kind",
                None,
                Some(key.clone()),
                format!("parameter `{key}` must have a valid key and be a number, bool or array of 2..4 numbers"),
                format!("Use parameter key `{key}` matching /^[a-z][A-Za-z0-9]{{0,31}}$/ and a number, bool or array of 2, 3 or 4 numbers."),
            ));
        }
    }
}

fn valid_key(key: &str) -> bool {
    let mut chars = key.chars();
    matches!(chars.next(), Some(first) if first.is_ascii_lowercase())
        && chars.count() <= 31
        && key.chars().all(|ch| ch.is_ascii_alphanumeric())
}

fn storage_format_list() -> &'static str {
    "rgba8unorm, rgba16float, r32float, rg32float, rgba32float, r32uint, rg32uint, rgba32uint"
}

fn line_fix(node: &NodeDoc, line: u32) -> String {
    format!("Correct line {line} of the WGSL of node `{}`.", node.id)
}

fn wgsl_issue(node: &NodeDoc, message: String, line: Option<u32>) -> Issue {
    let fix = line.map_or_else(
        || format!("Correct the WGSL of node `{}`.", node.id),
        |line| line_fix(node, line),
    );
    issue(node, "graph.wgsl", None, None, message, fix)
}

fn validate_user_wgsl(node: &NodeDoc, source: &str, issues: &mut Vec<Issue>) {
    let clean = remove_line_comments(source);
    let tokens = tokens(&clean);
    if !has_main_signature(&clean) {
        issues.push(wgsl_issue(
            node,
            "WGSL error: the WGSL does not define fn rg_main".to_string(),
            None,
        ));
    }
    for forbidden in [
        "@group",
        "@binding",
        "@compute",
        "@vertex",
        "@fragment",
        "@workgroup_size",
        "enable ",
        "override ",
    ] {
        for (index, text) in clean.lines().enumerate() {
            if !text.contains(forbidden) {
                continue;
            }
            let line = index as u32 + 1;
            issues.push(wgsl_issue(
                node,
                format!(
                    "WGSL error at line {line}, column 1: {} is not allowed in a compute node",
                    forbidden.trim()
                ),
                Some(line),
            ));
        }
    }
    for pair in tokens.windows(2) {
        if matches!(
            pair[0].text.as_str(),
            "fn" | "let" | "const" | "struct" | "alias"
        ) && pair[1].text.starts_with("rg_")
            && pair[1].text != "rg_main"
        {
            issues.push(wgsl_issue(
                node,
                format!("WGSL error at line {}, column 1: {} is reserved: identifiers starting rg_ belong to the generated module", pair[1].line, pair[1].text),
                Some(pair[1].line),
            ));
        }
    }
    for index in 0..tokens.len() {
        if tokens[index].text == "var" {
            let mut next = index + 1;
            if tokens.get(next).is_some_and(|token| token.text == "<") {
                while next < tokens.len() && tokens[next].text != ">" {
                    next += 1;
                }
                next += usize::from(next < tokens.len());
            }
            if let Some(name) = tokens
                .get(next)
                .filter(|token| token.text.starts_with("rg_") && token.text != "rg_main")
            {
                issues.push(wgsl_issue(
                    node,
                    format!("WGSL error at line {}, column 1: {} is reserved: identifiers starting rg_ belong to the generated module", name.line, name.text),
                    Some(name.line),
                ));
            }
        }
    }
}

#[derive(Clone, Debug)]
struct Token {
    text: String,
    line: u32,
}

fn remove_line_comments(source: &str) -> String {
    source
        .lines()
        .map(|line| line.split_once("//").map_or(line, |(code, _)| code))
        .collect::<Vec<_>>()
        .join("\n")
}

fn tokens(source: &str) -> Vec<Token> {
    let mut result = Vec::new();
    let mut line = 1;
    let mut current = String::new();
    let mut token_line = line;
    for ch in source.chars() {
        if ch == '\n' {
            if !current.is_empty() {
                result.push(Token {
                    text: std::mem::take(&mut current),
                    line: token_line,
                });
            }
            line += 1;
        } else if ch.is_ascii_alphanumeric() || ch == '_' {
            if current.is_empty() {
                token_line = line;
            }
            current.push(ch);
        } else {
            if !current.is_empty() {
                result.push(Token {
                    text: std::mem::take(&mut current),
                    line: token_line,
                });
            }
            if !ch.is_whitespace() {
                result.push(Token {
                    text: ch.to_string(),
                    line,
                });
            }
        }
    }
    if !current.is_empty() {
        result.push(Token {
            text: current,
            line: token_line,
        });
    }
    result
}

fn has_main_signature(source: &str) -> bool {
    const SIGNATURE: [&str; 11] = [
        "fn", "rg_main", "(", "p", ":", "vec2", "<", "u32", ">", ")", "{",
    ];
    let tokens = tokens(source);
    tokens
        .windows(SIGNATURE.len())
        .any(|window| window.iter().map(|token| token.text.as_str()).eq(SIGNATURE))
}

fn parse_location(message: &str) -> (Option<u32>, Option<u32>, &str) {
    let Some((location, detail)) = message.split_once(": ") else {
        return (None, None, message);
    };
    let Some((line, column)) = location.split_once(':') else {
        return (None, None, message);
    };
    match (line.parse().ok(), column.parse().ok()) {
        (Some(line), Some(column)) => (Some(line), Some(column), detail),
        _ => (None, None, message),
    }
}
