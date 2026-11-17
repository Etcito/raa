//! The `raa.render-graph` document types (S50).
//!
//! `crates/raa-graph/src/lib.rs`: out of scope for this share: consumers use the module
//! path (`raa_graph::format::parse_graph`). The parse stage sorts with `sort_issues`,
//! whose Handoff doc comment is at `crates/raa-graph/src/error.rs:112 sort_issues`
//! (card 'raa-graph core 2/5: issues, validation context and NodeFamily trait').

use raa_loader::json::JsonValue;

use crate::catalog::{GraphFormat, PortType, SizeSpec};
use crate::error::{Issue, Severity, Stage, sort_issues};

/// The document `format` field.
pub const FORMAT: &str = "raa.render-graph";
/// The document `version` field.
pub const VERSION: u32 = 1;

/// A render graph document.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphDoc {
    pub name: String,
    pub nodes: Vec<NodeDoc>,
    pub links: Vec<LinkDoc>,
    pub editor: Option<JsonValue>,
}

/// One node of a document.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeDoc {
    pub id: String,
    pub type_id: String,
    pub version: Option<u32>,
    pub params: Vec<(String, JsonValue)>,
    pub compute: Option<ComputeDecl>,
}

/// The compute declaration of a `compute.wgsl` node.
#[derive(Clone, Debug, PartialEq)]
pub struct ComputeDecl {
    pub inputs: Vec<ComputePortDecl>,
    pub outputs: Vec<ComputePortDecl>,
    pub math: JsonValue,
    pub wgsl: String,
}

/// One input or output of a compute declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct ComputePortDecl {
    pub key: String,
    pub ty: PortType,
    pub format: Option<GraphFormat>,
    pub size: Option<SizeSpec>,
}

/// One link between two ports.
#[derive(Clone, Debug, PartialEq)]
pub struct LinkDoc {
    pub from: PortRef,
    pub to: PortRef,
    pub history: bool,
}

/// A node id and port key.
#[derive(Clone, Debug, PartialEq)]
pub struct PortRef {
    pub node: String,
    pub port: String,
}

/// A document text is at most 1048576 bytes.
const MAX_TEXT_BYTES: usize = 1_048_576;
/// A document has at most 256 nodes.
const MAX_NODES: usize = 256;
/// A document has at most 1024 links.
const MAX_LINKS: usize = 1024;
/// One node's `wgsl` is at most 65536 bytes.
const MAX_WGSL_BYTES: usize = 65_536;
/// A document `name` has at most 64 characters.
const MAX_NAME_CHARS: usize = 64;
/// The node type that carries a compute declaration.
const COMPUTE_TYPE: &str = "compute.wgsl";

fn issue(code: &'static str, message: impl std::fmt::Display, fix: &'static str) -> Issue {
    Issue {
        code,
        severity: Severity::Error,
        stage: Stage::Parse,
        node: None,
        port: None,
        param: None,
        link: None,
        message: message.to_string(),
        fix: fix.to_string(),
    }
}

fn schema(issues: &mut Vec<Issue>, pointer: &str, what: impl std::fmt::Display) {
    issues.push(issue(
        "graph.schema",
        format!("{pointer}: {what}"),
        "Make the document match the raa.render-graph v1 schema.",
    ));
}

fn too_large(issues: &mut Vec<Issue>, pointer: &str, what: impl std::fmt::Display) {
    issues.push(issue(
        "graph.too_large",
        format!("{pointer}: {what}"),
        "Shorten the document to meet the S50 limits.",
    ));
}

/// The JSON pointer of `key` under `base`, with `~0` and `~1` escaping.
fn pointer(base: &str, key: &str) -> String {
    let mut out = String::with_capacity(base.len() + key.len() + 1);
    out.push_str(base);
    out.push('/');
    for ch in key.chars() {
        match ch {
            '~' => out.push_str("~0"),
            '/' => out.push_str("~1"),
            _ => out.push(ch),
        }
    }
    out
}

fn required_string(value: Option<&JsonValue>, at: &str, issues: &mut Vec<Issue>) -> String {
    match value {
        Some(JsonValue::String(text)) => text.clone(),
        Some(_) => {
            schema(issues, at, "expected a string");
            String::new()
        }
        None => {
            schema(issues, at, "missing required key");
            String::new()
        }
    }
}

fn graph_format_from_name(name: &str) -> Option<GraphFormat> {
    Some(match name {
        "r8unorm" => GraphFormat::R8Unorm,
        "rg8unorm" => GraphFormat::Rg8Unorm,
        "rgba8unorm" => GraphFormat::Rgba8Unorm,
        "rgba8unorm-srgb" => GraphFormat::Rgba8UnormSrgb,
        "bgra8unorm" => GraphFormat::Bgra8Unorm,
        "r16float" => GraphFormat::R16Float,
        "rg16float" => GraphFormat::Rg16Float,
        "rgba16float" => GraphFormat::Rgba16Float,
        "r32float" => GraphFormat::R32Float,
        "rg32float" => GraphFormat::Rg32Float,
        "rgba32float" => GraphFormat::Rgba32Float,
        "r32uint" => GraphFormat::R32Uint,
        "rg32uint" => GraphFormat::Rg32Uint,
        "rgba32uint" => GraphFormat::Rgba32Uint,
        "rgb10a2unorm" => GraphFormat::Rgb10a2Unorm,
        "rg11b10ufloat" => GraphFormat::Rg11b10Ufloat,
        "depth32float" => GraphFormat::Depth32Float,
        "surface" => GraphFormat::Surface,
        _ => return None,
    })
}

/// Parses and checks the parse stage of a `raa.render-graph` document (S50).
///
/// Handoff: `parse_graph` returns a `GraphDoc` whose `NodeDoc.params` keep document
/// order as raw `JsonValue`; card 'raa-graph format: canonical writer and round trip'
/// calls it for its round trip and card 'raa-graph validate 1/2: nodes, params, links
/// and canonical_doc' calls it for its fixture documents.
pub fn parse_graph(text: &str) -> Result<GraphDoc, Vec<Issue>> {
    if text.len() > MAX_TEXT_BYTES {
        return Err(vec![issue(
            "graph.too_large",
            format!(
                "document is {} bytes, over the {MAX_TEXT_BYTES}-byte limit",
                text.len()
            ),
            "Shorten the document text.",
        )]);
    }
    let value = raa_loader::json::parse(text).map_err(|error| {
        vec![issue(
            "graph.json",
            format!("{error}"),
            "Write the document as valid JSON.",
        )]
    })?;
    graph_from_json_value(&value)
}

/// Checks the parse stage of one JSON value and builds a [`GraphDoc`] (S50).
pub fn graph_from_json_value(value: &JsonValue) -> Result<GraphDoc, Vec<Issue>> {
    let mut issues = Vec::new();
    let JsonValue::Object(fields) = value else {
        schema(&mut issues, "", "the document must be a JSON object");
        return Err(issues);
    };
    let (mut format, mut version, mut name) = (None, None, None);
    let (mut nodes, mut links, mut editor) = (None, None, None);
    for (key, value) in fields {
        match key.as_str() {
            "format" => format = Some(value),
            "version" => version = Some(value),
            "name" => name = Some(value),
            "nodes" => nodes = Some(value),
            "links" => links = Some(value),
            "editor" => editor = Some(value.clone()),
            _ => schema(&mut issues, &pointer("", key), "unknown top-level key"),
        }
    }
    match format {
        Some(JsonValue::String(text)) if text == FORMAT => {}
        Some(_) => issues.push(issue(
            "graph.format",
            format!("format must be {FORMAT:?}"),
            "Set format to raa.render-graph.",
        )),
        None => issues.push(issue(
            "graph.format",
            "format is missing",
            "Set format to raa.render-graph.",
        )),
    }
    if !matches!(version, Some(JsonValue::Number(n)) if *n == 1.0) {
        let message = match version {
            Some(JsonValue::Number(n)) => format!("version {n} is not the integer 1"),
            Some(_) => "version must be the integer 1".to_string(),
            None => "version is missing".to_string(),
        };
        issues.push(issue("graph.version", message, "Set version to 1."));
    }
    let name = match name {
        Some(JsonValue::String(text)) => {
            let count = text.chars().count();
            if text.is_empty() {
                schema(&mut issues, &pointer("", "name"), "name is empty");
            } else if count > MAX_NAME_CHARS {
                too_large(
                    &mut issues,
                    &pointer("", "name"),
                    format!("{count} characters over the {MAX_NAME_CHARS}-character limit"),
                );
            }
            text.clone()
        }
        other => required_string(other, &pointer("", "name"), &mut issues),
    };
    let node_docs = parse_list(
        nodes,
        &pointer("", "nodes"),
        MAX_NODES,
        "node",
        &mut issues,
        parse_node,
    );
    let link_docs = parse_list(
        links,
        &pointer("", "links"),
        MAX_LINKS,
        "link",
        &mut issues,
        parse_link,
    );
    if issues.is_empty() {
        Ok(GraphDoc {
            name,
            nodes: node_docs,
            links: link_docs,
            editor,
        })
    } else {
        sort_issues(&mut issues, None);
        Err(issues)
    }
}

fn parse_list<T>(
    value: Option<&JsonValue>,
    at: &str,
    cap: usize,
    name: &str,
    issues: &mut Vec<Issue>,
    parse: impl Fn(&JsonValue, usize, &mut Vec<Issue>) -> Option<T>,
) -> Vec<T> {
    let Some(JsonValue::Array(items)) = value else {
        if value.is_some() {
            schema(issues, at, "expected an array");
        } else {
            schema(issues, at, "missing required key");
        }
        return Vec::new();
    };
    if items.len() > cap {
        too_large(
            issues,
            at,
            format!("{} {name}s over the {cap}-{name} limit", items.len()),
        );
    }
    items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| parse(item, index, issues))
        .collect()
}

fn parse_node(value: &JsonValue, index: usize, issues: &mut Vec<Issue>) -> Option<NodeDoc> {
    let at = format!("/nodes/{index}");
    let JsonValue::Object(fields) = value else {
        schema(issues, &at, "node must be a JSON object");
        return None;
    };
    let (mut id, mut type_id, mut v, mut params) = (None, None, None, None);
    let (mut inputs, mut outputs, mut math, mut wgsl) = (None, None, None, None);
    for (key, value) in fields {
        match key.as_str() {
            "id" => id = Some(value),
            "type" => type_id = Some(value),
            "v" => v = Some(value),
            "params" => params = Some(value),
            "inputs" => inputs = Some(value),
            "outputs" => outputs = Some(value),
            "math" => math = Some(value),
            "wgsl" => wgsl = Some(value),
            _ => schema(issues, &pointer(&at, key), "unknown node key"),
        }
    }
    let id = required_string(id, &pointer(&at, "id"), issues);
    let type_id = required_string(type_id, &pointer(&at, "type"), issues);
    let version = match v {
        Some(JsonValue::Number(n))
            if *n >= 1.0 && n.fract() == 0.0 && *n <= f64::from(u32::MAX) =>
        {
            Some(*n as u32)
        }
        Some(_) => {
            schema(issues, &pointer(&at, "v"), "v must be an integer >= 1");
            None
        }
        None => None,
    };
    let params = match params {
        Some(JsonValue::Object(entries)) => entries.clone(),
        Some(_) => {
            schema(issues, &pointer(&at, "params"), "params must be an object");
            Vec::new()
        }
        None => Vec::new(),
    };
    let compute = if type_id == COMPUTE_TYPE {
        Some(ComputeDecl {
            inputs: parse_ports(inputs, &pointer(&at, "inputs"), issues),
            outputs: parse_ports(outputs, &pointer(&at, "outputs"), issues),
            math: match math {
                Some(value) => value.clone(),
                None => {
                    schema(issues, &pointer(&at, "math"), "missing required key");
                    JsonValue::Null
                }
            },
            wgsl: parse_wgsl(wgsl, &pointer(&at, "wgsl"), issues),
        })
    } else {
        for (key, present) in [
            ("inputs", inputs.is_some()),
            ("outputs", outputs.is_some()),
            ("math", math.is_some()),
            ("wgsl", wgsl.is_some()),
        ] {
            if present {
                schema(
                    issues,
                    &pointer(&at, key),
                    format!("compute key on a node whose type is not {COMPUTE_TYPE}"),
                );
            }
        }
        None
    };
    Some(NodeDoc {
        id,
        type_id,
        version,
        params,
        compute,
    })
}

fn parse_wgsl(value: Option<&JsonValue>, at: &str, issues: &mut Vec<Issue>) -> String {
    match value {
        Some(JsonValue::String(text)) => {
            if text.len() > MAX_WGSL_BYTES {
                too_large(
                    issues,
                    at,
                    format!("{} bytes over the {MAX_WGSL_BYTES}-byte limit", text.len()),
                );
            }
            text.clone()
        }
        Some(_) => {
            schema(issues, at, "wgsl must be a string");
            String::new()
        }
        None => {
            schema(issues, at, "missing required key");
            String::new()
        }
    }
}

fn parse_ports(
    value: Option<&JsonValue>,
    at: &str,
    issues: &mut Vec<Issue>,
) -> Vec<ComputePortDecl> {
    let mut ports = Vec::new();
    let Some(value) = value else {
        schema(issues, at, "missing required compute port list");
        return ports;
    };
    let JsonValue::Array(items) = value else {
        schema(issues, at, "compute ports must be an array");
        return ports;
    };
    for (index, item) in items.iter().enumerate() {
        let at = format!("{at}/{index}");
        let JsonValue::Object(fields) = item else {
            schema(issues, &at, "compute port must be a JSON object");
            continue;
        };
        let (mut key, mut ty, mut format, mut size) = (None, None, None, None);
        for (name, value) in fields {
            match name.as_str() {
                "key" => key = Some(value),
                "type" => ty = Some(value),
                "format" => format = Some(value),
                "size" => size = Some(value),
                _ => schema(issues, &pointer(&at, name), "unknown compute port key"),
            }
        }
        let key = required_string(key, &pointer(&at, "key"), issues);
        let ty = match ty {
            Some(JsonValue::String(ty)) => ty.clone(),
            Some(_) => {
                schema(issues, &pointer(&at, "type"), "type must be a string");
                continue;
            }
            None => {
                schema(issues, &pointer(&at, "type"), "missing required key");
                continue;
            }
        };
        let ty = match ty.as_str() {
            "image2d" => PortType::Image2d,
            "depth2d" => PortType::Depth2d,
            "buffer" => PortType::Buffer,
            other => {
                schema(
                    issues,
                    &pointer(&at, "type"),
                    format!("{other:?} is not image2d, depth2d or buffer"),
                );
                continue;
            }
        };
        let format = match format {
            None => None,
            Some(JsonValue::String(name)) => match graph_format_from_name(name) {
                Some(format) => Some(format),
                None => {
                    schema(
                        issues,
                        &pointer(&at, "format"),
                        format!("{name:?} is not a WebGPU format name"),
                    );
                    None
                }
            },
            Some(_) => {
                schema(issues, &pointer(&at, "format"), "format must be a string");
                None
            }
        };
        let size = match size {
            None => None,
            Some(value) => parse_size(value, &pointer(&at, "size"), issues),
        };
        ports.push(ComputePortDecl {
            key,
            ty,
            format,
            size,
        });
    }
    ports
}

fn parse_size(value: &JsonValue, at: &str, issues: &mut Vec<Issue>) -> Option<SizeSpec> {
    let JsonValue::Object(fields) = value else {
        schema(issues, at, "size must be {scale} or {width,height}");
        return None;
    };
    if fields.len() == 1
        && let Some(("scale", JsonValue::Number(scale))) =
            fields.first().map(|(key, value)| (key.as_str(), value))
        && (*scale as f32).is_finite()
    {
        return Some(SizeSpec::Scale(*scale as f32));
    }
    if fields.len() == 2 {
        let number = |key: &str| {
            fields
                .iter()
                .find(|(name, _)| name == key)
                .and_then(|(_, value)| match value {
                    JsonValue::Number(n)
                        if n.fract() == 0.0 && *n >= 0.0 && *n <= f64::from(u32::MAX) =>
                    {
                        Some(*n as u32)
                    }
                    _ => None,
                })
        };
        if let (Some(width), Some(height)) = (number("width"), number("height")) {
            return Some(SizeSpec::Fixed { width, height });
        }
    }
    schema(issues, at, "size must be {scale} or {width,height}");
    None
}

fn parse_link(value: &JsonValue, index: usize, issues: &mut Vec<Issue>) -> Option<LinkDoc> {
    let at = format!("/links/{index}");
    let JsonValue::Object(fields) = value else {
        schema(issues, &at, "link must be a JSON object");
        return None;
    };
    let (mut from, mut to, mut history) = (None, None, None);
    for (key, value) in fields {
        match key.as_str() {
            "from" => from = Some(value),
            "to" => to = Some(value),
            "history" => history = Some(value),
            _ => schema(issues, &pointer(&at, key), "unknown link key"),
        }
    }
    let from = parse_port_ref(from, &pointer(&at, "from"), issues);
    let to = parse_port_ref(to, &pointer(&at, "to"), issues);
    let history = match history {
        None => false,
        Some(JsonValue::Bool(value)) => *value,
        Some(_) => {
            schema(
                issues,
                &pointer(&at, "history"),
                "history must be a boolean",
            );
            false
        }
    };
    match (from, to) {
        (Some(from), Some(to)) => Some(LinkDoc { from, to, history }),
        _ => None,
    }
}

fn parse_port_ref(value: Option<&JsonValue>, at: &str, issues: &mut Vec<Issue>) -> Option<PortRef> {
    let Some(value) = value else {
        schema(issues, at, "missing required key");
        return None;
    };
    let JsonValue::String(text) = value else {
        schema(issues, at, "link end must be a string");
        return None;
    };
    match text.split_once('.') {
        Some((node, port)) if !node.is_empty() && !port.is_empty() => Some(PortRef {
            node: node.to_string(),
            port: port.to_string(),
        }),
        _ => {
            schema(issues, at, "link end must be <nodeId>.<portKey>");
            None
        }
    }
}

/// The JSON name of a [`GraphFormat`] (S51): the WebGPU format name, `surface` for the canvas.
fn graph_format_name(format: &GraphFormat) -> &'static str {
    match format {
        GraphFormat::R8Unorm => "r8unorm",
        GraphFormat::Rg8Unorm => "rg8unorm",
        GraphFormat::Rgba8Unorm => "rgba8unorm",
        GraphFormat::Rgba8UnormSrgb => "rgba8unorm-srgb",
        GraphFormat::Bgra8Unorm => "bgra8unorm",
        GraphFormat::R16Float => "r16float",
        GraphFormat::Rg16Float => "rg16float",
        GraphFormat::Rgba16Float => "rgba16float",
        GraphFormat::R32Float => "r32float",
        GraphFormat::Rg32Float => "rg32float",
        GraphFormat::Rgba32Float => "rgba32float",
        GraphFormat::R32Uint => "r32uint",
        GraphFormat::Rg32Uint => "rg32uint",
        GraphFormat::Rgba32Uint => "rgba32uint",
        GraphFormat::Rgb10a2Unorm => "rgb10a2unorm",
        GraphFormat::Rg11b10Ufloat => "rg11b10ufloat",
        GraphFormat::Depth32Float => "depth32float",
        GraphFormat::Surface => "surface",
    }
}

/// The JSON name of a [`PortType`] (S51).
fn port_type_name(ty: &PortType) -> &'static str {
    match ty {
        PortType::Image2d => "image2d",
        PortType::Image2dArray => "image2dArray",
        PortType::Image3d => "image3d",
        PortType::ImageCube => "imageCube",
        PortType::Depth2d => "depth2d",
        PortType::Depth2dArray => "depth2dArray",
        PortType::Buffer => "buffer",
    }
}

fn size_value(size: &SizeSpec) -> JsonValue {
    match size {
        SizeSpec::Scale(scale) => JsonValue::Object(vec![(
            "scale".to_string(),
            JsonValue::Number(f64::from(*scale)),
        )]),
        SizeSpec::Fixed { width, height } => JsonValue::Object(vec![
            ("width".to_string(), JsonValue::Number(f64::from(*width))),
            ("height".to_string(), JsonValue::Number(f64::from(*height))),
        ]),
        SizeSpec::FromParam(key) => JsonValue::Object(vec![(
            "param".to_string(),
            JsonValue::String((*key).to_string()),
        )]),
        SizeSpec::Input(key) => JsonValue::Object(vec![(
            "input".to_string(),
            JsonValue::String((*key).to_string()),
        )]),
    }
}

/// `key`, `type`, `format` (when present), `size` (when present).
fn port_value(port: &ComputePortDecl) -> JsonValue {
    let mut fields = vec![
        ("key".to_string(), JsonValue::String(port.key.clone())),
        (
            "type".to_string(),
            JsonValue::String(port_type_name(&port.ty).to_string()),
        ),
    ];
    if let Some(format) = &port.format {
        fields.push((
            "format".to_string(),
            JsonValue::String(graph_format_name(format).to_string()),
        ));
    }
    if let Some(size) = &port.size {
        fields.push(("size".to_string(), size_value(size)));
    }
    JsonValue::Object(fields)
}

/// `id`, `type`, `v` (when present), `params` (when non-empty), then the compute keys.
fn node_value(node: &NodeDoc) -> JsonValue {
    let mut fields = vec![
        ("id".to_string(), JsonValue::String(node.id.clone())),
        ("type".to_string(), JsonValue::String(node.type_id.clone())),
    ];
    if let Some(version) = node.version {
        fields.push(("v".to_string(), JsonValue::Number(f64::from(version))));
    }
    if !node.params.is_empty() {
        fields.push(("params".to_string(), JsonValue::Object(node.params.clone())));
    }
    if let Some(compute) = &node.compute {
        fields.push((
            "inputs".to_string(),
            JsonValue::Array(compute.inputs.iter().map(port_value).collect()),
        ));
        fields.push((
            "outputs".to_string(),
            JsonValue::Array(compute.outputs.iter().map(port_value).collect()),
        ));
        fields.push(("math".to_string(), compute.math.clone()));
        fields.push(("wgsl".to_string(), JsonValue::String(compute.wgsl.clone())));
    }
    JsonValue::Object(fields)
}

fn port_ref_text(port: &PortRef) -> String {
    format!("{}.{}", port.node, port.port)
}

/// `from`, `to`, `history` (only when true).
fn link_value(link: &LinkDoc) -> JsonValue {
    let mut fields = vec![
        (
            "from".to_string(),
            JsonValue::String(port_ref_text(&link.from)),
        ),
        ("to".to_string(), JsonValue::String(port_ref_text(&link.to))),
    ];
    if link.history {
        fields.push(("history".to_string(), JsonValue::Bool(true)));
    }
    JsonValue::Object(fields)
}

/// The JSON value of `doc` with the S50 canonical key order.
///
/// The top level is `format`, `version`, `name`, `nodes`, `links`, then `editor` when
/// present; node params keep the order [`GraphDoc`] holds them in, so
/// `write_graph(doc: &GraphDoc) -> String` applied to the result of
/// `canonical_doc(doc: &GraphDoc, catalog: &Catalog) -> GraphDoc`
/// (`crates/raa-graph/src/validate.rs`) gives the S50 canonical text.
///
/// crates/raa-graph/src/lib.rs: out of scope for this share: consumers use the module path
pub fn graph_to_json_value(doc: &GraphDoc) -> JsonValue {
    let mut fields = vec![
        ("format".to_string(), JsonValue::String(FORMAT.to_string())),
        ("version".to_string(), JsonValue::Number(f64::from(VERSION))),
        ("name".to_string(), JsonValue::String(doc.name.clone())),
        (
            "nodes".to_string(),
            JsonValue::Array(doc.nodes.iter().map(node_value).collect()),
        ),
        (
            "links".to_string(),
            JsonValue::Array(doc.links.iter().map(link_value).collect()),
        ),
    ];
    if let Some(editor) = &doc.editor {
        fields.push(("editor".to_string(), editor.clone()));
    }
    JsonValue::Object(fields)
}

/// The canonical document text (S50): `raa_loader::json::write_pretty` of
/// [`graph_to_json_value`] plus a final LF.
pub fn write_graph(doc: &GraphDoc) -> String {
    let mut text = String::new();
    raa_loader::json::write_pretty(&graph_to_json_value(doc), &mut text);
    text.push('\n');
    text
}
