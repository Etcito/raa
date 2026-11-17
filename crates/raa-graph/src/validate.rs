//! The validation context and the device limits validation reads (S52, S53), and the
//! structure stage of the render graph validator: `validate` and `canonical_doc`.
//!
//! `crates/raa-graph/src/lib.rs`: out of scope for this share: consumers use the module
//! path (`raa_graph::validate::{validate, canonical_doc}`).

use crate::catalog::{
    Catalog, GraphFormat, NodeTypeSpec, ParamKind, ParamSpec, ParamValue, PortSpec, PortType,
};
use crate::error::{Issue, Severity, Stage, sort_issues};
use crate::format::{GraphDoc, LinkDoc, NodeDoc, PortRef};

/// The per-stage binding limits a graph is validated against (RG_LIMITS).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphLimits {
    pub sampled_textures: u32,
    pub samplers: u32,
    pub storage_textures: u32,
    pub storage_buffers: u32,
    pub uniform_buffers: u32,
}

impl GraphLimits {
    /// The device-free limits RG_LIMITS names, for code without a device.
    pub const CORE: GraphLimits = GraphLimits {
        sampled_textures: 16,
        samplers: 16,
        storage_textures: 4,
        storage_buffers: 8,
        uniform_buffers: 12,
    };
}

/// Everything validation reads besides the document itself (S53).
pub struct ValidateCtx<'a> {
    pub catalog: &'a Catalog,
    pub limits: GraphLimits,
    pub installed: &'a dyn Fn(&str) -> bool,
    pub viewports: u32,
    pub engine_options: &'a [(&'static str, ParamValue)],
}

/// The node types a graph must hold one node of each (`graph.required_node`).
const REQUIRED_TYPES: [&str; 3] = ["pass.prepass", "pass.forward", "pass.tonemap"];

/// The display output node type (`graph.output`).
const OUTPUT_TYPE: &str = "output.display";

/// `validate` checks the structure of `doc` against `ctx` (S52, first half of the structure
/// stage): node ids, node types and versions, parameters, links, history links, the output
/// node, the required nodes and the single-instance types. The issues come back sorted by
/// `sort_issues`.
///
/// Handoff: `validate` returns the structure issues this file collects today, sorted by
/// `sort_issues`. Card 'raa-graph validate 2/2: inputs, cycles, wiring, availability and
/// budget' extends it: that card adds its structure checks before the final `sort_issues`
/// call, then applies the stage stop and the budget stage. State passed: the issue list
/// being collected, with `doc` and `ctx`.
pub fn validate(doc: &GraphDoc, ctx: &ValidateCtx<'_>) -> Vec<Issue> {
    let mut issues = Vec::new();
    check_node_ids(doc, &mut issues);
    for node in &doc.nodes {
        match ctx.catalog.node(&node.type_id) {
            Some(spec) => check_node(node, spec, &mut issues),
            None => issues.push(node_issue(
                node,
                "graph.node_type",
                Severity::Error,
                format!(
                    "node `{}` has type `{}`, which the catalog does not define",
                    node.id, node.type_id
                ),
                format!(
                    "Change the type of node `{}` to a catalog node type, or delete the node.",
                    node.id
                ),
            )),
        }
    }
    check_instances(doc, ctx, &mut issues);
    check_links(doc, ctx, &mut issues);
    sort_issues(&mut issues, Some(doc));
    issues
}

/// `canonical_doc` returns a copy of `doc` whose node params follow the catalog key order
/// of each node's type. Keys the type does not declare, and every param of a node of an
/// unknown type, keep their document order after the declared keys. Nothing else changes.
///
/// Handoff: `canonical_doc` returns the copy that card 'raa-graph exec 4/5: GraphExecutor
/// apply, rollback and clear' stores as its active document, `canonical_doc(doc, &catalog)`.
/// Today it only builds and returns that copy; nothing in this crate stores it.
pub fn canonical_doc(doc: &GraphDoc, catalog: &Catalog) -> GraphDoc {
    let mut canonical = doc.clone();
    for node in &mut canonical.nodes {
        if let Some(spec) = catalog.node(&node.type_id) {
            // `sort_by_key` is stable: undeclared keys all map to `usize::MAX` and keep
            // their document order.
            node.params.sort_by_key(|(key, _)| {
                spec.params
                    .iter()
                    .position(|param| param.key == key.as_str())
                    .unwrap_or(usize::MAX)
            });
        }
    }
    canonical
}

/// A structure-stage issue with no node, port, parameter or link set yet.
fn structure(code: &'static str, severity: Severity, message: String, fix: String) -> Issue {
    Issue {
        code,
        severity,
        stage: Stage::Structure,
        node: None,
        port: None,
        param: None,
        link: None,
        message,
        fix,
    }
}

/// A structure-stage issue on node `node`.
fn node_issue(
    node: &NodeDoc,
    code: &'static str,
    severity: Severity,
    message: String,
    fix: String,
) -> Issue {
    Issue {
        node: Some(node.id.clone()),
        ..structure(code, severity, message, fix)
    }
}

/// `^[A-Za-z][A-Za-z0-9_]{0,63}$` (RG_NODE_ID).
fn valid_node_id(id: &str) -> bool {
    let mut chars = id.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic())
        && id.len() <= 64
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `^[a-z][A-Za-z0-9]{0,31}$` (RG_PORT_KEY).
fn valid_port_key(key: &str) -> bool {
    let mut chars = key.chars();
    chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && key.len() <= 32
        && chars.all(|c| c.is_ascii_alphanumeric())
}

/// `graph.node_id`: an id outside the node id pattern, or one an earlier node already has.
fn check_node_ids(doc: &GraphDoc, issues: &mut Vec<Issue>) {
    for (index, node) in doc.nodes.iter().enumerate() {
        let problem = if !valid_node_id(&node.id) {
            format!(
                "node id `{}` is not a letter followed by letters, digits or `_` (64 characters at most)",
                node.id
            )
        } else if doc.nodes[..index]
            .iter()
            .any(|earlier| earlier.id == node.id)
        {
            format!("node id `{}` is already used by an earlier node", node.id)
        } else {
            continue;
        };
        issues.push(node_issue(
            node,
            "graph.node_id",
            Severity::Error,
            problem,
            format!(
                "Give node `{}` an unused id that matches ^[A-Za-z][A-Za-z0-9_]{{0,63}}$.",
                node.id
            ),
        ));
    }
}

/// Checks a node of a known type: its version, its upgrade and its parameter values
/// (`graph.node_version`, `graph.node_upgraded`, `graph.param_range`).
fn check_node(node: &NodeDoc, spec: &NodeTypeSpec, issues: &mut Vec<Issue>) {
    let mut params = convert_params(node, spec, issues);
    let written = node.version.unwrap_or(spec.version);
    let mut ranged = true;
    if written > spec.version {
        issues.push(node_issue(
            node,
            "graph.node_version",
            Severity::Error,
            format!(
                "node `{}` is v{written}, but the catalog's `{}` is v{}",
                node.id, spec.id, spec.version
            ),
            format!(
                "Set `v` of node `{}` to {} or remove it.",
                node.id, spec.version
            ),
        ));
    } else if written < spec.version {
        ranged = upgrade(node, spec, written, &mut params, issues);
    }
    if ranged {
        check_ranges(node, spec, &params, issues);
    }
}

/// Converts the parameters the type declares, reporting `graph.param_unknown` and
/// `graph.param_kind`. Returns the converted values in document order.
fn convert_params(
    node: &NodeDoc,
    spec: &NodeTypeSpec,
    issues: &mut Vec<Issue>,
) -> Vec<(String, ParamValue)> {
    let mut converted = Vec::new();
    for (key, value) in &node.params {
        let Some(param) = spec.params.iter().find(|param| param.key == key.as_str()) else {
            issues.push(Issue {
                param: Some(key.clone()),
                ..node_issue(
                    node,
                    "graph.param_unknown",
                    Severity::Error,
                    format!(
                        "node `{}` sets `{key}`, which type `{}` does not declare",
                        node.id, spec.id
                    ),
                    format!("Delete `{key}` from node `{}`.", node.id),
                )
            });
            continue;
        };
        match ParamValue::from_json(&param.kind, value) {
            Ok(parsed) => converted.push((key.clone(), parsed)),
            Err(reason) => issues.push(Issue {
                param: Some(key.clone()),
                ..node_issue(
                    node,
                    "graph.param_kind",
                    Severity::Error,
                    format!("node `{}` param `{key}`: {reason}", node.id),
                    format!(
                        "Write `{key}` on node `{}` in the `{}` form the catalog declares.",
                        node.id,
                        kind_name(&param.kind)
                    ),
                )
            }),
        }
    }
    converted
}

/// The S51 JSON name of a parameter kind, for messages.
fn kind_name(kind: &ParamKind) -> &'static str {
    match kind {
        ParamKind::Number => "number",
        ParamKind::Int => "int",
        ParamKind::Bool => "bool",
        ParamKind::Enum => "enum",
        ParamKind::Vec2 => "vec2",
        ParamKind::Vec3 => "vec3",
        ParamKind::Vec4 => "vec4",
        ParamKind::String => "string",
    }
}

/// Runs the type's upgrade once per version step from `written` to the catalog version, on
/// the converted params. Keys the catalog does not declare are reported before this runs
/// (`graph.param_unknown`), so an upgrade only sees declared keys. A failed step is
/// `graph.node_version` and returns false; a run that succeeds is the warning
/// `graph.node_upgraded`.
fn upgrade(
    node: &NodeDoc,
    spec: &NodeTypeSpec,
    written: u32,
    params: &mut Vec<(String, ParamValue)>,
    issues: &mut Vec<Issue>,
) -> bool {
    for step in written..spec.version {
        if let Err(reason) = (spec.upgrade)(step, params) {
            issues.push(node_issue(
                node,
                "graph.node_version",
                Severity::Error,
                format!(
                    "node `{}` is v{written}, and its upgrade from v{step} failed: {reason}",
                    node.id
                ),
                format!(
                    "Set `v` of node `{}` to {} and write its params for that version.",
                    node.id, spec.version
                ),
            ));
            return false;
        }
    }
    issues.push(node_issue(
        node,
        "graph.node_upgraded",
        Severity::Warning,
        format!(
            "node `{}` is v{written}; its params were upgraded to the catalog's v{}",
            node.id, spec.version
        ),
        format!(
            "Set `v` of node `{}` to {} so the document matches the catalog.",
            node.id, spec.version
        ),
    ));
    true
}

/// `graph.param_range` for each converted value that its parameter's bounds or options refuse.
fn check_ranges(
    node: &NodeDoc,
    spec: &NodeTypeSpec,
    params: &[(String, ParamValue)],
    issues: &mut Vec<Issue>,
) {
    for (key, value) in params {
        let Some(param) = spec.params.iter().find(|param| param.key == key.as_str()) else {
            continue;
        };
        if let Some(problem) = range_problem(param, value) {
            issues.push(Issue {
                param: Some(key.clone()),
                ..node_issue(
                    node,
                    "graph.param_range",
                    Severity::Error,
                    format!("node `{}` param `{key}`: {problem}", node.id),
                    format!("Set `{key}` on node `{}` to {}.", node.id, allowed(param)),
                )
            });
        }
    }
}

/// What is wrong with a converted value of `param`: a number or int outside its `min` and
/// `max`, an enum outside its options, or an int outside its decimal options (amended (c)).
fn range_problem(param: &ParamSpec, value: &ParamValue) -> Option<String> {
    let listed = |text: &str| param.options.contains(&text);
    match value {
        ParamValue::Enum(choice) if !listed(choice) => {
            Some(format!("`{choice}` is not one of the options"))
        }
        ParamValue::Int(int) if !param.options.is_empty() && !listed(&int.to_string()) => {
            Some(format!("{int} is not one of the allowed integers"))
        }
        ParamValue::Int(int) => bounds_problem(param, *int as f64),
        ParamValue::Number(number) => bounds_problem(param, *number),
        _ => None,
    }
}

/// The problem with `number` when it lies outside the `min` and `max` of `param`.
fn bounds_problem(param: &ParamSpec, number: f64) -> Option<String> {
    let below = param.min.is_some_and(|min| number < min);
    let above = param.max.is_some_and(|max| number > max);
    (below || above).then(|| {
        format!(
            "{number} is outside [{}, {}]",
            bound(param.min, "-inf"),
            bound(param.max, "inf")
        )
    })
}

/// A bound as text, or `open` when the parameter has none.
fn bound(limit: Option<f64>, open: &str) -> String {
    limit.map_or_else(|| open.to_string(), |limit| limit.to_string())
}

/// The values a parameter accepts, as a fix hint.
fn allowed(param: &ParamSpec) -> String {
    if param.options.is_empty() {
        format!(
            "a value in [{}, {}]",
            bound(param.min, "-inf"),
            bound(param.max, "inf")
        )
    } else {
        format!("one of {}", param.options.join(", "))
    }
}

/// `graph.single_instance` for a repeat of a type that allows one node, then `graph.output`
/// and `graph.required_node` over the whole graph.
fn check_instances(doc: &GraphDoc, ctx: &ValidateCtx<'_>, issues: &mut Vec<Issue>) {
    for (index, node) in doc.nodes.iter().enumerate() {
        let single = ctx
            .catalog
            .node(&node.type_id)
            .is_some_and(|spec| !spec.multi_instance);
        if single
            && doc.nodes[..index]
                .iter()
                .any(|earlier| earlier.type_id == node.type_id)
        {
            issues.push(node_issue(
                node,
                "graph.single_instance",
                Severity::Error,
                format!(
                    "node `{}` is a second `{}` node, and that type allows one per graph",
                    node.id, node.type_id
                ),
                format!(
                    "Delete node `{}`, or change its type to one the graph does not already use.",
                    node.id
                ),
            ));
        }
    }
    let outputs = doc
        .nodes
        .iter()
        .filter(|node| node.type_id == OUTPUT_TYPE)
        .count();
    if outputs != 1 {
        let fix = if outputs == 0 {
            format!("Add one `{OUTPUT_TYPE}` node and link its `color` input.")
        } else {
            format!("Delete all but one `{OUTPUT_TYPE}` node.")
        };
        issues.push(structure(
            "graph.output",
            Severity::Error,
            format!("the graph has {outputs} `{OUTPUT_TYPE}` nodes, and it needs exactly one"),
            fix,
        ));
    }
    for required in REQUIRED_TYPES {
        if !doc.nodes.iter().any(|node| node.type_id == required) {
            issues.push(structure(
                "graph.required_node",
                Severity::Error,
                format!("the graph has no `{required}` node, and every graph needs one"),
                format!("Add one node of type `{required}`."),
            ));
        }
    }
}

/// The link checks, link by link in document order: `graph.port_unknown`, `graph.link_type`,
/// `graph.input_taken` and `graph.history`.
fn check_links(doc: &GraphDoc, ctx: &ValidateCtx<'_>, issues: &mut Vec<Issue>) {
    for (index, link) in doc.links.iter().enumerate() {
        let at = u32::try_from(index).unwrap_or(u32::MAX);
        let from = link_end(doc, ctx.catalog, &link.from, true, at, issues);
        let to = link_end(doc, ctx.catalog, &link.to, false, at, issues);
        if let (Some(from), Some(to)) = (from, to)
            && let Some((problem, fix)) = mismatch(from, to)
        {
            issues.push(Issue {
                node: Some(link.to.node.clone()),
                port: Some(link.to.port.clone()),
                link: Some(at),
                ..structure(
                    "graph.link_type",
                    Severity::Error,
                    format!(
                        "link {at} from `{}.{}` {problem}",
                        link.from.node, link.from.port
                    ),
                    format!("Link `{}.{}` from {fix}.", link.to.node, link.to.port),
                )
            });
        }
        if let Some(first) = doc.links[..index]
            .iter()
            .position(|earlier| earlier.to == link.to)
        {
            issues.push(Issue {
                node: Some(link.to.node.clone()),
                port: Some(link.to.port.clone()),
                link: Some(at),
                ..structure(
                    "graph.input_taken",
                    Severity::Error,
                    format!(
                        "input `{}.{}` already takes link {first}, so link {at} is a second link into it",
                        link.to.node, link.to.port
                    ),
                    format!("Delete link {at}, or link it into a different input."),
                )
            });
        }
        if link.history {
            history_issues(at, link, from, to, issues);
        }
    }
}

/// `graph.history` (on the link only): a history link into a port without `history`, or out
/// of a buffer port, since history carries images.
fn history_issues(
    at: u32,
    link: &LinkDoc,
    from: Option<&PortSpec>,
    to: Option<&PortSpec>,
    issues: &mut Vec<Issue>,
) {
    if let Some(to) = to
        && !to.history
    {
        issues.push(Issue {
            link: Some(at),
            ..structure(
                "graph.history",
                Severity::Error,
                format!(
                    "link {at} is a history link into `{}.{}`, which takes no history",
                    link.to.node, link.to.port
                ),
                format!("Remove the history flag from link {at}."),
            )
        });
    }
    if let Some(from) = from
        && matches!(from.ty, PortType::Buffer)
    {
        issues.push(Issue {
            link: Some(at),
            ..structure(
                "graph.history",
                Severity::Error,
                format!(
                    "link {at} is a history link out of the buffer `{}.{}`, and history carries images",
                    link.from.node, link.from.port
                ),
                format!("Remove the history flag from link {at}."),
            )
        });
    }
}

/// Resolves one end of link `at` to the port it names (`graph.port_unknown` when the end
/// names no node, no port key or no port of that side). An end of a node whose type is
/// unknown resolves to nothing, since `graph.node_type` already reports that node.
fn link_end<'c>(
    doc: &GraphDoc,
    catalog: &'c Catalog,
    end: &PortRef,
    outputs: bool,
    at: u32,
    issues: &mut Vec<Issue>,
) -> Option<&'c PortSpec> {
    let side = if outputs { "output" } else { "input" };
    let port_issue = |message: String, fix: String| Issue {
        node: Some(end.node.clone()),
        port: Some(end.port.clone()),
        link: Some(at),
        ..structure("graph.port_unknown", Severity::Error, message, fix)
    };
    let Some(node) = doc.nodes.iter().find(|node| node.id == end.node) else {
        issues.push(port_issue(
            format!(
                "link {at} names node `{}`, which the document does not declare",
                end.node
            ),
            format!("Point link {at} at a node the document declares, or delete the link."),
        ));
        return None;
    };
    if !valid_port_key(&end.port) {
        issues.push(port_issue(
            format!("link {at} names `{}` on node `{}`, which is not a port key", end.port, end.node),
            format!("Rename that port in link {at} to a lowercase letter followed by up to 31 letters or digits."),
        ));
        return None;
    }
    let spec = catalog.node(&node.type_id)?;
    let ports = if outputs { &spec.outputs } else { &spec.inputs };
    let found = ports.iter().find(|port| port.key == end.port);
    if found.is_none() {
        issues.push(port_issue(
            format!(
                "link {at} names {side} `{}` on node `{}`, which has no such {side}",
                end.port, end.node
            ),
            format!(
                "Name one of the {side} ports of `{}` in link {at}: {}.",
                spec.id,
                port_keys(ports)
            ),
        ));
    }
    found
}

/// The port keys of `ports` for a fix hint, or `none` when the list is empty.
fn port_keys(ports: &[PortSpec]) -> String {
    if ports.is_empty() {
        return "none".to_string();
    }
    ports
        .iter()
        .map(|port| port.key)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Why output `from` cannot feed input `to` (`graph.link_type`): a different port type, or an
/// input format the output does not have. Returns the problem and the source of the fix.
fn mismatch(from: &PortSpec, to: &PortSpec) -> Option<(String, String)> {
    if from.ty != to.ty {
        return Some((
            format!(
                "is of type {}, and the input takes {}",
                type_name(&from.ty),
                type_name(&to.ty)
            ),
            format!("an output of type {}", type_name(&to.ty)),
        ));
    }
    let wanted = to.format.as_ref()?;
    if from.format.as_ref() == Some(wanted) {
        return None;
    }
    Some((
        format!(
            "has format {}, and the input takes {}",
            from.format.as_ref().map_or("none", GraphFormat::name),
            wanted.name()
        ),
        format!("an output with format {}", wanted.name()),
    ))
}

/// The S51 JSON name of a port type, for messages.
fn type_name(ty: &PortType) -> &'static str {
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
