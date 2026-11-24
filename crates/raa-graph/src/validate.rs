//! The validation context and the device limits validation reads (S52, S53), and the render
//! graph validator: `validate` (the structure and budget stages) and `canonical_doc`.
//!
//! `crates/raa-graph/src/lib.rs`: out of scope for this share: consumers use the module
//! path (`raa_graph::validate::{validate, canonical_doc}`).

use crate::catalog::{
    Catalog, GraphFormat, NodeTypeSpec, ParamKind, ParamSpec, ParamValue, PortSpec, PortType,
    ShaderStage,
};
use crate::error::{Issue, Severity, Stage, sort_issues};
use crate::format::{GraphDoc, LinkDoc, NodeDoc, PortRef};

#[derive(Clone)]
struct EndPort {
    ty: PortType,
    format: Option<GraphFormat>,
    size: Option<crate::catalog::SizeSpec>,
    signals: &'static [crate::catalog::SignalContract],
    history: bool,
    generic: bool,
    default_source: &'static [&'static str],
}

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

/// `validate` checks `doc` against `ctx` (S52): the structure stage (node ids, types, versions,
/// parameters, links, history links, inputs, cycles, wiring, availability, the output node, the
/// required and single-instance nodes, and the family hooks), then the budget stage, which runs
/// only when the structure stage has no error. The issues come back sorted by `sort_issues`.
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
    check_inputs(doc, ctx, &mut issues);
    check_cycles(doc, ctx, &mut issues);
    check_availability(doc, ctx, &mut issues);
    family_hooks(doc, ctx, &mut issues);
    // The S52 stage stop: a structure error ends validation before the budget stage.
    if !structure_failed(&issues) {
        check_budgets(doc, ctx, &mut issues);
    }
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
    if node.compute.is_some() {
        return converted;
    }
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
/// `graph.input_taken`, `graph.fixed_wiring` and `graph.history`.
fn check_links(doc: &GraphDoc, ctx: &ValidateCtx<'_>, issues: &mut Vec<Issue>) {
    for (index, link) in doc.links.iter().enumerate() {
        let at = u32::try_from(index).unwrap_or(u32::MAX);
        let from = link_end(doc, ctx.catalog, &link.from, true, at, issues);
        let to = link_end(doc, ctx.catalog, &link.to, false, at, issues);
        if let (Some(from), Some(to)) = (&from, &to)
            && let Some((problem, fix)) =
                mismatch(from, to).or_else(|| signal_mismatch(from, to, link.history))
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
        if let (Some(_), Some(to)) = (&from, &to) {
            fixed_wiring(doc, link, at, to, issues);
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
            history_issues(at, link, from.as_ref(), to.as_ref(), issues);
        }
    }
}

/// `graph.history` (on the link only): a history link into a port without `history`, or out
/// of a buffer port, since history carries images.
fn history_issues(
    at: u32,
    link: &LinkDoc,
    from: Option<&EndPort>,
    to: Option<&EndPort>,
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
fn link_end(
    doc: &GraphDoc,
    catalog: &Catalog,
    end: &PortRef,
    outputs: bool,
    at: u32,
    issues: &mut Vec<Issue>,
) -> Option<EndPort> {
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
    if let Some(decl) = node
        .compute
        .as_ref()
        .filter(|_| node.type_id == "compute.wgsl")
    {
        let ports = if outputs { &decl.outputs } else { &decl.inputs };
        let found = ports.iter().find(|port| port.key == end.port);
        if let Some(port) = found {
            return Some(EndPort {
                ty: port.ty.clone(),
                format: port.format.clone(),
                size: port.size.clone(),
                signals: &[],
                history: !outputs,
                generic: !outputs,
                default_source: &[],
            });
        }
        issues.push(port_issue(
            format!(
                "link {at} names {side} `{}` on node `{}`, which has no such {side}",
                end.port, end.node
            ),
            format!(
                "Name one of the declared {side} ports of `{}` in link {at}.",
                node.id
            ),
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
    found.map(|port| EndPort {
        ty: port.ty.clone(),
        format: port.format.clone(),
        size: Some(port.size.clone()),
        signals: crate::catalog::port_signals(&node.type_id, port.key, outputs),
        history: port.history,
        generic: port.generic,
        default_source: port.default_source,
    })
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
fn mismatch(from: &EndPort, to: &EndPort) -> Option<(String, String)> {
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

/// History copies require COPY_SRC on the current-frame image before it can be sampled.
fn signal_mismatch(from: &EndPort, to: &EndPort, history: bool) -> Option<(String, String)> {
    use crate::catalog::SignalCadence;
    if to.signals.is_empty() {
        return None;
    }
    let cadence = if history {
        SignalCadence::Previous
    } else {
        SignalCadence::Current
    };
    let compatible = from.signals.first().is_some_and(|source| {
        to.signals.iter().any(|target| {
            let usage = if history {
                target.usage | wgpu::TextureUsages::COPY_SRC
            } else {
                target.usage
            };
            source.version == target.version
                && source.encoding == target.encoding
                && source.usage.contains(usage)
                && source.samples == target.samples
                && source.min_layers >= target.min_layers
                && source.max_layers <= target.max_layers
                && target.cadence == cadence
                && source.active_rect == target.active_rect
                && from.size == to.size
        })
    });
    if compatible {
        return None;
    }
    let expected = to
        .signals
        .iter()
        .map(|signal| format!("{} v{}", signal.encoding.name(), signal.version))
        .collect::<Vec<_>>()
        .join(" or ");
    let actual = from
        .signals
        .first()
        .map_or("untyped", |signal| signal.encoding.name());
    Some((
        format!("has signal {actual}, incompatible with the input contract ({expected})"),
        format!("an output providing {expected} with matching extent, cadence and resource layout"),
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

/// `graph.fixed_wiring` (amended (d)): a link into an input with `generic` false must come from
/// the first `default_source` entry the graph holds, since this version fixes the wiring.
fn fixed_wiring(doc: &GraphDoc, link: &LinkDoc, at: u32, to: &EndPort, issues: &mut Vec<Issue>) {
    let Some(source) = doc.nodes.iter().find(|node| node.id == link.from.node) else {
        return;
    };
    let given = (source.type_id.as_str(), link.from.port.as_str());
    let expected = first_present_source(doc, to.default_source);
    if to.generic || expected == Some(given) {
        return;
    }
    let target = format!("{}.{}", link.to.node, link.to.port);
    let (problem, fix) = match expected {
        Some((node_type, key)) => (
            format!("takes only its default source `{node_type}.{key}`"),
            format!("Link `{target}` from `{node_type}.{key}`, or delete link {at}."),
        ),
        None => (
            "has no default source in the graph".to_string(),
            match to.default_source.first() {
                Some(entry) => format!("Delete link {at}, or add a node for `{entry}`."),
                None => format!("Delete link {at}: this version takes no link into `{target}`."),
            },
        ),
    };
    issues.push(Issue {
        node: Some(link.to.node.clone()),
        port: Some(link.to.port.clone()),
        link: Some(at),
        ..structure(
            "graph.fixed_wiring",
            Severity::Error,
            format!(
                "fixed wiring in this version: link {at} from `{}.{}` into `{target}` {problem}",
                link.from.node, link.from.port
            ),
            fix,
        )
    });
}

/// Whether a link takes input `port` of `node`.
fn is_linked(doc: &GraphDoc, node: &NodeDoc, port: &str) -> bool {
    doc.links
        .iter()
        .any(|link| link.to.node == node.id && link.to.port == port)
}

/// The first `default_source` entry of `port` whose node type a node of the graph has, as its
/// node type and port key, split at the last `.`. An entry with no `.` is skipped.
fn first_present_source(
    doc: &GraphDoc,
    default_source: &[&'static str],
) -> Option<(&'static str, &'static str)> {
    default_source.iter().copied().find_map(|entry| {
        let (node_type, key) = entry.rsplit_once('.')?;
        doc.nodes
            .iter()
            .any(|node| node.type_id == node_type)
            .then_some((node_type, key))
    })
}

/// The default sources of `port` for a message, or `none`.
fn sources_text(default_source: &[&'static str]) -> String {
    if default_source.is_empty() {
        "none".to_string()
    } else {
        default_source.join(", ")
    }
}

/// `graph.input_missing`: an input no link takes that is `required`, or whose default sources
/// are all absent from the graph while `absent_ok` is false.
fn check_inputs(doc: &GraphDoc, ctx: &ValidateCtx<'_>, issues: &mut Vec<Issue>) {
    for node in &doc.nodes {
        if let Some(decl) = node
            .compute
            .as_ref()
            .filter(|_| node.type_id == "compute.wgsl")
        {
            for port in &decl.inputs {
                if !is_linked(doc, node, &port.key) {
                    issues.push(Issue {
                        port: Some(port.key.clone()),
                        ..node_issue(
                            node,
                            "graph.input_missing",
                            Severity::Error,
                            format!(
                                "input `{}.{}` is required, and no link takes it",
                                node.id, port.key
                            ),
                            format!(
                                "Link an output of type {} into `{}.{}`.",
                                type_name(&port.ty),
                                node.id,
                                port.key
                            ),
                        )
                    });
                }
            }
            continue;
        }
        let Some(spec) = ctx.catalog.node(&node.type_id) else {
            continue;
        };
        for port in &spec.inputs {
            if is_linked(doc, node, port.key) {
                continue;
            }
            let problem = if port.required {
                format!(
                    "input `{}.{}` is required, and no link takes it",
                    node.id, port.key
                )
            } else if !port.absent_ok && first_present_source(doc, port.default_source).is_none() {
                format!(
                    "input `{}.{}` has no link, and none of its default sources ({}) is in the graph",
                    node.id,
                    port.key,
                    sources_text(port.default_source)
                )
            } else {
                continue;
            };
            let fix = format!(
                "Link an output of type {} into `{}.{}`.",
                type_name(&port.ty),
                node.id,
                port.key
            );
            issues.push(Issue {
                port: Some(port.key.to_string()),
                ..node_issue(node, "graph.input_missing", Severity::Error, problem, fix)
            });
        }
    }
}

/// `graph.cycle`: nodes that read each other, over non-history links and the default source of
/// each input no link takes. Each cyclic group of nodes is reported once, on its first node in
/// document order.
fn check_cycles(doc: &GraphDoc, ctx: &ValidateCtx<'_>, issues: &mut Vec<Issue>) {
    let reads = read_edges(doc, ctx.catalog);
    let reach: Vec<Vec<bool>> = (0..reads.len())
        .map(|start| reached_from(&reads, start))
        .collect();
    // Two nodes share a cycle when each reaches the other.
    let mutual = |a: usize, b: usize| reach[a][b] && reach[b][a];
    for first in 0..reads.len() {
        let group: Vec<usize> = (0..reads.len())
            .filter(|&other| mutual(first, other))
            .collect();
        // A node in no cycle has an empty group, so only the group's first node reports it.
        if group.first() != Some(&first) {
            continue;
        }
        let names: Vec<&str> = group
            .iter()
            .filter_map(|&index| doc.nodes.get(index))
            .map(|node| node.id.as_str())
            .collect();
        let problem = match names.as_slice() {
            [only] => format!("node `{only}` reads itself, and no history link breaks that cycle"),
            _ => format!(
                "nodes {} read each other, and no history link breaks that cycle",
                quoted(&names)
            ),
        };
        if let Some(node) = doc.nodes.get(first) {
            let fix = "Set `history` on one link inside the cycle, or link one of its inputs from a node outside it.";
            issues.push(node_issue(
                node,
                "graph.cycle",
                Severity::Error,
                problem,
                fix.to_string(),
            ));
        }
    }
}

/// The nodes each node reads, by node index: the source of each non-history link, and the first
/// node of the first present default-source type for each input no link takes.
fn read_edges(doc: &GraphDoc, catalog: &Catalog) -> Vec<Vec<usize>> {
    let position = |id: &str| doc.nodes.iter().position(|node| node.id == id);
    let mut reads = vec![Vec::new(); doc.nodes.len()];
    for link in doc.links.iter().filter(|link| !link.history) {
        if let (Some(from), Some(to)) = (position(&link.from.node), position(&link.to.node))
            && let Some(list) = reads.get_mut(to)
        {
            list.push(from);
        }
    }
    for (to, node) in doc.nodes.iter().enumerate() {
        let Some(spec) = catalog.node(&node.type_id) else {
            continue;
        };
        for port in spec
            .inputs
            .iter()
            .filter(|port| !is_linked(doc, node, port.key))
        {
            let source =
                first_present_source(doc, port.default_source).and_then(|(node_type, _)| {
                    doc.nodes
                        .iter()
                        .position(|known| known.type_id == node_type)
                });
            if let (Some(from), Some(list)) = (source, reads.get_mut(to)) {
                list.push(from);
            }
        }
    }
    reads
}

/// Flags the nodes reachable from `start` through one or more `reads` edges.
fn reached_from(reads: &[Vec<usize>], start: usize) -> Vec<bool> {
    let mut reached = vec![false; reads.len()];
    let mut stack = Vec::new();
    stack.extend(reads.get(start).into_iter().flatten().copied());
    while let Some(next) = stack.pop() {
        if let Some(flag) = reached.get_mut(next).filter(|flag| !**flag) {
            *flag = true;
            stack.extend(reads.get(next).into_iter().flatten().copied());
        }
    }
    reached
}

/// The names as backquoted ids, separated by commas.
fn quoted(names: &[&str]) -> String {
    names
        .iter()
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `graph.unavailable` for each node whose pass the engine does not install, and
/// `graph.multi_view` when the engine has more than one viewport.
fn check_availability(doc: &GraphDoc, ctx: &ValidateCtx<'_>, issues: &mut Vec<Issue>) {
    for node in &doc.nodes {
        let Some(pass) = ctx.catalog.node(&node.type_id).and_then(|spec| spec.pass) else {
            continue;
        };
        if !(ctx.installed)(pass) {
            let problem = format!(
                "node `{}` runs pass `{pass}`, which this engine does not install",
                node.id
            );
            let fix = format!(
                "Delete node `{}`, or run this graph on an engine that installs `{pass}`.",
                node.id
            );
            issues.push(node_issue(
                node,
                "graph.unavailable",
                Severity::Error,
                problem,
                fix,
            ));
        }
    }
    if ctx.viewports > 1 {
        issues.push(structure(
            "graph.multi_view",
            Severity::Error,
            format!(
                "the engine has {} viewports, and a render graph drives one view",
                ctx.viewports
            ),
            "Turn off the extra viewports, or clear the render graph.".to_string(),
        ));
    }
}

/// Runs the `validate_node` hook of every node family, in registry order, on each node of a known
/// type in document order (S51).
fn family_hooks(doc: &GraphDoc, ctx: &ValidateCtx<'_>, issues: &mut Vec<Issue>) {
    for family in crate::registry::families_all() {
        family.validate_graph(doc, issues);
        for node in &doc.nodes {
            if let Some(spec) = ctx.catalog.node(&node.type_id) {
                family.validate_node(node, spec, ctx, issues);
            }
        }
    }
}

/// `graph.budget` (the budget stage): each node's own `budgets` entry, one shader stage at a
/// time, against `ctx.limits`. Counts are never summed over nodes or stages.
fn check_budgets(doc: &GraphDoc, ctx: &ValidateCtx<'_>, issues: &mut Vec<Issue>) {
    let limits = ctx.limits;
    for node in &doc.nodes {
        let Some(spec) = ctx.catalog.node(&node.type_id) else {
            continue;
        };
        if let Some(decl) = node
            .compute
            .as_ref()
            .filter(|_| node.type_id == "compute.wgsl")
        {
            let budget = [
                (
                    "sampled textures",
                    decl.inputs
                        .iter()
                        .filter(|p| matches!(p.ty, PortType::Image2d | PortType::Depth2d))
                        .count() as u32,
                    limits.sampled_textures,
                ),
                ("samplers", 1, limits.samplers),
                (
                    "storage textures",
                    decl.outputs.len() as u32,
                    limits.storage_textures,
                ),
                (
                    "storage buffers",
                    decl.inputs
                        .iter()
                        .filter(|p| p.ty == PortType::Buffer)
                        .count() as u32,
                    limits.storage_buffers,
                ),
                ("uniform buffers", 2, limits.uniform_buffers),
            ];
            for (kind, count, limit) in budget {
                if count > limit {
                    let problem = format!(
                        "node `{}` binds {count} {kind} in its compute stage, and the limit is {limit}",
                        node.id
                    );
                    let fix = format!(
                        "Reduce the {kind} of node `{}` in its compute stage to {limit} or fewer.",
                        node.id
                    );
                    issues.push(Issue {
                        stage: Stage::Budget,
                        ..node_issue(node, "graph.budget", Severity::Error, problem, fix)
                    });
                }
            }
            continue;
        }
        for budget in &spec.budgets {
            let stage = stage_name(&budget.stage);
            let kinds = [
                (
                    "sampled textures",
                    budget.sampled_textures,
                    limits.sampled_textures,
                ),
                ("samplers", budget.samplers, limits.samplers),
                (
                    "storage textures",
                    budget.storage_textures,
                    limits.storage_textures,
                ),
                (
                    "storage buffers",
                    budget.storage_buffers,
                    limits.storage_buffers,
                ),
                (
                    "uniform buffers",
                    budget.uniform_buffers,
                    limits.uniform_buffers,
                ),
            ];
            for (kind, count, limit) in kinds {
                if count > limit {
                    let problem = format!(
                        "node `{}` binds {count} {kind} in its {stage} stage, and the limit is {limit}",
                        node.id
                    );
                    let fix = format!(
                        "Reduce the {kind} of node `{}` in its {stage} stage to {limit} or fewer.",
                        node.id
                    );
                    issues.push(Issue {
                        stage: Stage::Budget,
                        ..node_issue(node, "graph.budget", Severity::Error, problem, fix)
                    });
                }
            }
        }
    }
}

/// The message name of a shader stage.
fn stage_name(stage: &ShaderStage) -> &'static str {
    match stage {
        ShaderStage::Vertex => "vertex",
        ShaderStage::Fragment => "fragment",
        ShaderStage::Compute => "compute",
    }
}

/// Whether the structure stage has an error; warnings do not count (S52).
fn structure_failed(issues: &[Issue]) -> bool {
    issues
        .iter()
        .any(|issue| issue.stage == Stage::Structure && issue.severity == Severity::Error)
}
