//! The planned form of a render graph (S53): the executed nodes in order, their inputs, the
//! resources they produce and the histories they read, and the planner that builds it from a
//! validated document (`plan_from_doc`, `plan_from_json`).

use crate::catalog::{
    Catalog, GraphFormat, NodeTypeSpec, ParamBinding, ParamValue, PortSpec, PortType, SizeSpec,
};
use crate::error::{Issue, Severity, Stage, sort_issues};
use crate::format::{ComputePortDecl, GraphDoc, NodeDoc, PortRef, parse_graph};
use crate::mesh_indirect::input_defaults;
use crate::validate::{GraphLimits, ValidateCtx, validate};

/// A graph after planning: ordered nodes, links, resources and the displayed output.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphPlan {
    pub name: String,
    pub nodes: Vec<PlanNode>,
    pub links: Vec<PlanLink>,
    pub resources: Vec<PlanResource>,
    pub histories: Vec<PlanHistory>,
    pub output: PlanPortRef,
}

impl GraphPlan {
    /// Resolves the current-frame producer contract of a planned input by instance and port.
    /// Frozen plans retain the same endpoints and catalog type versions as editable plans.
    pub fn input_signal(
        &self,
        node: &PlanNode,
        input: &str,
    ) -> Option<&'static crate::catalog::SignalContract> {
        let source = &node.inputs.iter().find(|feed| feed.port == input)?.source;
        let (id, port) = match source {
            InputSource::Link { node, port, .. } | InputSource::Default { node, port } => {
                (node, port)
            }
            InputSource::Absent => return None,
        };
        let producer = self.nodes.iter().find(|node| node.id == *id)?;
        crate::catalog::port_signals(&producer.type_id, port, true).first()
    }
}

/// One executed node of a plan.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanNode {
    pub id: String,
    pub type_id: String,
    pub type_version: u32,
    pub order_index: u32,
    pub params: Vec<(String, ParamValue)>,
    pub inputs: Vec<PlanInput>,
    pub compute: Option<PlanCompute>,
}

/// How one input of a planned node is fed.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanInput {
    pub port: String,
    pub source: InputSource,
}

/// The source of a planned input.
#[derive(Clone, Debug, PartialEq)]
pub enum InputSource {
    Link {
        node: String,
        port: String,
        history: bool,
    },
    Default {
        node: String,
        port: String,
    },
    Absent,
}

/// One link of a plan.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanLink {
    pub from: PlanPortRef,
    pub to: PlanPortRef,
    pub history: bool,
}

/// A node id and port key of a plan.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanPortRef {
    pub node: String,
    pub port: String,
}

/// One resource produced by a planned node.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanResource {
    pub id: u32,
    pub producer: PlanPortRef,
    pub ty: PortType,
    pub format: Option<GraphFormat>,
    pub size: SizeSpec,
    pub mips: u32,
    pub layers: u32,
    pub first_use: u32,
    pub last_use: u32,
    pub history: bool,
}

/// A history link and the resource it reads.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanHistory {
    pub resource: u32,
    pub consumer: PlanPortRef,
}

/// The compute declaration of a planned `compute.wgsl` node.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanCompute {
    pub wgsl: String,
    pub inputs: Vec<ComputePortDecl>,
    pub outputs: Vec<ComputePortDecl>,
    pub workgroup: [u32; 2],
}

/// The node type that displays the graph; validation allows exactly one (S51).
const OUTPUT_TYPE: &str = "output.display";
/// Contributors are executed with this node type, and placed right after it (S51, S53).
const FORWARD_TYPE: &str = "pass.forward";
/// The workgroup size of every compute node (S59).
const COMPUTE_WORKGROUP: [u32; 2] = [8, 8];

/// Validates `doc` against `ctx`, then plans it (S53). Any error returns every issue, sorted. A
/// plan returns its warnings: the validation warnings and `graph.unreachable` for each node that
/// is left out.
pub fn plan_from_doc(
    doc: &GraphDoc,
    ctx: &ValidateCtx<'_>,
) -> Result<(GraphPlan, Vec<Issue>), Vec<Issue>> {
    let mut issues = validate(doc, ctx);
    if has_error(&issues) {
        return Err(issues);
    }
    issues.extend(engine_option_issues(doc, ctx));
    // A planner refusal pushes its error onto `issues` and returns no plan.
    let plan = plan_nodes(doc, ctx, &mut issues);
    sort_issues(&mut issues, Some(doc));
    match plan {
        Some(plan) if !has_error(&issues) => Ok((plan, issues)),
        _ => Err(issues),
    }
}

/// Parses `text`, then plans it with the built-in catalog, every pass installed, one viewport and
/// `msaa` 4 as the engine's option (S53). The warnings are dropped.
pub fn plan_from_json(text: &str) -> Result<GraphPlan, Vec<Issue>> {
    let doc = parse_graph(text)?;
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
    plan_from_doc(&doc, &ctx).map(|(plan, _)| plan)
}

/// Whether any issue is an error.
fn has_error(issues: &[Issue]) -> bool {
    issues.iter().any(|found| found.severity == Severity::Error)
}

/// A planning issue on `node` (when `Some`), with the given code, severity and stage.
fn issue(
    code: &'static str,
    severity: Severity,
    stage: Stage,
    node: Option<&str>,
    message: String,
    fix: String,
) -> Issue {
    Issue {
        code,
        severity,
        stage,
        node: node.map(str::to_string),
        port: None,
        param: None,
        link: None,
        message,
        fix,
    }
}

/// The engine's value of `option`, when the engine reports one.
fn engine_value(ctx: &ValidateCtx<'_>, option: &str) -> Option<ParamValue> {
    ctx.engine_options
        .iter()
        .find(|(name, _)| *name == option)
        .map(|(_, value)| value.clone())
}

/// A parameter value as the message and the fix write it.
fn shown(value: &ParamValue) -> String {
    match value {
        ParamValue::Number(n) => n.to_string(),
        ParamValue::Int(i) => i.to_string(),
        ParamValue::Bool(b) => b.to_string(),
        ParamValue::Enum(s) | ParamValue::String(s) => s.clone(),
        vector => format!("{vector:?}"),
    }
}

/// `graph.engine_option` for each Engine-bound parameter a node writes with a value other than
/// the engine's (amended rule (e)). An absent parameter takes the engine's value instead.
fn engine_option_issues(doc: &GraphDoc, ctx: &ValidateCtx<'_>) -> Vec<Issue> {
    let mut issues = Vec::new();
    for node in &doc.nodes {
        let Some(spec) = ctx.catalog.node(&node.type_id) else {
            continue;
        };
        for param in &spec.params {
            let ParamBinding::Engine { option } = &param.binding else {
                continue;
            };
            let Some((_, written)) = node
                .params
                .iter()
                .find(|(key, _)| key.as_str() == param.key)
            else {
                continue;
            };
            let Ok(value) = ParamValue::from_json(&param.kind, written) else {
                continue;
            };
            let engine = engine_value(ctx, option).unwrap_or_else(|| param.default.clone());
            if value != engine {
                issues.push(Issue {
                    param: Some(param.key.to_string()),
                    ..issue(
                        "graph.engine_option",
                        Severity::Error,
                        Stage::Plan,
                        Some(&node.id),
                        format!(
                            "node `{}` sets `{}` to {}, but the engine has {} for option `{option}`",
                            node.id,
                            param.key,
                            shown(&value),
                            shown(&engine)
                        ),
                        format!("create the engine with {option}: {}", shown(&value)),
                    )
                });
            }
        }
    }
    issues
}

/// Where an input of a document node reads from: a link (with its history flag, by node index),
/// its first default source present in the graph, or nothing.
enum Feed {
    Link {
        from: usize,
        port: String,
        history: bool,
    },
    Default {
        from: usize,
        port: &'static str,
    },
    Absent,
}

/// The feed of `port` on `node`: the link into it, else its first default source present in the
/// graph. A default source is the first node of its type, as validation reads it.
fn feed(doc: &GraphDoc, node: &NodeDoc, port: &PortSpec) -> Feed {
    if let Some(link) = doc
        .links
        .iter()
        .find(|link| link.to.node == node.id && link.to.port == port.key)
    {
        return match doc
            .nodes
            .iter()
            .position(|known| known.id == link.from.node)
        {
            Some(from) => Feed::Link {
                from,
                port: link.from.port.clone(),
                history: link.history,
            },
            None => Feed::Absent,
        };
    }
    input_defaults(doc, node, port)
        .iter()
        .copied()
        .find_map(|entry| {
            let (node_type, key) = entry.rsplit_once('.')?;
            let from = doc
                .nodes
                .iter()
                .position(|known| known.type_id == node_type)?;
            Some(Feed::Default { from, port: key })
        })
        .unwrap_or(Feed::Absent)
}

fn compute_feed(doc: &GraphDoc, node: &NodeDoc, key: &str) -> Feed {
    doc.links
        .iter()
        .find(|link| link.to.node == node.id && link.to.port == key)
        .map_or(Feed::Absent, |link| {
            doc.nodes
                .iter()
                .position(|known| known.id == link.from.node)
                .map_or(Feed::Absent, |from| Feed::Link {
                    from,
                    port: link.from.port.clone(),
                    history: link.history,
                })
        })
}

/// Marks `executed` for every node reached from `stack` through non-history links and default
/// sources.
fn reach(feeds: &[Vec<Feed>], executed: &mut [bool], mut stack: Vec<usize>) {
    while let Some(node) = stack.pop() {
        if std::mem::replace(&mut executed[node], true) {
            continue;
        }
        for feed in &feeds[node] {
            if let Feed::Link {
                from,
                history: false,
                ..
            }
            | Feed::Default { from, .. } = feed
            {
                stack.push(*from);
            }
        }
    }
}

/// The executed nodes (S53, step 1): those reached backwards from `output.display` and from every
/// side-effect node. Each contributor is added, with its inputs, while a `pass.forward` is executed.
fn executed_set(doc: &GraphDoc, specs: &[&NodeTypeSpec], feeds: &[Vec<Feed>]) -> Vec<bool> {
    let n = doc.nodes.len();
    let mut executed = vec![false; n];
    let mut stack: Vec<usize> = (0..n)
        .filter(|&i| doc.nodes[i].type_id == OUTPUT_TYPE || specs[i].side_effect)
        .collect();
    loop {
        reach(feeds, &mut executed, stack);
        let forward_runs = (0..n).any(|i| executed[i] && doc.nodes[i].type_id == FORWARD_TYPE);
        stack = if forward_runs {
            (0..n)
                .filter(|&i| specs[i].contributor && !executed[i])
                .collect()
        } else {
            Vec::new()
        };
        if stack.is_empty() {
            return executed;
        }
    }
}

/// The source a planned input reads. A history link from a node that is not executed has no
/// resource, so the input reads nothing.
fn input_source(doc: &GraphDoc, feed: &Feed, executed: &[bool]) -> InputSource {
    match feed {
        Feed::Link {
            from,
            port,
            history,
        } if executed[*from] => InputSource::Link {
            node: doc.nodes[*from].id.clone(),
            port: port.clone(),
            history: *history,
        },
        Feed::Default { from, port } => InputSource::Default {
            node: doc.nodes[*from].id.clone(),
            port: port.to_string(),
        },
        Feed::Link { .. } | Feed::Absent => InputSource::Absent,
    }
}

/// A plan port reference naming the same node and port as a document port reference.
fn port_ref(end: &PortRef) -> PlanPortRef {
    PlanPortRef {
        node: end.node.clone(),
        port: end.port.clone(),
    }
}

/// The parameters of `node` in catalog key order (S53, step 3): the document's values, converted
/// and upgraded to the catalog version (S50), else the default. An Engine-bound parameter takes
/// the engine's value.
fn plan_params(
    node: &NodeDoc,
    spec: &NodeTypeSpec,
    ctx: &ValidateCtx<'_>,
    issues: &mut Vec<Issue>,
) -> Option<Vec<(String, ParamValue)>> {
    if node.compute.is_some() {
        return Some(
            node.params
                .iter()
                .filter_map(|(key, value)| infer_compute_param(key, value))
                .collect(),
        );
    }
    let mut given: Vec<(String, ParamValue)> = node
        .params
        .iter()
        .filter_map(|(key, value)| {
            let param = spec.params.iter().find(|param| param.key == key.as_str())?;
            Some((key.clone(), ParamValue::from_json(&param.kind, value).ok()?))
        })
        .collect();
    let written = node.version.unwrap_or(spec.version);
    for step in written..spec.version {
        if let Err(reason) = (spec.upgrade)(step, &mut given) {
            issues.push(issue(
                "graph.node_version",
                Severity::Error,
                Stage::Structure,
                Some(&node.id),
                format!(
                    "node `{}` is v{written}, and its upgrade from v{step} failed: {reason}",
                    node.id
                ),
                format!(
                    "Set `v` of node `{}` to {} and write its params for that version.",
                    node.id, spec.version
                ),
            ));
            return None;
        }
    }
    Some(
        spec.params
            .iter()
            .map(|param| {
                let value = match &param.binding {
                    ParamBinding::Engine { option } => engine_value(ctx, option),
                    _ => given
                        .iter()
                        .find(|(key, _)| key.as_str() == param.key)
                        .map(|(_, value)| value.clone()),
                };
                (
                    param.key.to_string(),
                    value.unwrap_or_else(|| param.default.clone()),
                )
            })
            .collect(),
    )
}

fn infer_compute_param(
    key: &str,
    value: &raa_loader::json::JsonValue,
) -> Option<(String, ParamValue)> {
    use raa_loader::json::JsonValue;
    let value = match value {
        JsonValue::Number(number) => ParamValue::Number(*number),
        JsonValue::Bool(value) => ParamValue::Bool(*value),
        JsonValue::Array(values) if (2..=4).contains(&values.len()) => {
            let numbers: Vec<f64> = values
                .iter()
                .map(|value| match value {
                    JsonValue::Number(n) => Some(*n),
                    _ => None,
                })
                .collect::<Option<_>>()?;
            match numbers.as_slice() {
                [x, y] => ParamValue::Vec2([*x, *y]),
                [x, y, z] => ParamValue::Vec3([*x, *y, *z]),
                [x, y, z, w] => ParamValue::Vec4([*x, *y, *z, *w]),
                _ => return None,
            }
        }
        _ => return None,
    };
    Some((key.to_string(), value))
}

/// The order of the executed nodes: Kahn's algorithm over the non-history links and default
/// sources, taking the lowest ready node by (default order, document index). A contributor waits
/// for `pass.forward`, then goes ahead of the other ready nodes, so it sits right after it.
fn execution_order(
    doc: &GraphDoc,
    specs: &[&NodeTypeSpec],
    feeds: &[Vec<Feed>],
    executed: &[bool],
    issues: &mut Vec<Issue>,
) -> Option<Vec<usize>> {
    let n = doc.nodes.len();
    let exec: Vec<usize> = (0..n).filter(|&i| executed[i]).collect();
    let forward = exec
        .iter()
        .copied()
        .find(|&i| doc.nodes[i].type_id == FORWARD_TYPE);
    let mut indegree = vec![0usize; n];
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &v in &exec {
        for feed in &feeds[v] {
            if let Feed::Link {
                from,
                history: false,
                ..
            }
            | Feed::Default { from, .. } = feed
            {
                successors[*from].push(v);
                indegree[v] += 1;
            }
        }
    }
    let mut placed = vec![false; n];
    let mut order = Vec::with_capacity(exec.len());
    while order.len() < exec.len() {
        let forward_placed = forward.is_some_and(|at| placed[at]);
        // Group 0: a contributor once forward is placed. Group 1: every other node. Group 2: a
        // contributor before forward, which only happens when forward depends on it.
        let rank = |v: usize| {
            let group: u8 = match (specs[v].contributor && forward.is_some(), forward_placed) {
                (false, _) => 1,
                (true, true) => 0,
                (true, false) => 2,
            };
            (group, specs[v].default_order, v)
        };
        let next = exec
            .iter()
            .copied()
            .filter(|&v| !placed[v] && indegree[v] == 0)
            .min_by_key(|&v| rank(v));
        let Some(next) = next else {
            let first = exec
                .iter()
                .copied()
                .find(|&v| !placed[v])
                .and_then(|v| doc.nodes.get(v));
            issues.push(issue(
                "graph.cycle",
                Severity::Error,
                Stage::Structure,
                first.map(|node| node.id.as_str()),
                "the executed nodes read each other, and no history link breaks that cycle"
                    .to_string(),
                "Set `history` on one link inside the cycle.".to_string(),
            ));
            return None;
        };
        placed[next] = true;
        order.push(next);
        for &after in &successors[next] {
            indegree[after] = indegree[after].saturating_sub(1);
        }
    }
    Some(order)
}

/// The resources (S53): one per output port of an executed node that feeds an input of an
/// executed node, numbered from 1 in first-use order. A history reader extends the lifetime to the
/// end of the frame.
fn planned_resources(
    doc: &GraphDoc,
    specs: &[&NodeTypeSpec],
    feeds: &[Vec<Feed>],
    order: &[usize],
    order_index: &[u32],
) -> Vec<PlanResource> {
    let mut resources: Vec<PlanResource> = Vec::new();
    for &producer in order {
        for out in output_descriptors(&doc.nodes[producer], specs[producer]) {
            // The order index and history flag of each input that reads this output port.
            let mut uses: Vec<(u32, bool)> = Vec::new();
            for &consumer in order {
                for feed in &feeds[consumer] {
                    match feed {
                        Feed::Link {
                            from,
                            port,
                            history,
                        } if *from == producer && port.as_str() == out.key => {
                            uses.push((order_index[consumer], *history));
                        }
                        Feed::Default { from, port } if *from == producer && *port == out.key => {
                            uses.push((order_index[consumer], false));
                        }
                        _ => {}
                    }
                }
            }
            if uses.is_empty() {
                continue;
            }
            let history = uses.iter().any(|&(_, history)| history);
            let last_use = if history {
                u32::MAX
            } else {
                uses.iter()
                    .map(|&(at, _)| at)
                    .max()
                    .unwrap_or(order_index[producer])
            };
            resources.push(PlanResource {
                id: u32::try_from(resources.len() + 1).unwrap_or(u32::MAX),
                producer: PlanPortRef {
                    node: doc.nodes[producer].id.clone(),
                    port: out.key.clone(),
                },
                ty: out.ty.clone(),
                format: out.format.clone(),
                size: out.size.clone(),
                mips: out.mips,
                layers: out.layers,
                first_use: order_index[producer],
                last_use,
                history,
            });
        }
    }
    resources
}

struct OutputDesc {
    key: String,
    ty: PortType,
    format: Option<GraphFormat>,
    size: SizeSpec,
    mips: u32,
    layers: u32,
}

fn output_descriptors(node: &NodeDoc, spec: &NodeTypeSpec) -> Vec<OutputDesc> {
    if let Some(decl) = &node.compute {
        return decl
            .outputs
            .iter()
            .map(|port| OutputDesc {
                key: port.key.clone(),
                ty: port.ty.clone(),
                format: port.format.clone(),
                size: port.size.clone().unwrap_or(SizeSpec::Scale(1.0)),
                mips: 1,
                layers: 1,
            })
            .collect();
    }
    spec.outputs
        .iter()
        .map(|port| OutputDesc {
            key: port.key.to_string(),
            ty: port.ty.clone(),
            format: port.format.clone(),
            size: port.size.clone(),
            mips: port.mips,
            layers: port.layers,
        })
        .collect()
}

/// Plans the executed nodes of a validated `doc`, pushing a warning for each node left out. When a
/// planning error stops the plan (a node type, an upgrade, a stalled order, no colour source), it
/// is pushed onto `issues` and no plan is returned.
fn plan_nodes(doc: &GraphDoc, ctx: &ValidateCtx<'_>, issues: &mut Vec<Issue>) -> Option<GraphPlan> {
    let mut specs: Vec<&NodeTypeSpec> = Vec::with_capacity(doc.nodes.len());
    for node in &doc.nodes {
        let Some(spec) = ctx.catalog.node(&node.type_id) else {
            issues.push(issue(
                "graph.node_type",
                Severity::Error,
                Stage::Structure,
                Some(&node.id),
                format!(
                    "node `{}` has type `{}`, which the catalog does not define",
                    node.id, node.type_id
                ),
                format!(
                    "Change the type of node `{}` to a catalog node type, or delete the node.",
                    node.id
                ),
            ));
            return None;
        };
        specs.push(spec);
    }
    let feeds: Vec<Vec<Feed>> = doc
        .nodes
        .iter()
        .zip(&specs)
        .map(|(node, spec)| {
            if let Some(decl) = &node.compute {
                return decl
                    .inputs
                    .iter()
                    .map(|port| compute_feed(doc, node, &port.key))
                    .collect();
            }
            spec.inputs
                .iter()
                .map(|port| feed(doc, node, port))
                .collect()
        })
        .collect();
    let executed = executed_set(doc, &specs, &feeds);
    for (node, _) in doc.nodes.iter().zip(&executed).filter(|(_, run)| !**run) {
        issues.push(issue(
            "graph.unreachable",
            Severity::Warning,
            Stage::Plan,
            Some(&node.id),
            format!(
                "node `{}` reaches neither `{OUTPUT_TYPE}` nor a side-effect node, so it is not executed",
                node.id
            ),
            format!(
                "Link an output of node `{}` towards `{OUTPUT_TYPE}`, or delete node `{}`.",
                node.id, node.id
            ),
        ));
    }
    let order = execution_order(doc, &specs, &feeds, &executed, issues)?;
    let mut order_index = vec![0u32; doc.nodes.len()];
    for (at, &v) in order.iter().enumerate() {
        order_index[v] = u32::try_from(at).unwrap_or(u32::MAX);
    }
    let resources = planned_resources(doc, &specs, &feeds, &order, &order_index);
    let mut nodes = Vec::with_capacity(order.len());
    for &v in &order {
        let (node, spec) = (&doc.nodes[v], specs[v]);
        nodes.push(PlanNode {
            id: node.id.clone(),
            type_id: node.type_id.clone(),
            type_version: spec.version,
            order_index: order_index[v],
            params: plan_params(node, spec, ctx, issues)?,
            inputs: node
                .compute
                .as_ref()
                .map_or_else(
                    || {
                        spec.inputs
                            .iter()
                            .map(|port| port.key.to_string())
                            .collect::<Vec<_>>()
                    },
                    |decl| {
                        decl.inputs
                            .iter()
                            .map(|port| port.key.clone())
                            .collect::<Vec<_>>()
                    },
                )
                .into_iter()
                .zip(&feeds[v])
                .map(|(port, feed)| PlanInput {
                    port,
                    source: input_source(doc, feed, &executed),
                })
                .collect(),
            compute: node.compute.as_ref().map(|decl| PlanCompute {
                wgsl: decl.wgsl.clone(),
                inputs: decl.inputs.clone(),
                outputs: decl.outputs.clone(),
                workgroup: COMPUTE_WORKGROUP,
            }),
        });
    }
    let is_planned = |id: &str| {
        doc.nodes
            .iter()
            .position(|node| node.id == id)
            .is_some_and(|at| executed[at])
    };
    let links = doc
        .links
        .iter()
        .filter(|link| is_planned(&link.from.node) && is_planned(&link.to.node))
        .map(|link| PlanLink {
            from: port_ref(&link.from),
            to: port_ref(&link.to),
            history: link.history,
        })
        .collect();
    let mut histories = Vec::new();
    for node in &nodes {
        for input in &node.inputs {
            let InputSource::Link {
                node: from,
                port,
                history: true,
            } = &input.source
            else {
                continue;
            };
            if let Some(resource) = resources
                .iter()
                .find(|resource| resource.producer.node == *from && resource.producer.port == *port)
            {
                histories.push(PlanHistory {
                    resource: resource.id,
                    consumer: PlanPortRef {
                        node: node.id.clone(),
                        port: input.port.clone(),
                    },
                });
            }
        }
    }
    let display = nodes.iter().find(|node| node.type_id == OUTPUT_TYPE);
    let output = display
        .and_then(|node| node.inputs.iter().find(|input| input.port == "color"))
        .and_then(|input| match &input.source {
            InputSource::Link { node, port, .. } | InputSource::Default { node, port } => {
                Some(PlanPortRef {
                    node: node.clone(),
                    port: port.clone(),
                })
            }
            InputSource::Absent => None,
        });
    let Some(output) = output else {
        issues.push(issue(
            "graph.input_missing",
            Severity::Error,
            Stage::Structure,
            display.map(|node| node.id.as_str()),
            format!(
                "input `{OUTPUT_TYPE}.color` has no link, and none of its default sources is in the graph"
            ),
            format!("Link an output into the `color` input of `{OUTPUT_TYPE}`."),
        ));
        return None;
    };
    Some(GraphPlan {
        name: doc.name.clone(),
        nodes,
        links,
        resources,
        histories,
        output,
    })
}
