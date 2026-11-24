//! Deterministic, device-free helpers used by the post-stage executor.

use super::*;

/// Handoff: card 'post_stages extension 1/2: install, values, cameras and stats' calls
/// `stage_nodes`, `stage_values`, `karis_nodes` and `bloom_composites`; card
/// 'post_stages extension 2/2: graph frame binding inputs, bloom composite chain and view aliases'
/// calls `stage_bindings`, `bloom_composites` and `view_aliases`. Today this function returns
/// post-stage (node id, type id, entry) tuples in plan order. Upstream `node_entries` at
/// `crates/raa-graph/src/exec_ext/wiring/plan_fns.rs:152` supplies the corresponding pipeline entries.
pub fn stage_nodes(plan: &GraphPlan) -> Vec<(String, String, String)> {
    plan.nodes
        .iter()
        .filter(|node| POST_STAGE_TYPES.contains(&node.type_id.as_str()))
        .map(|node| {
            (
                node.id.clone(),
                node.type_id.clone(),
                format!("{}#{}", node.type_id, node.id),
            )
        })
        .collect()
}

pub const POST_STAGE_TYPES: [&str; 9] = [
    "ao.trace",
    "ao.blur",
    "ssr.hiz",
    "ssr.trace",
    "ssr.composite",
    "ssr.incident",
    "ssr.replace",
    "bloom.down",
    "bloom.up",
];

pub fn stage_values(node: &PlanNode, spec: &NodeTypeSpec) -> Vec<(String, f64)> {
    node.params
        .iter()
        .filter_map(|(key, value)| {
            let param = spec.params.iter().find(|param| param.key == key)?;
            if param.binding != ParamBinding::Node {
                return None;
            }
            let value = match value {
                ParamValue::Bool(value) => {
                    if *value {
                        1.0
                    } else {
                        0.0
                    }
                }
                ParamValue::Int(value) => *value as f64,
                ParamValue::Number(value) => *value,
                ParamValue::Enum(value) => {
                    param.options.iter().position(|option| *option == value)? as f64
                }
                _ => return None,
            };
            Some((key.clone(), value))
        })
        .collect()
}

pub fn karis_nodes(plan: &GraphPlan) -> Vec<String> {
    plan.nodes
        .iter()
        .filter(|node| node.type_id == "bloom.down")
        .filter(|node| {
            node.inputs.iter().any(|input| {
                input.port == "src"
                    && matches!(&input.source, InputSource::Link { node: source, .. }
                        if plan.nodes.iter().find(|n| n.id == *source)
                            .is_some_and(|n| !matches!(n.type_id.as_str(), "bloom.down" | "bloom.up")))
            })
        })
        .map(|node| node.id.clone())
        .collect()
}

pub fn bloom_composites(
    plan: &GraphPlan,
    entries: &[(String, String)],
) -> Vec<(String, String, String)> {
    plan.nodes
        .iter()
        .filter(|node| node.type_id == "bloom.up")
        .filter_map(|node| {
            let input = node.inputs.iter().find(|input| input.port == "color")?;
            let InputSource::Link {
                node: source, port, ..
            } = &input.source
            else {
                return None;
            };
            Some((
                entry(entries, &node.id)?,
                entry(entries, source)?,
                port.clone(),
            ))
        })
        .collect()
}

/// Additional rays share the first ray's exact trace node and its input bindings.
pub fn incident_trace<'a>(plan: &'a GraphPlan, node: &PlanNode) -> Option<&'a PlanNode> {
    if node.type_id != "ssr.incident" || !node.params.iter().any(|(key, value)| {
        key == "extraRays" && matches!(value, ParamValue::Int(count) if *count > 0)
    }) {
        return None;
    }
    let input = node.inputs.iter().find(|input| input.port == "trace")?;
    let InputSource::Link { node: source, history: false, .. } = &input.source else {
        return None;
    };
    plan.nodes.iter().find(|node| node.id == *source && node.type_id == "ssr.trace")
}

pub fn stage_bindings(
    plan: &GraphPlan,
    catalog: &Catalog,
    entries: &[(String, String)],
) -> Vec<(String, &'static str, String, String)> {
    let mut result = Vec::new();
    for node in &plan.nodes {
        if !POST_STAGE_TYPES.contains(&node.type_id.as_str()) {
            continue;
        }
        let Some(spec) = catalog.node(&node.type_id) else {
            continue;
        };
        let Some(consumer_entry) = entry(entries, &node.id) else {
            continue;
        };
        for port_spec in &spec.inputs {
            let Some(input) = node.inputs.iter().find(|input| input.port == port_spec.key) else {
                continue;
            };
            if matches!(
                node.type_id.as_str(),
                "ssr.composite" | "ssr.replace" | "bloom.up"
            ) && input.port == "color"
            {
                continue;
            }
            let (source, port) = match &input.source {
                InputSource::Link {
                    node,
                    port,
                    history: false,
                }
                | InputSource::Default { node, port } => (node, port),
                _ => continue,
            };
            let Some(source_entry) = entry(entries, source) else {
                continue;
            };
            result.push((
                consumer_entry.clone(),
                port_spec.key,
                source_entry,
                port.clone(),
            ));
        }
    }
    // These inputs are inherited from the typed first-ray route, so a custom prepass or
    // depth pyramid cannot accidentally send the additional rays through another scene view.
    for incident in &plan.nodes {
        let Some(trace) = incident_trace(plan, incident) else { continue };
        let Some(consumer) = entry(entries, &incident.id) else { continue };
        let Some(producer) = entry(entries, &trace.id) else { continue };
        for port in ["depth", "normal", "hiz"] {
            if let Some((_, _, source, source_port)) = result.iter().find(|(node, key, _, _)| {
                node == &producer && *key == port
            }) {
                result.push((consumer.clone(), port, source.clone(), source_port.clone()));
            }
        }
    }
    result
}

pub fn view_aliases(plan: &GraphPlan) -> Vec<(String, String)> {
    let mut linked_types = Vec::new();
    for view in plan
        .nodes
        .iter()
        .filter(|node| node.type_id.starts_with("view."))
    {
        for input in &view.inputs {
            let InputSource::Link { node: source, .. } = &input.source else {
                continue;
            };
            let Some(stage) = plan.nodes.iter().find(|node| {
                node.id == *source && POST_STAGE_TYPES.contains(&node.type_id.as_str())
            }) else {
                continue;
            };
            if !linked_types.contains(&stage.type_id) {
                linked_types.push(stage.type_id.clone());
            }
        }
    }
    let mut aliases = Vec::new();
    for stage in plan.nodes.iter().filter(|node| {
        POST_STAGE_TYPES.contains(&node.type_id.as_str()) && linked_types.contains(&node.type_id)
    }) {
        if !aliases.iter().any(|(_, ty)| ty == &stage.type_id) {
            aliases.push((
                format!("{}#{}", stage.type_id, stage.id),
                stage.type_id.clone(),
            ));
        }
    }
    aliases
}

fn entry(entries: &[(String, String)], node: &str) -> Option<String> {
    entries
        .iter()
        .find(|(id, _)| id == node)
        .map(|(_, entry)| entry.clone())
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::plan_from_json;

    const GRAPH: &str = include_str!("../../../graphs/real-frame-model.json");

    #[test]
    fn extra_ssr_rays_reuse_the_selected_trace_route() {
        let result = GRAPH.replace(
            r#""type": "ssr.incident""#,
            r#""type": "ssr.incident", "params": { "extraRays": 3 }"#,
        );
        let catalog = Catalog::build();
        for (text, expected) in [(GRAPH, 0), (&result, 3)] {
            let plan = plan_from_json(text).expect("real-frame graph");
            let incident = plan.nodes.iter().find(|node| node.id == "incident").unwrap();
            assert!(incident.params.contains(&("extraRays".into(), ParamValue::Int(expected))));
            let entries = plan_fns::node_entries(&plan, &catalog);
            let binds = stage_bindings(&plan, &catalog, &entries);
            for port in ["depth", "normal", "hiz"] {
                let source = |entry: &str| binds.iter().find_map(|(node, key, source, output)| {
                    (node == entry && *key == port).then_some((source, output))
                });
                let trace = source("ssr.trace#ssrTrace").expect("linked first-ray input");
                assert_eq!(source("ssr.incident#incident"), (expected > 0).then_some(trace));
            }
        }
        let missing_trace = result.replace(r#""ssrTrace.trace""#, r#""forward.color""#);
        assert!(plan_from_json(&missing_trace).unwrap_err().iter().any(|issue| {
            issue.code == "graph.ssr_extra_rays_source"
        }));
    }
}
