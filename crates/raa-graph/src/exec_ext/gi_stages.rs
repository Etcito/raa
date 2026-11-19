//! Device-free plan mapping for GI stage nodes and the placeholder executor extension.
#![allow(dead_code)] // The next extension share consumes these plan helpers.

use crate::catalog::{Catalog, NodeTypeSpec, ParamValue, PortType};
use crate::error::{Issue, Severity, Stage};
use crate::exec::{ExecExtension, FeatureHandles};
use crate::plan::{GraphPlan, InputSource, PlanNode};
use raa_gi::stages::{GiStageKind, stage_entry};

/// The extension currently has no device work; `apply` succeeds and `clear` does nothing today.
pub fn extension() -> Box<dyn ExecExtension> {
    Box::new(GiStagesExtension)
}

struct GiStagesExtension;

impl ExecExtension for GiStagesExtension {
    fn name(&self) -> &'static str {
        "gi_stages"
    }
    fn apply(
        &mut self,
        _: &GraphPlan,
        _: Option<&GraphPlan>,
        _: &mut FeatureHandles<'_>,
    ) -> Result<(), Vec<Issue>> {
        Ok(())
    }
    fn clear(&mut self, _: &mut FeatureHandles<'_>) {}
}

/// Handoff: `stage_nodes` supplies stage identities to 'raa-graph exec gi_stages 2/3: install, attach, values, refusals and stats' and 'raa-graph exec gi_stages 3/3: graph frame binding stage inputs, executor-owned histories and forward slots'. It returns the plan-ordered `(node id, GiStageKind)` pairs; this file also provides their entries, bindings, restarts, slots, values and accumulate validation. Upstream `GiStageKind::from_type_id` and `stage_entry` are at `crates/raa-gi/src/stages.rs:41 GiStageKind` and `crates/raa-gi/src/stages.rs:133 stage_entry`.
pub(crate) fn stage_nodes(plan: &GraphPlan) -> Vec<(String, GiStageKind)> {
    plan.nodes
        .iter()
        .filter_map(|n| GiStageKind::from_type_id(&n.type_id).map(|k| (n.id.clone(), k)))
        .collect()
}

pub(crate) fn stage_entries(plan: &GraphPlan, catalog: &Catalog) -> Vec<(String, String)> {
    let mut seen: Vec<String> = Vec::new();
    plan.nodes
        .iter()
        .filter_map(|n| {
            let kind = GiStageKind::from_type_id(&n.type_id);
            let spec = catalog.node(&n.type_id)?;
            let entry = if let Some(kind) = kind {
                stage_entry(kind, &n.id)
            } else if n.type_id.starts_with("view.") {
                format!("view#{}", n.id)
            } else {
                if spec.contributor {
                    return None;
                }
                let pass = spec.pass?;
                let duplicate = spec.multi_instance && seen.contains(&n.type_id);
                seen.push(n.type_id.clone());
                if duplicate {
                    format!("{pass}#{}", n.id)
                } else {
                    pass.to_owned()
                }
            };
            Some((n.id.clone(), entry))
        })
        .collect()
}

fn entry(entries: &[(String, String)], id: &str) -> Option<String> {
    entries
        .iter()
        .find(|(node, _)| node == id)
        .map(|(_, e)| e.clone())
}
fn stage_id(id: &str) -> bool {
    GiStageKind::from_type_id(id).is_some()
}
fn bindings(
    plan: &GraphPlan,
    catalog: &Catalog,
    entries: &[(String, String)],
    buffers: bool,
) -> Vec<(String, &'static str, String, String)> {
    let mut out = Vec::new();
    for node in &plan.nodes {
        if !stage_id(&node.type_id) {
            continue;
        }
        let (Some(consumer), Some(spec)) = (entry(entries, &node.id), catalog.node(&node.type_id))
        else {
            continue;
        };
        for port in &spec.inputs {
            if matches!(port.ty, PortType::Buffer) != buffers {
                continue;
            }
            let source = node
                .inputs
                .iter()
                .find(|i| i.port == port.key)
                .and_then(|i| match &i.source {
                    InputSource::Link {
                        node,
                        port,
                        history: false,
                    }
                    | InputSource::Default { node, port } => Some((node, port)),
                    _ => None,
                });
            if let Some((source_node, source_port)) = source
                && let Some(source_entry) = entry(entries, source_node)
            {
                out.push((
                    consumer.clone(),
                    port.key,
                    source_entry,
                    source_port.clone(),
                ));
            }
        }
    }
    out
}
pub(crate) fn image_bindings(
    p: &GraphPlan,
    c: &Catalog,
    e: &[(String, String)],
) -> Vec<(String, &'static str, String, String)> {
    bindings(p, c, e, false)
}
pub(crate) fn buffer_sources(
    p: &GraphPlan,
    c: &Catalog,
    e: &[(String, String)],
) -> Vec<(String, &'static str, String, String)> {
    bindings(p, c, e, true)
}

pub(crate) fn stage_histories(
    plan: &GraphPlan,
    catalog: &Catalog,
    entries: &[(String, String)],
) -> Vec<(String, &'static str, String, String)> {
    plan.histories
        .iter()
        .filter_map(|h| {
            let consumer_node = plan.nodes.iter().find(|n| n.id == h.consumer.node)?;
            if !stage_id(&consumer_node.type_id) {
                return None;
            }
            let port = catalog
                .node(&consumer_node.type_id)?
                .inputs
                .iter()
                .find(|p| p.key == h.consumer.port)?
                .key;
            let producer = plan
                .resources
                .iter()
                .find(|r| r.id == h.resource)?
                .producer
                .clone();
            Some((
                entry(entries, &h.consumer.node)?,
                port,
                entry(entries, &producer.node)?,
                producer.port,
            ))
        })
        .collect()
}

pub(crate) fn restarted(previous: Option<&GraphPlan>, next: &GraphPlan) -> Vec<String> {
    next.nodes
        .iter()
        .filter(|n| {
            stage_id(&n.type_id)
                && previous
                    .and_then(|p| p.nodes.iter().find(|o| o.id == n.id))
                    .is_none_or(|o| o.type_id != n.type_id || o.params != n.params)
        })
        .map(|n| n.id.clone())
        .collect()
}
pub(crate) fn history_restarts(previous: Option<&GraphPlan>, next: &GraphPlan) -> Vec<String> {
    let changed = restarted(previous, next);
    next.histories
        .iter()
        .filter_map(|h| {
            let producer = next
                .resources
                .iter()
                .find(|r| r.id == h.resource)?
                .producer
                .clone();
            if !stage_id(&next.nodes.iter().find(|n| n.id == h.consumer.node)?.type_id) {
                return None;
            }
            let old_source = previous
                .and_then(|p| p.histories.iter().find(|x| x.consumer == h.consumer))
                .and_then(|x| previous?.resources.iter().find(|r| r.id == x.resource))
                .map(|r| &r.producer);
            (changed.contains(&h.consumer.node)
                || changed.contains(&producer.node)
                || old_source != Some(&producer))
            .then(|| format!("{}.{}", h.consumer.node, h.consumer.port))
        })
        .collect()
}
pub(crate) fn slot_sources(
    plan: &GraphPlan,
    catalog: &Catalog,
    entries: &[(String, String)],
) -> Vec<(u32, String, String)> {
    let mut out = Vec::new();
    for node in &plan.nodes {
        if node.type_id != "pass.forward" {
            continue;
        }
        let Some(spec) = catalog.node(&node.type_id) else {
            continue;
        };
        for input in &node.inputs {
            let Some(slot) = spec
                .inputs
                .iter()
                .find(|p| p.key == input.port)
                .and_then(|p| p.slot)
            else {
                continue;
            };
            if let InputSource::Link {
                node: src, port, ..
            } = &input.source
                && stage_id(
                    plan.nodes
                        .iter()
                        .find(|n| n.id == *src)
                        .map_or("", |n| n.type_id.as_str()),
                )
                && let Some(e) = entry(entries, src)
            {
                out.push((slot, e, port.clone()));
            }
        }
    }
    out
}
pub(crate) fn stage_values(node: &PlanNode, spec: &NodeTypeSpec) -> Vec<(String, f64)> {
    node.params
        .iter()
        .filter_map(|(key, value)| {
            let param = spec.params.iter().find(|p| p.key == key)?;
            let number = match value {
                ParamValue::Bool(v) => f64::from(*v),
                ParamValue::Int(v) => *v as f64,
                ParamValue::Number(v) => *v,
                ParamValue::Enum(v) => {
                    param.options.iter().position(|x| *x == v).unwrap_or(0) as f64
                }
                _ => return None,
            };
            Some((key.clone(), number))
        })
        .collect()
}
pub(crate) fn accumulate_issues(plan: &GraphPlan) -> Vec<Issue> {
    plan.nodes
        .iter()
        .filter(|n| n.type_id == "gi.accumulate")
        .filter_map(|n| {
            let linked = |key| {
                n.inputs
                    .iter()
                    .any(|i| i.port == key && matches!(i.source, InputSource::Link { .. }))
            };
            let (code, port, message, fix) = match (linked("probes"), linked("irradiance")) {
                (false, false) => (
                    "graph.input_missing",
                    "probes",
                    "GI accumulate needs exactly one of probes or irradiance.",
                    "Link either probes or irradiance.",
                ),
                (true, true) => (
                    "graph.input_taken",
                    "irradiance",
                    "GI accumulate has both probes and irradiance linked.",
                    "Remove the irradiance link or the probes link.",
                ),
                _ => return None,
            };
            Some(Issue {
                code,
                severity: Severity::Error,
                stage: Stage::Apply,
                node: Some(n.id.clone()),
                port: Some(port.into()),
                param: None,
                link: None,
                message: message.into(),
                fix: fix.into(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::plan_from_json;

    fn plan(nodes: &str, links: &str) -> GraphPlan {
        let text = format!(
            r#"{{"format":"raa.render-graph","version":1,"name":"test","nodes":[{{"id":"pre","type":"pass.prepass"}},{{"id":"forward","type":"pass.forward"}},{{"id":"tone","type":"pass.tonemap"}},{{"id":"out","type":"output.display"}}{nodes}],"links":[{{"from":"forward.color","to":"tone.color"}},{{"from":"tone.color","to":"out.color"}}{links}]}}"#
        );
        plan_from_json(&text).unwrap()
    }
    fn entry_pairs(p: &GraphPlan, c: &Catalog) -> Vec<(String, String)> {
        stage_entries(p, c)
    }

    #[test]
    fn device_free_stage_plan_functions() {
        let c = Catalog::build();
        let p = plan(
            r#",{"id":"a1","type":"gi.atrous"},{"id":"a2","type":"gi.atrous"},{"id":"p","type":"gi.pyramid"},{"id":"acc","type":"gi.accumulate"},{"id":"pr","type":"gi.probes"},{"id":"r","type":"gi.resolve"}"#,
            r#",{"from":"p.pyramid","to":"pr.pyramid"},{"from":"pr.probesFiltered","to":"acc.probes"},{"from":"acc.output","to":"acc.history","history":true},{"from":"acc.output","to":"a1.src"},{"from":"a1.dst","to":"a2.src"},{"from":"a2.dst","to":"r.irradiance"},{"from":"r.gi","to":"forward.gi"}"#,
        );
        let es = entry_pairs(&p, &c);
        assert!(es.contains(&("a1".into(), "gi.atrous#a1".into())));
        assert!(es.contains(&("pre".into(), "prepass".into())));
        assert!(!es.iter().any(|(id, _)| id == "out"));
        assert!(image_bindings(&p, &c, &es).contains(&(
            "gi.atrous#a2".into(),
            "src",
            "gi.atrous#a1".into(),
            "dst".into()
        )));
        assert!(image_bindings(&p, &c, &es).contains(&(
            "gi.probes#pr".into(),
            "depth",
            "prepass".into(),
            "depth".into()
        )));
        assert!(buffer_sources(&p, &c, &es).contains(&(
            "gi.accumulate#acc".into(),
            "probes",
            "gi.probes#pr".into(),
            "probesFiltered".into()
        )));
        assert!(stage_histories(&p, &c, &es).contains(&(
            "gi.accumulate#acc".into(),
            "history",
            "gi.accumulate#acc".into(),
            "output".into()
        )));
        assert!(slot_sources(&p, &c, &es).contains(&(14, "gi.resolve#r".into(), "gi".into())));
        let mut changed = p.clone();
        let atrous = changed.nodes.iter_mut().find(|n| n.id == "a1").unwrap();
        atrous
            .params
            .iter_mut()
            .find(|(k, _)| k == "step")
            .unwrap()
            .1 = ParamValue::Int(2);
        assert_eq!(restarted(None, &p).len(), 6);
        assert_eq!(history_restarts(None, &p), vec!["acc.history"]);
        assert!(restarted(Some(&p), &p).is_empty());
        assert_eq!(restarted(Some(&p), &changed), vec!["a1"]);
        assert!(history_restarts(Some(&p), &changed).is_empty());
        let mut history_changed = p.clone();
        history_changed
            .nodes
            .iter_mut()
            .find(|n| n.id == "acc")
            .unwrap()
            .params
            .iter_mut()
            .find(|(k, _)| k == "historyMax")
            .unwrap()
            .1 = ParamValue::Int(10);
        assert_eq!(
            history_restarts(Some(&p), &history_changed),
            vec!["acc.history"]
        );
    }

    #[test]
    fn defaults_values_and_accumulate_refusals() {
        let c = Catalog::build();
        let p = plan(
            r#",{"id":"trace","type":"ssgi.trace","params":{"quality":"high"}},{"id":"resolve","type":"ssgi.resolve"}"#,
            // The output path keeps both nodes in the plan.
            r#",{"from":"trace.raw","to":"resolve.raw"},{"from":"resolve.ssgi","to":"forward.gi"}"#,
        );
        let node = p.nodes.iter().find(|n| n.id == "trace").unwrap();
        assert!(
            stage_values(node, c.node("ssgi.trace").unwrap()).contains(&("quality".into(), 2.0))
        );
        let bool_spec = c.node("gi.probes").unwrap();
        let value_node = PlanNode {
            params: vec![
                ("screenTraces".into(), ParamValue::Bool(true)),
                ("probeVolumeSteps".into(), ParamValue::Int(4)),
            ],
            ..node.clone()
        };
        let values = stage_values(&value_node, bool_spec);
        assert!(values.contains(&("screenTraces".into(), 1.0)));
        assert!(values.contains(&("probeVolumeSteps".into(), 4.0)));
        let base = plan(
            r#",{"id":"pyramid","type":"gi.pyramid"},{"id":"probes","type":"gi.probes"},{"id":"acc","type":"gi.accumulate"},{"id":"atrous","type":"gi.atrous"},{"id":"resolve2","type":"gi.resolve"}"#,
            r#",{"from":"pyramid.pyramid","to":"probes.pyramid"},{"from":"probes.probesFiltered","to":"acc.probes"},{"from":"acc.output","to":"atrous.src"},{"from":"atrous.dst","to":"resolve2.irradiance"},{"from":"resolve2.gi","to":"forward.gi"}"#,
        );
        assert!(accumulate_issues(&base).is_empty());
        let mut neither = base.clone();
        neither
            .nodes
            .iter_mut()
            .find(|n| n.id == "acc")
            .unwrap()
            .inputs
            .clear();
        assert_eq!(accumulate_issues(&neither)[0].code, "graph.input_missing");
        let mut both = base;
        both.nodes
            .iter_mut()
            .find(|n| n.id == "acc")
            .unwrap()
            .inputs
            .push(crate::plan::PlanInput {
                port: "irradiance".into(),
                source: InputSource::Link {
                    node: "resolve".into(),
                    port: "gi".into(),
                    history: false,
                },
            });
        assert_eq!(
            accumulate_issues(&both)[0].port.as_deref(),
            Some("irradiance")
        );
    }
}
