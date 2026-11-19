//! Executor extension `compute`: per-node passes and device-free plan helpers.

use crate as raa_graph;
use crate::catalog::{Catalog, GraphFormat, ParamValue, PortType, SizeSpec};
use crate::error::{Issue, Severity, Stage};
use crate::exec::{ExecExtension, FeatureHandles, NodeStats, ResourceStat};
use crate::format::{ComputeDecl, NodeDoc};
use raa_loader::json::JsonValue;
use std::{cell::RefCell, rc::Rc};
#[path = "compute/pass.rs"]
mod pass;
use crate::plan::{GraphPlan, InputSource, PlanHistory, PlanNode, PlanPortRef, PlanResource};
use raa_core::RaaError;

#[allow(dead_code, clippy::duplicate_mod)]
#[path = "wiring/plan_fns.rs"]
mod plan_fns;

#[allow(dead_code)]
#[path = "compute/plan.rs"]
mod plan;

/// The extension, discovered by build.rs.
pub fn extension() -> Box<dyn ExecExtension> {
    Box::new(ComputeExtension::default())
}

#[derive(Default)]
pub(crate) struct ComputeExtension {
    nodes: Vec<Rc<RefCell<pass::ComputeNodeState>>>,
}

impl ExecExtension for ComputeExtension {
    fn name(&self) -> &'static str {
        "compute"
    }

    fn apply(
        &mut self,
        graph: &GraphPlan,
        _previous: Option<&GraphPlan>,
        handles: &mut FeatureHandles<'_>,
    ) -> Result<(), Vec<Issue>> {
        let nodes = plan::compute_nodes(graph);
        if !nodes.is_empty() && handles.lit.is_none() {
            let node = &nodes[0].id;
            return Err(vec![Issue {
                code: "graph.unavailable",
                severity: Severity::Error,
                stage: Stage::Structure,
                node: Some(node.clone()),
                port: None,
                param: None,
                link: None,
                message: "compute nodes require an installed lit renderer".to_owned(),
                fix: "Enable the lit renderer for this graph.".to_owned(),
            }]);
        }
        let catalog = Catalog::build();
        let lit = handles.lit.as_ref().map(|lit| lit.handle().frame_buffer());
        let mut prepared = Vec::new();
        for node in nodes {
            let Some(compute) = &node.compute else {
                continue;
            };
            let params = node
                .params
                .iter()
                .map(|(key, value)| (key.clone(), json_value(value)))
                .collect();
            let doc = NodeDoc {
                id: node.id.clone(),
                type_id: node.type_id.clone(),
                version: None,
                params,
                compute: Some(ComputeDecl {
                    inputs: compute.inputs.clone(),
                    outputs: compute.outputs.clone(),
                    math: JsonValue::Null,
                    wgsl: compute.wgsl.clone(),
                }),
            };
            let module = crate::registry::compute::codegen::generate_module(&doc)
                .map_err(|mut issue| {
                    issue.code = "graph.apply";
                    issue.stage = Stage::Apply;
                    issue.node = Some(node.id.clone());
                    vec![issue]
                })?
                .0;
            prepared.push((node, module));
        }
        let mut next = Vec::new();
        let mut installed: Vec<String> = Vec::new();
        let mut removed = Vec::new();
        for (node, module) in &prepared {
            if let Some(old) = self.nodes.iter().find(|old| {
                let old = old.borrow();
                old.node == node.id && old.module == *module
            }) {
                next.push(old.clone());
                continue;
            }
            let Some(frame) = lit.as_ref() else { continue };
            let entry = plan::compute_entry(&node.id);
            if self.nodes.iter().any(|old| old.borrow().node == node.id)
                && let Ok(old) = handles.pipeline.borrow_mut().remove(&entry)
            {
                removed.push(old);
            }
            let pass = pass::ComputeNodePass::new(
                handles.gpu,
                node,
                module.clone(),
                plan::compute_inputs(graph, &catalog, node),
                frame,
                handles.pipeline.borrow().surface_format(),
            );
            let state = pass.state.clone();
            let result = handles.pipeline.borrow_mut().add(Box::new(pass));
            if let Err(error) = result {
                for name in installed {
                    let _ = handles.pipeline.borrow_mut().remove(&name);
                }
                for old in removed {
                    let _ = handles.pipeline.borrow_mut().add(old);
                }
                return Err(vec![Issue {
                    code: "graph.apply",
                    severity: Severity::Error,
                    stage: Stage::Apply,
                    node: Some(node.id.clone()),
                    port: None,
                    param: None,
                    link: None,
                    message: error.to_string(),
                    fix: "Install the compute pass.".into(),
                }]);
            }
            installed.push(entry);
            next.push(state);
        }
        for old in &self.nodes {
            if !next.iter().any(|new| Rc::ptr_eq(old, new))
                && !next
                    .iter()
                    .any(|new| new.borrow().node == old.borrow().node)
            {
                let _ = handles
                    .pipeline
                    .borrow_mut()
                    .remove(&plan::compute_entry(&old.borrow().node));
            }
        }
        for (node, _) in &prepared {
            if let Some(state) = next.iter().find(|state| {
                state.borrow().node == node.id
                    && self.nodes.iter().any(|old| Rc::ptr_eq(old, state))
            }) {
                state.borrow_mut().update(
                    handles.gpu,
                    node,
                    plan::compute_inputs(graph, &catalog, node),
                );
            }
        }
        self.nodes = next;
        Ok(())
    }

    fn clear(&mut self, handles: &mut FeatureHandles<'_>) {
        for node in self.nodes.drain(..) {
            let _ = handles
                .pipeline
                .borrow_mut()
                .remove(&plan::compute_entry(&node.borrow().node));
        }
    }

    fn frame(&mut self, _handles: &mut FeatureHandles<'_>) -> Result<(), RaaError> {
        Ok(())
    }

    fn stats(&self, out: &mut Vec<NodeStats>) {
        for node in &self.nodes {
            let stats = node.borrow().stats();
            if let Some(existing) = out.iter_mut().find(|existing| existing.node == stats.node) {
                *existing = stats;
            } else {
                out.push(stats);
            }
        }
    }
}

fn json_value(value: &ParamValue) -> JsonValue {
    match value {
        ParamValue::Number(v) => JsonValue::Number(*v),
        ParamValue::Bool(v) => JsonValue::Bool(*v),
        ParamValue::Vec2(v) => JsonValue::Array(v.iter().copied().map(JsonValue::Number).collect()),
        ParamValue::Vec3(v) => JsonValue::Array(v.iter().copied().map(JsonValue::Number).collect()),
        ParamValue::Vec4(v) => JsonValue::Array(v.iter().copied().map(JsonValue::Number).collect()),
        _ => JsonValue::Null,
    }
}
