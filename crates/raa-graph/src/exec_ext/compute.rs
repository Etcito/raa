//! Executor extension `compute`: plan tracking and device-free helpers. Compute pass installation
//! is owned by the compute pass card.

use crate as raa_graph;
use crate::catalog::{Catalog, GraphFormat, ParamValue, PortType, SizeSpec};
use crate::error::{Issue, Severity, Stage};
use crate::exec::{ExecExtension, FeatureHandles, NodeStats};
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
    nodes: Vec<String>,
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
        self.nodes = nodes.into_iter().map(|node| node.id.clone()).collect();
        Ok(())
    }

    fn clear(&mut self, _handles: &mut FeatureHandles<'_>) {
        self.nodes.clear();
    }

    fn frame(&mut self, _handles: &mut FeatureHandles<'_>) -> Result<(), RaaError> {
        Ok(())
    }

    fn stats(&self, _out: &mut Vec<NodeStats>) {}
}
