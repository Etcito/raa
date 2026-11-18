//! Executor extension `gi_emissive` (contract 'rg-gi-emissive-param'): writes the `emissive`
//! parameters of the GI pass nodes of a plan through `GiHandle::set_emissive_sources`. The
//! extension keeps no copy of the previous sources: the executor's `rollback` re-applies the
//! previous plan (or calls `clear`) when a later extension refuses, and `clear` restores
//! `EmissiveSources::ALL`.

use raa_gi::EmissiveSources;

use crate::catalog::ParamValue;
use crate::error::Issue;
use crate::exec::{ExecExtension, FeatureHandles};
use crate::plan::GraphPlan;

/// The extension, discovered by build.rs (stem `gi_emissive`).
pub fn extension() -> Box<dyn ExecExtension> {
    Box::new(GiEmissiveExtension)
}

struct GiEmissiveExtension;

/// The sources a plan selects: each is the `emissive` of the first node of its type.
pub(crate) fn sources_of(plan: &GraphPlan) -> EmissiveSources {
    EmissiveSources {
        volume: source_of(plan, "pass.gi.volume"),
        screen: source_of(plan, "pass.gi.screen"),
        ssgi: source_of(plan, "pass.gi.ssgi"),
    }
}

/// Whether the first node of `type_id` emits. Only a Bool `false` turns emission off: an absent
/// node, an absent parameter and a value of another kind all leave it on.
fn source_of(plan: &GraphPlan, type_id: &str) -> bool {
    let node = plan.nodes.iter().find(|node| node.type_id == type_id);
    let value = node.and_then(|node| node.params.iter().find(|(key, _)| key == "emissive"));
    !matches!(value, Some((_, ParamValue::Bool(false))))
}

impl ExecExtension for GiEmissiveExtension {
    fn name(&self) -> &'static str {
        "gi_emissive"
    }

    /// Writes the plan's sources to the installed GI handle. Never refuses: with no GI handle
    /// installed nothing is written.
    fn apply(
        &mut self,
        plan: &GraphPlan,
        _previous: Option<&GraphPlan>,
        h: &mut FeatureHandles<'_>,
    ) -> Result<(), Vec<Issue>> {
        if let Some(gi) = h.handles.gi.as_ref() {
            gi.set_emissive_sources(sources_of(plan));
        }
        Ok(())
    }

    /// Restores every source to emission on.
    fn clear(&mut self, h: &mut FeatureHandles<'_>) {
        if let Some(gi) = h.handles.gi.as_ref() {
            gi.set_emissive_sources(EmissiveSources::ALL);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{PlanNode, PlanPortRef};

    fn node(type_id: &str, params: Vec<(String, ParamValue)>) -> PlanNode {
        PlanNode {
            id: format!("{type_id}-node"),
            type_id: type_id.into(),
            type_version: 1,
            order_index: 0,
            params,
            inputs: Vec::new(),
            compute: None,
        }
    }

    fn plan(nodes: Vec<PlanNode>) -> GraphPlan {
        GraphPlan {
            name: "test".into(),
            nodes,
            links: Vec::new(),
            resources: Vec::new(),
            histories: Vec::new(),
            output: PlanPortRef {
                node: "out".into(),
                port: "color".into(),
            },
        }
    }

    fn emissive(value: ParamValue) -> Vec<(String, ParamValue)> {
        vec![("emissive".into(), value)]
    }

    #[test]
    fn a_screen_node_set_to_false_turns_only_the_screen_source_off() {
        let plan = plan(vec![node(
            "pass.gi.screen",
            emissive(ParamValue::Bool(false)),
        )]);
        assert_eq!(
            sources_of(&plan),
            EmissiveSources {
                volume: true,
                screen: false,
                ssgi: true,
            }
        );
    }

    #[test]
    fn a_plan_without_gi_nodes_keeps_every_source_on() {
        let plan = plan(vec![node("pass.forward", Vec::new())]);
        assert_eq!(sources_of(&plan), EmissiveSources::ALL);
    }

    #[test]
    fn an_integer_value_counts_as_on() {
        let plan = plan(vec![node("pass.gi.ssgi", emissive(ParamValue::Int(0)))]);
        assert_eq!(sources_of(&plan), EmissiveSources::ALL);
    }
}
