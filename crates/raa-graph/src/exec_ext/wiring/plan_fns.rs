//! Device-free functions of the wiring extension: the resource pooling (S57) and the history
//! restarts of the executor-owned histories. The file names its types only through `super::`, so it
//! compiles twice: in `exec_ext/wiring.rs`, and in the integration tests that include it with
//! `#[path]`, because build.rs declares executor-extension modules privately.

#[allow(unused_imports)]
use super::*;
use super::{GraphPlan, PlanHistory, PlanNode, PlanPortRef, PlanResource};

/// Handoff: `pool_resources` is called by card 'Wiring 3/5: colour-chain graph frame, resource pool
/// and order from links', which pools the colour-chain resources of a plan. Today it returns one
/// slot number per input resource, aligned with the input order: resources are placed in ascending
/// `id` order, each into the lowest existing slot whose resources share its (ty, format, size, mips,
/// layers) and have lifetimes disjoint from it, with no history; otherwise into a new slot numbered
/// by creation. A history resource always takes a new slot. Empty input gives an empty result.
pub fn pool_resources(resources: &[PlanResource]) -> Vec<u32> {
    let mut order: Vec<usize> = (0..resources.len()).collect();
    order.sort_by_key(|&index| resources[index].id);
    let mut slots: Vec<Vec<usize>> = Vec::new();
    let mut out = vec![0; resources.len()];
    for index in order {
        let resource = &resources[index];
        let fit = if resource.history {
            None
        } else {
            slots.iter().position(|members| {
                members
                    .iter()
                    .all(|&member| shares_slot(&resources[member], resource))
            })
        };
        let slot = fit.unwrap_or_else(|| {
            slots.push(Vec::new());
            slots.len() - 1
        });
        slots[slot].push(index);
        out[index] = slot as u32;
    }
    out
}

/// Whether `resource` may join a slot that holds `member`: neither is a history, both have the same
/// descriptor, and their lifetimes are disjoint (touching ends overlap).
fn shares_slot(member: &PlanResource, resource: &PlanResource) -> bool {
    !member.history
        && !resource.history
        && member.ty == resource.ty
        && member.format == resource.format
        && member.size == resource.size
        && member.mips == resource.mips
        && member.layers == resource.layers
        && (member.last_use < resource.first_use || resource.last_use < member.first_use)
}

/// Handoff: `history_resets` is taken over by card 'Wiring 5/5: executor-owned histories', which calls
/// `history_resets(previous: Option<&GraphPlan>, next: &GraphPlan) -> Vec<String>` from this file
/// through the re-export `pub use plan_fns::{history_resets, pool_resources};` of
/// `exec_ext/wiring.rs`. Today it returns the keys `"<consumer node>.<consumer port>"` of the
/// `next.histories` entries, in that order, whose history restarts: there is no previous plan; the
/// previous plan has no history with the same consumer and the same producer port; or the producer
/// node differs from its node of the same id in `previous` in type id, type version or params, or is
/// absent from either plan.
pub fn history_resets(previous: Option<&GraphPlan>, next: &GraphPlan) -> Vec<String> {
    next.histories
        .iter()
        .filter(|history| restarts(previous, next, history))
        .map(|history| format!("{}.{}", history.consumer.node, history.consumer.port))
        .collect()
}

/// Whether the history `history` of `next` restarts, given the plan it replaces.
fn restarts(previous: Option<&GraphPlan>, next: &GraphPlan, history: &PlanHistory) -> bool {
    let Some(previous) = previous else {
        return true;
    };
    let Some(producer) = producer_of(next, history.resource) else {
        return true;
    };
    let linked = previous.histories.iter().any(|old| {
        old.consumer == history.consumer && producer_of(previous, old.resource) == Some(producer)
    });
    !linked || producer_changed(previous, next, &producer.node)
}

/// The producer port of the resource `resource` of `plan`.
fn producer_of(plan: &GraphPlan, resource: u32) -> Option<&PlanPortRef> {
    plan.resources
        .iter()
        .find(|candidate| candidate.id == resource)
        .map(|candidate| &candidate.producer)
}

/// Whether the node `node` differs between the two plans, or is absent from one of them.
fn producer_changed(previous: &GraphPlan, next: &GraphPlan, node: &str) -> bool {
    match (find_node(previous, node), find_node(next, node)) {
        (Some(old), Some(new)) => {
            old.type_id != new.type_id
                || old.type_version != new.type_version
                || old.params != new.params
        }
        _ => true,
    }
}

/// The node of `plan` with the id `id`.
fn find_node<'a>(plan: &'a GraphPlan, id: &str) -> Option<&'a PlanNode> {
    plan.nodes.iter().find(|node| node.id == id)
}

/// Whether the node writes a pooled chain image.
/// no handoff: Wiring 1/5: node family wiring (generic colour chain, TAA history input, multi-instance post) and its tests takes nothing from this card.
pub fn chain_type(ty: &str) -> bool {
    matches!(
        ty,
        "pass.forward"
            | "pass.gi.history"
            | "pass.ssr"
            | "pass.fog"
            | "pass.sunshafts"
            | "pass.opaque.copy"
            | "pass.transparent.oit"
            | "pass.particles.blend"
            | "pass.taa"
            | "pass.dof"
            | "pass.bloom"
            | "pass.grading"
            | "pass.cel.resolve"
    )
}

/// Colour outputs that belong to the HDR chain.
pub fn chain_resources(plan: &GraphPlan, _catalog: &raa_graph::Catalog) -> Vec<PlanResource> {
    plan.resources
        .iter()
        .filter(|resource| {
            resource.producer.port == "color"
                && plan
                    .nodes
                    .iter()
                    .any(|node| node.id == resource.producer.node && chain_type(&node.type_id))
        })
        .cloned()
        .collect()
}

/// Pipeline entries driven by executed nodes.
pub fn node_entries(plan: &GraphPlan, catalog: &raa_graph::Catalog) -> Vec<(String, String)> {
    let mut seen = Vec::new();
    plan.nodes
        .iter()
        .filter_map(|node| {
            let spec = catalog.node(&node.type_id)?;
            let entry = if node.type_id.starts_with("view.") {
                format!("view#{}", node.id)
            } else {
                if spec.contributor {
                    return None;
                }
                let pass = spec.pass?;
                let duplicate = spec.multi_instance && seen.contains(&node.type_id);
                seen.push(node.type_id.clone());
                if duplicate {
                    format!("{pass}#{}", node.id)
                } else {
                    pass.to_owned()
                }
            };
            Some((node.id.clone(), entry))
        })
        .collect()
}

/// Pass order from executed nodes, preserving positions of unplanned installed passes.
pub fn link_order(
    plan: &GraphPlan,
    entries: &[(String, String)],
    installed: &[raa_pipeline::PassInfo],
) -> Vec<String> {
    let tonemap = installed
        .iter()
        .position(|p| p.name == "tonemap")
        .unwrap_or(installed.len());
    let mut result = Vec::new();
    for node in &plan.nodes {
        if let Some((_, entry)) = entries.iter().find(|(id, _)| *id == node.id)
            && let Some((index, _)) = installed
                .iter()
                .enumerate()
                .find(|(_, p)| p.name == *entry && p.role == raa_pipeline::PassRole::Pass)
            && !(result.contains(entry) || entry.starts_with("view#") && index > tonemap)
        {
            result.push(entry.clone());
        }
    }
    for (index, pass) in installed
        .iter()
        .enumerate()
        .filter(|(_, p)| p.role == raa_pipeline::PassRole::Pass)
    {
        if result.contains(&pass.name) {
            continue;
        }
        let at = installed[..index]
            .iter()
            .rev()
            .filter(|p| p.role == raa_pipeline::PassRole::Pass)
            .find_map(|p| result.iter().position(|name| *name == p.name))
            .map_or(0, |i| i + 1);
        result.insert(at, pass.name.clone());
    }
    result
}

/// Maximum number of distinct views bound during one frame.
pub fn bound_view_count(plan: &GraphPlan, catalog: &raa_graph::Catalog) -> usize {
    let resources = chain_resources(plan, catalog);
    let slots = pool_resources(&resources);
    let count = slots.iter().max().map_or(0, |slot| *slot as usize + 1);
    let extra = plan
        .nodes
        .iter()
        .find(|n| n.type_id == "pass.tonemap")
        .and_then(|n| n.inputs.iter().find(|i| i.port == "color"))
        .is_some_and(|input| {
            let source = match &input.source {
                raa_graph::plan::InputSource::Link { node, port, .. }
                | raa_graph::plan::InputSource::Default { node, port } => Some((node, port)),
                _ => None,
            };
            !resources.iter().any(|r| {
                source.is_some_and(|(node, port)| {
                    *node == r.producer.node && *port == r.producer.port
                })
            })
        });
    count + usize::from(extra)
}

/// The (install, keep, remove) lists of [`instance_plan`].
pub type InstancePlan = (Vec<(String, String)>, Vec<(String, String)>, Vec<String>);

/// The instance passes to install, keep and remove, from `kept` (the stored instances) and `wanted`
/// (the plan's instance nodes), both as (node id, node type id). `install` and `keep` follow
/// `wanted`; `remove` lists the node ids of the kept entries that `wanted` does not match (rule E4).
pub fn instance_plan(kept: &[(String, String)], wanted: &[(String, String)]) -> InstancePlan {
    let install = wanted
        .iter()
        .filter(|w| !kept.contains(w))
        .cloned()
        .collect();
    let keep = wanted
        .iter()
        .filter(|w| kept.contains(w))
        .cloned()
        .collect();
    let remove = kept
        .iter()
        .filter(|k| !wanted.contains(k))
        .map(|(id, _)| id.clone())
        .collect();
    (install, keep, remove)
}
