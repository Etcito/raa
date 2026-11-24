//! Family `wiring`: the generic colour chain, the TAA history input and the multi-instance post
//! nodes (contract rg-wiring-family). `extend_types` edits existing types only and adds no node
//! type: the `color` ports of the chain passes become generic, pass.taa gains the `history` input
//! that raa-taa reads when bound (contract rg-taa-history-input), the history outputs of
//! pass.taa and pass.gi.history are marked as history producers, and seven post node types become
//! multi-instance. The budgets are unchanged: the TAA history input takes the binding the TAA
//! resolve group already has for its own history.

use crate::catalog::{GraphFormat, NodeFamily, NodeTypeSpec, PortSpec, PortType, SizeSpec};

/// The `wiring` family, discovered by build.rs; registered after `view`, so it sees every family.
pub fn family() -> Box<dyn NodeFamily> {
    Box::new(Wiring)
}

/// The chain passes whose `color` input and output become generic.
const CHAIN: [&str; 13] = [
    "pass.gi.history",
    "pass.ssr",
    "pass.fog",
    "pass.sunshafts",
    "pass.opaque.copy",
    "pass.transparent.oit",
    "pass.particles.blend",
    "pass.taa",
    "neural.super-resolution",
    "pass.dof",
    "pass.bloom",
    "pass.grading",
    "pass.cel.resolve",
];

/// The post node types that accept several instances.
const MULTI_INSTANCE: [&str; 7] = [
    "pass.ssr",
    "pass.dof",
    "pass.sunshafts",
    "pass.bloom",
    "pass.grading",
    "pass.cel.resolve",
    "pass.taa",
];

struct Wiring;

impl NodeFamily for Wiring {
    fn name(&self) -> &'static str {
        "wiring"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        Vec::new()
    }

    fn extend_types(&self, types: &mut Vec<NodeTypeSpec>) {
        for id in CHAIN {
            let node = node_mut(types, id);
            port_mut(&mut node.inputs, "color", id).generic = true;
            port_mut(&mut node.outputs, "color", id).generic = true;
        }
        let forward = node_mut(types, "pass.forward");
        port_mut(&mut forward.outputs, "color", "pass.forward").generic = true;
        let tonemap = node_mut(types, "pass.tonemap");
        port_mut(&mut tonemap.inputs, "color", "pass.tonemap").generic = true;

        for id in ["pass.taa", "neural.super-resolution"] {
            let taa = node_mut(types, id);
            taa.inputs.push(history_input());
            port_mut(&mut taa.outputs, "history", id).history = true;
        }
        let gi_history = node_mut(types, "pass.gi.history");
        port_mut(&mut gi_history.outputs, "history", "pass.gi.history").history = true;

        for id in MULTI_INSTANCE {
            node_mut(types, id).multi_instance = true;
        }
    }
}

/// The node type `id`; a missing one is a catalog defect.
fn node_mut<'a>(types: &'a mut [NodeTypeSpec], id: &str) -> &'a mut NodeTypeSpec {
    types
        .iter_mut()
        .find(|node| node.id == id)
        .unwrap_or_else(|| panic!("wiring: no node type `{id}`"))
}

/// The port `key` of node type `node`; a missing one is a catalog defect.
fn port_mut<'a>(ports: &'a mut [PortSpec], key: &str, node: &str) -> &'a mut PortSpec {
    ports
        .iter_mut()
        .find(|port| port.key == key)
        .unwrap_or_else(|| panic!("wiring: `{node}` has no port `{key}`"))
}

/// The `history` input of pass.taa: the previous frame's image of its own TAA history, read when
/// a history link or a bound executor input provides one (absent otherwise).
fn history_input() -> PortSpec {
    PortSpec {
        key: "history",
        label: "History",
        ty: PortType::Image2d,
        format: Some(GraphFormat::Rgba16Float),
        size: SizeSpec::Scale(1.0),
        mips: 1,
        layers: 1,
        required: false,
        default_source: &[],
        absent_ok: true,
        generic: true,
        history: true,
        slot: None,
    }
}
