//! Family `gi_emissive`: the `emissive` parameter of the three GI pass nodes (contract
//! 'rg-gi-emissive-param'). `extend_types` appends one Bool parameter to pass.gi.volume,
//! pass.gi.screen and pass.gi.ssgi, in that order, and sets the Compute storage buffers of the
//! screen and SSGI nodes to what their shaders bind: 3 and 1, the read-only voxel buffer at group 2
//! binding 6 of contract 'gi-screen-emission-wgsl'. No node type is added. The executor extension
//! `gi_emissive` (exec_ext/gi_emissive.rs) writes the value of a plan through `GiHandle`.

use crate::catalog::{
    NodeFamily, NodeTypeSpec, ParamBinding, ParamKind, ParamSpec, ParamValue, Rebuild, ShaderStage,
};

/// The `gi_emissive` family, discovered by build.rs.
pub fn family() -> Box<dyn NodeFamily> {
    Box::new(GiEmissive)
}

/// The GI pass node types that carry `emissive`, in the order `extend_types` extends them.
const NODE_TYPES: [&str; 3] = ["pass.gi.volume", "pass.gi.screen", "pass.gi.ssgi"];

/// The file that defines the emission the parameter selects (`raa_gi::EmissiveSources`).
const SOURCE: &str = "crates/raa-gi/src/emissive.rs";

struct GiEmissive;

impl NodeFamily for GiEmissive {
    fn name(&self) -> &'static str {
        "gi_emissive"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        Vec::new()
    }

    fn extend_types(&self, types: &mut Vec<NodeTypeSpec>) {
        for id in NODE_TYPES {
            node_mut(types, id).params.push(emissive_param());
        }
        compute_storage_buffers(types, "pass.gi.screen", 3);
        compute_storage_buffers(types, "pass.gi.ssgi", 1);
    }
}

/// The node type `id`; a missing one is a catalog defect.
fn node_mut<'a>(types: &'a mut [NodeTypeSpec], id: &str) -> &'a mut NodeTypeSpec {
    types
        .iter_mut()
        .find(|node| node.id == id)
        .unwrap_or_else(|| panic!("gi_emissive: no node type `{id}`"))
}

/// Sets the storage buffers of the Compute budget of node type `id`, leaving its other fields.
fn compute_storage_buffers(types: &mut [NodeTypeSpec], id: &str, storage_buffers: u32) {
    let budget = node_mut(types, id)
        .budgets
        .iter_mut()
        .find(|budget| matches!(budget.stage, ShaderStage::Compute))
        .unwrap_or_else(|| panic!("gi_emissive: `{id}` has no Compute budget"));
    budget.storage_buffers = storage_buffers;
}

/// The `emissive` parameter: a Bool that defaults to on and rebuilds the pipeline.
fn emissive_param() -> ParamSpec {
    ParamSpec {
        key: "emissive",
        label: "Emissive",
        group: "Emissive",
        kind: ParamKind::Bool,
        default: ParamValue::Bool(true),
        min: None,
        max: None,
        step: None,
        options: &[],
        unit: "",
        rebuild: Rebuild::Pipeline,
        primary: false,
        unstated_range: false,
        binding: ParamBinding::Node,
        source: SOURCE,
    }
}
