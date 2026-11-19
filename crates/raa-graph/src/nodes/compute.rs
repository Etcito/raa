//! Family `compute`: the `compute.wgsl` node whose WGSL body comes from the math graph or is
//! written by hand (S59). This file declares the catalog type; the module generator lives in
//! `compute/codegen.rs`, and card 'raa-graph compute 2/7: declaration and WGSL validation' adds
//! the declaration and WGSL validation hook.

use crate::catalog::{Category, GraphFormat, NodeFamily, NodeTypeSpec, ParamValue, PortType};
use crate::error::{Issue, Severity, Stage};
use crate::format::NodeDoc;
use raa_loader::json::JsonValue;

#[allow(dead_code)] // the compute pass and validation cards call these; the tests compile the same file
#[path = "compute/codegen.rs"]
pub(crate) mod codegen;

/// The `compute` family, discovered by build.rs.
pub fn family() -> Box<dyn NodeFamily> {
    Box::new(ComputeFamily)
}

struct ComputeFamily;

fn identity_upgrade(_from: u32, _params: &mut Vec<(String, ParamValue)>) -> Result<(), String> {
    Ok(())
}

impl NodeFamily for ComputeFamily {
    fn name(&self) -> &'static str {
        "compute"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        vec![NodeTypeSpec {
            id: "compute.wgsl",
            version: 1,
            category: Category::Compute,
            label: "Compute (WGSL)",
            description: "A compute pass whose WGSL body comes from the math graph or is written by hand",
            pass: None,
            contributor: false,
            default_order: 46,
            multi_instance: true,
            side_effect: false,
            required_in_graph: false,
            inputs: Vec::new(),
            outputs: Vec::new(),
            params: Vec::new(),
            budgets: Vec::new(),
            upgrade: identity_upgrade,
            family: "compute",
        }]
    }
}
