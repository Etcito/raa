//! Family `output`: the single `output.display` node that receives the final image (S51).

use crate::catalog::{
    Category, NodeFamily, NodeTypeSpec, ParamValue, PortSpec, PortType, SizeSpec,
};

/// The `output` family, discovered by build.rs.
pub fn family() -> Box<dyn NodeFamily> {
    Box::new(OutputFamily)
}

struct OutputFamily;

#[allow(clippy::ptr_arg)]
fn no_upgrade(from: u32, _params: &mut Vec<(String, ParamValue)>) -> Result<(), String> {
    Err(format!("no upgrade from version {from}"))
}

impl NodeFamily for OutputFamily {
    fn name(&self) -> &'static str {
        "output"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        vec![NodeTypeSpec {
            id: "output.display",
            version: 1,
            category: Category::Output,
            label: "Display",
            description: "The image shown in the view; exactly one per graph.",
            pass: None,
            contributor: false,
            default_order: 0,
            multi_instance: false,
            side_effect: false,
            required_in_graph: false,
            inputs: vec![PortSpec {
                key: "color",
                label: "Color",
                ty: PortType::Image2d,
                format: None,
                size: SizeSpec::Scale(1.0),
                mips: 1,
                layers: 1,
                required: true,
                default_source: &[],
                absent_ok: false,
                generic: true,
                history: false,
                slot: None,
            }],
            outputs: Vec::new(),
            params: Vec::new(),
            budgets: Vec::new(),
            upgrade: no_upgrade,
            family: "output",
        }]
    }
}
