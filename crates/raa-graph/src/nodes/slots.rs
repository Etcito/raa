//! Family `slots`: the forward-class slot inputs and the G-buffer inputs that take a bound image,
//! the overlay depth input and the multi-instance ao and contact nodes (contract slots-family).
//! `extend_types` edits existing types and adds no node type. `validate_node` refuses a node whose
//! fixed slot set exceeds the fragment sampled-texture limit (`graph.budget`): a graph never adds a
//! slot, and a link only replaces the view of an existing one. Every other G-buffer input keeps
//! `generic: false`, so `graph.fixed_wiring` refuses an alternative source instead of ignoring it.

use crate::catalog::{GraphFormat, NodeFamily, NodeTypeSpec, PortSpec, PortType, SizeSpec};
use crate::error::{Issue, Severity, Stage};
use crate::format::NodeDoc;
use crate::validate::ValidateCtx;

/// The `slots` family, discovered by build.rs; registered after `passes` and before `view`.
pub fn family() -> Box<dyn NodeFamily> {
    Box::new(Slots)
}

/// The forward-class slot inputs of pass.forward that accept any image of their type.
const FORWARD_SLOTS: [&str; 9] = [
    "shadowMap",
    "envSpecular",
    "ao",
    "contact",
    "gi",
    "areaShadow",
    "cloudTransmittance",
    "opaqueColor",
    "opaqueDepth",
];

/// The G-buffer inputs whose pass reads the bound image (contract slots-family, row 2). The
/// overlay's depth input is pushed by `overlay_depth`.
const GBUFFER_GENERIC: [(&str, &[&str]); 14] = [
    ("pass.ao.gtao", &["depth", "normal"]),
    ("pass.shadow.contact", &["depth"]),
    ("pass.light.tiles", &["depth"]),
    ("pass.ssr", &["depth", "normal", "material"]),
    ("pass.taa", &["depth", "motion"]),
    ("neural.super-resolution", &["depth", "motion"]),
    ("pass.gi.screen", &["depth", "normal", "motion"]),
    ("pass.gi.ssgi", &["depth", "normal", "motion"]),
    ("pass.gi.area-shadow", &["depth", "normal", "motion"]),
    ("pass.dof", &["depth"]),
    ("pass.sunshafts", &["depth"]),
    ("pass.transparent.oit", &["depth"]),
    ("pass.sky.cloud-screen", &["depth"]),
    ("pass.tonemap", &["depth"]),
];

/// The node types that accept several instances (contract slots-family, rule 4).
const MULTI_INSTANCE: [&str; 2] = ["pass.ao.gtao", "pass.shadow.contact"];

struct Slots;

impl NodeFamily for Slots {
    fn name(&self) -> &'static str {
        "slots"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        Vec::new()
    }

    fn extend_types(&self, types: &mut Vec<NodeTypeSpec>) {
        let forward = node_mut(types, "pass.forward");
        for key in FORWARD_SLOTS {
            port_mut(&mut forward.inputs, key, "pass.forward").generic = true;
        }
        for (id, keys) in GBUFFER_GENERIC {
            let node = node_mut(types, id);
            for key in keys {
                port_mut(&mut node.inputs, key, id).generic = true;
            }
        }
        node_mut(types, "pass.overlay").inputs.push(overlay_depth());
        for id in MULTI_INSTANCE {
            node_mut(types, id).multi_instance = true;
        }
    }

    fn validate_node(
        &self,
        node: &NodeDoc,
        spec: &NodeTypeSpec,
        ctx: &ValidateCtx<'_>,
        issues: &mut Vec<Issue>,
    ) {
        let slots = spec
            .inputs
            .iter()
            .filter(|port| port.slot.is_some())
            .count() as u32;
        let limit = ctx.limits.sampled_textures;
        if slots <= limit {
            return;
        }
        issues.push(Issue {
            code: "graph.budget",
            severity: Severity::Error,
            stage: Stage::Budget,
            node: Some(node.id.clone()),
            port: None,
            param: None,
            link: None,
            message: format!(
                "node `{}` keeps a fixed slot set of {slots} sampled textures, and the limit is {limit}",
                node.id
            ),
            fix: format!(
                "Raise maxSampledTexturesPerShaderStage to {slots} or more; a graph never adds a slot."
            ),
        });
    }
}

/// The node type `id`; a missing one is a catalog defect.
fn node_mut<'a>(types: &'a mut [NodeTypeSpec], id: &str) -> &'a mut NodeTypeSpec {
    types
        .iter_mut()
        .find(|node| node.id == id)
        .unwrap_or_else(|| panic!("slots: no node type `{id}`"))
}

/// The port `key` of node type `node`; a missing one is a catalog defect.
fn port_mut<'a>(ports: &'a mut [PortSpec], key: &str, node: &str) -> &'a mut PortSpec {
    ports
        .iter_mut()
        .find(|port| port.key == key)
        .unwrap_or_else(|| panic!("slots: `{node}` has no port `{key}`"))
}

/// The `depth` input of pass.overlay: the depth attachment of the prepass or the virtual-geometry
/// G-buffer, read as a bound image when a link provides one (the budget is unchanged).
fn overlay_depth() -> PortSpec {
    PortSpec {
        key: "depth",
        label: "Depth",
        ty: PortType::Depth2d,
        format: Some(GraphFormat::Depth32Float),
        size: SizeSpec::Scale(1.0),
        mips: 1,
        layers: 1,
        required: false,
        default_source: &["pass.prepass.depth", "pass.vg.gbuffer.depth"],
        absent_ok: false,
        generic: true,
        history: false,
        slot: None,
    }
}
