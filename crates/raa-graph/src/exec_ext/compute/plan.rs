//! Device-free plan helpers for compute nodes.

use super::*;
use super::{GraphPlan, PlanNode};

/// Handoff: `compute_nodes` is used by card 'raa-graph compute 5/7: compute pass, pipeline, bind groups and dispatch' to install compute passes, and by card 'raa-graph compute 7/7: executor-owned history pairs of compute inputs' to add history handling. Today `ComputeExtension::apply` stores its ids after checking for a lit renderer, `clear` forgets them, and no pass is installed; see `crates/raa-graph/src/exec_ext/compute.rs`.
pub fn compute_nodes(plan: &GraphPlan) -> Vec<&PlanNode> {
    plan.nodes
        .iter()
        .filter(|node| node.type_id == "compute.wgsl" && node.compute.is_some())
        .collect()
}

pub fn compute_entry(id: &str) -> String {
    format!("compute#{id}")
}

pub fn source_entry(plan: &GraphPlan, catalog: &Catalog, node: &str) -> Option<String> {
    if plan
        .nodes
        .iter()
        .any(|candidate| candidate.id == node && candidate.type_id == "compute.wgsl")
    {
        return Some(compute_entry(node));
    }
    if let Some((_, entry)) = super::plan_fns::node_entries(plan, catalog)
        .into_iter()
        .find(|(id, _)| id == node)
    {
        return Some(entry);
    }
    plan.nodes
        .iter()
        .find(|candidate| candidate.id == node)
        .map(|candidate| format!("{}#{}", candidate.type_id, candidate.id))
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComputeInput {
    pub key: String,
    pub ty: PortType,
    pub source: Option<(String, String)>,
    pub history: bool,
    pub filterable: bool,
}

pub fn compute_inputs(plan: &GraphPlan, catalog: &Catalog, node: &PlanNode) -> Vec<ComputeInput> {
    let Some(compute) = &node.compute else {
        return Vec::new();
    };
    compute
        .inputs
        .iter()
        .map(|decl| {
            let input = node.inputs.iter().find(|input| input.port == decl.key);
            let history = matches!(
                input.map(|input| &input.source),
                Some(InputSource::Link { history: true, .. })
            );
            let source = input
                .and_then(|input| match &input.source {
                    InputSource::Link {
                        node: source,
                        port,
                        history: false,
                    } => Some((source, port)),
                    _ => None,
                })
                .and_then(|(source, port)| {
                    source_entry(plan, catalog, source).map(|entry| (entry, port.clone()))
                });
            let format = input.and_then(|input| match &input.source {
                InputSource::Link {
                    node: source, port, ..
                }
                | InputSource::Default { node: source, port } => plan
                    .nodes
                    .iter()
                    .find(|candidate| candidate.id == *source)
                    .and_then(|producer| {
                        if let Some(compute) = &producer.compute {
                            compute
                                .outputs
                                .iter()
                                .find(|output| output.key == *port)
                                .and_then(|output| output.format.as_ref())
                        } else {
                            catalog
                                .node(&producer.type_id)
                                .and_then(|spec| {
                                    spec.outputs.iter().find(|output| output.key == *port)
                                })
                                .and_then(|output| output.format.as_ref())
                        }
                    }),
                InputSource::Absent => None,
            });
            let filterable = !matches!(
                format,
                Some(GraphFormat::R32Float | GraphFormat::Rg32Float | GraphFormat::Rgba32Float)
            );
            ComputeInput {
                key: decl.key.clone(),
                ty: decl.ty.clone(),
                source,
                history,
                filterable,
            }
        })
        .collect()
}

/// History-linked compute inputs, in plan and input declaration order.
pub fn history_inputs(plan: &GraphPlan) -> Vec<(String, String, String, String)> {
    plan.nodes
        .iter()
        .filter(|node| node.type_id == "compute.wgsl")
        .flat_map(|consumer| {
            consumer.inputs.iter().filter_map(move |input| {
                let InputSource::Link {
                    node: producer,
                    port,
                    history: true,
                } = &input.source
                else {
                    return None;
                };
                Some((
                    consumer.id.clone(),
                    input.port.clone(),
                    producer.clone(),
                    port.clone(),
                ))
            })
        })
        .collect()
}

pub fn pack_params(params: &[(String, ParamValue)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut offset = 0usize;
    for (_, value) in params {
        let (alignment, values): (usize, Vec<[u8; 4]>) = match value {
            ParamValue::Number(v) => (4, vec![(*v as f32).to_le_bytes()]),
            ParamValue::Bool(v) => (4, vec![u32::from(*v).to_le_bytes()]),
            ParamValue::Vec2(v) => (8, v.iter().map(|x| (*x as f32).to_le_bytes()).collect()),
            ParamValue::Vec3(v) => (16, v.iter().map(|x| (*x as f32).to_le_bytes()).collect()),
            ParamValue::Vec4(v) => (16, v.iter().map(|x| (*x as f32).to_le_bytes()).collect()),
            _ => continue,
        };
        offset = offset.div_ceil(alignment) * alignment;
        bytes.resize(offset, 0);
        for value in values {
            bytes.extend_from_slice(&value);
            offset += 4;
        }
    }
    let size = bytes.len().div_ceil(16) * 16;
    bytes.resize(size.max(16), 0);
    bytes
}

pub fn output_size(size: Option<&SizeSpec>, surface: (u32, u32)) -> (u32, u32) {
    match size {
        Some(SizeSpec::Scale(scale)) => (
            (surface.0 as f32 * scale).floor() as u32,
            (surface.1 as f32 * scale).floor() as u32,
        ),
        Some(SizeSpec::Fixed { width, height }) => (*width, *height),
        _ => surface,
    }
}

pub fn frame_rows(size: (u32, u32), frame: u64, time_s: f32) -> [u8; 32] {
    let mut bytes = [0; 32];
    for (offset, value) in [(0, size.0.to_le_bytes()), (4, size.1.to_le_bytes())] {
        bytes[offset..offset + 4].copy_from_slice(&value);
    }
    bytes[8..12].copy_from_slice(&(1.0 / size.0 as f32).to_le_bytes());
    bytes[12..16].copy_from_slice(&(1.0 / size.1 as f32).to_le_bytes());
    bytes[16..20].copy_from_slice(&(frame as u32).to_le_bytes());
    bytes[20..24].copy_from_slice(&time_s.to_le_bytes());
    bytes
}

pub const RG_FRAME_BYTES: u64 = 368;
pub const COMPUTE_ORDER: u32 = 46;
