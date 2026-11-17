//! The planned form of a render graph (S53): the executed nodes in order, their
//! inputs, the resources they produce and the histories they read.

use crate::catalog::{GraphFormat, ParamValue, PortType, SizeSpec};
use crate::format::ComputePortDecl;

/// A graph after planning: ordered nodes, links, resources and the displayed output.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphPlan {
    pub name: String,
    pub nodes: Vec<PlanNode>,
    pub links: Vec<PlanLink>,
    pub resources: Vec<PlanResource>,
    pub histories: Vec<PlanHistory>,
    pub output: PlanPortRef,
}

/// One executed node of a plan.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanNode {
    pub id: String,
    pub type_id: String,
    pub type_version: u32,
    pub order_index: u32,
    pub params: Vec<(String, ParamValue)>,
    pub inputs: Vec<PlanInput>,
    pub compute: Option<PlanCompute>,
}

/// How one input of a planned node is fed.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanInput {
    pub port: String,
    pub source: InputSource,
}

/// The source of a planned input.
#[derive(Clone, Debug, PartialEq)]
pub enum InputSource {
    Link {
        node: String,
        port: String,
        history: bool,
    },
    Default {
        node: String,
        port: String,
    },
    Absent,
}

/// One link of a plan.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanLink {
    pub from: PlanPortRef,
    pub to: PlanPortRef,
    pub history: bool,
}

/// A node id and port key of a plan.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanPortRef {
    pub node: String,
    pub port: String,
}

/// One resource produced by a planned node.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanResource {
    pub id: u32,
    pub producer: PlanPortRef,
    pub ty: PortType,
    pub format: Option<GraphFormat>,
    pub size: SizeSpec,
    pub mips: u32,
    pub layers: u32,
    pub first_use: u32,
    pub last_use: u32,
    pub history: bool,
}

/// A history link and the resource it reads.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanHistory {
    pub resource: u32,
    pub consumer: PlanPortRef,
}

/// The compute declaration of a planned `compute.wgsl` node.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanCompute {
    pub wgsl: String,
    pub inputs: Vec<ComputePortDecl>,
    pub outputs: Vec<ComputePortDecl>,
    pub workgroup: [u32; 2],
}
