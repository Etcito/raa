//! The `raa.render-graph` document types (S50).

use raa_loader::json::JsonValue;

use crate::catalog::{GraphFormat, PortType, SizeSpec};

/// The document `format` field.
pub const FORMAT: &str = "raa.render-graph";
/// The document `version` field.
pub const VERSION: u32 = 1;

/// A render graph document.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphDoc {
    pub name: String,
    pub nodes: Vec<NodeDoc>,
    pub links: Vec<LinkDoc>,
    pub editor: Option<JsonValue>,
}

/// One node of a document.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeDoc {
    pub id: String,
    pub type_id: String,
    pub version: Option<u32>,
    pub params: Vec<(String, JsonValue)>,
    pub compute: Option<ComputeDecl>,
}

/// The compute declaration of a `compute.wgsl` node.
#[derive(Clone, Debug, PartialEq)]
pub struct ComputeDecl {
    pub inputs: Vec<ComputePortDecl>,
    pub outputs: Vec<ComputePortDecl>,
    pub math: JsonValue,
    pub wgsl: String,
}

/// One input or output of a compute declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct ComputePortDecl {
    pub key: String,
    pub ty: PortType,
    pub format: Option<GraphFormat>,
    pub size: Option<SizeSpec>,
}

/// One link between two ports.
#[derive(Clone, Debug, PartialEq)]
pub struct LinkDoc {
    pub from: PortRef,
    pub to: PortRef,
    pub history: bool,
}

/// A node id and port key.
#[derive(Clone, Debug, PartialEq)]
pub struct PortRef {
    pub node: String,
    pub port: String,
}
