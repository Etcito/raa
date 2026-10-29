//! S19 document model. Numbers read from the document stay `f64` in [`Num::doc`]
//! for canonical writing; the deterministic runtime only reads [`Num::rt`].

use raa_anim::{MAX_BONES, RootMotion};
use raa_det::Tick;

/// Transition source matching every state of the machine.
pub const WILDCARD: &str = "*";

/// A document number with its runtime value, converted once at load.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Num {
    pub doc: f64,
    pub rt: f32,
}

impl Num {
    pub fn new(doc: f64) -> Self {
        Self {
            doc,
            rt: doc as f32,
        }
    }
}

/// Count limits for graph documents. All are inclusive and excess counts are
/// refused with `graph.cap`, never truncated. S19 only says "caps follow
/// GRAPH_CAPS" and gives no numbers: every value below is an implementation
/// limit chosen here, except `max_mask_bones`, which is the S16 `MAX_BONES`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphCaps {
    pub max_params: usize,
    pub max_blend_spaces: usize,
    pub max_points_per_space: usize,
    pub max_layers: usize,
    pub max_machines: usize,
    pub max_states_per_machine: usize,
    pub max_transitions_per_machine: usize,
    pub max_conditions_per_transition: usize,
    pub max_mask_bones: usize,
}

pub const GRAPH_CAPS: GraphCaps = GraphCaps {
    max_params: 64,
    max_blend_spaces: 64,
    max_points_per_space: 64,
    max_layers: 16,
    max_machines: 16,
    max_states_per_machine: 256,
    max_transitions_per_machine: 1024,
    max_conditions_per_transition: 8,
    max_mask_bones: MAX_BONES,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamType {
    Float,
    Int,
    Bool,
    Trigger,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: ParamType,
    /// Absent in the document for triggers; zero then.
    pub default: Num,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CondOp {
    Trigger,
    Eq,
    Ne,
    Gt,
    Lt,
    Ge,
    Le,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Condition {
    pub param: String,
    pub op: CondOp,
    /// Comparison operand; zero for `trigger`.
    pub value: Num,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlendPoint {
    pub clip: String,
    /// One coordinate per blend-space param (1D or 2D).
    pub at: Vec<Num>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlendSpace {
    pub name: String,
    pub params: Vec<String>,
    pub points: Vec<BlendPoint>,
    pub sync: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct State {
    pub name: String,
    /// A clip name or a blend space name.
    pub motion: String,
    pub looping: bool,
    pub speed_param: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Transition {
    /// A state name or [`WILDCARD`].
    pub from: String,
    pub to: String,
    pub when: Vec<Condition>,
    pub blend_ticks: Tick,
    pub exit_tick: Option<Tick>,
    pub priority: i32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Machine {
    pub name: String,
    pub entry: String,
    pub states: Vec<State>,
    pub transitions: Vec<Transition>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    pub name: String,
    pub machine: String,
    /// `None` applies to all bones; otherwise bones whose descendants are included.
    pub mask: Option<Vec<String>>,
    pub weight: Num,
    pub additive: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphRootMotion {
    pub bone: String,
    pub mode: RootMotion,
}

/// A validated S19 document.
#[derive(Debug, Clone, PartialEq)]
pub struct Graph {
    pub name: String,
    pub rig: String,
    pub params: Vec<Param>,
    pub blend_spaces: Vec<BlendSpace>,
    pub layers: Vec<Layer>,
    pub machines: Vec<Machine>,
    pub root_motion: GraphRootMotion,
}
