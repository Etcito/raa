//! Animation graph brick for RAA (S19): document model, refusals and caps.
//! Written from the S19 contract; Sketchbook `GraphEntity` is not ported.

pub mod error;
pub mod model;

pub use error::{GraphError, codes};
pub use model::{
    BlendPoint, BlendSpace, CondOp, Condition, GRAPH_CAPS, Graph, GraphCaps, GraphRootMotion,
    Layer, Machine, Num, Param, ParamType, State, Transition, WILDCARD,
};
