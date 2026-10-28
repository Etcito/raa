//! Animation graph brick for RAA (S19): document model, refusals and caps.
//! Written from the S19 contract; Sketchbook `GraphEntity` is not ported.

pub mod canon;
pub mod decode;
pub mod error;
pub mod load;
pub mod model;

pub use canon::{graph_value, to_canonical};
pub use decode::{GraphParts, decode_conditions, decode_document};
pub use error::{GraphError, codes};
pub use load::decode_graph;
pub use model::{
    BlendPoint, BlendSpace, CondOp, Condition, GRAPH_CAPS, Graph, GraphCaps, GraphRootMotion,
    Layer, Machine, Num, Param, ParamType, State, Transition, WILDCARD,
};
