//! Animation graph brick for RAA (S19): document model, refusals and caps.
//! Written from the S19 contract; Sketchbook `GraphEntity` is not ported.

pub mod blend_weights;
pub mod canon;
pub mod decode;
pub mod error;
pub mod layers;
pub mod load;
pub mod model;
pub mod motion;
pub mod params; pub mod state;
pub mod triangulation;
pub mod transitions;

pub use blend_weights::{BlendWeight, evaluate, evaluate_1d, evaluate_2d};
pub use canon::{graph_value, to_canonical};
pub use decode::{GraphParts, decode_conditions, decode_document};
pub use error::{GraphError, codes};
pub use load::{Compiled, decode_graph};
pub use model::{
    BlendPoint, BlendSpace, CondOp, Condition, GRAPH_CAPS, Graph, GraphCaps, GraphRootMotion,
    Layer, Machine, Num, Param, ParamType, State, Transition, WILDCARD,
};
pub use motion::{Contribution, blend_local};
pub use {params::{ParamValue, Params}, state::{Crossfade, GraphState, LayerState}};
pub use triangulation::{Triangulation, triangulate};
