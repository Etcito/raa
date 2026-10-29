//! Animation graph brick for RAA (S19): document model, refusals and caps.
//! Written from the S19 contract; Sketchbook `GraphEntity` is not ported.
//! The public tick is [`GraphRuntime::step`]; its ordering is documented in `runtime`.

pub mod blend_weights;
pub mod canon;
pub mod decode;
pub mod error;
pub mod layers;
pub mod load;
pub mod model;
pub mod motion;
pub mod output;
pub mod params; pub mod state;
pub mod runtime;
pub mod scenario;
pub mod scenario_data;
pub mod snapshot;
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
pub use output::{GraphEvent, RootDelta, TickOut};
pub use runtime::{Frame, GraphRuntime};
pub use scenario::{Rollback, RouteEntry, SCENARIO_TICKS, ScenarioEvent, ScenarioRun, ScenarioTick,
    run, run_penguin};
pub use {params::{ParamValue, Params}, state::{Crossfade, GraphState, LayerState}};
pub use triangulation::{Triangulation, triangulate};
