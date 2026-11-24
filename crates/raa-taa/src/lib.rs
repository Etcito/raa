//! Temporal anti-aliasing, analytic super-resolution and reusable temporal histories.
//!
//! The resolve reads single-sample guides on the HDR source grid. A larger graph-bound
//! HDR destination makes the resolve and its history display-sized. Scene-space post
//! passes must precede the resolve for a display-sized HDR chain.

mod feature;
mod history;
mod resources;
mod shaders;
mod tunables;

pub use feature::{TaaFeature, TaaHandle};
pub use history::{HistoryPair, HistoryState};
pub use shaders::{
    REPROJECT_WGSL, TAA_ENTRY, TAA_PARAMS_SIZE, TAA_WGSL, TAA_WORKGROUP, TEMPORAL_FRAME_SIZE,
    taa_module_source,
};
pub use tunables::tunables;

/// S12 name of the pass.
pub const TAA_PASS_NAME: &str = "taa";
/// S12 order of the pass (after `particles.blend`, before `dof`).
pub const TAA_ORDER: u32 = 100;
