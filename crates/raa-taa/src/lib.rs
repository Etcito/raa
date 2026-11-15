//! Temporal anti-aliasing and reusable temporal histories.
//!
//! The TAA resolve accumulates at render resolution 1:1 and reuses its view-local history
//! across frames; the `taa` pass and its shaders arrive in later shares. The
//! [`HistoryState`] and [`HistoryPair`] ping-pong are also reused by the M6 shadow-mask and
//! M7 reservoir histories.

mod history;

pub use history::{HistoryPair, HistoryState};

/// S12 name of the pass.
pub const TAA_PASS_NAME: &str = "taa";
/// S12 order of the pass (after `particles.blend`, before `dof`).
pub const TAA_ORDER: u32 = 100;
