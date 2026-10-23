//! Frozen wasm-bindgen API v0 of RAA; no consumer is named.
//!
//! `convert` holds the pure-Rust helpers (error mapping, handle table,
//! descriptor conversion) that are tested natively.

pub mod convert;

// Used by the engine and the api files (wasm32); unit-tested on the host.
#[allow(dead_code)]
pub(crate) mod feature_map;

// The engine state (host and wasm32); the engine and the api files consume it.
#[allow(dead_code)]
pub(crate) mod state;

#[cfg(target_arch = "wasm32")]
mod engine;
#[cfg(target_arch = "wasm32")]
pub use engine::Engine;

/// The api files `src/api/*.rs`, discovered by build.rs: `install_all` and `frame_all`
/// call each file's `install` and `frame` in file-name order. The generated code names
/// `InstallCtx`, `EngineState`, `GpuContext`, `FrameInput` and `RaaError` unqualified;
/// they are imported here.
#[cfg(target_arch = "wasm32")]
pub(crate) mod api {
    use crate::engine::InstallCtx;
    use crate::state::EngineState;
    use raa_core::{GpuContext, RaaError};
    use raa_web::FrameInput;

    include!(concat!(env!("OUT_DIR"), "/api_registry.rs"));
}
