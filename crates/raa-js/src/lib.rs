//! Frozen wasm-bindgen API v0 of RAA; no consumer is named.
//!
//! `convert` holds the pure-Rust helpers (error mapping, handle table,
//! descriptor conversion) that are tested natively.

pub mod convert;

// Used by the engine and the api files (wasm32); unit-tested on the host.
#[allow(dead_code)]
pub(crate) mod feature_map;

#[cfg(target_arch = "wasm32")]
mod engine;
#[cfg(target_arch = "wasm32")]
pub use engine::Engine;

/// The api files `src/api/*.rs`, discovered by build.rs: `install_all` and `frame_all`
/// call each file's `install` and `frame` in file-name order. The generated code names
/// `InstallCtx`, `EngineState`, `GpuContext`, `FrameInput` and `RaaError` unqualified;
/// the engine task imports them here and drops the `raa_js_engine` gate once
/// `InstallCtx` and `EngineState` exist.
#[cfg(all(target_arch = "wasm32", raa_js_engine))]
#[allow(dead_code)]
pub(crate) mod api {
    include!(concat!(env!("OUT_DIR"), "/api_registry.rs"));
}
