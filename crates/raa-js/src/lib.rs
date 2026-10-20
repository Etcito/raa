//! Frozen wasm-bindgen API v0 of RAA; no consumer is named.
//!
//! `convert` holds the pure-Rust helpers (error mapping, handle table,
//! descriptor conversion) that are tested natively.

pub mod convert;

#[cfg(target_arch = "wasm32")]
mod engine;
#[cfg(target_arch = "wasm32")]
pub use engine::Engine;
