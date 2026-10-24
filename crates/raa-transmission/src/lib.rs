//! Transmission and refraction for the RAA lit renderer.
//!
//! [`absorption`] holds the CPU reference of the formulas the WGSL
//! `Transmission` slot uses: Beer-Lambert absorption over a clamped path
//! length, refraction with total internal reflection, the roughness-driven
//! opaque-colour mip and Schlick Fresnel. The `screen` provider of the `Transmission`
//! slot ([`SCREEN_PROVIDER`], priority 10) evaluates them in WGSL on the opaque colour and
//! depth copies, which the crate-private `opaque.copy` pass (order 70) fills while a
//! Transparent draw exists. [`TRANSMISSIVE_BUILTINS`] holds the `raa.ice` and `raa.glass`
//! material functions.

pub mod absorption;
mod builtins;
mod copy;
mod slot;

pub use builtins::TRANSMISSIVE_BUILTINS;
pub use slot::SCREEN_PROVIDER;
