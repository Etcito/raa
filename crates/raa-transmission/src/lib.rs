//! Transmission and refraction for the RAA lit renderer.
//!
//! [`absorption`] holds the CPU reference of the formulas the WGSL
//! `Transmission` slot uses: Beer-Lambert absorption over a clamped path
//! length, refraction with total internal reflection, the roughness-driven
//! opaque-colour mip and Schlick Fresnel.

pub mod absorption;
