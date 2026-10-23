//! Layered BRDF of the lit renderer (S11 `Direct` slot): base, sheen and clearcoat.
//!
//! The base is Lambert x (1 - metallic) plus GGX with height-correlated Smith visibility and
//! the multiscatter compensation `1 + F0 (1 / E(mu) - 1)`; the Lambert lobe is weighted by the
//! albedo the compensated specular leaves. The sheen is the Charlie NDF with Neubelt
//! visibility and scales the base by its directional albedo. The clearcoat is GGX with F0 0.04
//! and Kelemen visibility on the geometric normal and attenuates the layers below by its
//! Fresnel at `n.v`.
//!
//! [`eval_direct`] is the CPU reference of that model, which the WGSL provider mirrors line
//! by line: it returns the reflected radiance per unit irradiance x cos (`f(v, l) n.l`).
//! [`eval_under_coat`] and [`eval_base`] are the same model without the clearcoat, and without
//! the clearcoat and the sheen; [`eval_direct`] returns them exactly when those weights are 0.
//! [`ggx_dfg_fit`] and [`sheen_albedo_fit`] are the analytic albedo fits the model uses, the
//! same in WGSL; both over-estimate the albedo they approximate, which is what keeps the
//! layered albedo at or below 1 (see `tests/energy.rs`).
//!
//! [`providers`] are the `layered` slot providers at priority 20 (above raa-ibl's `ibl` at
//! 10): `Direct` is [`BRDF_WGSL`], the WGSL mirror of [`eval_direct`]; `Ambient` is raa-ibl's
//! image-based base terms plus the clearcoat reflection and the sheen indirect, attenuated by
//! the clearcoat as in `Direct`. [`install`] adds both to a [`raa_lit::LitRenderer`]. The S11
//! slots receive no geometric normal, so in WGSL the clearcoat uses the shading normal.

mod cpu;
mod shaders;

pub use cpu::{
    CLEARCOAT_F0, DirectTerms, MIN_N_DOT_V, MIN_ROUGHNESS, MIN_SHEEN_ROUGHNESS, eval_base,
    eval_direct, eval_under_coat, f0_from_ior, ggx_dfg_fit, sheen_albedo_fit,
};
pub use shaders::{BRDF_WGSL, install, providers};
