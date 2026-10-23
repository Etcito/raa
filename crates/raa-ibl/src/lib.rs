//! Image-based lighting from the sky cube: CPU SH9 reference, the IBL WGSL library and the
//! `ibl` provider of the `Ambient` slot; GPU filtering and the BRDF LUT are added by later
//! tasks.

mod filter;
mod lut;
mod provider;
mod sh;

/// IBL contract functions over the S10 group-1 environment, reused by other providers:
/// `raa_ibl_irradiance(n) -> vec3` (SH9, cosine-convolved / pi, so a constant environment
/// c gives c), `raa_ibl_specular(r, roughness) -> vec3` (prefiltered cube at roughness x
/// max mip), `raa_ibl_dfg(n_dot_v, roughness) -> vec2` (BRDF LUT scale, bias) and
/// `raa_specular_occlusion(n_dot_v, ao, roughness) -> f32` (Lagarde).
pub const IBL_LIB_WGSL: &str = provider::IBL_LIB_SOURCE;
