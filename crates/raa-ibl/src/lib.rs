//! Image-based lighting derived from the sky cube (S10 group 1 bindings 3-5).
//!
//! [`IblFeature::install`] builds the BRDF LUT once and sets it on the pipeline's lighting,
//! creates the filter and adds the `ibl` provider of the `Ambient` slot (priority 10) to the
//! lit renderer. The returned [`IblHandle`] is driven by `sky.env`: `filter` records the
//! downsample, GGX prefilter and SH9 reduction of the sky cube into the frame's encoder;
//! `apply_pending` (every prepare) maps the coefficient readbacks of earlier frames and sets
//! the env once they land, so the SH reaches the lighting asynchronously, at least one frame
//! after the filter. Until then the provider falls back to the flat S1 ambient.

mod feature;
mod filter;
mod lut;
mod provider;
mod sh;

pub use feature::{IblFeature, IblHandle};

/// IBL contract functions over the S10 group-1 environment, reused by other providers:
/// `raa_ibl_irradiance(n) -> vec3` (SH9, cosine-convolved / pi, so a constant environment
/// c gives c), `raa_ibl_specular(r, roughness) -> vec3` (prefiltered cube at roughness x
/// max mip), `raa_ibl_dfg(n_dot_v, roughness) -> vec2` (BRDF LUT scale, bias) and
/// `raa_specular_occlusion(n_dot_v, ao, roughness) -> f32` (Lagarde).
pub const IBL_LIB_WGSL: &str = provider::IBL_LIB_SOURCE;
