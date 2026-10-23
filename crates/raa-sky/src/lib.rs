//! Procedural sky of the lit renderer (S12 `sky.env`, `sky.background`).
//!
//! [`SkyDesc`] describes a zenith/horizon/ground gradient, a sun disc and a procedural cloud
//! layer ([`CloudDesc`]) in scene-referred linear HDR. [`SkyDesc::preset`] gives the built-in
//! moods (`day`, `snow`, `jungle`, `night`) and [`SkyDesc::validate`] refuses invalid
//! descriptions with `sky.invalid`.

mod background;
mod desc;
mod shaders;
mod state;

pub use desc::{CloudDesc, EnvUpdate, SkyDesc};

/// WGSL of the procedural sky: defines
/// `fn raa_sky_radiance(dir: vec3<f32>, env: EnvGpu, sun_dir: vec3<f32>, sun_color: vec3<f32>, time: f32) -> vec3<f32>`
/// plus private helpers prefixed `raa_sky_`.
///
/// It composes after `raa_shader::PRELUDE_WGSL` (for `EnvGpu`) and `raa_shader::LIB_WGSL` (for
/// `raa_fbm3`) and reads only its parameters, so the env cube pass and the background
/// contributor share it. Algorithm: zenith/horizon/ground gradient (horizon blend by
/// `pow(1 - max(dir.y, 0), softness)`), Henyey-Greenstein (g 0.76) sun glow, sun disc with limb
/// darkening, cloud layer by ray-plane intersection at the cloud altitude with a 5-octave
/// domain-warped fbm density over the coverage threshold, a 4-step light march toward the sun
/// with Beer-powder transmittance, a silver lining from the HG forward lobe (g 0.6), and a fade
/// toward the horizon. Finite for every unit `dir`, including zero coverage.
pub const SKY_WGSL: &str = include_str!("wgsl/sky.wgsl");
