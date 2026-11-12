//! Procedural sky of the lit renderer (S12 `sky.env`, `sky.background`).
//!
//! [`SkyDesc`] describes a zenith/horizon/ground gradient, a sun disc and a procedural cloud
//! layer ([`CloudDesc`]) in scene-referred linear HDR. [`SkyDesc::preset`] gives the built-in
//! moods (`day`, `snow`, `jungle`, `night`) and [`SkyDesc::validate`] refuses invalid
//! descriptions with `sky.invalid`.
//!
//! [`SkyFeature::install`] adds the FeaturePasses `sky.atmosphere` (order 5: records the
//! Milky Way bake, LUTs and the directional transmission), `sky.cloud-lighting` (order 8: the
//! backend's panorama/transmittance work), `sky.env` (order 10: renders the SKY_ENV cube
//! from [`SKY_WGSL`]'s `raa_sky_radiance` and filters it through raa-ibl, when the sky changed
//! or every frame per [`EnvUpdate`]), `sky.cloud-screen` (order 55: the screen march after the
//! prepass) and the forward contributor `sky.background` (order 900: the per-pixel sky at
//! depth 1). The returned [`SkyHandle`] sets and reads the description and
//! [`SkyHandle::diagnostics`] reports the typed per-view submission lifecycle. Background
//! mode: `sky.env` writes the constant mode each frame and `sky.background` switches it to
//! sky while enabled, so disabling the contributor brings back the flat background colour.

/// Holds the native GPU test lock for the rest of a test. Unserialized tests that create
/// software GPU devices in parallel crash the process with SIGSEGV inside `libvulkan`
/// (reached from `wgpu` device creation, `GpuContext::from_adapter`, on a test thread).
/// Every test that creates a software GPU device holds this guard from its first line, so
/// the guard is released only after the test's device and its resources have been dropped.
#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn gpu_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU_TESTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

mod atmosphere;
mod background;
mod cloud_backend;
mod desc;
mod env_pass;
mod feature;
mod observations;
mod shaders;
mod state;
mod time_of_day;
mod weather;

pub use cloud_backend::{
    CLOUD_SCREEN_UNKNOWN_ALPHA, CloudFrame, CloudInvalidation, CloudViews, M2_CLOUD_SOURCE_PRESENT,
    cloud_backend_names, registered_cloud_backend_names,
};
pub use desc::{
    AtmosphereDesc, CloudDesc, EnvUpdate, LegacyCloudFields, SkyDesc, SkyModel, SkyPreset,
    SkyTransaction,
};
pub use feature::{SkyFeature, SkyHandle};
pub use observations::*;
pub use time_of_day::*;
pub use weather::*;

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
