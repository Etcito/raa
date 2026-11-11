//! WGSL modules of the M2 volumetric cloud backend.
//!
//! The cloud noise bake is self-contained: it has no prelude, groups 0 and 1 are empty
//! and both storage outputs are explicit group-2 bindings. The cloud march modules
//! compose `cloud_volume_wgsl()`, their output binding 10 and their compute function.

/// The cloud noise bake: shape (128^3), detail (32^3) and weather (512^2).
const NOISE_SRC: &str = include_str!("wgsl/noise.wgsl");

/// The noise compute entries in dispatch order.
pub(super) const NOISE_ENTRIES: [&str; 3] = [
    "raa_cloud_noise_shape",
    "raa_cloud_noise_detail",
    "raa_cloud_noise_weather",
];

/// The standalone cloud noise module; group 2 binding 0 is the 3D volume write and
/// binding 1 the 2D weather-map write, both `rgba8unorm`.
pub(super) fn noise_module_wgsl() -> String {
    NOISE_SRC.to_string()
}

/// The coverage CDF of the weather map's red channel (256 bins).
const COVERAGE_SRC: &str = include_str!("wgsl/coverage.wgsl");

/// The coverage compute entries in dispatch order.
pub(super) const COVERAGE_ENTRIES: [&str; 2] = [
    "raa_cloud_coverage_histogram",
    "raa_cloud_coverage_accumulate",
];

/// The standalone coverage module; group 2 binding 0 is the weather map, 1 the
/// atomic bins and 2 the cumulative distribution.
pub(super) fn coverage_module_wgsl() -> String {
    COVERAGE_SRC.to_string()
}

/// The read-only physical atmosphere helpers every cloud march module composes first.
const ATMOSPHERE_SRC: &str = include_str!("../wgsl/atmosphere.wgsl");

/// The uniform record and density model; the march share appends its lighting and march.
const CLOUD_VOLUME_SRC: &str = include_str!("wgsl/cloud_volume.wgsl");

/// The private group-2 bindings 0-9 every cloud march module declares.
const CLOUD_BINDINGS: &str = "\
@group(2) @binding(0) var<uniform> raa_cloud: CloudGpu;
@group(2) @binding(1) var<uniform> raa_cloud_sky: PhysicalSkyGpu;
@group(2) @binding(2) var raa_cloud_shape: texture_3d<f32>;
@group(2) @binding(3) var raa_cloud_detail: texture_3d<f32>;
@group(2) @binding(4) var raa_cloud_weather_map: texture_2d<f32>;
@group(2) @binding(5) var raa_cloud_transmittance_lut: texture_2d<f32>;
@group(2) @binding(6) var raa_cloud_sky_view: texture_2d<f32>;
@group(2) @binding(7) var raa_cloud_noise_sampler: sampler;
@group(2) @binding(8) var raa_cloud_lut_sampler: sampler;
@group(2) @binding(9) var<storage, read> raa_cloud_coverage_cdf: array<f32, 256>;";

/// The common part of every cloud march module: the atmosphere helpers, the group-2
/// bindings 0-9 and the whole cloud uniform/density text, with no compute function.
/// A pass module appends its output binding and its compute function.
pub(super) fn cloud_volume_wgsl() -> String {
    format!("{ATMOSPHERE_SRC}\n{CLOUD_BINDINGS}\n{CLOUD_VOLUME_SRC}")
}

/// The march output every cloud march module declares after the common text.
const CLOUD_OUTPUT: &str =
    "@group(2) @binding(10) var raa_cloud_output: texture_storage_2d<rgba16float, write>;";

/// The ground transmittance pass: 16 density samples toward the dominant light.
const SHADOW_SRC: &str = include_str!("wgsl/shadow.wgsl");

/// The ground shadow module (entry `raa_cloud_shadow_main`): the common cloud text,
/// the output binding 10 and the shadow pass.
pub(super) fn shadow_module_wgsl() -> String {
    format!("{}\n{CLOUD_OUTPUT}\n{SHADOW_SRC}", cloud_volume_wgsl())
}

/// The mean of the ground shadow map, written to texel (0,0) of layer 1.
const SHADOW_MEAN_SRC: &str = include_str!("wgsl/shadow_mean.wgsl");

/// The standalone shadow mean module (entry `raa_cloud_shadow_mean_main`); group 2
/// binding 0 is the map and binding 1 the `rgba16float` mean write.
pub(super) fn shadow_mean_module_wgsl() -> String {
    SHADOW_MEAN_SRC.to_string()
}

/// The cloud panorama pass: 32 primary and 4 light samples per sky-view direction.
const PANORAMA_SRC: &str = include_str!("wgsl/panorama.wgsl");

/// The panorama module (entry `raa_cloud_panorama_main`): the common cloud text, the
/// output binding 10 and the panorama pass.
pub(super) fn panorama_module_wgsl() -> String {
    format!("{}\n{CLOUD_OUTPUT}\n{PANORAMA_SRC}", cloud_volume_wgsl())
}
