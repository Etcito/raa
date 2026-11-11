//! WGSL modules of the M2 volumetric cloud backend.
//!
//! The cloud noise bake is self-contained: it has no prelude, groups 0 and 1 are empty
//! and both storage outputs are explicit group-2 bindings. Later cards add the cloud
//! march modules here.

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

// Contract check until the ground-shadow card builds a march pass from this builder.
const _: fn() -> String = cloud_volume_wgsl;
