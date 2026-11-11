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
