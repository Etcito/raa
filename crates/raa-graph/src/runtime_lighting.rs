//! Parameter conversion shared by spherical GI validation and the executor.

use raa_sdf_trace::mesh::{SphericalGiDebug, SphericalGiSettings};

use crate::catalog::{FeatureId, ParamValue};

pub(crate) const SPHERICAL_TYPE: &str = "pass.gi.spherical";
pub(crate) const SPHERICAL: FeatureId = FeatureId::Other("sphericalGi");
pub(crate) const ISOLATION: FeatureId = FeatureId::Other("lightingIsolation");
pub(crate) const SPHERICAL_KEYS: &[&str] = &[
    "sphereSubdivision",
    "exponentialPower",
    "clipStart",
    "zMax",
    "channelMask",
    "debug",
    "maxProbes",
    "captureFaces",
    "captureMs",
    "burstMs",
    "probeCache",
    "skySamples",
    "lightSamples",
    "parallaxRange",
    "autoRange",
    "enabled",
];

const DEBUG: [(&str, SphericalGiDebug); 5] = [
    ("off", SphericalGiDebug::Off),
    ("boundaries", SphericalGiDebug::Boundaries),
    ("occupancy", SphericalGiDebug::Occupancy),
    ("age", SphericalGiDebug::Age),
    ("light-hash", SphericalGiDebug::LightHash),
];
pub(crate) fn spherical_get(settings: SphericalGiSettings, key: &str) -> Option<ParamValue> {
    let grid = settings.scheduler.grid;
    Some(match key {
        "enabled" => ParamValue::Bool(settings.enabled),
        "sphereSubdivision" => ParamValue::Int(i64::from(grid.angular_resolution.ilog2())),
        "exponentialPower" => ParamValue::Number(f64::from(grid.radial_growth).log2()),
        "clipStart" => ParamValue::Number(f64::from(grid.clip_start)),
        "zMax" => ParamValue::Number(f64::from(grid.z_max)),
        "channelMask" => ParamValue::Int(i64::from(settings.channels)),
        "maxProbes" => ParamValue::Int(i64::from(settings.scheduler.max_cells)),
        "captureFaces" => ParamValue::Int(i64::from(settings.budget.near)),
        "captureMs" => ParamValue::Number(f64::from(settings.capture_ms)),
        "burstMs" => ParamValue::Number(f64::from(settings.burst_ms)),
        "probeCache" => ParamValue::Int(i64::from(settings.scheduler.cache_cells)),
        "skySamples" => ParamValue::Int(i64::from(settings.sky_samples)),
        "lightSamples" => ParamValue::Int(i64::from(settings.light_samples)),
        "parallaxRange" => ParamValue::Number(f64::from(settings.parallax_range)),
        "autoRange" => ParamValue::Bool(settings.scheduler.auto_range),
        "debug" => ParamValue::Enum(
            DEBUG
                .iter()
                .find(|(_, value)| *value == settings.debug)?
                .0
                .into(),
        ),
        _ => return None,
    })
}

pub(crate) fn spherical_set(
    settings: &mut SphericalGiSettings,
    key: &str,
    value: &ParamValue,
) -> Result<(), String> {
    let grid = &mut settings.scheduler.grid;
    match (key, value) {
        ("enabled", ParamValue::Bool(enabled)) => settings.enabled = *enabled,
        ("sphereSubdivision", ParamValue::Int(value)) if (0..=6).contains(value) => {
            grid.angular_resolution = 1 << *value as u32;
        }
        ("exponentialPower", ParamValue::Number(value)) if (0.0..=2.0).contains(value) => {
            grid.radial_growth = value.exp2() as f32;
        }
        ("clipStart", ParamValue::Number(value)) if (0.0..=f64::from(f32::MAX)).contains(value) => {
            grid.clip_start = *value as f32;
        }
        ("zMax", ParamValue::Number(value)) if (0.0..=f64::from(f32::MAX)).contains(value) => {
            grid.z_max = *value as f32;
        }
        ("channelMask", ParamValue::Int(value)) if (0..=7).contains(value) => {
            settings.channels = *value as u32;
        }
        ("maxProbes", ParamValue::Int(value)) if (16..=4096).contains(value) => {
            settings.scheduler.max_cells = *value as u32;
        }
        // Far shells get half of the near face budget, as in the original 24/12 split.
        ("captureFaces", ParamValue::Int(value)) if (1..=1024).contains(value) => {
            settings.budget.near = *value as u32;
            settings.budget.far = (*value as u32).div_ceil(2);
        }
        ("captureMs", ParamValue::Number(value)) if (0.0..=50.0).contains(value) => {
            settings.capture_ms = *value as f32;
        }
        ("burstMs", ParamValue::Number(value)) if (0.0..=50.0).contains(value) => {
            settings.burst_ms = *value as f32;
        }
        ("probeCache", ParamValue::Int(value)) if (0..=4096).contains(value) => {
            settings.scheduler.cache_cells = *value as u32;
        }
        ("skySamples", ParamValue::Int(value)) if (0..=64).contains(value) => {
            settings.sky_samples = *value as u32;
        }
        ("lightSamples", ParamValue::Int(value)) if (0..=64).contains(value) => {
            settings.light_samples = *value as u32;
        }
        ("parallaxRange", ParamValue::Number(value)) if (0.0..=10000.0).contains(value) => {
            settings.parallax_range = *value as f32;
        }
        ("autoRange", ParamValue::Bool(value)) => settings.scheduler.auto_range = *value,
        ("debug", ParamValue::Enum(value)) => {
            settings.debug = DEBUG
                .iter()
                .find(|(name, _)| name == value)
                .map(|(_, debug)| *debug)
                .ok_or_else(|| format!("unknown spherical diagnostic '{value}'"))?;
        }
        _ => return Err(format!("invalid spherical setting {key}")),
    }
    Ok(())
}
