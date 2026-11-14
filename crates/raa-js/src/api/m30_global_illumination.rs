//! m30 global illumination, share 1 (S36): the pure helpers of the global-illumination API (the
//! settings and quality patch parsers, the settings, quality, instance and stats entries), the
//! `GiWire` model they convert through and the api state `GiApiState`.
//!
//! The helpers are plain Rust, so `tests/global_illumination.rs` includes this file and exercises
//! them natively: the patch refusals (`gi.invalid` with the key name), the settings and quality
//! key tables, the instance entries and the stats entries. The wasm32 runtime (the feature
//! install, the frame hook and the seven S36 Engine methods) is a later card; this file carries
//! no wasm-bindgen code today.
//!
//! `SETTING_KEYS` and `QUALITY_KEYS` are the S36 key orders. Every refusal is `gi.invalid` with
//! the message `<key>: <reason>`; ranges and domains are left to `GiSettingsPatch::apply` and
//! `GiQualityPatch::apply`, which validate the merged record atomically.

// Some helpers are reached only by the host tests or only by the wasm32 runtime of a later card,
// so they are dead in one of the two builds.
#![allow(dead_code)]

use std::collections::BTreeMap;

use raa_core::RaaError;
use raa_gi::error::invalid;
use raa_gi::{
    DebugView, Driver, GiQuality, GiQualityPatch, GiSettings, GiSettingsPatch, GiStats, InstanceGi,
};

/// The S36 settings keys in their key order: the `GlobalIlluminationSettings` names of the JS API.
pub(crate) const SETTING_KEYS: [&str; 7] = [
    "enabled",
    "intensity",
    "environmentDiffuse",
    "localLightShadows",
    "reflections",
    "screenTraces",
    "debugView",
];

/// The S36 quality keys in their key order: the `GiQuality` names of the JS API.
pub(crate) const QUALITY_KEYS: [&str; 3] = ["resolution", "levels", "probeRays"];

/// A JS value at the wasm boundary, reduced to the shapes the S36 patches read: `Undefined` is an
/// omission, `Null` and every other type are refusals, `Other` stands for any value that is not a
/// boolean, number or string.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum GiWire {
    Undefined,
    Null,
    Bool(bool),
    Number(f64),
    Text(String),
    Other,
}

/// Parses a `setGlobalIllumination` patch in entry order. `Undefined` means omission for a known
/// key; an unknown key is refused whatever its value. `Null` and a wrong type are refused here;
/// the ranges are `GiSettingsPatch::apply`'s. Every refusal is `gi.invalid` with the message
/// `<key>: <reason>`.
///
/// Handoff: `api-m30: global_illumination.rs tests` calls `parse_settings_patch`,
/// `parse_quality_patch`, `settings_entries`, `quality_entries`, `instance_entries`,
/// `stats_entries`, `GiWire` and `GiApiState` exported by this file through `#[path]`; no state
/// crosses (pure functions over `GiWire`). Today this function walks the entries and returns the
/// `raa_gi::GiSettingsPatch`; it does not merge the patch.
///
/// Handoff: `api-m30: wasm runtime, Engine methods and observe_submission forwarding` calls
/// `parse_settings_patch`, `parse_quality_patch`, `settings_entries`, `quality_entries`,
/// `instance_entries` and `stats_entries` from its `web` module and stores requests in
/// `GiApiState.requests`. Today `GiApiState` holds only `requests`.
pub(crate) fn parse_settings_patch(
    entries: &[(String, GiWire)],
) -> Result<GiSettingsPatch, RaaError> {
    let mut patch = GiSettingsPatch::default();
    for (key, value) in entries {
        let key = key.as_str();
        if !SETTING_KEYS.contains(&key) {
            return Err(invalid(key, "unknown key"));
        }
        if *value == GiWire::Undefined {
            continue;
        }
        match key {
            "enabled" => patch.enabled = Some(boolean(key, value)?),
            "intensity" => patch.intensity = Some(number(key, value)?),
            "environmentDiffuse" => patch.environment_diffuse = Some(number(key, value)?),
            "localLightShadows" => patch.local_light_shadows = Some(boolean(key, value)?),
            "reflections" => patch.reflections = Some(boolean(key, value)?),
            "screenTraces" => patch.screen_traces = Some(boolean(key, value)?),
            // `debugView`, the last key of SETTING_KEYS.
            _ => patch.debug_view = Some(debug_view(value)?),
        }
    }
    Ok(patch)
}

/// Parses a `setGiQuality` patch in entry order with the same rules as
/// [`parse_settings_patch`]: an unknown key, `Null` and a wrong type are refused here, and the
/// domains are `GiQualityPatch::apply`'s. All three keys are numbers.
pub(crate) fn parse_quality_patch(
    entries: &[(String, GiWire)],
) -> Result<GiQualityPatch, RaaError> {
    let mut patch = GiQualityPatch::default();
    for (key, value) in entries {
        let key = key.as_str();
        if !QUALITY_KEYS.contains(&key) {
            return Err(invalid(key, "unknown key"));
        }
        if *value == GiWire::Undefined {
            continue;
        }
        let parsed = number(key, value)?;
        match key {
            "resolution" => patch.resolution = Some(parsed),
            "levels" => patch.levels = Some(parsed),
            // `probeRays`, the last key of QUALITY_KEYS.
            _ => patch.probe_rays = Some(parsed),
        }
    }
    Ok(patch)
}

/// `null` and a value of another type, named by the expected type.
fn scalar_refusal(key: &str, value: &GiWire, expected: &str) -> RaaError {
    match value {
        GiWire::Null => invalid(key, "must not be null"),
        _ => invalid(key, expected),
    }
}

fn boolean(key: &str, value: &GiWire) -> Result<bool, RaaError> {
    match value {
        GiWire::Bool(b) => Ok(*b),
        _ => Err(scalar_refusal(key, value, "expected a boolean")),
    }
}

fn number(key: &str, value: &GiWire) -> Result<f64, RaaError> {
    match value {
        GiWire::Number(n) => Ok(*n),
        _ => Err(scalar_refusal(key, value, "expected a number")),
    }
}

fn debug_view(value: &GiWire) -> Result<DebugView, RaaError> {
    const REASON: &str = "expected one of none, voxels, radiance, distance, levels";
    match value {
        GiWire::Text(name) => DebugView::parse(name).ok_or_else(|| invalid("debugView", REASON)),
        _ => Err(invalid("debugView", REASON)),
    }
}

/// The resolved settings as `GlobalIlluminationSettings` entries, in [`SETTING_KEYS`] order.
pub(crate) fn settings_entries(s: &GiSettings) -> Vec<(&'static str, GiWire)> {
    vec![
        ("enabled", GiWire::Bool(s.enabled)),
        ("intensity", GiWire::Number(f64::from(s.intensity))),
        (
            "environmentDiffuse",
            GiWire::Number(f64::from(s.environment_diffuse)),
        ),
        ("localLightShadows", GiWire::Bool(s.local_light_shadows)),
        ("reflections", GiWire::Bool(s.reflections)),
        ("screenTraces", GiWire::Bool(s.screen_traces)),
        ("debugView", GiWire::Text(s.debug_view.as_str().to_string())),
    ]
}

/// The resolved quality as `GiQuality` entries, in [`QUALITY_KEYS`] order.
pub(crate) fn quality_entries(q: &GiQuality) -> Vec<(&'static str, GiWire)> {
    vec![
        ("resolution", GiWire::Number(f64::from(q.resolution))),
        ("levels", GiWire::Number(f64::from(q.levels))),
        ("probeRays", GiWire::Number(f64::from(q.probe_rays))),
    ]
}

/// The `InstanceGlobalIllumination` entries: `requested` (the explicit request, `Null` when
/// following the default), `voxelized`, `driver` ('forward' | 'virtual', or `Null` when the group
/// is not voxelized), `code` and `message` (the first refusal, or `Null`).
pub(crate) fn instance_entries(i: &InstanceGi) -> Vec<(&'static str, GiWire)> {
    let driver = match i.driver {
        Some(Driver::Forward) => GiWire::Text("forward".to_string()),
        Some(Driver::Virtual) => GiWire::Text("virtual".to_string()),
        None => GiWire::Null,
    };
    vec![
        ("requested", i.requested.map_or(GiWire::Null, GiWire::Bool)),
        ("voxelized", GiWire::Bool(i.voxelized)),
        ("driver", driver),
        ("code", text_or_null(i.code)),
        ("message", text_or_null(i.message)),
    ]
}

/// The `GiStats` entries in the S36 key order. `quality` is a placeholder (`Other`): the wasm
/// runtime replaces it with the [`quality_entries`] object at its position. Counters that are
/// `None` are `Null`; `allocatedBytes` is a `u64` widened to `f64`.
pub(crate) fn stats_entries(s: &GiStats) -> Vec<(&'static str, GiWire)> {
    vec![
        ("active", GiWire::Bool(s.active)),
        ("inactiveReason", text_or_null(s.inactive_reason)),
        ("quality", GiWire::Other),
        (
            "finestVoxelMetres",
            GiWire::Number(f64::from(s.finest_voxel_metres)),
        ),
        (
            "voxelizedForwardInstances",
            GiWire::Number(f64::from(s.voxelized_forward_instances)),
        ),
        (
            "voxelizedVirtualInstances",
            GiWire::Number(f64::from(s.voxelized_virtual_instances)),
        ),
        ("pendingRebuild", GiWire::Bool(s.pending_rebuild)),
        (
            "framesSinceRebuild",
            GiWire::Number(f64::from(s.frames_since_rebuild)),
        ),
        ("allocatedBytes", GiWire::Number(s.allocated_bytes as f64)),
        (
            "statsFrameIndex",
            number_or_null(s.stats_frame_index.map(|index| index as f64)),
        ),
        ("pending", GiWire::Bool(s.pending)),
    ]
}

fn text_or_null(value: Option<&'static str>) -> GiWire {
    value.map_or(GiWire::Null, |text| GiWire::Text(text.to_string()))
}

fn number_or_null(value: Option<f64>) -> GiWire {
    value.map_or(GiWire::Null, GiWire::Number)
}

/// The api state of the feature: the explicit instance requests (JS instance-group handle ->
/// request, the `setInstanceGlobalIllumination` values).
#[derive(Debug, Default)]
pub(crate) struct GiApiState {
    pub requests: BTreeMap<u32, bool>,
}
