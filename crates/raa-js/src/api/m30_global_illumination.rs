//! m30 global illumination, share 1 (S36): the pure helpers of the global-illumination API (the
//! settings and quality patch parsers, the settings, quality, instance and stats entries), the
//! `GiWire` model they convert through and the api state `GiApiState`.
//!
//! The helpers are plain Rust, so `tests/global_illumination.rs` includes this file and exercises
//! them natively: the patch refusals (`gi.invalid` with the key name), the settings and quality
//! key tables, the instance entries and the stats entries. The wasm32 part (`web`) holds the
//! feature install, the frame hook and the seven S36 Engine methods, each run through
//! `Engine::with_state`.
//!
//! crates/raa-js/src/lib.rs: out of scope for this share: consumers use the module path.
//!
//! `SETTING_KEYS` and `QUALITY_KEYS` are the S36 key orders. The screen-space keys `screenSpace`
//! and `screenSpaceQuality` (`gi.ssgi`, `SsgiQuality`) are the last two settings. Every refusal is
//! `gi.invalid` with the message `<key>: <reason>`; ranges and domains are left to
//! `GiSettingsPatch::apply` and `GiQualityPatch::apply`, which validate the merged record
//! atomically. `screenSpaceQuality` is refused unless it names `low`, `medium` or `high`.

// Some helpers are reached only by the host tests or only by the wasm32 runtime of a later card,
// so they are dead in one of the two builds.
#![allow(dead_code)]

use std::collections::BTreeMap;

use raa_core::RaaError;
use raa_gi::error::invalid;
use raa_gi::{
    DebugView, Driver, GiPreset, GiQuality, GiQualityPatch, GiSettings, GiSettingsPatch, GiStats,
    IndirectMode, InstanceGi, SsgiQuality,
};

/// The S36 settings keys in their key order: the `GlobalIlluminationSettings` names of the JS API.
pub(crate) const SETTING_KEYS: [&str; 18] = [
    "enabled",
    "intensity",
    "environmentDiffuse",
    "localLightShadows",
    "reflections",
    "screenTraces",
    "debugView",
    "screenSpace",
    "screenSpaceQuality",
    "indirect",
    "historyMax",
    "clampGamma",
    "spatialPasses",
    "restirMCap",
    "voxelFullRelight",
    "historyMin",
    "voxelHistoryMax",
    "voxelRays",
];

/// Expands a per-vertex colour buffer of `components` floats per vertex (3 = rgb, 4 = rgba) to the
/// four-floats-per-vertex rgba stream `GiHandle::set_vertex_colors` takes; rgb gets alpha 1.
/// `components` other than 3 or 4, or a length that is not a multiple of it, is `gi.invalid`.
/// Finiteness is checked by raa-gi.
pub(crate) fn colors_to_rgba(values: &[f32], components: u32) -> Result<Vec<f32>, RaaError> {
    if components != 3 && components != 4 {
        return Err(invalid("components", "must be 3 or 4"));
    }
    let n = components as usize;
    if !values.len().is_multiple_of(n) {
        return Err(invalid(
            "colors",
            &format!("length must be a multiple of {components}"),
        ));
    }
    if n == 4 {
        return Ok(values.to_vec());
    }
    Ok(values
        .chunks_exact(3)
        .flat_map(|c| [c[0], c[1], c[2], 1.0])
        .collect())
}

/// The native-testable body of the colour setters: `None` (no buffer) stays `None`, meaning the
/// albedo is the material base colour x instance tint; a `Float32Array` is read as is and a
/// `Uint8Array` is normalised by 1/255. `components` is 3 or 4 (default 4).
pub(crate) fn colour_data(
    floats: Option<&[f32]>,
    bytes: Option<&[u8]>,
    components: Option<u32>,
) -> Result<Option<Vec<f32>>, RaaError> {
    let n = components.unwrap_or(4);
    match (floats, bytes) {
        (Some(f), _) => colors_to_rgba(f, n).map(Some),
        (None, Some(b)) => {
            let raw: Vec<f32> = b.iter().map(|&v| f32::from(v) / 255.0).collect();
            colors_to_rgba(&raw, n).map(Some)
        }
        (None, None) => Ok(None),
    }
}

/// Forwards one colour stream (or `None`) to every id through `sink`, which is
/// `GiHandle::set_vertex_colors` in the wasm method.
pub(crate) fn forward_colors<I: Copy>(
    ids: &[I],
    data: Option<&[f32]>,
    mut sink: impl FnMut(I, Option<&[f32]>) -> Result<(), RaaError>,
) -> Result<(), RaaError> {
    for &id in ids {
        sink(id, data)?;
    }
    Ok(())
}

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
            "debugView" => patch.debug_view = Some(debug_view(value)?),
            "screenSpace" => patch.screen_space = Some(boolean(key, value)?),
            "screenSpaceQuality" => patch.screen_space_quality = Some(screen_space_quality(value)?),
            "indirect" => patch.indirect = Some(indirect_mode(value)?),
            "historyMax" => patch.history_max = Some(number(key, value)?),
            "clampGamma" => patch.clamp_gamma = Some(number(key, value)?),
            "spatialPasses" => patch.spatial_passes = Some(number(key, value)?),
            "restirMCap" => patch.restir_m_cap = Some(number(key, value)?),
            "voxelFullRelight" => patch.voxel_full_relight = Some(boolean(key, value)?),
            "historyMin" => patch.history_min = Some(number(key, value)?),
            "voxelHistoryMax" => patch.voxel_history_max = Some(number(key, value)?),
            // `voxelRays`, the last key of SETTING_KEYS.
            _ => patch.voxel_rays = Some(number(key, value)?),
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
    const REASON: &str = "expected one of none, voxels, radiance, distance, levels, probes, \
                          hitType, history, historyLength, reservoirs, rawDenoised";
    match value {
        GiWire::Text(name) => DebugView::parse(name).ok_or_else(|| invalid("debugView", REASON)),
        _ => Err(invalid("debugView", REASON)),
    }
}

/// The `screenSpaceQuality` name: exactly `low`, `medium` or `high`. Any other value, or a value
/// that is not text, is refused with the same reason.
fn screen_space_quality(value: &GiWire) -> Result<SsgiQuality, RaaError> {
    const REASON: &str = "expected one of low, medium, high";
    let parsed = match value {
        GiWire::Text(name) => SsgiQuality::parse(name),
        _ => None,
    };
    parsed.ok_or_else(|| invalid("screenSpaceQuality", REASON))
}

/// The `indirect` name: exactly `probes` or `restir`.
fn indirect_mode(value: &GiWire) -> Result<IndirectMode, RaaError> {
    const REASON: &str = "expected one of probes, restir";
    let parsed = match value {
        GiWire::Text(name) => IndirectMode::parse(name),
        _ => None,
    };
    parsed.ok_or_else(|| invalid("indirect", REASON))
}

/// The `setGiPreset` name: exactly `performance`, `balanced` or `quality`.
pub(crate) fn preset(value: &GiWire) -> Result<GiPreset, RaaError> {
    const REASON: &str = "expected one of performance, balanced, quality";
    let parsed = match value {
        GiWire::Text(name) => GiPreset::parse(name),
        _ => None,
    };
    parsed.ok_or_else(|| invalid("preset", REASON))
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
        ("screenSpace", GiWire::Bool(s.screen_space)),
        (
            "screenSpaceQuality",
            GiWire::Text(s.screen_space_quality.as_str().to_string()),
        ),
        ("indirect", GiWire::Text(s.indirect.as_str().to_string())),
        ("historyMax", GiWire::Number(f64::from(s.history_max))),
        ("clampGamma", GiWire::Number(f64::from(s.clamp_gamma))),
        ("spatialPasses", GiWire::Number(f64::from(s.spatial_passes))),
        ("restirMCap", GiWire::Number(f64::from(s.restir_m_cap))),
        ("voxelFullRelight", GiWire::Bool(s.voxel_full_relight)),
        ("historyMin", GiWire::Number(f64::from(s.history_min))),
        (
            "voxelHistoryMax",
            GiWire::Number(f64::from(s.voxel_history_max)),
        ),
        ("voxelRays", GiWire::Number(f64::from(s.voxel_rays))),
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
        (
            "historyLengthMean",
            number_or_null(s.history_length_mean.map(f64::from)),
        ),
        (
            "rejectedPercent",
            number_or_null(s.rejected_percent.map(f64::from)),
        ),
        (
            "clampedPercent",
            number_or_null(s.clamped_percent.map(f64::from)),
        ),
        (
            "litHistoryAge",
            number_or_null(s.lit_history_age.map(|age| age as f64)),
        ),
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

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install};

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use crate::api::ViewsState;
    use crate::convert::{UNKNOWN_HANDLE, api_error};
    use crate::engine::{Engine, InstallCtx};
    use crate::feature_map::FEATURE_MISSING;
    use crate::state::EngineState;
    use js_sys::{Array, Object, Reflect};
    use raa_core::GpuContext;
    use raa_gi::{GiFeature, GiHandle};
    use raa_ibl::IblHandle;
    use raa_vg::VgHandle;
    use raa_web::FrameInput;
    use wasm_bindgen::{JsCast, JsValue, prelude::wasm_bindgen};

    /// Installs the feature after `m08_sky_ibl` and `m29_virtual_geometry` (file-name order). The
    /// m29 `VgHandle` and the m08 `IblHandle` are passed when present; `GiFeature::install`
    /// returns the `GiHandle`, which is inserted with a default `GiApiState`. The engine's SDF
    /// trace handle, inserted before `api::install_all`, is attached so the mid-tier rays also
    /// see SDF content.
    ///
    /// Handoff: card 'raa-gi: GiShared depth-pyramid and trace-scene seams, screen pass publishes
    /// its pyramid, raa-taa and raa-sdf-trace dependencies' supplies
    /// [`GiHandle::set_trace_scene`] in `crates/raa-gi/src/feature.rs:288`.
    pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
        let vg = ctx.features.get::<VgHandle>().cloned();
        let ibl = ctx.features.get::<IblHandle>().cloned();
        let handle = GiFeature::install(ctx.gpu, ctx.pipeline, ctx.lit, vg, ibl)?;
        if let Some(trace) = ctx.features.get::<raa_sdf_trace::TraceHandle>().cloned() {
            handle.set_trace_scene(Some(trace));
        }
        ctx.features.insert(handle);
        ctx.features.insert(GiApiState::default());
        Ok(())
    }

    /// Runs before `lit.prepare` once per Engine invocation: without a `GiHandle` it returns.
    /// Otherwise the camera, viewport 0 and the submitted-view count go to the handle, with the
    /// sun and the local-light count; a camera cut invalidates the history; the requests of
    /// removed instance groups are forgotten; then `prepare_frame` runs.
    pub(crate) fn frame(
        st: &mut EngineState,
        gpu: &GpuContext,
        input: &FrameInput,
    ) -> Result<(), RaaError> {
        let Some(gi) = st.features.get::<GiHandle>().cloned() else {
            return Ok(());
        };
        let camera = st.scene.camera;
        let viewport = viewport0(st, input);
        let submitted = submitted_views(st);
        gi.set_view(&camera, viewport, submitted);
        gi.set_sun(&st.scene.light);
        gi.set_local_light_count(st.lit.local_lights().active_count());
        if st.scene_invocation.is_discontinuous(0) {
            gi.invalidate_history();
        }
        forget_vanished(st, &gi);
        gi.prepare_frame(gpu, &mut st.scene, &mut st.lit);
        Ok(())
    }

    /// Drops the request of every instance group that no longer exists (`removeInstance`) from
    /// the handle and from `GiApiState.requests`.
    fn forget_vanished(st: &mut EngineState, gi: &GiHandle) {
        let vanished: Vec<u32> = st
            .features
            .get::<GiApiState>()
            .map_or_else(Vec::new, |api| {
                api.requests
                    .keys()
                    .copied()
                    .filter(|&tag| st.instance_group(tag).is_err())
                    .collect()
            });
        let Some(api) = st.features.get_mut::<GiApiState>() else {
            return;
        };
        for tag in vanished {
            gi.forget_instance(tag);
            api.requests.remove(&tag);
        }
    }

    /// The submitted-view count: the viewport count of `ViewsState`, or 1 without it.
    fn submitted_views(st: &EngineState) -> u8 {
        st.features.get::<ViewsState>().map_or(1, |views| {
            u8::try_from(views.viewports().len()).unwrap_or(u8::MAX)
        })
    }

    /// The size in backing pixels of viewport 0, computed as `Engine::update` does; the canvas
    /// size without a `ViewsState`.
    fn viewport0(st: &EngineState, input: &FrameInput) -> (u32, u32) {
        let Some(rect) = st
            .features
            .get::<ViewsState>()
            .and_then(|views| views.viewports().first())
        else {
            return input.size;
        };
        let (w, h) = (f64::from(input.size.0), f64::from(input.size.1));
        let x = (rect.x * w).round() as u32;
        let y = (rect.y * h).round() as u32;
        let right = ((rect.x + rect.width) * w).round() as u32;
        let bottom = ((rect.y + rect.height) * h).round() as u32;
        (right.saturating_sub(x), bottom.saturating_sub(y))
    }

    /// The own enumerable keys of a JS object, in order.
    fn own_keys(value: &JsValue) -> Vec<String> {
        Object::keys(value.unchecked_ref::<Object>())
            .iter()
            .filter_map(|key| key.as_string())
            .collect()
    }

    /// A wire value of a JS value: `undefined`, `null`, a boolean, a number, a string, or
    /// `Other` for anything else.
    fn wire(value: &JsValue) -> GiWire {
        if value.is_undefined() {
            GiWire::Undefined
        } else if value.is_null() {
            GiWire::Null
        } else if let Some(b) = value.as_bool() {
            GiWire::Bool(b)
        } else if let Some(n) = value.as_f64() {
            GiWire::Number(n)
        } else if let Some(text) = value.as_string() {
            GiWire::Text(text)
        } else {
            GiWire::Other
        }
    }

    /// A JS value of an entry. `Other` never reaches JS: `stats_js` replaces `quality`.
    fn wire_js(value: &GiWire) -> JsValue {
        match value {
            GiWire::Undefined => JsValue::UNDEFINED,
            GiWire::Null | GiWire::Other => JsValue::NULL,
            GiWire::Bool(b) => JsValue::from_bool(*b),
            GiWire::Number(n) => JsValue::from_f64(*n),
            GiWire::Text(text) => JsValue::from_str(text),
        }
    }

    /// A fresh plain object of the entries, in order.
    fn entries_js(entries: Vec<(&'static str, GiWire)>) -> JsValue {
        let out = Object::new();
        for (key, value) in entries {
            let _ = Reflect::set(&out, &JsValue::from_str(key), &wire_js(&value));
        }
        out.into()
    }

    /// The global-illumination feature handle, or `api.feature_missing` when it is not installed.
    fn gi_handle(st: &EngineState) -> Result<GiHandle, RaaError> {
        st.features
            .get::<GiHandle>()
            .cloned()
            .ok_or_else(|| api_error(FEATURE_MISSING, "global illumination is not installed"))
    }

    /// The entries of a patch argument: a plain object whose own keys are the patch. A non-object
    /// or an array is `gi.invalid` naming `name` (`<name>: must be an object`).
    fn patch_entries(patch: &JsValue, name: &str) -> Result<Vec<(String, GiWire)>, RaaError> {
        if !patch.is_object() || Array::is_array(patch) {
            return Err(invalid(name, "must be an object"));
        }
        Ok(own_keys(patch)
            .into_iter()
            .map(|key| {
                let value = wire(
                    &Reflect::get(patch, &JsValue::from_str(&key)).unwrap_or(JsValue::UNDEFINED),
                );
                (key, value)
            })
            .collect())
    }

    /// `setGlobalIllumination(patch)`: the patch is parsed, then applied to the handle; a refusal
    /// changes nothing.
    fn set_settings(st: &mut EngineState, patch: &JsValue) -> Result<(), RaaError> {
        let parsed = parse_settings_patch(&patch_entries(patch, "settings")?)?;
        gi_handle(st)?.set(parsed)
    }

    /// `setGiQuality(patch)`: the patch is parsed, then applied to the handle; a refusal changes
    /// nothing.
    fn set_quality_patch(st: &mut EngineState, patch: &JsValue) -> Result<(), RaaError> {
        let parsed = parse_quality_patch(&patch_entries(patch, "quality")?)?;
        gi_handle(st)?.set_quality(parsed)
    }

    /// `setGiPreset(name)`: the preset's settings and quality patches, applied in that order;
    /// an unknown name changes nothing.
    fn set_preset(st: &mut EngineState, name: &JsValue) -> Result<(), RaaError> {
        let chosen = preset(&wire(name))?;
        let gi = gi_handle(st)?;
        gi.set(chosen.settings_patch())?;
        gi.set_quality(chosen.quality_patch())
    }

    /// `setInstanceGlobalIllumination(instance, value)`: the group is checked first
    /// (`api.unknown_handle`), then `value` (true, false or null); the handle and the api state
    /// are updated together.
    fn set_instance_request(
        st: &mut EngineState,
        instance: u32,
        value: &JsValue,
    ) -> Result<(), RaaError> {
        st.instance_group(instance)?;
        let request = if value.is_null() {
            None
        } else if let Some(b) = value.as_bool() {
            Some(b)
        } else {
            return Err(invalid("value", "expected a boolean or null"));
        };
        gi_handle(st)?.set_instance(instance, request);
        let api = st
            .features
            .require_mut::<GiApiState>("global illumination api state")?;
        if let Some(b) = request {
            api.requests.insert(instance, b);
        } else {
            api.requests.remove(&instance);
        }
        Ok(())
    }

    /// `instanceGlobalIllumination(instance)`: the request and the membership of a group, as a
    /// fresh object. The group is checked first.
    fn instance_js(st: &EngineState, instance: u32) -> Result<JsValue, RaaError> {
        st.instance_group(instance)?;
        Ok(entries_js(instance_entries(
            &gi_handle(st)?.instance(instance),
        )))
    }

    /// `giStats()`: a fresh object in the S36 key order; `quality` is a fresh object of its own.
    fn stats_js(st: &EngineState) -> Result<JsValue, RaaError> {
        let stats = gi_handle(st)?.stats();
        let out = Object::new();
        for (key, value) in stats_entries(&stats) {
            let js = if key == "quality" {
                entries_js(quality_entries(&stats.quality))
            } else {
                wire_js(&value)
            };
            let _ = Reflect::set(&out, &JsValue::from_str(key), &js);
        }
        Ok(out.into())
    }

    /// The optional colour buffer of a setter: `None` for null or undefined (the stream is
    /// removed), otherwise rgba floats. A `Float32Array` is read as is, a `Uint8Array` is
    /// normalised by 1/255; anything else is `gi.invalid`.
    fn colors_arg(value: &JsValue, components: Option<u32>) -> Result<Option<Vec<f32>>, RaaError> {
        if value.is_null() || value.is_undefined() {
            return Ok(None);
        }
        if let Some(f) = value.dyn_ref::<js_sys::Float32Array>() {
            colour_data(Some(&f.to_vec()), None, components)
        } else if let Some(u) = value.dyn_ref::<js_sys::Uint8Array>() {
            colour_data(None, Some(&u.to_vec()), components)
        } else {
            Err(invalid(
                "colors",
                "must be a Float32Array, a Uint8Array or null",
            ))
        }
    }

    /// `setMeshVertexColors(mesh, colors, components?)`: forwards to `GiHandle::set_vertex_colors`
    /// for every sub-mesh of the JS mesh handle.
    fn set_mesh_colors(
        st: &mut EngineState,
        mesh: u32,
        colors: &JsValue,
        components: Option<u32>,
    ) -> Result<(), RaaError> {
        let lit = st.mesh(mesh)?.lit;
        let ids = st
            .lit
            .submesh_ids(lit)
            .map(|ids| ids.to_vec())
            .ok_or_else(|| api_error(UNKNOWN_HANDLE, format!("mesh {mesh} does not exist")))?;
        let data = colors_arg(colors, components)?;
        let gi = gi_handle(st)?;
        forward_colors(&ids, data.as_deref(), |id, c| gi.set_vertex_colors(id, c))
    }

    #[wasm_bindgen(js_class = Engine)]
    impl Engine {
        /// Applies a `GlobalIlluminationPatch` atomically: an unknown key, null, a wrong type or an
        /// out-of-domain value throws `gi.invalid` naming the key, and nothing changes.
        #[wasm_bindgen(js_name = setGlobalIllumination)]
        pub fn set_global_illumination(&mut self, patch: JsValue) -> Result<(), JsValue> {
            self.with_state(|_, st| set_settings(st, &patch))
        }

        /// The resolved `GlobalIlluminationSettings`, as a fresh object.
        #[wasm_bindgen(js_name = globalIllumination)]
        pub fn global_illumination(&self) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| Ok(entries_js(settings_entries(&gi_handle(st)?.get()))))
        }

        /// Applies a `GiQualityPatch` atomically, with the same refusals as
        /// `setGlobalIllumination`; a change reallocates the volume at the next active frame.
        #[wasm_bindgen(js_name = setGiQuality)]
        pub fn set_gi_quality(&mut self, patch: JsValue) -> Result<(), JsValue> {
            self.with_state(|_, st| set_quality_patch(st, &patch))
        }

        /// Applies a GI preset (`performance`, `balanced`, `quality`): it sets `historyMax`,
        /// `clampGamma`, `spatialPasses`, `restirMCap`, `voxelFullRelight` and the probe rays,
        /// and leaves every other setting as it is. Another name throws `gi.invalid`.
        #[wasm_bindgen(js_name = setGiPreset)]
        pub fn set_gi_preset(&mut self, name: JsValue) -> Result<(), JsValue> {
            self.with_state(|_, st| set_preset(st, &name))
        }

        /// The current `GiQuality`, as a fresh object.
        #[wasm_bindgen(js_name = giQuality)]
        pub fn gi_quality(&self) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| Ok(entries_js(quality_entries(&gi_handle(st)?.quality()))))
        }

        /// Sets the request of an instance group: true, false, or null for the default membership.
        #[wasm_bindgen(js_name = setInstanceGlobalIllumination)]
        pub fn set_instance_global_illumination(
            &mut self,
            instance: u32,
            value: JsValue,
        ) -> Result<(), JsValue> {
            self.with_state(|_, st| set_instance_request(st, instance, &value))
        }

        /// The `InstanceGlobalIllumination` of an instance group, as a fresh object.
        #[wasm_bindgen(js_name = instanceGlobalIllumination)]
        pub fn instance_global_illumination(&self, instance: u32) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| instance_js(st, instance))
        }

        /// The `GiStats` of the latest published submission, as a fresh object.
        #[wasm_bindgen(js_name = giStats)]
        pub fn gi_stats(&self) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| stats_js(st))
        }
        /// Registers (Float32Array rgb(a) or Uint8Array rgb(a)) or removes (null/undefined) the
        /// per-vertex colour stream of a mesh for the forward GI voxelizer. `components` is 3 or
        /// 4 (default 4).
        #[wasm_bindgen(js_name = setMeshVertexColors)]
        pub fn set_mesh_vertex_colors(
            &mut self,
            mesh: u32,
            colors: JsValue,
            components: Option<u32>,
        ) -> Result<(), JsValue> {
            self.with_state(|_, st| set_mesh_colors(st, mesh, &colors, components))
        }

        /// As `setMeshVertexColors`, for section `ordinal` of the mesh's cluster store, one colour
        /// per `ClusterMesh` vertex: the virtualized-geometry raster draws the section with them
        /// (base colour x vertex colour, as the forward path) and the virtualized-geometry GI
        /// voxelizer reads them. Each installed feature of the two receives the stream; with
        /// neither installed it is `feature.missing` (global illumination).
        #[wasm_bindgen(js_name = setMeshClusterVertexColors)]
        pub fn set_mesh_cluster_vertex_colors(
            &mut self,
            mesh: u32,
            ordinal: u32,
            colors: JsValue,
            components: Option<u32>,
        ) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                st.mesh(mesh)?;
                let data = colors_arg(&colors, components)?;
                let vg = st.features.get::<VgHandle>().cloned();
                let gi = st.features.get::<GiHandle>().cloned();
                if vg.is_none() && gi.is_none() {
                    gi_handle(st)?;
                }
                if let Some(vg) = vg {
                    vg.set_cluster_vertex_colors(mesh, ordinal as usize, data.as_deref())?;
                }
                if let Some(gi) = gi {
                    gi.set_cluster_vertex_colors(mesh, ordinal as usize, data.as_deref())?;
                }
                Ok(())
            })
        }
    }
}
