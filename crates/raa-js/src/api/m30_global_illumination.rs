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

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install};

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use crate::api::ViewsState;
    use crate::convert::api_error;
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
    /// returns the `GiHandle`, which is inserted with a default `GiApiState`.
    pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
        let vg = ctx.features.get::<VgHandle>().cloned();
        let ibl = ctx.features.get::<IblHandle>().cloned();
        let handle = GiFeature::install(ctx.gpu, ctx.pipeline, ctx.lit, vg, ibl)?;
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
    }
}
