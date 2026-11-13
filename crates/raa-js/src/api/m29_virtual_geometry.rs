//! m29 virtual geometry, share 1 (S31): the pure helpers of the virtualized-geometry API (the
//! patch parser, the settings, instance and stats entries), the `WireValue` model they convert
//! through, the api state `VgApiState`, and the wasm32 runtime that installs the feature and
//! drives its frame (`install`, `frame`).
//!
//! The helpers are plain Rust, so `tests/virtual_geometry.rs` includes this file and exercises
//! them natively: the patch refusals (`vg.invalid` with the key name), the defaults, the settings
//! table in `SETTING_KEYS` order, the instance entries and the stats entries. This file writes no
//! `#[wasm_bindgen]` method: the Engine methods are attached by the card named in the handoff.
//!
//! The wasm32 part (`web`) runs after `m29_cluster_bake` (file-name order). `install` takes the
//! shared cluster store and the m09 shadow handle from the feature map and inserts the `VgHandle`
//! and a `VgApiState`. `frame` feeds the camera, the viewport and the submitted-view count to the
//! handle, keeps the mesh keys and the instance requests in step with the live handles, calls
//! `prepare_frame` before `lit.prepare` and starts the runtime bake of each requested mesh.

// Some helpers are reached only by the host tests or only by the Engine methods of a later card,
// so they are dead in one of the two builds.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};

use raa_core::RaaError;
use raa_vg::{DebugView, InstancePath, VgSettings, VgSettingsPatch, VgStats, error::invalid};

/// The S31 settings keys in their key order: the `VirtualGeometrySettings` names of the JS API.
pub(crate) const SETTING_KEYS: [&str; 9] = [
    "enabled",
    "errorPixels",
    "softwareRaster",
    "occlusion",
    "shadows",
    "runtimeBake",
    "instanceDefault",
    "maxVisibleClusters",
    "debugView",
];

/// A JS value at the wasm boundary, reduced to the shapes the S31 patch reads: `Undefined` is an
/// omission, `Null` and every other type are refusals, `Other` stands for any value that is not
/// a boolean, number or string.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum WireValue {
    Undefined,
    Null,
    Bool(bool),
    Number(f64),
    Text(String),
    Other,
}

/// The api state of the feature: the explicit instance requests (JS instance handle -> request,
/// the `setInstanceVirtualGeometry` values) and the JS mesh handles fed to `VgHandle::set_mesh_key`.
#[derive(Debug, Default)]
pub(crate) struct VgApiState {
    pub requests: BTreeMap<u32, bool>,
    pub mesh_keys: BTreeSet<u32>,
}

/// Parses a `setVirtualGeometry` patch in entry order. `Undefined` is omission for a known key;
/// an unknown key is refused whatever its value. `Null`, a wrong type and a non-integer
/// `maxVisibleClusters` are refused here; the range checks are `VgSettingsPatch::apply`'s. Every
/// refusal is `vg.invalid` with the message `<key>: <reason>`.
///
/// Handoff: `raa-js: m29 virtual-geometry Engine methods and the submission hook` takes
/// `parse_settings_patch`, `settings_entries`, `instance_entries`, `stats_entries`, `VgApiState`
/// and `WireValue` exported by crates/raa-js/src/api/m29_virtual_geometry.rs; it attaches its
/// `#[wasm_bindgen(js_class = Engine)] impl Engine` block to this file and reads the FeatureMap
/// through `Engine::with_state`. Today `install` inserts `VgApiState::default()` next to the
/// `VgHandle` that `VgFeature::install` returns. State/keys: `VgApiState.requests` (JS instance
/// handle -> explicit request) and `VgApiState.mesh_keys` (the JS mesh handles fed to
/// `VgHandle::set_mesh_key`).
pub(crate) fn parse_settings_patch(
    entries: &[(String, WireValue)],
) -> Result<VgSettingsPatch, RaaError> {
    let mut patch = VgSettingsPatch::default();
    for (key, value) in entries {
        let key = key.as_str();
        if !SETTING_KEYS.contains(&key) {
            return Err(invalid(key, "unknown setting"));
        }
        if *value == WireValue::Undefined {
            continue;
        }
        match key {
            "enabled" => patch.enabled = Some(boolean(key, value)?),
            "errorPixels" => patch.error_pixels = Some(number(key, value)? as f32),
            "softwareRaster" => patch.software_raster = Some(boolean(key, value)?),
            "occlusion" => patch.occlusion = Some(boolean(key, value)?),
            "shadows" => patch.shadows = Some(boolean(key, value)?),
            "runtimeBake" => patch.runtime_bake = Some(boolean(key, value)?),
            "instanceDefault" => patch.instance_default = Some(boolean(key, value)?),
            "maxVisibleClusters" => patch.max_visible_clusters = Some(count(key, value)?),
            // `debugView`, the last key of SETTING_KEYS.
            _ => patch.debug_view = Some(debug_view(value)?),
        }
    }
    Ok(patch)
}

/// `null` and a value of another type, named by the expected type (`must be a <expected>`).
fn scalar_refusal(key: &str, value: &WireValue, expected: &str) -> RaaError {
    match value {
        WireValue::Null => invalid(key, "must not be null"),
        _ => invalid(key, format!("must be a {expected}")),
    }
}

fn boolean(key: &str, value: &WireValue) -> Result<bool, RaaError> {
    match value {
        WireValue::Bool(b) => Ok(*b),
        _ => Err(scalar_refusal(key, value, "boolean")),
    }
}

fn number(key: &str, value: &WireValue) -> Result<f64, RaaError> {
    match value {
        WireValue::Number(n) => Ok(*n),
        _ => Err(scalar_refusal(key, value, "number")),
    }
}

/// A whole number for `maxVisibleClusters`. A value outside the `u32` range saturates and then
/// fails the range check of `VgSettingsPatch::apply`.
fn count(key: &str, value: &WireValue) -> Result<u32, RaaError> {
    let n = number(key, value)?;
    if !n.is_finite() || n.fract() != 0.0 {
        return Err(invalid(key, "must be an integer"));
    }
    Ok(n as u32)
}

fn debug_view(value: &WireValue) -> Result<DebugView, RaaError> {
    match value {
        WireValue::Text(name) => DebugView::from_name(name).ok_or_else(|| {
            invalid(
                "debugView",
                format!("must be none, clusters, lod, raster or occlusion, got {name}"),
            )
        }),
        _ => Err(scalar_refusal("debugView", value, "string")),
    }
}

/// The resolved settings as `ResolvedVirtualGeometry` entries, in `SETTING_KEYS` order.
/// `debugView` is its JS name.
pub(crate) fn settings_entries(s: &VgSettings) -> Vec<(&'static str, WireValue)> {
    vec![
        ("enabled", WireValue::Bool(s.enabled)),
        ("errorPixels", WireValue::Number(f64::from(s.error_pixels))),
        ("softwareRaster", WireValue::Bool(s.software_raster)),
        ("occlusion", WireValue::Bool(s.occlusion)),
        ("shadows", WireValue::Bool(s.shadows)),
        ("runtimeBake", WireValue::Bool(s.runtime_bake)),
        ("instanceDefault", WireValue::Bool(s.instance_default)),
        (
            "maxVisibleClusters",
            WireValue::Number(f64::from(s.max_visible_clusters)),
        ),
        (
            "debugView",
            WireValue::Text(s.debug_view.name().to_string()),
        ),
    ]
}

/// The `InstanceVirtualGeometry` entries: `requested` (the explicit request, `Null` when following
/// `instanceDefault`), `path` ('virtual' | 'forward'), `pending`, `code` and `message`. The code
/// and the message are the path's own values, `Null` when the path carries none.
pub(crate) fn instance_entries(
    path: &InstancePath,
    requested: Option<bool>,
) -> Vec<(&'static str, WireValue)> {
    let (name, pending, code, message) = match path {
        InstancePath::Virtual => ("virtual", false, None, None),
        InstancePath::Forward {
            pending,
            code,
            message,
        } => ("forward", *pending, *code, message.clone()),
    };
    vec![
        (
            "requested",
            requested.map_or(WireValue::Null, WireValue::Bool),
        ),
        ("path", WireValue::Text(name.to_string())),
        ("pending", WireValue::Bool(pending)),
        ("code", text_or_null(code)),
        ("message", message.map_or(WireValue::Null, WireValue::Text)),
    ]
}

/// The `VirtualGeometryStats` entries in the S31 key order. `refused` is a placeholder
/// (`Other`): the wasm side serialises the object from `stats.refused` at its position.
/// Counters that are `None` are `Null`.
pub(crate) fn stats_entries(stats: &VgStats) -> Vec<(&'static str, WireValue)> {
    vec![
        ("active", WireValue::Bool(stats.active)),
        ("inactiveReason", text_or_null(stats.inactive_reason)),
        (
            "virtualInstances",
            WireValue::Number(f64::from(stats.virtual_instances)),
        ),
        (
            "forwardInstances",
            WireValue::Number(f64::from(stats.forward_instances)),
        ),
        ("refused", WireValue::Other),
        (
            "bakingMeshes",
            WireValue::Number(f64::from(stats.baking_meshes)),
        ),
        (
            "readyMeshes",
            WireValue::Number(f64::from(stats.ready_meshes)),
        ),
        (
            "drawnClusters",
            number_or_null(stats.drawn_clusters.map(f64::from)),
        ),
        (
            "drawnTriangles",
            number_or_null(stats.drawn_triangles.map(f64::from)),
        ),
        (
            "hardwareClusters",
            number_or_null(stats.hardware_clusters.map(f64::from)),
        ),
        (
            "softwareClusters",
            number_or_null(stats.software_clusters.map(f64::from)),
        ),
        (
            "shadowClusters",
            number_or_null(stats.shadow_clusters.map(f64::from)),
        ),
        (
            "droppedClusters",
            number_or_null(stats.dropped_clusters.map(f64::from)),
        ),
        (
            "statsFrameIndex",
            number_or_null(stats.stats_frame_index.map(|i| i as f64)),
        ),
        ("pending", WireValue::Bool(stats.pending)),
    ]
}

fn text_or_null(value: Option<&'static str>) -> WireValue {
    value.map_or(WireValue::Null, |text| WireValue::Text(text.to_string()))
}

fn number_or_null(value: Option<f64>) -> WireValue {
    value.map_or(WireValue::Null, WireValue::Number)
}

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install};

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use crate::api::ViewsState;
    use crate::api::m29_cluster_bake::{
        VG_INVALID, check_vertex_colors, cluster_error, vg_invalid,
    };
    use crate::convert::HandleMap;
    use crate::engine::InstallCtx;
    use crate::state::{EngineState, InstanceGroup, MeshEntry, MeshSources};
    use raa_cluster::{BuildOptions, ClusterStore};
    use raa_core::GpuContext;
    use raa_lit::LitRenderer;
    use raa_shadow::ShadowHandle;
    use raa_vg::{VgFeature, VgHandle};
    use raa_web::FrameInput;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Installs the feature after `m29_cluster_bake` (file-name order). The shared cluster store
    /// (`vg.invalid` naming `store` when absent) and the m09 shadow handle when present go to
    /// `VgFeature::install`; the returned `VgHandle` and a default `VgApiState` are inserted.
    pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
        let store = ctx
            .features
            .get::<Rc<RefCell<ClusterStore>>>()
            .cloned()
            .ok_or_else(|| vg_invalid("store", "the cluster store is not installed"))?;
        let shadow = ctx.features.get::<ShadowHandle>().cloned();
        let handle = VgFeature::install(ctx.gpu, ctx.pipeline, ctx.lit, store, shadow)?;
        ctx.features.insert(handle);
        ctx.features.insert(VgApiState::default());
        Ok(())
    }

    /// Runs before `lit.prepare` once per Engine invocation: the camera, viewport 0 and the
    /// submitted-view count go to the handle, the mesh keys and instance requests follow the live
    /// handles, then `prepare_frame` runs and each mesh it asks to bake starts its runtime bake
    /// (when `runtimeBake` is on). A bake that cannot start is recorded in the store, never thrown.
    pub(crate) fn frame(
        st: &mut EngineState,
        gpu: &GpuContext,
        input: &FrameInput,
    ) -> Result<(), RaaError> {
        let Some(vg) = st.features.get::<VgHandle>().cloned() else {
            return Ok(());
        };
        let camera = st.scene.camera;
        let viewport = viewport0(st, input);
        let submitted = submitted_views(st);
        if st.scene_invocation.is_discontinuous(0) {
            vg.invalidate_history();
        }
        vg.set_view(&camera, viewport, submitted);
        let Some(api) = st.features.get_mut::<VgApiState>() else {
            return Ok(());
        };
        sync_mesh_keys(&vg, &st.meshes, &st.lit, &mut api.mesh_keys);
        sync_requests(&vg, &st.instances, &mut api.requests)?;
        let report = vg.prepare_frame(gpu, &mut st.scene, &mut st.lit);
        if vg.get().runtime_bake && !report.bake_requests.is_empty() {
            let store = st.features.get::<Rc<RefCell<ClusterStore>>>().cloned();
            if let Some(store) = store {
                for &key in &report.bake_requests {
                    start_runtime_bake(st, &store, key);
                }
            }
        }
        Ok(())
    }

    /// Maps every live JS mesh handle to its lit sub-mesh ids and removes the keys of handles
    /// that vanished (`removeMesh`). `keys` keeps the handles now mapped.
    fn sync_mesh_keys(
        vg: &VgHandle,
        meshes: &HandleMap<MeshEntry>,
        lit: &LitRenderer,
        keys: &mut BTreeSet<u32>,
    ) {
        let mut live = BTreeSet::new();
        for (handle, entry) in meshes.iter() {
            if let Some(ids) = lit.submesh_ids(entry.lit) {
                vg.set_mesh_key(handle, &ids);
                live.insert(handle);
            }
        }
        for vanished in keys.difference(&live).copied().collect::<Vec<u32>>() {
            vg.remove_mesh_key(vanished);
        }
        *keys = live;
    }

    /// Drops the explicit request of every instance group that no longer exists
    /// (`removeInstance`): the handle's request goes back to following `instanceDefault`.
    fn sync_requests(
        vg: &VgHandle,
        instances: &HandleMap<InstanceGroup>,
        requests: &mut BTreeMap<u32, bool>,
    ) -> Result<(), RaaError> {
        let vanished: Vec<u32> = requests
            .keys()
            .copied()
            .filter(|&tag| instances.get_ref(tag).is_err())
            .collect();
        for tag in vanished {
            vg.set_instance(tag, None)?;
            requests.remove(&tag);
        }
        Ok(())
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

    /// Starts the runtime bake of mesh `key` or records why it cannot: no retained geometry is
    /// `vg.invalid`, a colour stream `vg.mesh_vertex_colors`, and a cluster error its vg code.
    fn start_runtime_bake(st: &EngineState, store: &RefCell<ClusterStore>, key: u32) {
        let Some(source) = st
            .features
            .get::<MeshSources>()
            .and_then(|sources| sources.get(key))
        else {
            store
                .borrow_mut()
                .refuse(key, VG_INVALID, "no retained geometry".to_string());
            return;
        };
        let started = check_vertex_colors(source.has_colors, "mesh").and_then(|()| {
            let inputs = source.cluster_inputs().map_err(|e| cluster_error(&e))?;
            store
                .borrow_mut()
                .start_bake(key, inputs, &BuildOptions::default())
                .map_err(|e| cluster_error(&e))
        });
        if let Err(error) = started {
            record_refusal(store, key, error);
        }
    }

    /// Stores a refusal under its code and message; any other error variant is `vg.invalid`.
    fn record_refusal(store: &RefCell<ClusterStore>, key: u32, error: RaaError) {
        let (code, message) = match error {
            RaaError::FrameGraph { code, message } => (code, message),
            other => (VG_INVALID, other.to_string()),
        };
        store.borrow_mut().refuse(key, code, message);
    }
}
