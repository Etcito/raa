//! Installation, settings and statistics of the cached local-light cube shadows
//! (`shadow.local`): point and spot lights with `shadows: true` are shadowed by cube faces traced
//! once through the shared mesh BVH (present while a BVH-consuming GI such as DDGI is active).

use js_sys::{Object, Reflect};
use raa_core::RaaError;
use raa_sdf_trace::mesh::{
    LocalShadowFeature, LocalShadowHandle, LocalShadowSettings, MeshTraceHandle,
};
use wasm_bindgen::prelude::*;

use crate::convert::{INVALID_ARGUMENT, api_error, js};
use crate::engine::{Engine, InstallCtx};
use crate::state::EngineState;

pub(crate) fn install_for_graph(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
    let mesh = ctx
        .features
        .require_mut::<MeshTraceHandle>("mesh trace")?
        .clone();
    let handle = LocalShadowFeature::install(&mut ctx.pipeline.borrow_mut(), mesh)?;
    ctx.features.insert(handle);
    Ok(())
}

/// Gives the pass the frame's camera position (nearest lights are traced first).
pub(crate) fn prepare(st: &EngineState, view: usize) {
    if view != 0 {
        return;
    }
    if let Some(handle) = st.features.get::<LocalShadowHandle>() {
        handle.borrow_mut().set_eye(st.scene.camera.eye.to_array());
    }
}

fn put(object: &Object, key: &str, value: JsValue) {
    Reflect::set(object, &key.into(), &value).expect("fresh object");
}

fn whole(options: &JsValue, key: &str, previous: u32) -> Result<u32, RaaError> {
    let Some(raw) = js::optional_f64(options, key)? else {
        return Ok(previous);
    };
    if raw.fract() != 0.0 || !(0.0..=4096.0).contains(&raw) {
        return Err(api_error(
            "shadow.local_invalid",
            format!("{key} must be a whole number"),
        ));
    }
    Ok(raw as u32)
}

#[wasm_bindgen]
impl Engine {
    /// Configures the local-light cube shadows: `{ enabled, resolution, facesPerFrame }`
    /// (defaults true, 128 texels per face side, 2 faces traced per frame at most). Lights opt in
    /// with `shadows: true`; their `radius` is the emitter radius the shadow rays start from.
    #[wasm_bindgen(js_name = setLocalShadows)]
    pub fn set_local_shadows(&self, options: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            if !options.is_object() || options.is_null() || js_sys::Array::is_array(&options) {
                return Err(api_error(
                    INVALID_ARGUMENT,
                    "local shadow options must be an object",
                ));
            }
            let handle = st
                .features
                .get::<LocalShadowHandle>()
                .cloned()
                .ok_or_else(|| {
                    api_error("api.feature_missing", "local shadows are not installed")
                })?;
            let previous = handle.borrow().settings;
            let settings = LocalShadowSettings {
                enabled: js::optional_bool(&options, "enabled")?.unwrap_or(previous.enabled),
                resolution: whole(&options, "resolution", previous.resolution)?,
                faces_per_frame: whole(&options, "facesPerFrame", previous.faces_per_frame)?,
            };
            handle.borrow_mut().configure(settings)
        })
    }

    /// Re-traces every cached face (A/B captures, diagnostics).
    #[wasm_bindgen(js_name = invalidateLocalShadows)]
    pub fn invalidate_local_shadows(&self) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            if let Some(handle) = st.features.get::<LocalShadowHandle>() {
                handle.borrow_mut().invalidate();
            }
            Ok(())
        })
    }

    /// Settings and the last frame's state of the local-light cube shadows, or null.
    #[wasm_bindgen(js_name = localShadowStats)]
    pub fn local_shadow_stats(&self) -> Result<JsValue, JsValue> {
        self.with_state(|_, st| {
            let Some(handle) = st.features.get::<LocalShadowHandle>() else {
                return Ok(JsValue::NULL);
            };
            let state = handle.borrow();
            let s = state.settings;
            let stats = state.stats();
            let result = Object::new();
            put(&result, "enabled", s.enabled.into());
            put(&result, "resolution", s.resolution.into());
            put(&result, "facesPerFrame", s.faces_per_frame.into());
            put(&result, "lights", stats.lights.into());
            put(&result, "capacity", stats.capacity.into());
            put(&result, "readyFaces", stats.ready_faces.into());
            put(&result, "pendingFaces", stats.pending_faces.into());
            put(&result, "tracedFaces", stats.traced_faces.into());
            put(&result, "tracedTotal", (stats.traced_total as f64).into());
            put(&result, "triangles", stats.triangles.into());
            put(&result, "bytes", (stats.bytes as f64).into());
            Ok(result.into())
        })
    }
}
