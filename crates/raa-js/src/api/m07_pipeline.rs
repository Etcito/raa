//! m07 pipeline API: the pass list and its configuration (`passes`, `setPassEnabled`,
//! `setPassOrder`), tone mapping (`setToneMapping`), the display-referred background
//! (`setBackground`) and the device capabilities (`capabilities`).
//!
//! Pipeline refusals (`pipeline.*`) and tone mapping refusals (`post.invalid`) are thrown
//! with their own code; malformed JS arguments throw `api.invalid_argument`.

use js_sys::{Array, Object, Reflect};
use raa_core::{GpuContext, RaaError};
use raa_pipeline::{PassInfo, PassRole, Pipeline};
use raa_tonemap::{Look, ToneMapping, TonemapControl};
use raa_web::FrameInput;
use wasm_bindgen::prelude::*;

use crate::convert::{INVALID_ARGUMENT, api_error, js, tone_mapper};
use crate::engine::{Engine, InstallCtx};
use crate::state::EngineState;

/// `InstallCtx` does not carry the tonemap control (engine.rs keeps it in
/// `EngineState::tonemap`), so it is stored in the feature map by [`frame`].
pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
    let _ = ctx;
    Ok(())
}

/// Stores the tonemap control in the feature map once, when engine.rs did not (one
/// allocation on the first frame, none after).
pub(crate) fn frame(
    st: &mut EngineState,
    _gpu: &GpuContext,
    _input: &FrameInput,
) -> Result<(), RaaError> {
    if st.features.get::<TonemapControl>().is_none() {
        st.features.insert(st.tonemap.clone());
    }
    Ok(())
}

fn invalid(message: String) -> RaaError {
    api_error(INVALID_ARGUMENT, message)
}

/// Runs `f` on the mutably borrowed pipeline (`pipeline.busy` while a frame executes).
fn with_pipeline<R>(
    st: &EngineState,
    f: impl FnOnce(&mut Pipeline) -> Result<R, RaaError>,
) -> Result<R, RaaError> {
    let mut p = st.pipeline.try_borrow_mut().map_err(|_| {
        api_error(
            "pipeline.busy",
            "the pipeline is borrowed (called while a frame executes)",
        )
    })?;
    f(&mut p)
}

/// The JS name of an AgX look: 'none', 'punchy' or 'golden'.
fn look(name: &str) -> Option<Look> {
    match name {
        "none" => Some(Look::None),
        "punchy" => Some(Look::Punchy),
        "golden" => Some(Look::Golden),
        _ => None,
    }
}

/// Sets `key` on a fresh plain object; this cannot throw, so the result is ignored.
fn set(obj: &Object, key: &str, value: &JsValue) {
    let _ = Reflect::set(obj, &JsValue::from_str(key), value);
}

fn pass_info_to_js(p: &PassInfo) -> JsValue {
    let o = Object::new();
    let role = match p.role {
        PassRole::Pass => "pass",
        PassRole::Contributor => "contributor",
    };
    set(&o, "name", &JsValue::from_str(&p.name));
    set(&o, "role", &JsValue::from_str(role));
    set(&o, "enabled", &JsValue::from_bool(p.enabled));
    set(&o, "order", &JsValue::from(p.order));
    set(&o, "required", &JsValue::from_bool(p.required));
    o.into()
}

/// The enabled WebGPU features of `features`, as their WebGPU names (`float32-filterable`).
fn feature_names(features: wgpu::Features) -> Array {
    wgpu::Features::from(features.features_webgpu)
        .iter()
        .filter_map(|f| f.as_str())
        .map(JsValue::from_str)
        .collect()
}

#[wasm_bindgen(js_class = Engine)]
impl Engine {
    /// Every pass and forward contributor in execution order:
    /// `{name, role: 'pass' | 'contributor', enabled, order, required}`.
    pub fn passes(&self) -> Result<Array, JsValue> {
        self.with_state(|_, st| {
            let list = with_pipeline(st, |p| Ok(p.passes()))?;
            Ok(list.iter().map(pass_info_to_js).collect())
        })
    }

    /// Enables or disables a pass or contributor (`pipeline.unknown_pass`,
    /// `pipeline.required_pass`).
    #[wasm_bindgen(js_name = setPassEnabled)]
    pub fn set_pass_enabled(&mut self, name: &str, enabled: bool) -> Result<(), JsValue> {
        self.with_state(|_, st| with_pipeline(st, |p| p.set_enabled(name, enabled)))
    }

    /// Sets the execution order of the passes: every pass-role name exactly once
    /// (`pipeline.order_incomplete`, `pipeline.unknown_pass`, `pipeline.required_order`).
    #[wasm_bindgen(js_name = setPassOrder)]
    pub fn set_pass_order(&mut self, names: Vec<String>) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            let names: Vec<&str> = names.iter().map(String::as_str).collect();
            with_pipeline(st, |p| p.set_order(&names))
        })
    }

    /// Merges `{mapper?: 'agx' | 'aces' | 'linear', exposureEv?: number (finite),
    /// look?: 'none' | 'punchy' | 'golden', dither?: boolean}` over the current tone
    /// mapping. Unknown names or a non-finite ev throw `api.invalid_argument`; nothing
    /// changes on a refusal.
    #[wasm_bindgen(js_name = setToneMapping)]
    pub fn set_tone_mapping(&mut self, settings: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            let mut t: ToneMapping = st.tonemap.get();
            if let Some(name) = js::optional_string(&settings, "mapper")? {
                t.mapper = tone_mapper(&name).ok_or_else(|| {
                    invalid(format!(
                        "'mapper' must be 'agx', 'aces' or 'linear', got '{name}'"
                    ))
                })?;
            }
            if let Some(ev) = js::optional_f64(&settings, "exposureEv")? {
                let ev = ev as f32;
                if !ev.is_finite() {
                    return Err(invalid(format!(
                        "'exposureEv' must be finite as f32, got {ev}"
                    )));
                }
                t.exposure_ev = ev;
            }
            if let Some(name) = js::optional_string(&settings, "look")? {
                t.look = look(&name).ok_or_else(|| {
                    invalid(format!(
                        "'look' must be 'none', 'punchy' or 'golden', got '{name}'"
                    ))
                })?;
            }
            if let Some(d) = js::optional_bool(&settings, "dither")? {
                t.dither = d;
            }
            st.tonemap.set(t)
        })
    }

    /// Sets the display-referred linear background colour (S12), finite and >= 0.
    #[wasm_bindgen(js_name = setBackground)]
    pub fn set_background(&mut self, color: &[f32]) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            let rgb: [f32; 3] = color.try_into().map_err(|_| {
                invalid(format!(
                    "'color' must have 3 components, got {}",
                    color.len()
                ))
            })?;
            if !rgb.iter().all(|c| c.is_finite() && *c >= 0.0) {
                return Err(invalid(format!(
                    "'color' must hold finite numbers >= 0, got {rgb:?}"
                )));
            }
            st.lit.set_background(rgb);
            Ok(())
        })
    }

    /// `{coreWebGpu: true, features: string[] (enabled device features, WebGPU names),
    /// msaa, hdrFormat: 'rgba16float', maxStorageBufferBindingSize}`.
    pub fn capabilities(&self) -> Result<JsValue, JsValue> {
        self.with_state(|gpu, st| {
            let o = Object::new();
            set(&o, "coreWebGpu", &JsValue::TRUE);
            set(&o, "features", &feature_names(gpu.features()));
            set(&o, "msaa", &JsValue::from(st.lit.handle().msaa()));
            set(&o, "hdrFormat", &JsValue::from_str("rgba16float"));
            let max = gpu.limits().max_storage_buffer_binding_size;
            set(
                &o,
                "maxStorageBufferBindingSize",
                &JsValue::from(max as f64),
            );
            Ok(o.into())
        })
    }
}
