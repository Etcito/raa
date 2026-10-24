//! m11 transmission API: refraction with absorption and sorted translucency.
//!
//! `install` adds the transmission feature (opaque copy pass, `screen` slot provider and the
//! `raa.ice`, `raa.slime`, `raa.water`, `raa.glass` built-ins), the linked-list OIT (its
//! handle is kept in the feature map) and pushes `TRANSMISSIVE_BUILTINS` onto the
//! `ExtraBuiltins` of the feature map, so `addSurfaceMaterial` accepts them with their params
//! and `materialFunctions()` lists them. `frame` is empty. `setTransparency` merges
//! `{maxFragments?, poolFactor?}` over the current settings (`oit.invalid` from the handle,
//! `api.invalid_argument` for malformed JS); `transparencyStats` returns
//! `{poolNodes, overflowFrames}`.

use js_sys::{Object, Reflect};
use raa_core::{GpuContext, RaaError};
use raa_matfn::ExtraBuiltins;
use raa_oit::{OitFeature, OitHandle};
use raa_transmission::{TRANSMISSIVE_BUILTINS, TransmissionFeature};
use raa_web::FrameInput;
use wasm_bindgen::prelude::*;

use crate::convert::{INVALID_ARGUMENT, api_error};
use crate::engine::{Engine, InstallCtx};
use crate::feature_map::FEATURE_MISSING;
use crate::state::EngineState;

pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
    TransmissionFeature::install(ctx.gpu, ctx.pipeline, ctx.lit)?;
    let oit = OitFeature::install(ctx.gpu, ctx.pipeline, ctx.lit)?;
    ctx.features.insert(oit);
    match ctx.features.get_mut::<ExtraBuiltins>() {
        Some(extras) => extras.push(TRANSMISSIVE_BUILTINS),
        None => {
            let mut extras = ExtraBuiltins::default();
            extras.push(TRANSMISSIVE_BUILTINS);
            ctx.features.insert(extras);
        }
    }
    Ok(())
}

pub(crate) fn frame(
    _st: &mut EngineState,
    _gpu: &GpuContext,
    _input: &FrameInput,
) -> Result<(), RaaError> {
    Ok(())
}

fn invalid(message: String) -> RaaError {
    api_error(INVALID_ARGUMENT, message)
}

/// The number at `key`, if present (`api.invalid_argument` for a non-number). Integer and
/// range checks are left to `OitHandle::set`, so it refuses with `oit.invalid`.
fn optional_number(obj: &JsValue, key: &str) -> Result<Option<f64>, RaaError> {
    let v = Reflect::get(obj, &JsValue::from_str(key))
        .map_err(|_| invalid(format!("cannot read '{key}'")))?;
    if v.is_undefined() || v.is_null() {
        return Ok(None);
    }
    v.as_f64()
        .map(Some)
        .ok_or_else(|| invalid(format!("'{key}' must be a number")))
}

/// Converts a JS number to a u32 setting; anything that is not a non-negative integer
/// becomes `u32::MAX`, which `OitHandle::set` refuses with `oit.invalid`.
fn to_setting(x: f64) -> u32 {
    if x.is_finite() && x >= 0.0 && x.fract() == 0.0 && x <= f64::from(u32::MAX) {
        x as u32
    } else {
        u32::MAX
    }
}

#[wasm_bindgen(js_class = Engine)]
impl Engine {
    /// Sets the OIT: `{maxFragments?: 4..32, poolFactor?: 2..32}` merged over the current
    /// settings. Non-integers and out-of-range values throw `oit.invalid` naming the field;
    /// nothing changes on a refusal.
    #[wasm_bindgen(js_name = setTransparency)]
    pub fn set_transparency(&mut self, settings: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            if !settings.is_object() {
                return Err(invalid("setTransparency expects an object".to_string()));
            }
            let oit = st.features.require_mut::<OitHandle>("transparency")?;
            let mut d = oit.get();
            if let Some(x) = optional_number(&settings, "maxFragments")? {
                d.max_fragments = to_setting(x);
            }
            if let Some(x) = optional_number(&settings, "poolFactor")? {
                d.pool_factor = to_setting(x);
            }
            oit.set(d)
        })
    }

    /// `{poolNodes, overflowFrames}`: the OIT node pool in use and the overflow frames
    /// counted so far (debugging information).
    #[wasm_bindgen(js_name = transparencyStats)]
    pub fn transparency_stats(&self) -> Result<Object, JsValue> {
        self.with_state(|_, st| {
            let oit = st.features.get::<OitHandle>().ok_or_else(|| {
                api_error(
                    FEATURE_MISSING,
                    "feature `transparency` is not installed in this engine".to_string(),
                )
            })?;
            let s = oit.stats();
            let o = Object::new();
            let _ = Reflect::set(
                &o,
                &"poolNodes".into(),
                &JsValue::from_f64(f64::from(s.pool_nodes)),
            );
            let _ = Reflect::set(
                &o,
                &"overflowFrames".into(),
                &JsValue::from_f64(f64::from(s.overflow_frames)),
            );
            Ok(o)
        })
    }
}
