//! m11 water API: one still water body per scene (caustics-design.md).
//!
//! `setWater(desc | null)` sets the body: `{level, lo: [x, z], hi: [x, z], slope?, frequency?,
//! time?, ior?, sigma?: [r, g, b], caustics?: bool}`. Its 16 lanes go to `FrameGpu.reserved`
//! (water.wgsl). The water's material shader takes its normal from them, and with `caustics`
//! (default true) the forward lights submerged receivers with the refracted sun, tested where
//! its path enters the water, and concentrates it by the ripple's caustics (raa_water_sun,
//! raa_water_caustic, per pixel: no pass, no texture, nothing without water); the DDGI update's
//! sun follows the same refracted path. `caustics: false` keeps the straight, unrefracted sun.

use js_sys::Reflect;
use raa_core::{GpuContext, RaaError};
use raa_shader::WaterBody;
use raa_web::FrameInput;
use wasm_bindgen::prelude::*;

use crate::convert::{INVALID_ARGUMENT, api_error, js};
use crate::engine::{Engine, InstallCtx};
use crate::state::EngineState;

/// The water body's frame lanes (feature map entry); `dirty` until the renderer has them.
#[derive(Default)]
pub(crate) struct WaterState {
    lanes: [f32; 16],
    dirty: bool,
}

pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
    ctx.features.insert(WaterState::default());
    Ok(())
}

pub(crate) fn frame(st: &mut EngineState, _: &GpuContext, _: &FrameInput) -> Result<(), RaaError> {
    let Some(water) = st.features.get_mut::<WaterState>() else {
        return Ok(());
    };
    if water.dirty {
        water.dirty = false;
        let lanes = water.lanes;
        st.lit.set_water_frame(lanes);
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> RaaError {
    api_error(INVALID_ARGUMENT, message.into())
}

fn number(desc: &JsValue, key: &str, default: f32) -> Result<f32, RaaError> {
    let v = js::optional_f64(desc, key)?.map_or(default, |v| v as f32);
    if !v.is_finite() {
        return Err(invalid(format!("'{key}' must be finite")));
    }
    Ok(v)
}

fn array<const N: usize>(desc: &JsValue, key: &str) -> Result<Option<[f32; N]>, RaaError> {
    let v = Reflect::get(desc, &JsValue::from_str(key))
        .map_err(|_| invalid(format!("cannot read '{key}'")))?;
    if v.is_undefined() || v.is_null() {
        return Ok(None);
    }
    js::floats::<N>(key, &v).map(Some)
}

#[wasm_bindgen(js_class = Engine)]
impl Engine {
    /// Sets (or with null clears) the scene's water body.
    #[wasm_bindgen(js_name = setWater)]
    pub fn set_water(&self, desc: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            let lanes = if desc.is_null() || desc.is_undefined() {
                [0.0f32; 16]
            } else {
                if !desc.is_object() {
                    return Err(invalid("water must be an object or null"));
                }
                let lo = array::<2>(&desc, "lo")?.ok_or_else(|| invalid("'lo' is required"))?;
                let hi = array::<2>(&desc, "hi")?.ok_or_else(|| invalid("'hi' is required"))?;
                if hi[0] <= lo[0] || hi[1] <= lo[1] {
                    return Err(invalid("'hi' must exceed 'lo'"));
                }
                let body = WaterBody {
                    level: number(&desc, "level", 0.0)?,
                    slope: number(&desc, "slope", 0.0)?.clamp(0.0, 1.0),
                    frequency: number(&desc, "frequency", 1.0)?.max(1e-3),
                    time: number(&desc, "time", 0.0)?,
                    lo,
                    hi,
                    ior: number(&desc, "ior", 1.33)?.clamp(1.0, 3.0),
                    sigma: array::<3>(&desc, "sigma")?
                        .unwrap_or([0.0; 3])
                        .map(|s| s.max(0.0)),
                };
                body.frame_lanes(js::optional_bool(&desc, "caustics")?.unwrap_or(true))
            };
            let water = st
                .features
                .get_mut::<WaterState>()
                .ok_or_else(|| invalid("water is not installed"))?;
            water.dirty |= water.lanes != lanes;
            water.lanes = lanes;
            Ok(())
        })
    }
}
