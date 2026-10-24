//! m14 camera effects API: screen-space reflections (`setSsr`), depth of field (`setDof`)
//! and volumetric sun shafts (`setSunShafts`); height fog and colour grading are installed
//! here too.
//!
//! `install` adds the SSR, fog, sun shafts, DOF and grading features in that order; `frame`
//! pushes the scene camera (and the sun, for the shafts) to their handles every frame (the
//! passes have no scene access). Enabling and disabling stays `setPassEnabled('ssr' |
//! 'dof' | 'sunshafts' | 'fog' | 'grading', on)`. Refusals (`post.invalid`, naming the
//! field) are thrown with their own code; malformed JS arguments throw
//! `api.invalid_argument`.

use js_sys::{Array, Reflect};
use raa_core::{GpuContext, RaaError};
use raa_dof::{DofFeature, DofHandle};
use raa_fog::FogFeature;
use raa_grading::GradingFeature;
use raa_shafts::{ShaftsFeature, ShaftsHandle};
use raa_ssr::{SsrFeature, SsrHandle};
use raa_web::FrameInput;
use wasm_bindgen::prelude::*;

use crate::convert::{INVALID_ARGUMENT, api_error};
use crate::engine::{Engine, InstallCtx};
use crate::state::EngineState;

pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
    let ssr = SsrFeature::install(ctx.gpu, ctx.pipeline)?;
    let fog = FogFeature::install(ctx.gpu, ctx.pipeline, ctx.lit)?;
    let shafts = ShaftsFeature::install(ctx.gpu, ctx.pipeline)?;
    let dof = DofFeature::install(ctx.gpu, ctx.pipeline)?;
    let grading = GradingFeature::install(ctx.gpu, ctx.pipeline)?;
    ctx.features.insert(ssr);
    ctx.features.insert(fog);
    ctx.features.insert(shafts);
    ctx.features.insert(dof);
    ctx.features.insert(grading);
    Ok(())
}

/// Pushes the scene camera to the SSR, DOF and shafts handles and the sun to the shafts
/// (the handles keep their previous camera on a singular matrix). No allocation.
pub(crate) fn frame(
    st: &mut EngineState,
    _gpu: &GpuContext,
    _input: &FrameInput,
) -> Result<(), RaaError> {
    let cam = &st.scene.camera;
    let view = cam.view();
    let proj = cam.projection();
    if let Some(ssr) = st.features.get::<SsrHandle>() {
        ssr.set_camera(&view, &proj);
    }
    if let Some(dof) = st.features.get::<DofHandle>() {
        dof.set_camera(&view, &proj);
    }
    if let Some(shafts) = st.features.get::<ShaftsHandle>() {
        shafts.set_camera(&view, &proj);
        let light = &st.scene.light;
        let c = light.color;
        let i = light.intensity;
        shafts.set_sun(light.direction, [c[0] * i, c[1] * i, c[2] * i]);
    }
    Ok(())
}

fn invalid(message: String) -> RaaError {
    api_error(INVALID_ARGUMENT, message)
}

/// The value at `key`, or None when it is missing, undefined or null.
fn get(obj: &JsValue, key: &str) -> Result<Option<JsValue>, RaaError> {
    if !obj.is_object() {
        return Err(invalid(format!(
            "expected an object holding '{key}', got a non-object"
        )));
    }
    let v = Reflect::get(obj, &JsValue::from_str(key))
        .map_err(|_| invalid(format!("cannot read '{key}'")))?;
    Ok(if v.is_undefined() || v.is_null() {
        None
    } else {
        Some(v)
    })
}

/// The number at `key`, if present (`api.invalid_argument` for a non-number). Range and
/// finiteness are left to the feature's own validation, so it refuses with its code.
fn optional_f64(obj: &JsValue, key: &str) -> Result<Option<f64>, RaaError> {
    match get(obj, key)? {
        None => Ok(None),
        Some(v) => v
            .as_f64()
            .map(Some)
            .ok_or_else(|| invalid(format!("'{key}' must be a number"))),
    }
}

/// Overwrites `*slot` with the number at `key` when present.
fn read_f32(obj: &JsValue, key: &str, slot: &mut f32) -> Result<(), RaaError> {
    if let Some(x) = optional_f64(obj, key)? {
        *slot = x as f32;
    }
    Ok(())
}

/// Checks that `settings` is an object (`api.invalid_argument` otherwise).
fn require_object(settings: &JsValue, method: &str) -> Result<(), RaaError> {
    if settings.is_object() {
        Ok(())
    } else {
        Err(invalid(format!("{method} expects an object")))
    }
}

/// The array of `N` numbers at `key`, if present (`api.invalid_argument` otherwise).
fn optional_array<const N: usize>(obj: &JsValue, key: &str) -> Result<Option<[f32; N]>, RaaError> {
    let Some(v) = get(obj, key)? else {
        return Ok(None);
    };
    let bad = || invalid(format!("'{key}' must be an array of {N} numbers"));
    if !Array::is_array(&v) {
        return Err(bad());
    }
    let a = Array::from(&v);
    if a.length() != N as u32 {
        return Err(bad());
    }
    let mut out = [0.0; N];
    for (i, dst) in out.iter_mut().enumerate() {
        *dst = a.get(i as u32).as_f64().ok_or_else(bad)? as f32;
    }
    Ok(Some(out))
}

/// The array of 3 numbers at `key`, if present (used by the fog and grading setters).
#[allow(dead_code)]
fn optional_vec3(obj: &JsValue, key: &str) -> Result<Option<[f32; 3]>, RaaError> {
    optional_array::<3>(obj, key)
}

#[wasm_bindgen(js_class = Engine)]
impl Engine {
    /// Sets the screen-space reflections: `{intensity?, maxRoughness?, thickness?}` merged
    /// over the current settings. Out-of-range values throw `post.invalid` naming the
    /// field; nothing changes on a refusal.
    #[wasm_bindgen(js_name = setSsr)]
    pub fn set_ssr(&mut self, settings: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            require_object(&settings, "setSsr")?;
            let ssr = st.features.require_mut::<SsrHandle>("ssr")?;
            let mut d = ssr.get();
            read_f32(&settings, "intensity", &mut d.intensity)?;
            read_f32(&settings, "maxRoughness", &mut d.max_roughness)?;
            read_f32(&settings, "thickness", &mut d.thickness_m)?;
            ssr.set(d)
        })
    }

    /// Sets the depth of field: `{focusDistance?, fStop?, focalLength?, autofocus?: [u, v]
    /// | null, maxCoc?}` merged over the current settings (`autofocus: null` turns it off,
    /// absent keeps it). Out-of-range values throw `post.invalid` naming the field; nothing
    /// changes on a refusal.
    #[wasm_bindgen(js_name = setDof)]
    pub fn set_dof(&mut self, settings: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            require_object(&settings, "setDof")?;
            let dof = st.features.require_mut::<DofHandle>("dof")?;
            let mut d = dof.get();
            read_f32(&settings, "focusDistance", &mut d.focus_distance_m)?;
            read_f32(&settings, "fStop", &mut d.f_stop)?;
            read_f32(&settings, "focalLength", &mut d.focal_length_mm)?;
            read_f32(&settings, "maxCoc", &mut d.max_coc_px)?;
            let af = Reflect::get(&settings, &JsValue::from_str("autofocus"))
                .map_err(|_| invalid("cannot read 'autofocus'".to_string()))?;
            if af.is_null() {
                d.autofocus = None;
            } else if !af.is_undefined() {
                d.autofocus = optional_array::<2>(&settings, "autofocus")?;
            }
            dof.set(d)
        })
    }

    /// Sets the sun shafts: `{intensity?, density?, maxDistance?}` merged over the current
    /// settings. Out-of-range values throw `post.invalid` naming the field; nothing changes
    /// on a refusal.
    #[wasm_bindgen(js_name = setSunShafts)]
    pub fn set_sun_shafts(&mut self, settings: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            require_object(&settings, "setSunShafts")?;
            let shafts = st.features.require_mut::<ShaftsHandle>("sun shafts")?;
            let mut d = shafts.get();
            read_f32(&settings, "intensity", &mut d.intensity)?;
            read_f32(&settings, "density", &mut d.density)?;
            read_f32(&settings, "maxDistance", &mut d.max_distance_m)?;
            shafts.set(d)
        })
    }
}
