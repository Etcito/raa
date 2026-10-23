//! m09 shadows and ambient occlusion API: cascaded soft sun shadows (`setShadows`), GTAO
//! (`setAo`) and screen-space contact shadows (`setContactShadows`).
//!
//! `install` adds the shadow, GTAO and contact features in that order; `frame` pushes the
//! scene camera and sun to their handles every frame (the passes have no scene access).
//! Enabling and disabling stays `setPassEnabled('shadow.csm' | 'ao.gtao' |
//! 'shadow.contact', on)`. Refusals (`shadow.invalid` / `ao.invalid`, naming the field) are
//! thrown with their own code; malformed JS arguments throw `api.invalid_argument`.

use js_sys::Reflect;
use raa_contact::{ContactFeature, ContactHandle};
use raa_core::{GpuContext, RaaError};
use raa_gtao::{GtaoFeature, GtaoHandle};
use raa_shadow::{ShadowFeature, ShadowHandle};
use raa_web::FrameInput;
use wasm_bindgen::prelude::*;

use crate::convert::{INVALID_ARGUMENT, api_error};
use crate::engine::{Engine, InstallCtx};
use crate::state::EngineState;

pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
    let shadow = ShadowFeature::install(ctx.gpu, ctx.pipeline, ctx.lit)?;
    let gtao = GtaoFeature::install(ctx.gpu, ctx.pipeline, ctx.lit)?;
    let contact = ContactFeature::install(ctx.gpu, ctx.pipeline, ctx.lit)?;
    ctx.features.insert(shadow);
    ctx.features.insert(gtao);
    ctx.features.insert(contact);
    Ok(())
}

/// Pushes the scene camera and sun to the three handles; skipped when the projection has
/// no inverse. No allocation.
pub(crate) fn frame(
    st: &mut EngineState,
    _gpu: &GpuContext,
    _input: &FrameInput,
) -> Result<(), RaaError> {
    let cam = &st.scene.camera;
    let view = cam.view();
    let proj = cam.projection();
    let Some(inv_proj) = proj.inverse() else {
        return Ok(());
    };
    let sun_dir = st.scene.light.direction;
    if let Some(shadow) = st.features.get::<ShadowHandle>() {
        shadow.set_view(&view, cam.fov_y_rad, cam.aspect, cam.near, cam.far, sun_dir);
    }
    if let Some(gtao) = st.features.get::<GtaoHandle>() {
        gtao.set_camera(&proj, &inv_proj);
    }
    if let Some(contact) = st.features.get::<ContactHandle>() {
        contact.set_camera(&view, &proj, &inv_proj, sun_dir);
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

#[wasm_bindgen(js_class = Engine)]
impl Engine {
    /// Sets the cascaded sun shadows: `{softness?, distance?, resolution?: 1024|2048|4096}`
    /// merged over the current settings. Out-of-range values throw `shadow.invalid` naming
    /// the field; nothing changes on a refusal.
    #[wasm_bindgen(js_name = setShadows)]
    pub fn set_shadows(&mut self, settings: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            require_object(&settings, "setShadows")?;
            let shadow = st.features.require_mut::<ShadowHandle>("shadows")?;
            let mut d = shadow.get();
            read_f32(&settings, "softness", &mut d.softness)?;
            read_f32(&settings, "distance", &mut d.distance_m)?;
            if let Some(r) = optional_f64(&settings, "resolution")? {
                if !(r.fract() == 0.0 && (0.0..=f64::from(u32::MAX)).contains(&r)) {
                    return Err(api_error(
                        "shadow.invalid",
                        format!("resolution must be 1024, 2048 or 4096, got {r}"),
                    ));
                }
                d.resolution = r as u32;
            }
            shadow.set(d)
        })
    }

    /// Sets the ambient occlusion: `{radius?, intensity?, power?}` merged over the current
    /// settings. Out-of-range values throw `ao.invalid` naming the field; nothing changes on
    /// a refusal.
    #[wasm_bindgen(js_name = setAo)]
    pub fn set_ao(&mut self, settings: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            require_object(&settings, "setAo")?;
            let ao = st.features.require_mut::<GtaoHandle>("ao")?;
            let mut d = ao.get();
            read_f32(&settings, "radius", &mut d.radius_m)?;
            read_f32(&settings, "intensity", &mut d.intensity)?;
            read_f32(&settings, "power", &mut d.power)?;
            ao.set(d)
        })
    }

    /// Sets the contact shadows: `{distance?, thickness?, intensity?}` merged over the
    /// current settings. Out-of-range values throw `shadow.invalid` naming the field;
    /// nothing changes on a refusal.
    #[wasm_bindgen(js_name = setContactShadows)]
    pub fn set_contact_shadows(&mut self, settings: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            require_object(&settings, "setContactShadows")?;
            let contact = st
                .features
                .require_mut::<ContactHandle>("contact shadows")?;
            let mut d = contact.get();
            read_f32(&settings, "distance", &mut d.distance_m)?;
            read_f32(&settings, "thickness", &mut d.thickness_m)?;
            read_f32(&settings, "intensity", &mut d.intensity)?;
            contact.set(d)
        })
    }
}
