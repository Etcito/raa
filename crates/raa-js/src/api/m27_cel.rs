//! Cel resolve and opt-in material toon lighting. Styles are supplied by the host.
use crate::engine::{Engine, InstallCtx};
use crate::state::EngineState;
use js_sys::{Object, Reflect};
use raa_cel::{CelFeature, CelHandle, CelSettings, Dither};
use raa_core::{GpuContext, RaaError};
use raa_web::FrameInput;
use wasm_bindgen::prelude::*;

pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
    raa_cel::install(ctx.lit)?;
    let handle = CelFeature::install(ctx.gpu, ctx.pipeline)?;
    ctx.pipeline
        .borrow_mut()
        .set_enabled("cel.resolve", false)?;
    ctx.features.insert(handle);
    Ok(())
}
pub(crate) fn frame(
    st: &mut EngineState,
    _gpu: &GpuContext,
    _input: &FrameInput,
) -> Result<(), RaaError> {
    if let Some(h) = st.features.get::<CelHandle>() {
        h.set_camera(&st.scene.camera.view(), &st.scene.camera.projection());
    }
    Ok(())
}
fn invalid(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: "cel.invalid",
        message: message.into(),
    }
}
fn get(obj: &JsValue, key: &str) -> Result<Option<JsValue>, RaaError> {
    let v = Reflect::get(obj, &JsValue::from_str(key))
        .map_err(|_| invalid(format!("cannot read {key}")))?;
    Ok((!v.is_undefined()).then_some(v))
}
fn number(obj: &JsValue, key: &str, lo: f64, hi: f64) -> Result<Option<f64>, RaaError> {
    get(obj, key)?
        .map(|v| {
            v.as_f64()
                .filter(|v| v.is_finite() && (lo..=hi).contains(v))
                .ok_or_else(|| invalid(format!("{key} must be a finite number in [{lo}, {hi}]")))
        })
        .transpose()
}
fn boolean(obj: &JsValue, key: &str, target: &mut bool) -> Result<(), RaaError> {
    if let Some(v) = get(obj, key)? {
        *target = v
            .as_bool()
            .ok_or_else(|| invalid(format!("{key} must be boolean")))?;
    }
    Ok(())
}
fn read(obj: &JsValue, mut d: CelSettings) -> Result<CelSettings, RaaError> {
    if !obj.is_object() || obj.is_null() || js_sys::Array::is_array(obj) {
        return Err(invalid("setCel expects an object"));
    }
    for (key, target, lo, hi) in [
        ("bands", &mut d.bands, 2.0, 32.0),
        ("hues", &mut d.hues, 4.0, 32.0),
        ("chroma", &mut d.chroma, 2.0, 12.0),
    ] {
        if let Some(v) = number(obj, key, lo, hi)? {
            if v.fract() != 0.0 {
                return Err(invalid(format!("{key} must be an integer")));
            }
            *target = v as u32;
        }
    }
    for (key, target, lo, hi) in [
        ("ditherAmount", &mut d.dither_amount, 0.0, 1.0),
        ("outlineWidth", &mut d.outline_width, 0.5, 6.0),
        ("outlineSharpness", &mut d.outline_sharpness, 0.0, 1.0),
        ("edgeRetention", &mut d.edge_retention, 0.0, 1.0),
        ("worldScale", &mut d.world_scale, 1.0, 512.0),
        ("softness", &mut d.softness, 0.0, 1.0),
        ("specular", &mut d.specular, 0.0, 2.0),
    ] {
        if let Some(v) = number(obj, key, lo, hi)? {
            *target = v as f32;
        }
    }
    for (key, target) in [
        ("interiorEdges", &mut d.interior_edges),
        ("worldSpace", &mut d.world_space),
        ("toon", &mut d.toon),
        ("palette", &mut d.palette),
        ("outlines", &mut d.outlines),
        ("ditherOutlines", &mut d.dither_outlines),
    ] {
        boolean(obj, key, target)?;
    }
    if let Some(v) = get(obj, "dither")? {
        d.dither = Dither::from_name(
            &v.as_string()
                .ok_or_else(|| invalid("dither must be a string"))?,
        )?;
    }
    d.validate()?;
    Ok(d)
}
fn write(d: CelSettings) -> JsValue {
    let obj = Object::new();
    for (key, value) in [
        ("bands", d.bands as f64),
        ("hues", d.hues as f64),
        ("chroma", d.chroma as f64),
        ("ditherAmount", d.dither_amount as f64),
        ("outlineWidth", d.outline_width as f64),
        ("outlineSharpness", d.outline_sharpness as f64),
        ("edgeRetention", d.edge_retention as f64),
        ("worldScale", d.world_scale as f64),
        ("softness", d.softness as f64),
        ("specular", d.specular as f64),
    ] {
        let _ = Reflect::set(&obj, &key.into(), &JsValue::from_f64(value));
    }
    for (key, value) in [
        ("interiorEdges", d.interior_edges),
        ("worldSpace", d.world_space),
        ("toon", d.toon),
        ("palette", d.palette),
        ("outlines", d.outlines),
        ("ditherOutlines", d.dither_outlines),
    ] {
        let _ = Reflect::set(&obj, &key.into(), &JsValue::from_bool(value));
    }
    let _ = Reflect::set(&obj, &"dither".into(), &d.dither.name().into());
    obj.into()
}
#[wasm_bindgen(js_class = Engine)]
impl Engine {
    /// Merges controls, enables cel and selects a neutral linear display transform.
    #[wasm_bindgen(js_name = setCel)]
    pub fn set_cel(&mut self, settings: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            let h = st.features.require_mut::<CelHandle>("cel")?;
            let d = read(&settings, h.get())?;
            h.set(d)?;
            st.pipeline.borrow_mut().set_enabled("cel.resolve", true)?;
            st.tonemap.set(raa_tonemap::ToneMapping {
                mapper: raa_tonemap::ToneMapper::Linear,
                exposure_ev: 0.0,
                look: raa_tonemap::Look::None,
                dither: false,
            })?;
            Ok(())
        })
    }
    #[wasm_bindgen(js_name = getCel)]
    pub fn get_cel(&mut self) -> Result<JsValue, JsValue> {
        self.with_state(|_, st| Ok(write(st.features.require_mut::<CelHandle>("cel")?.get())))
    }
}
