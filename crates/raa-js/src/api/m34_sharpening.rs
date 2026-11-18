//! m34 sharpening API (S47 'JS API (m34)'): `setSharpening` and `sharpening`, plus the install
//! of pass `sharpen` (disabled by default) and the frame hook mirroring the pipeline flag into
//! the handle. The pass has no per-view state.
//!
//! The pure part (`merge_sharpening`) compiles on the host and is tested by
//! `tests/sharpening.rs`; the wasm32 part sits in `web`.

use raa_core::RaaError;
use raa_sharpen::SharpenDesc;

/// `current` with each Some field replaced (sharpness cast to f32), validated (`post.invalid`).
pub(crate) fn merge_sharpening(
    current: SharpenDesc,
    enabled: Option<bool>,
    sharpness: Option<f64>,
) -> Result<SharpenDesc, RaaError> {
    let mut d = current;
    if let Some(v) = enabled {
        d.enabled = v;
    }
    if let Some(v) = sharpness {
        d.sharpness = v as f32;
    }
    d.validate()?;
    Ok(d)
}

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install};

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use crate::convert::{INVALID_ARGUMENT, api_error};
    use crate::engine::{Engine, InstallCtx};
    use crate::state::EngineState;
    use js_sys::{Object, Reflect};
    use raa_core::GpuContext;
    use raa_sharpen::{SHARPEN_PASS_NAME, SharpenFeature, SharpenHandle};
    use raa_web::FrameInput;
    use wasm_bindgen::prelude::*;

    pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
        let handle = SharpenFeature::install(ctx.gpu, ctx.pipeline)?;
        ctx.features.insert(handle);
        Ok(())
    }

    fn pipeline_enabled(st: &EngineState) -> bool {
        st.pipeline
            .borrow()
            .entry_enabled(SHARPEN_PASS_NAME)
            .unwrap_or(false)
    }

    pub(crate) fn frame(
        st: &mut EngineState,
        _: &GpuContext,
        _: &FrameInput,
    ) -> Result<(), RaaError> {
        let enabled = pipeline_enabled(st);
        if let Some(h) = st.features.get::<SharpenHandle>() {
            h.set_enabled_flag(enabled);
        }
        Ok(())
    }

    fn invalid(message: String) -> RaaError {
        api_error(INVALID_ARGUMENT, message)
    }

    /// The value at `key`, or None when it is missing, undefined or null.
    fn get(obj: &JsValue, key: &str) -> Result<Option<JsValue>, RaaError> {
        let v = Reflect::get(obj, &JsValue::from_str(key))
            .map_err(|_| invalid(format!("cannot read '{key}'")))?;
        Ok(if v.is_undefined() || v.is_null() {
            None
        } else {
            Some(v)
        })
    }

    fn optional_f64(obj: &JsValue, key: &str) -> Result<Option<f64>, RaaError> {
        match get(obj, key)? {
            None => Ok(None),
            Some(v) => v
                .as_f64()
                .map(Some)
                .ok_or_else(|| invalid(format!("`{key}` must be a number"))),
        }
    }

    fn parse_patch(settings: &JsValue) -> Result<(Option<bool>, Option<f64>), RaaError> {
        if !settings.is_object() {
            return Err(invalid("setSharpening expects an object".to_string()));
        }
        let enabled = match get(settings, "enabled")? {
            None => None,
            Some(v) => Some(
                v.as_bool()
                    .ok_or_else(|| invalid("`enabled` must be a boolean".to_string()))?,
            ),
        };
        Ok((enabled, optional_f64(settings, "sharpness")?))
    }

    fn main_handle(st: &EngineState) -> Result<SharpenHandle, RaaError> {
        st.features
            .get::<SharpenHandle>()
            .cloned()
            .ok_or_else(|| invalid("sharpen is not installed".to_string()))
    }

    fn set(obj: &Object, key: &str, value: JsValue) {
        let _ = Reflect::set(obj, &JsValue::from_str(key), &value);
    }

    fn apply(st: &mut EngineState, settings: &JsValue) -> Result<(), RaaError> {
        let (enabled, sharpness) = parse_patch(settings)?;
        let handle = main_handle(st)?;
        let mut current = handle.get();
        current.enabled = pipeline_enabled(st);
        let merged = merge_sharpening(current, enabled, sharpness)?;
        let mut pipeline = st
            .pipeline
            .try_borrow_mut()
            .map_err(|_| api_error("pipeline.busy", "a frame is executing".to_string()))?;
        let prev = handle.get();
        handle.set(merged)?;
        if let Err(e) = pipeline.set_enabled(SHARPEN_PASS_NAME, merged.enabled) {
            handle.set(prev)?;
            return Err(e);
        }
        handle.set_enabled_flag(merged.enabled);
        Ok(())
    }

    #[wasm_bindgen(js_class = Engine)]
    impl Engine {
        #[wasm_bindgen(js_name = setSharpening)]
        pub fn set_sharpening(&mut self, settings: JsValue) -> Result<(), JsValue> {
            self.with_state(|_, st| apply(st, &settings))
        }

        #[wasm_bindgen(js_name = sharpening)]
        pub fn sharpening(&self) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| {
                let d = main_handle(st)?.get();
                let obj = Object::new();
                set(&obj, "enabled", JsValue::from_bool(pipeline_enabled(st)));
                set(&obj, "sharpness", JsValue::from_f64(d.sharpness as f64));
                Ok(obj.into())
            })
        }
    }
}
