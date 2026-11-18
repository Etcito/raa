//! m33 automatic exposure API (S46 'JS API (m33)'): `setAutoExposure`, `autoExposure` and
//! `autoExposureState`, plus the install of pass `exposure.auto` (disabled by default) and the
//! frame hook mirroring the pipeline flag into the main and per-view handles.
//!
//! The pure part (`AutoExposurePatch`, `merge_auto_exposure`) compiles on the host and is tested
//! by `tests/auto_exposure.rs`; the wasm32 part sits in `web`.

use raa_core::RaaError;
use raa_exposure::AutoExposureDesc;

/// The keys present in a `setAutoExposure` argument (None = absent, undefined or null).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct AutoExposurePatch {
    pub(crate) enabled: Option<bool>,
    pub(crate) compensation_ev: Option<f64>,
    pub(crate) low_percent: Option<f64>,
    pub(crate) high_percent: Option<f64>,
    pub(crate) speed_up: Option<f64>,
    pub(crate) speed_down: Option<f64>,
    pub(crate) min_ev: Option<f64>,
    pub(crate) max_ev: Option<f64>,
}

/// `current` with each present key of `patch` replaced, validated (`post.invalid`).
pub(crate) fn merge_auto_exposure(
    current: AutoExposureDesc,
    patch: &AutoExposurePatch,
) -> Result<AutoExposureDesc, RaaError> {
    let mut d = current;
    if let Some(v) = patch.enabled {
        d.enabled = v;
    }
    let fields = [
        (patch.compensation_ev, &mut d.compensation_ev),
        (patch.low_percent, &mut d.low_percent),
        (patch.high_percent, &mut d.high_percent),
        (patch.speed_up, &mut d.speed_up),
        (patch.speed_down, &mut d.speed_down),
        (patch.min_ev, &mut d.min_ev),
        (patch.max_ev, &mut d.max_ev),
    ];
    for (value, slot) in fields {
        if let Some(v) = value {
            *slot = v as f32;
        }
    }
    d.validate()?;
    Ok(d)
}

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install};

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use crate::api::m07_pipeline::view_auto_exposure;
    use crate::convert::{INVALID_ARGUMENT, api_error};
    use crate::engine::{Engine, InstallCtx};
    use crate::state::EngineState;
    use js_sys::{Object, Reflect};
    use raa_core::GpuContext;
    use raa_exposure::{AUTO_EXPOSURE_PASS_NAME, AutoExposureFeature, AutoExposureHandle};
    use raa_web::FrameInput;
    use wasm_bindgen::prelude::*;

    pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
        let handle = AutoExposureFeature::install(ctx.gpu, ctx.pipeline)?;
        ctx.features.insert(handle);
        Ok(())
    }

    fn pipeline_enabled(st: &EngineState) -> bool {
        st.pipeline
            .borrow()
            .entry_enabled(AUTO_EXPOSURE_PASS_NAME)
            .unwrap_or(false)
    }

    pub(crate) fn frame(
        st: &mut EngineState,
        _: &GpuContext,
        _: &FrameInput,
    ) -> Result<(), RaaError> {
        let enabled = pipeline_enabled(st);
        if let Some(h) = st.features.get::<AutoExposureHandle>() {
            h.set_enabled_flag(enabled);
        }
        let mut index = 0;
        while let Some(h) = view_auto_exposure(st, index) {
            h.set_enabled_flag(enabled);
            index += 1;
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

    fn parse_patch(settings: &JsValue) -> Result<AutoExposurePatch, RaaError> {
        if !settings.is_object() {
            return Err(invalid("setAutoExposure expects an object".to_string()));
        }
        let enabled = match get(settings, "enabled")? {
            None => None,
            Some(v) => Some(
                v.as_bool()
                    .ok_or_else(|| invalid("`enabled` must be a boolean".to_string()))?,
            ),
        };
        Ok(AutoExposurePatch {
            enabled,
            compensation_ev: optional_f64(settings, "compensationEv")?,
            low_percent: optional_f64(settings, "lowPercent")?,
            high_percent: optional_f64(settings, "highPercent")?,
            speed_up: optional_f64(settings, "speedUp")?,
            speed_down: optional_f64(settings, "speedDown")?,
            min_ev: optional_f64(settings, "minEv")?,
            max_ev: optional_f64(settings, "maxEv")?,
        })
    }

    fn main_handle(st: &EngineState) -> Result<AutoExposureHandle, RaaError> {
        st.features
            .get::<AutoExposureHandle>()
            .cloned()
            .ok_or_else(|| invalid("exposure.auto is not installed".to_string()))
    }

    fn set(obj: &Object, key: &str, value: JsValue) {
        let _ = Reflect::set(obj, &JsValue::from_str(key), &value);
    }

    fn apply(st: &mut EngineState, settings: &JsValue) -> Result<(), RaaError> {
        let patch = parse_patch(settings)?;
        let handle = main_handle(st)?;
        let mut current = handle.get();
        current.enabled = pipeline_enabled(st);
        let merged = merge_auto_exposure(current, &patch)?;
        let mut pipeline = st
            .pipeline
            .try_borrow_mut()
            .map_err(|_| api_error("pipeline.busy", "a frame is executing".to_string()))?;
        let prev = handle.get();
        handle.set(merged)?;
        if let Err(e) = pipeline.set_enabled(AUTO_EXPOSURE_PASS_NAME, merged.enabled) {
            handle.set(prev)?;
            return Err(e);
        }
        handle.set_enabled_flag(merged.enabled);
        Ok(())
    }

    #[wasm_bindgen(js_class = Engine)]
    impl Engine {
        #[wasm_bindgen(js_name = setAutoExposure)]
        pub fn set_auto_exposure(&mut self, settings: JsValue) -> Result<(), JsValue> {
            self.with_state(|_, st| apply(st, &settings))
        }

        #[wasm_bindgen(js_name = autoExposure)]
        pub fn auto_exposure(&self) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| {
                let d = main_handle(st)?.get();
                let obj = Object::new();
                set(&obj, "enabled", JsValue::from_bool(pipeline_enabled(st)));
                let numbers = [
                    ("compensationEv", d.compensation_ev),
                    ("lowPercent", d.low_percent),
                    ("highPercent", d.high_percent),
                    ("speedUp", d.speed_up),
                    ("speedDown", d.speed_down),
                    ("minEv", d.min_ev),
                    ("maxEv", d.max_ev),
                ];
                for (key, value) in numbers {
                    set(&obj, key, JsValue::from_f64(value as f64));
                }
                Ok(obj.into())
            })
        }

        #[wasm_bindgen(js_name = autoExposureState)]
        pub fn auto_exposure_state(&self, viewport: Option<u32>) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| {
                let index = viewport.unwrap_or(0);
                let handle = match view_auto_exposure(st, index as usize) {
                    Some(h) => h,
                    None if index == 0 => main_handle(st)?,
                    None => return Err(invalid(format!("viewport {index} does not exist"))),
                };
                let s = handle.state();
                let number = |v: Option<f32>| v.map_or(JsValue::NULL, |x| (x as f64).into());
                let obj = Object::new();
                set(&obj, "exposureEv", number(s.exposure_ev));
                set(&obj, "averageLuminance", number(s.average_luminance));
                set(&obj, "settled", JsValue::from_bool(s.settled));
                set(&obj, "pending", JsValue::from_bool(s.pending));
                Ok(obj.into())
            })
        }
    }
}
