//! Animation brick exports and deterministic case registration.
use crate::Case;

/// Public brick contract version, independent of the Rust workspace version.
pub const BRICK_VERSION: &str = "0.5.0";
pub const BRICK_NAME: &str = "raa-anim";

// Installed by the automatically discovered clips module.
pub(crate) const CLIP_SAMPLE: Case = Case {
    name: "clip-sample",
    run: crate::clips::clip_sample_selftest,
};

// Installed by the automatically discovered graph module (M9).
pub(crate) const GRAPH_RUN: Case = Case {
    name: "graph-run",
    run: crate::graph::graph_run_selftest,
};

pub fn register(cases: &mut Vec<Case>) {
    cases.push(Case {
        name: "det-selftest",
        run: raa_det::selftest,
    });
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::{BRICK_NAME, BRICK_VERSION};
    use crate::{ApiError, dispatch};
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(js_name = brickInfo)]
    pub fn brick_info() -> Result<JsValue, JsValue> {
        let object = js_sys::Object::new();
        js_sys::Reflect::set(&object, &"name".into(), &BRICK_NAME.into())?;
        js_sys::Reflect::set(&object, &"version".into(), &BRICK_VERSION.into())?;
        Ok(object.into())
    }

    fn error(error: ApiError) -> JsValue {
        let (code, message) = match error {
            ApiError::UnknownCase(case) => ("api.unknown_case", format!("Unknown case: {case}")),
            ApiError::InvalidInput("case") => ("api.invalid_input", "case must be a string".into()),
            ApiError::InvalidInput(field) => {
                ("api.invalid_input", format!("{field} must be a u32 number"))
            }
        };
        let value = js_sys::TypeError::new(&message);
        js_sys::Reflect::set(&value, &"code".into(), &code.into()).expect("fresh error object");
        value.into()
    }

    #[wasm_bindgen(js_name = selftest)]
    pub fn selftest(case: JsValue, seed: JsValue, ticks: JsValue) -> Result<JsValue, JsValue> {
        let case = case
            .as_string()
            .ok_or_else(|| error(ApiError::InvalidInput("case")))?;
        let seed = seed
            .as_f64()
            .ok_or_else(|| error(ApiError::InvalidInput("seed")))?;
        let ticks = ticks
            .as_f64()
            .ok_or_else(|| error(ApiError::InvalidInput("ticks")))?;
        let result = dispatch(&case, seed, ticks).map_err(error)?;
        let object = js_sys::Object::new();
        for (key, value) in [
            ("case", JsValue::from(result.case)),
            ("seed", JsValue::from(result.seed)),
            ("ticks", JsValue::from(result.ticks)),
            ("hash", JsValue::from(result.hash)),
        ] {
            js_sys::Reflect::set(&object, &key.into(), &value)?;
        }
        Ok(object.into())
    }
}
