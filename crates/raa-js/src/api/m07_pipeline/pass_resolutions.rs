//! Dimensions of images published by the most recent single-view scene submission.

use js_sys::{Array, Object};
use wasm_bindgen::prelude::*;

use super::{Engine, set, with_pipeline};

#[wasm_bindgen(js_class = Engine)]
impl Engine {
    /// Published pass images as `{ pass, port, width, height, layers, mips }`.
    /// The resource table belongs to the most recently recorded view, so return no dimensions
    /// for a multi-view, incomplete or failed submission.
    #[wasm_bindgen(js_name = passImageExtents)]
    pub fn pass_image_extents(&self) -> Result<Array, JsValue> {
        self.with_state(|_, st| {
            with_pipeline(st, |pipeline| {
                let submission = pipeline.scene_submission();
                if submission.record_failed
                    || submission.pending_mask != 0
                    || submission.submitted_mask.count_ones() != 1
                {
                    return Ok(Array::new());
                }
                Ok(pipeline
                    .resources()
                    .entries()
                    .into_iter()
                    .map(|(pass, port, image)| {
                        let row = Object::new();
                        set(&row, "pass", &JsValue::from_str(&pass));
                        set(&row, "port", &JsValue::from_str(&port));
                        set(&row, "width", &JsValue::from(image.width));
                        set(&row, "height", &JsValue::from(image.height));
                        set(&row, "layers", &JsValue::from(image.layers));
                        set(&row, "mips", &JsValue::from(image.mips));
                        JsValue::from(row)
                    })
                    .collect())
            })
        })
    }
}
