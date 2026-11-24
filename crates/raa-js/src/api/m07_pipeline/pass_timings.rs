//! On-demand timestamp timings for the most recent sampled pipeline frame.

use js_sys::{Array, Object};
use raa_pipeline::PassGpuTimings;
use wasm_bindgen::prelude::*;

use super::{Engine, set, with_pipeline};

#[wasm_bindgen(js_class = Engine)]
impl Engine {
    /// Returns the latest GPU pass timing sample and requests another real frame sample if idle.
    #[wasm_bindgen(js_name = passGpuTimings)]
    pub fn pass_gpu_timings(&self) -> Result<JsValue, JsValue> {
        self.with_state(|gpu, st| {
            with_pipeline(st, |pipeline| {
                let out = Object::new();
                match pipeline.pass_gpu_timings(gpu) {
                    PassGpuTimings::Unavailable(reason) => {
                        set(&out, "status", &JsValue::from_str("unavailable"));
                        set(&out, "reason", &JsValue::from_str(reason));
                    }
                    PassGpuTimings::Pending => {
                        set(&out, "status", &JsValue::from_str("pending"));
                    }
                    PassGpuTimings::Sample(sample) => {
                        set(&out, "status", &JsValue::from_str("sample"));
                        set(&out, "frame", &JsValue::from_f64(sample.frame as f64));
                        set(&out, "width", &JsValue::from(sample.size.0));
                        set(&out, "height", &JsValue::from(sample.size.1));
                        let passes: Array = sample
                            .passes
                            .iter()
                            .map(|(name, ms)| {
                                let pass = Object::new();
                                set(&pass, "pass", &JsValue::from_str(name));
                                set(&pass, "ms", &ms.map_or(JsValue::NULL, JsValue::from_f64));
                                JsValue::from(pass)
                            })
                            .collect();
                        set(&out, "passes", &passes);
                    }
                }
                Ok(out.into())
            })
        })
    }
}
