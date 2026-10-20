//! wasm-bindgen API of the contact sheet demo.

use raa_web::{RuntimeOptions, WebRuntime, error_to_js, stats_to_js};
use wasm_bindgen::prelude::*;

use crate::app::ContactSheet;

/// Reads `options[key]` as a positive integer size, if present and valid.
fn size_option(options: &JsValue, key: &str) -> Option<u32> {
    if !options.is_object() {
        return None;
    }
    let value = js_sys::Reflect::get(options, &JsValue::from_str(key))
        .ok()?
        .as_f64()?;
    // Rejects NaN, non-positive and out-of-range values before the cast.
    (1.0..=f64::from(u32::MAX))
        .contains(&value)
        .then_some(value as u32)
}

/// The cumulative allocation counter, linked only by `alloc-count` builds.
#[cfg(feature = "alloc-count")]
const ALLOC_PROBE: Option<fn() -> u64> = Some(raa_alloc_count::allocations);
#[cfg(not(feature = "alloc-count"))]
const ALLOC_PROBE: Option<fn() -> u64> = None;

/// The running demo, driven from JS.
#[wasm_bindgen]
pub struct Demo {
    runtime: WebRuntime,
}

/// Starts the demo on `canvas`. `options` may carry `width` and `height` (canvas
/// backing size in pixels); the canvas's current size is used otherwise.
#[wasm_bindgen]
pub async fn start(canvas: web_sys::HtmlCanvasElement, options: JsValue) -> Result<Demo, JsValue> {
    let opts = RuntimeOptions {
        width: size_option(&options, "width").unwrap_or(canvas.width()),
        height: size_option(&options, "height").unwrap_or(canvas.height()),
        alloc_probe: ALLOC_PROBE,
        ..RuntimeOptions::default()
    };
    let runtime = WebRuntime::new(canvas, Box::new(ContactSheet::default()), opts)
        .await
        .map_err(|e| error_to_js(&e))?;
    Ok(Demo { runtime })
}

#[wasm_bindgen]
impl Demo {
    pub fn frame(&mut self, now_ms: f64) -> Result<(), JsValue> {
        self.runtime.frame(now_ms).map_err(|e| error_to_js(&e))
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.runtime.resize(width, height);
    }

    pub fn stats(&self) -> JsValue {
        stats_to_js(&self.runtime.stats(), self.runtime.adapter())
    }

    pub async fn probe_non_empty(&mut self, now_ms: f64) -> Result<bool, JsValue> {
        self.runtime
            .probe_non_empty(now_ms)
            .await
            .map_err(|e| error_to_js(&e))
    }
}
