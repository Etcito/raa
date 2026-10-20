//! Conversions from Rust results to JavaScript values (wasm32 only).
//!
//! [`stats_to_js`] builds the plain stats object the demo page and the GPU client read,
//! and [`error_to_js`] turns a [`RaaError`] into a JS `Error` carrying the stable code
//! property. Both use [`js_sys::Reflect::set`] on freshly created objects instead of serde.

use js_sys::Reflect;
use raa_core::{AdapterSummary, RaaError};
use wasm_bindgen::JsValue;

use crate::stats::FrameStats;

/// Sets `key` on a freshly created plain JS object.
///
/// The `Reflect::set` result is ignored deliberately: setting a property on a fresh plain
/// object cannot fail, because the object is extensible and none of the keys used here is
/// an accessor.
fn set(object: &JsValue, key: &str, value: &JsValue) {
    let _ = Reflect::set(object, &JsValue::from_str(key), value);
}

/// A JS number (doubles are the only number type in JS).
fn number(value: f64) -> JsValue {
    JsValue::from_f64(value)
}

/// A JS number, or `null` when the value is not known yet.
fn optional_number(value: Option<f64>) -> JsValue {
    value.map_or(JsValue::NULL, JsValue::from_f64)
}

/// Builds a plain object with exactly these keys: `adapter` (a nested object with
/// `vendor`, `architecture`, `device`, `description` and `isFallbackAdapter`), `hardware`
/// (from [`AdapterSummary::is_hardware`]), `firstFrameMs` (number or `null`), `frames`,
/// `cpuMsMedian`, `cpuMsP95`, `instances`, `drawCalls`, `triangles` and
/// `allocsPerFrameTotal` (number or `null`). Every number is an f64, as JS requires.
pub fn stats_to_js(stats: &FrameStats, adapter: &AdapterSummary) -> JsValue {
    let adapter_object = js_sys::Object::new();
    let adapter_value = adapter_object.as_ref();
    set(adapter_value, "vendor", &JsValue::from_str(&adapter.vendor));
    set(
        adapter_value,
        "architecture",
        &JsValue::from_str(&adapter.architecture),
    );
    set(adapter_value, "device", &JsValue::from_str(&adapter.device));
    set(
        adapter_value,
        "description",
        &JsValue::from_str(&adapter.description),
    );
    set(
        adapter_value,
        "isFallbackAdapter",
        &JsValue::from_bool(adapter.is_fallback),
    );

    let object = js_sys::Object::new();
    let value = object.as_ref();
    set(value, "adapter", adapter_value);
    set(
        value,
        "hardware",
        &JsValue::from_bool(adapter.is_hardware()),
    );
    set(
        value,
        "firstFrameMs",
        &optional_number(stats.first_frame_ms),
    );
    set(value, "frames", &number(stats.frames as f64));
    set(value, "cpuMsMedian", &number(stats.cpu_ms_median));
    set(value, "cpuMsP95", &number(stats.cpu_ms_p95));
    set(
        value,
        "instances",
        &number(f64::from(stats.counters.instances)),
    );
    set(
        value,
        "drawCalls",
        &number(f64::from(stats.counters.draw_calls)),
    );
    set(value, "triangles", &number(stats.counters.triangles as f64));
    set(
        value,
        "allocsPerFrameTotal",
        &optional_number(stats.allocs_last_frame.map(|allocs| allocs as f64)),
    );
    object.into()
}

/// Builds a JS `Error` whose name is `RaaError`, whose message is the `Display` output
/// (code and message) and which carries the stable code in a `code` property.
pub fn error_to_js(error: &RaaError) -> JsValue {
    let js_error = js_sys::Error::new(&error.to_string());
    js_error.set_name("RaaError");
    // As in `set`: a freshly created Error is extensible and `code` is not an accessor,
    // so setting the property cannot fail; the result is ignored deliberately.
    let _ = Reflect::set(
        js_error.as_ref(),
        &JsValue::from_str("code"),
        &JsValue::from_str(error.code()),
    );
    js_error.into()
}
