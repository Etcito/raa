//! Non-blocking pipeline creation (vendored wgpu `webgpu::async_pipelines`) and the pipeline
//! warm-up of a load: the host turns the mode on, renders hidden frames with
//! `setPipelineWarmup(true)` (every instance selected, so every material variant is
//! requested) until `pipelineStats().pending` is 0 and a frame creates no new pipeline, then
//! shows the first frame. In game, a pipeline still compiling makes its pass skip its draws.

use js_sys::{Object, Reflect};
use wasm_bindgen::prelude::*;
use wgpu::webgpu::async_pipelines;

use crate::engine::Engine;

fn put(object: &Object, key: &str, value: JsValue) {
    Reflect::set(object, &key.into(), &value).expect("fresh object");
}

/// The module-level switch of the async pipeline mode (the same as
/// `Engine.setAsyncPipelines`), so a host turns it on before `createEngine` and the
/// pipelines the engine installs at creation compile asynchronously too.
#[wasm_bindgen(js_name = setAsyncPipelines)]
pub fn set_async_pipelines_module(on: bool) {
    async_pipelines::set_enabled(on);
}

#[wasm_bindgen]
impl Engine {
    /// Creates new render/compute pipelines with `createRenderPipelineAsync` /
    /// `createComputePipelineAsync` (on) instead of the synchronous calls (off, the default).
    /// While a pipeline compiles, the passes that set it skip their draws and dispatches.
    /// The mode is shared by every engine of the page.
    #[wasm_bindgen(js_name = setAsyncPipelines)]
    pub fn set_async_pipelines(&self, on: bool) {
        async_pipelines::set_enabled(on);
    }

    /// `{enabled, pending, pendingInstall, createdAsync, createdSync, forcedSync, failed, skippedDraws,
    /// skippedDispatches, maxCompileMs}`: `pending` pipelines are still compiling,
    /// `pendingInstall` of them were made outside a frame (render graph / feature install):
    /// wait for those before rendering, a pass may use its pipeline once (a bake). The skip
    /// counters and `maxCompileMs` count since the last `resetPipelineCounters()`.
    #[wasm_bindgen(js_name = pipelineStats)]
    pub fn pipeline_stats(&self) -> JsValue {
        let s = async_pipelines::stats();
        let o = Object::new();
        put(&o, "enabled", s.enabled.into());
        put(&o, "pending", s.pending.into());
        put(&o, "pendingInstall", s.pending_install.into());
        put(&o, "createdAsync", (s.created_async as f64).into());
        put(&o, "createdSync", (s.created_sync as f64).into());
        put(&o, "forcedSync", (s.forced_sync as f64).into());
        put(&o, "failed", (s.failed as f64).into());
        put(&o, "skippedDraws", (s.skipped_draws as f64).into());
        put(
            &o,
            "skippedDispatches",
            (s.skipped_dispatches as f64).into(),
        );
        put(&o, "maxCompileMs", s.max_compile_ms.into());
        let labels = js_sys::Array::new();
        for label in async_pipelines::skipped_labels() {
            labels.push(&label.into());
        }
        put(&o, "skippedLabels", labels.into());
        o.into()
    }

    /// Clears the skip counters and `maxCompileMs` of `pipelineStats()`.
    #[wasm_bindgen(js_name = resetPipelineCounters)]
    pub fn reset_pipeline_counters(&self) {
        async_pipelines::reset_counters();
    }

    /// Pipeline warm-up: while on, the camera selection draws every instance (no frustum
    /// test), so a rendered frame requests the pipelines of every material in the scene.
    /// Render such frames hidden (behind the loading notice) and turn it off before showing.
    #[wasm_bindgen(js_name = setPipelineWarmup)]
    pub fn set_pipeline_warmup(&self, on: bool) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            st.lit.handle().set_select_all(on);
            Ok(())
        })
    }
}
