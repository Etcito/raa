//! Installation, statistics and checkpoints of the neural radiance cache
//! (`neural.radiance-cache`). The render graph configures it; there is no standalone setter.

use js_sys::{Array, Object, Reflect, Uint8Array};
use raa_core::RaaError;
use raa_sdf_trace::mesh::{MeshTraceHandle, NrcFeature, NrcHandle};
use wasm_bindgen::prelude::*;

use crate::engine::{Engine, InstallCtx};

pub(crate) fn install_for_graph(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
    let mesh = ctx
        .features
        .require_mut::<MeshTraceHandle>("mesh trace")?
        .clone();
    let handle = NrcFeature::install(&mut ctx.pipeline.borrow_mut(), mesh)?;
    ctx.features.insert(handle);
    Ok(())
}

fn put(object: &Object, key: &str, value: JsValue) {
    Reflect::set(object, &key.into(), &value).expect("fresh object");
}

#[wasm_bindgen]
impl Engine {
    /// Snapshot of the active neural radiance cache, or null.
    #[wasm_bindgen(js_name = nrcStats)]
    pub fn nrc_stats(&self) -> Result<JsValue, JsValue> {
        self.with_state(|_, st| {
            let Some(handle) = st.features.get::<NrcHandle>() else {
                return Ok(JsValue::NULL);
            };
            let state = handle.borrow();
            let result = Object::new();
            put(&result, "active", state.active().into());
            let s = state.settings;
            put(&result, "hiddenLayers", s.hidden_layers.into());
            put(&result, "frequencies", s.frequencies.into());
            put(&result, "trainSamples", s.train_samples.into());
            put(&result, "train", s.train.into());
            put(&result, "bounces", s.bounces.into());
            put(&result, "divisor", s.divisor.into());
            if let Some(stats) = state.stats() {
                put(&result, "weights", stats.weights.into());
                put(&result, "steps", (stats.steps as f64).into());
                put(&result, "frames", (stats.frames as f64).into());
                put(&result, "batch", stats.batch.into());
                put(&result, "stepsPerFrame", stats.steps_per_frame.into());
                let size = Array::new();
                size.push(&stats.query_size.0.into());
                size.push(&stats.query_size.1.into());
                put(&result, "querySize", size.into());
                put(&result, "gpuBytes", (stats.gpu_bytes as f64).into());
            }
            Ok(result.into())
        })
    }

    /// Discards the learned cache and restarts from the random initialization.
    #[wasm_bindgen(js_name = resetNrc)]
    pub fn reset_nrc(&self) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            if let Some(handle) = st.features.get::<NrcHandle>() {
                handle.borrow_mut().reset();
            }
            Ok(())
        })
    }

    /// Warm start: loads weights, their moving average and the Adam state from a checkpoint
    /// produced by `nrcCheckpoint`. The hidden layer count must match the node.
    #[wasm_bindgen(js_name = setNrcCheckpoint)]
    pub fn set_nrc_checkpoint(&self, bytes: &[u8]) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            let handle = st
                .features
                .get::<NrcHandle>()
                .ok_or_else(|| RaaError::FrameGraph {
                    code: "gi.nrc.unavailable",
                    message: "The neural radiance cache is not installed.".into(),
                })?;
            handle.borrow_mut().load_checkpoint(bytes)
        })
    }

    /// Asks for a checkpoint of the cache as of the next rendered frame.
    #[wasm_bindgen(js_name = requestNrcCheckpoint)]
    pub fn request_nrc_checkpoint(&self) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            if let Some(handle) = st.features.get::<NrcHandle>() {
                handle.borrow_mut().request_checkpoint();
            }
            Ok(())
        })
    }

    /// The requested checkpoint bytes once their GPU readback has completed, else null.
    #[wasm_bindgen(js_name = nrcCheckpoint)]
    pub fn nrc_checkpoint(&self) -> Result<JsValue, JsValue> {
        self.with_state(|_, st| {
            let bytes = st
                .features
                .get::<NrcHandle>()
                .and_then(|handle| handle.borrow_mut().take_checkpoint());
            Ok(bytes.map_or(JsValue::NULL, |bytes| {
                Uint8Array::from(bytes.as_slice()).into()
            }))
        })
    }
}
