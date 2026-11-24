//! Offline capture of one submitted real view and independent reference samples.

use js_sys::{Object, Reflect, Uint8Array};
use raa_core::RaaError;
use raa_sdf_trace::mesh::MeshTraceHandle;
use wasm_bindgen::prelude::*;

use crate::{convert::api_error, engine::Engine};

fn invalid(reason: &str) -> RaaError {
    api_error("capture.invalid", reason)
}

#[wasm_bindgen]
impl Engine {
    /// Reads the already submitted real frame. This method does not advance renderer history.
    #[wasm_bindgen(js_name = captureTrainingTaps)]
    pub async fn capture_training_taps(&mut self, frame_id: f64) -> Result<JsValue, JsValue> {
        let runtime = self.capture_runtime()?;
        if runtime.stats().frames.checked_sub(1).map(|id| id as f64) != Some(frame_id) {
            return Err(raa_web::error_to_js(&invalid(
                "training taps require the latest real frame",
            )));
        }
        let (textures, jitter_values) = self.with_state(|gpu, st| {
            if !gpu.summary().is_hardware() {
                return Err(invalid("training capture requires hardware WebGPU"));
            }
            let pipeline = st
                .pipeline
                .try_borrow()
                .map_err(|_| invalid("pipeline is busy"))?;
            let submission = pipeline.scene_submission();
            if submission.record_failed
                || submission.submitted_mask != 1
                || submission.pending_mask != 0
            {
                return Err(invalid("training taps require one submitted primary view"));
            }
            let targets = pipeline
                .targets()
                .ok_or_else(|| invalid("missing real-frame targets"))?;
            let resources = pipeline.resources();
            let hdr = resources
                .get("trace.mesh", "color")
                .or_else(|| resources.get("forward", "color"))
                .ok_or_else(|| invalid("missing scene-linear HDR tap"))?;
            let mut textures = vec![
                ("hdr", hdr.view.texture().clone()),
                ("depth", targets.depth_texture().clone()),
                ("normal", targets.normal_texture().clone()),
                ("motion", targets.motion_texture().clone()),
            ];
            if let Some(trace) = resources.get("ssr.confidence", "trace") {
                textures.push(("reflectionHit", trace.view.texture().clone()));
            }
            let reflection = [
                ("reflectionTrace", "ssr.trace#ssrTrace", "trace"),
                ("reflectionMeta", "ssr.trace#ssrTrace", "hitMeta"),
                ("reflectionIncident", "ssr.incident#incident", "incident"),
                (
                    "reflectionReconstruction",
                    "neural.reflection-reconstruction#reflectionModel",
                    "incident",
                ),
            ];
            for (name, producer, port) in reflection {
                if let Some(image) = resources.get(producer, port) {
                    textures.push((name, image.view.texture().clone()));
                }
            }
            let size = targets.size();
            if textures
                .iter()
                .any(|(_, texture)| (texture.width(), texture.height()) != size)
            {
                return Err(invalid("scene-aligned training tap extents differ"));
            }
            Ok((textures, st.lit.handle().camera_jitter_ndc()))
        })?;
        let output = Object::new();
        let jitter = js_sys::Array::new();
        for value in jitter_values {
            jitter.push(&value.into());
        }
        Reflect::set(&output, &"jitterNdc".into(), &jitter)?;
        for (name, texture) in textures {
            let bytes = raa_web::read_capture_texture(runtime.gpu(), &texture)
                .await
                .map_err(|err| raa_web::error_to_js(&err))?;
            Reflect::set(&output, &name.into(), &Uint8Array::from(bytes.as_slice()))?;
        }
        Ok(output.into())
    }

    /// Renders a seeded offline reference sample batch at the caller's fixed pose and time.
    #[wasm_bindgen(js_name = captureTrainingReference)]
    pub async fn capture_training_reference(
        &mut self,
        time_ms: f64,
        seed: u32,
        samples: u32,
    ) -> Result<JsValue, JsValue> {
        if !(1..=32).contains(&samples) {
            return Err(raa_web::error_to_js(&invalid(
                "offline reference samples must be within 1..=32",
            )));
        }
        let handle = self.with_state(|gpu, st| {
            if !gpu.summary().is_hardware() {
                return Err(invalid("training capture requires hardware WebGPU"));
            }
            let handle = st
                .features
                .get::<MeshTraceHandle>()
                .cloned()
                .ok_or_else(|| invalid("offline reference requires mesh tracing"))?;
            if !handle.borrow().settings.reference {
                return Err(invalid(
                    "offline reference requires the reference path mode",
                ));
            }
            Ok(handle)
        })?;
        handle
            .borrow_mut()
            .set_reference_sample(Some((seed, samples)));
        let result = self.capture_mesh_lighting(time_ms, true, Some(true)).await;
        handle.borrow_mut().set_reference_sample(None);
        result
    }
}
