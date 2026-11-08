//! Native checks for the trace binding's packed input contract.

#[cfg(test)]
mod tests {
    #[test]
    fn trace_rejects_bend_without_rail_record() {
        let mut words = vec![raa_sdf_trace::MAGIC, 1, 1, 1, 1, 15, 1, 0];
        words.extend([0, 1, 0, 15]);
        words.extend([-1.0_f32, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0].map(f32::to_bits));
        words.push(0);
        words.push(raa_sdf_trace::opcode::SPHERE);
        let matrix = [
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ];
        words.extend(matrix.map(f32::to_bits));
        words.extend([1.0_f32, 0.0, 0.5].map(f32::to_bits));
        words.extend([0; 8]);
        words.extend(matrix.map(f32::to_bits));
        words.extend(matrix.map(f32::to_bits));
        words.extend([
            1.0_f32.to_bits(),
            0,
            u32::MAX,
            0,
            1,
            0.5_f32.to_bits(),
            2.0_f32.to_bits(),
            40,
        ]);
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        assert!(
            raa_sdf_trace::parse(&bytes).is_err(),
            "missing bend payload must be refused"
        );
    }
}

#[cfg(target_arch = "wasm32")]
mod binding {
    use crate::{convert::api_error, engine::Engine, state::EngineState};
    use js_sys::{Float32Array, Object, Reflect, Uint8Array};
    use raa_core::RaaError;
    use raa_sdf_trace::{FRAME_BYTES, TraceError, TraceHandle, target_size};
    use wasm_bindgen::prelude::*;

    fn trace_error(error: TraceError) -> RaaError {
        api_error(error.code(), error.to_string())
    }

    pub(crate) fn prepare_view(
        st: &mut EngineState,
        index: usize,
        size: (u32, u32),
    ) -> Result<(), RaaError> {
        let Some(handle) = st.features.get::<TraceHandle>() else {
            return Ok(());
        };
        let mut camera = st.scene.camera;
        if let Some(view) = st
            .features
            .get::<crate::api::ViewsState>()
            .and_then(|v| v.camera(index))
        {
            camera.eye = raa_math::Vec3::from_array(view.eye);
            camera.target = raa_math::Vec3::from_array(view.target);
            camera.up = raa_math::Vec3::Y;
            camera.fov_y_rad = view.fov_y_deg.to_radians();
            camera.near = view.near;
            camera.far = view.far;
            camera.ortho_height = None;
        }
        camera.set_viewport(size.0, size.1);
        let original = std::mem::replace(&mut st.scene.camera, camera);
        let mut frame = [0u8; raa_lit::FRAME_GPU_SIZE];
        let result = raa_lit::write_frame_gpu(
            &mut frame,
            &st.scene,
            &raa_lit::FrameParams {
                viewport: size,
                ..Default::default()
            },
        );
        st.scene.camera = original;
        if result.is_ok() {
            handle
                .borrow_mut()
                .frame
                .copy_from_slice(&frame[..FRAME_BYTES]);
        }
        result
    }

    #[wasm_bindgen]
    impl Engine {
        #[wasm_bindgen(js_name = setTraceScene)]
        pub fn set_trace_scene(&self, program: Uint8Array) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                st.features
                    .require_mut::<TraceHandle>("trace")?
                    .borrow_mut()
                    .set_program(&program.to_vec())
                    .map_err(trace_error)
            })
        }

        #[wasm_bindgen(js_name = updateTraceInstances)]
        pub fn update_trace_instances(&self, words: Float32Array) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                st.features
                    .require_mut::<TraceHandle>("trace")?
                    .borrow_mut()
                    .update_instances(&words.to_vec())
                    .map_err(trace_error)
            })
        }

        #[wasm_bindgen(js_name = clearTraceScene)]
        pub fn clear_trace_scene(&self) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                st.features
                    .require_mut::<TraceHandle>("trace")?
                    .borrow_mut()
                    .clear();
                Ok(())
            })
        }

        #[wasm_bindgen(js_name = traceStats)]
        pub fn trace_stats(&self) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| {
                let stats = st
                    .features
                    .require_mut::<TraceHandle>("trace")?
                    .borrow()
                    .stats(target_size(self.viewport.0, self.viewport.1));
                let out = Object::new();
                for (name, value) in [
                    ("sources", stats.sources),
                    ("instances", stats.instances),
                    ("bvhNodes", stats.bvh_nodes),
                    ("width", stats.width as usize),
                    ("height", stats.height as usize),
                    ("placeholders", stats.placeholders),
                ] {
                    let _ = Reflect::set(&out, &name.into(), &(value as f64).into());
                }
                Ok(out.into())
            })
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use binding::prepare_view;
