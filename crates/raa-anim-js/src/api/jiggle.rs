//! Visual jiggle bones (raa_anim::jiggle) as the JavaScript `JiggleWorld` class: register one chain per worn
//! armature instance, then step every visible one in a single call per rendered frame. Visual only: callers run it
//! after the main pose, may skip it under budget, and never feed it back into a simulation.
#[cfg(target_arch = "wasm32")]
use {crate::clips::AnimationRig, wasm_bindgen::prelude::*};

pub fn register(_cases: &mut Vec<crate::Case>) {}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use raa_anim::AnimError;

    fn error(e: AnimError) -> JsValue {
        let value = js_sys::Error::new(&e.message);
        js_sys::Reflect::set(&value, &"code".into(), &e.code.into()).expect("fresh error");
        js_sys::Reflect::set(&value, &"path".into(), &e.path.into()).expect("fresh error");
        value.into()
    }

    /// All jiggle chains of one view.
    #[wasm_bindgen]
    pub struct JiggleWorld {
        inner: raa_anim::JiggleWorld,
    }

    #[wasm_bindgen]
    impl JiggleWorld {
        #[wasm_bindgen(constructor)]
        pub fn new() -> JiggleWorld {
            JiggleWorld { inner: raa_anim::JiggleWorld::new() }
        }

        /// Register one instance of `rig` (its bones' `jiggle` settings); returns its handle.
        pub fn add(&mut self, rig: &AnimationRig) -> Result<u32, JsValue> {
            self.inner.add(rig.shared_rig()).map_err(error)
        }

        /// Forget a handle (a no-op for an unknown one).
        pub fn remove(&mut self, handle: u32) {
            self.inner.remove(handle);
        }

        /// Bones of the handle's rig: its palette holds 16 floats per bone.
        #[wasm_bindgen(js_name = boneCount)]
        pub fn bone_count(&self, handle: u32) -> Result<u32, JsValue> {
            Ok(self.inner.chain(handle).map_err(error)?.bone_count() as u32)
        }

        /// Bones of the handle's rig that jiggle.
        #[wasm_bindgen(js_name = jiggleCount)]
        pub fn jiggle_count(&self, handle: u32) -> Result<u32, JsValue> {
            Ok(self.inner.chain(handle).map_err(error)?.jiggle_count() as u32)
        }

        /// The instance's animated pose: model matrices in its mount frame, 16 floats per bone; empty = rest.
        #[wasm_bindgen(js_name = setPose)]
        pub fn set_pose(&mut self, handle: u32, model: &[f32]) -> Result<(), JsValue> {
            self.inner.chain_mut(handle).map_err(error)?.set_pose(model).map_err(error)
        }

        /// Animated pose as skin matrices (model * inverse bind, 16 floats per bone); empty restores rest.
        #[wasm_bindgen(js_name = setPoseSkin)]
        pub fn set_pose_skin(&mut self, handle: u32, skin: &[f32]) -> Result<(), JsValue> {
            self.inner.chain_mut(handle).map_err(error)?.set_skin(skin).map_err(error)
        }

        /// The next step of this handle starts from its animated pose (teleport, respawn, cut).
        pub fn reset(&mut self, handle: u32) -> Result<(), JsValue> {
            self.inner.chain_mut(handle).map_err(error)?.reset();
            Ok(())
        }

        /// Live handles.
        #[wasm_bindgen(getter)]
        pub fn size(&self) -> u32 {
            self.inner.len() as u32
        }

        /// One frame of `dt` real seconds for the listed handles; `frames` holds 32 floats per handle (world matrix,
        /// then mount matrix, column-major). Returns their skin palettes concatenated in handle order.
        pub fn step(&mut self, dt: f64, handles: &[u32], frames: &[f32]) -> Result<Vec<f32>, JsValue> {
            self.inner.step(dt as f32, handles, frames).map_err(error)
        }
    }

    impl Default for JiggleWorld {
        fn default() -> Self {
            Self::new()
        }
    }
}
