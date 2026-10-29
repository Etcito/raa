//! m18 skin API (S18): mesh-space, column-major poses of instances whose mesh was
//! uploaded with `joints` and `weights` (`setSkinPose`, `setSkinPoses`).
//!
//! A pose holds 16 floats per joint. One instance's slice is 16 x one past the highest
//! joint its mesh's skin references, so a batch over meshes with different joint counts
//! is the concatenation of those slices in the order of `instances`. Refusals:
//! `api.unknown_handle` (instance), `skin.not_skinned`, `skin.pose_length`,
//! `skin.joint_range` and `skin.cap_joints`.

use js_sys::{Float32Array, Uint32Array};
use raa_core::{GpuContext, RaaError};
use raa_web::FrameInput;
use wasm_bindgen::prelude::*;

use crate::engine::{Engine, InstallCtx};
use crate::state::EngineState;

pub(crate) fn install(_ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
    Ok(())
}

pub(crate) fn frame(
    _st: &mut EngineState,
    _gpu: &GpuContext,
    _input: &FrameInput,
) -> Result<(), RaaError> {
    Ok(())
}

#[wasm_bindgen(js_class = Engine)]
impl Engine {
    /// Sets the S16 skin matrices of `instance` in mesh space: 16 column-major floats per
    /// joint, exactly the joints its mesh's skin references. An instance without a pose
    /// renders at rest. A refusal changes nothing.
    #[wasm_bindgen(js_name = setSkinPose)]
    pub fn set_skin_pose(&mut self, instance: u32, matrices: Float32Array) -> Result<(), JsValue> {
        self.with_state(|gpu, st| st.set_skin_pose(gpu, instance, &matrices.to_vec()))
    }

    /// Sets many poses at once. `matrices` concatenates one slice per `instances` entry,
    /// in order, each 16 x that instance's mesh skin joints; mixed joint counts are
    /// allowed. The whole batch is validated before any pose is updated.
    #[wasm_bindgen(js_name = setSkinPoses)]
    pub fn set_skin_poses(
        &mut self,
        instances: Uint32Array,
        matrices: Float32Array,
    ) -> Result<(), JsValue> {
        self.with_state(|gpu, st| st.set_skin_poses(gpu, &instances.to_vec(), &matrices.to_vec()))
    }
}
