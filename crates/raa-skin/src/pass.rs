//! The `skin` compute FeaturePass (S12): order 15, feature bit 20. Rest inputs are
//! shared by every instance of a mesh; the pose buffer and lit skinned view (private
//! stream 0, tangents, edge segments) are per instance. Vertices are dispatched before
//! edges; a missing pose or a disabled pass draws rest resources.

use std::collections::BTreeMap;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError, wgpu};
use raa_lit::{RestMeshView, SkinViews};
use raa_pipeline::{FeaturePass, LightingResources, RecordCtx};

use crate::pose::PoseBuffer;
use crate::{SkinError, SkinVertexInputs, SkinVertexKernel};

/// S12 name of the pass.
pub const SKIN_PASS: &str = "skin";
/// S12 order: before `prepass` (30) and every lit draw.
pub const SKIN_ORDER: u32 = 15;
/// Feature-bit index: the next row after S12's 0..19.
pub const SKIN_FEATURE_BIT: u32 = 20;

type Instance = (Rc<SkinVertexInputs>, RestMeshView, Option<PoseBuffer>);

/// The skin pass and its per-instance resources.
pub struct SkinPass {
    kernel: SkinVertexKernel,
    instances: BTreeMap<u32, Instance>,
    views: SkinViews,
    enabled: bool,
}

impl SkinPass {
    pub fn new(gpu: &GpuContext) -> Self {
        SkinPass {
            kernel: SkinVertexKernel::new(gpu),
            instances: BTreeMap::new(),
            views: SkinViews::default(),
            enabled: true,
        }
    }

    /// Binds `instance` to uploaded rest `inputs` (shared by its mesh's instances) and its
    /// lit rest view; vertex and edge counts must match. It draws rest until posed.
    pub fn add_instance(
        &mut self,
        instance: u32,
        inputs: Rc<SkinVertexInputs>,
        rest: RestMeshView,
    ) -> Result<(), RaaError> {
        self.views.remove_skinned_view(instance);
        if rest.vertices != inputs.vertices || rest.edge_count != inputs.edges {
            return Err(refusal(
                "render.invalid_mesh",
                format!(
                    "instance {instance} rest view is {} vertices/{} edges; the skin has {}/{}",
                    rest.vertices, rest.edge_count, inputs.vertices, inputs.edges
                ),
            ));
        }
        self.instances.insert(instance, (inputs, rest, None));
        Ok(())
    }

    /// Uploads the S16 skin matrices of `instance` (16 column-major floats per joint) and
    /// draws its posed view. Refusals: `skin.pose_length`, `skin.cap_joints`,
    /// `api.unknown_handle`.
    pub fn set_pose(
        &mut self,
        gpu: &GpuContext,
        instance: u32,
        matrices: &[f32],
    ) -> Result<(), RaaError> {
        let Some((_, rest, pose)) = self.instances.get_mut(&instance) else {
            return Err(refusal(
                "api.unknown_handle",
                format!("instance {instance} is not registered"),
            ));
        };
        let first = pose.is_none();
        match pose {
            Some(pose) => pose.write(gpu, instance, matrices).map_err(skin)?,
            None => *pose = Some(PoseBuffer::new(gpu, instance, matrices).map_err(skin)?),
        }
        if first {
            self.views.add_skinned_view(gpu, instance, rest.clone())?;
            self.views.set_skinning_enabled(self.enabled);
        }
        Ok(())
    }

    /// Drops the pose and posed outputs of `instance`; it draws rest again.
    pub fn clear_pose(&mut self, instance: u32) {
        if let Some((_, _, pose)) = self.instances.get_mut(&instance) {
            *pose = None;
        }
        self.views.remove_skinned_view(instance);
    }

    /// Drops every resource of `instance`: pose, skinned outputs and registration.
    pub fn remove_instance(&mut self, instance: u32) {
        self.instances.remove(&instance);
        self.views.remove_skinned_view(instance);
    }

    /// The lit views the draw passes bind.
    pub fn views(&self) -> &SkinViews {
        &self.views
    }

    /// Dispatches every posed instance: vertices first, then that instance's edges.
    pub fn encode(&self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder) {
        if !self.enabled {
            return;
        }
        for (&instance, (inputs, _, pose)) in &self.instances {
            let Some((pose, view)) = pose.as_ref().zip(self.views.skinned_view(instance)) else {
                continue;
            };
            self.kernel.dispatch(
                gpu,
                encoder,
                inputs,
                pose.buffer(),
                &view.stream0,
                &view.tangents,
                view.edges.as_ref(),
            );
        }
    }
}

impl FeaturePass for SkinPass {
    fn name(&self) -> &str {
        SKIN_PASS
    }

    fn order(&self) -> u32 {
        SKIN_ORDER
    }

    /// Disabling selects rest streams; re-enabling keeps the views and the next record
    /// refreshes every posed output, so nothing stale is drawn.
    fn set_enabled(&mut self, enabled: bool, _lighting: &mut LightingResources) {
        self.enabled = enabled;
        self.views.set_skinning_enabled(enabled);
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        self.encode(ctx.gpu, ctx.encoder);
        Ok(())
    }
}

fn refusal(code: &'static str, message: String) -> RaaError {
    RaaError::FrameGraph { code, message }
}

fn skin(e: SkinError) -> RaaError {
    refusal(e.code, e.message)
}
