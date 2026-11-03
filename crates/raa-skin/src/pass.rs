//! The `skin` FeaturePass (S12): order 15, feature bit 20.
//!
//! Instanced skinning: a pose is written into the shared joint palette of
//! [`SkinViews`](raa_lit::SkinViews) (one block per instance, no per-instance buffer); the
//! lit vertex stage skins every posed instance from it, so posed instances of one mesh and
//! material are ONE instanced draw per pass. Each frame this pass uploads the palette with
//! one `write_buffer` and dispatches compute ONLY for editor overlay geometry (posed
//! stream 0 and edge segments) of instances whose outline or wireframe is on. A missing
//! pose or a disabled pass draws rest resources.

use std::collections::BTreeMap;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError, wgpu};
use raa_lit::{PALETTE_MATRIX_BYTES, RestMeshView, SkinViews};
use raa_pipeline::{FeaturePass, LightingResources, RecordCtx};

use crate::{SkinError, SkinVertexInputs, SkinVertexKernel, validate_pose};

/// S12 name of the pass.
pub const SKIN_PASS: &str = "skin";
/// S12 order: before `prepass` (30) and every lit draw.
pub const SKIN_ORDER: u32 = 15;
/// Feature-bit index: the next row after S12's 0..19.
pub const SKIN_FEATURE_BIT: u32 = 20;

/// What one [`SkinPass::encode`] issued (structural counts, no timing).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct SkinFrameStats {
    /// Posed instances (palette blocks in use).
    pub posed: u32,
    /// Palette uploads: 0 or 1 `write_buffer` per frame.
    pub palette_writes: u32,
    /// Compute dispatches (editor overlays only: vertices + edges per overlaid instance).
    pub dispatches: u32,
    /// Live overlay views after this frame.
    pub overlays: u32,
}

/// The skin pass and its per-instance registrations.
pub struct SkinPass {
    kernel: SkinVertexKernel,
    instances: BTreeMap<u32, (Rc<SkinVertexInputs>, RestMeshView)>,
    views: SkinViews,
    enabled: bool,
    last: SkinFrameStats,
}

impl SkinPass {
    pub fn new(gpu: &GpuContext) -> Self {
        SkinPass {
            kernel: SkinVertexKernel::new(gpu),
            instances: BTreeMap::new(),
            views: SkinViews::default(),
            enabled: true,
            last: SkinFrameStats::default(),
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
        self.views.clear_pose(instance);
        if rest.vertices != inputs.vertices || rest.edge_count != inputs.edges {
            return Err(refusal(
                "render.invalid_mesh",
                format!(
                    "instance {instance} rest view is {} vertices/{} edges; the skin has {}/{}",
                    rest.vertices, rest.edge_count, inputs.vertices, inputs.edges
                ),
            ));
        }
        self.instances.insert(instance, (inputs, rest));
        Ok(())
    }

    /// Writes the S16 skin matrices of `instance` (16 column-major floats per joint) into
    /// its palette block; it draws posed from the next frame. Refusals: `skin.pose_length`,
    /// `skin.cap_joints`, `api.unknown_handle`.
    pub fn set_pose(
        &mut self,
        gpu: &GpuContext,
        instance: u32,
        matrices: &[f32],
    ) -> Result<(), RaaError> {
        let Some((inputs, _)) = self.instances.get(&instance) else {
            return Err(refusal(
                "api.unknown_handle",
                format!("instance {instance} is not registered"),
            ));
        };
        validate_pose(matrices).map_err(skin)?;
        gpu.ensure_live()?;
        let skin = inputs.buffer.clone();
        self.views.set_pose(gpu, instance, &skin, matrices);
        Ok(())
    }

    /// Drops the pose and overlay outputs of `instance`; it draws rest again.
    pub fn clear_pose(&mut self, instance: u32) {
        self.views.clear_pose(instance);
        self.views.remove_skinned_view(instance);
    }

    /// Drops every resource of `instance`: pose, overlay outputs and registration.
    pub fn remove_instance(&mut self, instance: u32) {
        self.instances.remove(&instance);
        self.clear_pose(instance);
    }

    /// The lit views the draw passes bind.
    pub fn views(&self) -> &SkinViews {
        &self.views
    }

    /// Counts of the last [`encode`](Self::encode).
    pub fn last_stats(&self) -> SkinFrameStats {
        self.last
    }

    /// Uploads the palette (one write when a pose changed), then refreshes the overlay
    /// geometry of posed instances the editor overlays: vertices first, then that
    /// instance's edges. Overlay views of instances no longer overlaid are dropped.
    pub fn encode(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
    ) -> SkinFrameStats {
        let mut stats = SkinFrameStats {
            posed: self.views.posed_tags().len() as u32,
            ..SkinFrameStats::default()
        };
        if !self.enabled {
            stats.overlays = self.views.overlay_views() as u32;
            self.last = stats;
            return stats;
        }
        stats.palette_writes = self.views.flush_palette(gpu);
        let Some(palette) = self.views.palette_buffer() else {
            stats.overlays = self.views.overlay_views() as u32;
            self.last = stats;
            return stats;
        };
        for (&instance, (inputs, rest)) in &self.instances {
            let wanted = self.views.has_pose(instance) && self.views.overlay_wanted(instance);
            if !wanted {
                if self.views.skinned_view(instance).is_some() {
                    self.views.remove_skinned_view(instance);
                }
                continue;
            }
            if self.views.skinned_view(instance).is_none()
                && self
                    .views
                    .add_skinned_view(gpu, instance, rest.clone())
                    .is_err()
            {
                continue;
            }
            let (Some((base, capacity)), Some(view)) = (
                self.views.palette_block(instance),
                self.views.skinned_view(instance),
            ) else {
                continue;
            };
            let pose = wgpu::BufferBinding {
                buffer: &palette,
                offset: u64::from(base) * PALETTE_MATRIX_BYTES,
                size: wgpu::BufferSize::new(u64::from(capacity) * PALETTE_MATRIX_BYTES),
            };
            stats.dispatches += self.kernel.dispatch_binding(
                gpu,
                encoder,
                inputs,
                pose,
                &view.stream0,
                &view.tangents,
                view.edges.as_ref(),
            );
        }
        stats.overlays = self.views.overlay_views() as u32;
        self.last = stats;
        stats
    }
}

impl FeaturePass for SkinPass {
    fn name(&self) -> &str {
        SKIN_PASS
    }

    fn order(&self) -> u32 {
        SKIN_ORDER
    }

    /// Disabling selects rest streams; re-enabling keeps the poses and the next record
    /// refreshes every overlay output, so nothing stale is drawn.
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
