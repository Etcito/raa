//! Feature passes (S12): the trait every pass of the lit pipeline implements, what its
//! prepare and record receive, and the `PassInfo` rows `Pipeline::passes` lists.

use raa_core::{GpuContext, RaaError};

use crate::{LightingResources, Targets};

/// One feature pass of the lit pipeline (sky, shadows, prepass, forward, bloom, tonemap...).
/// The pipeline runs the enabled passes in order inside its one `raa.pipeline` graph pass:
/// every prepare first, then every record into the frame's encoder.
pub trait FeaturePass {
    /// Unique name in its pipeline (the S12 table name, e.g. `forward`).
    fn name(&self) -> &str;

    /// Default position in the execution order (S12 table); lower runs first.
    fn order(&self) -> u32;

    /// True when the pass cannot be disabled (`pipeline.required_pass`). Default: false.
    fn required(&self) -> bool {
        false
    }

    /// True for a display-space overlay that paints the backbuffer after tone mapping.
    /// Such passes may follow tonemap; they must not modify the HDR chain.
    fn output_overlay(&self) -> bool {
        false
    }

    /// Called before the pass's first prepare and after the targets were recreated for a new
    /// size. Default: nothing.
    fn resize(&mut self, _gpu: &GpuContext, _targets: &Targets) {}

    /// Called by `Pipeline::set_enabled`. Disabling resets the pass's group-1 resources to the
    /// dummies (`clear_*`) so the frame stays valid without it (S12). Default: nothing.
    fn set_enabled(&mut self, _enabled: bool, _lighting: &mut LightingResources) {}

    /// CPU work and uploads of the frame, run for every enabled pass before any records.
    /// Default: nothing.
    fn prepare(&mut self, _ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        Ok(())
    }

    /// Records the pass's commands into `ctx.encoder`.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError>;
}

/// What [`FeaturePass::prepare`] receives.
pub struct PrepareCtx<'a> {
    /// Device and queue.
    pub gpu: &'a GpuContext,
    /// Frame size in pixels.
    pub size: (u32, u32),
    /// Frame index, 0 for the first executed frame.
    pub frame: u64,
    /// Time of the frame in seconds (`Pipeline::set_time`).
    pub time_s: f32,
    /// Seconds since the previous `set_time` (0 on the first).
    pub dt_s: f32,
    /// The surface-sized targets of this frame.
    pub targets: &'a Targets,
    /// Group-1 lighting resources; changes made here are bound in the same frame.
    pub lighting: &'a mut LightingResources,
}

/// What [`FeaturePass::record`] receives: the frame's shared encoder and the backbuffer.
pub struct RecordCtx<'a> {
    /// Device and queue.
    pub gpu: &'a GpuContext,
    /// The one encoder every pass of the frame records into.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// Frame size in pixels.
    pub size: (u32, u32),
    /// Frame index, 0 for the first executed frame.
    pub frame: u64,
    /// Time of the frame in seconds (`Pipeline::set_time`).
    pub time_s: f32,
    /// The surface-sized targets of this frame.
    pub targets: &'a Targets,
    /// Group-1 lighting resources, uploaded for this frame.
    pub lighting: &'a LightingResources,
    /// View of the frame's backbuffer (the canvas texture or an offscreen target).
    pub backbuffer: &'a wgpu::TextureView,
    /// Format of `backbuffer`.
    pub backbuffer_format: wgpu::TextureFormat,
    /// True when the surface format is *Srgb (the hardware encodes; tonemap must not).
    pub surface_is_srgb: bool,
}

/// What an entry of the pipeline is: a pass it runs, or a contributor drawn by its host
/// (`forward`) inside the host's render pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassRole {
    /// A feature pass the pipeline prepares and records.
    Pass,
    /// A draw inside the forward pass; the pipeline only lists and toggles it.
    Contributor,
}

/// One row of `Pipeline::passes`, in execution order.
#[derive(Clone, Debug, PartialEq)]
pub struct PassInfo {
    /// Entry name.
    pub name: String,
    /// Pass or contributor.
    pub role: PassRole,
    /// Whether the entry runs this frame.
    pub enabled: bool,
    /// The entry's declared order (a contributor's is its position inside its host).
    pub order: u32,
    /// True when the entry cannot be disabled.
    pub required: bool,
}
