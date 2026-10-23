//! Extension point of the forward pass: code drawing into the forward HDR target (MSAA,
//! after the renderer's opaque draws) with the renderer's meshes, instance stream, group-0
//! bind group and pipeline cache.

use raa_core::RaaError;
use raa_pipeline::PrepareCtx;

use crate::handle::LitHandle;

/// Draws contributed to the forward pass. Contributors run in ascending [`order`]
/// (then registration order); each one prepares once per frame, then records into the
/// forward render pass, whose group 0 and 1 the renderer has bound.
///
/// [`order`]: ForwardContributor::order
pub trait ForwardContributor {
    /// Name in diagnostics and refusals.
    fn name(&self) -> &str;

    /// Recording order among the contributors (lower first).
    fn order(&self) -> u32;

    /// Per-frame CPU and upload work before any pass records (create pipelines with
    /// [`LitHandle::pipeline_for`], write buffers through `ctx.gpu`). Nothing by default.
    fn prepare(&mut self, _lit: &LitHandle, _ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        Ok(())
    }

    /// Records the contributor's draws into the forward pass (bind pipelines, group 2,
    /// meshes with [`LitHandle::bind_mesh`], then draw).
    fn record(&self, pass: &mut wgpu::RenderPass<'_>, lit: &LitHandle) -> Result<(), RaaError>;
}
