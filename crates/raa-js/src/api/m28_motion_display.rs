//! Installs the optional motion diagnostic into the existing pass registry.
use crate::{engine::InstallCtx, state::EngineState};
use raa_core::{GpuContext, RaaError};
use raa_web::FrameInput;

pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
    raa_lit::install_motion_display(ctx.gpu, ctx.pipeline)
}

pub(crate) fn frame(_: &mut EngineState, _: &GpuContext, _: &FrameInput) -> Result<(), RaaError> {
    Ok(())
}
