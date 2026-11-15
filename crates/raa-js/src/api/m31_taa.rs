//! Installs the temporal anti-aliasing resolve (pass `taa`, order 100, between
//! `particles.blend` and `dof`) at render resolution 1:1, disabled by default.
use crate::{engine::InstallCtx, state::EngineState};
use raa_core::{GpuContext, RaaError};
use raa_web::FrameInput;

pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
    let frame = ctx.lit.handle().frame_buffer();
    let taa = raa_taa::TaaFeature::install(ctx.gpu, ctx.pipeline, &frame)?;
    ctx.features.insert(taa);
    Ok(())
}

pub(crate) fn frame(_: &mut EngineState, _: &GpuContext, _: &FrameInput) -> Result<(), RaaError> {
    Ok(())
}
