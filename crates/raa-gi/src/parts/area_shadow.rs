//! The M6 mid tier of the area-light shadows (S44): stratified shadow rays per pixel and mid-tier
//! slot, a reprojected and spatially denoised shadow weight and visibility pair (temporal and
//! spatial bilateral), and the ratio estimator that writes the visibility mask. This module root
//! holds the CPU twins in `cpu`; the trace, temporal and spatial kernels, their builders, the
//! resources and the part are added by later shares of the module.

#[path = "area_shadow/cpu.rs"]
pub mod cpu;

#[path = "area_shadow/passes.rs"]
pub mod passes;

#[path = "area_shadow/resources.rs"]
mod resources;

use super::PartsCtx;
use raa_core::RaaError;

/// Handoff: card 'raa-gi: AreaShadowPart and the gi.area-shadow feature pass (order 43) publishing
/// the mask' replaces this body. Today `install` adds nothing to `ctx` and returns `Ok(())`; the
/// generated `install_parts` of `parts.rs` calls it once per GI feature install, in file-name order.
pub(crate) fn install(ctx: &mut PartsCtx<'_>) -> Result<(), RaaError> {
    let _ = ctx;
    Ok(())
}
