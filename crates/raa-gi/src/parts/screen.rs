//! The milestone-2 part `screen` of the GI volume (S39 screen probes): the module root of the
//! part, registered by the generated registry in `parts.rs`. Its CPU twins live in `screen/cpu.rs`.
//! Adapted from solstice@7e465b5 `crates/solstice-render/src/gi/screen.rs` and
//! `crates/solstice-render/src/gi/depth_pyramid.rs`. MIT licence, Copyright (c) 2026 SamG-Coder.

use super::PartsCtx;
use raa_core::RaaError;

#[path = "screen/cpu.rs"]
pub mod cpu;

#[path = "screen/passes.rs"]
pub mod passes;

/// Handoff: `install` is the body that card 'raa-gi screen: ScreenPart, gi.screen pass and install'
/// replaces. Today `install` adds nothing to `ctx` and returns `Ok(())`; the generated
/// `install_parts` of `parts.rs` calls it once per GI feature install, in file-name order.
pub(crate) fn install(ctx: &mut PartsCtx<'_>) -> Result<(), RaaError> {
    let _ = ctx;
    Ok(())
}
