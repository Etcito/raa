//! WGSL of the `height` Fog slot provider and of the `fog` fullscreen pass.
//!
//! Standalone (no `crate::` paths) so tests/variants.rs can compile it with `#[path]`.

use std::borrow::Cow;

use raa_shader::{PRELUDE_WGSL, SlotId, SlotProvider};

/// The `Fog` slot provider `height`: `raa_slot_fog` plus private `raa_height_*` helpers
/// (height fog optical depth mirroring `integral.rs`, env/sun fog radiance, aerial
/// perspective). `raa_height_apply(color, world_pos, camera, sky)` is also used by the pass.
pub const FOG_SLOT_WGSL: &str = include_str!("wgsl/fog_slot.wgsl");

/// Fullscreen pass body, composed after PRELUDE and [`FOG_SLOT_WGSL`] by
/// [`fog_pass_source`]. Entry points `vs_main` (3-vertex triangle) and `fs_main` (HDR).
/// Group 2 (private): 0 depth (`texture_depth_2d`), 1 hdr.src(), 2 linear clamp sampler.
pub const FOG_PASS_WGSL: &str = include_str!("wgsl/fog_pass.wgsl");

/// Provider name of the fog slot provider.
pub const HEIGHT_PROVIDER: &str = "height";

/// Priority of the `height` provider (above raa-lit's defaults at 0).
pub const HEIGHT_PRIORITY: i32 = 10;

/// The composed fog pass module: PRELUDE_WGSL + [`FOG_SLOT_WGSL`] + [`FOG_PASS_WGSL`].
pub fn fog_pass_source() -> String {
    let mut src =
        String::with_capacity(PRELUDE_WGSL.len() + FOG_SLOT_WGSL.len() + FOG_PASS_WGSL.len() + 2);
    src.push_str(PRELUDE_WGSL);
    src.push('\n');
    src.push_str(FOG_SLOT_WGSL);
    src.push('\n');
    src.push_str(FOG_PASS_WGSL);
    src
}

/// The `Fog` slot provider `height`, priority 10.
pub fn slot_provider() -> SlotProvider {
    SlotProvider {
        slot: SlotId::Fog,
        priority: HEIGHT_PRIORITY,
        provider: HEIGHT_PROVIDER,
        wgsl: Cow::Borrowed(FOG_SLOT_WGSL),
    }
}
