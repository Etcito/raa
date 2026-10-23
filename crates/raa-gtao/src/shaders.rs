//! WGSL source of the `ao.gtao` compute pass, its depth-aware denoise pass, the size of
//! their shared uniform and the `gtao` provider of the `Ao` slot.
//!
//! Standalone (no `crate::` paths) so tests/variants.rs can compile it with `#[path]`.

use std::borrow::Cow;

use raa_shader::{SlotId, SlotProvider};

/// Compute module, entry `gtao` (8x8), group 0 only: 0 depth, 1 normal, 2 `GtaoParams`,
/// 3 R32Float storage output (AO, 1 = unoccluded).
pub(crate) const GTAO_WGSL: &str = include_str!("wgsl/gtao.wgsl");

/// Denoise module, entries `denoise_h` and `denoise_v` (8x8), group 0 only: 0 source AO,
/// 1 depth, 2 R32Float storage (horizontal), 3 Rgba16Float storage (vertical, AO in `.r`),
/// 4 `GtaoParams`.
pub(crate) const DENOISE_WGSL: &str = include_str!("wgsl/denoise.wgsl");

/// Byte size of `GtaoParams`: proj @0, inv_proj @64, viewport @128, params @144.
pub(crate) const PARAMS_SIZE: u64 = 160;

/// `raa_slot_ao` of provider `gtao`: `textureLoad` of the S10 group-1 AO texture at the
/// fragment's pixel, value in `.r`.
pub(crate) const SLOT_AO_WGSL: &str = include_str!("wgsl/slot_ao.wgsl");

/// The `Ao` slot provider `gtao`, priority 10 (above the raa-lit defaults at 0).
pub(crate) fn ao_provider() -> SlotProvider {
    SlotProvider {
        slot: SlotId::Ao,
        priority: 10,
        provider: "gtao",
        wgsl: Cow::Borrowed(SLOT_AO_WGSL),
    }
}
