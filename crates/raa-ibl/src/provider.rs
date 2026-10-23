//! The `ibl` provider of the `Ambient` slot (S11): IBL_LIB_WGSL followed by
//! `wgsl/ambient.wgsl`.

use std::borrow::Cow;

use raa_shader::{SlotId, SlotProvider};

/// Provider name of the IBL ambient slot.
pub(crate) const IBL_PROVIDER: &str = "ibl";

/// Priority of the IBL ambient provider (above the raa-lit defaults at 0).
pub(crate) const IBL_PRIORITY: i32 = 10;

/// The IBL contract functions (`raa_ibl_irradiance`, `raa_ibl_specular`, `raa_ibl_dfg`,
/// `raa_specular_occlusion`).
pub(crate) const IBL_LIB_SOURCE: &str = include_str!("wgsl/ibl_lib.wgsl");

/// `raa_slot_ambient` of provider `ibl` and its private helpers.
pub(crate) const AMBIENT_WGSL: &str = include_str!("wgsl/ambient.wgsl");

/// The `Ambient` slot provider `ibl`, priority 10: the IBL library followed by the ambient
/// slot function.
#[allow(dead_code, reason = "installed by IblFeature::install")]
pub(crate) fn ambient_provider() -> SlotProvider {
    let mut wgsl = String::with_capacity(IBL_LIB_SOURCE.len() + 1 + AMBIENT_WGSL.len());
    wgsl.push_str(IBL_LIB_SOURCE);
    wgsl.push('\n');
    wgsl.push_str(AMBIENT_WGSL);
    SlotProvider {
        slot: SlotId::Ambient,
        priority: IBL_PRIORITY,
        provider: IBL_PROVIDER,
        wgsl: Cow::Owned(wgsl),
    }
}
