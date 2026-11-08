//! Priority 30 Direct provider. The previous layered implementation is retained as fallback.
use raa_core::RaaError;
use raa_lit::LitRenderer;
use raa_shader::{SlotId, SlotProvider};
use std::borrow::Cow;
/// SurfaceMaterialGpu.header.z opt-in bit.
pub const FLAG_CEL: u32 = 128;
pub fn provider() -> SlotProvider {
    let fallback = raa_brdf::BRDF_WGSL
        .replace("raa_slot_direct", "raa_cel_fallback")
        .replace("raa_layered_", "raa_cel_layered_")
        .replace("RAA_LAYERED_", "raa_cel_LAYERED_");
    SlotProvider {
        slot: SlotId::Direct,
        priority: 30,
        provider: "cel",
        wgsl: Cow::Owned(format!(
            "{}\n{}",
            fallback,
            include_str!("wgsl/direct.wgsl")
        )),
    }
}
pub fn install(lit: &mut LitRenderer) -> Result<(), RaaError> {
    lit.add_slot(provider())
}
