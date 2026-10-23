//! The `csm` provider of the `Shadow` slot (S11): `wgsl/csm.wgsl`.
//!
//! Standalone (only `raa_shader::` paths) so the variants test can include this file
//! with `#[path]`; the later provider task adds `mod provider;` to `lib.rs`.

use std::borrow::Cow;

use raa_shader::{SlotId, SlotProvider};

/// Provider name of the CSM shadow slot.
pub(crate) const CSM_PROVIDER: &str = "csm";

/// Priority of the CSM shadow provider (above the raa-lit defaults at 0).
pub(crate) const CSM_PRIORITY: i32 = 10;

/// `raa_slot_shadow` of provider `csm` and its private `raa_csm_` helpers.
pub(crate) const CSM_WGSL: &str = include_str!("wgsl/csm.wgsl");

/// The `Shadow` slot provider `csm`, priority 10: cascaded soft sun shadows with a PCSS
/// blocker search and filter, multiplied by the contact-shadow texture.
#[allow(dead_code, reason = "installed by ShadowFeature::install")]
pub(crate) fn csm_provider() -> SlotProvider {
    SlotProvider {
        slot: SlotId::Shadow,
        priority: CSM_PRIORITY,
        provider: CSM_PROVIDER,
        wgsl: Cow::Borrowed(CSM_WGSL),
    }
}
