//! The `squash-jiggle` `Deform` provider (S11, priority 10) and its installation.

use std::borrow::Cow;

use raa_core::RaaError;
use raa_lit::LitRenderer;
use raa_shader::{SlotId, SlotProvider};

/// Name of the provider (`Deform` slot).
const PROVIDER: &str = "squash-jiggle";

/// Priority of the provider, above the raa-lit defaults at 0.
const PRIORITY: i32 = 10;

/// `raa_slot_deform` of provider `squash-jiggle` and its `raa_squash_jiggle_` helpers.
const DEFORM_WGSL: &str = include_str!("wgsl/deform.wgsl");

/// The `Deform` slot provider `squash-jiggle`: squash-and-stretch followed by a
/// height-weighted jiggle offset and the inverse-transpose normal tilt.
pub fn provider() -> SlotProvider {
    SlotProvider {
        slot: SlotId::Deform,
        priority: PRIORITY,
        provider: PROVIDER,
        wgsl: Cow::Borrowed(DEFORM_WGSL),
    }
}

/// Adds the [`provider`] to `lit` (which clears its pipeline cache).
///
/// Refusals (nothing installed): the `add_slot` refusal with its code
/// (`shader.slot_conflict` when installed twice or when another provider holds priority 10
/// on `Deform`).
pub fn install(lit: &mut LitRenderer) -> Result<(), RaaError> {
    lit.add_slot(provider())
}
