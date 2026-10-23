//! The default slot providers (S11): one per [`SlotId`], priority 0, provider `raa.default`,
//! installed by `LitRenderer::new`. Their WGSL lives in `wgsl/slots_default.wgsl`, one
//! `// @slot <name>` section per slot.

use std::borrow::Cow;

use raa_shader::{SlotId, SlotProvider};

/// Provider name of the default slot implementations.
pub(crate) const DEFAULT_PROVIDER: &str = "raa.default";

/// Every default slot section; [`default_slot_providers`] cuts it at the `// @slot` lines.
pub(crate) const SLOTS_DEFAULT_WGSL: &str = include_str!("wgsl/slots_default.wgsl");

/// Line that starts each section of [`SLOTS_DEFAULT_WGSL`].
const SECTION_MARKER: &str = "// @slot ";

/// Name of the section of `slot` in [`SLOTS_DEFAULT_WGSL`].
pub(crate) fn section_name(slot: SlotId) -> &'static str {
    match slot {
        SlotId::Deform => "deform",
        SlotId::Shadow => "shadow",
        SlotId::Ao => "ao",
        SlotId::Ambient => "ambient",
        SlotId::Direct => "direct",
        SlotId::Transmission => "transmission",
        SlotId::Fog => "fog",
    }
}

/// The section `name` of [`SLOTS_DEFAULT_WGSL`]: from its marker line to the next marker
/// or the end of the file; empty when the file has no such section.
fn section(name: &str) -> &'static str {
    let src = SLOTS_DEFAULT_WGSL;
    let mut offset = 0;
    for line in src.split_inclusive('\n') {
        let start = offset;
        offset += line.len();
        if line.trim_end().strip_prefix(SECTION_MARKER) == Some(name) {
            let rest = &src[offset..];
            let len = rest
                .find(&format!("\n{SECTION_MARKER}"))
                .map_or(rest.len(), |i| i + 1);
            return &src[start..offset + len];
        }
    }
    ""
}

/// One provider per [`SlotId`] in [`SlotId::ALL`] order, priority 0, provider
/// [`DEFAULT_PROVIDER`], with the S11 defaults: identity deform, shadow 1, AO 1, the S1
/// PbrLite ambient term, S1 GGX + Lambert direct light, no transmission, no fog.
#[allow(dead_code, reason = "installed by LitRenderer::new")]
pub(crate) fn default_slot_providers() -> Vec<SlotProvider> {
    SlotId::ALL
        .iter()
        .map(|&slot| SlotProvider {
            slot,
            priority: 0,
            provider: DEFAULT_PROVIDER,
            wgsl: Cow::Borrowed(section(section_name(slot))),
        })
        .collect()
}
