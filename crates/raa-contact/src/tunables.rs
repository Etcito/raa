//! S56 tunables of raa-contact (docs/render-graph/tunables.md, raa-contact rows).

use raa_pipeline::TunableSpec;

/// The raa-contact rows of the triage table, in table order.
pub(crate) const TUNABLES: &[TunableSpec] = &[TunableSpec {
    key: "steps",
    default: 24.0,
    min: None,
    max: None,
    integer: true,
    rebuild: "pipeline",
    source: "crates/raa-contact/src/wgsl/contact.wgsl:23",
}];

/// Handoff: card 'Catalog family tunables_lighting' calls `tunables` and attaches each spec by
/// key to a `pass.shadow.contact` ParamSpec with
/// `ParamBinding::Tunable { feature: FeatureId::Contact, key }`. Today it returns the
/// raa-contact rows in table order (`steps`: override STEPS, default 24, rebuild pipeline).
pub fn tunables() -> &'static [TunableSpec] {
    TUNABLES
}
