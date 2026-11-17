//! S56 tunables of raa-shafts (docs/render-graph/tunables.md rows of the crate).

use raa_pipeline::TunableSpec;

const TUNABLES: &[TunableSpec] = &[
    TunableSpec {
        key: "steps",
        default: 32.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "pipeline",
        source: "crates/raa-shafts/src/wgsl/shafts.wgsl:66",
    },
    TunableSpec {
        key: "hgG",
        default: 0.7,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-shafts/src/wgsl/shafts.wgsl:67",
    },
];

/// Handoff: card 'Catalog family tunables_post' calls `tunables`; today it returns the
/// raa-shafts rows in table order (`steps`, `hgG`, both WGSL overrides, rebuild pipeline).
/// That card attaches each spec by key to a pass.sunshafts ParamSpec with
/// ParamBinding::Tunable { feature: FeatureId::Shafts, key }.
pub fn tunables() -> &'static [TunableSpec] {
    TUNABLES
}
