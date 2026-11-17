//! S56 tunables of raa-gtao (rows of docs/render-graph/tunables.md).

use raa_pipeline::TunableSpec;

static TUNABLES: [TunableSpec; 2] = [
    TunableSpec {
        key: "slices",
        default: 4.0,
        min: Some(1.0),
        max: Some(8.0),
        integer: true,
        rebuild: "none",
        source: "crates/raa-gtao/src/desc.rs:62",
    },
    TunableSpec {
        key: "steps",
        default: 8.0,
        min: Some(1.0),
        max: Some(16.0),
        integer: true,
        rebuild: "none",
        source: "crates/raa-gtao/src/desc.rs:63",
    },
];

/// The raa-gtao tunables. Today it returns the raa-gtao rows in table order: `slices`, then
/// `steps` (no WGSL row: denoise.wgsl `TAPS` is classed layout).
///
/// Handoff: card 'Catalog family tunables_lighting' attaches each spec by key (key ->
/// TunableSpec) to a pass.ao.gtao ParamSpec with
/// `ParamBinding::Tunable { feature: FeatureId::Gtao, key }`.
pub fn tunables() -> &'static [TunableSpec] {
    &TUNABLES
}
