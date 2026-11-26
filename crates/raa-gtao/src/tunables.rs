//! S56 tunables of raa-gtao (rows of docs/render-graph/tunables.md).

use raa_pipeline::TunableSpec;

static TUNABLES: [TunableSpec; 4] = [
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
    TunableSpec {
        key: "scale",
        default: 1.0,
        min: Some(1.0),
        max: Some(8.0),
        integer: true,
        rebuild: "none",
        source: "crates/raa-gtao/src/feature.rs:77",
    },
    TunableSpec {
        key: "upsample",
        default: 1.0,
        min: Some(0.0),
        max: Some(1.0),
        integer: true,
        rebuild: "none",
        source: "crates/raa-gtao/src/feature.rs:78",
    },
];

/// The raa-gtao tunables. Today it returns the raa-gtao rows in table order: `slices`, then
/// `steps` (no WGSL row: denoise.wgsl `TAPS` is classed layout), then the pass-scale study
/// rows `scale` (pixels per AO texel side, 1 = full resolution) and `upsample` (0 bilinear,
/// 1 depth-aware bilateral) of the reconstruction to full resolution.
///
/// Handoff: card 'Catalog family tunables_lighting' attaches each spec by key (key ->
/// TunableSpec) to a pass.ao.gtao ParamSpec with
/// `ParamBinding::Tunable { feature: FeatureId::Gtao, key }`.
pub fn tunables() -> &'static [TunableSpec] {
    &TUNABLES
}
