//! S56 tunables of raa-dof (rows of docs/render-graph/tunables.md, in table order).

use raa_pipeline::TunableSpec;

/// The specs of [`tunables`], indexed like `DofState.tunables`.
pub(crate) static TUNABLES: [TunableSpec; 2] = [
    TunableSpec {
        key: "taps",
        default: 48.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "pipeline",
        source: "crates/raa-dof/src/wgsl/dof_gather.wgsl:37",
    },
    TunableSpec {
        key: "radiusPerCoc",
        default: 0.5,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-dof/src/wgsl/dof_gather.wgsl:41",
    },
];

/// WGSL override names of [`TUNABLES`], in the same order.
pub(crate) const OVERRIDES: [&str; 2] = ["TAPS", "RADIUS_PER_COC"];

/// The raa-dof tunable specs. Today it returns the raa-dof rows of
/// docs/render-graph/tunables.md in table order (`taps`, `radiusPerCoc`), both WGSL
/// overrides of the gather module with rebuild `pipeline`.
///
/// Handoff: card 'Catalog family tunables_post' calls `tunables` as its key registry
/// (key -> TunableSpec) and attaches each spec by key to a pass.dof ParamSpec with
/// ParamBinding::Tunable { feature: FeatureId::Dof, key }.
pub fn tunables() -> &'static [TunableSpec] {
    &TUNABLES
}

/// The current values as pipeline constants of the gather module.
pub(crate) fn constants(values: &[f64; 2]) -> [(&'static str, f64); 2] {
    [(OVERRIDES[0], values[0]), (OVERRIDES[1], values[1])]
}
