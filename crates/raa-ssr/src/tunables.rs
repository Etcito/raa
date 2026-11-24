//! S56 tunables of raa-ssr: the `override` constants of `ssr_trace.wgsl`.

use raa_pipeline::TunableSpec;

/// WGSL override names, in the order of [`TUNABLES`].
pub(crate) const OVERRIDE_NAMES: [&str; 3] = ["MAX_STEPS", "REFINE_STEPS", "EDGE_FADE"];

static TUNABLES: [TunableSpec; 3] = [
    TunableSpec {
        key: "maxSteps",
        default: 64.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "pipeline",
        source: "crates/raa-ssr/src/wgsl/ssr_trace.wgsl:60",
    },
    TunableSpec {
        key: "refineSteps",
        default: 8.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "pipeline",
        source: "crates/raa-ssr/src/wgsl/ssr_trace.wgsl:61",
    },
    TunableSpec {
        key: "edgeFade",
        default: 0.1,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-ssr/src/wgsl/ssr_trace.wgsl:63",
    },
];

/// The raa-ssr tunable specs (docs/render-graph/tunables.md).
///
/// Handoff: card 'Catalog family tunables_post' calls `tunables`; today it returns the
/// raa-ssr rows in table order (maxSteps, refineSteps, edgeFade), maxSteps among them, as
/// the key -> TunableSpec registry that card attaches by key to a pass.ssr ParamSpec with
/// ParamBinding::Tunable { feature: FeatureId::Ssr, key }.
pub fn tunables() -> &'static [TunableSpec] {
    &TUNABLES
}

/// Default values of [`tunables`], in its order.
pub(crate) fn defaults() -> [f64; 3] {
    [
        TUNABLES[0].default,
        TUNABLES[1].default,
        TUNABLES[2].default,
    ]
}
