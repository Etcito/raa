//! The TAA resolve's tunables (contract `tunable-promotion`): the `override`s of `taa.wgsl`.

use raa_pipeline::TunableSpec;

const fn spec(key: &'static str, default: f64, line: &'static str) -> TunableSpec {
    TunableSpec {
        key,
        default,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: line,
    }
}

/// The number of TAA tunables.
pub(crate) const N: usize = 12;

static TUNABLES: [TunableSpec; N] = [
    spec(
        "filterSharpness",
        8.0,
        "crates/raa-taa/src/wgsl/taa.wgsl:30",
    ),
    spec("clipTolerance", 1.25, "crates/raa-taa/src/wgsl/taa.wgsl:31"),
    spec("blendBase", 0.04, "crates/raa-taa/src/wgsl/taa.wgsl:32"),
    spec(
        "blendMotionGain",
        0.03,
        "crates/raa-taa/src/wgsl/taa.wgsl:33",
    ),
    spec("blendMin", 0.04, "crates/raa-taa/src/wgsl/taa.wgsl:34"),
    spec("blendMax", 0.5, "crates/raa-taa/src/wgsl/taa.wgsl:35"),
    spec(
        "historySharpness",
        0.6,
        "crates/raa-taa/src/wgsl/taa.wgsl:36",
    ),
    spec("blendStatic", 0.04, "crates/raa-taa/src/wgsl/taa.wgsl:37"),
    spec("clipStatic", 1.0, "crates/raa-taa/src/wgsl/taa.wgsl:38"),
    spec(
        "upsampleConfidence",
        0.0,
        "crates/raa-taa/src/wgsl/taa.wgsl:39",
    ),
    spec(
        "upsampleClipScene",
        0.0,
        "crates/raa-taa/src/wgsl/taa.wgsl:40",
    ),
    spec("upsampleKernel", 0.0, "crates/raa-taa/src/wgsl/taa.wgsl:41"),
];

/// The tunables of the `taa` pass.
///
/// Handoff: card 'Catalog family tunables_post' calls `tunables`; today it returns the twelve
/// pass.taa specs in pinned order (filterSharpness, clipTolerance, blendBase,
/// blendMotionGain, blendMin, blendMax, historySharpness, blendStatic, clipStatic,
/// upsampleConfidence, upsampleClipScene, upsampleKernel). The key registry it passes is key -> TunableSpec
/// (default, min, max, integer, rebuild, source); that card attaches each spec by key to a
/// pass.taa ParamSpec with `ParamBinding::Tunable { feature: FeatureId::Taa, key }`.
pub fn tunables() -> &'static [TunableSpec] {
    &TUNABLES
}

/// Defaults in table order.
pub(crate) fn defaults() -> [f64; N] {
    std::array::from_fn(|i| TUNABLES[i].default)
}

/// Pairs each override name with its value, in table order.
pub(crate) fn taa_constants(values: &[f64; N]) -> [(&'static str, f64); N] {
    [
        ("RAA_TAA_FILTER_SHARPNESS", values[0]),
        ("RAA_TAA_CLIP_TOLERANCE", values[1]),
        ("RAA_TAA_BLEND_BASE", values[2]),
        ("RAA_TAA_BLEND_MOTION_GAIN", values[3]),
        ("RAA_TAA_BLEND_MIN", values[4]),
        ("RAA_TAA_BLEND_MAX", values[5]),
        ("RAA_TAA_HISTORY_SHARPNESS", values[6]),
        ("RAA_TAA_BLEND_STATIC", values[7]),
        ("RAA_TAA_CLIP_STATIC", values[8]),
        ("RAA_TAA_UPSAMPLE_CONFIDENCE", values[9]),
        ("RAA_TAA_UPSAMPLE_CLIP_SCENE", values[10]),
        ("RAA_TAA_UPSAMPLE_KERNEL", values[11]),
    ]
}
