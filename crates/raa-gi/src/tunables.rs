//! The GI tunable store (S56, contract 'gi-tunables-store'): the promoted values of the gi.* passes
//! with their keys, defaults and ranges ([`GI_TUNABLES`]), and [`GiTunables`], the current values
//! with the generation counter that a pass compares to decide when to rebuild its pipelines.

use raa_core::RaaError;
use raa_pipeline::{TunableSpec, validate_tunable};

use crate::error::invalid;

/// Every promoted GI value, in table order (`docs/render-graph/tunables.md`, crate raa-gi). The
/// three rows are the `pass.gi.volume` values: `behind` is `RAA_GI_BEHIND` of volume_write.wgsl,
/// `resetRays` and `directChange` are `RAA_GI_RESET_RAYS` and `RAA_GI_DIRECT_CHANGE` of
/// voxel_light.wgsl. Each is a WGSL `override` passed as a pipeline constant (rebuild
/// `pipeline`). The eighteen `pass.gi.screen` rows follow: the accumulation and occlusion values of
/// screen/accumulate.wgsl, the a-trous weights of screen/atrous.wgsl, the probe trace values of
/// screen/probe_trace.wgsl and `screenEdgeFade` of screen/screen_trace.wgsl, each an `override`.
/// The thirteen ReSTIR GI rows of `pass.gi.screen` follow: nine `override`s of
/// screen/restir_gi_common.wgsl and screen/restir_gi_main.wgsl, then the four params-uniform values
/// of parts/screen/restir_gi_cpu.rs (rebuild `none`), written each prepare.
/// The five `pass.gi.area-shadow` override rows follow: `restirDepthTolerance` of
/// area_shadow/restir_core.wgsl, `spatialStride` and `normalPower` of area_shadow/spatial.wgsl,
/// `temporalAlpha` and `depthTolerance` of area_shadow/temporal.wgsl.
/// The table states no range for any of them, so none is given.
pub const GI_TUNABLES: &[TunableSpec] = &[
    TunableSpec {
        key: "behind",
        default: 0.5,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/volume_write.wgsl:59",
    },
    TunableSpec {
        key: "resetRays",
        default: 8.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/voxel_light.wgsl:57",
    },
    TunableSpec {
        key: "directChange",
        default: 0.05,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/voxel_light.wgsl:65",
    },
    TunableSpec {
        key: "accDepthTolerance",
        default: 0.04,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/accumulate.wgsl:44",
    },
    TunableSpec {
        key: "accNormalCos",
        default: 0.9,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/accumulate.wgsl:46",
    },
    TunableSpec {
        key: "accMinWeight",
        default: 0.05,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/accumulate.wgsl:48",
    },
    TunableSpec {
        key: "accChangeSlack",
        default: 0.05,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/accumulate.wgsl:51",
    },
    TunableSpec {
        key: "accChangeMin",
        default: 4.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/accumulate.wgsl:53",
    },
    TunableSpec {
        key: "occlusionDirections",
        default: 3.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/accumulate.wgsl:57",
    },
    TunableSpec {
        key: "occlusionSteps",
        default: 4.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/accumulate.wgsl:58",
    },
    TunableSpec {
        key: "occlusionRadius",
        default: 0.9,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/accumulate.wgsl:59",
    },
    TunableSpec {
        key: "atrousSigmaLuminance",
        default: 4.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/atrous.wgsl:31",
    },
    TunableSpec {
        key: "atrousNormalPower",
        default: 32.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/atrous.wgsl:33",
    },
    TunableSpec {
        key: "atrousPlane",
        default: 0.01,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/atrous.wgsl:35",
    },
    TunableSpec {
        key: "probeScreenDistance",
        default: 8.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/probe_trace.wgsl:35",
    },
    TunableSpec {
        key: "probeScreenSteps",
        default: 32.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/probe_trace.wgsl:36",
    },
    TunableSpec {
        key: "probeScreenThickness",
        default: 0.15,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/probe_trace.wgsl:37",
    },
    TunableSpec {
        key: "probeVolumeDistance",
        default: 320.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/probe_trace.wgsl:38",
    },
    TunableSpec {
        key: "probeVolumeSteps",
        default: 40.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/probe_trace.wgsl:39",
    },
    TunableSpec {
        key: "probeVolumeHitVoxels",
        default: 0.85,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/probe_trace.wgsl:40",
    },
    TunableSpec {
        key: "screenEdgeFade",
        default: 0.08,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/screen_trace.wgsl:34",
    },
    TunableSpec {
        key: "restirJacobianMax",
        default: 10.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/restir_gi_common.wgsl:63",
    },
    TunableSpec {
        key: "restirVisibilitySteps",
        default: 32.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/restir_gi_common.wgsl:223",
    },
    TunableSpec {
        key: "restirVisibilityDistance",
        default: 320.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/restir_gi_common.wgsl:224",
    },
    TunableSpec {
        key: "restirScreenDistance",
        default: 8.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/restir_gi_main.wgsl:14",
    },
    TunableSpec {
        key: "restirScreenSteps",
        default: 32.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/restir_gi_main.wgsl:15",
    },
    TunableSpec {
        key: "restirScreenThickness",
        default: 0.15,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/restir_gi_main.wgsl:16",
    },
    TunableSpec {
        key: "restirVolumeDistance",
        default: 320.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/restir_gi_main.wgsl:17",
    },
    TunableSpec {
        key: "restirVolumeSteps",
        default: 40.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/restir_gi_main.wgsl:18",
    },
    TunableSpec {
        key: "restirVolumeHitVoxels",
        default: 0.85,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/screen/restir_gi_main.wgsl:19",
    },
    TunableSpec {
        key: "restirSpatialSamples",
        default: 4.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "none",
        source: "crates/raa-gi/src/parts/screen/restir_gi_cpu.rs:13",
    },
    TunableSpec {
        key: "restirSpatialRadius",
        default: 24.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "none",
        source: "crates/raa-gi/src/parts/screen/restir_gi_cpu.rs:15",
    },
    TunableSpec {
        key: "restirBoilingFactor",
        default: 20.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "none",
        source: "crates/raa-gi/src/parts/screen/restir_gi_cpu.rs:19",
    },
    TunableSpec {
        key: "restirNormalCos",
        default: 0.9,
        min: None,
        max: None,
        integer: false,
        rebuild: "none",
        source: "crates/raa-gi/src/parts/screen/restir_gi_cpu.rs:21",
    },
    TunableSpec {
        key: "restirDepthTolerance",
        default: 0.1,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/area_shadow/restir_core.wgsl:12",
    },
    TunableSpec {
        key: "spatialStride",
        default: 2.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/area_shadow/spatial.wgsl:44",
    },
    TunableSpec {
        key: "normalPower",
        default: 8.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/area_shadow/spatial.wgsl:47",
    },
    TunableSpec {
        key: "temporalAlpha",
        default: 0.1,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/area_shadow/temporal.wgsl:41",
    },
    TunableSpec {
        key: "depthTolerance",
        default: 0.05,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/area_shadow/temporal.wgsl:44",
    },
    TunableSpec {
        key: "areaRestirCandidates",
        default: 8.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "none",
        source: "crates/raa-gi/src/parts/area_shadow/restir_cpu.rs:21",
    },
    TunableSpec {
        key: "areaRestirSpatialSamples",
        default: 4.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "none",
        source: "crates/raa-gi/src/parts/area_shadow/restir_cpu.rs:24",
    },
    TunableSpec {
        key: "areaRestirSpatialRadius",
        default: 30.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "none",
        source: "crates/raa-gi/src/parts/area_shadow/restir_cpu.rs:27",
    },
    TunableSpec {
        key: "areaRestirMCap",
        default: 20.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "none",
        source: "crates/raa-gi/src/parts/area_shadow/restir_cpu.rs:30",
    },
    TunableSpec {
        key: "areaRestirBoilingFactor",
        default: 8.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "none",
        source: "crates/raa-gi/src/parts/area_shadow/restir_cpu.rs:33",
    },
    TunableSpec {
        key: "areaRestirNormalCos",
        default: 0.9,
        min: None,
        max: None,
        integer: false,
        rebuild: "none",
        source: "crates/raa-gi/src/parts/area_shadow/restir_cpu.rs:36",
    },
    TunableSpec {
        key: "changeSigma",
        default: 3.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-gi/src/wgsl/ssgi/resolve.wgsl:35",
    },
    TunableSpec {
        key: "slicesLow",
        default: 2.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "none",
        source: "crates/raa-gi/src/settings.rs:157",
    },
    TunableSpec {
        key: "slicesMedium",
        default: 4.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "none",
        source: "crates/raa-gi/src/settings.rs:158",
    },
    TunableSpec {
        key: "slicesHigh",
        default: 6.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "none",
        source: "crates/raa-gi/src/settings.rs:159",
    },
    TunableSpec {
        key: "stepsLow",
        default: 6.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "none",
        source: "crates/raa-gi/src/settings.rs:166",
    },
    TunableSpec {
        key: "stepsMedium",
        default: 8.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "none",
        source: "crates/raa-gi/src/settings.rs:167",
    },
    TunableSpec {
        key: "stepsHigh",
        default: 12.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "none",
        source: "crates/raa-gi/src/settings.rs:168",
    },
];

/// The specs of [`GI_TUNABLES`], as the crate root re-exports them.
///
/// Handoff: `tunables` is called by card 'Catalog family tunables_gi'. Today it returns
/// `GI_TUNABLES` with every raa-gi row in table order (gi.volume, gi.screen, ReSTIR GI,
/// area-shadow, then the SSGI rows changeSigma, slicesLow..High and stepsLow..High). The key
/// registry is key -> `TunableSpec`; that card attaches each spec by key to a ParamSpec of
/// pass.gi.volume, pass.gi.screen, pass.gi.area-shadow or pass.gi.ssgi with
/// `ParamBinding::Tunable { feature: FeatureId::Gi, key }`. The store lifecycle is the one stated
/// at the Handoff doc comment of `GiTunables` (this file) and is not restated here.
pub fn tunables() -> &'static [TunableSpec] {
    GI_TUNABLES
}

/// The position of `key` in [`GI_TUNABLES`] and in the values of a [`GiTunables`].
fn index_of(key: &str) -> Option<usize> {
    GI_TUNABLES.iter().position(|spec| spec.key == key)
}

/// `GiTunables` holds the current GI tunable values (contract 'gi-tunables-store'): one value per
/// [`GI_TUNABLES`] row in the same order, and a generation that [`GiTunables::set`] advances on
/// each change.
///
/// Handoff: `GiTunables` (this file) is taken over by the cards
/// 'raa-gi 2/5: gi.screen probe, accumulation and a-trous tunables',
/// 'raa-gi 3/5: ReSTIR GI tunables', 'raa-gi 4/5: area-shadow tunables' and
/// 'raa-gi 5/5: SSGI per-quality slices and steps'; each appends its rows to `GI_TUNABLES`.
/// Today the store holds the three gi.volume rows. The entry points those cards call are
/// `GiTunables::constants(&self, overrides: &[(&'static str, &'static str)]) -> Vec<(&'static str, f64)>`,
/// which returns the (override name, current value) pair of each (override name, key) input in
/// input order and skips an unknown key; `GiTunables::generation(&self) -> u64`, compared in
/// prepare with the generation a pipeline was built with; and
/// `GiShared::tunables(&self) -> GiTunables` (crates/raa-gi/src/parts.rs), a clone of the published
/// copy that allocates, so it is called only when the generation differs. The state they pass is
/// the key registry `GI_TUNABLES` (key -> current value, read with `GiTunables::value`).
///
/// LIFECYCLE: `GiTunables` is created in `GiShared::new` (crates/raa-gi/src/parts.rs:217) at the
/// defaults and lives as long as `GiShared`. `GiHandle::set_tunable` writes the core copy and
/// publishes it to `GiShared` immediately. Each GI pass and part reads
/// `GiShared::tunables_generation()` in its `prepare` (pipeline.rs:1002-1010 runs resize then
/// prepare for each enabled pass, in order, before any record). Only when it differs from the
/// generation its pipelines were built with does it clone `GiShared::tunables()` once and rebuild
/// its own pipelines. Uniform rows are read each prepare with `GiShared::tunable_value(key)`. Both
/// per-frame reads are allocation-free (Q4). No pass or part drops or replaces the store. Today
/// the gi.volume pass (`GiVolumePass::prepare`, crates/raa-gi/src/feature.rs) is the one reader.
#[derive(Clone, Debug, PartialEq)]
pub struct GiTunables {
    values: Vec<f64>,
    generation: u64,
}

impl GiTunables {
    /// Every default of [`GI_TUNABLES`], at generation 0.
    pub fn new() -> GiTunables {
        GiTunables {
            values: GI_TUNABLES.iter().map(|spec| spec.default).collect(),
            generation: 0,
        }
    }

    /// The current value of `key`, or `None` for a key that [`GI_TUNABLES`] does not list.
    pub fn value(&self, key: &str) -> Option<f64> {
        index_of(key).and_then(|index| self.values.get(index).copied())
    }

    /// Validates `value` for `key` with [`validate_tunable`] over [`GI_TUNABLES`] and stores it.
    /// Returns `Ok(true)` when the value changed, which advances the generation; `Ok(false)` when
    /// it was already stored. A refusal (`gi.invalid`, the validator's message) changes nothing.
    pub fn set(&mut self, key: &str, value: f64) -> Result<bool, RaaError> {
        validate_tunable(GI_TUNABLES, key, value).map_err(|message| invalid(key, &message))?;
        let Some(slot) = index_of(key).and_then(|index| self.values.get_mut(index)) else {
            return Err(invalid(key, "unknown tunable"));
        };
        if *slot == value {
            return Ok(false);
        }
        *slot = value;
        self.generation = self.generation.wrapping_add(1);
        Ok(true)
    }

    /// The number of changes stored since construction.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Each (override name, key) pair as (override name, current value), in input order. A key
    /// that is not in [`GI_TUNABLES`] is skipped. The result is passed as
    /// `compilation_options.constants` of the pipelines that declare those overrides.
    pub fn constants(
        &self,
        overrides: &[(&'static str, &'static str)],
    ) -> Vec<(&'static str, f64)> {
        overrides
            .iter()
            .filter_map(|&(name, key)| self.value(key).map(|value| (name, value)))
            .collect()
    }
}

impl Default for GiTunables {
    fn default() -> GiTunables {
        GiTunables::new()
    }
}
