//! S56 tunables of raa-sky: the one key registry and the one value store.

use raa_core::RaaError;
use raa_pipeline::{TunableSpec, validate_tunable};

/// Every sky tunable, atmosphere rows first (docs/render-graph/tunables.md order).
///
/// Handoff: raa-sky 2/3: cloud tunables in CloudFrame and the cloud volume. SKY_TUNABLES is
/// the one key registry of raa-sky; `SkyTunables::set` (called by `SkyHandle::set_tunable`)
/// stores every key of it, including the cloud rows card 2/3 appends after the first
/// `ATMOSPHERE_ROWS` entries, and bumps `atmosphere_generation` only for atmosphere rows.
/// Card 2/3 reads cloud values with `SkyTunables::value` to build `CloudTunables`.
pub const SKY_TUNABLES: &[TunableSpec] = &[
    TunableSpec {
        key: "skyViewSteps",
        default: 32.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "pipeline",
        source: "crates/raa-sky/src/wgsl/atmosphere_luts.wgsl:248",
    },
    TunableSpec {
        key: "horizonFade",
        default: 0.006,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-sky/src/wgsl/atmosphere_luts.wgsl:314",
    },
    TunableSpec {
        key: "cloudWeatherTile",
        default: 36000.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-sky/src/clouds/wgsl/cloud_volume.wgsl:57",
    },
    TunableSpec {
        key: "cloudShapeTile",
        default: 7000.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-sky/src/clouds/wgsl/cloud_volume.wgsl:58",
    },
    TunableSpec {
        key: "cloudDetailTile",
        default: 700.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-sky/src/clouds/wgsl/cloud_volume.wgsl:59",
    },
    TunableSpec {
        key: "cloudExtinction",
        default: 0.05,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-sky/src/clouds/wgsl/cloud_volume.wgsl:62",
    },
    TunableSpec {
        key: "cloudAlbedo",
        default: 0.98,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-sky/src/clouds/wgsl/cloud_volume.wgsl:64",
    },
    TunableSpec {
        key: "cloudCirrusHeight",
        default: 9000.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-sky/src/clouds/wgsl/cloud_volume.wgsl:65",
    },
    TunableSpec {
        key: "cloudMaxDistance",
        default: 90000.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-sky/src/clouds/wgsl/cloud_volume.wgsl:67",
    },
    TunableSpec {
        key: "cloudMaxPath",
        default: 40000.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-sky/src/clouds/wgsl/cloud_volume.wgsl:69",
    },
    TunableSpec {
        key: "cloudDomeBelow",
        default: 0.05,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-sky/src/clouds/wgsl/dome.wgsl:20",
    },
    TunableSpec {
        key: "cloudDomeLightSamples",
        default: 5.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "pipeline",
        source: "crates/raa-sky/src/clouds/wgsl/dome.wgsl:21",
    },
    TunableSpec {
        key: "cloudDomeAnimatedFloor",
        default: 0.0625,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-sky/src/clouds/wgsl/dome.wgsl:23",
    },
    TunableSpec {
        key: "cloudDomeBurst",
        default: 2.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "pipeline",
        source: "crates/raa-sky/src/clouds/wgsl/dome.wgsl:25",
    },
    TunableSpec {
        key: "cloudScreenLightSamples",
        default: 5.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "pipeline",
        source: "crates/raa-sky/src/clouds/wgsl/screen.wgsl:52",
    },
    TunableSpec {
        key: "cloudScreenMovingBlend",
        default: 0.12,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-sky/src/clouds/wgsl/screen.wgsl:54",
    },
    TunableSpec {
        key: "cloudScreenPrior",
        default: 6.0,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-sky/src/clouds/wgsl/screen.wgsl:56",
    },
    TunableSpec {
        key: "cloudScreenAnimatedFloor",
        default: 0.0625,
        min: None,
        max: None,
        integer: false,
        rebuild: "pipeline",
        source: "crates/raa-sky/src/clouds/wgsl/screen.wgsl:58",
    },
    TunableSpec {
        key: "cloudScreenBurst",
        default: 2.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "pipeline",
        source: "crates/raa-sky/src/clouds/wgsl/screen.wgsl:60",
    },
    TunableSpec {
        key: "cloudShadowSamples",
        default: 16.0,
        min: None,
        max: None,
        integer: true,
        rebuild: "pipeline",
        source: "crates/raa-sky/src/clouds/wgsl/shadow.wgsl:31",
    },
];

/// The number of atmosphere rows, the first entries of [`SKY_TUNABLES`].
pub(crate) const ATMOSPHERE_ROWS: usize = 2;

/// The WGSL override names of the atmosphere rows, in [`SKY_TUNABLES`] order.
const ATMOSPHERE_OVERRIDES: [&str; ATMOSPHERE_ROWS] =
    ["RAA_SKY_VIEW_STEPS", "raa_atmosphere_horizon_fade"];

/// The static list of raa-sky tunables.
///
/// Handoff: Catalog family tunables_sky. `tunables` returns SKY_TUNABLES today: the atmosphere
/// rows, then every cloud row (cloud_volume.wgsl, dome.wgsl, screen.wgsl, shadow.wgsl), in the
/// order of docs/render-graph/tunables.md. The key registry is key -> TunableSpec; that card
/// attaches each spec by key to a ParamSpec of pass.sky.atmosphere, pass.sky.cloud-lighting or
/// pass.sky.cloud-screen with ParamBinding::Tunable { feature: FeatureId::Sky, key }. The cloud
/// rows reach the pipelines as in the Handoff at tunables.rs `CloudTunables`.
pub fn tunables() -> &'static [TunableSpec] {
    SKY_TUNABLES
}

/// The one store of every sky tunable value (atmosphere and cloud keys alike).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SkyTunables {
    values: Vec<f64>,
    atmosphere_generation: u64,
}

impl SkyTunables {
    pub(crate) fn new() -> SkyTunables {
        SkyTunables {
            values: SKY_TUNABLES.iter().map(|s| s.default).collect(),
            atmosphere_generation: 0,
        }
    }

    /// The current value of `key`, allocation-free.
    #[allow(dead_code)] // read by the cloud rows' CloudTunables (raa-sky 2/3)
    pub(crate) fn value(&self, key: &str) -> Option<f64> {
        let index = SKY_TUNABLES.iter().position(|s| s.key == key)?;
        self.values.get(index).copied()
    }

    /// Validates and stores `value`; Ok(true) when it changed. A refusal changes nothing.
    pub(crate) fn set(&mut self, key: &str, value: f64) -> Result<bool, RaaError> {
        validate_tunable(SKY_TUNABLES, key, value).map_err(crate::state::invalid)?;
        let Some(index) = SKY_TUNABLES.iter().position(|s| s.key == key) else {
            return Ok(false);
        };
        let Some(slot) = self.values.get_mut(index) else {
            return Ok(false);
        };
        if *slot == value {
            return Ok(false);
        }
        *slot = value;
        if index < ATMOSPHERE_ROWS {
            self.atmosphere_generation += 1;
        }
        Ok(true)
    }

    pub(crate) fn atmosphere_generation(&self) -> u64 {
        self.atmosphere_generation
    }

    /// The atmosphere LUT pipeline constants at the current values.
    pub(crate) fn atmosphere_constants(&self) -> [(&'static str, f64); ATMOSPHERE_ROWS] {
        std::array::from_fn(|i| {
            let value = self
                .values
                .get(i)
                .copied()
                .unwrap_or(SKY_TUNABLES[i].default);
            (ATMOSPHERE_OVERRIDES[i], value)
        })
    }
}

impl Default for SkyTunables {
    fn default() -> Self {
        SkyTunables::new()
    }
}

/// The number of cloud rows, the entries of [`SKY_TUNABLES`] after the atmosphere rows.
const CLOUD_ROWS: usize = SKY_TUNABLES.len() - ATMOSPHERE_ROWS;

/// The WGSL override names of the cloud rows (clouds/wgsl/cloud_volume.wgsl, dome.wgsl,
/// screen.wgsl and shadow.wgsl), in table order.
const CLOUD_OVERRIDES: [&str; CLOUD_ROWS] = [
    "RAA_CLOUD_WEATHER_TILE",
    "RAA_CLOUD_SHAPE_TILE",
    "RAA_CLOUD_DETAIL_TILE",
    "RAA_CLOUD_EXTINCTION",
    "RAA_CLOUD_ALBEDO",
    "RAA_CLOUD_CIRRUS_HEIGHT",
    "RAA_CLOUD_MAX_DISTANCE",
    "RAA_CLOUD_MAX_PATH",
    "RAA_CLOUD_DOME_BELOW",
    "RAA_CLOUD_DOME_LIGHT_SAMPLES",
    "RAA_CLOUD_DOME_ANIMATED_FLOOR",
    "RAA_CLOUD_DOME_BURST",
    "RAA_CLOUD_SCREEN_LIGHT_SAMPLES",
    "RAA_CLOUD_SCREEN_MOVING_BLEND",
    "RAA_CLOUD_SCREEN_PRIOR",
    "RAA_CLOUD_SCREEN_ANIMATED_FLOOR",
    "RAA_CLOUD_SCREEN_BURST",
    "RAA_CLOUD_SHADOW_SAMPLES",
];

/// A per-frame copy of the cloud rows of [`SKY_TUNABLES`], in table order; never a store.
///
/// Handoff: raa-sky 3/3: screen, dome and shadow cloud tunables. CloudTunables holds one f32
/// per cloud row of `SKY_TUNABLES` (CLOUD_ROWS = rows after `ATMOSPHERE_ROWS`, so appended
/// rows grow it); `CloudTunables::from_sky` copies them from SkyState's `SkyTunables` in
/// `cloud_frame` (cloud_backend.rs), which folds them into `CloudFrame::signature`.
/// `CloudTunables::constants` pairs each row with its `CLOUD_OVERRIDES` name and
/// clouds/passes.rs `CloudPasses::rebuild` passes them to every march pipeline (all compose
/// cloud_volume.wgsl); clouds/mod.rs `VolumetricClouds::prepare` rebuilds when
/// `CloudFrame.tunables` differs from the copy the pipelines were built with. The store and
/// `SkyHandle::set_tunable` are card 1/3's, see its Handoff at tunables.rs `SKY_TUNABLES`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudTunables {
    values: [f32; CLOUD_ROWS],
}

impl Default for CloudTunables {
    fn default() -> Self {
        CloudTunables {
            values: std::array::from_fn(|i| SKY_TUNABLES[ATMOSPHERE_ROWS + i].default as f32),
        }
    }
}

impl CloudTunables {
    /// The value of the cloud row `key`; None for an atmosphere or unknown key.
    pub fn value(&self, key: &str) -> Option<f64> {
        let index = SKY_TUNABLES
            .iter()
            .skip(ATMOSPHERE_ROWS)
            .position(|s| s.key == key)?;
        self.values.get(index).map(|v| f64::from(*v))
    }

    /// Copies the current cloud values out of the one store.
    pub(crate) fn from_sky(store: &SkyTunables) -> CloudTunables {
        let mut out = CloudTunables::default();
        for (slot, value) in out
            .values
            .iter_mut()
            .zip(store.values.iter().skip(ATMOSPHERE_ROWS))
        {
            *slot = *value as f32;
        }
        out
    }

    /// The bits of every value, in table order, for the frame signature.
    pub(crate) fn bits(&self) -> [u32; CLOUD_ROWS] {
        self.values.map(f32::to_bits)
    }

    /// The pipeline constants of every cloud override at these values; clouds/passes.rs passes
    /// each module only the ones it declares.
    pub(crate) fn constants(&self) -> [(&'static str, f64); CLOUD_ROWS] {
        std::array::from_fn(|i| (CLOUD_OVERRIDES[i], f64::from(self.values[i])))
    }
}
