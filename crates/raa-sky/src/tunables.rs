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
];

/// The number of atmosphere rows, the first entries of [`SKY_TUNABLES`].
pub(crate) const ATMOSPHERE_ROWS: usize = 2;

/// The WGSL override names of the atmosphere rows, in [`SKY_TUNABLES`] order.
const ATMOSPHERE_OVERRIDES: [&str; ATMOSPHERE_ROWS] =
    ["RAA_SKY_VIEW_STEPS", "raa_atmosphere_horizon_fade"];

/// The static list of raa-sky tunables.
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
