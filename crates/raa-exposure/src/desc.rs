//! Automatic exposure settings (S46 `AutoExposureDesc`) and their validation (`post.invalid`).

use raa_core::RaaError;

/// Automatic exposure settings. The JS keys are `enabled`, `compensationEv`, `lowPercent`,
/// `highPercent`, `speedUp`, `speedDown`, `minEv` and `maxEv`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AutoExposureDesc {
    /// Whether the `exposure.auto` pass runs; off by default.
    pub enabled: bool,
    /// Exposure compensation in EV, finite in [-8, 8]; the default is log2 1.2.
    pub compensation_ev: f32,
    /// Lower end of the histogram band, finite in [0, 1] and below `high_percent`.
    pub low_percent: f32,
    /// Upper end of the histogram band, finite in [0, 1].
    pub high_percent: f32,
    /// Adaptation rate towards a brighter scene, in 1/s; finite and > 0.
    pub speed_up: f32,
    /// Adaptation rate towards a darker scene, in 1/s; finite and > 0.
    pub speed_down: f32,
    /// Lowest exposure in EV; finite and <= `max_ev`.
    pub min_ev: f32,
    /// Highest exposure in EV; finite.
    pub max_ev: f32,
}

impl Default for AutoExposureDesc {
    fn default() -> Self {
        AutoExposureDesc {
            enabled: false,
            compensation_ev: 0.263,
            low_percent: 0.2,
            high_percent: 0.7,
            speed_up: 2.4,
            speed_down: 1.1,
            min_ev: -6.0,
            max_ev: 6.0,
        }
    }
}

impl AutoExposureDesc {
    /// Refuses with code post.invalid and a message naming the JS key of the first failing row,
    /// in the S46 row order; returns Ok and changes nothing otherwise.
    ///
    /// Handoff: raa-exposure 6/6: exposure.auto pass, install and handle. This is the entry point
    /// `AutoExposureDesc::validate(&self) -> Result<(), RaaError>`, exported by
    /// `crates/raa-exposure/src/desc.rs` and re-exported by `crates/raa-exposure/src/lib.rs`.
    /// Today it is a pure check: it reads the fields and returns the first refusal, with no
    /// registry and no state. The handle takes the desc value only and calls this before it
    /// stores the desc; a refusal is `RaaError::FrameGraph { code: "post.invalid", message }`.
    pub fn validate(&self) -> Result<(), RaaError> {
        let (compensation, low, high) = (self.compensation_ev, self.low_percent, self.high_percent);
        if !finite_in(compensation, -8.0, 8.0) {
            return refuse(format!(
                "`compensationEv` must be finite in [-8, 8], got {compensation}"
            ));
        }
        if !finite_in(low, 0.0, 1.0) {
            return refuse(format!("`lowPercent` must be finite in [0, 1], got {low}"));
        }
        if high.is_finite() && low >= high {
            return refuse(format!(
                "`lowPercent` must be below `highPercent`, got {low} >= {high}"
            ));
        }
        if !finite_in(high, 0.0, 1.0) {
            return refuse(format!(
                "`highPercent` must be finite in [0, 1], got {high}"
            ));
        }
        if !(self.speed_up.is_finite() && self.speed_up > 0.0) {
            return refuse(format!(
                "`speedUp` must be finite and > 0, got {}",
                self.speed_up
            ));
        }
        if !(self.speed_down.is_finite() && self.speed_down > 0.0) {
            return refuse(format!(
                "`speedDown` must be finite and > 0, got {}",
                self.speed_down
            ));
        }
        let (min, max) = (self.min_ev, self.max_ev);
        if !min.is_finite() {
            return refuse(format!("`minEv` must be finite, got {min}"));
        }
        if max.is_finite() && min > max {
            return refuse(format!("`minEv` must be <= `maxEv`, got {min} > {max}"));
        }
        if !max.is_finite() {
            return refuse(format!("`maxEv` must be finite, got {max}"));
        }
        Ok(())
    }
}

fn finite_in(v: f32, lo: f32, hi: f32) -> bool {
    v.is_finite() && (lo..=hi).contains(&v)
}

fn refuse(message: String) -> Result<(), RaaError> {
    Err(RaaError::FrameGraph {
        code: "post.invalid",
        message,
    })
}
