//! Ambient-occlusion description and validation (`ao.invalid`).

use raa_core::RaaError;

/// Ground-truth ambient occlusion settings of the `ao.gtao` pass (S12).
///
/// The values follow the GTAO row of the architecture values table: the horizon search
/// covers `radius_m` metres at full resolution with `intensity` scaling the occlusion and
/// `power` shaping its falloff. [`AoDesc::validate`] refuses anything outside those ranges
/// with `ao.invalid` and a message naming the offending field.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AoDesc {
    /// World-space sampling radius in metres, in `(0, 10]`.
    pub radius_m: f32,
    /// Occlusion strength, in `[0, 1]` (0 disables AO, 1 full darkening).
    pub intensity: f32,
    /// Exponent applied to the visibility before mixing, in `(0, 8]`.
    pub power: f32,
}

impl Default for AoDesc {
    /// GTAO defaults: radius 0.8 m, intensity 1, power 1.5.
    fn default() -> Self {
        AoDesc {
            radius_m: 0.8,
            intensity: 1.0,
            power: 1.5,
        }
    }
}

impl AoDesc {
    /// Refuses with `ao.invalid` and a message naming the first offending field.
    ///
    /// `radius_m` must be finite and in `(0, 10]`; `intensity` finite and in `[0, 1]`;
    /// `power` finite and in `(0, 8]`. NaN and infinities are refused by the finiteness
    /// check before the range is tested, so every bound has its own refusal.
    pub fn validate(&self) -> Result<(), RaaError> {
        open_closed("radius_m", self.radius_m, 0.0, 10.0)?;
        closed("intensity", self.intensity, 0.0, 1.0)?;
        open_closed("power", self.power, 0.0, 8.0)?;
        Ok(())
    }
}

/// The sampling effort of the `ao.gtao` pass: `slices` directions per pixel, each marching
/// `steps` horizon samples per side. The default (4 x 8) is the GTAO row of the values
/// table; fewer is cheaper and noisier. Kept apart from [`AoDesc`] so a look (radius,
/// intensity, power) and a player's quality choice never overwrite each other.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AoQuality {
    /// Slice directions per pixel, in `1..=8`.
    pub slices: u32,
    /// Horizon steps per side of a slice, in `1..=16`.
    pub steps: u32,
}

impl Default for AoQuality {
    /// 4 slices x 8 steps.
    fn default() -> Self {
        AoQuality {
            slices: 4,
            steps: 8,
        }
    }
}

impl AoQuality {
    /// Refuses with `ao.invalid` when `slices` is outside `1..=8` or `steps` outside
    /// `1..=16`.
    pub fn validate(&self) -> Result<(), RaaError> {
        if !(1..=8).contains(&self.slices) {
            return Err(invalid(format!(
                "`slices` must be in 1..=8, got {}",
                self.slices
            )));
        }
        if !(1..=16).contains(&self.steps) {
            return Err(invalid(format!(
                "`steps` must be in 1..=16, got {}",
                self.steps
            )));
        }
        Ok(())
    }
}

fn invalid(message: String) -> RaaError {
    RaaError::FrameGraph {
        code: "ao.invalid",
        message,
    }
}

fn finite(field: &str, v: f32) -> Result<(), RaaError> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(invalid(format!("`{field}` must be finite, got {v}")))
    }
}

/// Finite and in the half-open interval `(min, max]`.
fn open_closed(field: &str, v: f32, min: f32, max: f32) -> Result<(), RaaError> {
    finite(field, v)?;
    if v > min && v <= max {
        Ok(())
    } else {
        Err(invalid(format!(
            "`{field}` must be in ({min}, {max}], got {v}"
        )))
    }
}

/// Finite and in the closed interval `[min, max]`.
fn closed(field: &str, v: f32, min: f32, max: f32) -> Result<(), RaaError> {
    finite(field, v)?;
    if (min..=max).contains(&v) {
        Ok(())
    } else {
        Err(invalid(format!(
            "`{field}` must be in [{min}, {max}], got {v}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_default_is_4_by_8_and_bounds_are_refused() {
        assert_eq!(
            AoQuality::default(),
            AoQuality {
                slices: 4,
                steps: 8
            }
        );
        for (slices, steps) in [(1, 1), (8, 16)] {
            AoQuality { slices, steps }.validate().unwrap();
        }
        for (slices, steps) in [(0, 8), (9, 8), (4, 0), (4, 17)] {
            let err = AoQuality { slices, steps }.validate().unwrap_err();
            assert_eq!(err.code(), "ao.invalid");
        }
    }

    #[test]
    fn default_matches_contract() {
        assert_eq!(
            AoDesc::default(),
            AoDesc {
                radius_m: 0.8,
                intensity: 1.0,
                power: 1.5,
            }
        );
    }
}
