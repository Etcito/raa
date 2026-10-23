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
