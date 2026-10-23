//! Contact shadow description and validation (`shadow.invalid`).

use raa_core::RaaError;

/// Screen-space contact shadow parameters (CONTACT values by default).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ContactDesc {
    /// Length of the march toward the sun in metres, in (0, 5].
    pub distance_m: f32,
    /// Depth a surface is assumed to have behind the depth buffer in metres, in (0, 1].
    pub thickness_m: f32,
    /// Strength of the darkening, in [0, 1] (0 = no effect).
    pub intensity: f32,
}

impl Default for ContactDesc {
    /// CONTACT: distance 0.35 m, thickness 0.05 m, full intensity.
    fn default() -> Self {
        ContactDesc {
            distance_m: 0.35,
            thickness_m: 0.05,
            intensity: 1.0,
        }
    }
}

impl ContactDesc {
    /// Refuses with `shadow.invalid` and a message naming the field when a value is not
    /// finite or out of its range.
    pub fn validate(&self) -> Result<(), RaaError> {
        open_closed("distance_m", self.distance_m, 5.0)?;
        open_closed("thickness_m", self.thickness_m, 1.0)?;
        if !self.intensity.is_finite() || !(0.0..=1.0).contains(&self.intensity) {
            return Err(invalid(format!(
                "intensity must be finite and in [0, 1], got {}",
                self.intensity
            )));
        }
        Ok(())
    }
}

/// `v` finite and in (0, max].
fn open_closed(field: &str, v: f32, max: f32) -> Result<(), RaaError> {
    if v.is_finite() && v > 0.0 && v <= max {
        Ok(())
    } else {
        Err(invalid(format!(
            "{field} must be finite and in (0, {max}], got {v}"
        )))
    }
}

fn invalid(message: String) -> RaaError {
    RaaError::FrameGraph {
        code: "shadow.invalid",
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refused(d: ContactDesc, field: &str) {
        let err = d.validate().expect_err(field);
        assert_eq!(err.code(), "shadow.invalid");
        assert!(err.to_string().contains(field), "{field}: {err}");
    }

    #[test]
    fn default_is_contact_values_and_valid() {
        let d = ContactDesc::default();
        assert_eq!(
            (d.distance_m, d.thickness_m, d.intensity),
            (0.35, 0.05, 1.0)
        );
        assert!(d.validate().is_ok());
    }

    #[test]
    fn range_limits_are_accepted() {
        let d = ContactDesc {
            distance_m: 5.0,
            thickness_m: 1.0,
            intensity: 0.0,
        };
        assert!(d.validate().is_ok());
    }

    #[test]
    fn distance_out_of_range_is_refused() {
        for v in [0.0, -1.0, 5.01, f32::NAN, f32::INFINITY] {
            refused(
                ContactDesc {
                    distance_m: v,
                    ..ContactDesc::default()
                },
                "distance_m",
            );
        }
    }

    #[test]
    fn thickness_out_of_range_is_refused() {
        for v in [0.0, -0.1, 1.5, f32::NAN, f32::NEG_INFINITY] {
            refused(
                ContactDesc {
                    thickness_m: v,
                    ..ContactDesc::default()
                },
                "thickness_m",
            );
        }
    }

    #[test]
    fn intensity_out_of_range_is_refused() {
        for v in [-0.01, 1.01, f32::NAN, f32::INFINITY] {
            refused(
                ContactDesc {
                    intensity: v,
                    ..ContactDesc::default()
                },
                "intensity",
            );
        }
    }
}
