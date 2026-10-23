//! Shadow description and validation (`shadow.invalid`).

use raa_core::RaaError;

/// Sun shadow settings.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ShadowDesc {
    /// Multiplies the sun angular radius; in (0, 16].
    pub softness: f32,
    /// Shadow distance in metres (> 0).
    pub distance_m: f32,
    /// Cascade resolution: 1024, 2048 or 4096.
    pub resolution: u32,
}

impl Default for ShadowDesc {
    fn default() -> Self {
        ShadowDesc {
            softness: 1.0,
            distance_m: 60.0,
            resolution: 2048,
        }
    }
}

fn invalid(message: String) -> RaaError {
    RaaError::FrameGraph {
        code: "shadow.invalid",
        message,
    }
}

impl ShadowDesc {
    /// Refuses with `shadow.invalid`, the message naming the offending field.
    pub fn validate(&self) -> Result<(), RaaError> {
        if !(self.softness.is_finite() && self.softness > 0.0 && self.softness <= 16.0) {
            return Err(invalid(format!(
                "softness must be finite and in (0, 16], got {}",
                self.softness
            )));
        }
        if !(self.distance_m.is_finite() && self.distance_m > 0.0) {
            return Err(invalid(format!(
                "distance_m must be finite and > 0, got {}",
                self.distance_m
            )));
        }
        if !matches!(self.resolution, 1024 | 2048 | 4096) {
            return Err(invalid(format!(
                "resolution must be 1024, 2048 or 4096, got {}",
                self.resolution
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refused(d: ShadowDesc, field: &str) {
        match d.validate() {
            Err(RaaError::FrameGraph { code, message }) => {
                assert_eq!(code, "shadow.invalid");
                assert!(!message.is_empty());
                assert!(message.contains(field), "{message}");
            }
            other => panic!("expected refusal, got {other:?}"),
        }
    }

    #[test]
    fn default_is_valid() {
        assert!(ShadowDesc::default().validate().is_ok());
    }

    #[test]
    fn softness_refused() {
        for v in [0.0, -1.0, 16.5, f32::NAN, f32::INFINITY] {
            refused(
                ShadowDesc {
                    softness: v,
                    ..Default::default()
                },
                "softness",
            );
        }
    }

    #[test]
    fn distance_refused() {
        for v in [0.0, -5.0, f32::NAN, f32::INFINITY] {
            refused(
                ShadowDesc {
                    distance_m: v,
                    ..Default::default()
                },
                "distance_m",
            );
        }
    }

    #[test]
    fn resolution_refused() {
        for v in [0, 512, 3000, 8192] {
            refused(
                ShadowDesc {
                    resolution: v,
                    ..Default::default()
                },
                "resolution",
            );
        }
    }
}
