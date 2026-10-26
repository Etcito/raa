use raa_core::RaaError;

const POST_INVALID: &str = "post.invalid";

/// Screen-space reflection settings.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SsrDesc {
    /// Scale of the reflection composite, in `[0, 2]`.
    pub intensity: f32,
    /// Surfaces above this roughness get no reflection, in `[0, 1]`.
    pub max_roughness: f32,
    /// Depth tolerance of a hit along the view ray, in metres, in `(0, 10]`.
    pub thickness_m: f32,
}

impl Default for SsrDesc {
    fn default() -> Self {
        SsrDesc {
            intensity: 1.0,
            max_roughness: 0.6,
            thickness_m: 0.15,
        }
    }
}

impl SsrDesc {
    /// Refuses with `post.invalid`, naming the offending field; non-finite values are refused.
    pub fn validate(&self) -> Result<(), RaaError> {
        check(
            "intensity",
            self.intensity,
            |v| (0.0..=2.0).contains(&v),
            "in [0, 2]",
        )?;
        check(
            "max_roughness",
            self.max_roughness,
            |v| (0.0..=1.0).contains(&v),
            "in [0, 1]",
        )?;
        check(
            "thickness_m",
            self.thickness_m,
            |v| v > 0.0 && v <= 10.0,
            "in (0, 10]",
        )
    }
}

fn check(
    field: &str,
    v: f32,
    in_range: impl Fn(f32) -> bool,
    expected: &str,
) -> Result<(), RaaError> {
    let message = if !v.is_finite() {
        format!("`{field}` must be finite, got {v}")
    } else if !in_range(v) {
        format!("`{field}` must be {expected}, got {v}")
    } else {
        return Ok(());
    };
    Err(RaaError::FrameGraph {
        code: POST_INVALID,
        message,
    })
}
