use raa_core::RaaError;

/// Sun shafts parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShaftsDesc {
    pub intensity: f32,
    pub density: f32,
    pub max_distance_m: f32,
}

impl Default for ShaftsDesc {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            density: 0.02,
            max_distance_m: 60.0,
        }
    }
}

impl ShaftsDesc {
    /// Refuses with `post.invalid`, naming the offending field.
    pub fn validate(&self) -> Result<(), RaaError> {
        check("intensity", self.intensity, 0.0, 8.0, true)?;
        check("density", self.density, 0.0, 1.0, true)?;
        check("max_distance_m", self.max_distance_m, 0.0, 1000.0, false)
    }
}

fn check(field: &str, v: f32, lo: f32, hi: f32, lo_inclusive: bool) -> Result<(), RaaError> {
    let ok_lo = if lo_inclusive { v >= lo } else { v > lo };
    let message = if !v.is_finite() {
        format!("`{field}` must be finite, got {v}")
    } else if !ok_lo || v > hi {
        let open = if lo_inclusive { '[' } else { '(' };
        format!("`{field}` must be in {open}{lo}, {hi}], got {v}")
    } else {
        return Ok(());
    };
    Err(RaaError::FrameGraph {
        code: "post.invalid",
        message,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refused(d: ShaftsDesc, field: &str) {
        let err = d.validate().unwrap_err();
        assert_eq!(err.code(), "post.invalid");
        assert!(err.to_string().contains(field), "{err}");
    }

    #[test]
    fn defaults_validate() {
        let d = ShaftsDesc::default();
        assert_eq!((d.intensity, d.density, d.max_distance_m), (1.0, 0.02, 60.0));
        d.validate().unwrap();
    }

    #[test]
    fn invalid_fields_are_refused_by_name() {
        let base = ShaftsDesc::default();
        for v in [-0.1, 8.5, f32::NAN, f32::INFINITY] {
            refused(ShaftsDesc { intensity: v, ..base }, "intensity");
        }
        for v in [-0.1, 1.5, f32::NAN] {
            refused(ShaftsDesc { density: v, ..base }, "density");
        }
        for v in [0.0, -1.0, 1000.5, f32::NAN, f32::INFINITY] {
            refused(ShaftsDesc { max_distance_m: v, ..base }, "max_distance_m");
        }
    }

    #[test]
    fn bounds_are_accepted() {
        let d = ShaftsDesc { intensity: 8.0, density: 0.0, max_distance_m: 1000.0 };
        d.validate().unwrap();
    }
}
