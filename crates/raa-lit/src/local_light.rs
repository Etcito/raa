//! S17 local-light descriptors and stable handles; linear RGB at this boundary.
//!
//! Handoff: bounded handles (`crates/raa-lit/src/local_lights.rs`) call
//! `pub fn LocalLight::validate(self) -> Result<Self, RaaError>` from
//! `crates/raa-lit/src/local_light.rs`, attached by `mod local_light;` and the re-exports in
//! `crates/raa-lit/src/lib.rs`, before mutating their capacity-256 `LightHandle`-keyed table.
//! Handoff: attachments (`crates/raa-lit/src/local_lights.rs`) call the same `validate` after
//! applying an instance transform and keep `range`/`size` in metres.
//! Handoff: Engine (`crates/raa-js/src/engine.rs`) builds `LocalLight::default()` descriptors,
//! calls `validate` before mutating state and keys the renderer registry by `LightHandle`.
//! Handoff: `LightGpu` (same file) writes its record from a validated `LocalLight` using
//! `LocalLightKind::rank` and `effective_inner_angle_deg`.

use raa_core::RaaError;

/// Stable identifier of a local light: positive, at most [`MAX_LIGHT_HANDLE`] and never
/// reused for a different light.
pub type LightHandle = u64;

/// Largest valid [`LightHandle`], the maximum integer a JS number represents exactly.
pub const MAX_LIGHT_HANDLE: LightHandle = 9_007_199_254_740_991;

/// The four S17 local-light kinds; [`LocalLightKind::rank`] is the GPU kind word.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum LocalLightKind {
    /// Point light, GPU rank 0.
    #[default]
    Point,
    /// Spot light, GPU rank 1.
    Spot,
    /// Directional light, GPU rank 2.
    Directional,
    /// Area (rectangle) light, GPU rank 3.
    Area,
}

impl LocalLightKind {
    /// Kind word of `LightGpu` lane 3: point 0, spot 1, directional 2, area 3.
    pub const fn rank(self) -> u32 {
        match self {
            LocalLightKind::Point => 0,
            LocalLightKind::Spot => 1,
            LocalLightKind::Directional => 2,
            LocalLightKind::Area => 3,
        }
    }
}

/// An S17 local-light descriptor: metres, full cone angles in degrees and linear RGB.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LocalLight {
    pub kind: LocalLightKind,
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    pub angle_deg: f32,
    pub inner_angle_deg: Option<f32>,
    pub size: [f32; 2],
    pub radius: f32,
    pub shadows: bool,
}

impl Default for LocalLight {
    /// S17 defaults: point, origin, [0, 0, -1], white, intensity 2, range 6, angle 35,
    /// no inner angle, size [1, 1], radius 0, shadows off.
    fn default() -> Self {
        LocalLight {
            kind: LocalLightKind::Point,
            position: [0.0; 3],
            direction: [0.0, 0.0, -1.0],
            color: [1.0; 3],
            intensity: 2.0,
            range: 6.0,
            angle_deg: 35.0,
            inner_angle_deg: None,
            size: [1.0, 1.0],
            radius: 0.0,
            shadows: false,
        }
    }
}

fn invalid(message: String) -> RaaError {
    RaaError::FrameGraph {
        code: "light.invalid_desc",
        message,
    }
}

fn check_finite(field: &str, v: f32) -> Result<(), RaaError> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(invalid(format!("{field} must be finite (got {v})")))
    }
}

fn check_min(field: &str, v: f32, min: f32, inclusive: bool) -> Result<(), RaaError> {
    check_finite(field, v)?;
    if (inclusive && v >= min) || (!inclusive && v > min) {
        Ok(())
    } else {
        let op = if inclusive { ">=" } else { ">" };
        Err(invalid(format!("{field} must be {op} {min} (got {v})")))
    }
}

fn check_angle(field: &str, v: f32) -> Result<(), RaaError> {
    check_finite(field, v)?;
    if (1.0..=179.0).contains(&v) {
        Ok(())
    } else {
        Err(invalid(format!("{field} must be in [1, 179] (got {v})")))
    }
}

fn normalized_direction(direction: [f32; 3]) -> Result<[f32; 3], RaaError> {
    for (i, v) in direction.iter().enumerate() {
        check_finite(&format!("direction[{i}]"), *v)?;
    }
    let length = direction
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        .sqrt();
    if !length.is_finite() || length <= 0.0 {
        return Err(invalid(format!(
            "direction must be a nonzero finite vector (got {direction:?})"
        )));
    }
    let mut out = [0.0; 3];
    for (unit, v) in out.iter_mut().zip(direction) {
        *unit = (f64::from(v) / length) as f32;
    }
    Ok(out)
}

impl LocalLight {
    /// Handoff: `LocalLight::write_gpu(self, out: &mut [u8; LIGHT_GPU_SIZE]) -> ()`
    /// in `crates/raa-lit/src/local_light.rs` encodes a validated world descriptor.
    /// `LocalLights::write_gpu` attaches this to its ascending `LightHandle` registry;
    /// handles, radius and shadows stay outside the S10 record.
    pub(crate) fn write_gpu(self, out: &mut [u8; raa_shader::LIGHT_GPU_SIZE]) {
        // Compute in f64 to avoid intermediate overflow; extremely small positive
        // ranges saturate at the largest finite GPU value instead of emitting infinity.
        let inverse_range_squared =
            (1.0 / f64::from(self.range).powi(2)).min(f64::from(f32::MAX)) as f32;
        let words = [
            self.position[0],
            self.position[1],
            self.position[2],
            self.range,
            self.color[0] * self.intensity,
            self.color[1] * self.intensity,
            self.color[2] * self.intensity,
            inverse_range_squared,
            self.direction[0],
            self.direction[1],
            self.direction[2],
            (self.angle_deg.to_radians() * 0.5).cos(),
            (self.effective_inner_angle_deg().to_radians() * 0.5).cos(),
            0.0,
            self.size[0],
            self.size[1],
        ];
        for (bytes, value) in out.chunks_exact_mut(4).zip(words) {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        out[52..56].copy_from_slice(&self.kind.rank().to_le_bytes());
    }

    /// Effective full inner cone angle: the authored value, or 80 percent of `angle_deg`.
    pub fn effective_inner_angle_deg(self) -> f32 {
        self.inner_angle_deg.unwrap_or(self.angle_deg * 0.8)
    }

    /// Validates without mutating the caller's stored value. Every numeric field is finite;
    /// colour is in [0, 1], intensity and radius are nonnegative, range and size positive,
    /// angles are full degrees in [1, 179] and an explicit inner angle does not exceed the
    /// outer one. A nonzero finite `direction` is returned normalized. Refusals have code
    /// `light.invalid_desc` and name the field.
    pub fn validate(self) -> Result<Self, RaaError> {
        for (i, v) in self.position.iter().enumerate() {
            check_finite(&format!("position[{i}]"), *v)?;
        }
        let direction = normalized_direction(self.direction)?;
        for (i, v) in self.color.iter().enumerate() {
            check_finite(&format!("color[{i}]"), *v)?;
            if !(0.0..=1.0).contains(v) {
                return Err(invalid(format!("color[{i}] must be in [0, 1] (got {v})")));
            }
        }
        check_min("intensity", self.intensity, 0.0, true)?;
        check_min("range", self.range, 0.0, false)?;
        check_angle("angle_deg", self.angle_deg)?;
        if let Some(inner) = self.inner_angle_deg {
            check_angle("inner_angle_deg", inner)?;
            if inner > self.angle_deg {
                return Err(invalid(format!(
                    "inner_angle_deg must not exceed angle_deg (got {inner} > {})",
                    self.angle_deg
                )));
            }
        }
        for (i, v) in self.size.iter().enumerate() {
            check_min(&format!("size[{i}]"), *v, 0.0, false)?;
        }
        check_min("radius", self.radius, 0.0, true)?;
        Ok(LocalLight { direction, ..self })
    }
}

#[cfg(test)]
#[rustfmt::skip]
mod tests {
    use super::*;
    use LocalLightKind::{Area, Directional, Point, Spot};

    const NAN: f32 = f32::NAN; const INF: f32 = f32::INFINITY;

    fn lit(edit: impl FnOnce(&mut LocalLight)) -> LocalLight {
        let mut light = LocalLight::default();
        edit(&mut light);
        light
    }

    fn assert_refused(light: LocalLight, field: &str) {
        let err = light.validate().expect_err("descriptor must be refused");
        assert_eq!(err.code(), "light.invalid_desc");
        assert!(err.message().contains(field), "{} lacks {field}", err.message());
    }

    #[test]
    fn defaults_and_kind_ranks() {
        let light = LocalLight::default();
        assert_eq!((light.kind, light.position), (Point, [0.0; 3]));
        assert_eq!((light.direction, light.color), ([0.0, 0.0, -1.0], [1.0; 3]));
        assert_eq!((light.intensity, light.range, light.angle_deg), (2.0, 6.0, 35.0));
        assert_eq!((light.inner_angle_deg, light.size, light.radius, light.shadows), (None, [1.0, 1.0], 0.0, false));
        assert_eq!(light.validate().unwrap(), light);
        let ranks = [Point.rank(), Spot.rank(), Directional.rank(), Area.rank()];
        assert_eq!(ranks, [0, 1, 2, 3]);
        assert_eq!(MAX_LIGHT_HANDLE, 9_007_199_254_740_991);
    }

    #[test]
    fn direction_is_normalized_on_acceptance_only() {
        let authored = lit(|l| l.direction = [3.0, 4.0, 0.0]);
        let accepted = authored.validate().unwrap();
        assert!((accepted.direction[0] - 0.6).abs() < 1e-6);
        assert!((accepted.direction[1] - 0.8).abs() < 1e-6);
        assert_eq!(accepted.direction[2], 0.0);
        assert_eq!(authored.direction, [3.0, 4.0, 0.0]);
        let huge = lit(|l| l.direction = [1e30, 0.0, 0.0]).validate().unwrap();
        assert_eq!(huge.direction, [1.0, 0.0, 0.0]);
    }
    #[test]
    fn finite_and_range_rules_are_refused() {
        let cases = [
            (lit(|l| l.position = [NAN, 0.0, 0.0]), "position"),
            (lit(|l| l.direction = [0.0; 3]), "direction"),
            (lit(|l| l.direction = [INF, 0.0, 0.0]), "direction"),
            (lit(|l| l.color = [0.0, INF, 0.0]), "color"),
            (lit(|l| l.color = [-0.1, 0.0, 0.0]), "color"),
            (lit(|l| l.color = [0.0, 1.1, 0.0]), "color"),
            (lit(|l| l.intensity = NAN), "intensity"),
            (lit(|l| l.intensity = -1.0), "intensity"),
            (lit(|l| l.range = INF), "range"),
            (lit(|l| l.range = 0.0), "range"),
            (lit(|l| l.angle_deg = NAN), "angle_deg"),
            (lit(|l| l.angle_deg = 0.0), "angle_deg"),
            (lit(|l| l.angle_deg = 180.0), "angle_deg"),
            (lit(|l| l.size = [0.0, 1.0]), "size"),
            (lit(|l| l.size = [1.0, NAN]), "size"),
            (lit(|l| l.radius = -0.5), "radius"),
            (lit(|l| l.radius = INF), "radius"),
        ];
        for (light, field) in cases { assert_refused(light, field); }
        let edges = LocalLight {
            color: [0.0, 1.0, 0.5],
            intensity: 0.0,
            angle_deg: 1.0,
            size: [f32::MIN_POSITIVE, 2.0],
            ..LocalLight::default()
        };
        assert!(edges.validate().is_ok());
        assert!(lit(|l| l.angle_deg = 179.0).validate().is_ok());
    }
    #[test]
    fn inner_angles_are_optional_equal_or_retained() {
        let omitted = LocalLight::default().validate().unwrap();
        assert_eq!(omitted.inner_angle_deg, None);
        assert!((omitted.effective_inner_angle_deg() - 28.0).abs() < 1e-6);
        let equal = lit(|l| l.inner_angle_deg = Some(35.0)).validate().unwrap();
        assert_eq!(equal.effective_inner_angle_deg(), 35.0);
        let bad = [
            (lit(|l| l.inner_angle_deg = Some(36.0)), "inner_angle_deg"),
            (lit(|l| l.inner_angle_deg = Some(0.5)), "inner_angle_deg"),
            (lit(|l| l.inner_angle_deg = Some(180.0)), "inner_angle_deg"),
            (lit(|l| l.inner_angle_deg = Some(NAN)), "inner_angle_deg"),
        ];
        for (light, field) in bad { assert_refused(light, field); }
        let authored = LocalLight {
            angle_deg: 60.0,
            inner_angle_deg: Some(45.0),
            ..LocalLight::default()
        };
        for outer in [60.0, 90.0, 179.0] {
            let mut updated = authored;
            updated.angle_deg = outer;
            let accepted = updated.validate().unwrap();
            assert_eq!((accepted.angle_deg, accepted.inner_angle_deg), (outer, Some(45.0)));
            assert_eq!(accepted.effective_inner_angle_deg(), 45.0);
        }
        let mut too_narrow = authored;
        too_narrow.angle_deg = 40.0;
        assert_eq!(too_narrow.validate().unwrap_err().code(), "light.invalid_desc");
        assert_eq!((authored.angle_deg, authored.inner_angle_deg), (60.0, Some(45.0)));
    }
    #[test]
    fn metadata_and_kinds_are_retained() {
        for kind in [Point, Spot, Directional, Area] {
            let light = LocalLight {
                kind,
                radius: 2.5,
                shadows: true,
                ..LocalLight::default()
            };
            let accepted = light.validate().unwrap();
            assert_eq!((accepted.kind, accepted.radius, accepted.shadows), (kind, 2.5, true));
        }
    }
}
