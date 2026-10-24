//! Depth-of-field description, validation (`post.invalid`) and the thin-lens CoC.

use raa_core::RaaError;

/// Refusal code of invalid depth-of-field settings.
pub const POST_INVALID: &str = "post.invalid";

/// Sensor width in millimetres the thin-lens circle of confusion is mapped onto.
pub const SENSOR_WIDTH_MM: f32 = 36.0;

/// Viewport height the `max_coc_px` value is expressed at (S12 default 24 px at 720p).
pub const REFERENCE_VIEWPORT_H: f32 = 720.0;

/// Depth-of-field parameters.
///
/// `Default` is DOF: focus at 10 m, f/2.8, 50 mm lens, no autofocus, at most 24 px of
/// CoC at 720p.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DofDesc {
    /// Focus distance in metres.
    pub focus_distance_m: f32,
    /// Aperture f-number `N` in `[0.7, 64]` (smaller = wider aperture = more blur).
    pub f_stop: f32,
    /// Focal length in millimetres, in `[8, 600]`.
    pub focal_length_mm: f32,
    /// Optional autofocus sample in screen UV (`x` right, `y` down), each in `[0, 1]`.
    pub autofocus: Option<[f32; 2]>,
    /// CoC clamp in pixels at 720p, in `(0, 64]`; scaled by `viewport_h / 720`.
    pub max_coc_px: f32,
}

impl Default for DofDesc {
    fn default() -> Self {
        DofDesc {
            focus_distance_m: 10.0,
            f_stop: 2.8,
            focal_length_mm: 50.0,
            autofocus: None,
            max_coc_px: 24.0,
        }
    }
}

impl DofDesc {
    /// Refuses with `post.invalid`, naming the offending field; every value must be finite
    /// and the focus distance must exceed the focal length in metres.
    pub fn validate(&self) -> Result<(), RaaError> {
        finite("focus_distance_m", self.focus_distance_m)?;
        finite("f_stop", self.f_stop)?;
        finite("focal_length_mm", self.focal_length_mm)?;
        finite("max_coc_px", self.max_coc_px)?;

        let focal_m = self.focal_length_mm / 1000.0;
        if self.focus_distance_m <= focal_m {
            return Err(invalid(format!(
                "`focus_distance_m` must be greater than the focal length ({focal_m} m), got {}",
                self.focus_distance_m
            )));
        }
        if !(0.7..=64.0).contains(&self.f_stop) {
            return Err(invalid(format!(
                "`f_stop` must be in [0.7, 64], got {}",
                self.f_stop
            )));
        }
        if !(8.0..=600.0).contains(&self.focal_length_mm) {
            return Err(invalid(format!(
                "`focal_length_mm` must be in [8, 600], got {}",
                self.focal_length_mm
            )));
        }
        if !(self.max_coc_px > 0.0 && self.max_coc_px <= 64.0) {
            return Err(invalid(format!(
                "`max_coc_px` must be in (0, 64], got {}",
                self.max_coc_px
            )));
        }
        if let Some(uv) = self.autofocus {
            for (axis, v) in ["x", "y"].into_iter().zip(uv) {
                if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                    return Err(invalid(format!(
                        "`autofocus[{axis}]` must be finite and in [0, 1], got {v}"
                    )));
                }
            }
        }
        Ok(())
    }
}

fn finite(field: &str, v: f32) -> Result<(), RaaError> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(invalid(format!("`{field}` must be finite, got {v}")))
    }
}

fn invalid(message: String) -> RaaError {
    RaaError::FrameGraph {
        code: POST_INVALID,
        message,
    }
}

/// Signed circle of confusion in pixels at `viewport_h`.
///
/// Thin lens with a 36 mm sensor mapped onto the viewport height:
///
/// ```text
/// A      = f / N                              aperture diameter (m)
/// coc_mm = A * f * (z - s) / (z * (s - f))    signed CoC on the sensor (mm)
/// px     = coc_mm / 36 * viewport_h
/// ```
///
/// with `f` the focal length in metres, `N` the f-number, `s` the focus distance and `z`
/// the depth of the sample in metres. The result is negative in front of the focus plane
/// (the near layer) and positive behind it (the far layer), and is clamped to
/// `+-max_coc_px * viewport_h / 720`. A non-finite or non-positive `z`, or a description
/// the formula cannot use (focus not beyond the focal point), returns 0.
pub fn coc_px(z: f32, d: &DofDesc, viewport_h: f32) -> f32 {
    if !z.is_finite() || z <= 0.0 {
        return 0.0;
    }
    if !viewport_h.is_finite() || viewport_h <= 0.0 {
        return 0.0;
    }
    if !(d.focus_distance_m.is_finite()
        && d.f_stop.is_finite()
        && d.focal_length_mm.is_finite()
        && d.max_coc_px.is_finite())
    {
        return 0.0;
    }

    let f_mm = f64::from(d.focal_length_mm);
    let f = f_mm / 1000.0;
    let n = f64::from(d.f_stop);
    let s = f64::from(d.focus_distance_m);
    let z = f64::from(z);
    if f <= 0.0 || n <= 0.0 || s <= f {
        return 0.0;
    }

    let aperture = f / n;
    let coc_mm = aperture * f * (z - s) / (z * (s - f)) * 1000.0;
    let px = coc_mm / f64::from(SENSOR_WIDTH_MM) * f64::from(viewport_h);
    let limit = f64::from(d.max_coc_px) * f64::from(viewport_h) / f64::from(REFERENCE_VIEWPORT_H);
    px.clamp(-limit, limit) as f32
}
