//! CPU twins of the squash-and-jiggle vertex functions (`squash-jiggle` provider).

use raa_math::Vec3;

const MIN_AMOUNT: f32 = -0.9;
const FALLBACK_AXIS: Vec3 = Vec3::Y;

fn clamp_amount(s: f32) -> f32 {
    if s.is_finite() {
        s.max(MIN_AMOUNT)
    } else {
        0.0
    }
}

fn unit_axis(axis: [f32; 3]) -> Vec3 {
    let a = Vec3::from_array(axis);
    if !a.is_finite() {
        return FALLBACK_AXIS;
    }
    let unit = a.normalize_or_zero();
    if unit == Vec3::ZERO {
        FALLBACK_AXIS
    } else {
        unit
    }
}

/// Squash `p` along `axis` by amount `s`, volume preserving: `p' = (p.a) a (1 + s) + (p - (p.a) a) / sqrt(1 + s)`.
pub fn squash_point(p: [f32; 3], axis: [f32; 3], s: f32) -> [f32; 3] {
    let a = unit_axis(axis);
    let s = clamp_amount(s);
    let p = Vec3::from_array(p);
    let parallel = a * p.dot(a);
    let perpendicular = p - parallel;
    (parallel * (1.0 + s) + perpendicular / (1.0 + s).sqrt()).to_array()
}

/// Inverse-transpose normal of [`squash_point`]: parallel part over `1 + s`, perpendicular part times `sqrt(1 + s)`, renormalised.
/// A zero or non-finite `n` yields `(0, 1, 0)`.
pub fn squash_normal(n: [f32; 3], axis: [f32; 3], s: f32) -> [f32; 3] {
    let a = unit_axis(axis);
    let s = clamp_amount(s);
    let n = Vec3::from_array(n).normalize_or_zero();
    if n == Vec3::ZERO {
        return FALLBACK_AXIS.to_array();
    }
    let parallel = a * n.dot(a);
    let perpendicular = n - parallel;
    let out = parallel / (1.0 + s) + perpendicular * (1.0 + s).sqrt();
    out.normalize_or_zero().to_array()
}

/// Jiggle weight `smoothstep(0, 1, clamp(y / h, 0, 1))^2`; zero when `h <= 0` (or NaN).
pub fn jiggle_offset_weight(y: f32, h: f32) -> f32 {
    if !(h > 0.0) {
        return 0.0;
    }
    let t = (y / h).clamp(0.0, 1.0);
    let smooth = t * t * (3.0 - 2.0 * t);
    smooth * smooth
}
