//! `Vec3` and `Vec4`.

use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// A 3-component f32 vector.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3::new(0.0, 0.0, 0.0);
    pub const ONE: Vec3 = Vec3::new(1.0, 1.0, 1.0);
    pub const X: Vec3 = Vec3::new(1.0, 0.0, 0.0);
    pub const Y: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    pub const Z: Vec3 = Vec3::new(0.0, 0.0, 1.0);

    pub const fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }

    pub const fn splat(v: f32) -> Vec3 {
        Vec3::new(v, v, v)
    }

    pub fn dot(self, rhs: Vec3) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Right-handed cross product: `X.cross(Y) == Z`.
    pub fn cross(self, rhs: Vec3) -> Vec3 {
        Vec3::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Unit vector in the same direction, or `ZERO` when the length is zero or not finite.
    pub fn normalize_or_zero(self) -> Vec3 {
        let len = self.length();
        if len > 0.0 && len.is_finite() {
            self / len
        } else {
            Vec3::ZERO
        }
    }

    pub fn min(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x.min(rhs.x), self.y.min(rhs.y), self.z.min(rhs.z))
    }

    pub fn max(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x.max(rhs.x), self.y.max(rhs.y), self.z.max(rhs.z))
    }

    pub fn abs(self) -> Vec3 {
        Vec3::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    /// `a + (b - a) * t`.
    pub fn lerp(a: Vec3, b: Vec3, t: f32) -> Vec3 {
        a + (b - a) * t
    }

    pub const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    pub const fn from_array(a: [f32; 3]) -> Vec3 {
        Vec3::new(a[0], a[1], a[2])
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: f32) -> Vec3 {
        Vec3::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, rhs: f32) -> Vec3 {
        Vec3::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

/// Component-wise product.
impl Mul<Vec3> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Vec3) {
        *self = *self + rhs;
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Vec3) {
        *self = *self - rhs;
    }
}

/// A 4-component f32 vector.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
        Vec4 { x, y, z, w }
    }

    pub const fn from_vec3(v: Vec3, w: f32) -> Vec4 {
        Vec4::new(v.x, v.y, v.z, w)
    }

    pub const fn xyz(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    pub fn dot(self, rhs: Vec4) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }

    pub const fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }
}

impl Add for Vec4 {
    type Output = Vec4;
    fn add(self, rhs: Vec4) -> Vec4 {
        Vec4::new(
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
            self.w + rhs.w,
        )
    }
}

impl Sub for Vec4 {
    type Output = Vec4;
    fn sub(self, rhs: Vec4) -> Vec4 {
        Vec4::new(
            self.x - rhs.x,
            self.y - rhs.y,
            self.z - rhs.z,
            self.w - rhs.w,
        )
    }
}

impl Mul<f32> for Vec4 {
    type Output = Vec4;
    fn mul(self, rhs: f32) -> Vec4 {
        Vec4::new(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approx_eq;

    #[test]
    fn cross_of_x_and_y_is_z() {
        assert_eq!(Vec3::X.cross(Vec3::Y), Vec3::Z);
        assert_eq!(Vec3::Y.cross(Vec3::X), -Vec3::Z);
    }

    #[test]
    fn dot_sums_component_products() {
        assert_eq!(
            Vec3::new(1.0, 2.0, 3.0).dot(Vec3::new(4.0, -5.0, 6.0)),
            12.0
        );
        assert_eq!(Vec3::X.dot(Vec3::Y), 0.0);
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(a.dot(a), 30.0);
    }

    #[test]
    fn normalize_of_3_4_0_is_unit_and_scaled() {
        let n = Vec3::new(3.0, 4.0, 0.0).normalize_or_zero();
        assert!(approx_eq(n.x, 0.6, 1e-6));
        assert!(approx_eq(n.y, 0.8, 1e-6));
        assert_eq!(n.z, 0.0);
        assert!(approx_eq(n.length(), 1.0, 1e-6));
        assert_eq!(Vec3::new(3.0, 4.0, 0.0).length_squared(), 25.0);
    }

    #[test]
    fn normalize_or_zero_of_zero_or_non_finite_is_zero() {
        assert_eq!(Vec3::ZERO.normalize_or_zero(), Vec3::ZERO);
        assert_eq!(
            Vec3::new(f32::NAN, 0.0, 0.0).normalize_or_zero(),
            Vec3::ZERO
        );
        assert_eq!(Vec3::splat(f32::INFINITY).normalize_or_zero(), Vec3::ZERO);
    }

    #[test]
    fn min_max_abs_are_component_wise() {
        let a = Vec3::new(1.0, -5.0, 3.0);
        let b = Vec3::new(-2.0, 4.0, 3.5);
        assert_eq!(a.min(b), Vec3::new(-2.0, -5.0, 3.0));
        assert_eq!(a.max(b), Vec3::new(1.0, 4.0, 3.5));
        assert_eq!(a.abs(), Vec3::new(1.0, 5.0, 3.0));
    }

    #[test]
    fn lerp_hits_endpoints_and_midpoint() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(3.0, -2.0, 7.0);
        assert_eq!(Vec3::lerp(a, b, 0.0), a);
        assert_eq!(Vec3::lerp(a, b, 0.5), Vec3::new(2.0, 0.0, 5.0));
        assert_eq!(Vec3::lerp(a, b, 1.0), b);
    }

    #[test]
    fn array_round_trip_preserves_components() {
        let v = Vec3::new(1.5, -2.25, 3.0);
        assert_eq!(v.to_array(), [1.5, -2.25, 3.0]);
        assert_eq!(Vec3::from_array(v.to_array()), v);
        let w = Vec4::from_vec3(v, 2.0);
        assert_eq!(w.to_array(), [1.5, -2.25, 3.0, 2.0]);
        assert_eq!(w.xyz(), v);
    }

    #[test]
    fn is_finite_rejects_nan_and_infinity() {
        assert!(Vec3::ONE.is_finite());
        assert!(!Vec3::new(0.0, f32::NAN, 0.0).is_finite());
        assert!(!Vec3::new(0.0, 0.0, f32::NEG_INFINITY).is_finite());
    }

    #[test]
    fn operators_are_component_wise() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a + b, Vec3::new(5.0, 7.0, 9.0));
        assert_eq!(b - a, Vec3::splat(3.0));
        assert_eq!(a * 2.0, Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(b / 2.0, Vec3::new(2.0, 2.5, 3.0));
        assert_eq!(a * b, Vec3::new(4.0, 10.0, 18.0));
        let mut c = a;
        c += b;
        assert_eq!(c, a + b);
        c -= b;
        assert_eq!(c, a);
        let p = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let q = Vec4::new(0.5, 0.5, 0.5, 0.5);
        assert_eq!(p + q, Vec4::new(1.5, 2.5, 3.5, 4.5));
        assert_eq!(p - q, Vec4::new(0.5, 1.5, 2.5, 3.5));
        assert_eq!(p * 2.0, Vec4::new(2.0, 4.0, 6.0, 8.0));
        assert!(approx_eq(1.0, 1.0 + 1e-7, 1e-6));
        assert!(!approx_eq(1.0, 1.1, 1e-3));
    }
}
