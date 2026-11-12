//! Private f64 vector helpers of the cluster build: the arithmetic that the weld, simplify
//! and hierarchy stages share in double precision, so no stage rounds in f32 twice.
//! The crate's own code: this module has no solstice source.

use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::iter::Sum;

/// A 2-component f64 vector.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct DVec2 {
    pub x: f64,
    pub y: f64,
}

impl DVec2 {
    pub const fn new(x: f64, y: f64) -> DVec2 {
        DVec2 { x, y }
    }

    pub fn from_f32(v: [f32; 2]) -> DVec2 {
        DVec2::new(f64::from(v[0]), f64::from(v[1]))
    }

    pub fn dot(self, o: DVec2) -> f64 {
        self.x * o.x + self.y * o.y
    }

    /// The signed area of the parallelogram: positive when `o` is counter-clockwise of `self`.
    pub fn perp_dot(self, o: DVec2) -> f64 {
        self.x * o.y - self.y * o.x
    }

    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn distance(self, o: DVec2) -> f64 {
        (self - o).length()
    }
}

impl Add for DVec2 {
    type Output = DVec2;

    fn add(self, rhs: DVec2) -> DVec2 {
        DVec2::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Sub for DVec2 {
    type Output = DVec2;

    fn sub(self, rhs: DVec2) -> DVec2 {
        DVec2::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Mul<f64> for DVec2 {
    type Output = DVec2;

    fn mul(self, rhs: f64) -> DVec2 {
        DVec2::new(self.x * rhs, self.y * rhs)
    }
}

/// A 3-component f64 vector.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct DVec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl DVec3 {
    pub const ZERO: DVec3 = DVec3::new(0.0, 0.0, 0.0);
    pub const X: DVec3 = DVec3::new(1.0, 0.0, 0.0);
    pub const Y: DVec3 = DVec3::new(0.0, 1.0, 0.0);

    pub const fn new(x: f64, y: f64, z: f64) -> DVec3 {
        DVec3 { x, y, z }
    }

    pub fn from_vec3(v: raa_math::Vec3) -> DVec3 {
        DVec3::new(f64::from(v.x), f64::from(v.y), f64::from(v.z))
    }

    pub fn to_vec3(self) -> raa_math::Vec3 {
        raa_math::Vec3::new(self.x as f32, self.y as f32, self.z as f32)
    }

    pub fn dot(self, o: DVec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: DVec3) -> DVec3 {
        DVec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }

    pub fn distance(self, o: DVec3) -> f64 {
        (self - o).length()
    }

    pub fn distance_squared(self, o: DVec3) -> f64 {
        (self - o).length_squared()
    }

    /// Unit vector in the direction of `self`, or `fallback` when the length is zero or not finite.
    pub fn normalize_or(self, fallback: DVec3) -> DVec3 {
        let len = self.length();
        if len > 0.0 && len.is_finite() {
            self / len
        } else {
            fallback
        }
    }

    pub fn normalize_or_zero(self) -> DVec3 {
        self.normalize_or(DVec3::ZERO)
    }

    /// A unit vector orthogonal to `self`, deterministic for the same input.
    ///
    /// Branchless orthonormal basis of Duff et al. 2017 on the normalized input, so the
    /// result is unit and orthogonal even for unnormalized arguments.
    pub fn any_orthonormal_vector(self) -> DVec3 {
        let n = self.normalize_or(DVec3::X);
        let sign = 1.0_f64.copysign(n.z);
        let a = -1.0 / (sign + n.z);
        let b = n.x * n.y * a;
        DVec3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x)
    }
}

impl Add for DVec3 {
    type Output = DVec3;

    fn add(self, rhs: DVec3) -> DVec3 {
        DVec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for DVec3 {
    type Output = DVec3;

    fn sub(self, rhs: DVec3) -> DVec3 {
        DVec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Neg for DVec3 {
    type Output = DVec3;

    fn neg(self) -> DVec3 {
        DVec3::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f64> for DVec3 {
    type Output = DVec3;

    fn mul(self, rhs: f64) -> DVec3 {
        DVec3::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Div<f64> for DVec3 {
    type Output = DVec3;

    fn div(self, rhs: f64) -> DVec3 {
        DVec3::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl AddAssign for DVec3 {
    fn add_assign(&mut self, rhs: DVec3) {
        *self = *self + rhs;
    }
}

impl Sum for DVec3 {
    fn sum<I: Iterator<Item = DVec3>>(iter: I) -> DVec3 {
        iter.fold(DVec3::ZERO, |acc, v| acc + v)
    }
}

/// A 3x3 matrix of f64, stored as three column vectors.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct DMat3 {
    pub cols: [DVec3; 3],
}

impl DMat3 {
    pub const ZERO: DMat3 = DMat3 {
        cols: [DVec3::ZERO, DVec3::ZERO, DVec3::ZERO],
    };

    /// Adds the outer product `d d^T` to the matrix.
    pub fn add_outer(&mut self, d: DVec3) {
        self.cols[0] += d * d.x;
        self.cols[1] += d * d.y;
        self.cols[2] += d * d.z;
    }

    pub fn mul_vec(&self, v: DVec3) -> DVec3 {
        self.cols[0] * v.x + self.cols[1] * v.y + self.cols[2] * v.z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= EPS
    }

    fn close_vec(a: DVec3, b: DVec3) -> bool {
        close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z)
    }

    #[test]
    fn dot_and_cross_identities() {
        let x = DVec3::X;
        let y = DVec3::Y;
        let z = DVec3::new(0.0, 0.0, 1.0);
        assert!(close(x.dot(x), 1.0));
        assert!(close(x.dot(y), 0.0));
        assert!(close_vec(x.cross(y), z));
        assert!(close_vec(y.cross(x), -z));
        assert!(close(x.dot(y.cross(z)), 1.0));
        assert!(close(x.dot(x.cross(y)), 0.0));

        let a = DVec3::new(1.0, 2.0, 3.0);
        let b = DVec3::new(-4.0, 5.0, 0.5);
        assert!(close(a.dot(b), b.dot(a)));
        assert!(close_vec(a.cross(b), -b.cross(a)));
        assert!(close(a.dot(a.cross(b)), 0.0));
        assert!(close(b.dot(a.cross(b)), 0.0));
        assert!(close(
            a.cross(b).length_squared(),
            a.length_squared() * b.length_squared() - a.dot(b) * a.dot(b)
        ));
        assert!(close_vec(a + b - b, a));
        assert!(close_vec(a * 2.0, a + a));
        assert!(close_vec(a / 2.0 + a / 2.0, a));
    }

    #[test]
    fn normalize_or_uses_the_fallback_for_a_zero_vector() {
        let fallback = DVec3::new(0.0, 1.0, 0.0);
        assert_eq!(DVec3::ZERO.normalize_or(fallback), fallback);
        assert_eq!(DVec3::ZERO.normalize_or_zero(), DVec3::ZERO);
        let unit = DVec3::new(3.0, 0.0, 4.0).normalize_or(fallback);
        assert!(close(unit.length(), 1.0));
        assert!(close_vec(unit, DVec3::new(0.6, 0.0, 0.8)));
    }

    #[test]
    fn any_orthonormal_vector_is_unit_and_orthogonal() {
        for v in [
            DVec3::X,
            DVec3::Y,
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(1.0, 2.0, 3.0),
        ] {
            let o = v.any_orthonormal_vector();
            assert!(close(o.length(), 1.0), "not unit for {v:?}: {o:?}");
            assert!(close(o.dot(v), 0.0), "not orthogonal for {v:?}: {o:?}");
            assert_eq!(o, v.any_orthonormal_vector(), "not deterministic for {v:?}");
        }
    }

    #[test]
    fn perp_dot_signs_and_distance() {
        let x = DVec2::new(1.0, 0.0);
        let y = DVec2::new(0.0, 1.0);
        assert!(close(x.perp_dot(y), 1.0));
        assert!(close(y.perp_dot(x), -1.0));
        assert!(close(x.perp_dot(x), 0.0));
        assert!(close(x.dot(y), 0.0));
        assert!(close(DVec2::from_f32([3.0, 4.0]).length(), 5.0));
        assert!(close(
            DVec2::new(1.0, 1.0).distance(DVec2::new(4.0, 5.0)),
            5.0
        ));
        assert!(close_vec(
            DVec3::new(1.0, 2.0, 3.0).cross(DVec3::new(2.0, 4.0, 6.0)),
            DVec3::ZERO
        ));
        assert!(close(
            DVec3::new(1.0, 2.0, 3.0).distance_squared(DVec3::new(1.0, 2.0, 6.0)),
            9.0
        ));
    }

    #[test]
    fn add_outer_then_mul_vec_equals_d_times_dot() {
        let mut m = DMat3::ZERO;
        let d = DVec3::new(0.5, -1.5, 2.0);
        let v = DVec3::new(3.0, -2.0, 1.0);
        m.add_outer(d);
        assert!(close_vec(m.mul_vec(v), d * d.dot(v)));
        assert_eq!(m.mul_vec(DVec3::ZERO), DVec3::ZERO);

        let mut twice = DMat3::ZERO;
        twice.add_outer(d);
        twice.add_outer(d);
        assert!(close_vec(twice.mul_vec(v), d * (2.0 * d.dot(v))));
    }

    #[test]
    fn vec3_conversion_round_trips() {
        let v = raa_math::Vec3::new(1.5, -2.25, 3.75);
        let d = DVec3::from_vec3(v);
        assert_eq!(d, DVec3::new(1.5, -2.25, 3.75));
        assert_eq!(d.to_vec3(), v);
        assert_eq!(DVec3::from_vec3(raa_math::Vec3::ZERO), DVec3::ZERO);

        let sum: DVec3 = [DVec3::X, DVec3::Y, DVec3::new(0.0, 0.0, 1.0)]
            .into_iter()
            .sum();
        assert_eq!(sum, DVec3::new(1.0, 1.0, 1.0));
    }
}
