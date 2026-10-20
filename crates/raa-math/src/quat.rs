//! `Quat`: unit quaternion rotations, stored `[x, y, z, w]`.

use core::ops::Mul;

use crate::Vec3;

/// A quaternion `x i + y j + z k + w`. Rotations use unit quaternions.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    pub const IDENTITY: Quat = Quat::from_xyzw(0.0, 0.0, 0.0, 1.0);

    pub const fn from_xyzw(x: f32, y: f32, z: f32, w: f32) -> Quat {
        Quat { x, y, z, w }
    }

    pub const fn from_array(a: [f32; 4]) -> Quat {
        Quat::from_xyzw(a[0], a[1], a[2], a[3])
    }

    pub const fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// Rotation of `radians` around `axis` (normalised here). A zero axis gives IDENTITY.
    pub fn from_axis_angle(axis: Vec3, radians: f32) -> Quat {
        let axis = axis.normalize_or_zero();
        if axis == Vec3::ZERO {
            return Quat::IDENTITY;
        }
        let (s, c) = (radians * 0.5).sin_cos();
        Quat::from_xyzw(axis.x * s, axis.y * s, axis.z * s, c)
    }

    pub fn from_rotation_y(radians: f32) -> Quat {
        let (s, c) = (radians * 0.5).sin_cos();
        Quat::from_xyzw(0.0, s, 0.0, c)
    }

    /// Hamilton product: `self.mul(rhs)` applies `rhs` first, then `self`.
    // Named in the raa-math contract; `impl Mul` below forwards to it.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, rhs: Quat) -> Quat {
        let (a, b) = (self, rhs);
        Quat::from_xyzw(
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        )
    }

    /// Rotates `v` by this (unit) quaternion: v + 2w(q x v) + 2 q x (q x v).
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let q = Vec3::new(self.x, self.y, self.z);
        let t = q.cross(v) * 2.0;
        v + t * self.w + q.cross(t)
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    /// Unit-length copy; IDENTITY for a zero-length (or non-finite) quaternion.
    pub fn normalize(self) -> Quat {
        let len = self.length();
        if len == 0.0 || !len.is_finite() {
            return Quat::IDENTITY;
        }
        let inv = 1.0 / len;
        Quat::from_xyzw(self.x * inv, self.y * inv, self.z * inv, self.w * inv)
    }

    /// Inverse rotation for a unit quaternion.
    pub fn conjugate(self) -> Quat {
        Quat::from_xyzw(-self.x, -self.y, -self.z, self.w)
    }

    fn dot(self, rhs: Quat) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }

    /// Spherical interpolation along the shortest path; normalised lerp when nearly parallel.
    pub fn slerp(a: Quat, b: Quat, t: f32) -> Quat {
        let mut b = b;
        let mut d = a.dot(b);
        if d < 0.0 {
            b = Quat::from_xyzw(-b.x, -b.y, -b.z, -b.w);
            d = -d;
        }
        let (wa, wb) = if d > 0.9995 {
            (1.0 - t, t)
        } else {
            let theta = d.acos();
            let sin_theta = theta.sin();
            (
                ((1.0 - t) * theta).sin() / sin_theta,
                (t * theta).sin() / sin_theta,
            )
        };
        Quat::from_xyzw(
            a.x * wa + b.x * wb,
            a.y * wa + b.y * wb,
            a.z * wa + b.z * wb,
            a.w * wa + b.w * wb,
        )
        .normalize()
    }
}

impl Mul for Quat {
    type Output = Quat;
    fn mul(self, rhs: Quat) -> Quat {
        Quat::mul(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approx_eq;
    use core::f32::consts::FRAC_PI_2;

    fn vec_close(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x, 1e-5) && approx_eq(a.y, b.y, 1e-5) && approx_eq(a.z, b.z, 1e-5)
    }

    fn quat_close(a: Quat, b: Quat) -> bool {
        a.to_array()
            .iter()
            .zip(b.to_array())
            .all(|(x, y)| approx_eq(*x, y, 1e-5))
    }

    #[test]
    fn rotation_y_90_degrees_turns_x_into_negative_z() {
        let q = Quat::from_rotation_y(FRAC_PI_2);
        assert!(vec_close(q.rotate(Vec3::X), -Vec3::Z));
        assert!(vec_close(q.rotate(Vec3::Z), Vec3::X));
        assert!(vec_close(q.rotate(Vec3::Y), Vec3::Y));
    }

    #[test]
    fn axis_angle_around_y_matches_rotation_y() {
        for a in [0.0, 0.3, FRAC_PI_2, 2.5, -1.2] {
            assert!(quat_close(
                Quat::from_axis_angle(Vec3::Y * 3.0, a),
                Quat::from_rotation_y(a)
            ));
        }
        assert_eq!(Quat::from_axis_angle(Vec3::ZERO, 1.0), Quat::IDENTITY);
    }

    #[test]
    fn rotations_compose_under_mul() {
        let a = Quat::from_axis_angle(Vec3::new(1.0, 2.0, -0.5), 0.7);
        let b = Quat::from_axis_angle(Vec3::new(-0.3, 0.1, 1.0), 1.9);
        let v = Vec3::new(0.4, -1.5, 2.0);
        assert!(vec_close((a * b).rotate(v), a.rotate(b.rotate(v))));
        assert_eq!(a * b, a.mul(b));
        assert!(approx_eq((a * b).length(), 1.0, 1e-5));
    }

    #[test]
    fn conjugate_undoes_a_rotation() {
        let q = Quat::from_axis_angle(Vec3::new(0.2, -1.0, 0.7), 2.2);
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert!(vec_close(q.conjugate().rotate(q.rotate(v)), v));
        assert!(quat_close(q * q.conjugate(), Quat::IDENTITY));
    }

    #[test]
    fn normalize_scales_to_unit_and_zero_gives_identity() {
        let q = Quat::from_xyzw(0.0, 3.0, 0.0, 4.0).normalize();
        assert!(quat_close(q, Quat::from_xyzw(0.0, 0.6, 0.0, 0.8)));
        assert_eq!(Quat::default().normalize(), Quat::IDENTITY);
        let a = [0.1, 0.2, 0.3, 0.4];
        assert_eq!(Quat::from_array(a).to_array(), a);
    }

    #[test]
    fn slerp_hits_endpoints_and_midpoint_is_unit() {
        let a = Quat::from_axis_angle(Vec3::X, 0.4);
        let b = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 1.0), 2.0);
        assert!(quat_close(Quat::slerp(a, b, 0.0), a));
        assert!(quat_close(Quat::slerp(a, b, 1.0), b));
        assert!(approx_eq(Quat::slerp(a, b, 0.5).length(), 1.0, 1e-5));
        // Shortest path: -b is the same rotation, so the endpoint rotates like b.
        let neg_b = Quat::from_xyzw(-b.x, -b.y, -b.z, -b.w);
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert!(vec_close(Quat::slerp(a, neg_b, 1.0).rotate(v), b.rotate(v)));
        // Nearly parallel inputs take the normalised-lerp branch.
        let c = Quat::from_rotation_y(0.001);
        let m = Quat::slerp(Quat::IDENTITY, c, 0.5);
        assert!(quat_close(m, Quat::from_rotation_y(0.0005)));
    }
}
