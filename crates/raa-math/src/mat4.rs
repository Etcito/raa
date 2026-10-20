//! `Mat4`: column-major 4x4 matrix, column vectors (`M * v`).

use core::ops::Mul;

use crate::{Quat, Vec3, Vec4};

/// Determinants below this magnitude are treated as singular.
const SINGULAR_DET: f32 = 1e-12;

/// A 4x4 matrix stored as 4 columns; `cols[c][r]` is row `r` of column `c`.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Mat4 {
    pub cols: [[f32; 4]; 4],
}

impl Mat4 {
    pub const IDENTITY: Mat4 = Mat4 {
        cols: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    pub const fn from_cols(c0: Vec4, c1: Vec4, c2: Vec4, c3: Vec4) -> Mat4 {
        Mat4 {
            cols: [c0.to_array(), c1.to_array(), c2.to_array(), c3.to_array()],
        }
    }

    pub const fn from_translation(t: Vec3) -> Mat4 {
        let mut m = Mat4::IDENTITY;
        m.cols[3] = [t.x, t.y, t.z, 1.0];
        m
    }

    pub const fn from_scale(s: Vec3) -> Mat4 {
        let mut m = Mat4::IDENTITY;
        m.cols[0][0] = s.x;
        m.cols[1][1] = s.y;
        m.cols[2][2] = s.z;
        m
    }

    /// Rotation matrix of a unit quaternion.
    pub fn from_quat(q: Quat) -> Mat4 {
        let (x, y, z, w) = (q.x, q.y, q.z, q.w);
        let (x2, y2, z2) = (x + x, y + y, z + z);
        let (xx, yy, zz) = (x * x2, y * y2, z * z2);
        let (xy, xz, yz) = (x * y2, x * z2, y * z2);
        let (wx, wy, wz) = (w * x2, w * y2, w * z2);
        Mat4 {
            cols: [
                [1.0 - (yy + zz), xy + wz, xz - wy, 0.0],
                [xy - wz, 1.0 - (xx + zz), yz + wx, 0.0],
                [xz + wy, yz - wx, 1.0 - (xx + yy), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// `T * R * S`: scale first, then rotate, then translate.
    pub fn from_trs(t: Vec3, r: Quat, s: Vec3) -> Mat4 {
        let mut m = Mat4::from_quat(r);
        for (col, k) in m.cols.iter_mut().zip([s.x, s.y, s.z]) {
            for v in col.iter_mut().take(3) {
                *v *= k;
            }
        }
        m.cols[3] = [t.x, t.y, t.z, 1.0];
        m
    }

    /// Matrix product `self * rhs` (`rhs` is applied first).
    // Named in the raa-math contract; `impl Mul` below forwards to it.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(&self, rhs: &Mat4) -> Mat4 {
        let mut out = [[0.0; 4]; 4];
        for (o, c) in out.iter_mut().zip(rhs.cols) {
            *o = self.mul_vec4(Vec4::new(c[0], c[1], c[2], c[3])).to_array();
        }
        Mat4 { cols: out }
    }

    pub fn mul_vec4(&self, v: Vec4) -> Vec4 {
        let [c0, c1, c2, c3] = self.cols;
        let row = |r: usize| c0[r] * v.x + c1[r] * v.y + c2[r] * v.z + c3[r] * v.w;
        Vec4::new(row(0), row(1), row(2), row(3))
    }

    /// Transforms a point (w = 1); divides by the resulting w when it is neither 1 nor 0.
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        let r = self.mul_vec4(Vec4::from_vec3(p, 1.0));
        if r.w != 1.0 && r.w != 0.0 {
            r.xyz() / r.w
        } else {
            r.xyz()
        }
    }

    /// Transforms a direction (w = 0): translation is ignored.
    pub fn transform_vector(&self, v: Vec3) -> Vec3 {
        self.mul_vec4(Vec4::from_vec3(v, 0.0)).xyz()
    }

    pub fn transpose(&self) -> Mat4 {
        let mut out = [[0.0; 4]; 4];
        for (c, col) in self.cols.iter().enumerate() {
            for (r, v) in col.iter().enumerate() {
                out[r][c] = *v;
            }
        }
        Mat4 { cols: out }
    }

    /// The 2x2 sub-determinants used by `determinant` and `inverse`.
    /// `a(i, j)` reads `cols[i][j]`; the formulas hold for either storage order
    /// because inverse and transpose commute.
    fn minors(&self) -> ([f32; 6], [f32; 6]) {
        let a = |i: usize, j: usize| self.cols[i][j];
        let s = [
            a(0, 0) * a(1, 1) - a(1, 0) * a(0, 1),
            a(0, 0) * a(1, 2) - a(1, 0) * a(0, 2),
            a(0, 0) * a(1, 3) - a(1, 0) * a(0, 3),
            a(0, 1) * a(1, 2) - a(1, 1) * a(0, 2),
            a(0, 1) * a(1, 3) - a(1, 1) * a(0, 3),
            a(0, 2) * a(1, 3) - a(1, 2) * a(0, 3),
        ];
        let c = [
            a(2, 0) * a(3, 1) - a(3, 0) * a(2, 1),
            a(2, 0) * a(3, 2) - a(3, 0) * a(2, 2),
            a(2, 0) * a(3, 3) - a(3, 0) * a(2, 3),
            a(2, 1) * a(3, 2) - a(3, 1) * a(2, 2),
            a(2, 1) * a(3, 3) - a(3, 1) * a(2, 3),
            a(2, 2) * a(3, 3) - a(3, 2) * a(2, 3),
        ];
        (s, c)
    }

    pub fn determinant(&self) -> f32 {
        let (s, c) = self.minors();
        s[0] * c[5] - s[1] * c[4] + s[2] * c[3] + s[3] * c[2] - s[4] * c[1] + s[5] * c[0]
    }

    /// Inverse by cofactor expansion; `None` when `abs(det) < 1e-12` (or det is not finite).
    pub fn inverse(&self) -> Option<Mat4> {
        let (s, c) = self.minors();
        let det = s[0] * c[5] - s[1] * c[4] + s[2] * c[3] + s[3] * c[2] - s[4] * c[1] + s[5] * c[0];
        if det.abs() < SINGULAR_DET || !det.is_finite() {
            return None;
        }
        let inv = 1.0 / det;
        let a = |i: usize, j: usize| self.cols[i][j];
        let b = [
            [
                a(1, 1) * c[5] - a(1, 2) * c[4] + a(1, 3) * c[3],
                -a(0, 1) * c[5] + a(0, 2) * c[4] - a(0, 3) * c[3],
                a(3, 1) * s[5] - a(3, 2) * s[4] + a(3, 3) * s[3],
                -a(2, 1) * s[5] + a(2, 2) * s[4] - a(2, 3) * s[3],
            ],
            [
                -a(1, 0) * c[5] + a(1, 2) * c[2] - a(1, 3) * c[1],
                a(0, 0) * c[5] - a(0, 2) * c[2] + a(0, 3) * c[1],
                -a(3, 0) * s[5] + a(3, 2) * s[2] - a(3, 3) * s[1],
                a(2, 0) * s[5] - a(2, 2) * s[2] + a(2, 3) * s[1],
            ],
            [
                a(1, 0) * c[4] - a(1, 1) * c[2] + a(1, 3) * c[0],
                -a(0, 0) * c[4] + a(0, 1) * c[2] - a(0, 3) * c[0],
                a(3, 0) * s[4] - a(3, 1) * s[2] + a(3, 3) * s[0],
                -a(2, 0) * s[4] + a(2, 1) * s[2] - a(2, 3) * s[0],
            ],
            [
                -a(1, 0) * c[3] + a(1, 1) * c[1] - a(1, 2) * c[0],
                a(0, 0) * c[3] - a(0, 1) * c[1] + a(0, 2) * c[0],
                -a(3, 0) * s[3] + a(3, 1) * s[1] - a(3, 2) * s[0],
                a(2, 0) * s[3] - a(2, 1) * s[1] + a(2, 2) * s[0],
            ],
        ];
        let mut out = [[0.0; 4]; 4];
        for (o, row) in out.iter_mut().zip(b) {
            for (v, x) in o.iter_mut().zip(row) {
                *v = x * inv;
            }
        }
        Some(Mat4 { cols: out })
    }

    /// Inverse-transpose of the upper 3x3 as 3 columns with w = 0 (for normals).
    /// Identity columns when the 3x3 is singular (`abs(det) < 1e-12`).
    pub fn normal_matrix(&self) -> [[f32; 4]; 3] {
        let col = |i: usize| Vec3::new(self.cols[i][0], self.cols[i][1], self.cols[i][2]);
        let (c0, c1, c2) = (col(0), col(1), col(2));
        // Rows of inverse(M) are the cross products / det, so they are the columns of its transpose.
        let (r0, r1, r2) = (c1.cross(c2), c2.cross(c0), c0.cross(c1));
        let det = c0.dot(r0);
        if det.abs() < SINGULAR_DET || !det.is_finite() {
            return [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ];
        }
        let inv = 1.0 / det;
        [r0, r1, r2].map(|r| Vec4::from_vec3(r * inv, 0.0).to_array())
    }

    /// Right-handed perspective projection with clip depth [0, 1] (WebGPU).
    /// A view-space point at `z = -near` maps to depth 0 and one at `z = -far` to depth 1.
    pub fn perspective_rh_zo(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let f = 1.0 / (0.5 * fov_y_radians).tan();
        let r = far / (near - far);
        Mat4 {
            cols: [
                [f / aspect, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, r, -1.0],
                [0.0, 0.0, r * near, 0.0],
            ],
        }
    }

    /// Right-handed view matrix: `eye` goes to the origin and `target` onto the -Z axis.
    pub fn look_at_rh(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
        let f = (target - eye).normalize_or_zero();
        let s = f.cross(up).normalize_or_zero();
        let u = s.cross(f);
        Mat4 {
            cols: [
                [s.x, u.x, -f.x, 0.0],
                [s.y, u.y, -f.y, 0.0],
                [s.z, u.z, -f.z, 0.0],
                [-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0],
            ],
        }
    }

    /// The 16 floats in column-major order.
    pub fn to_cols_array(&self) -> [f32; 16] {
        let mut out = [0.0; 16];
        for (o, v) in out.iter_mut().zip(self.cols.iter().flatten()) {
            *o = *v;
        }
        out
    }

    /// Writes the 16 floats in column order as little-endian bytes into `out[..64]`.
    ///
    /// Invariant: `out.len() >= 64`. A shorter slice is a caller bug (debug-asserted);
    /// in release only the bytes that fit are written.
    pub fn write_le_bytes(&self, out: &mut [u8]) {
        debug_assert!(out.len() >= 64, "Mat4::write_le_bytes needs 64 bytes");
        for (chunk, v) in out.chunks_mut(4).zip(self.cols.iter().flatten()) {
            let bytes = v.to_le_bytes();
            let n = chunk.len();
            chunk.copy_from_slice(&bytes[..n]);
        }
    }
}

impl Mul for Mat4 {
    type Output = Mat4;
    fn mul(self, rhs: Mat4) -> Mat4 {
        Mat4::mul(&self, &rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approx_eq;

    fn mat_close(a: &Mat4, b: &Mat4, eps: f32) -> bool {
        a.cols
            .iter()
            .flatten()
            .zip(b.cols.iter().flatten())
            .all(|(x, y)| approx_eq(*x, *y, eps))
    }

    fn vec_close(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x, 1e-5) && approx_eq(a.y, b.y, 1e-5) && approx_eq(a.z, b.z, 1e-5)
    }

    fn sample_trs() -> (Vec3, Quat, Vec3) {
        (
            Vec3::new(1.5, -2.0, 3.25),
            Quat::from_axis_angle(Vec3::new(0.3, 1.0, -0.6), 1.1),
            Vec3::new(2.0, 0.5, 3.0),
        )
    }

    #[test]
    fn from_trs_equals_product_of_translation_rotation_scale() {
        let (t, r, s) = sample_trs();
        let product = Mat4::from_translation(t) * Mat4::from_quat(r) * Mat4::from_scale(s);
        assert!(mat_close(&Mat4::from_trs(t, r, s), &product, 1e-6));
        assert_eq!(Mat4::IDENTITY * product, product);
    }

    #[test]
    fn translation_moves_points_but_not_vectors() {
        let m = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let p = Vec3::new(-4.0, 0.5, 2.0);
        assert_eq!(m.transform_point(p), Vec3::new(-3.0, 2.5, 5.0));
        assert_eq!(m.transform_vector(p), p);
    }

    #[test]
    fn transform_point_divides_by_w_only_when_needed() {
        let mut m = Mat4::IDENTITY;
        m.cols[3][3] = 2.0;
        assert_eq!(
            m.transform_point(Vec3::new(2.0, 4.0, 6.0)),
            Vec3::new(1.0, 2.0, 3.0)
        );
        m.cols[3][3] = 0.0;
        assert_eq!(
            m.transform_point(Vec3::new(2.0, 4.0, 6.0)),
            Vec3::new(2.0, 4.0, 6.0)
        );
    }

    #[test]
    fn inverse_times_matrix_is_identity_for_non_uniform_trs() {
        let (t, r, s) = sample_trs();
        let m = Mat4::from_trs(t, r, s);
        let inv = m.inverse().expect("TRS with non-zero scale is invertible");
        assert!(mat_close(&(inv * m), &Mat4::IDENTITY, 1e-5));
        assert!(mat_close(&(m * inv), &Mat4::IDENTITY, 1e-5));
        assert!(approx_eq(m.determinant() * inv.determinant(), 1.0, 1e-4));
    }

    #[test]
    fn singular_matrix_has_no_inverse() {
        assert_eq!(Mat4::from_scale(Vec3::new(1.0, 0.0, 2.0)).inverse(), None);
        let mut m = Mat4::IDENTITY;
        m.cols[1] = m.cols[0];
        assert_eq!(m.inverse(), None);
    }

    #[test]
    fn determinant_of_scale_is_product_of_factors() {
        assert_eq!(
            Mat4::from_scale(Vec3::new(2.0, 3.0, 4.0)).determinant(),
            24.0
        );
        assert_eq!(Mat4::IDENTITY.determinant(), 1.0);
    }

    #[test]
    fn transpose_swaps_rows_and_columns() {
        let (t, r, s) = sample_trs();
        let m = Mat4::from_trs(t, r, s);
        let mt = m.transpose();
        assert_eq!(mt.cols[3][0], m.cols[0][3]);
        assert_eq!(mt.cols[0][3], t.x);
        assert_eq!(mt.transpose(), m);
    }

    #[test]
    fn from_quat_agrees_with_quat_rotate() {
        let q = Quat::from_axis_angle(Vec3::new(-0.4, 0.9, 0.2), 2.3);
        let m = Mat4::from_quat(q);
        for v in [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.3, -2.0, 1.5),
            Vec3::new(-1.0, 4.0, -0.25),
        ] {
            assert!(vec_close(m.transform_vector(v), q.rotate(v)));
            assert!(vec_close(m.transform_point(v), q.rotate(v)));
        }
    }

    #[test]
    fn normal_matrix_of_non_uniform_scale_is_inverse_scale() {
        let n = Mat4::from_scale(Vec3::new(2.0, 4.0, 0.5)).normal_matrix();
        let expected = [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.25, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
        ];
        for (col, exp) in n.iter().zip(expected) {
            for (x, y) in col.iter().zip(exp) {
                assert!(approx_eq(*x, y, 1e-6));
            }
        }
    }

    #[test]
    fn normal_matrix_ignores_translation_and_falls_back_when_singular() {
        let (t, r, s) = sample_trs();
        let m = Mat4::from_trs(t, r, s);
        let n = m.normal_matrix();
        let it = m.inverse().expect("invertible").transpose();
        for (col, exp) in n.iter().zip(it.cols) {
            assert_eq!(col[3], 0.0);
            for (x, y) in col.iter().zip(exp).take(3) {
                assert!(approx_eq(*x, y, 1e-5));
            }
        }
        let flat = Mat4::from_scale(Vec3::new(1.0, 1.0, 0.0)).normal_matrix();
        assert_eq!(
            flat,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]
            ]
        );
    }
    #[test]
    fn perspective_maps_near_to_zero_and_far_to_one() {
        let (near, far) = (0.1, 100.0);
        let p = Mat4::perspective_rh_zo(1.0, 16.0 / 9.0, near, far);
        let depth = |z: f32| {
            let c = p.mul_vec4(Vec4::new(0.0, 0.0, z, 1.0));
            c.z / c.w
        };
        assert!(approx_eq(depth(-near), 0.0, 1e-6));
        assert!(approx_eq(depth(-far), 1.0, 1e-6));
    }

    #[test]
    fn look_at_puts_target_on_negative_z() {
        let v = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        assert!(vec_close(
            v.transform_point(Vec3::ZERO),
            Vec3::new(0.0, 0.0, -5.0)
        ));
        assert!(vec_close(
            v.transform_point(Vec3::new(0.0, 0.0, 5.0)),
            Vec3::ZERO
        ));
        assert!(vec_close(v.transform_vector(Vec3::Y), Vec3::Y));
    }

    #[test]
    fn to_cols_array_round_trips_through_from_cols() {
        let (t, r, s) = sample_trs();
        let m = Mat4::from_trs(t, r, s);
        let a = m.to_cols_array();
        let col = |i: usize| Vec4::new(a[4 * i], a[4 * i + 1], a[4 * i + 2], a[4 * i + 3]);
        assert_eq!(Mat4::from_cols(col(0), col(1), col(2), col(3)), m);
    }

    #[test]
    fn write_le_bytes_writes_columns_in_order() {
        let (t, r, s) = sample_trs();
        let m = Mat4::from_trs(t, r, s);
        let mut out = [0u8; 72];
        m.write_le_bytes(&mut out);
        assert_eq!(out[0..4], m.cols[0][0].to_le_bytes());
        assert_eq!(out[48..52], t.x.to_le_bytes());
        assert_eq!(out[60..64], 1.0f32.to_le_bytes());
        assert_eq!(out[64..], [0u8; 8]);
    }
}
