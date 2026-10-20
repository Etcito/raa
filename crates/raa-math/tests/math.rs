//! Integration tests for raa-math through its public API.

use std::f32::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, SQRT_2};

use raa_math::{Aabb, Mat4, Quat, Vec3, Vec4, approx_eq};

const EPS: f32 = 1e-5;

fn assert_vec3_near(a: Vec3, b: Vec3, eps: f32) {
    assert!(
        approx_eq(a.x, b.x, eps) && approx_eq(a.y, b.y, eps) && approx_eq(a.z, b.z, eps),
        "{a:?} != {b:?} (eps {eps})"
    );
}

fn assert_mat4_near(a: &Mat4, b: &Mat4, eps: f32) {
    for (x, y) in a.to_cols_array().iter().zip(b.to_cols_array()) {
        assert!(approx_eq(*x, y, eps), "{a:?} != {b:?} (eps {eps})");
    }
}

fn sample_trs() -> [(Vec3, Quat, Vec3); 3] {
    [
        (
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_rotation_y(FRAC_PI_3),
            Vec3::new(2.0, 2.0, 2.0),
        ),
        (
            Vec3::new(-4.0, 0.5, 7.0),
            Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0), 1.1),
            Vec3::new(0.5, 3.0, 1.5),
        ),
        (
            Vec3::new(0.0, -10.0, 0.25),
            Quat::from_axis_angle(Vec3::new(-0.3, 0.2, 0.9), -2.4),
            Vec3::new(1.0, 0.25, 4.0),
        ),
    ]
}

#[test]
fn trs_applies_scale_then_rotation_then_translation() {
    let p = Vec3::new(0.3, -1.2, 2.5);
    for (t, r, s) in sample_trs() {
        let m = Mat4::from_trs(t, r, s);
        let expected = r.rotate(p * s) + t;
        assert_vec3_near(m.transform_point(p), expected, EPS);
        let composed = Mat4::from_translation(t) * Mat4::from_quat(r) * Mat4::from_scale(s);
        assert_mat4_near(&m, &composed, EPS);
    }
}

#[test]
fn inverse_round_trips_on_three_trs_matrices() {
    for (t, r, s) in sample_trs() {
        let m = Mat4::from_trs(t, r, s);
        let inv = m.inverse().expect("TRS with non-zero scale is invertible");
        assert_mat4_near(&(m * inv), &Mat4::IDENTITY, 1e-5);
        assert_mat4_near(&(inv * m), &Mat4::IDENTITY, 1e-5);
        let p = Vec3::new(1.5, -0.5, 2.0);
        assert_vec3_near(inv.transform_point(m.transform_point(p)), p, 1e-4);
    }
}

#[test]
fn quaternion_rotate_matches_matrix_on_five_vectors() {
    let q = Quat::from_axis_angle(Vec3::new(0.2, -0.7, 0.4), 0.9);
    let m = Mat4::from_quat(q);
    let vectors = [
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(-0.5, 4.0, -2.25),
    ];
    for v in vectors {
        assert_vec3_near(q.rotate(v), m.transform_vector(v), EPS);
    }
}

#[test]
fn perspective_maps_near_to_zero_and_far_to_one() {
    let (near, far) = (0.1, 100.0);
    let proj = Mat4::perspective_rh_zo(FRAC_PI_4, 16.0 / 9.0, near, far);
    let depth = |z: f32| {
        let c = proj.mul_vec4(Vec4::new(0.0, 0.0, z, 1.0));
        assert!(
            c.w > 0.0,
            "point in front of the camera has positive clip w"
        );
        c.z / c.w
    };
    assert!(approx_eq(depth(-near), 0.0, 1e-6));
    assert!(approx_eq(depth(-far), 1.0, 1e-6));
    let mid = depth(-10.0);
    assert!(mid > 0.0 && mid < 1.0);
}

#[test]
fn look_at_forward_is_negative_z() {
    let eye = Vec3::new(3.0, 2.0, 5.0);
    let target = Vec3::new(-1.0, 0.5, -2.0);
    let view = Mat4::look_at_rh(eye, target, Vec3::Y);
    assert_vec3_near(view.transform_point(eye), Vec3::ZERO, EPS);
    let forward = (target - eye).normalize_or_zero();
    assert_vec3_near(view.transform_vector(forward), -Vec3::Z, EPS);
    let in_front = view.transform_point(eye + forward * 2.0);
    assert!(
        in_front.z < 0.0,
        "point in front of the eye has view z {}",
        in_front.z
    );
    assert_vec3_near(in_front, Vec3::new(0.0, 0.0, -2.0), 1e-4);
    // Up stays up: a point above the eye has positive view y.
    assert!(view.transform_point(eye + Vec3::Y).y > 0.0);
}

#[test]
fn aabb_transform_of_unit_box_rotated_45_degrees_about_y() {
    let unit = Aabb::from_points(&[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]);
    let rotated = unit.transform(&Mat4::from_quat(Quat::from_rotation_y(FRAC_PI_4)));
    let h = SQRT_2 * 0.5;
    assert_vec3_near(rotated.min, Vec3::new(-h, -0.5, -h), EPS);
    assert_vec3_near(rotated.max, Vec3::new(h, 0.5, h), EPS);
    // A quarter turn maps the unit box onto itself.
    let quarter = unit.transform(&Mat4::from_quat(Quat::from_rotation_y(FRAC_PI_2)));
    assert_vec3_near(quarter.min, unit.min, EPS);
    assert_vec3_near(quarter.max, unit.max, EPS);
}
