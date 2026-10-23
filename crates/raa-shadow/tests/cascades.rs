//! Cascade split and snapping tests (CPU only).

use raa_math::{Mat4, Vec3, Vec4};
use raa_shadow::{cascade_matrix, cascade_splits};

const FOV: f32 = 0.9;
const ASPECT: f32 = 16.0 / 9.0;
const RES: u32 = 2048;

fn sun() -> Vec3 {
    Vec3::new(-0.4, -0.8, -0.3)
}

fn camera(eye: Vec3, yaw: f32) -> Mat4 {
    let forward = Vec3::new(-yaw.sin(), -0.25, -yaw.cos());
    Mat4::look_at_rh(eye, eye + forward, Vec3::Y)
}

/// Upper 3x3 of the matrix (rotation x scale); it only depends on the sun and the extent.
fn linear_part(m: &Mat4) -> [u32; 9] {
    let mut out = [0; 9];
    for c in 0..3 {
        for r in 0..3 {
            out[c * 3 + r] = m.cols[c][r].to_bits();
        }
    }
    out
}

/// Half extent in metres (the ortho x scale is 1 / r and the light view is a rotation).
fn extent(m: &Mat4) -> f32 {
    let row0 = Vec3::new(m.cols[0][0], m.cols[1][0], m.cols[2][0]);
    1.0 / row0.length()
}

/// World origin projected to shadow-map texel coordinates (x, y).
fn origin_texels(m: &Mat4) -> (f32, f32) {
    let p = m.mul_vec4(Vec4::new(0.0, 0.0, 0.0, 1.0));
    let half = RES as f32 * 0.5;
    (p.x * half, p.y * half)
}

#[test]
fn splits_are_monotonic() {
    for lambda in [0.0, 0.75, 1.0] {
        for (near, far) in [(0.1, 60.0), (0.5, 20.0), (0.01, 500.0), (1.0, 2.0)] {
            for n in 1..=4 {
                let s = cascade_splits(near, far, lambda, n);
                assert!(
                    s[0] > near,
                    "lambda {lambda} near {near} far {far} n {n}: {s:?}"
                );
                for i in 1..n {
                    assert!(
                        s[i] > s[i - 1],
                        "lambda {lambda} near {near} far {far} n {n}: {s:?}"
                    );
                }
                for v in &s[n..] {
                    assert_eq!(*v, far, "unused entries equal far: {s:?}");
                }
            }
        }
    }
}

#[test]
fn splits_cover_near_to_distance() {
    let s = cascade_splits(0.1, 60.0, 0.75, 4);
    assert!(s[0] > 0.1, "{s:?}");
    assert_eq!(s[3], 60.0, "{s:?}");
    // n is clamped to 1..=4 and the last used entry is far exactly.
    assert_eq!(cascade_splits(0.1, 60.0, 0.75, 0), [60.0; 4]);
    assert_eq!(cascade_splits(0.1, 60.0, 0.75, 9), s);
    let two = cascade_splits(0.1, 60.0, 0.75, 2);
    assert!(two[0] > 0.1 && two[0] < 60.0, "{two:?}");
    assert_eq!(&two[1..], &[60.0; 3]);
}

#[test]
fn snapping_is_stable_under_sub_texel_moves() {
    let splits = cascade_splits(0.1, 60.0, 0.75, 4);
    let eye = Vec3::new(3.0, 2.0, 5.0);
    let mut near = 0.1;
    for far in splits {
        let base = cascade_matrix(&camera(eye, 0.3), FOV, ASPECT, (near, far), sun(), RES);
        let texel = 2.0 * extent(&base) / RES as f32;
        let (bx, by) = origin_texels(&base);
        for off in [
            Vec3::new(0.3, 0.0, 0.0),
            Vec3::new(0.0, 0.45, 0.0),
            Vec3::new(0.0, 0.0, -0.7),
            Vec3::new(0.2, -0.35, 0.5),
            Vec3::new(-0.55, 0.1, -0.25),
        ] {
            let moved_eye = eye + off * texel;
            let m = cascade_matrix(
                &camera(moved_eye, 0.3),
                FOV,
                ASPECT,
                (near, far),
                sun(),
                RES,
            );
            assert_eq!(
                linear_part(&m),
                linear_part(&base),
                "split ({near}, {far}) offset {off:?}: scale changed"
            );
            let (x, y) = origin_texels(&m);
            for d in [x - bx, y - by] {
                assert!(
                    (d - d.round()).abs() < 1e-3,
                    "split ({near}, {far}) offset {off:?}: origin moved {d} texels"
                );
            }
        }
        near = far;
    }
}

#[test]
fn extent_is_rotation_invariant() {
    let splits = cascade_splits(0.1, 60.0, 0.75, 4);
    let eye = Vec3::new(-1.0, 1.7, 4.0);
    let mut near = 0.1;
    for far in splits {
        let base = cascade_matrix(&camera(eye, 0.0), FOV, ASPECT, (near, far), sun(), RES);
        let r = extent(&base);
        assert!(r > 0.0 && r.is_finite(), "extent {r}");
        // The radius is a multiple of 1/16 m.
        let steps = r * 16.0;
        assert!((steps - steps.round()).abs() < 1e-3, "extent {r}");
        for yaw in [0.1, 0.77, 1.6, 2.9, -2.2, 4.0] {
            let m = cascade_matrix(&camera(eye, yaw), FOV, ASPECT, (near, far), sun(), RES);
            assert_eq!(
                linear_part(&m),
                linear_part(&base),
                "split ({near}, {far}) yaw {yaw}: extent {} vs {r}",
                extent(&m)
            );
        }
        near = far;
    }
}

#[test]
fn degenerate_inputs_do_not_panic() {
    let view = camera(Vec3::new(0.0, 2.0, 5.0), 0.4);
    let finite = |m: &Mat4| m.to_cols_array().iter().all(|v| v.is_finite());

    let vertical = cascade_matrix(
        &view,
        FOV,
        ASPECT,
        (0.1, 8.0),
        Vec3::new(0.0, -1.0, 0.0),
        RES,
    );
    assert!(finite(&vertical), "vertical sun: {vertical:?}");
    assert_ne!(vertical, Mat4::IDENTITY);
    let up = cascade_matrix(
        &view,
        FOV,
        ASPECT,
        (0.1, 8.0),
        Vec3::new(0.0, 1.0, 0.0),
        RES,
    );
    assert!(finite(&up), "upward sun: {up:?}");

    let zero = cascade_matrix(&view, FOV, ASPECT, (0.1, 8.0), Vec3::ZERO, RES);
    assert!(finite(&zero), "zero sun_dir: {zero:?}");

    let singular = Mat4::from_scale(Vec3::new(1.0, 0.0, 1.0));
    assert_eq!(
        cascade_matrix(&singular, FOV, ASPECT, (0.1, 8.0), sun(), RES),
        Mat4::IDENTITY
    );
    assert_eq!(
        cascade_matrix(&Mat4::default(), FOV, ASPECT, (0.1, 8.0), sun(), RES),
        Mat4::IDENTITY
    );

    // Empty slice, zero resolution and non-finite fov stay finite or fall back to identity.
    let empty = cascade_matrix(&view, FOV, ASPECT, (3.0, 3.0), sun(), RES);
    assert!(finite(&empty), "empty slice: {empty:?}");
    let no_res = cascade_matrix(&view, FOV, ASPECT, (0.1, 8.0), sun(), 0);
    assert!(finite(&no_res), "zero resolution: {no_res:?}");
    let nan_fov = cascade_matrix(&view, f32::NAN, ASPECT, (0.1, 8.0), sun(), RES);
    assert!(finite(&nan_fov), "nan fov: {nan_fov:?}");

    let s = cascade_splits(0.0, 60.0, 0.75, 4);
    assert_eq!(s[3], 60.0);
    assert!(s.iter().all(|v| v.is_finite()), "{s:?}");
}
