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

// ---- Scene cover (no sun without shadow inside the scene) ----

use raa_math::Aabb;
use raa_shadow::{CascadeFit, cascade_frame, cascade_frame_covered, cover_distance};

/// An atrium-like scene: 40 x 16 x 18 m.
fn atrium() -> Aabb {
    Aabb {
        min: Vec3::new(-20.0, -1.0, -9.0),
        max: Vec3::new(20.0, 15.0, 9.0),
    }
}

fn corners(b: &Aabb) -> [Vec3; 8] {
    let mut out = [Vec3::ZERO; 8];
    for (k, p) in out.iter_mut().enumerate() {
        *p = Vec3::new(
            if k & 1 == 0 { b.min.x } else { b.max.x },
            if k & 2 == 0 { b.min.y } else { b.max.y },
            if k & 4 == 0 { b.min.z } else { b.max.z },
        );
    }
    out
}

#[test]
fn covered_frame_without_scene_is_the_plain_frame() {
    let view = camera(Vec3::new(-9.0, 1.6, -0.9), -1.5);
    let (m, s) = cascade_frame(&view, FOV, ASPECT, 0.1, 37.2, 20.0, sun(), RES);
    let c = cascade_frame_covered(&view, FOV, ASPECT, 0.1, 37.2, 20.0, sun(), RES, None);
    assert_eq!(c.splits, s);
    assert_eq!(c.matrices, m);
    assert_eq!(c.fits, [CascadeFit::Slice; 4]);
    assert_eq!(c.cover_m, None);
}

#[test]
fn covered_frame_keeps_the_near_cascades_and_reaches_every_scene_corner() {
    let scene = atrium();
    for (eye, yaw, far) in [
        (Vec3::new(-9.0, 1.6, -0.9), -1.5, 37.2),
        (Vec3::new(12.5, 1.8, -0.6), 1.5, 60.0),
        (Vec3::new(0.0, 7.0, 5.0), 0.7, 1000.0),
    ] {
        let view = camera(eye, yaw);
        let (m, s) = cascade_frame(&view, FOV, ASPECT, 0.1, far, 20.0, sun(), RES);
        let c = cascade_frame_covered(&view, FOV, ASPECT, 0.1, far, 20.0, sun(), RES, Some(&scene));
        // Near quality: the first three cascades are the plain ones.
        assert_eq!(&c.splits[..3], &s[..3]);
        assert_eq!(&c.matrices[..3], &m[..3]);
        // The last split reaches the farthest scene corner (capped by far).
        let farthest = corners(&scene)
            .iter()
            .map(|p| (*p - eye).length())
            .fold(0.0f32, f32::max);
        assert!(c.splits[3] >= farthest.min(far) - 1e-3, "{c:?} {farthest}");
        // Every scene point the camera can see past split 2 is inside the last cascade box.
        let last = c.matrices[3];
        for p in corners(&scene) {
            let q = last.mul_vec4(Vec4::new(p.x, p.y, p.z, 1.0));
            assert!(q.x.abs() <= 1.0 && q.y.abs() <= 1.0, "{p:?} -> {q:?}");
            assert!(q.z > 0.0 && q.z < 1.0, "{p:?} -> {q:?}");
        }
        assert_eq!(c.fits[3], CascadeFit::Scene, "{c:?}");
    }
}

#[test]
fn scene_fitted_cascade_does_not_move_with_the_camera() {
    let scene = atrium();
    let a = cascade_frame_covered(
        &camera(Vec3::new(-9.0, 1.6, -0.9), -1.5),
        FOV,
        ASPECT,
        0.1,
        60.0,
        20.0,
        sun(),
        RES,
        Some(&scene),
    );
    let b = cascade_frame_covered(
        &camera(Vec3::new(-7.3, 1.9, 0.4), -1.1),
        FOV,
        ASPECT,
        0.1,
        60.0,
        20.0,
        sun(),
        RES,
        Some(&scene),
    );
    assert_eq!(a.fits[3], CascadeFit::Scene);
    assert_eq!(a.matrices[3], b.matrices[3]);
}

#[test]
fn huge_scene_keeps_the_slice_sphere() {
    let scene = Aabb {
        min: Vec3::splat(-5000.0),
        max: Vec3::splat(5000.0),
    };
    let view = camera(Vec3::new(0.0, 2.0, 0.0), 0.3);
    let c = cascade_frame_covered(
        &view,
        FOV,
        ASPECT,
        0.1,
        300.0,
        20.0,
        sun(),
        RES,
        Some(&scene),
    );
    assert_eq!(c.fits, [CascadeFit::Slice; 4]);
    assert_eq!(c.splits[3], 300.0);
    assert!(c.texels_m[3] > c.texels_m[2]);
}

#[test]
fn cover_distance_is_laddered() {
    let view = camera(Vec3::new(-9.0, 1.6, -0.9), -1.5);
    let d = cover_distance(&view, &atrium()).unwrap();
    let k = (d.log2() * 4.0).round();
    assert!((d.log2() * 4.0 - k).abs() < 1e-4, "{d}");
    // A small move keeps the same value.
    let e = cover_distance(&camera(Vec3::new(-9.1, 1.6, -0.9), -1.5), &atrium()).unwrap();
    assert_eq!(d, e);
}
