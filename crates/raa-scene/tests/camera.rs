use raa_math::{Vec3, Vec4, approx_eq};
use raa_scene::{Camera, OrbitController};

const EPS: f32 = 1e-5;

fn assert_vec3_near(a: Vec3, b: Vec3, eps: f32) {
    assert!(
        approx_eq(a.x, b.x, eps) && approx_eq(a.y, b.y, eps) && approx_eq(a.z, b.z, eps),
        "{a:?} != {b:?}"
    );
}

#[test]
fn default_camera_maps_target_to_clip_centre_with_depth_in_range() {
    let cam = Camera::default();
    let clip = cam.view_proj().mul_vec4(Vec4::from_vec3(cam.target, 1.0));
    assert!(clip.w > 0.0);
    let ndc = clip.xyz() / clip.w;
    assert!(approx_eq(ndc.x, 0.0, EPS), "x = {}", ndc.x);
    assert!(approx_eq(ndc.y, 0.0, EPS), "y = {}", ndc.y);
    assert!(ndc.z > 0.0 && ndc.z < 1.0, "depth = {}", ndc.z);
}

#[test]
fn set_viewport_sets_aspect_and_never_divides_by_zero() {
    let mut cam = Camera::default();
    cam.set_viewport(1280, 720);
    assert!(approx_eq(cam.aspect, 16.0 / 9.0, EPS));
    cam.set_viewport(100, 0);
    assert!(cam.aspect.is_finite());
    assert!(approx_eq(cam.aspect, 100.0, EPS));
    cam.set_viewport(0, 50);
    assert!(approx_eq(cam.aspect, 1.0 / 50.0, EPS));
}

#[test]
fn orbit_with_zero_angles_puts_eye_on_positive_z() {
    let target = Vec3::new(1.0, 2.0, 3.0);
    let orbit = OrbitController::new(target, 5.0);
    let mut cam = Camera::default();
    orbit.apply(&mut cam);
    assert_vec3_near(cam.eye, target + Vec3::new(0.0, 0.0, 5.0), EPS);
    assert_eq!(cam.target, target);
    assert_eq!(cam.up, Vec3::Y);
}

#[test]
fn pitch_clamps_at_89_degrees() {
    let limit = 89.0_f32.to_radians();
    let mut orbit = OrbitController::new(Vec3::ZERO, 4.0);
    orbit.rotate(0.3, 10.0);
    assert!(approx_eq(orbit.pitch_rad, limit, EPS));
    assert!(approx_eq(orbit.yaw_rad, 0.3, EPS));
    orbit.rotate(0.0, -20.0);
    assert!(approx_eq(orbit.pitch_rad, -limit, EPS));
    let mut cam = Camera::default();
    orbit.apply(&mut cam);
    assert!(cam.eye.y < 0.0);
    assert!(cam.view().cols.iter().flatten().all(|v| v.is_finite()));
}

#[test]
fn zoom_clamps_and_ignores_bad_factors() {
    let mut orbit = OrbitController::new(Vec3::ZERO, 10.0);
    assert_eq!((orbit.min_distance, orbit.max_distance), (0.1, 1000.0));
    orbit.zoom(0.5);
    assert!(approx_eq(orbit.distance, 5.0, EPS));
    orbit.zoom(1e6);
    assert_eq!(orbit.distance, 1000.0);
    orbit.zoom(1e-9);
    assert_eq!(orbit.distance, 0.1);
    orbit.zoom(0.0);
    orbit.zoom(-2.0);
    orbit.zoom(f32::NAN);
    orbit.zoom(f32::INFINITY);
    assert_eq!(orbit.distance, 0.1);
    assert_eq!(OrbitController::new(Vec3::ZERO, 0.0).distance, 0.1);
    assert_eq!(OrbitController::new(Vec3::ZERO, 5000.0).distance, 1000.0);
}

#[test]
fn pan_moves_target_and_keeps_distance() {
    let mut orbit = OrbitController::new(Vec3::ZERO, 4.0);
    let mut cam = Camera::default();
    orbit.apply(&mut cam);
    let eye_before = cam.eye;
    orbit.pan(0.5, 0.25);
    // Yaw 0, pitch 0: right = +X, up = +Y, both scaled by the distance.
    assert_vec3_near(orbit.target, Vec3::new(2.0, 1.0, 0.0), EPS);
    assert_eq!(orbit.distance, 4.0);
    orbit.apply(&mut cam);
    assert!(approx_eq((cam.eye - cam.target).length(), 4.0, EPS));
    assert_vec3_near(cam.eye - eye_before, Vec3::new(2.0, 1.0, 0.0), EPS);

    // At a rotated orbit the target still moves in the camera plane.
    let mut orbit = OrbitController::new(Vec3::ZERO, 3.0);
    orbit.rotate(0.7, 0.4);
    orbit.apply(&mut cam);
    let view_dir = (cam.target - cam.eye).normalize_or_zero();
    orbit.pan(0.2, -0.3);
    let moved = orbit.target;
    assert!(approx_eq(moved.dot(view_dir), 0.0, EPS));
    assert!(approx_eq(
        moved.length(),
        3.0 * (0.2f32 * 0.2 + 0.3 * 0.3).sqrt(),
        EPS
    ));
    orbit.apply(&mut cam);
    assert!(approx_eq((cam.eye - cam.target).length(), 3.0, EPS));
}
