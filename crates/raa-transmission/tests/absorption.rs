use raa_transmission::absorption::*;

const N: [f32; 3] = [0.0, 1.0, 0.0];
const DOWN: [f32; 3] = [0.0, -1.0, 0.0];

fn close(a: [f32; 3], b: [f32; 3]) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() < 1e-5)
}

#[test]
fn half_red_attenuation_over_unit_path() {
    let t = transmittance_for([0.5, 1.0, 1.0], 1.0, 1.0, DOWN, N);
    assert!(close(t, [0.5, 1.0, 1.0]), "{t:?}");
}

#[test]
fn doubling_thickness_squares_transmittance() {
    let c = [0.3, 0.6, 0.9];
    let t1 = transmittance_for(c, 0.7, 0.4, DOWN, N);
    let t2 = transmittance_for(c, 0.7, 0.8, DOWN, N);
    assert!(close(t2, t1.map(|x| x * x)), "{t1:?} {t2:?}");
}

#[test]
fn grazing_angle_bounded_by_clamp() {
    let grazing = [1.0, -0.001, 0.0];
    assert!((path_length(1.0, grazing, N) - 5.0).abs() < 1e-5);
    let t = transmittance_for([0.5, 1.0, 1.0], 1.0, 1.0, grazing, N);
    assert!((t[0] - 0.5f32.powi(5)).abs() < 1e-5, "{t:?}");
}

#[test]
fn infinite_distance_gives_full_transmittance() {
    let t = transmittance_for([0.1, 0.2, 0.0], NO_ABSORPTION_DISTANCE, 3.0, DOWN, N);
    assert_eq!(t, [1.0, 1.0, 1.0]);
}

#[test]
fn zero_component_is_clamped() {
    let s = sigma([0.0, 1.0, 1.0], 1.0);
    assert!(s[0].is_finite() && s[0] > 0.0);
}

#[test]
fn ior_one_leaves_direction_unchanged() {
    let v = [0.6f32, 0.8, 0.0];
    let d = refract_view(v, N, 1.0);
    assert!(close(d, [-0.6, -0.8, 0.0]), "{d:?}");
}

#[test]
fn total_internal_reflection_reflects() {
    // Inside a dense medium, grazing ray: eta = 1.5.
    let i = [0.9f32, -0.43589, 0.0];
    let d = refract(i, N, 1.5);
    assert!(close(d, [0.9, 0.43589, 0.0]), "{d:?}");
}

#[test]
fn lod_zero_for_smooth_and_grows_with_roughness() {
    assert_eq!(transmission_lod(0.0, 720.0, 0.5, 3.0), 0.0);
    let a = transmission_lod(0.3, 720.0, 0.5, 3.0);
    let b = transmission_lod(0.6, 720.0, 0.5, 3.0);
    assert!(a > 0.0 && b > a, "{a} {b}");
}

#[test]
fn schlick_endpoints() {
    assert!((schlick_fresnel(0.04, 1.0) - 0.04).abs() < 1e-6);
    assert!((schlick_fresnel(0.04, 0.0) - 1.0).abs() < 1e-6);
}
