//! Subject tests of the bounded differentiable foliage update (M7, contract `m7-compute`):
//! the hand-derived one-step quadratic descent, the bounded and finite step, zero-gradient
//! stability, the degenerate independent normal's defined fallback, the terminating opacity
//! anneal and the frozen terminal opacity with its finite-readback refusal. The module is
//! path-compiled until optimizer integration; no test runs GPU work or builds the optimizer
//! allocation.

#![cfg(not(target_arch = "wasm32"))]

#[allow(dead_code)]
#[path = "../src/foliage_bins.rs"]
mod foliage_bins;
#[allow(dead_code)]
#[path = "../src/foliage_gradients.rs"]
mod foliage_gradients;
#[allow(dead_code)]
#[path = "../src/foliage_raster.rs"]
mod foliage_raster;
#[allow(dead_code)]
#[allow(clippy::wrong_self_convention)]
#[path = "../src/foliage_resources.rs"]
mod foliage_resources;
#[allow(dead_code)]
#[path = "../src/foliage_update.rs"]
mod foliage_update;

use foliage_gradients::GradientRecord;
use foliage_resources::MicroTriangle;
use foliage_update::{
    FOLIAGE_UPDATE_WGSL, MAX_POSITION_STEP, OPACITY_ANNEAL_STEP, TRIANGLE_RECORD_STRIDE,
    anneal_opacity, bounded_step, freeze_terminal, resolve_normal, triangle_to_record,
    update_triangle,
};
use raa_core::RaaError;

fn unit_triangle() -> MicroTriangle {
    MicroTriangle {
        p0: [0.0, 0.0, 0.0],
        p1: [1.0, 0.0, 0.0],
        p2: [0.0, 1.0, 0.0],
        normal: [0.0, 0.0, 1.0],
        opacity: 1.0,
    }
}

fn bake_failed_code() -> &'static str {
    RaaError::FrameGraph {
        code: "foliage.bake_failed",
        message: String::new(),
    }
    .code()
}

#[test]
fn one_quadratic_step_matches_the_hand_derived_descent() {
    // L(p) = 0.5 |p - c|^2 has the gradient p - c. From x = 3 toward c = 1 with rate
    // 0.01 the step is -0.01 * 2 = -0.02, leaving 2.98 and the reduced loss 1.9602.
    let step = bounded_step([2.0, 0.0, 0.0], 0.01, MAX_POSITION_STEP);
    assert!((step[0] + 0.02).abs() < 1e-6, "step {step:?}");
    assert_eq!(step[1], 0.0);
    assert_eq!(step[2], 0.0);
    let mut triangle = unit_triangle();
    triangle.p0 = [3.0, 0.0, 0.0];
    triangle.p1 = [3.0, 1.0, 0.0];
    triangle.p2 = [3.0, 0.0, 1.0];
    let gradient = GradientRecord {
        d_position: [[2.0, 0.0, 0.0]; 3],
        d_normal: [0.0; 3],
        d_opacity: 0.0,
    };
    let updated = update_triangle(triangle, gradient, 0.01, 1.0);
    assert!((updated.p0[0] - 2.98).abs() < 1e-6, "p0 {updated:?}");
    assert!((updated.p1[0] - 2.98).abs() < 1e-6);
    assert!((updated.p2[0] - 2.98).abs() < 1e-6);
    let before = 0.5 * (3.0f32 - 1.0).powi(2);
    let after = 0.5 * (updated.p0[0] - 1.0).powi(2);
    assert!(after < before);
    // A step longer than the bound is scaled back to exactly the bound.
    let bounded = bounded_step([1.0, 0.0, 0.0], 1000.0, MAX_POSITION_STEP);
    let length =
        (bounded[0] * bounded[0] + bounded[1] * bounded[1] + bounded[2] * bounded[2]).sqrt();
    assert!((length - MAX_POSITION_STEP).abs() < 1e-6);
    assert_eq!(bounded_step([f32::NAN, 0.0, 0.0], 0.5, 1.0), [0.0; 3]);
    assert_eq!(bounded_step([1.0, 0.0, 0.0], f32::INFINITY, 1.0), [0.0; 3]);
}

#[test]
fn zero_gradient_is_stable() {
    let triangle = unit_triangle();
    let updated = update_triangle(triangle, GradientRecord::ZERO, 0.5, 1.0);
    assert_eq!(updated, triangle);
}

#[test]
fn degenerate_normals_take_the_defined_fallback() {
    let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    assert_eq!(
        resolve_normal([0.0; 3], [0.0; 3], positions),
        [0.0, 0.0, 1.0]
    );
    assert_eq!(
        resolve_normal([f32::NAN; 3], [0.0, 4.0, 0.0], positions),
        [0.0, 1.0, 0.0]
    );
    assert_eq!(
        resolve_normal([0.0; 3], [0.0; 3], [[2.0; 3]; 3]),
        [0.0, 1.0, 0.0]
    );
    let mut triangle = unit_triangle();
    triangle.normal = [0.0; 3];
    let updated = update_triangle(triangle, GradientRecord::ZERO, 0.5, 1.0);
    assert_eq!(updated.normal, [0.0, 0.0, 1.0]);
    let length = updated
        .normal
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!((length - 1.0).abs() < 1e-6);
}

#[test]
fn opacity_annealing_terminates_exactly_on_its_target() {
    let mut opacity = 0.0f32;
    for _ in 0..10 {
        opacity = anneal_opacity(opacity, 1.0);
    }
    assert_eq!(opacity, 1.0);
    assert_eq!(anneal_opacity(0.95, 1.0), 1.0);
    assert_eq!(anneal_opacity(f32::NAN, 1.0), 1.0);
    assert!((anneal_opacity(0.5, f32::NAN) - 0.6).abs() < 1e-6);
    let step = anneal_opacity(0.0, 1.0);
    assert!((step - OPACITY_ANNEAL_STEP).abs() < 1e-6);
}

#[test]
fn terminal_readback_is_finite_and_exactly_opaque() {
    let mut first = unit_triangle();
    first.opacity = 0.3;
    let mut second = unit_triangle();
    second.opacity = 0.0;
    second.normal = [0.0, 0.0, 2.0];
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&triangle_to_record(first));
    bytes.extend_from_slice(&triangle_to_record(second));
    assert_eq!(bytes.len(), 2 * TRIANGLE_RECORD_STRIDE);
    let terminal = freeze_terminal(&bytes, 2).expect("a finite readback freezes");
    assert_eq!(terminal.len(), 2);
    assert!(terminal.iter().all(|triangle| triangle.opacity == 1.0));
    assert_eq!(terminal[0].p0, first.p0);
    assert_eq!(terminal[1].normal, [0.0, 0.0, 1.0]);
    // A NaN or infinite value is a numerical failure, never a published triangle.
    let mut broken = triangle_to_record(first);
    broken[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
    let error = freeze_terminal(&broken, 1).expect_err("NaN must be refused");
    assert_eq!(error.code(), bake_failed_code());
    let short = freeze_terminal(&bytes[..TRIANGLE_RECORD_STRIDE - 4], 2)
        .expect_err("a short readback must be refused");
    assert_eq!(
        short.code(),
        RaaError::FrameGraph {
            code: "foliage.invalid",
            message: String::new(),
        }
        .code()
    );
}

#[test]
fn triangle_record_writes_the_five_contract_lanes() {
    let mut triangle = unit_triangle();
    triangle.opacity = 0.75;
    let bytes = triangle_to_record(triangle);
    let at = |lane: usize, component: usize| {
        let offset = 4 * (4 * lane + component);
        f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    };
    assert_eq!(
        [at(0, 0), at(0, 1), at(0, 2), at(0, 3)],
        [0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!([at(1, 0), at(1, 1), at(1, 3)], [1.0, 0.0, 0.0]);
    assert_eq!(
        [at(3, 0), at(3, 1), at(3, 2), at(3, 3)],
        [0.0, 0.0, 1.0, 0.75]
    );
    assert!(bytes[64..80].iter().all(|byte| *byte == 0));
}

#[test]
fn shader_validates_and_uses_the_update_layout() {
    raa_shader::Composer::validate(FOLIAGE_UPDATE_WGSL).expect("the update shader validates");
    assert!(FOLIAGE_UPDATE_WGSL.contains("@compute @workgroup_size(64)"));
    assert!(FOLIAGE_UPDATE_WGSL.contains("fn update_triangles"));
    assert!(FOLIAGE_UPDATE_WGSL.contains("@group(0) @binding(0) var<uniform> params"));
    assert!(
        FOLIAGE_UPDATE_WGSL.contains("@group(0) @binding(1) var<storage, read_write> triangles")
    );
    assert!(FOLIAGE_UPDATE_WGSL.contains("@group(0) @binding(2) var<storage, read> gradients"));
    assert!(!FOLIAGE_UPDATE_WGSL.contains("texture_2d"));
    assert!(!FOLIAGE_UPDATE_WGSL.contains("sampler"));
}
