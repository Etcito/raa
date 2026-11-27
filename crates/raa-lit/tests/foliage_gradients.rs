//! Subject tests of differentiable foliage gradients (M7, contract `m7-compute`): the
//! 1-cosine normal loss, the shared sigmoid derivative, the squared-edge gradient, the
//! silhouette pair and the 80-byte five-lane gradient record, plus the composed shader
//! that includes the shared coverage helper exactly once. The module is path-compiled
//! until production registration; no test runs GPU work or builds the optimizer allocation.

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

use foliage_bins::dispatch_size;
use foliage_gradients::{
    GRADIENT_RECORD_STRIDE, GradientRecord, edge_gradient, edge_loss, gradients_wgsl, normal_loss,
    silhouette_gradient, silhouette_loss, soft_coverage_derivative,
};

#[test]
fn cosine_loss_is_zero_for_identical_and_one_for_orthogonal_normals() {
    assert_eq!(normal_loss([0.0, 0.0, 2.0], [0.0, 0.0, 5.0]), 0.0);
    assert_eq!(normal_loss([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), 1.0);
    let diagonal = 1.0 - 0.5f32.sqrt();
    assert!((normal_loss([1.0, 1.0, 0.0], [1.0, 0.0, 0.0]) - diagonal).abs() < 1e-6);
    assert_eq!(normal_loss([0.0; 3], [0.0, 1.0, 0.0]), 0.0);
}

#[test]
fn shared_sigmoid_derivative_is_a_quarter_at_temperature_one() {
    assert!((soft_coverage_derivative(0.0, 1.0) - 0.25).abs() < 1e-6);
    let h = 1e-4;
    let finite = (foliage_raster::soft_coverage(h, 1.0) - foliage_raster::soft_coverage(-h, 1.0))
        / (2.0 * h);
    assert!((finite - 0.25).abs() < 1e-3);
    assert!((soft_coverage_derivative(0.0, 2.0) - 0.125).abs() < 1e-6);
}

#[test]
fn squared_edge_gradient_is_twice_the_edge() {
    let a = [1.5, -2.0, 0.25];
    let b = [-0.5, 1.0, 2.0];
    let difference = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let (gradient_a, gradient_b) = edge_gradient(a, b);
    assert_eq!(
        gradient_a,
        [
            2.0 * difference[0],
            2.0 * difference[1],
            2.0 * difference[2]
        ]
    );
    assert_eq!(
        gradient_b,
        [
            -2.0 * difference[0],
            -2.0 * difference[1],
            -2.0 * difference[2]
        ]
    );
    assert!(
        (edge_loss(a, b)
            - (difference[0] * difference[0]
                + difference[1] * difference[1]
                + difference[2] * difference[2]))
            .abs()
            < 1e-6
    );
    let h = 1e-3;
    let moved = [a[0] + h, a[1], a[2]];
    let finite = (edge_loss(moved, b) - edge_loss(a, b)) / h;
    assert!((finite - gradient_a[0]).abs() < 1e-2);
}

#[test]
fn silhouette_loss_and_gradient_agree() {
    assert_eq!(silhouette_loss(0.75, 0.5), 0.03125);
    assert_eq!(silhouette_gradient(0.75, 0.5), 0.25);
    let h = 1e-4;
    let finite = (silhouette_loss(0.75 + h, 0.5) - silhouette_loss(0.75 - h, 0.5)) / (2.0 * h);
    assert!((finite - 0.25).abs() < 1e-3);
}

#[test]
fn gradient_record_writes_the_five_contract_lanes() {
    let record = GradientRecord {
        d_position: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        d_normal: [10.0, 11.0, 12.0],
        d_opacity: 13.0,
    };
    let bytes = record.to_bytes();
    assert_eq!(bytes.len(), GRADIENT_RECORD_STRIDE);
    let at = |lane: usize, component: usize| {
        let offset = 4 * (4 * lane + component);
        f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    };
    assert_eq!(
        [at(0, 0), at(0, 1), at(0, 2), at(0, 3)],
        [1.0, 2.0, 3.0, 0.0]
    );
    assert_eq!(
        [at(1, 0), at(1, 1), at(1, 2), at(1, 3)],
        [4.0, 5.0, 6.0, 0.0]
    );
    assert_eq!(
        [at(2, 0), at(2, 1), at(2, 2), at(2, 3)],
        [7.0, 8.0, 9.0, 0.0]
    );
    assert_eq!(
        [at(3, 0), at(3, 1), at(3, 2), at(3, 3)],
        [10.0, 11.0, 12.0, 13.0]
    );
    assert!(bytes[64..80].iter().all(|byte| *byte == 0));
    assert_eq!(
        GradientRecord::ZERO.to_bytes(),
        [0u8; GRADIENT_RECORD_STRIDE]
    );
}

#[test]
fn shader_composes_the_shared_helper_exactly_once() {
    let source = gradients_wgsl();
    assert_eq!(source.matches("fn foliage_soft_coverage").count(), 1);
    assert_eq!(source.matches("1 / (4 * temperature)").count(), 1);
    assert!(source.contains("@compute @workgroup_size(64)"));
    assert!(source.contains("fn gradient_triangles"));
    raa_shader::Composer::validate(&source).expect("the gradient shader validates");
}

#[test]
fn gradient_dispatch_is_one_workgroup_per_64_triangles() {
    assert_eq!([0, 1, 64, 65].map(dispatch_size), [0, 1, 1, 2]);
}
