//! Subject tests of the differentiable foliage raster (M7, contracts `m7-compute` and
//! `m7-soft-coverage`): the inward-signed pixel edge distance, the analytic soft coverage
//! at the edge and one known distance with its temperature derivative, the background
//! record, the normalized weighted normal and the reusable WGSL helper. The module is
//! path-compiled until production registration; no test runs GPU work.

#![cfg(not(target_arch = "wasm32"))]

#[allow(dead_code)]
#[path = "../src/foliage_bins.rs"]
mod foliage_bins;
#[allow(dead_code)]
#[path = "../src/foliage_raster.rs"]
mod foliage_raster;
#[allow(dead_code)]
#[allow(clippy::wrong_self_convention)]
#[path = "../src/foliage_resources.rs"]
mod foliage_resources;

use foliage_raster::{
    FOLIAGE_RASTER_WGSL, PIXEL_RECORD_STRIDE, RasterPixel, RasterSample, depth_visibility,
    resolve_pixel, signed_edge_distance, soft_coverage, soft_coverage_wgsl, triangle_soft_coverage,
};

#[test]
fn signed_edge_distance_is_zero_on_the_edge_and_signed_inside() {
    let a = [0.0, 0.0];
    let b = [4.0, 0.0];
    assert!(signed_edge_distance(a, b, [2.0, 0.0]).abs() < 1e-6);
    assert!((signed_edge_distance(a, b, [2.0, 1.5]) - 1.5).abs() < 1e-6);
    assert!((signed_edge_distance(a, b, [2.0, -0.25]) + 0.25).abs() < 1e-6);
}

#[test]
fn soft_coverage_is_half_at_the_edge_and_sigmoid_elsewhere() {
    let temperature = 0.75;
    assert!((soft_coverage(0.0, temperature) - 0.5).abs() < 1e-6);
    let sigmoid_one = 1.0 / (1.0 + (-1.0f32).exp());
    assert!((soft_coverage(temperature, temperature) - sigmoid_one).abs() < 1e-6);
    // A triangle whose three edge distances all sit on their zero crossing contributes
    // 0.5 per edge, 0.125 before its opacity.
    assert!((soft_coverage(0.0, temperature).powi(3) - 0.125).abs() < 1e-6);
    // The temperature derivative convention at the edge.
    let h = 1e-4;
    let derivative = (soft_coverage(h, temperature) - soft_coverage(-h, temperature)) / (2.0 * h);
    assert!((derivative - 1.0 / (4.0 * temperature)).abs() < 1e-3);
}

#[test]
fn triangle_coverage_multiplies_the_three_inward_edges() {
    let unit = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let on_edge = triangle_soft_coverage(unit, [0.5, 0.0], 1.0);
    assert!(on_edge > 0.0 && on_edge < 0.5);
    let inside = triangle_soft_coverage(unit, [0.25, 0.25], 1.0);
    assert!(
        inside > on_edge,
        "inside {inside} is not above the edge {on_edge}"
    );
    assert!(triangle_soft_coverage(unit, [10.0, 10.0], 0.25) < 1e-6);
    assert_eq!(triangle_soft_coverage([[0.0; 2]; 3], [0.0, 0.0], 1.0), 0.0);
}

#[test]
fn background_has_zero_coverage_and_a_defined_zero_normal() {
    let pixel = resolve_pixel(&[]);
    assert_eq!(pixel, RasterPixel::BACKGROUND);
    assert_eq!(pixel.normal, [0.0; 3]);
    assert_eq!(pixel.coverage, 0.0);
    assert_eq!(pixel.depth, 1.0);
    assert_eq!(depth_visibility(0.0, 0.0, 0.4, 1.0, 128), 0.0);
    let front = depth_visibility(1.0, 0.5, 0.25, 1.0, 128);
    let back = depth_visibility(1.0, 0.5, 0.75, 1.0, 128);
    assert!(front > 0.5, "a nearer surface {front} is not preferred");
    assert!(back < 0.5, "a farther surface {back} is not ordered behind");
}

#[test]
fn weighted_normals_are_normalized_into_the_32_byte_record() {
    let samples = [
        RasterSample {
            coverage: 0.5,
            visibility: 1.0,
            normal: [0.0, 1.0, 0.0],
            depth: 0.2,
        },
        RasterSample {
            coverage: 0.5,
            visibility: 1.0,
            normal: [0.0, 1.0, 0.0],
            depth: 0.4,
        },
    ];
    let pixel = resolve_pixel(&samples);
    assert!((pixel.coverage - 1.0).abs() < 1e-6);
    assert_eq!(pixel.normal, [0.0, 1.0, 0.0]);
    assert!((pixel.depth - 0.3).abs() < 1e-6);
    let bytes = pixel.to_bytes();
    assert_eq!(bytes.len(), PIXEL_RECORD_STRIDE);
    assert_eq!(f32::from_le_bytes(bytes[12..16].try_into().unwrap()), 1.0);
    let recorded = f32::from_le_bytes(bytes[16..20].try_into().unwrap());
    assert!((recorded - 0.3).abs() < 1e-6);
    assert!(bytes[20..32].iter().all(|byte| *byte == 0));
    // A reference-background sample keeps its silhouette but votes for no normal.
    let silent = [RasterSample {
        coverage: 0.5,
        visibility: 0.0,
        normal: [1.0, 0.0, 0.0],
        depth: 2.0,
    }];
    let pixel = resolve_pixel(&silent);
    assert_eq!(pixel.normal, [0.0; 3]);
    assert!((pixel.coverage - 0.5).abs() < 1e-6);
    assert_eq!(pixel.depth, 1.0);
}

#[test]
fn shader_validates_and_exposes_the_shared_soft_coverage_section() {
    raa_shader::Composer::validate(FOLIAGE_RASTER_WGSL).expect("the raster shader validates");
    let helper = soft_coverage_wgsl();
    assert!(helper.contains("fn foliage_soft_coverage(edge_distance: f32, temperature: f32)"));
    assert!(helper.contains("1 / (4 * temperature)"));
    assert!(!helper.contains("@group"));
    assert!(!helper.contains("@compute"));
}
