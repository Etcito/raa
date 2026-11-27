//! Subject tests of deterministic surface Poisson disk initialization (M7, contract
//! `m7-compute`): the requested count on a planar fixture, the separation of the accepted
//! micro-triangles, the independent source shading normals, the deterministic seed and the
//! empty source. The module is path-compiled until production registration.

#![cfg(not(target_arch = "wasm32"))]

mod foliage_resources {
    pub use raa_lit::foliage_resources::*;
}

#[path = "../src/foliage_sampling.rs"]
mod foliage_sampling;

#[path = "../src/foliage_poisson.rs"]
mod foliage_poisson;

use foliage_poisson::initialize_foliage;
use foliage_resources::{FoliageSource, MicroTriangle, SourceTriangle};
use raa_core::RaaError;

fn triangle(positions: [[f32; 3]; 3], normals: [[f32; 3]; 3], foliage: bool) -> SourceTriangle {
    SourceTriangle {
        positions,
        normals,
        uv: [[0.0; 2]; 3],
        material: 1,
        source_tag: 2,
        foliage,
    }
}

fn source(triangles: Vec<SourceTriangle>) -> FoliageSource {
    FoliageSource {
        triangles,
        min: [-1.0; 3],
        max: [101.0; 3],
        root_world: raa_math::Mat4::IDENTITY,
        generation: 3,
    }
}

fn plane_source() -> FoliageSource {
    let normal = [[0.0, 1.0, 0.0]; 3];
    let first = triangle(
        [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 0.0, 100.0]],
        normal,
        true,
    );
    let second = triangle(
        [[100.0, 0.0, 0.0], [100.0, 0.0, 100.0], [0.0, 0.0, 100.0]],
        normal,
        true,
    );
    source(vec![first, second])
}

fn centroid(t: &MicroTriangle) -> [f32; 3] {
    [
        (t.p0[0] + t.p1[0] + t.p2[0]) / 3.0,
        (t.p0[1] + t.p1[1] + t.p2[1]) / 3.0,
        (t.p0[2] + t.p1[2] + t.p2[2]) / 3.0,
    ]
}

fn initial_radius(area: f32, count: u32) -> f32 {
    (2.0 * area / (3.0f32.sqrt() * count as f32)).sqrt()
}

fn invalid_code() -> &'static str {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message: String::new(),
    }
    .code()
}

#[test]
fn planar_fixture_returns_the_requested_count() {
    let triangles = initialize_foliage(&plane_source(), 64, 7).expect("initialization succeeds");
    assert_eq!(triangles.len(), 64);
    assert!(triangles.iter().all(|t| t.opacity == 1.0));
    for t in &triangles {
        let normal = t.normal;
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        assert!((length - 1.0).abs() < 1e-5, "normal length {length}");
        assert!((normal[1] - 1.0).abs() < 1e-5, "normal {normal:?}");
    }
}

#[test]
fn planar_fixture_keeps_the_poisson_separation() {
    let count = 64;
    let triangles =
        initialize_foliage(&plane_source(), count, 11).expect("initialization succeeds");
    let radius = initial_radius(10_000.0, count);
    for (left, left_triangle) in triangles.iter().enumerate() {
        let a = centroid(left_triangle);
        for right_triangle in triangles.iter().skip(left + 1) {
            let b = centroid(right_triangle);
            let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
            let distance = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!(
                distance >= radius * 0.25,
                "sample {left} is {distance} from a later sample, radius {radius}"
            );
        }
    }
}

#[test]
fn normals_come_from_the_source_shading_normals() {
    let shading = [[0.0, 0.6, 0.8]; 3];
    let a = triangle(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 4.0]],
        shading,
        true,
    );
    let triangles = initialize_foliage(&source(vec![a]), 4, 3).expect("initialization succeeds");
    for t in &triangles {
        assert!((t.normal[1] - 0.6).abs() < 1e-5, "normal {:?}", t.normal);
        assert!((t.normal[2] - 0.8).abs() < 1e-5, "normal {:?}", t.normal);
    }
}

#[test]
fn same_seed_repeats_and_a_changed_seed_differs() {
    let src = plane_source();
    let first = initialize_foliage(&src, 32, 21).expect("initialization succeeds");
    let again = initialize_foliage(&src, 32, 21).expect("initialization succeeds");
    let other = initialize_foliage(&src, 32, 22).expect("initialization succeeds");
    assert_eq!(first, again);
    assert_ne!(first, other);
}

#[test]
fn empty_and_wood_only_sources_return_no_triangle() {
    assert!(
        initialize_foliage(&source(Vec::new()), 8, 1)
            .expect("initialization succeeds")
            .is_empty()
    );
    let wood = triangle(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 1.0, 0.0]; 3],
        false,
    );
    assert!(
        initialize_foliage(&source(vec![wood]), 8, 1)
            .expect("initialization succeeds")
            .is_empty()
    );
}

#[test]
fn positive_count_is_required() {
    let err = initialize_foliage(&plane_source(), 0, 1).expect_err("zero count is refused");
    assert_eq!(err.code(), invalid_code());
}
