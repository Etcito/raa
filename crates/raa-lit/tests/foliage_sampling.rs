//! Subject tests of the area-weighted foliage sampling module (M7, contract
//! `m7-sampling`): the hand-derived two-triangle stratum distribution, the total proxy
//! area, the seeded sequence and repeatability, the changed seed, the independent source
//! shading normals and the refusals. The module is path-compiled until production
//! registration.

#![cfg(not(target_arch = "wasm32"))]

mod foliage_resources {
    pub use raa_lit::foliage_resources::*;
}

#[path = "../src/foliage_sampling.rs"]
mod foliage_sampling;

use foliage_resources::{FoliageSource, SourceTriangle};
use foliage_sampling::{FoliageRng, sample_foliage};
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
        max: [20.0; 3],
        root_world: raa_math::Mat4::IDENTITY,
        generation: 3,
    }
}

fn corner_area(corners: [[f32; 3]; 3]) -> f32 {
    let a = [
        corners[1][0] - corners[0][0],
        corners[1][1] - corners[0][1],
        corners[1][2] - corners[0][2],
    ];
    let b = [
        corners[2][0] - corners[0][0],
        corners[2][1] - corners[0][1],
        corners[2][2] - corners[0][2],
    ];
    let n = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
}

fn invalid_code() -> &'static str {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message: String::new(),
    }
    .code()
}

#[test]
fn two_triangles_follow_the_hand_derived_cumulative_area() {
    let a = triangle(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0]; 3],
        true,
    );
    let b = triangle(
        [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 3.0, 0.0]],
        [[0.0, 0.0, 1.0]; 3],
        true,
    );
    let samples = sample_foliage(&source(vec![a, b]), 4, 1).expect("sampling succeeds");
    let indices: Vec<u32> = samples.iter().map(|s| s.source_triangle).collect();
    assert_eq!(indices, vec![0, 1, 1, 1]);
    let mut total = 0.0;
    for sample in &samples {
        assert_eq!(sample.represented_area, 0.5);
        let area = corner_area(sample.corners);
        assert!((area - 0.5).abs() < 1e-5, "corner area {area}");
        total += area;
    }
    assert!((total - 2.0).abs() < 1e-4, "proxy total {total}");
}

#[test]
fn only_eligible_triangles_participate() {
    let wood = triangle(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
        [[0.0, 0.0, 1.0]; 3],
        false,
    );
    let foliage = triangle(
        [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0]; 3],
        true,
    );
    let degenerate = triangle(
        [[20.0, 0.0, 0.0], [21.0, 0.0, 0.0], [22.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0]; 3],
        true,
    );
    let samples =
        sample_foliage(&source(vec![wood, foliage, degenerate]), 3, 9).expect("sampling succeeds");
    assert_eq!(samples.len(), 3);
    assert!(samples.iter().all(|s| s.source_triangle == 1));
    let total: f32 = samples.iter().map(|s| corner_area(s.corners)).sum();
    assert!((total - 0.5).abs() < 1e-5, "proxy total {total}");
    let expected = 0.5 / 3.0;
    assert!((samples[0].represented_area - expected).abs() < 1e-7);
}

#[test]
fn no_eligible_triangle_returns_empty() {
    let wood = triangle(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0]; 3],
        false,
    );
    assert!(
        sample_foliage(&source(vec![wood]), 8, 1)
            .unwrap()
            .is_empty()
    );
    assert!(
        sample_foliage(&source(Vec::new()), 8, 1)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn positive_count_is_required() {
    let a = triangle(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0]; 3],
        true,
    );
    let err = sample_foliage(&source(vec![a]), 0, 1).expect_err("zero count is refused");
    assert_eq!(err.code(), invalid_code());
}

fn reference_next(state: &mut u32) -> u32 {
    *state = state.wrapping_add(0x9e37_79b9);
    let mut z = *state;
    z = (z ^ (z >> 16)).wrapping_mul(0x85eb_ca6b);
    z = (z ^ (z >> 13)).wrapping_mul(0xc2b2_ae35);
    z ^ (z >> 16)
}

#[test]
fn seeded_stream_matches_the_documented_sequence() {
    for seed in [0u32, 7, 0xdead_beef] {
        let mut rng = FoliageRng::new(seed);
        let mut state = seed;
        for draw in 0..32 {
            assert_eq!(
                rng.next_u32(),
                reference_next(&mut state),
                "seed {seed} draw {draw}"
            );
        }
        let value = rng.next_f32();
        assert!((0.0..1.0).contains(&value), "next_f32 {value}");
    }
}

#[test]
fn same_seed_repeats_and_a_changed_seed_differs() {
    let a = triangle(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0]; 3],
        true,
    );
    let src = source(vec![a]);
    let first = sample_foliage(&src, 32, 11).expect("sampling succeeds");
    let again = sample_foliage(&src, 32, 11).expect("sampling succeeds");
    let other = sample_foliage(&src, 32, 12).expect("sampling succeeds");
    assert_eq!(first, again);
    assert_ne!(first, other);
    assert!(
        first
            .iter()
            .zip(&other)
            .any(|(l, r)| l.corners != r.corners || l.barycentric != r.barycentric)
    );
}

#[test]
fn shading_normals_stay_independent_of_the_geometric_plane() {
    let a = triangle(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0]; 3],
        true,
    );
    let samples = sample_foliage(&source(vec![a]), 2, 5).expect("sampling succeeds");
    for sample in &samples {
        assert!(sample.normal[0].abs() < 1e-6);
        assert!(sample.normal[1].abs() < 1e-6);
        assert!((sample.normal[2] - 1.0).abs() < 1e-6);
        let sum: f32 = sample.barycentric.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(sample.barycentric.iter().all(|w| (0.0..=1.0).contains(w)));
        for corner in sample.corners {
            let radial = [
                corner[0] - sample.position[0],
                corner[1] - sample.position[1],
                corner[2] - sample.position[2],
            ];
            let along = radial[0] * sample.normal[0]
                + radial[1] * sample.normal[1]
                + radial[2] * sample.normal[2];
            assert!(
                along.abs() < 1e-5,
                "corner leaves the tangent plane: {along}"
            );
        }
    }
}
