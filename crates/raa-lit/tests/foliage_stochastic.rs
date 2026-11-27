//! Subject tests of the stochastic foliage assembler (M7, contract `m7-stochastic`): the
//! actual emitted foliage area, the triangle-constant sampled source normals, empty foliage
//! with retained wood and the repeated-input equality compared byte for byte. The module is
//! path-compiled until production registration.

#![cfg(not(target_arch = "wasm32"))]

mod foliage_resources {
    pub use raa_lit::foliage_resources::*;
}

#[allow(dead_code)]
#[path = "../src/foliage_sampling.rs"]
mod foliage_sampling;
#[allow(dead_code)]
#[path = "../src/foliage_stochastic.rs"]
mod foliage_stochastic;
#[allow(dead_code)]
#[path = "../src/foliage_volume.rs"]
mod foliage_volume;

use foliage_resources::{FoliageSource, SourceTriangle};
use foliage_sampling::{FoliageSample, sample_foliage};
use foliage_stochastic::build_stochastic_foliage;
use raa_core::RaaError;
use raa_geom::MeshData;

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
        max: [3.0, 2.0, 2.0],
        root_world: raa_math::Mat4::IDENTITY,
        generation: 5,
    }
}

fn invalid_code() -> &'static str {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message: String::new(),
    }
    .code()
}

fn corners_cross(corners: [[f32; 3]; 3]) -> [f32; 3] {
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
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn triangle_area(corners: [[f32; 3]; 3]) -> f32 {
    let n = corners_cross(corners);
    0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
}

fn mesh_triangle(mesh: &MeshData, triangle: usize) -> ([[f32; 3]; 3], [[f32; 3]; 3]) {
    let mut positions = [[0.0f32; 3]; 3];
    let mut normals = [[0.0f32; 3]; 3];
    for (slot, index) in mesh.indices[3 * triangle..3 * triangle + 3]
        .iter()
        .enumerate()
    {
        positions[slot] = mesh.positions[*index as usize];
        normals[slot] = mesh.normals[*index as usize];
    }
    (positions, normals)
}

fn mixed_source() -> FoliageSource {
    let foliage_a = triangle(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0]; 3],
        true,
    );
    let wood = triangle(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        false,
    );
    let foliage_b = triangle(
        [[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        [[0.0, 0.0, 1.0]; 3],
        true,
    );
    source(vec![foliage_a, wood, foliage_b])
}

#[test]
fn emitted_proxy_area_equals_the_eligible_source_area() {
    let source = mixed_source();
    let samples = sample_foliage(&source, 24, 7).expect("sampling succeeds");
    let albedo = vec![[0.25f32, 0.5, 0.75]; samples.len()];
    let bake = build_stochastic_foliage(&source, &samples, &albedo, 16).expect("assembly succeeds");

    assert_eq!(bake.foliage_triangles, 24);
    assert_eq!(bake.wood_triangles, 1);
    assert_eq!(bake.mesh.vertex_count(), 3 * 25);
    assert_eq!(bake.mesh.triangle_count(), 25);
    assert_eq!(bake.mesh.validate(), Ok(()));

    let (wood_corners, wood_normals) = mesh_triangle(&bake.mesh, 0);
    assert_eq!(wood_corners, source.triangles[1].positions);
    assert_eq!(wood_normals, source.triangles[1].normals);

    let mut total = 0.0f32;
    for (index, sample) in samples.iter().enumerate() {
        let (corners, _) = mesh_triangle(&bake.mesh, 1 + index);
        let area = triangle_area(corners);
        assert_eq!(corners, sample.corners);
        assert!(
            (area - sample.represented_area).abs() < 1e-5,
            "stratum {index} area {area} represents {}",
            sample.represented_area
        );
        total += area;
    }
    assert!((total - 1.5).abs() < 1e-4, "emitted foliage area {total}");

    assert_eq!(bake.volume.len(), 4 * 16 * 16 * 16);
    let mut occupied = 0usize;
    for pixel in bake.volume.as_chunks::<4>().0 {
        if pixel[3] == 0 {
            continue;
        }
        occupied += 1;
        assert_eq!(&pixel[..3], &[64, 128, 191]);
    }
    assert!(occupied > 0, "the sampled foliage must occupy the volume");
}

#[test]
fn foliage_normals_are_the_sampled_source_normals_not_the_proxy_plane() {
    let source = source(vec![triangle(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        [[0.0, 1.0, 0.0]; 3],
        false,
    )]);
    let normal = [0.0, 1.0, 0.0];
    let samples = [
        FoliageSample {
            source_triangle: 0,
            barycentric: [0.25, 0.25, 0.5],
            position: [0.0, 0.5, 0.0],
            normal,
            corners: [[-1.0, 0.5, 0.0], [1.0, 0.5, 0.0], [0.0, 0.5, 1.0]],
            represented_area: 0.5,
        },
        FoliageSample {
            source_triangle: 0,
            barycentric: [0.5, 0.5, 0.0],
            position: [0.5, 0.5, 0.5],
            normal,
            corners: [[0.0, 0.5, 0.0], [0.0, 0.5, 1.0], [-1.0, 0.5, 0.0]],
            represented_area: 0.5,
        },
    ];
    let albedo = vec![[0.5f32; 3]; samples.len()];
    let bake = build_stochastic_foliage(&source, &samples, &albedo, 16).expect("assembly succeeds");

    assert_eq!(bake.foliage_triangles, 2);
    for (index, sample) in samples.iter().enumerate() {
        let (corners, normals) = mesh_triangle(&bake.mesh, bake.wood_triangles as usize + index);
        assert_eq!(corners, sample.corners);
        assert_eq!(normals, [sample.normal; 3]);
        let cross = corners_cross(corners);
        assert!(
            cross[1] < 0.0,
            "proxy geometric normal must not drive the shading normal: {cross:?}"
        );
    }
}

#[test]
fn empty_foliage_keeps_the_wood_and_a_zero_volume() {
    let wood_a = triangle(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0]; 3],
        false,
    );
    let wood_b = triangle(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        false,
    );
    let source = source(vec![wood_a, wood_b]);
    let samples = sample_foliage(&source, 8, 1).expect("sampling succeeds");
    assert!(samples.is_empty());

    for resolution in [16u32, 32] {
        let bake = build_stochastic_foliage(&source, &samples, &[], resolution).expect("assembly");
        assert_eq!(bake.foliage_triangles, 0);
        assert_eq!(bake.wood_triangles, 2);
        assert_eq!(bake.mesh.triangle_count(), 2);
        assert_eq!(bake.mesh.indices, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(bake.volume.len(), 4 * (resolution as usize).pow(3));
        assert!(bake.volume.iter().all(|byte| *byte == 0));
        assert_eq!(
            mesh_triangle(&bake.mesh, 0),
            (wood_a.positions, wood_a.normals)
        );
        assert_eq!(
            mesh_triangle(&bake.mesh, 1),
            (wood_b.positions, wood_b.normals)
        );
    }
}

#[test]
fn repeated_inputs_return_equal_meshes_and_equal_volume_bytes() {
    let source = mixed_source();
    let samples = sample_foliage(&source, 12, 42).expect("sampling succeeds");
    let albedo = vec![[0.1f32, 0.2, 0.3]; samples.len()];
    let first =
        build_stochastic_foliage(&source, &samples, &albedo, 32).expect("assembly succeeds");
    let second =
        build_stochastic_foliage(&source, &samples, &albedo, 32).expect("assembly succeeds");

    assert_eq!(first, second);
    assert_eq!(first.mesh.positions, second.mesh.positions);
    assert_eq!(first.mesh.normals, second.mesh.normals);
    assert_eq!(first.mesh.indices, second.mesh.indices);
    assert_eq!(first.volume, second.volume);
}

#[test]
fn albedo_mismatch_and_invalid_albedo_are_refused() {
    let source = mixed_source();
    let samples = sample_foliage(&source, 4, 3).expect("sampling succeeds");
    let valid = vec![[0.5f32, 0.5, 0.5]; samples.len()];
    for length in [samples.len() - 1, samples.len() + 1] {
        let albedo = vec![[0.5f32, 0.5, 0.5]; length];
        let err = build_stochastic_foliage(&source, &samples, &albedo, 16)
            .expect_err("albedo length must match the samples");
        assert_eq!(err.code(), invalid_code());
    }
    let mut out_of_range = valid.clone();
    out_of_range[0] = [1.5, 0.0, 0.0];
    let err = build_stochastic_foliage(&source, &samples, &out_of_range, 16)
        .expect_err("albedo must be linear RGB in 0..=1");
    assert_eq!(err.code(), invalid_code());
    let err = build_stochastic_foliage(&source, &samples, &valid, 8)
        .expect_err("resolution must be 16 or 32");
    assert_eq!(err.code(), invalid_code());
}
