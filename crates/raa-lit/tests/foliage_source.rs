//! Source retention for native foliage baking (M7, contract m7-source): `LitHandle::
//! foliage_source` reads the immutable `AttributedMesh` kept beside each uploaded lit mesh.
//!
//! The software-GPU tests (SKIP without an adapter) cover upload, update and removal,
//! repeated sub-mesh instances in input draw order, source-root transforms with independent
//! shading normals, and the refusal of stale ids and invalid transforms.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{skip, software_gpu};
use raa_core::{GpuContext, MaterialId, MeshId, RaaError, wgpu};
use raa_geom::MeshData;
use raa_lit::{AttributedMesh, LitOptions, LitRenderer, SubMesh};
use raa_math::{Mat4, Vec3};
use raa_pipeline::Pipeline;

const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// A shading normal deliberately different from the triangle's geometric normal.
const SHADING: [f32; 3] = [0.6, 0.8, 0.0];

fn setup(test: &str) -> Option<(GpuContext, LitRenderer)> {
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return None;
    };
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, SURFACE, false)));
    let lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit renderer");
    Some((gpu, lit))
}

/// A one-triangle mesh in the z = 0 plane translated by `dx` on x, with a shading normal
/// that is not the geometric `+z`.
fn triangle(dx: f32) -> AttributedMesh {
    let mut base = MeshData::new();
    base.positions = vec![[dx, 0.0, 0.0], [dx + 1.0, 0.0, 0.0], [dx, 1.0, 0.0]];
    base.normals = vec![SHADING; 3];
    base.indices = vec![0, 1, 2];
    let mut mesh = AttributedMesh::from_mesh(base);
    mesh.uvs = Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    mesh.submeshes = vec![SubMesh {
        first_index: 0,
        index_count: 3,
        material: None,
    }];
    mesh
}

/// A two-sub-mesh mesh, one triangle per range.
fn two_submeshes() -> AttributedMesh {
    let mut base = MeshData::new();
    base.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    base.normals = vec![SHADING; 4];
    base.indices = vec![0, 1, 2, 1, 3, 2];
    let mut mesh = AttributedMesh::from_mesh(base);
    mesh.uvs = Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
    mesh.submeshes = vec![
        SubMesh {
            first_index: 0,
            index_count: 3,
            material: None,
        },
        SubMesh {
            first_index: 3,
            index_count: 3,
            material: None,
        },
    ];
    mesh
}

fn code(err: RaaError) -> &'static str {
    err.code()
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-5
}

fn assert_point(got: [f32; 3], want: [f32; 3]) {
    assert!(
        got.iter().zip(want).all(|(a, b)| close(*a, b)),
        "{got:?} != {want:?}"
    );
}

#[test]
fn source_snapshot_follows_upload_update_and_removal() {
    let test = "source_snapshot_follows_upload_update_and_removal";
    let Some((gpu, mut lit)) = setup(test) else {
        return;
    };
    let handle = lit.handle();
    let lit_id = lit.upload(&gpu, &triangle(0.0)).expect("upload");
    let mesh = lit.submesh_ids(lit_id).expect("sub-meshes")[0];

    let source = handle
        .foliage_source(
            &[(mesh, MaterialId(3), 9, Mat4::IDENTITY, true)],
            Mat4::IDENTITY,
            4,
        )
        .expect("source");
    assert_eq!(source.generation, 4);
    assert_eq!(source.root_world, Mat4::IDENTITY);
    assert_eq!(source.triangles.len(), 1);
    let first = source.triangles[0];
    assert_point(first.positions[1], [1.0, 0.0, 0.0]);
    assert_point(first.normals[0], SHADING);
    assert_eq!(first.uv, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    assert_eq!((first.material, first.source_tag), (3, 9));
    assert!(first.foliage);
    assert_point(source.min, [0.0, 0.0, 0.0]);
    assert_point(source.max, [1.0, 1.0, 0.0]);

    lit.update(&gpu, lit_id, &triangle(2.0)).expect("update");
    let updated = handle
        .foliage_source(
            &[(mesh, MaterialId(3), 9, Mat4::IDENTITY, false)],
            Mat4::IDENTITY,
            5,
        )
        .expect("updated source");
    assert_eq!(updated.generation, 5);
    assert!(!updated.triangles[0].foliage);
    assert_point(updated.triangles[0].positions[0], [2.0, 0.0, 0.0]);
    assert_point(updated.max, [3.0, 1.0, 0.0]);

    lit.remove(lit_id);
    let stale = handle
        .foliage_source(
            &[(mesh, MaterialId(0), 0, Mat4::IDENTITY, false)],
            Mat4::IDENTITY,
            6,
        )
        .expect_err("removed mesh");
    assert_eq!(code(stale), "foliage.invalid");
}

#[test]
fn repeated_submeshes_keep_draw_order_and_source_ranges() {
    let test = "repeated_submeshes_keep_draw_order_and_source_ranges";
    let Some((gpu, mut lit)) = setup(test) else {
        return;
    };
    let handle = lit.handle();
    let lit_id = lit.upload(&gpu, &two_submeshes()).expect("upload");
    let ids = lit.submesh_ids(lit_id).expect("sub-meshes").to_vec();
    assert_eq!(ids.len(), 2);

    let shifted = Mat4::from_translation(Vec3::new(0.0, 0.0, 4.0));
    let source = handle
        .foliage_source(
            &[
                (ids[0], MaterialId(1), 11, Mat4::IDENTITY, true),
                (ids[0], MaterialId(2), 12, shifted, false),
                (ids[1], MaterialId(3), 13, Mat4::IDENTITY, true),
            ],
            Mat4::IDENTITY,
            7,
        )
        .expect("source");
    assert_eq!(source.triangles.len(), 3);
    let tags: Vec<u32> = source.triangles.iter().map(|t| t.source_tag).collect();
    assert_eq!(tags, [11, 12, 13]);
    let materials: Vec<u32> = source.triangles.iter().map(|t| t.material).collect();
    assert_eq!(materials, [1, 2, 3]);
    // The repeated first sub-mesh draws the same triangle; only the model-to-root differs.
    assert_point(source.triangles[0].positions[0], [0.0, 0.0, 0.0]);
    assert_point(source.triangles[1].positions[0], [0.0, 0.0, 4.0]);
    // The second sub-mesh starts at its own source triangle.
    assert_point(source.triangles[2].positions[0], [1.0, 0.0, 0.0]);
}

#[test]
fn source_root_transform_and_normals_are_independent() {
    let test = "source_root_transform_and_normals_are_independent";
    let Some((gpu, mut lit)) = setup(test) else {
        return;
    };
    let handle = lit.handle();
    let lit_id = lit.upload(&gpu, &triangle(0.0)).expect("upload");
    let mesh = lit.submesh_ids(lit_id).expect("sub-meshes")[0];

    let model_to_root = Mat4::from_scale(Vec3::new(2.0, 1.0, 1.0));
    let root_world = Mat4::from_translation(Vec3::new(10.0, 20.0, 30.0));
    let source = handle
        .foliage_source(
            &[(mesh, MaterialId(0), 0, model_to_root, true)],
            root_world,
            8,
        )
        .expect("source");
    let first = source.triangles[0];
    assert_point(first.positions[1], [2.0, 0.0, 0.0]);
    // Inverse transpose of diag(2, 1, 1) gives (0.3, 0.8, 0) for SHADING, normalised.
    let len = 0.3f32.hypot(0.8);
    assert_point(first.normals[0], [0.3 / len, 0.8 / len, 0.0]);
    // The positions stay in the geometric z = 0 plane, so the normal is never recomputed.
    assert!(first.normals[0][2].abs() < 1e-6);
    assert_eq!(source.root_world, root_world);
}

#[test]
fn invalid_draws_are_refused_and_empty_draws_are_empty() {
    let test = "invalid_draws_are_refused_and_empty_draws_are_empty";
    let Some((gpu, mut lit)) = setup(test) else {
        return;
    };
    let handle = lit.handle();
    let lit_id = lit.upload(&gpu, &triangle(0.0)).expect("upload");
    let mesh = lit.submesh_ids(lit_id).expect("sub-meshes")[0];

    let empty = handle
        .foliage_source(&[], Mat4::IDENTITY, 9)
        .expect("empty source");
    assert!(empty.triangles.is_empty());
    assert_eq!((empty.min, empty.max), ([0.0; 3], [0.0; 3]));

    let unknown = handle
        .foliage_source(
            &[(MeshId(999), MaterialId(0), 0, Mat4::IDENTITY, false)],
            Mat4::IDENTITY,
            0,
        )
        .expect_err("unknown mesh");
    assert_eq!(code(unknown), "foliage.invalid");

    let mut bad = Mat4::IDENTITY;
    bad.cols[0][0] = f32::NAN;
    let nonfinite = handle
        .foliage_source(&[(mesh, MaterialId(0), 0, bad, false)], Mat4::IDENTITY, 0)
        .expect_err("non-finite transform");
    assert_eq!(code(nonfinite), "foliage.invalid");

    let singular = Mat4::from_scale(Vec3::new(0.0, 1.0, 1.0));
    let flat = handle
        .foliage_source(
            &[(mesh, MaterialId(0), 0, singular, false)],
            Mat4::IDENTITY,
            0,
        )
        .expect_err("singular transform");
    assert_eq!(code(flat), "foliage.invalid");

    lit.remove(lit_id);
    let stale = handle
        .foliage_source(
            &[(mesh, MaterialId(0), 0, Mat4::IDENTITY, false)],
            Mat4::IDENTITY,
            0,
        )
        .expect_err("stale id");
    assert_eq!(code(stale), "foliage.invalid");
}
