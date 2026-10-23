//! S13 refusals (one case per `mesh.*` code, non-empty message) and valid round-trips.

use raa_core::MaterialId;
use raa_geom::MeshData;
use raa_mesh::{AttributedMesh, Colors, MeshError, SubMesh, from_js_arrays};

/// A unit quad in the XY plane facing +Z: 4 vertices, 2 triangles.
struct Quad {
    positions: Vec<f32>,
    normals: Vec<f32>,
    uvs: Vec<f32>,
    tangents: Vec<f32>,
    colors: Vec<f32>,
    indices: Vec<u32>,
}

fn quad() -> Quad {
    Quad {
        positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        normals: [0.0, 0.0, 1.0].repeat(4),
        uvs: vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        tangents: [1.0, 0.0, 0.0, 1.0].repeat(4),
        colors: [1.0, 0.5, 0.25, 1.0].repeat(4),
        indices: vec![0, 1, 2, 0, 2, 3],
    }
}

fn sub(first: u32, count: u32) -> SubMesh {
    SubMesh {
        first_index: first,
        index_count: count,
        material: None,
    }
}

fn build(q: &Quad, subs: &[SubMesh]) -> Result<AttributedMesh, MeshError> {
    from_js_arrays(
        &q.positions,
        &q.normals,
        Some(&q.uvs),
        Some(&q.tangents),
        Some(Colors::Linear(&q.colors)),
        &q.indices,
        subs,
    )
}

fn refused(result: Result<AttributedMesh, MeshError>, code: &str, mentions: &str) {
    let err = result.expect_err("mesh must be refused");
    assert_eq!(err.code(), code, "unexpected refusal: {err}");
    assert!(!err.message.is_empty(), "{code} has an empty message");
    assert!(
        err.message.contains(mentions),
        "{code} message {:?} does not mention {mentions:?}",
        err.message
    );
    assert!(err.to_string().contains(code));
}

#[test]
fn empty_positions_or_indices_are_refused() {
    let mut q = quad();
    q.indices.clear();
    refused(build(&q, &[]), "mesh.empty", "indices");
    let refused_mesh = AttributedMesh::default().validate();
    assert_eq!(refused_mesh.map_err(|e| e.code), Err("mesh.empty"));
}

#[test]
fn length_mismatch_names_the_array() {
    let mut q = quad();
    q.normals.pop();
    refused(build(&q, &[]), "mesh.length_mismatch", "normals");
    let mut q = quad();
    q.uvs.push(0.0);
    refused(build(&q, &[]), "mesh.length_mismatch", "uvs");
    let mut q = quad();
    q.tangents.truncate(12);
    refused(build(&q, &[]), "mesh.length_mismatch", "tangents");
    let q = quad();
    let srgb = [255u8; 12];
    let result = from_js_arrays(
        &q.positions,
        &q.normals,
        None,
        None,
        Some(Colors::Srgb8(&srgb)),
        &q.indices,
        &[],
    );
    refused(result, "mesh.length_mismatch", "colors");
    let mut q = quad();
    q.indices.push(1);
    refused(build(&q, &[]), "mesh.length_mismatch", "indices");
}

#[test]
fn not_finite_names_array_and_index() {
    let mut q = quad();
    q.positions[7] = f32::NAN;
    refused(build(&q, &[]), "mesh.not_finite", "positions[2][1]");
    let mut q = quad();
    q.colors[5] = f32::INFINITY;
    refused(build(&q, &[]), "mesh.not_finite", "colors[1][1]");
}

#[test]
fn index_range_reports_the_first_bad_index() {
    let mut q = quad();
    q.indices[4] = 9;
    q.indices[5] = 4;
    refused(build(&q, &[]), "mesh.index_range", "indices[4] = 9");
}

#[test]
fn zero_normal_is_refused() {
    let mut q = quad();
    q.normals[6..9].copy_from_slice(&[0.0, 0.0, 0.0]);
    refused(build(&q, &[]), "mesh.invalid_normal", "normals[2]");
}

#[test]
fn invalid_tangent_w_or_zero_xyz_is_refused() {
    let mut q = quad();
    q.tangents[7] = 0.5;
    refused(build(&q, &[]), "mesh.invalid_tangent", "tangents[1]");
    let mut q = quad();
    q.tangents[12..15].copy_from_slice(&[0.0, 0.0, 0.0]);
    refused(build(&q, &[]), "mesh.invalid_tangent", "tangents[3]");
}

#[test]
fn submesh_range_rules_are_enforced() {
    let q = quad();
    refused(
        build(&q, &[sub(1, 3)]),
        "mesh.submesh_range",
        "subMeshes[0]",
    );
    refused(build(&q, &[sub(0, 0)]), "mesh.submesh_range", "count is 0");
    refused(
        build(&q, &[sub(0, 3), sub(3, 2)]),
        "mesh.submesh_range",
        "subMeshes[1]",
    );
    refused(build(&q, &[sub(3, 6)]), "mesh.submesh_range", "past");
}

#[test]
fn overlapping_submeshes_are_refused() {
    let q = quad();
    refused(
        build(&q, &[sub(3, 3), sub(0, 6)]),
        "mesh.submesh_overlap",
        "subMeshes[0] and subMeshes[1]",
    );
}

#[test]
fn valid_desc_round_trips() {
    let mut q = quad();
    q.normals[0..3].copy_from_slice(&[0.0, 0.0, 2.0]);
    let subs = [
        SubMesh {
            first_index: 0,
            index_count: 3,
            material: Some(MaterialId(4)),
        },
        sub(3, 3),
    ];
    let mesh = build(&q, &subs).expect("valid mesh");
    assert_eq!(mesh.vertex_count(), 4);
    assert_eq!(mesh.triangle_count(), 2);
    assert_eq!(mesh.base.positions[2], [1.0, 1.0, 0.0]);
    assert_eq!(mesh.base.normals, vec![[0.0, 0.0, 1.0]; 4]);
    assert_eq!(mesh.base.indices, q.indices);
    assert_eq!(mesh.submeshes, subs.to_vec());
    assert_eq!(mesh.uvs.as_deref().map(|u| u[2]), Some([1.0, 1.0]));
    assert_eq!(
        mesh.tangents.as_deref().map(|t| t[0]),
        Some([1.0, 0.0, 0.0, 1.0])
    );
    assert_eq!(
        mesh.colors.as_deref().map(|c| c[3]),
        Some([1.0, 0.5, 0.25, 1.0])
    );
    let b = mesh.bounds();
    assert_eq!((b.min.x, b.max.y, b.max.z), (0.0, 1.0, 0.0));
    assert_eq!(mesh.validate(), Ok(()));
}

#[test]
fn absent_submeshes_cover_all_indices() {
    let q = quad();
    let mesh = from_js_arrays(&q.positions, &q.normals, None, None, None, &q.indices, &[])
        .expect("valid mesh");
    assert_eq!(mesh.submeshes, vec![sub(0, 6)]);
    assert!(mesh.uvs.is_none() && mesh.tangents.is_none() && mesh.colors.is_none());
}

#[test]
fn srgb8_colours_use_the_exact_eotf() {
    let q = quad();
    let srgb = [
        255, 128, 0, 128, 10, 10, 10, 255, 0, 0, 0, 0, 255, 255, 255, 255,
    ];
    let mesh = from_js_arrays(
        &q.positions,
        &q.normals,
        None,
        None,
        Some(Colors::Srgb8(&srgb)),
        &q.indices,
        &[],
    )
    .expect("valid mesh");
    let colors = mesh.colors.expect("colours kept");
    let [r, g, b, a] = colors[0];
    assert_eq!(r, 1.0);
    assert!((g - 0.215_861).abs() < 1e-5, "128 -> {g}");
    assert_eq!(b, 0.0);
    assert!(
        (a - 128.0 / 255.0).abs() < 1e-7,
        "alpha is linear /255: {a}"
    );
    assert!(
        (colors[1][0] - (10.0 / 255.0) / 12.92).abs() < 1e-7,
        "linear segment"
    );
    assert_eq!(colors[3], [1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn from_mesh_is_one_submesh_without_attributes() {
    let q = quad();
    let base = MeshData {
        positions: q
            .positions
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect(),
        normals: vec![[0.0, 0.0, 1.0]; 4],
        indices: q.indices.clone(),
    };
    let mesh = AttributedMesh::from_mesh(base);
    assert_eq!(mesh.submeshes, vec![sub(0, 6)]);
    assert_eq!(mesh.validate(), Ok(()));
}
