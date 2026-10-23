//! Byte layout of the S1 stream 0, S10 stream 1 and index buffer.

use raa_geom::MeshData;
use raa_mesh::{AttributedMesh, default_tangent};

fn triangle() -> AttributedMesh {
    AttributedMesh::from_mesh(MeshData {
        positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        normals: vec![[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        indices: vec![0, 1, 2],
    })
}

fn f32_at(bytes: &[u8], offset: usize) -> f32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&bytes[offset..offset + 4]);
    f32::from_le_bytes(b)
}

fn floats_at<const N: usize>(bytes: &[u8], offset: usize) -> [f32; N] {
    std::array::from_fn(|k| f32_at(bytes, offset + 4 * k))
}

#[test]
fn stream0_is_24_bytes_per_vertex_with_position_then_normal() {
    let mesh = triangle();
    let mut out = Vec::new();
    mesh.write_stream0_bytes(&mut out);
    assert_eq!(out.len(), 3 * 24);
    for (i, (p, n)) in mesh
        .base
        .positions
        .iter()
        .zip(&mesh.base.normals)
        .enumerate()
    {
        assert_eq!(floats_at::<3>(&out, i * 24), *p, "position {i} @0");
        assert_eq!(floats_at::<3>(&out, i * 24 + 12), *n, "normal {i} @12");
    }
    let mut reference = Vec::new();
    mesh.base.write_vertex_bytes(&mut reference);
    assert_eq!(out, reference);
}

#[test]
fn stream1_is_40_bytes_per_vertex_with_uv_tangent_colour() {
    let mut mesh = triangle();
    mesh.uvs = Some(vec![[0.0, 0.25], [0.5, 0.75], [1.0, 0.125]]);
    mesh.tangents = Some(vec![
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, -1.0],
        [0.0, 1.0, 0.0, 1.0],
    ]);
    mesh.colors = Some(vec![
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 0.0, 1.0],
    ]);
    let mut out = Vec::new();
    mesh.write_stream1_bytes(&mut out);
    assert_eq!(out.len(), 3 * 40);
    for i in 0..3 {
        let base = i * 40;
        assert_eq!(
            floats_at::<2>(&out, base),
            mesh.uvs.as_ref().unwrap()[i],
            "uv {i} @0"
        );
        assert_eq!(
            floats_at::<4>(&out, base + 8),
            mesh.tangents.as_ref().unwrap()[i],
            "tangent {i} @8"
        );
        assert_eq!(
            floats_at::<4>(&out, base + 24),
            mesh.colors.as_ref().unwrap()[i],
            "colour {i} @24"
        );
    }
}

#[test]
fn stream1_uses_s10_defaults_when_attributes_are_absent() {
    let mesh = triangle();
    let mut out = Vec::new();
    mesh.write_stream1_bytes(&mut out);
    assert_eq!(out.len(), 3 * 40);
    for (i, n) in mesh.base.normals.iter().enumerate() {
        let base = i * 40;
        assert_eq!(floats_at::<2>(&out, base), [0.0, 0.0], "default uv {i}");
        assert_eq!(
            floats_at::<4>(&out, base + 8),
            default_tangent(*n),
            "default tangent {i}"
        );
        assert_eq!(
            floats_at::<4>(&out, base + 24),
            [1.0, 1.0, 1.0, 1.0],
            "default colour {i}"
        );
    }
}

#[test]
fn stream1_defaults_only_the_absent_attribute() {
    let mut mesh = triangle();
    mesh.colors = Some(vec![[0.25, 0.5, 0.75, 1.0]; 3]);
    let mut out = Vec::new();
    mesh.write_stream1_bytes(&mut out);
    assert_eq!(floats_at::<2>(&out, 40), [0.0, 0.0]);
    assert_eq!(floats_at::<4>(&out, 48), default_tangent([0.0, 1.0, 0.0]));
    assert_eq!(floats_at::<4>(&out, 64), [0.25, 0.5, 0.75, 1.0]);
}

#[test]
fn index_bytes_are_4_per_index_little_endian() {
    let mut mesh = triangle();
    mesh.base.indices = vec![0, 2, 1, 1, 2, 0x0102_0304];
    let mut out = Vec::new();
    mesh.write_index_bytes(&mut out);
    assert_eq!(out.len(), mesh.base.indices.len() * 4);
    for (i, index) in mesh.base.indices.iter().enumerate() {
        assert_eq!(out[i * 4..i * 4 + 4], index.to_le_bytes());
    }
}

#[test]
fn writers_append_without_clearing() {
    let mut mesh = triangle();
    mesh.ensure_tangents();
    let prefix = [0xAAu8, 0xBB, 0xCC];

    let mut out = prefix.to_vec();
    mesh.write_stream0_bytes(&mut out);
    assert_eq!(out.len(), 3 + 3 * 24);
    assert_eq!(out[..3], prefix);

    let mut out = prefix.to_vec();
    mesh.write_stream1_bytes(&mut out);
    assert_eq!(out.len(), 3 + 3 * 40);
    assert_eq!(out[..3], prefix);

    let mut out = prefix.to_vec();
    mesh.write_index_bytes(&mut out);
    assert_eq!(out.len(), 3 + 3 * 4);
    assert_eq!(out[..3], prefix);
}
