//! GPU byte streams of an attributed mesh: S1 vertex stream 0, S10 vertex stream 1 and
//! the u32 index buffer. Little-endian via `to_le_bytes`; every writer appends to `out`.

use crate::AttributedMesh;
use crate::tangents::default_tangent;

/// Stride of vertex stream 0 (S1): position f32x3 @0, normal f32x3 @12.
const STREAM0_STRIDE: usize = 24;
/// Stride of vertex stream 1 (S10): uv f32x2 @0, tangent f32x4 @8, colour f32x4 @24.
const STREAM1_STRIDE: usize = 40;

const DEFAULT_UV: [f32; 2] = [0.0, 0.0];
const DEFAULT_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

impl AttributedMesh {
    /// Appends the S1 vertex stream (24 B per vertex: position @0, normal @12) to `out`.
    /// The bytes equal `MeshData::write_vertex_bytes`, without clearing `out`.
    pub fn write_stream0_bytes(&self, out: &mut Vec<u8>) {
        out.reserve(self.base.positions.len() * STREAM0_STRIDE);
        for (i, position) in self.base.positions.iter().enumerate() {
            let normal = self.base.normals.get(i).copied().unwrap_or([0.0; 3]);
            put(out, position);
            put(out, &normal);
        }
    }

    /// Appends the S10 vertex stream 1 (40 B per vertex: uv @0, tangent @8, linear rgba
    /// colour @24) to `out`. Absent attributes take the S10 defaults: uv (0, 0),
    /// tangent `default_tangent(normal)`, colour (1, 1, 1, 1). Tangents are not generated
    /// here: call [`AttributedMesh::ensure_tangents`] first for meshes with uvs.
    pub fn write_stream1_bytes(&self, out: &mut Vec<u8>) {
        let uvs = self.uvs.as_deref().unwrap_or(&[]);
        let tangents = self.tangents.as_deref().unwrap_or(&[]);
        let colors = self.colors.as_deref().unwrap_or(&[]);
        out.reserve(self.base.positions.len() * STREAM1_STRIDE);
        for i in 0..self.base.positions.len() {
            let uv = uvs.get(i).copied().unwrap_or(DEFAULT_UV);
            let tangent = tangents.get(i).copied().unwrap_or_else(|| {
                default_tangent(self.base.normals.get(i).copied().unwrap_or([0.0, 0.0, 1.0]))
            });
            let color = colors.get(i).copied().unwrap_or(DEFAULT_COLOR);
            put(out, &uv);
            put(out, &tangent);
            put(out, &color);
        }
    }

    /// Appends the indices as little-endian u32 (4 B each) to `out`.
    pub fn write_index_bytes(&self, out: &mut Vec<u8>) {
        out.reserve(self.base.indices.len() * 4);
        for index in &self.base.indices {
            out.extend_from_slice(&index.to_le_bytes());
        }
    }
}

fn put(out: &mut Vec<u8>, values: &[f32]) {
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}
