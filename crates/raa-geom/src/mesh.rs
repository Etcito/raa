//! `MeshData`: the indexed triangle list shared by providers and consumers (S2).

use raa_math::{Aabb, Mat4, Vec3};

use crate::GeomError;

/// Indexed triangle list in its own space (S2).
///
/// Invariants, checked by [`MeshData::validate`]: `positions.len() == normals.len()`,
/// `indices.len() % 3 == 0`, every index is in range and every value is finite.
/// Hard edges (caps, box faces) duplicate vertices; smooth surfaces share them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

impl MeshData {
    pub fn new() -> MeshData {
        MeshData::default()
    }

    /// Empties the mesh; every buffer keeps its capacity.
    pub fn clear(&mut self) {
        self.positions.clear();
        self.normals.clear();
        self.indices.clear();
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Checks the S2 rules; the first broken rule becomes the `geom.invalid_mesh` reason.
    pub fn validate(&self) -> Result<(), GeomError> {
        if self.positions.len() != self.normals.len() {
            return Err(GeomError::InvalidMesh(format!(
                "positions length {} does not match normals length {}",
                self.positions.len(),
                self.normals.len()
            )));
        }
        if !self.indices.len().is_multiple_of(3) {
            return Err(GeomError::InvalidMesh(format!(
                "index count {} is not a multiple of 3",
                self.indices.len()
            )));
        }
        let vertex_count = self.positions.len();
        for (offset, index) in self.indices.iter().enumerate() {
            if *index as usize >= vertex_count {
                return Err(GeomError::InvalidMesh(format!(
                    "index {index} at offset {offset} is out of range for {vertex_count} vertices"
                )));
            }
        }
        for (i, position) in self.positions.iter().enumerate() {
            if !position.iter().all(|v| v.is_finite()) {
                return Err(GeomError::InvalidMesh(format!(
                    "positions[{i}] is not finite"
                )));
            }
        }
        for (i, normal) in self.normals.iter().enumerate() {
            if !normal.iter().all(|v| v.is_finite()) {
                return Err(GeomError::InvalidMesh(format!(
                    "normals[{i}] is not finite"
                )));
            }
        }
        Ok(())
    }

    /// Appends `other`, transformed by `m`.
    ///
    /// Appended indices are offset by this mesh's current vertex count. Positions go
    /// through [`Mat4::transform_point`]; normals go through [`Mat4::normal_matrix`] and
    /// are renormalised (a zero normal stays zero). When `det(m) < 0` every appended
    /// triangle's winding is flipped (2nd and 3rd index swapped) so it follows the mirror.
    pub fn append_transformed(&mut self, other: &MeshData, m: &Mat4) {
        let base = self.positions.len() as u32;
        let normal_matrix = m.normal_matrix();
        let flip = m.determinant() < 0.0;
        self.positions.reserve(other.positions.len());
        for position in &other.positions {
            self.positions
                .push(m.transform_point(Vec3::from_array(*position)).to_array());
        }
        self.normals.reserve(other.normals.len());
        for normal in &other.normals {
            let v = Vec3::from_array(*normal);
            let t = Vec3::new(
                normal_matrix[0][0] * v.x + normal_matrix[1][0] * v.y + normal_matrix[2][0] * v.z,
                normal_matrix[0][1] * v.x + normal_matrix[1][1] * v.y + normal_matrix[2][1] * v.z,
                normal_matrix[0][2] * v.x + normal_matrix[1][2] * v.y + normal_matrix[2][2] * v.z,
            );
            self.normals.push(t.normalize_or_zero().to_array());
        }
        self.indices.reserve(other.indices.len());
        for triangle in other.indices.chunks_exact(3) {
            if let [a, b, c] = triangle {
                self.indices.push(*a + base);
                if flip {
                    self.indices.push(*c + base);
                    self.indices.push(*b + base);
                } else {
                    self.indices.push(*b + base);
                    self.indices.push(*c + base);
                }
            }
        }
    }

    /// Bounds of the positions; `Aabb::EMPTY` for an empty mesh.
    pub fn bounds(&self) -> Aabb {
        Aabb::from_points(&self.positions)
    }

    /// Clears `out` and writes the S1 vertex layout: position f32x3 then normal f32x3,
    /// little-endian, 24 bytes per vertex. A missing normal (invalid mesh) is written
    /// as zero.
    pub fn write_vertex_bytes(&self, out: &mut Vec<u8>) {
        out.clear();
        out.reserve(self.positions.len() * 24);
        for (i, position) in self.positions.iter().enumerate() {
            for v in position {
                out.extend_from_slice(&v.to_le_bytes());
            }
            let normal = self.normals.get(i).copied().unwrap_or([0.0; 3]);
            for v in &normal {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    /// Clears `out` and writes the indices as little-endian u32, 4 bytes each.
    pub fn write_index_bytes(&self, out: &mut Vec<u8>) {
        out.clear();
        out.reserve(self.indices.len() * 4);
        for index in &self.indices {
            out.extend_from_slice(&index.to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_math::{Quat, approx_eq};

    fn valid_triangle() -> MeshData {
        MeshData {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            indices: vec![0, 1, 2],
        }
    }

    fn assert_invalid(mesh: &MeshData) -> String {
        let err = mesh.validate().expect_err("mesh must be refused");
        assert_eq!(err.code(), "geom.invalid_mesh");
        err.to_string()
    }

    #[test]
    fn validate_accepts_a_well_formed_mesh() {
        assert_eq!(valid_triangle().validate(), Ok(()));
    }

    #[test]
    fn validate_refuses_positions_normals_length_mismatch() {
        let mut mesh = valid_triangle();
        mesh.normals.pop();
        let message = assert_invalid(&mesh);
        assert!(message.contains("positions length 3"), "{message}");
        assert!(message.contains("normals length 2"), "{message}");
    }

    #[test]
    fn validate_refuses_index_count_not_a_multiple_of_three() {
        let mut mesh = valid_triangle();
        mesh.indices = vec![0, 1];
        let message = assert_invalid(&mesh);
        assert!(message.contains("not a multiple of 3"), "{message}");
    }

    #[test]
    fn validate_refuses_out_of_range_index_and_names_the_first_one() {
        let mut mesh = valid_triangle();
        mesh.indices = vec![0, 1, 2, 2, 7, 0, 9, 0, 1];
        let message = assert_invalid(&mesh);
        assert!(message.contains("index 7 at offset 4"), "{message}");
        assert!(!message.contains("index 9"), "{message}");
    }

    #[test]
    fn validate_refuses_non_finite_positions_and_normals() {
        let mut mesh = valid_triangle();
        mesh.positions[1][2] = f32::NAN;
        let message = assert_invalid(&mesh);
        assert!(message.contains("positions[1]"), "{message}");

        let mut mesh = valid_triangle();
        mesh.normals[0][0] = f32::INFINITY;
        let message = assert_invalid(&mesh);
        assert!(message.contains("normals[0]"), "{message}");

        assert_eq!(valid_triangle().validate(), Ok(()));
    }

    #[test]
    fn append_transformed_offsets_indices_and_keeps_winding() {
        let source = valid_triangle();
        let mut mesh = valid_triangle();
        let m = Mat4::from_trs(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_rotation_y(0.5),
            Vec3::splat(2.0),
        );
        mesh.append_transformed(&source, &m);
        assert_eq!(mesh.vertex_count(), 6);
        assert_eq!(mesh.triangle_count(), 2);
        assert_eq!(mesh.indices, vec![0, 1, 2, 3, 4, 5]);
        assert!(mesh.positions[3..].iter().all(|p| p[2] > 0.0));
        assert_eq!(mesh.validate(), Ok(()));
    }

    #[test]
    fn append_transformed_mirror_flips_winding_and_keeps_normals_unit() {
        let mut source = valid_triangle();
        source.normals = vec![[1.0, 1.0, 1.0]; 3];
        let mirror = Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0));
        let mut mesh = MeshData::new();
        mesh.append_transformed(&source, &mirror);
        assert_eq!(mesh.indices, vec![0, 2, 1]);
        assert_eq!(mesh.positions[0], [0.0, 0.0, 0.0]);
        assert_eq!(mesh.positions[1], [-1.0, 0.0, 0.0]);
        assert_eq!(mesh.positions[2], [0.0, 1.0, 0.0]);
        for normal in &mesh.normals {
            assert!(approx_eq(Vec3::from_array(*normal).length(), 1.0, 1e-6));
        }
        assert_eq!(mesh.validate(), Ok(()));
    }

    #[test]
    fn append_transformed_uses_the_inverse_transpose_for_normals() {
        let mut source = valid_triangle();
        source.normals = vec![[1.0, 1.0, 0.0]; 3];
        let rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let m = Mat4::from_trs(Vec3::ZERO, rotation, Vec3::new(2.0, 1.0, 1.0));
        let mut mesh = MeshData::new();
        mesh.append_transformed(&source, &m);
        let expected = rotation
            .rotate(Vec3::new(0.5, 1.0, 0.0))
            .normalize_or_zero();
        for normal in &mesh.normals {
            let actual = Vec3::from_array(*normal);
            assert!(approx_eq(actual.x, expected.x, 1e-5));
            assert!(approx_eq(actual.y, expected.y, 1e-5));
            assert!(approx_eq(actual.z, expected.z, 1e-5));
        }
    }

    #[test]
    fn append_transformed_leaves_zero_normals_zero_and_renormalises() {
        let mut source = valid_triangle();
        source.normals = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]];
        let mut mesh = MeshData::new();
        mesh.append_transformed(&source, &Mat4::from_scale(Vec3::new(2.0, 1.0, 0.5)));
        assert_eq!(mesh.normals[0], [0.0, 0.0, 0.0]);
        assert_eq!(mesh.normals[2], [0.0, 0.0, 0.0]);
        let middle = Vec3::from_array(mesh.normals[1]);
        assert!(approx_eq(middle.length(), 1.0, 1e-6));
        assert!(middle.x < middle.y && middle.y < middle.z);
    }

    #[test]
    fn bounds_cover_all_positions_and_empty_mesh_is_empty() {
        let bounds = valid_triangle().bounds();
        assert_eq!(bounds.min, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(bounds.max, Vec3::new(1.0, 1.0, 0.0));
        assert_eq!(MeshData::new().bounds(), Aabb::EMPTY);
    }

    #[test]
    fn write_vertex_bytes_writes_24_bytes_per_vertex_little_endian() {
        let mesh = valid_triangle();
        let mut out = vec![0xAA_u8; 7];
        mesh.write_vertex_bytes(&mut out);
        assert_eq!(out.len(), 24 * mesh.vertex_count());
        assert_eq!(out[0..4], mesh.positions[0][0].to_le_bytes());
        assert_eq!(out[12..16], mesh.normals[0][0].to_le_bytes());
        assert_eq!(out[20..24], mesh.normals[0][2].to_le_bytes());
        assert_eq!(out.len(), 72);
    }

    #[test]
    fn write_index_bytes_writes_little_endian_u32() {
        let mesh = valid_triangle();
        let mut out = vec![0xAA_u8; 5];
        mesh.write_index_bytes(&mut out);
        assert_eq!(out.len(), 4 * mesh.indices.len());
        assert_eq!(out[0..4], 0_u32.to_le_bytes());
        assert_eq!(out[8..12], 2_u32.to_le_bytes());
    }

    #[test]
    fn clear_empties_the_mesh_and_keeps_capacity() {
        let mut mesh = valid_triangle();
        mesh.positions.reserve(16);
        mesh.normals.reserve(16);
        mesh.indices.reserve(16);
        let capacities = (
            mesh.positions.capacity(),
            mesh.normals.capacity(),
            mesh.indices.capacity(),
        );
        mesh.clear();
        assert_eq!(mesh.vertex_count(), 0);
        assert_eq!(mesh.triangle_count(), 0);
        assert!(mesh.positions.is_empty() && mesh.normals.is_empty() && mesh.indices.is_empty());
        assert_eq!(mesh.positions.capacity(), capacities.0);
        assert_eq!(mesh.normals.capacity(), capacities.1);
        assert_eq!(mesh.indices.capacity(), capacities.2);
    }
}
