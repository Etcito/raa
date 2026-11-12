//! Inputs of the cluster build: the borrowed mesh with optional uvs, the owned copy the sliced
//! bake keeps, and the `BuildOptions` with their validation.
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-builder/src/dag.rs`, lines 23-41, revision 7e465b5.

use raa_geom::MeshData;
use raa_math::Vec3;

use crate::error::{ClusterError, EMPTY, INVALID_MESH, INVALID_OPTIONS};

/// A borrowed build input: the mesh and, when present, one uv per vertex.
#[derive(Clone, Copy, Debug)]
pub struct ClusterInput<'a> {
    pub mesh: &'a MeshData,
    pub uvs: Option<&'a [[f32; 2]]>,
}

impl ClusterInput<'_> {
    /// Checks the S2 mesh rules, the uv count and finiteness, then that a triangle exists.
    pub fn validate(&self) -> Result<(), ClusterError> {
        self.mesh
            .validate()
            .map_err(|err| ClusterError::at(INVALID_MESH, "mesh", err.to_string()))?;
        if let Some(uvs) = self.uvs {
            if uvs.len() != self.mesh.positions.len() {
                return Err(ClusterError::at(
                    INVALID_MESH,
                    "uvs",
                    format!(
                        "has {} entries for {} vertices",
                        uvs.len(),
                        self.mesh.positions.len()
                    ),
                ));
            }
            if let Some((i, _)) = uvs
                .iter()
                .enumerate()
                .find(|(_, uv)| !uv.iter().all(|v| v.is_finite()))
            {
                return Err(ClusterError::at(
                    INVALID_MESH,
                    format!("uvs[{i}]"),
                    "must be finite",
                ));
            }
        }
        if self.mesh.triangle_count() == 0 {
            return Err(ClusterError::new(EMPTY, "the mesh has no triangle"));
        }
        Ok(())
    }
}

/// An owned build input: the sliced bake copies it so every step can read the same data.
#[derive(Clone, Debug)]
pub struct OwnedClusterInput {
    pub mesh: MeshData,
    pub uvs: Option<Vec<[f32; 2]>>,
}

impl OwnedClusterInput {
    /// The borrowed view of this input.
    pub fn as_input(&self) -> ClusterInput<'_> {
        ClusterInput {
            mesh: &self.mesh,
            uvs: self.uvs.as_deref(),
        }
    }

    /// A copy of a borrowed input.
    pub fn from_input(input: &ClusterInput<'_>) -> OwnedClusterInput {
        OwnedClusterInput {
            mesh: input.mesh.clone(),
            uvs: input.uvs.map(|uvs| uvs.to_vec()),
        }
    }

    /// Copies the triangles of `first_index..first_index + index_count` with only the vertices
    /// they use, in first-use order. The range must be whole triangles inside `indices`.
    pub fn from_index_range(
        mesh: &MeshData,
        uvs: Option<&[[f32; 2]]>,
        first_index: usize,
        index_count: usize,
    ) -> Result<OwnedClusterInput, ClusterError> {
        let end = first_index
            .checked_add(index_count)
            .ok_or_else(|| ClusterError::at(INVALID_MESH, "indices", "index range overflows"))?;
        if end > mesh.indices.len() {
            return Err(ClusterError::at(
                INVALID_MESH,
                "indices",
                format!(
                    "range {first_index}..{end} is outside {} indices",
                    mesh.indices.len()
                ),
            ));
        }
        if !index_count.is_multiple_of(3) {
            return Err(ClusterError::at(
                INVALID_MESH,
                "indices",
                format!("index count {index_count} is not a multiple of 3"),
            ));
        }
        let range = mesh.indices.get(first_index..end).ok_or_else(|| {
            ClusterError::at(INVALID_MESH, "indices", "index range is outside indices")
        })?;
        let mut remap = vec![u32::MAX; mesh.positions.len()];
        let mut out = OwnedClusterInput {
            mesh: MeshData::new(),
            uvs: uvs.map(|_| Vec::new()),
        };
        out.mesh.indices.reserve(index_count);
        out.mesh.positions.reserve(index_count);
        out.mesh.normals.reserve(index_count);
        for index in range {
            let src = *index as usize;
            let position = *mesh.positions.get(src).ok_or_else(|| missing_vertex(src))?;
            let normal = *mesh.normals.get(src).ok_or_else(|| missing_vertex(src))?;
            let uv = match uvs {
                Some(uvs) => Some(*uvs.get(src).ok_or_else(|| {
                    ClusterError::at(
                        INVALID_MESH,
                        "uvs",
                        format!("has no entry for vertex {src}"),
                    )
                })?),
                None => None,
            };
            let mapped = match remap.get_mut(src) {
                Some(slot) if *slot != u32::MAX => *slot,
                Some(slot) => {
                    let new_index = out.mesh.positions.len() as u32;
                    *slot = new_index;
                    out.mesh.positions.push(position);
                    out.mesh.normals.push(normal);
                    if let (Some(out_uvs), Some(uv)) = (out.uvs.as_mut(), uv) {
                        out_uvs.push(uv);
                    }
                    new_index
                }
                None => return Err(missing_vertex(src)),
            };
            out.mesh.indices.push(mapped);
        }
        Ok(out)
    }

    /// Bounds of the positions and, when present, of the uvs; all zero without vertices or uvs.
    pub fn bounds(&self) -> InputBounds {
        let mut bounds = InputBounds::default();
        if let Some((first, rest)) = self.mesh.positions.split_first() {
            bounds.min = Vec3::from_array(*first);
            bounds.max = bounds.min;
            for position in rest {
                let p = Vec3::from_array(*position);
                bounds.min = bounds.min.min(p);
                bounds.max = bounds.max.max(p);
            }
        }
        if let Some(uvs) = self.uvs.as_deref()
            && let Some((first, rest)) = uvs.split_first()
        {
            bounds.uv_min = *first;
            bounds.uv_max = *first;
            for uv in rest {
                bounds.uv_min = [bounds.uv_min[0].min(uv[0]), bounds.uv_min[1].min(uv[1])];
                bounds.uv_max = [bounds.uv_max[0].max(uv[0]), bounds.uv_max[1].max(uv[1])];
            }
        }
        bounds
    }
}

fn missing_vertex(index: usize) -> ClusterError {
    ClusterError::at(
        INVALID_MESH,
        "indices",
        format!("vertex {index} is missing"),
    )
}

/// Bounds of an owned input: position box and uv rectangle.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct InputBounds {
    pub min: Vec3,
    pub max: Vec3,
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
}

/// Tuning parameters of the cluster build.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BuildOptions {
    pub max_cluster_triangles: usize,
    pub max_cluster_vertices: usize,
    /// A group that cannot shrink below this fraction of its triangles is left as is for this
    /// level and retried in a differently shaped group at the next one.
    pub min_reduction: f32,
    /// Fraction of triangles a group tries to keep per level.
    pub target_ratio: f32,
}

impl Default for BuildOptions {
    fn default() -> BuildOptions {
        BuildOptions {
            max_cluster_triangles: 128,
            max_cluster_vertices: 255,
            min_reduction: 0.85,
            target_ratio: 0.5,
        }
    }
}

impl BuildOptions {
    /// Checks the S29 ranges; the path names the refused field.
    pub fn validate(&self) -> Result<(), ClusterError> {
        if !(8..=128).contains(&self.max_cluster_triangles) {
            return Err(ClusterError::at(
                INVALID_OPTIONS,
                "max_cluster_triangles",
                format!("must be in 8..=128, got {}", self.max_cluster_triangles),
            ));
        }
        if !(16..=255).contains(&self.max_cluster_vertices) {
            return Err(ClusterError::at(
                INVALID_OPTIONS,
                "max_cluster_vertices",
                format!("must be in 16..=255, got {}", self.max_cluster_vertices),
            ));
        }
        if !(self.min_reduction > 0.0 && self.min_reduction < 1.0) {
            return Err(ClusterError::at(
                INVALID_OPTIONS,
                "min_reduction",
                format!("must be in (0, 1), got {}", self.min_reduction),
            ));
        }
        if !(self.target_ratio > 0.0 && self.target_ratio < 1.0) {
            return Err(ClusterError::at(
                INVALID_OPTIONS,
                "target_ratio",
                format!("must be in (0, 1), got {}", self.target_ratio),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad() -> MeshData {
        MeshData {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            indices: vec![0, 1, 2, 1, 3, 2],
        }
    }

    fn input<'a>(mesh: &'a MeshData, uvs: Option<&'a [[f32; 2]]>) -> ClusterInput<'a> {
        ClusterInput { mesh, uvs }
    }

    fn options(triangles: usize, vertices: usize, min: f32, ratio: f32) -> BuildOptions {
        BuildOptions {
            max_cluster_triangles: triangles,
            max_cluster_vertices: vertices,
            min_reduction: min,
            target_ratio: ratio,
        }
    }

    fn refuse(options: BuildOptions, field: &str) {
        let err = options.validate().expect_err("options must be refused");
        assert_eq!(err.code(), "cluster.invalid_options", "{field}");
        assert_eq!(err.path, field);
    }

    fn assert_invalid_mesh(input: &ClusterInput<'_>, path: &str) {
        let err = input.validate().expect_err("mesh must be refused");
        assert_eq!(err.code(), "cluster.invalid_mesh");
        assert_eq!(err.path, path);
    }

    #[test]
    fn default_options_validate() {
        let options = BuildOptions::default();
        assert_eq!(options.validate(), Ok(()));
        assert_eq!(options.max_cluster_triangles, 128);
        assert_eq!(options.max_cluster_vertices, 255);
        assert_eq!(options.min_reduction, 0.85);
        assert_eq!(options.target_ratio, 0.5);
    }

    #[test]
    fn options_outside_the_s29_ranges_are_refused() {
        refuse(options(7, 255, 0.85, 0.5), "max_cluster_triangles");
        refuse(options(129, 255, 0.85, 0.5), "max_cluster_triangles");
        refuse(options(128, 15, 0.85, 0.5), "max_cluster_vertices");
        refuse(options(128, 256, 0.85, 0.5), "max_cluster_vertices");
        refuse(options(128, 255, 0.0, 0.5), "min_reduction");
        refuse(options(128, 255, 1.0, 0.5), "min_reduction");
        refuse(options(128, 255, f32::NAN, 0.5), "min_reduction");
        refuse(options(128, 255, 0.85, 0.0), "target_ratio");
        refuse(options(128, 255, 0.85, 1.0), "target_ratio");
        refuse(options(128, 255, 0.85, f32::NAN), "target_ratio");
    }

    #[test]
    fn input_validate_refuses_an_invalid_mesh() {
        let mesh = quad();
        let uvs = vec![[0.0, 0.0]; 4];
        assert_eq!(input(&mesh, Some(&uvs)).validate(), Ok(()));

        let mut bad_index = quad();
        bad_index.indices[0] = 9;
        assert_invalid_mesh(&input(&bad_index, None), "mesh");

        let mut bad_normals = quad();
        bad_normals.normals.pop();
        assert_invalid_mesh(&input(&bad_normals, None), "mesh");

        let short_uvs = vec![[0.0, 0.0]; 3];
        assert_invalid_mesh(&input(&mesh, Some(&short_uvs)), "uvs");

        let nan_uvs = vec![[0.0, 0.0], [f32::NAN, 0.0], [0.0, 0.0], [0.0, 0.0]];
        assert_invalid_mesh(&input(&mesh, Some(&nan_uvs)), "uvs[1]");
    }

    #[test]
    fn input_validate_refuses_a_mesh_without_triangles() {
        let mesh = MeshData {
            positions: vec![[0.0, 0.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]],
            indices: Vec::new(),
        };
        let err = input(&mesh, None)
            .validate()
            .expect_err("empty mesh must be refused");
        assert_eq!(err.code(), "cluster.empty");
    }

    #[test]
    fn from_input_copies_the_borrowed_input() {
        let mesh = quad();
        let uvs = vec![[0.0, 0.0]; 4];
        let owned = OwnedClusterInput::from_input(&input(&mesh, Some(&uvs)));
        let back = owned.as_input();
        assert_eq!(back.mesh, &mesh);
        assert_eq!(back.uvs, Some(&uvs[..]));
        assert_eq!(back.validate(), Ok(()));
    }

    #[test]
    fn from_index_range_keeps_the_used_vertices_in_first_use_order() {
        let mesh = quad();
        let uvs = vec![[0.0, 0.0], [0.25, 0.0], [0.0, 0.25], [0.25, 0.25]];
        let whole =
            OwnedClusterInput::from_index_range(&mesh, Some(&uvs), 0, 6).expect("range is valid");
        assert_eq!(whole.mesh.positions, mesh.positions);
        assert_eq!(whole.mesh.normals, mesh.normals);
        assert_eq!(whole.mesh.indices, vec![0, 1, 2, 1, 3, 2]);
        assert_eq!(whole.uvs.as_deref(), Some(&uvs[..]));

        let tail =
            OwnedClusterInput::from_index_range(&mesh, Some(&uvs), 3, 3).expect("range is valid");
        assert_eq!(
            tail.mesh.positions,
            vec![mesh.positions[1], mesh.positions[3], mesh.positions[2]]
        );
        assert_eq!(tail.mesh.indices, vec![0, 1, 2]);
        assert_eq!(tail.uvs, Some(vec![uvs[1], uvs[3], uvs[2]]));
    }

    #[test]
    fn from_index_range_refuses_a_range_outside_the_indices() {
        let mesh = quad();
        for (first, count) in [(0, 4), (0, 9), (3, 6), (6, 3), (usize::MAX, 3)] {
            let err = OwnedClusterInput::from_index_range(&mesh, None, first, count)
                .expect_err("range must be refused");
            assert_eq!(err.code(), "cluster.invalid_mesh", "{first}..+{count}");
        }
        let mut missing = quad();
        missing.indices[1] = 12;
        let err = OwnedClusterInput::from_index_range(&missing, None, 0, 6)
            .expect_err("missing vertex must be refused");
        assert_eq!(err.code(), "cluster.invalid_mesh");
    }

    #[test]
    fn bounds_cover_positions_and_uvs_of_a_two_triangle_mesh() {
        let mesh = quad();
        let uvs = vec![[0.1, 0.2], [0.7, 0.4], [0.3, 0.8], [0.5, 0.6]];
        let owned =
            OwnedClusterInput::from_index_range(&mesh, Some(&uvs), 0, 6).expect("range is valid");
        let bounds = owned.bounds();
        assert_eq!(bounds.min, Vec3::ZERO);
        assert_eq!(bounds.max, Vec3::new(1.0, 1.0, 0.0));
        assert_eq!(bounds.uv_min, [0.1, 0.2]);
        assert_eq!(bounds.uv_max, [0.7, 0.8]);
    }

    #[test]
    fn bounds_without_uvs_or_vertices_are_zero() {
        let owned =
            OwnedClusterInput::from_index_range(&quad(), None, 0, 6).expect("range is valid");
        assert_eq!(owned.bounds().uv_min, [0.0, 0.0]);
        assert_eq!(owned.bounds().uv_max, [0.0, 0.0]);
        let empty = OwnedClusterInput {
            mesh: MeshData::new(),
            uvs: None,
        };
        assert_eq!(empty.bounds(), InputBounds::default());
    }
}
