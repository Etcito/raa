//! The cluster mesh of a bake: `ClusterMesh` with its section header, the section bytes, the
//! structural `validate()`, and the uniform cut of a finished DAG.
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-asset/src/scene.rs`, lines 10-35, revision 7e465b5.

use crate::error::{ClusterError, INVALID_DATA};
use crate::packed::{
    MAX_CLUSTER_TRIANGLES, MAX_CLUSTER_VERTICES, PACKED_CLUSTER_SIZE, PACKED_NODE_SIZE,
    PackedCluster, PackedNode, PackedVertex, unpack_triangle,
};

/// The baked DAG of one mesh in the S29 packed records. A cluster index is its position in
/// `clusters`; `vertex_offset` and `triangle_offset` are relative to `vertices` and `triangles`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClusterMesh {
    pub vertices: Vec<PackedVertex>,
    pub triangles: Vec<u32>,
    pub clusters: Vec<PackedCluster>,
    pub nodes: Vec<PackedNode>,
    pub root_node: u32,
    pub hierarchy_depth: u32,
    pub levels: u32,
    pub quant_origin: [f32; 3],
    pub quant_step: f32,
    pub uv_origin: [f32; 2],
    pub uv_step: [f32; 2],
    pub bounds_center: [f32; 3],
    pub bounds_radius: f32,
    pub source_triangles: u64,
}

/// The scalar fields of a `ClusterMesh`, as the saved section carries them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SectionHeader {
    pub root_node: u32,
    pub hierarchy_depth: u32,
    pub levels: u32,
    pub quant_origin: [f32; 3],
    pub quant_step: f32,
    pub uv_origin: [f32; 2],
    pub uv_step: [f32; 2],
    pub bounds_center: [f32; 3],
    pub bounds_radius: f32,
    pub source_triangles: u64,
}

impl ClusterMesh {
    /// The scalar fields of this mesh.
    pub fn header(&self) -> SectionHeader {
        SectionHeader {
            root_node: self.root_node,
            hierarchy_depth: self.hierarchy_depth,
            levels: self.levels,
            quant_origin: self.quant_origin,
            quant_step: self.quant_step,
            uv_origin: self.uv_origin,
            uv_step: self.uv_step,
            bounds_center: self.bounds_center,
            bounds_radius: self.bounds_radius,
            source_triangles: self.source_triangles,
        }
    }

    /// The cluster records, then the node records, 48 bytes each, little-endian.
    pub fn to_section_bytes(&self) -> (Vec<u8>, Vec<u8>) {
        let mut clusters = Vec::new();
        for cluster in &self.clusters {
            cluster.write_le_bytes(&mut clusters);
        }
        let mut nodes = Vec::new();
        for node in &self.nodes {
            node.write_le_bytes(&mut nodes);
        }
        (clusters, nodes)
    }

    /// Builds a mesh from the section arrays (8 u16 per vertex, one u32 per triangle, 48 bytes per
    /// cluster and per node) and its header, then validates it. Every refusal is `invalid_data`.
    pub fn from_section_parts(
        vertices: &[u16],
        triangles: &[u32],
        clusters: &[u8],
        nodes: &[u8],
        header: SectionHeader,
    ) -> Result<ClusterMesh, ClusterError> {
        if !vertices.len().is_multiple_of(8) {
            return Err(refusal(
                "vertices",
                "the vertex slice is not a multiple of 8 u16",
            ));
        }
        if !clusters.len().is_multiple_of(PACKED_CLUSTER_SIZE) {
            return Err(refusal(
                "clusters",
                "the cluster bytes are not a multiple of 48",
            ));
        }
        if !nodes.len().is_multiple_of(PACKED_NODE_SIZE) {
            return Err(refusal("nodes", "the node bytes are not a multiple of 48"));
        }
        let vertices = vertices
            .chunks_exact(8)
            .map(|chunk| {
                <[u16; 8]>::try_from(chunk)
                    .ok()
                    .map(|data| PackedVertex { data })
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| refusal("vertices", "a vertex record is incomplete"))?;
        let clusters = clusters
            .chunks_exact(PACKED_CLUSTER_SIZE)
            .map(PackedCluster::read_le_bytes)
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| refusal("clusters", "a cluster record is incomplete"))?;
        let nodes = nodes
            .chunks_exact(PACKED_NODE_SIZE)
            .map(PackedNode::read_le_bytes)
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| refusal("nodes", "a node record is incomplete"))?;
        let mesh = ClusterMesh {
            vertices,
            triangles: triangles.to_vec(),
            clusters,
            nodes,
            root_node: header.root_node,
            hierarchy_depth: header.hierarchy_depth,
            levels: header.levels,
            quant_origin: header.quant_origin,
            quant_step: header.quant_step,
            uv_origin: header.uv_origin,
            uv_step: header.uv_step,
            bounds_center: header.bounds_center,
            bounds_radius: header.bounds_radius,
            source_triangles: header.source_triangles,
        };
        mesh.validate()?;
        Ok(mesh)
    }

    /// The structural validation of the section: every range, count and sphere, the reachability
    /// of every cluster exactly once from `root_node`, and the monotonic error toward the root.
    /// It returns the first failure, with the field path it names.
    pub fn validate(&self) -> Result<(), ClusterError> {
        if self.vertices.is_empty() {
            return Err(refusal("vertices", "no vertex"));
        }
        if self.triangles.is_empty() {
            return Err(refusal("triangles", "no triangle"));
        }
        if self.clusters.is_empty() {
            return Err(refusal("clusters", "no cluster"));
        }
        if self.nodes.is_empty() {
            return Err(refusal("nodes", "no node"));
        }
        self.validate_header()?;
        self.validate_clusters()?;
        self.validate_nodes()?;
        self.validate_walk()
    }

    fn validate_header(&self) -> Result<(), ClusterError> {
        if self.root_node as usize >= self.nodes.len() {
            return Err(refusal("rootNode", "rootNode is not a node index"));
        }
        if !(1..=64).contains(&self.hierarchy_depth) {
            return Err(refusal("hierarchyDepth", "hierarchyDepth must be 1..=64"));
        }
        if !(1..=256).contains(&self.levels) {
            return Err(refusal("levels", "levels must be 1..=256"));
        }
        if !(self.quant_step.is_finite() && self.quant_step > 0.0) {
            return Err(refusal(
                "quantStep",
                "quantStep must be finite and above zero",
            ));
        }
        if !self.uv_step.iter().all(|step| non_negative(*step)) {
            return Err(refusal("uvStep", "uvStep must be finite and not negative"));
        }
        if !finite(&self.quant_origin) {
            return Err(refusal("quantOrigin", "quantOrigin must be finite"));
        }
        if !finite(&self.uv_origin) {
            return Err(refusal("uvOrigin", "uvOrigin must be finite"));
        }
        if !finite(&self.bounds_center) {
            return Err(refusal("boundsCenter", "boundsCenter must be finite"));
        }
        if !non_negative(self.bounds_radius) {
            return Err(refusal(
                "boundsRadius",
                "boundsRadius must be finite and not negative",
            ));
        }
        Ok(())
    }

    fn validate_clusters(&self) -> Result<(), ClusterError> {
        for (i, cluster) in self.clusters.iter().enumerate() {
            let vertices = cluster.vertex_count();
            let triangles = cluster.triangle_count();
            let counts_ok = cluster.counts >> 24 == 0
                && (1..=MAX_CLUSTER_VERTICES as u32).contains(&vertices)
                && (1..=MAX_CLUSTER_TRIANGLES as u32).contains(&triangles)
                && cluster.level() < self.levels;
            if !counts_ok {
                return Err(refusal(
                    format!("clusters[{i}].counts"),
                    "vertex count 1..=255, triangle count 1..=128, level below levels",
                ));
            }
            if !(sphere_ok(cluster.center, cluster.radius)
                && sphere_ok(cluster.lod_center, cluster.lod_radius)
                && non_negative(cluster.lod_error))
            {
                return Err(refusal(
                    format!("clusters[{i}].bounds"),
                    "bounds and lod_error must be finite, radii and errors not negative",
                ));
            }
            if span(&self.vertices, cluster.vertex_offset, vertices).is_none() {
                return Err(refusal(
                    format!("clusters[{i}].vertex_offset"),
                    "the vertex range is outside vertices",
                ));
            }
            let Some(packed) = span(&self.triangles, cluster.triangle_offset, triangles) else {
                return Err(refusal(
                    format!("clusters[{i}].triangle_offset"),
                    "the triangle range is outside triangles",
                ));
            };
            for (j, triangle) in packed.iter().enumerate() {
                if unpack_triangle(*triangle)
                    .iter()
                    .any(|&index| u32::from(index) >= vertices)
                {
                    return Err(refusal(
                        format!("clusters[{i}].triangle[{j}]"),
                        format!("a corner names a vertex outside the cluster's {vertices}"),
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_nodes(&self) -> Result<(), ClusterError> {
        for (i, node) in self.nodes.iter().enumerate() {
            if node.is_leaf > 1 {
                return Err(refusal(
                    format!("nodes[{i}].is_leaf"),
                    "is_leaf must be 0 or 1",
                ));
            }
            if node.count == 0 {
                return Err(refusal(
                    format!("nodes[{i}].count"),
                    "count must be at least 1",
                ));
            }
            let limit = if node.is_leaf == 1 {
                self.clusters.len()
            } else {
                self.nodes.len()
            };
            if u64::from(node.first) + u64::from(node.count) > limit as u64 {
                return Err(refusal(
                    format!("nodes[{i}].first"),
                    "first + count leaves the cluster or node array",
                ));
            }
            if !(sphere_ok(node.center, node.radius) && sphere_ok(node.lod_center, node.lod_radius))
            {
                return Err(refusal(
                    format!("nodes[{i}].bounds"),
                    "bounds must be finite, radii not negative",
                ));
            }
            if !non_negative(node.max_parent_error) {
                return Err(refusal(
                    format!("nodes[{i}].max_parent_error"),
                    "max_parent_error must be finite and not negative",
                ));
            }
        }
        Ok(())
    }

    /// Walks the hierarchy from `root_node` with an explicit stack: every node once, within
    /// `hierarchy_depth`, every cluster through exactly one leaf, and no error below its parent's.
    fn validate_walk(&self) -> Result<(), ClusterError> {
        let mut visited = vec![false; self.nodes.len()];
        let mut reached = vec![false; self.clusters.len()];
        let mut stack = vec![(self.root_node as usize, 1u32, None::<usize>)];
        while let Some((index, depth, parent)) = stack.pop() {
            let Some(node) = self.nodes.get(index) else {
                return Err(refusal("nodes", "a reached node index is out of range"));
            };
            let Some(seen) = visited.get_mut(index) else {
                return Err(refusal("nodes", "a reached node index is out of range"));
            };
            if *seen {
                let path = match parent {
                    Some(parent) => format!("nodes[{parent}].first"),
                    None => "rootNode".to_string(),
                };
                return Err(refusal(
                    path,
                    "a node is reached twice: a cycle or a shared node",
                ));
            }
            *seen = true;
            if depth > self.hierarchy_depth {
                return Err(refusal(
                    "hierarchyDepth",
                    "a node lies deeper than hierarchyDepth",
                ));
            }
            let first = node.first as usize;
            let end = first + node.count as usize;
            if node.is_leaf == 1 {
                for cluster in first..end {
                    let (Some(flag), Some(data)) =
                        (reached.get_mut(cluster), self.clusters.get(cluster))
                    else {
                        return Err(refusal("clusters", "a leaf names a cluster out of range"));
                    };
                    if *flag {
                        return Err(refusal(
                            format!("nodes[{index}].first"),
                            format!("cluster {cluster} is listed by more than one leaf"),
                        ));
                    }
                    *flag = true;
                    if data.lod_error > node.max_parent_error {
                        return Err(refusal(
                            format!("nodes[{index}].max_parent_error"),
                            "a cluster error is above its leaf's max_parent_error",
                        ));
                    }
                }
            } else {
                for child in (first..end).rev() {
                    let Some(child_node) = self.nodes.get(child) else {
                        return Err(refusal("nodes", "a child index is out of range"));
                    };
                    if child_node.max_parent_error > node.max_parent_error {
                        return Err(refusal(
                            format!("nodes[{index}].max_parent_error"),
                            "a child error is above its parent's max_parent_error",
                        ));
                    }
                    stack.push((child, depth + 1, Some(index)));
                }
            }
        }
        match reached.iter().position(|seen| !*seen) {
            Some(cluster) => Err(refusal(
                format!("clusters[{cluster}]"),
                "the cluster is not reached from rootNode through exactly one leaf",
            )),
            None => Ok(()),
        }
    }
}

fn refusal(path: impl Into<String>, message: impl Into<String>) -> ClusterError {
    ClusterError::at(INVALID_DATA, path, message)
}

fn finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn non_negative(value: f32) -> bool {
    value.is_finite() && value >= 0.0
}

fn sphere_ok(center: [f32; 3], radius: f32) -> bool {
    finite(&center) && non_negative(radius)
}

/// The `count` items at `offset`, or None when the range leaves `items`.
fn span<T>(items: &[T], offset: u32, count: u32) -> Option<&[T]> {
    items.get(offset as usize..)?.get(..count as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bounds::Sphere;
    use crate::packed::{ROOT_PARENT_ERROR, pack_counts, pack_triangle};
    use raa_math::Vec3;

    fn unit_sphere(x: f32) -> Sphere {
        Sphere {
            center: Vec3::new(x, 0.0, 0.0),
            radius: 1.0,
        }
    }

    fn node(
        sphere: Sphere,
        max_parent_error: f32,
        first: u32,
        count: u32,
        is_leaf: u32,
    ) -> PackedNode {
        PackedNode {
            center: sphere.center.to_array(),
            radius: sphere.radius,
            lod_center: sphere.center.to_array(),
            lod_radius: sphere.radius,
            max_parent_error,
            first,
            count,
            is_leaf,
        }
    }

    /// Appends one cluster of three vertices and one triangle at the end of `mesh`.
    fn push_cluster(mesh: &mut ClusterMesh, x: f32, lod_error: f32) {
        let cluster = PackedCluster {
            center: [x, 0.0, 0.0],
            radius: 1.0,
            lod_center: [x, 0.0, 0.0],
            lod_radius: 1.0,
            lod_error,
            vertex_offset: mesh.vertices.len() as u32,
            triangle_offset: mesh.triangles.len() as u32,
            counts: pack_counts(3, 1, 0),
        };
        mesh.vertices.extend([PackedVertex::default(); 3]);
        mesh.triangles.push(pack_triangle(0, 1, 2));
        mesh.clusters.push(cluster);
    }

    /// Two clusters under one leaf (node 0), and the root (node 1), the only inner node.
    fn two_cluster_mesh() -> ClusterMesh {
        let mut mesh = ClusterMesh {
            hierarchy_depth: 2,
            levels: 1,
            root_node: 1,
            quant_step: 0.01,
            uv_step: [0.001, 0.001],
            source_triangles: 2,
            ..ClusterMesh::default()
        };
        push_cluster(&mut mesh, 0.0, 0.1);
        push_cluster(&mut mesh, 2.0, 0.2);
        let leaf = Sphere::merged_all([unit_sphere(0.0), unit_sphere(2.0)]);
        mesh.nodes.push(node(leaf, 0.5, 0, 2, 1));
        mesh.nodes.push(node(leaf, ROOT_PARENT_ERROR, 0, 1, 0));
        mesh.bounds_center = leaf.center.to_array();
        mesh.bounds_radius = leaf.radius;
        mesh
    }

    fn vertex_words(mesh: &ClusterMesh) -> Vec<u16> {
        mesh.vertices.iter().flat_map(|v| v.data).collect()
    }

    #[test]
    fn two_cluster_mesh_validates() {
        assert_eq!(two_cluster_mesh().validate(), Ok(()));
    }

    #[test]
    fn leaf_and_root_spheres_contain_the_cluster_spheres() {
        let mesh = two_cluster_mesh();
        let leaf = Sphere {
            center: Vec3::from_array(mesh.nodes[0].center),
            radius: mesh.nodes[0].radius,
        };
        assert!(leaf.contains(unit_sphere(0.0)));
        assert!(leaf.contains(unit_sphere(2.0)));
        assert!(!leaf.contains(unit_sphere(9.0)));
    }

    #[test]
    fn section_bytes_then_section_parts_give_an_equal_mesh() {
        let mesh = two_cluster_mesh();
        let (clusters, nodes) = mesh.to_section_bytes();
        assert_eq!(clusters.len(), 2 * PACKED_CLUSTER_SIZE);
        assert_eq!(nodes.len(), 2 * PACKED_NODE_SIZE);
        let back = ClusterMesh::from_section_parts(
            &vertex_words(&mesh),
            &mesh.triangles,
            &clusters,
            &nodes,
            mesh.header(),
        );
        assert_eq!(back, Ok(mesh));
    }

    #[test]
    fn section_parts_refuse_bad_byte_lengths_with_the_field_path() {
        let mesh = two_cluster_mesh();
        let (clusters, nodes) = mesh.to_section_bytes();
        let words = vertex_words(&mesh);
        let header = mesh.header();
        let cases = [
            (
                ClusterMesh::from_section_parts(
                    &words[..15],
                    &mesh.triangles,
                    &clusters,
                    &nodes,
                    header,
                ),
                "vertices",
            ),
            (
                ClusterMesh::from_section_parts(
                    &words,
                    &mesh.triangles,
                    &clusters[..47],
                    &nodes,
                    header,
                ),
                "clusters",
            ),
            (
                ClusterMesh::from_section_parts(
                    &words,
                    &mesh.triangles,
                    &clusters,
                    &nodes[..47],
                    header,
                ),
                "nodes",
            ),
        ];
        for (result, path) in cases {
            let err = result.unwrap_err();
            assert_eq!(err.code(), INVALID_DATA);
            assert_eq!(err.path, path);
        }
    }

    #[test]
    fn refuses_each_rule_with_its_path() {
        type Mutate = fn(&mut ClusterMesh);
        let cases: [(Mutate, &str); 25] = [
            (|m| m.vertices.clear(), "vertices"),
            (|m| m.triangles.clear(), "triangles"),
            (|m| m.clusters.clear(), "clusters"),
            (
                |m| m.triangles[0] = pack_triangle(0, 1, 3),
                "clusters[0].triangle[0]",
            ),
            (
                |m| m.clusters[1].counts = pack_counts(0, 1, 0),
                "clusters[1].counts",
            ),
            (
                |m| m.clusters[0].counts = pack_counts(3, 1, 1),
                "clusters[0].counts",
            ),
            (|m| m.clusters[0].radius = f32::NAN, "clusters[0].bounds"),
            (
                |m| m.clusters[1].vertex_offset = 5,
                "clusters[1].vertex_offset",
            ),
            (
                |m| m.clusters[1].triangle_offset = 2,
                "clusters[1].triangle_offset",
            ),
            (|m| m.nodes[0].is_leaf = 2, "nodes[0].is_leaf"),
            (|m| m.nodes[1].count = 3, "nodes[1].first"),
            (|m| m.nodes[0].radius = -1.0, "nodes[0].bounds"),
            (
                |m| m.nodes[0].max_parent_error = f32::INFINITY,
                "nodes[0].max_parent_error",
            ),
            (
                |m| m.nodes[0].max_parent_error = 0.15,
                "nodes[0].max_parent_error",
            ),
            (
                |m| m.nodes[1].max_parent_error = 0.1,
                "nodes[1].max_parent_error",
            ),
            (|m| m.nodes[1].first = 1, "nodes[1].first"),
            (|m| push_cluster(m, 4.0, 0.3), "clusters[2]"),
            (|m| m.root_node = 2, "rootNode"),
            (|m| m.hierarchy_depth = 0, "hierarchyDepth"),
            (|m| m.levels = 0, "levels"),
            (|m| m.quant_step = 0.0, "quantStep"),
            (|m| m.uv_step[1] = -1.0, "uvStep"),
            (|m| m.bounds_center[0] = f32::NAN, "boundsCenter"),
            (|m| m.bounds_radius = f32::NAN, "boundsRadius"),
            (
                |m| {
                    let (leaf, root) = (m.nodes[0], m.nodes[1]);
                    m.nodes = vec![
                        leaf,
                        PackedNode { count: 1, ..leaf },
                        PackedNode { count: 2, ..root },
                    ];
                    m.root_node = 2;
                },
                "nodes[1].first",
            ),
        ];
        for (mutate, path) in cases {
            let mut mesh = two_cluster_mesh();
            mutate(&mut mesh);
            let err = mesh.validate().unwrap_err();
            assert_eq!(err.code(), INVALID_DATA, "{err}");
            assert_eq!(err.path, path, "{err}");
        }
    }
}
