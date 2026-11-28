//! Triangle input for the shared trace BVH. Refit the world bounds for rigid motion.

use std::ops::Range;

use raa_core::RaaError;
use raa_math::{Mat4, Vec3};

use crate::bvh::{Bvh, LEAF, NODE_WORDS, Node, build_bounds};

const POSITION_NORMAL_WORDS: usize = 6;
const VERTEX_WORDS: usize = 9;
const TRIANGLE_WORDS: usize = 4;
const TRIANGLE_HAS_UV: u32 = 1 << 31;
pub(super) const ATTRIBUTE_HEADER_BYTES: u64 = 4;
/// The first storage word gives the vertex stride, followed by indexed triangles and vertices.
pub(super) const GEOMETRY_HEADER_BYTES: u64 = 4;
type TraceVertex = [f32; VERTEX_WORDS];
type TraceTriangle = [u32; TRIANGLE_WORDS];
type TriangleBounds = ([f32; 3], [f32; 3]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Bounds {
    pub(super) min: [f32; 3],
    pub(super) max: [f32; 3],
}

impl Bounds {
    pub(super) fn include(&mut self, other: Self) {
        for axis in 0..3 {
            self.min[axis] = self.min[axis].min(other.min[axis]);
            self.max[axis] = self.max[axis].max(other.max[axis]);
        }
    }

    pub(super) fn overlaps(self, min: [f64; 3], max: [f64; 3]) -> bool {
        (0..3).all(|i| f64::from(self.min[i]) <= max[i] && min[i] <= f64::from(self.max[i]))
    }

    pub(super) fn radial_range(self, origin: [f32; 3]) -> (f64, f64) {
        let mut min_sq = 0.0;
        let mut max_sq = 0.0;
        for (i, origin) in origin.into_iter().map(f64::from).enumerate() {
            let lo = f64::from(self.min[i]) - origin;
            let hi = f64::from(self.max[i]) - origin;
            min_sq += 0.0_f64.max(lo).max(-hi).powi(2);
            max_sq += lo.abs().max(hi.abs()).powi(2);
        }
        (min_sq.sqrt(), max_sq.sqrt())
    }
}

// Bound the number of dirty regions when an update moves many independent meshes.
const MAX_DIRTY_BOUNDS: usize = 32;

fn record_dirty(dirty: &mut Vec<Bounds>, bounds: Option<Bounds>) {
    let Some(bounds) = bounds else { return };
    if dirty.len() == MAX_DIRTY_BOUNDS {
        dirty.last_mut().expect("dirty bounds").include(bounds);
    } else {
        dirty.push(bounds);
    }
}

/// Byte ranges rewritten by the latest geometry updates since the GPU last consumed them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct DirtyBytes {
    pub(super) full: bool,
    pub(super) ranges: Vec<Range<u64>>,
}

impl DirtyBytes {
    pub(super) fn clear(&mut self) {
        self.full = false;
        self.ranges.clear();
    }

    pub(super) fn mark_full(&mut self) {
        self.full = true;
        self.ranges.clear();
    }

    fn add_blocks(&mut self, mut blocks: Vec<u64>, size: u64) {
        if self.full || blocks.is_empty() {
            return;
        }
        blocks.sort_unstable();
        let mut start = blocks[0];
        let mut end = start + size;
        for &block in &blocks[1..] {
            if block <= end {
                end = end.max(block + size);
            } else {
                self.ranges.push(start..end);
                start = block;
                end = block + size;
            }
        }
        self.ranges.push(start..end);
        self.ranges.sort_unstable_by_key(|range| range.start);
        let mut merged: Vec<Range<u64>> = Vec::with_capacity(self.ranges.len());
        for range in self.ranges.drain(..) {
            if let Some(last) = merged.last_mut()
                && range.start <= last.end
            {
                last.end = last.end.max(range.end);
            } else {
                merged.push(range);
            }
        }
        self.ranges = merged;
    }
}

/// One submesh range with the renderer's current material and world transform.
#[derive(Clone, Copy)]
pub struct MeshInput<'a> {
    pub key: u64,
    pub mesh: u32,
    pub positions: &'a [[f32; 3]],
    pub normals: &'a [[f32; 3]],
    pub colors: Option<&'a [[f32; 4]]>,
    pub uvs: Option<&'a [[f32; 2]]>,
    pub indices: &'a [u32],
    pub material: u32,
    pub world: Mat4,
}

#[derive(Clone, Copy, PartialEq)]
struct Part {
    id: u32,
    key: u64,
    material: u32,
    count: usize,
    vertices: usize,
    colored: bool,
    indexed: bool,
    has_uv: bool,
    world: Mat4,
}

impl Part {
    fn new(value: &MeshInput<'_>, normal_matrix: Option<Mat4>) -> Self {
        let count = value.indices.len() / 3;
        Self {
            id: value.mesh,
            key: value.key,
            material: value.material,
            count,
            // Singular transforms use a separate geometric normal at each face.
            vertices: if normal_matrix.is_some() {
                value.positions.len()
            } else {
                count * 3
            },
            colored: value.colors.is_some(),
            indexed: normal_matrix.is_some(),
            has_uv: value
                .uvs
                .is_some_and(|uvs| uvs.len() == value.positions.len()),
            world: value.world,
        }
    }
}

fn same_geometry(a: &Part, b: &Part) -> bool {
    (a.id, a.key, a.count, a.vertices, a.indexed) == (b.id, b.key, b.count, b.vertices, b.indexed)
}

#[derive(Clone, Default)]
pub(super) struct Geometry {
    parts: Vec<Part>,
    bounds: Vec<TriangleBounds>,
    part_bounds: Vec<Option<Bounds>>,
    pub(super) dirty_bounds: Vec<Bounds>,
    /// Bounds changed by every update since a consumer last took them (`take_dirty_log`): unlike `dirty_bounds` (the
    /// latest update only) it survives several updates between two frames of a consumer.
    pub(super) dirty_log: Vec<Bounds>,
    triangles: Vec<TraceTriangle>,
    vertices: Vec<TraceVertex>,
    bvh: Bvh,
    parents: Vec<u32>,
    triangle_node: Vec<u32>,
    triangle_position: Vec<u32>,
    refit_mark: Vec<u32>,
    refit_generation: u32,
    pub revision: u64,
    pub mesh_bytes: Vec<u8>,
    pub node_bytes: Vec<u8>,
    pub(super) mesh_dirty: DirtyBytes,
    pub(super) node_dirty: DirtyBytes,
    pub uv_bytes: Vec<u8>,
    pub uv_revision: u64,
}

impl Geometry {
    pub(super) fn take_dirty_log(&mut self) -> Vec<Bounds> {
        std::mem::take(&mut self.dirty_log)
    }

    pub fn count(&self) -> u32 {
        self.triangles.len() as u32
    }

    pub fn update(
        &mut self,
        inputs: &[MeshInput<'_>],
        storage_limit: u64,
        material_bytes: u64,
    ) -> Result<(), RaaError> {
        let normals: Vec<_> = inputs
            .iter()
            .map(|input| input.world.inverse().map(|matrix| matrix.transpose()))
            .collect();
        let parts: Vec<_> = inputs
            .iter()
            .zip(&normals)
            .map(|(input, &normal)| Part::new(input, normal))
            .collect();
        let count: usize = parts.iter().map(|part| part.count).sum();
        let vertices: usize = parts.iter().map(|part| part.vertices).sum();
        let vertex_words = if parts.iter().any(|part| part.colored) {
            VERTEX_WORDS
        } else {
            POSITION_NORMAL_WORDS
        };
        let mesh_size = geometry_size(count as u64, vertices as u64, vertex_words);
        let uv_size = uv_size(vertices as u64, parts.iter().any(|part| part.has_uv));
        let attr_size = ATTRIBUTE_HEADER_BYTES + uv_size + material_bytes;
        // Round leaf count up to a complete binary tree for a preflight upper bound.
        let max_nodes = 2 * (count as u64).div_ceil(LEAF as u64).next_power_of_two() - 1;
        let bvh_size = max_nodes * (NODE_WORDS * 4) as u64;
        if mesh_size.max(bvh_size).max(attr_size) > storage_limit {
            return Err(RaaError::FrameGraph {
                code: "trace.mesh_capacity",
                message: format!(
                    "Mesh tracing needs {count} triangles in this view. Indexed geometry requires {mesh_size} bytes and the BVH requires up to {bvh_size} bytes. UV and material attributes require {attr_size} bytes. The device storage binding limit is {storage_limit} bytes. Select a render profile without mesh tracing to render this scene at full geometry."
                ),
            });
        }
        if self.parts == parts {
            return Ok(());
        }
        let aligned = self.parts.len() == parts.len();
        let rebuild = !aligned
            || self
                .parts
                .iter()
                .zip(&parts)
                .any(|(old, next)| !same_geometry(old, next));

        // Refit world transforms independently of the local UV attribute stream.
        if rebuild
            || self
                .parts
                .iter()
                .zip(&parts)
                .any(|(a, b)| a.has_uv != b.has_uv)
        {
            self.uv_bytes.clear();
            self.uv_bytes.reserve_exact(uv_size as usize);
            if uv_size != 0 {
                for ((input, part), normal_matrix) in inputs.iter().zip(&parts).zip(&normals) {
                    let uvs = input.uvs.filter(|_| part.has_uv);
                    if let Some(uvs) = uvs {
                        if normal_matrix.is_some() {
                            self.uv_bytes
                                .extend(uvs.iter().flatten().flat_map(|uv| uv.to_le_bytes()));
                        } else {
                            self.uv_bytes.extend(
                                input.indices[..part.count * 3]
                                    .iter()
                                    .flat_map(|&index| uvs[index as usize])
                                    .flat_map(f32::to_le_bytes),
                            );
                        }
                    } else {
                        self.uv_bytes
                            .resize(self.uv_bytes.len() + part.vertices * 8, 0);
                    }
                }
            }
            debug_assert_eq!(self.uv_bytes.len() as u64, uv_size);
            self.uv_revision = self.uv_revision.wrapping_add(1);
        }

        // An unchanged suffix can survive insertions or removals without marking its bounds dirty.
        let prefix = self
            .parts
            .iter()
            .zip(&parts)
            .take_while(|(old, next)| old == next)
            .count();
        let suffix = self.parts[prefix..]
            .iter()
            .rev()
            .zip(parts[prefix..].iter().rev())
            .take_while(|(old, next)| old == next)
            .count();
        let old_end = self.parts.len() - suffix;
        let new_end = parts.len() - suffix;

        self.dirty_bounds.clear();
        for i in prefix..old_end {
            if !aligned || self.parts[i] != parts[i] {
                record_dirty(&mut self.dirty_bounds, self.part_bounds[i]);
            }
        }
        self.triangles.resize(count, [0; TRIANGLE_WORDS]);
        self.vertices.resize(vertices, [0.; VERTEX_WORDS]);
        self.bounds.resize(count, ([0.; 3], [0.; 3]));
        self.part_bounds.resize(parts.len(), None);
        let mut triangle_start = 0;
        let mut vertex_start = 0;
        for (i, ((input, &part), normal_matrix)) in
            inputs.iter().zip(&parts).zip(normals).enumerate()
        {
            if rebuild || self.parts[i] != part {
                self.part_bounds[i] =
                    self.write_part(input, normal_matrix, triangle_start, vertex_start);
                if (prefix..new_end).contains(&i) && (!aligned || self.parts[i] != part) {
                    record_dirty(&mut self.dirty_bounds, self.part_bounds[i]);
                }
            }
            triangle_start += part.count;
            vertex_start += part.vertices;
        }
        let old_vertex_words = self.mesh_bytes.get(..4).map_or(0, |word| {
            u32::from_le_bytes(word.try_into().expect("vertex stride")) as usize
        });
        if rebuild {
            self.bvh = if count == 0 {
                Bvh::default()
            } else {
                build_bounds(&self.bounds)
            };
            self.build_index();
            self.serialize_mesh_full(vertex_words, mesh_size);
            self.serialize_nodes_full();
            self.mesh_dirty.mark_full();
            self.node_dirty.mark_full();
        } else {
            let mut triangle_starts = Vec::new();
            let mut vertex_starts: Vec<(u64, usize, usize)> = Vec::new();
            let mut vertex_blocks = Vec::new();
            let mut leaves = Vec::new();
            let (mut triangle_start, mut vertex_start) = (0, 0);
            for (i, part) in parts.iter().enumerate() {
                if self.parts[i] != *part {
                    for triangle in triangle_start..triangle_start + part.count {
                        let position = self.triangle_position[triangle] as u64;
                        triangle_starts
                            .push(GEOMETRY_HEADER_BYTES + position * (TRIANGLE_WORDS * 4) as u64);
                        leaves.push(self.triangle_node[triangle]);
                    }
                    if part.vertices != 0 {
                        let base =
                            GEOMETRY_HEADER_BYTES + count as u64 * (TRIANGLE_WORDS * 4) as u64;
                        vertex_starts.push((
                            base + vertex_start as u64 * (vertex_words * 4) as u64,
                            vertex_start,
                            part.vertices,
                        ));
                        vertex_blocks.extend((0..part.vertices).map(|vertex| {
                            base + (vertex_start + vertex) as u64 * (vertex_words * 4) as u64
                        }));
                    }
                }
                triangle_start += part.count;
                vertex_start += part.vertices;
            }
            let mut nodes = self.dirty_ancestors(&leaves);
            nodes.sort_unstable_by(|a, b| b.cmp(a));
            let mut node_starts = Vec::with_capacity(nodes.len());
            for i in nodes {
                self.refit(i);
                self.write_node(i);
                node_starts.push(i as u64 * (NODE_WORDS * 4) as u64);
            }
            self.node_dirty
                .add_blocks(node_starts, (NODE_WORDS * 4) as u64);

            if old_vertex_words != vertex_words || self.mesh_bytes.len() as u64 != mesh_size {
                self.serialize_mesh_full(vertex_words, mesh_size);
                self.mesh_dirty.mark_full();
            } else {
                for &start in &triangle_starts {
                    let triangle =
                        ((start - GEOMETRY_HEADER_BYTES) / (TRIANGLE_WORDS * 4) as u64) as usize;
                    self.write_triangle(triangle, start);
                }
                for &(start, first, length) in &vertex_starts {
                    self.write_vertices(first, length, start, vertex_words);
                }
                self.mesh_dirty
                    .add_blocks(triangle_starts, (TRIANGLE_WORDS * 4) as u64);
                self.mesh_dirty
                    .add_blocks(vertex_blocks, (vertex_words * 4) as u64);
            }
        }
        self.parts = parts;
        for i in 0..self.dirty_bounds.len() {
            let b = self.dirty_bounds[i];
            record_dirty(&mut self.dirty_log, Some(b));
        }
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    fn build_index(&mut self) {
        let nodes = self.bvh.nodes.len();
        self.parents.clear();
        self.parents.resize(nodes, u32::MAX);
        self.triangle_node.clear();
        self.triangle_node.resize(self.bvh.order.len(), u32::MAX);
        self.triangle_position.clear();
        self.triangle_position
            .resize(self.bvh.order.len(), u32::MAX);
        self.refit_mark.clear();
        self.refit_mark.resize(nodes, 0);
        self.refit_generation = 0;
        for i in 0..nodes {
            let node = self.bvh.nodes[i];
            let first = node.left as usize;
            if node.count == 0 {
                if first < nodes {
                    self.parents[first] = i as u32;
                }
                if first + 1 < nodes {
                    self.parents[first + 1] = i as u32;
                }
            } else {
                for offset in 0..node.count as usize {
                    let triangle = self.bvh.order[first + offset] as usize;
                    self.triangle_node[triangle] = i as u32;
                    self.triangle_position[triangle] = (first + offset) as u32;
                }
            }
        }
    }

    fn dirty_ancestors(&mut self, leaves: &[u32]) -> Vec<usize> {
        self.refit_generation = self.refit_generation.wrapping_add(1);
        if self.refit_generation == 0 {
            self.refit_mark.fill(0);
            self.refit_generation = 1;
        }
        let generation = self.refit_generation;
        let mut dirty = Vec::new();
        for &leaf in leaves {
            let mut node = leaf as usize;
            while node < self.refit_mark.len() && self.refit_mark[node] != generation {
                self.refit_mark[node] = generation;
                dirty.push(node);
                let parent = self.parents[node];
                if parent == u32::MAX {
                    break;
                }
                node = parent as usize;
            }
        }
        dirty
    }

    fn write_triangle(&mut self, triangle: usize, mut offset: u64) {
        for word in self.triangles[triangle] {
            let end = offset as usize + 4;
            self.mesh_bytes[offset as usize..end].copy_from_slice(&word.to_le_bytes());
            offset += 4;
        }
    }

    fn write_vertices(
        &mut self,
        first: usize,
        length: usize,
        mut offset: u64,
        vertex_words: usize,
    ) {
        for vertex in &self.vertices[first..first + length] {
            for &word in &vertex[..vertex_words] {
                let end = offset as usize + 4;
                self.mesh_bytes[offset as usize..end].copy_from_slice(&word.to_le_bytes());
                offset += 4;
            }
        }
    }

    fn write_node(&mut self, index: usize) {
        let Node {
            min,
            max,
            left,
            count,
        } = self.bvh.nodes[index];
        let offset = index * NODE_WORDS * 4;
        let words = min
            .map(f32::to_bits)
            .into_iter()
            .chain([left])
            .chain(max.map(f32::to_bits))
            .chain([count]);
        for (i, word) in words.enumerate() {
            let start = offset + i * 4;
            self.node_bytes[start..start + 4].copy_from_slice(&word.to_le_bytes());
        }
    }

    fn serialize_mesh_full(&mut self, vertex_words: usize, mesh_size: u64) {
        self.mesh_bytes.clear();
        self.mesh_bytes.reserve_exact(mesh_size as usize);
        self.mesh_bytes
            .extend_from_slice(&(vertex_words as u32).to_le_bytes());
        for &index in &self.bvh.order {
            for word in self.triangles[index as usize] {
                self.mesh_bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
        for vertex in &self.vertices {
            for &word in &vertex[..vertex_words] {
                self.mesh_bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
        debug_assert_eq!(self.mesh_bytes.len() as u64, mesh_size);
    }

    fn serialize_nodes_full(&mut self) {
        self.node_bytes.clear();
        self.node_bytes
            .reserve_exact(self.bvh.nodes.len() * NODE_WORDS * 4);
        for &Node {
            min,
            max,
            left,
            count,
        } in &self.bvh.nodes
        {
            for word in min
                .map(f32::to_bits)
                .into_iter()
                .chain([left])
                .chain(max.map(f32::to_bits))
                .chain([count])
            {
                self.node_bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
    }

    fn write_part(
        &mut self,
        input: &MeshInput<'_>,
        normal_matrix: Option<Mat4>,
        triangle_start: usize,
        vertex_start: usize,
    ) -> Option<Bounds> {
        let position = |i| {
            input
                .world
                .transform_point(Vec3::from_array(input.positions[i]))
        };
        let color = |i: usize| input.colors.map_or([1.; 4], |colors| colors[i]);
        if let Some(normal_matrix) = normal_matrix {
            for i in 0..input.positions.len() {
                let normal = normal_matrix
                    .transform_vector(Vec3::from_array(input.normals[i]))
                    .normalize_or_zero();
                self.vertices[vertex_start + i] = trace_vertex(position(i), normal, color(i));
            }
        }

        let has_uv = input
            .uvs
            .is_some_and(|uvs| uvs.len() == input.positions.len());
        let surface = input.material | if has_uv { TRIANGLE_HAS_UV } else { 0 };
        let mut part: Option<Bounds> = None;
        for (i, indices) in input.indices.chunks_exact(3).enumerate() {
            let [a, b, c] = [indices[0], indices[1], indices[2]].map(|index| index as usize);
            let vertex_ids = if normal_matrix.is_some() {
                [a, b, c].map(|index| vertex_start + index)
            } else {
                let points = [position(a), position(b), position(c)];
                let normal = (points[1] - points[0])
                    .cross(points[2] - points[0])
                    .normalize_or_zero();
                let vertex_ids = [0, 1, 2].map(|corner| vertex_start + i * 3 + corner);
                for ((id, point), index) in vertex_ids.into_iter().zip(points).zip([a, b, c]) {
                    self.vertices[id] = trace_vertex(point, normal, color(index));
                }
                vertex_ids
            };

            let points = vertex_ids.map(|index| vertex_position(&self.vertices[index]));
            let min = std::array::from_fn(|axis| {
                points[0][axis].min(points[1][axis]).min(points[2][axis])
            });
            let max = std::array::from_fn(|axis| {
                points[0][axis].max(points[1][axis]).max(points[2][axis])
            });
            let [a, b, c] = vertex_ids.map(|index| index as u32);
            self.triangles[triangle_start + i] = [a, b, c, surface];
            self.bounds[triangle_start + i] = (min, max);
            let bounds = Bounds { min, max };
            match &mut part {
                Some(part) => part.include(bounds),
                slot => *slot = Some(bounds),
            }
        }
        part
    }

    pub(super) fn world_bounds(&self) -> Option<Bounds> {
        self.bvh.nodes.first().map(|root| Bounds {
            min: root.min,
            max: root.max,
        })
    }

    /// Visit candidate triangles in the same BVH used by mesh rays. The caller's
    /// narrow phase decides whether to accept each triangle.
    pub(super) fn find_triangle<R>(
        &self,
        min: [f64; 3],
        max: [f64; 3],
        mut visit: impl FnMut([[f32; 3]; 3]) -> Option<R>,
    ) -> Option<R> {
        self.find_triangle_in(0, min, max, &mut visit)
    }

    fn find_triangle_in<R>(
        &self,
        index: usize,
        min: [f64; 3],
        max: [f64; 3],
        visit: &mut impl FnMut([[f32; 3]; 3]) -> Option<R>,
    ) -> Option<R> {
        let node = self.bvh.nodes.get(index)?;
        if !(Bounds {
            min: node.min,
            max: node.max,
        })
        .overlaps(min, max)
        {
            return None;
        }
        let first = node.left as usize;
        if node.count == 0 {
            return self
                .find_triangle_in(first, min, max, visit)
                .or_else(|| self.find_triangle_in(first + 1, min, max, visit));
        }
        for &index in &self.bvh.order[first..first + node.count as usize] {
            let index = index as usize;
            let (lo, hi) = self.bounds[index];
            if !(Bounds { min: lo, max: hi }).overlaps(min, max) {
                continue;
            }
            let [a, b, c, _] = self.triangles[index];
            let points = [a, b, c].map(|id| vertex_position(&self.vertices[id as usize]));
            if let Some(hit) = visit(points) {
                return Some(hit);
            }
        }
        None
    }

    fn refit(&mut self, i: usize) {
        let node = self.bvh.nodes[i];
        let first = node.left as usize;
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        let mut merge = |a: [f32; 3], b: [f32; 3]| {
            for axis in 0..3 {
                lo[axis] = lo[axis].min(a[axis]);
                hi[axis] = hi[axis].max(b[axis]);
            }
        };
        if node.count == 0 {
            for child in &self.bvh.nodes[first..first + 2] {
                merge(child.min, child.max);
            }
        } else {
            for &index in &self.bvh.order[first..first + node.count as usize] {
                let (a, b) = self.bounds[index as usize];
                merge(a, b);
            }
        }
        self.bvh.nodes[i].min = lo;
        self.bvh.nodes[i].max = hi;
    }
}

fn geometry_size(triangles: u64, vertices: u64, vertex_words: usize) -> u64 {
    GEOMETRY_HEADER_BYTES + 4 * (triangles * TRIANGLE_WORDS as u64 + vertices * vertex_words as u64)
}

fn uv_size(vertices: u64, has_uv: bool) -> u64 {
    if has_uv { 8 * vertices } else { 0 }
}

pub(super) fn storage_limit(limits: wgpu::Limits) -> u64 {
    limits
        .max_buffer_size
        .min(u64::from(limits.max_storage_buffer_binding_size))
        & !3
}

fn trace_vertex(position: Vec3, normal: Vec3, color: [f32; 4]) -> TraceVertex {
    let [x, y, z] = position.to_array();
    let [nx, ny, nz] = normal.to_array();
    let [r, g, b, _] = color;
    [x, y, z, nx, ny, nz, r, g, b]
}

fn vertex_position(vertex: &TraceVertex) -> [f32; 3] {
    [vertex[0], vertex[1], vertex[2]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_refusal_preserves_the_previous_geometry_and_allows_a_later_update() {
        let original = MeshInput {
            key: 1,
            mesh: 0,
            positions: &[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
            normals: &[[0., 0., 1.]; 3],
            colors: None,
            uvs: None,
            indices: &[0, 1, 2],
            material: 0,
            world: Mat4::IDENTITY,
        };
        let mut geometry = Geometry::default();
        geometry
            .update(std::slice::from_ref(&original), 128, 0)
            .unwrap();
        let revision = geometry.revision;
        let too_many = vec![0; 30];
        let error = geometry
            .update(
                &[MeshInput {
                    indices: &too_many,
                    ..original
                }],
                128,
                0,
            )
            .unwrap_err();
        assert_eq!(error.code(), "trace.mesh_capacity");
        assert!(error.to_string().contains("10 triangles"));
        assert!(error.to_string().contains("128 bytes"));
        assert_eq!(geometry.revision, revision);
        assert_eq!(geometry.count(), 1);

        geometry.update(&[], 128, 0).unwrap();
        assert_eq!(geometry.count(), 0);
        assert_ne!(geometry.revision, revision);
    }

    #[test]
    fn packs_linear_rgb_in_index_order_and_uses_white_for_uncolored_meshes() {
        let colors = [
            [1.5, 0.3, 0.1, 1.0],
            [0.1, 0.7, 0.2, 0.0],
            [0.25, 0.5, 1.0, 0.2],
        ];
        let mut input = MeshInput {
            key: 1,
            mesh: 0,
            positions: &[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
            normals: &[[0., 0., 1.]; 3],
            colors: Some(&colors),
            uvs: None,
            indices: &[2, 0, 1],
            material: 7,
            world: Mat4::IDENTITY,
        };
        let mut geometry = Geometry::default();
        geometry
            .update(std::slice::from_ref(&input), 128, 0)
            .unwrap();
        let lanes = |geometry: &Geometry, corner: usize| {
            let vertex = geometry.triangles[0][corner] as usize;
            let start = 4 + TRIANGLE_WORDS * 4 + (vertex * VERTEX_WORDS + 6) * 4;
            std::array::from_fn::<_, 3, _>(|i| {
                let word = start + i * 4;
                f32::from_le_bytes(geometry.mesh_bytes[word..word + 4].try_into().unwrap())
            })
        };
        for (corner, index) in input.indices.iter().enumerate() {
            let [r, g, b, _] = colors[*index as usize];
            assert_eq!(lanes(&geometry, corner), [r, g, b]);
        }
        assert_eq!(geometry.mesh_bytes.len(), 128);

        input.colors = None;
        input.key += 1;
        geometry.update(&[input], 128, 0).unwrap();
        assert_eq!(geometry.mesh_bytes.len(), 92);
        assert_eq!(geometry.mesh_bytes[0], POSITION_NORMAL_WORDS as u8);
        assert!(
            geometry
                .vertices
                .iter()
                .all(|vertex| vertex[6..] == [1.; 3])
        );
    }
    #[test]
    fn full_intel_payload_keeps_exact_colors_and_uvs_within_separate_128_mib_bindings() {
        // Original Intel Sponza. Four blended placements remain in raster, and
        // 58 of the ray-eligible placements carry authored vertex colors.
        let (triangles, vertices, materials) = (3_738_878, 2_042_098, 28);
        let geometry = geometry_size(triangles, vertices, VERTEX_WORDS);
        let attributes = ATTRIBUTE_HEADER_BYTES + uv_size(vertices, true) + materials * 64;
        let binding_limit = 128 << 20;
        assert_eq!(geometry, 133_337_580);
        assert_eq!(attributes, 16_338_580);
        assert!(geometry < binding_limit);
        assert!(attributes < binding_limit);
        assert!(geometry + attributes > binding_limit);
    }

    #[test]
    fn exact_indexed_uvs_preserve_geometry_and_default_without_coordinates() {
        let positions = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]];
        let texcoords = [[-1.125, 0.125], [2.75, 0.3125], [0.33333334, -2.875]];
        let mut input = MeshInput {
            key: 1,
            mesh: 0,
            positions: &positions,
            normals: &[[0., 0., 1.]; 3],
            colors: None,
            uvs: None,
            indices: &[2, 0, 1],
            material: 7,
            world: Mat4::IDENTITY,
        };
        let mut geometry = Geometry::default();
        geometry
            .update(std::slice::from_ref(&input), 128, 64)
            .unwrap();
        let before = geometry.mesh_bytes.clone();
        let nodes = geometry.node_bytes.clone();
        assert!(geometry.uv_bytes.is_empty());
        input.uvs = Some(&texcoords);
        geometry
            .update(std::slice::from_ref(&input), 128, 64)
            .unwrap();
        assert_eq!(geometry.node_bytes, nodes);
        assert_eq!(&before[..16], &geometry.mesh_bytes[..16]);
        assert_eq!(&before[20..], &geometry.mesh_bytes[20..]);
        assert_eq!(geometry.triangles[0], [2, 0, 1, 7 | TRIANGLE_HAS_UV]);
        assert_eq!(
            geometry.uv_bytes,
            texcoords
                .into_iter()
                .flatten()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        let uv_revision = geometry.uv_revision;
        input.world.cols[3][0] = 1.;
        geometry
            .update(std::slice::from_ref(&input), 128, 64)
            .unwrap();
        assert_eq!(geometry.uv_revision, uv_revision);
        input.world.cols[2][2] = 0.;
        geometry
            .update(std::slice::from_ref(&input), 128, 64)
            .unwrap();
        assert_ne!(geometry.uv_revision, uv_revision);
        assert_eq!(geometry.triangles[0], [0, 1, 2, 7 | TRIANGLE_HAS_UV]);
        assert_eq!(
            geometry.uv_bytes,
            [texcoords[2], texcoords[0], texcoords[1]]
                .into_iter()
                .flatten()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        input.uvs = Some(&texcoords[..1]);
        geometry.update(&[input], 128, 64).unwrap();
        assert_eq!(geometry.triangles[0][3], 7);
        assert!(geometry.uv_bytes.is_empty());
    }

    fn strip(offset: f32, triangles: usize) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
        let mut positions = Vec::with_capacity(triangles * 3);
        let mut indices = Vec::with_capacity(triangles * 3);
        for triangle in 0..triangles {
            let x = offset + triangle as f32;
            positions.extend([[x, 0.0, 0.0], [x + 0.5, 0.0, 0.0], [x, 1.0, 0.0]]);
            let base = (triangle * 3) as u32;
            indices.extend([base, base + 1, base + 2]);
        }
        let normals = vec![[0.0, 0.0, 1.0]; triangles * 3];
        (positions, normals, indices)
    }

    #[test]
    fn refit_of_a_moved_part_matches_a_full_rebuild_bounds() {
        let (p0, n0, i0) = strip(0.0, 2);
        let (p1, n1, i1) = strip(10.0, 2);
        let (p2, n2, i2) = strip(20.0, 2);
        let input = |key, mesh, positions, normals, indices, world| MeshInput {
            key,
            mesh,
            positions,
            normals,
            colors: None,
            uvs: None,
            indices,
            material: 0,
            world,
        };
        let original = [
            input(1, 0, &p0, &n0, &i0, Mat4::IDENTITY),
            input(2, 1, &p1, &n1, &i1, Mat4::IDENTITY),
            input(3, 2, &p2, &n2, &i2, Mat4::IDENTITY),
        ];
        let moved_world = Mat4::from_translation(Vec3::new(3.0, 2.0, 1.0));
        let moved = [
            input(1, 0, &p0, &n0, &i0, moved_world),
            original[1],
            original[2],
        ];
        let mut geometry = Geometry::default();
        geometry.update(&original, 1 << 20, 0).unwrap();
        geometry.update(&moved, 1 << 20, 0).unwrap();

        let mut rebuilt = Geometry::default();
        rebuilt.update(&moved, 1 << 20, 0).unwrap();
        assert_eq!(geometry.bounds, rebuilt.bounds);
        assert_eq!(geometry.part_bounds, rebuilt.part_bounds);
        assert_eq!(geometry.world_bounds(), rebuilt.world_bounds());
        assert_eq!(geometry.count(), rebuilt.count());
    }

    #[test]
    fn incremental_update_writes_only_the_moved_ranges_and_matches_full_serialization() {
        let (p0, n0, i0) = strip(0.0, 2);
        let (p1, n1, i1) = strip(10.0, 2);
        let (p2, n2, i2) = strip(20.0, 2);
        let input = |key, mesh, positions, normals, indices, world| MeshInput {
            key,
            mesh,
            positions,
            normals,
            colors: None,
            uvs: None,
            indices,
            material: 0,
            world,
        };
        let original = [
            input(1, 0, &p0, &n0, &i0, Mat4::IDENTITY),
            input(2, 1, &p1, &n1, &i1, Mat4::IDENTITY),
            input(3, 2, &p2, &n2, &i2, Mat4::IDENTITY),
        ];
        let moved_world = Mat4::from_translation(Vec3::new(3.0, 2.0, 1.0));
        let moved = [
            input(1, 0, &p0, &n0, &i0, moved_world),
            original[1],
            original[2],
        ];
        let mut geometry = Geometry::default();
        geometry.update(&original, 1 << 20, 0).unwrap();
        let old_part_bounds = geometry.part_bounds[0];
        geometry.mesh_dirty.clear();
        geometry.node_dirty.clear();
        geometry.update(&moved, 1 << 20, 0).unwrap();

        let mut expected_mesh = DirtyBytes::default();
        let count = geometry.count() as u64;
        let vertex_words = if geometry.parts.iter().any(|part| part.colored) {
            VERTEX_WORDS
        } else {
            POSITION_NORMAL_WORDS
        };
        let triangles = (0..2)
            .map(|triangle| {
                GEOMETRY_HEADER_BYTES
                    + geometry.triangle_position[triangle] as u64 * (TRIANGLE_WORDS * 4) as u64
            })
            .collect();
        expected_mesh.add_blocks(triangles, (TRIANGLE_WORDS * 4) as u64);
        expected_mesh.add_blocks(
            (0..6)
                .map(|vertex| {
                    GEOMETRY_HEADER_BYTES
                        + count * (TRIANGLE_WORDS * 4) as u64
                        + vertex as u64 * (vertex_words * 4) as u64
                })
                .collect(),
            (vertex_words * 4) as u64,
        );
        assert!(!geometry.mesh_dirty.full);
        assert_eq!(geometry.mesh_dirty.ranges, expected_mesh.ranges);

        let mut leaves = vec![geometry.triangle_node[0], geometry.triangle_node[1]];
        let mut nodes = Vec::new();
        while let Some(leaf) = leaves.pop() {
            let mut node = leaf as usize;
            loop {
                if nodes.contains(&node) {
                    break;
                }
                nodes.push(node);
                let parent = geometry.parents[node];
                if parent == u32::MAX {
                    break;
                }
                node = parent as usize;
            }
        }
        let mut expected_nodes = DirtyBytes::default();
        expected_nodes.add_blocks(
            nodes
                .into_iter()
                .map(|node| node as u64 * (NODE_WORDS * 4) as u64)
                .collect(),
            (NODE_WORDS * 4) as u64,
        );
        assert!(!geometry.node_dirty.full);
        assert_eq!(geometry.node_dirty.ranges, expected_nodes.ranges);

        assert_eq!(geometry.dirty_bounds[0], old_part_bounds.unwrap());
        assert_eq!(geometry.dirty_bounds[1], geometry.part_bounds[0].unwrap());

        let mut reserialized = geometry.clone();
        let mesh_size = geometry.mesh_bytes.len() as u64;
        reserialized.serialize_mesh_full(vertex_words, mesh_size);
        reserialized.serialize_nodes_full();
        assert_eq!(geometry.mesh_bytes, reserialized.mesh_bytes);
        assert_eq!(geometry.node_bytes, reserialized.node_bytes);
    }
}
