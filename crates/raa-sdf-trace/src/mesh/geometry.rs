//! Triangle input for the shared trace BVH. Refit the world bounds for rigid motion.

use raa_core::RaaError;
use raa_math::{Mat4, Vec3};

use crate::bvh::{Bvh, Node, build_bounds};

const VERTEX_WORDS: usize = 12;
pub(super) const TRIANGLE_BYTES: u64 = (TRIANGLE_WORDS * 4) as u64;
const TRIANGLE_WORDS: usize = 3 * VERTEX_WORDS;
const MAX_TRIANGLES: usize = 250_000;
type TriangleBounds = ([f32; 3], [f32; 3]);

#[derive(Clone, Copy)]
pub(super) struct Bounds {
    min: [f32; 3],
    max: [f32; 3],
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

/// One submesh range with the renderer's current material and world transform.
pub struct MeshInput<'a> {
    pub key: u64,
    pub mesh: u32,
    pub positions: &'a [[f32; 3]],
    pub normals: &'a [[f32; 3]],
    pub colors: Option<&'a [[f32; 4]]>,
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
    world: Mat4,
}

impl From<&MeshInput<'_>> for Part {
    fn from(value: &MeshInput<'_>) -> Self {
        Self {
            id: value.mesh,
            key: value.key,
            material: value.material,
            count: value.indices.len() / 3,
            world: value.world,
        }
    }
}

fn same_geometry(a: &Part, b: &Part) -> bool {
    (a.id, a.key, a.count) == (b.id, b.key, b.count)
}

#[derive(Default)]
pub(super) struct Geometry {
    parts: Vec<Part>,
    bounds: Vec<TriangleBounds>,
    part_bounds: Vec<Option<Bounds>>,
    pub(super) dirty_bounds: Vec<Bounds>,
    triangles: Vec<[u32; TRIANGLE_WORDS]>,
    bvh: Bvh,
    pub revision: u64,
    pub triangle_bytes: Vec<u8>,
    pub node_bytes: Vec<u8>,
}

impl Geometry {
    pub fn count(&self) -> u32 {
        self.triangles.len() as u32
    }

    pub fn update(&mut self, inputs: &[MeshInput<'_>]) -> Result<(), RaaError> {
        let count: usize = inputs.iter().map(|input| input.indices.len() / 3).sum();
        if count > MAX_TRIANGLES {
            return Err(RaaError::Resource(format!(
                "mesh tracing supports {MAX_TRIANGLES} triangles per view, received {count}"
            )));
        }
        let rebuild = self.parts.len() != inputs.len()
            || inputs
                .iter()
                .zip(&self.parts)
                .any(|(input, part)| !same_geometry(&input.into(), part));
        if !rebuild
            && inputs
                .iter()
                .zip(&self.parts)
                .all(|(input, part)| &Part::from(input) == part)
        {
            return Ok(());
        }

        // Find an unchanged suffix when insertions or removals shift part indices.
        // This keeps common scene edits local even when the existing BVH must rebuild.
        let prefix = self
            .parts
            .iter()
            .zip(inputs)
            .take_while(|(old, input)| **old == Part::from(*input))
            .count();
        let suffix = self.parts[prefix..]
            .iter()
            .rev()
            .zip(inputs[prefix..].iter().rev())
            .take_while(|(old, input)| **old == Part::from(*input))
            .count();
        let aligned = self.parts.len() == inputs.len();
        let old_end = self.parts.len() - suffix;
        let new_end = inputs.len() - suffix;

        self.dirty_bounds.clear();
        for i in prefix..old_end {
            if !aligned || self.parts[i] != Part::from(&inputs[i]) {
                record_dirty(&mut self.dirty_bounds, self.part_bounds[i]);
            }
        }
        self.triangles.resize(count, [0; TRIANGLE_WORDS]);
        self.bounds.resize(count, ([0.; 3], [0.; 3]));
        self.part_bounds.resize(inputs.len(), None);
        let mut start = 0;
        for (i, input) in inputs.iter().enumerate() {
            let part = Part::from(input);
            if rebuild || self.parts[i] != part {
                self.part_bounds[i] = self.write_part(input, start);
                if (prefix..new_end).contains(&i) && (!aligned || self.parts[i] != part) {
                    record_dirty(&mut self.dirty_bounds, self.part_bounds[i]);
                }
            }
            start += part.count;
        }
        if rebuild {
            self.bvh = if count == 0 {
                Bvh::default()
            } else {
                build_bounds(&self.bounds)
            };
            self.parts.clear();
            self.parts.extend(inputs.iter().map(Part::from));
        } else {
            for (part, input) in self.parts.iter_mut().zip(inputs) {
                *part = Part::from(input);
            }
            for i in (0..self.bvh.nodes.len()).rev() {
                self.refit(i);
            }
        }

        self.triangle_bytes.clear();
        self.node_bytes.clear();
        for &index in &self.bvh.order {
            self.triangle_bytes.extend(
                self.triangles[index as usize]
                    .iter()
                    .flat_map(|word| word.to_le_bytes()),
            );
        }
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
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    fn write_part(&mut self, input: &MeshInput<'_>, start: usize) -> Option<Bounds> {
        let mut part: Option<Bounds> = None;
        let normal_matrix = input.world.inverse().map(|matrix| matrix.transpose());
        for (i, indices) in input.indices.chunks_exact(3).enumerate() {
            let indices = [indices[0], indices[1], indices[2]].map(|index| index as usize);
            let positions = indices.map(|j| {
                input
                    .world
                    .transform_point(Vec3::from_array(input.positions[j]))
            });
            let geometric = (positions[1] - positions[0])
                .cross(positions[2] - positions[0])
                .normalize_or_zero();
            let mut words = [0u32; TRIANGLE_WORDS];
            let mut lo = [f32::INFINITY; 3];
            let mut hi = [f32::NEG_INFINITY; 3];
            for (corner, index) in indices.into_iter().enumerate() {
                let p = positions[corner].to_array();
                let normal = normal_matrix
                    .map_or(geometric, |matrix| {
                        matrix
                            .transform_vector(Vec3::from_array(input.normals[index]))
                            .normalize_or_zero()
                    })
                    .to_array();
                let color = input.colors.map_or([1.0; 4], |colors| colors[index]);
                let offset = corner * VERTEX_WORDS;
                words[offset..offset + 3].copy_from_slice(&p.map(f32::to_bits));
                words[offset + 4..offset + 7].copy_from_slice(&normal.map(f32::to_bits));
                words[offset + 8..offset + 11]
                    .copy_from_slice(&[color[0], color[1], color[2]].map(f32::to_bits));
                for axis in 0..3 {
                    lo[axis] = lo[axis].min(p[axis]);
                    hi[axis] = hi[axis].max(p[axis]);
                }
            }
            words[3] = input.material;
            self.triangles[start + i] = words;
            self.bounds[start + i] = (lo, hi);
            let bounds = Bounds { min: lo, max: hi };
            match &mut part {
                Some(part) => part.include(bounds),
                slot => *slot = Some(bounds),
            }
        }
        part
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
            let words = &self.triangles[index];
            let points = std::array::from_fn(|i| {
                std::array::from_fn(|axis| f32::from_bits(words[i * VERTEX_WORDS + axis]))
            });
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

#[cfg(test)]
mod tests {
    use super::*;

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
            indices: &[2, 0, 1],
            material: 7,
            world: Mat4::IDENTITY,
        };
        let mut geometry = Geometry::default();
        geometry.update(std::slice::from_ref(&input)).unwrap();
        let lanes = |geometry: &Geometry, corner: usize| {
            let start = corner * VERTEX_WORDS + 8;
            std::array::from_fn::<_, 3, _>(|i| f32::from_bits(geometry.triangles[0][start + i]))
        };
        for (corner, index) in input.indices.iter().enumerate() {
            let [r, g, b, _] = colors[*index as usize];
            assert_eq!(lanes(&geometry, corner), [r, g, b]);
        }
        assert_eq!(geometry.triangle_bytes.len() as u64, TRIANGLE_BYTES);

        input.colors = None;
        input.key += 1;
        geometry.update(&[input]).unwrap();
        for corner in 0..3 {
            assert_eq!(lanes(&geometry, corner), [1.0; 3]);
        }
    }
}
