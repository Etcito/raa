//! Triangle input for the shared trace BVH. Refit the world bounds for rigid motion.

use raa_core::RaaError;
use raa_math::{Mat4, Vec3};

use crate::bvh::{Bvh, Node, build_bounds};

const TRIANGLE_WORDS: usize = 24;
const MAX_TRIANGLES: usize = 250_000;
type Bounds = ([f32; 3], [f32; 3]);

/// One submesh range with the renderer's current material and world transform.
pub struct MeshInput<'a> {
    pub key: u64,
    pub mesh: u32,
    pub positions: &'a [[f32; 3]],
    pub normals: &'a [[f32; 3]],
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
    bounds: Vec<Bounds>,
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

        self.triangles.resize(count, [0; TRIANGLE_WORDS]);
        self.bounds.resize(count, ([0.; 3], [0.; 3]));
        let mut start = 0;
        for (i, input) in inputs.iter().enumerate() {
            let part = Part::from(input);
            if rebuild || self.parts[i] != part {
                self.write_part(input, start);
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

    fn write_part(&mut self, input: &MeshInput<'_>, start: usize) {
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
                let offset = corner * 8;
                words[offset..offset + 3].copy_from_slice(&p.map(f32::to_bits));
                words[offset + 4..offset + 7].copy_from_slice(&normal.map(f32::to_bits));
                for axis in 0..3 {
                    lo[axis] = lo[axis].min(p[axis]);
                    hi[axis] = hi[axis].max(p[axis]);
                }
            }
            words[3] = input.material;
            self.triangles[start + i] = words;
            self.bounds[start + i] = (lo, hi);
        }
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
