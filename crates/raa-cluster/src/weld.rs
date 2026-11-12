//! Grid weld of the input mesh: positions snap to the 16-bit quantisation grid, coincident
//! vertices merge, collapsed triangles drop, and octahedral normals are packed.
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-builder/src/dag.rs`, lines 419-533, revision 7e465b5.

use std::collections::BTreeMap;

use raa_math::Vec3;

use crate::input::{InputBounds, OwnedClusterInput};
use crate::simplify::{Attribute, Corner, Triangle};

/// Input triangles welded by one `WeldState::step`: one bounded work unit.
pub const WELD_BATCH: usize = 4096;
/// How far the grid of a mesh made of separate pieces extends beyond its bounds on each side, as a
/// fraction of its largest dimension (solstice dag.rs:46).
pub const FOLIAGE_GRID_MARGIN: f32 = 0.25;

/// Largest grid index of one axis.
const GRID_MAX: f32 = 65535.0;
/// Cache value of a vertex that is not welded yet.
const UNSET: u32 = u32::MAX;

/// A welded mesh on one 16-bit grid. `Corner.position` indexes `positions`, `Corner.attribute`
/// indexes `attributes`; positions are grid points `origin + grid * step`.
#[derive(Clone, Debug)]
pub struct WeldedMesh {
    pub positions: Vec<Vec3>,
    pub attributes: Vec<Attribute>,
    pub triangles: Vec<Triangle>,
    pub origin: Vec3,
    pub step: f32,
    pub uv_origin: [f32; 2],
    pub uv_step: [f32; 2],
}

impl WeldedMesh {
    /// The nearest grid point to `p`, each axis clamped to the grid.
    pub fn snap(&self, p: Vec3) -> Vec3 {
        grid_point(self.origin, self.step, self.grid(p))
    }

    /// The grid index of `p` on each axis, clamped to 0..=65535.
    pub fn grid(&self, p: Vec3) -> [u16; 3] {
        grid_index(self.origin, self.step, p)
    }
}

/// Grid index of `p`, each axis rounded and clamped; NaN gives 0.
fn grid_index(origin: Vec3, step: f32, p: Vec3) -> [u16; 3] {
    let q = (p - origin) / step;
    [axis_index(q.x), axis_index(q.y), axis_index(q.z)]
}

fn axis_index(v: f32) -> u16 {
    v.round().clamp(0.0, GRID_MAX) as u16
}

fn grid_point(origin: Vec3, step: f32, g: [u16; 3]) -> Vec3 {
    origin + Vec3::new(f32::from(g[0]), f32::from(g[1]), f32::from(g[2])) * step
}

/// Unit vector of `v`, or +Y for a zero vector.
fn unit_or_up(v: Vec3) -> Vec3 {
    let n = v.normalize_or_zero();
    if n == Vec3::ZERO { Vec3::Y } else { n }
}

/// Encodes a unit vector to 16-bit signed octahedral coordinates (solstice dag.rs:518-533); a zero
/// vector encodes +Y.
pub fn encode_octahedral(normal: Vec3) -> [i16; 2] {
    let n = unit_or_up(normal);
    let sum = n.x.abs() + n.y.abs() + n.z.abs();
    let (px, py) = (n.x / sum, n.y / sum);
    let (fx, fy) = if n.z < 0.0 {
        (
            (1.0 - py.abs()) * px.signum(),
            (1.0 - px.abs()) * py.signum(),
        )
    } else {
        (px, py)
    };
    [octahedral_word(fx), octahedral_word(fy)]
}

fn octahedral_word(v: f32) -> i16 {
    (v.clamp(-1.0, 1.0) * 32767.0).round() as i16
}

/// Decodes the octahedral coordinates of `encode_octahedral` to a unit vector.
pub fn decode_octahedral(v: [i16; 2]) -> Vec3 {
    let (x, y) = (f32::from(v[0]) / 32767.0, f32::from(v[1]) / 32767.0);
    let z = 1.0 - x.abs() - y.abs();
    let (x, y) = if z < 0.0 {
        ((1.0 - y.abs()) * x.signum(), (1.0 - x.abs()) * y.signum())
    } else {
        (x, y)
    };
    unit_or_up(Vec3::new(x, y, z))
}

/// Quantises one uv component to its grid index, clamped to 0..=65535.
fn uv_index(v: f32, origin: f32, step: f32) -> u32 {
    ((v - origin) / step).round().clamp(0.0, GRID_MAX) as u32
}

/// Welds an owned input one batch of triangles at a time. `new` fixes the grid from the bounds,
/// `step` welds the next `WELD_BATCH` triangles and `finish` returns the mesh once `step` has
/// reported completion. Position and attribute ids are first-use order; a vertex's ids are kept
/// in per-vertex caches so it is welded once.
pub struct WeldState {
    origin: Vec3,
    step: f32,
    uv_origin: [f32; 2],
    uv_step: [f32; 2],
    position_ids: BTreeMap<[u16; 3], u32>,
    attribute_ids: BTreeMap<[u32; 4], u32>,
    position_cache: Vec<u32>,
    attribute_cache: Vec<u32>,
    positions: Vec<Vec3>,
    attributes: Vec<Attribute>,
    triangles: Vec<Triangle>,
    next_triangle: usize,
    done: bool,
}

impl WeldState {
    /// The grid of a mesh with `bounds`, padded by `margin` times its largest dimension on each
    /// side.
    pub fn new(bounds: &InputBounds, margin: f32) -> WeldState {
        let size = bounds.max - bounds.min;
        let pad = size.x.max(size.y).max(size.z) * margin;
        let min = bounds.min - Vec3::splat(pad);
        let extent = bounds.max + Vec3::splat(pad) - min;
        let uv_size = [
            bounds.uv_max[0] - bounds.uv_min[0],
            bounds.uv_max[1] - bounds.uv_min[1],
        ];
        WeldState {
            origin: min,
            step: (extent.x.max(extent.y).max(extent.z) / GRID_MAX).max(1.0e-9),
            uv_origin: bounds.uv_min,
            uv_step: uv_size.map(|s| (s / GRID_MAX).max(1.0e-9)),
            position_ids: BTreeMap::new(),
            attribute_ids: BTreeMap::new(),
            position_cache: Vec::new(),
            attribute_cache: Vec::new(),
            positions: Vec::new(),
            attributes: Vec::new(),
            triangles: Vec::new(),
            next_triangle: 0,
            done: false,
        }
    }

    /// Welds the corners of the next `WELD_BATCH` input triangles in index order. A triangle with
    /// two equal position ids is dropped, and one referencing a missing vertex is dropped too.
    /// Returns true once every input triangle was visited; later calls change nothing.
    pub fn step(&mut self, input: &OwnedClusterInput) -> bool {
        if self.done {
            return true;
        }
        let vertices = input.mesh.positions.len();
        if self.position_cache.len() != vertices {
            self.position_cache = vec![UNSET; vertices];
        }
        if self.attribute_cache.len() != vertices {
            self.attribute_cache = vec![UNSET; vertices];
        }
        let indices = &input.mesh.indices;
        let total = indices.len() / 3;
        let start = self.next_triangle.min(total);
        let end = (start + WELD_BATCH).min(total);
        let batch = indices.get(start * 3..end * 3).unwrap_or(&[]);
        for face in batch.chunks_exact(3) {
            if let &[a, b, c] = face
                && let Some(triangle) = self.weld_face(input, [a, b, c])
            {
                self.triangles.push(triangle);
            }
        }
        self.next_triangle = end;
        self.done = end >= total;
        self.done
    }

    /// The welded mesh. Call it after `step` returned true; earlier it holds the batches so far.
    pub fn finish(self) -> WeldedMesh {
        WeldedMesh {
            positions: self.positions,
            attributes: self.attributes,
            triangles: self.triangles,
            origin: self.origin,
            step: self.step,
            uv_origin: self.uv_origin,
            uv_step: self.uv_step,
        }
    }

    fn weld_face(&mut self, input: &OwnedClusterInput, [a, b, c]: [u32; 3]) -> Option<Triangle> {
        let corners = [
            self.corner(input, a)?,
            self.corner(input, b)?,
            self.corner(input, c)?,
        ];
        let [t0, t1, t2] = corners;
        let kept =
            t0.position != t1.position && t1.position != t2.position && t2.position != t0.position;
        kept.then_some(corners)
    }

    fn corner(&mut self, input: &OwnedClusterInput, vertex: u32) -> Option<Corner> {
        let v = usize::try_from(vertex).ok()?;
        Some(Corner {
            position: self.position_id(input, v)?,
            attribute: self.attribute_id(input, v)?,
        })
    }

    fn position_id(&mut self, input: &OwnedClusterInput, v: usize) -> Option<u32> {
        let cached = *self.position_cache.get(v)?;
        if cached != UNSET {
            return Some(cached);
        }
        let key = grid_index(
            self.origin,
            self.step,
            Vec3::from_array(*input.mesh.positions.get(v)?),
        );
        let id = match self.position_ids.get(&key) {
            Some(&id) => id,
            None => {
                let id = u32::try_from(self.positions.len()).ok()?;
                self.positions.push(grid_point(self.origin, self.step, key));
                self.position_ids.insert(key, id);
                id
            }
        };
        *self.position_cache.get_mut(v)? = id;
        Some(id)
    }

    fn attribute_id(&mut self, input: &OwnedClusterInput, v: usize) -> Option<u32> {
        let cached = *self.attribute_cache.get(v)?;
        if cached != UNSET {
            return Some(cached);
        }
        let normal = Vec3::from_array(*input.mesh.normals.get(v)?);
        let uv = input
            .uvs
            .as_ref()
            .and_then(|uvs| uvs.get(v))
            .copied()
            .unwrap_or([0.0, 0.0]);
        let octahedral = encode_octahedral(normal);
        let key = [
            u32::from(octahedral[0] as u16),
            u32::from(octahedral[1] as u16),
            uv_index(uv[0], self.uv_origin[0], self.uv_step[0]),
            uv_index(uv[1], self.uv_origin[1], self.uv_step[1]),
        ];
        let id = match self.attribute_ids.get(&key) {
            Some(&id) => id,
            None => {
                let id = u32::try_from(self.attributes.len()).ok()?;
                self.attributes.push(Attribute {
                    normal: unit_or_up(normal),
                    uv,
                });
                self.attribute_ids.insert(key, id);
                id
            }
        };
        *self.attribute_cache.get_mut(v)? = id;
        Some(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_geom::MeshData;

    fn input(positions: Vec<[f32; 3]>, indices: Vec<u32>) -> OwnedClusterInput {
        let normals = vec![[0.0, 0.0, 1.0]; positions.len()];
        OwnedClusterInput {
            mesh: MeshData {
                positions,
                normals,
                indices,
            },
            uvs: None,
        }
    }

    fn weld_all(input: &OwnedClusterInput, margin: f32) -> WeldedMesh {
        let mut state = WeldState::new(&input.bounds(), margin);
        while !state.step(input) {}
        state.finish()
    }

    #[test]
    fn octahedral_round_trip_keeps_the_direction() {
        let axes = [
            Vec3::X,
            -Vec3::X,
            Vec3::Y,
            -Vec3::Y,
            Vec3::Z,
            -Vec3::Z,
            Vec3::new(1.0, 2.0, 3.0).normalize_or_zero(),
            Vec3::new(-1.0, -2.0, -3.0).normalize_or_zero(),
        ];
        for n in axes {
            let back = decode_octahedral(encode_octahedral(n));
            assert!((back - n).length() <= 1.0e-3, "{n:?} decoded to {back:?}");
            assert!((back.length() - 1.0).abs() <= 1.0e-3);
        }
    }

    #[test]
    fn zero_normal_encodes_plus_y() {
        assert_eq!(encode_octahedral(Vec3::ZERO), encode_octahedral(Vec3::Y));
        assert!((decode_octahedral(encode_octahedral(Vec3::ZERO)) - Vec3::Y).length() <= 1.0e-3);
    }

    #[test]
    fn two_vertices_within_one_step_share_a_position() {
        // Bounds span one unit, so the step is 1/65535 and 1e-6 rounds onto the first vertex.
        let mesh = input(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0e-6, 0.0, 0.0],
            ],
            vec![0, 1, 2, 3, 1, 2],
        );
        let welded = weld_all(&mesh, 0.0);
        assert_eq!(welded.positions.len(), 3);
        assert_eq!(welded.triangles.len(), 2);
        assert_eq!(
            welded.triangles[0][0].position,
            welded.triangles[1][0].position
        );
        assert_eq!(welded.attributes.len(), 1);
    }

    #[test]
    fn triangle_collapsed_by_quantisation_is_dropped() {
        let mesh = input(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0e-6, 0.0, 0.0],
            ],
            vec![0, 1, 2, 0, 3, 2],
        );
        assert_eq!(weld_all(&mesh, 0.0).triangles.len(), 1);
    }

    #[test]
    fn snap_returns_a_grid_point_and_clamps_outside_the_grid() {
        let mesh = WeldedMesh {
            positions: Vec::new(),
            attributes: Vec::new(),
            triangles: Vec::new(),
            origin: Vec3::ZERO,
            step: 0.5,
            uv_origin: [0.0; 2],
            uv_step: [1.0; 2],
        };
        assert_eq!(
            mesh.snap(Vec3::new(0.26, 0.74, 0.0)),
            Vec3::new(0.5, 0.5, 0.0)
        );
        assert_eq!(mesh.snap(Vec3::new(-5.0, 0.0, 0.0)), Vec3::ZERO);
        assert_eq!(mesh.snap(Vec3::new(1.0e9, 0.0, 0.0)).x, 65535.0 * 0.5);
        assert_eq!(mesh.grid(Vec3::new(1.0e9, -3.0, 0.2)), [65535, 0, 0]);
        let on_grid = mesh.snap(Vec3::new(1.3, 2.2, -0.4));
        assert_eq!(mesh.snap(on_grid), on_grid);
    }

    #[test]
    fn batched_steps_match_one_loop_over_a_generated_grid() {
        // 100 x 50 quads: 10000 triangles over 5151 vertices.
        let (nx, ny) = (100usize, 50usize);
        let mut positions = Vec::new();
        for j in 0..=ny {
            for i in 0..=nx {
                positions.push([i as f32 * 0.1, j as f32 * 0.1, 0.0]);
            }
        }
        let mut indices = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                let a = (j * (nx + 1) + i) as u32;
                let (b, c) = (a + 1, a + (nx + 1) as u32);
                indices.extend_from_slice(&[a, b, c + 1, a, c + 1, c]);
            }
        }
        let mesh = input(positions, indices);
        let mut state = WeldState::new(&mesh.bounds(), 0.0);
        let mut calls = 1;
        while !state.step(&mesh) {
            calls += 1;
        }
        assert_eq!(calls, 10000usize.div_ceil(WELD_BATCH));
        let batched = state.finish();
        let looped = weld_all(&mesh, 0.0);
        assert_eq!(batched.triangles.len(), 10000);
        assert_eq!(batched.positions.len(), 5151);
        assert_eq!(batched.attributes.len(), 1);
        assert_eq!(batched.triangles, looped.triangles);
        assert_eq!(batched.positions, looped.positions);
        assert_eq!(batched.attributes, looped.attributes);
    }

    #[test]
    fn margin_moves_the_origin_by_a_quarter_of_the_largest_dimension() {
        let bounds = InputBounds {
            min: Vec3::ZERO,
            max: Vec3::new(2.0, 1.0, 4.0),
            uv_min: [0.0; 2],
            uv_max: [0.0; 2],
        };
        let state = WeldState::new(&bounds, FOLIAGE_GRID_MARGIN);
        assert_eq!(state.origin, Vec3::splat(-1.0));
    }

    #[test]
    fn attribute_stores_the_unit_normal_and_the_vertex_uv() {
        let mut mesh = input(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![0, 1, 2],
        );
        mesh.mesh.normals = vec![[0.0, 0.0, 5.0], [0.0; 3], [0.0, 0.0, 5.0]];
        mesh.uvs = Some(vec![[0.25, 0.5], [0.0, 0.0], [0.25, 0.5]]);
        let welded = weld_all(&mesh, 0.0);
        assert_eq!(welded.attributes.len(), 2);
        assert_eq!(welded.attributes[0].normal, Vec3::Z);
        assert_eq!(welded.attributes[0].uv, [0.25, 0.5]);
        assert_eq!(welded.attributes[1].normal, Vec3::Y);
    }
}
