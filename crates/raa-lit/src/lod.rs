//! CPU geometric error measurement and stateless detail selection.
use std::collections::{HashMap, HashSet};

use raa_core::{MeshId, RaaError};
use raa_geom::MeshData;

/// Maximum chain length, including the full-resolution mesh.
pub const MAX_LOD_LEVELS: usize = 8;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LodLevel {
    pub mesh: MeshId,
    /// Geometric error in object units.
    pub error: f32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LodParams {
    pub threshold_px: f32,
    pub hysteresis: f32,
    pub bias: f32,
}

impl Default for LodParams {
    fn default() -> Self {
        Self {
            threshold_px: 1.0,
            hysteresis: 0.2,
            bias: 0.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LodRegistry {
    chains: HashMap<MeshId, Vec<LodLevel>>,
}

fn invalid(message: &str) -> RaaError {
    RaaError::FrameGraph {
        code: "render.invalid_lod",
        message: message.into(),
    }
}

impl LodRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Atomically replaces a chain; `levels` excludes the full mesh.
    pub fn set(&mut self, full: MeshId, levels: &[LodLevel]) -> Result<(), RaaError> {
        if levels.is_empty() || levels.len() >= MAX_LOD_LEVELS || self.is_level(full) {
            return Err(invalid("Invalid detail chain length or full mesh"));
        }
        let mut seen = HashSet::new();
        let mut error = 0.0;
        for level in levels {
            if !level.error.is_finite()
                || level.error <= error
                || level.mesh == full
                || !seen.insert(level.mesh)
                || self.chains.contains_key(&level.mesh)
                || self.chains.iter().any(|(owner, chain)| {
                    *owner != full && chain[1..].iter().any(|l| l.mesh == level.mesh)
                })
            {
                return Err(invalid(
                    "Detail meshes must be unique with increasing positive errors",
                ));
            }
            error = level.error;
        }
        let mut chain = Vec::with_capacity(levels.len() + 1);
        chain.push(LodLevel {
            mesh: full,
            error: 0.0,
        });
        chain.extend_from_slice(levels);
        self.chains.insert(full, chain);
        Ok(())
    }

    pub fn clear(&mut self, full: MeshId) {
        self.chains.remove(&full);
    }
    pub fn chain(&self, full: MeshId) -> Option<&[LodLevel]> {
        self.chains.get(&full).map(Vec::as_slice)
    }
    pub fn is_level(&self, mesh: MeshId) -> bool {
        self.chains
            .values()
            .any(|c| c[1..].iter().any(|l| l.mesh == mesh))
    }
}

pub fn camera_level(
    chain: &[LodLevel],
    world_scale: f32,
    distance: f32,
    focal_px: f32,
    prev: Option<usize>,
    p: &LodParams,
) -> usize {
    if distance <= 0.0
        || ![world_scale, distance, focal_px]
            .iter()
            .all(|x| x.is_finite())
    {
        return 0;
    }
    select(chain, world_scale * focal_px / distance, prev, p)
}

pub fn shadow_level(
    chain: &[LodLevel],
    world_scale: f32,
    texel_world: f32,
    prev: Option<usize>,
    p: &LodParams,
) -> usize {
    if texel_world <= 0.0 || ![world_scale, texel_world].iter().all(|x| x.is_finite()) {
        return 0;
    }
    select(chain, world_scale / texel_world, prev, p)
}

fn select(chain: &[LodLevel], k: f32, prev: Option<usize>, p: &LodParams) -> usize {
    if chain.len() <= 1
        || ![k, p.threshold_px, p.hysteresis, p.bias]
            .iter()
            .all(|x| x.is_finite())
        || chain.iter().any(|l| !l.error.is_finite())
    {
        return 0;
    }
    let t = p.threshold_px * p.bias.exp2();
    let h = p.hysteresis.clamp(0.0, 0.9);
    let best = |threshold| {
        chain
            .iter()
            .rposition(|l| l.error * k <= threshold)
            .unwrap_or(0)
    };
    let c = best(t);
    let Some(prev) = prev else {
        return c;
    };
    let q = prev.min(chain.len() - 1);
    if c > q {
        best(t * (1.0 - h)).max(q)
    } else if c < q && chain[q].error * k > t * (1.0 + h) {
        c
    } else {
        q
    }
}

type Point = [f64; 3];
fn sub(a: Point, b: Point) -> Point {
    std::array::from_fn(|i| a[i] - b[i])
}
fn dot(a: Point, b: Point) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn along(a: Point, b: Point, t: f64) -> Point {
    std::array::from_fn(|i| a[i] + b[i] * t)
}

// Closest point using the vertex, edge and face Voronoi regions.
fn triangle_distance(p: Point, [a, b, c]: [Point; 3]) -> f64 {
    #[cfg(test)]
    TRIANGLE_VISITS.with(|v| v.set(v.get() + 1));
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    let closest = if d1 <= 0.0 && d2 <= 0.0 {
        a
    } else {
        let bp = sub(p, b);
        let d3 = dot(ab, bp);
        let d4 = dot(ac, bp);
        if d3 >= 0.0 && d4 <= d3 {
            b
        } else {
            let vc = d1 * d4 - d3 * d2;
            if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
                along(a, ab, d1 / (d1 - d3))
            } else {
                let cp = sub(p, c);
                let d5 = dot(ab, cp);
                let d6 = dot(ac, cp);
                if d6 >= 0.0 && d5 <= d6 {
                    c
                } else {
                    let vb = d5 * d2 - d1 * d6;
                    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
                        along(a, ac, d2 / (d2 - d6))
                    } else {
                        let va = d3 * d6 - d5 * d4;
                        if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
                            along(b, sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6)))
                        } else {
                            let inv = 1.0 / (va + vb + vc);
                            along(along(a, ab, vb * inv), ac, vc * inv)
                        }
                    }
                }
            }
        }
    };
    let delta = sub(p, closest);
    dot(delta, delta)
}

/// Bounding volume hierarchy over the nondegenerate triangles. The nearest-triangle search
/// visits a few boxes per vertex whatever the triangle density; the former grid of
/// diagonal/32 cells held hundreds of grass blades per cell and scanned all of them, and
/// the 26 cells around them, for every vertex.
struct Grid {
    triangles: Vec<[Point; 3]>,
    nodes: Vec<Node>,
}

#[derive(Copy, Clone)]
struct Node {
    lo: Point,
    hi: Point,
    /// Leaf (`count > 0`): first triangle. Inner node: the right child; the left one follows it.
    start: u32,
    count: u32,
}

const LEAF_TRIANGLES: usize = 4;

fn bounds<'a>(points: impl Iterator<Item = &'a Point>) -> (Point, Point) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in points {
        for i in 0..3 {
            lo[i] = lo[i].min(p[i]);
            hi[i] = hi[i].max(p[i]);
        }
    }
    (lo, hi)
}

fn box_distance(p: Point, node: &Node) -> f64 {
    (0..3)
        .map(|i| {
            let d = (node.lo[i] - p[i]).max(p[i] - node.hi[i]).max(0.0);
            d * d
        })
        .sum()
}

// Median split on the longest axis of the box: depth log2(n), every leaf at most LEAF_TRIANGLES.
fn build(triangles: &[[Point; 3]], order: &mut [usize], offset: usize, nodes: &mut Vec<Node>) {
    let (lo, hi) = bounds(order.iter().flat_map(|&t| &triangles[t]));
    let index = nodes.len();
    nodes.push(Node {
        lo,
        hi,
        start: offset as u32,
        count: order.len() as u32,
    });
    if order.len() <= LEAF_TRIANGLES {
        return;
    }
    let axis = (0..3)
        .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
        .unwrap_or(0);
    let key = |t: usize| triangles[t].iter().map(|p| p[axis]).sum::<f64>();
    let mid = order.len() / 2;
    order.select_nth_unstable_by(mid, |&a, &b| key(a).total_cmp(&key(b)));
    let (left, right) = order.split_at_mut(mid);
    build(triangles, left, offset, nodes);
    nodes[index].start = nodes.len() as u32;
    nodes[index].count = 0;
    build(triangles, right, offset + mid, nodes);
}

impl Grid {
    fn new(mesh: &MeshData) -> Result<Self, RaaError> {
        if mesh.indices.len() < 3
            || !mesh.indices.len().is_multiple_of(3)
            || mesh.positions.iter().flatten().any(|v| !v.is_finite())
            || mesh
                .indices
                .iter()
                .any(|i| *i as usize >= mesh.positions.len())
        {
            return Err(invalid(
                "Detail measurement requires finite indexed triangles",
            ));
        }
        let points: Vec<Point> = mesh.positions.iter().map(|p| p.map(f64::from)).collect();
        let (origin, max) = bounds(points.iter());
        let diagonal = sub(max, origin);
        if dot(diagonal, diagonal) == 0.0 {
            return Err(invalid("Detail mesh is degenerate"));
        }
        let mut triangles = Vec::with_capacity(mesh.indices.len() / 3);
        for indices in mesh.indices.as_chunks::<3>().0 {
            let triangle: [Point; 3] = std::array::from_fn(|i| points[indices[i] as usize]);
            let ab = sub(triangle[1], triangle[0]);
            let ac = sub(triangle[2], triangle[0]);
            let cross = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            // Zero-area triangles carry no surface; a wholly degenerate mesh is refused.
            if dot(cross, cross) != 0.0 {
                triangles.push(triangle);
            }
        }
        if triangles.is_empty() {
            return Err(invalid("Detail mesh has no nondegenerate triangle"));
        }
        let mut order: Vec<usize> = (0..triangles.len()).collect();
        let mut nodes = Vec::with_capacity(2 * triangles.len() / LEAF_TRIANGLES + 1);
        build(&triangles, &mut order, 0, &mut nodes);
        let triangles = order.iter().map(|&t| triangles[t]).collect();
        Ok(Self { triangles, nodes })
    }

    /// Squared distance from `p` to the surface, exact when it is above `cutoff`. Once a
    /// triangle within `cutoff` is found the search stops and returns that (smaller) distance:
    /// `lod_error` keeps a maximum, which such a vertex cannot raise. `hint` is the triangle
    /// nearest to the previous query; neighbouring vertices usually share it.
    fn distance(
        &self,
        p: Point,
        cutoff: f64,
        hint: &mut usize,
        stack: &mut Vec<(usize, f64)>,
    ) -> f64 {
        let mut best = triangle_distance(p, self.triangles[*hint]);
        if best <= cutoff {
            return best;
        }
        stack.clear();
        stack.push((0usize, box_distance(p, &self.nodes[0])));
        while let Some((index, bound)) = stack.pop() {
            // Every triangle under a box is at least as far as the box.
            if bound > best {
                continue;
            }
            let node = &self.nodes[index];
            if node.count > 0 {
                let start = node.start as usize;
                for (i, triangle) in self.triangles[start..start + node.count as usize]
                    .iter()
                    .enumerate()
                {
                    let d = triangle_distance(p, *triangle);
                    if d < best {
                        best = d;
                        *hint = start + i;
                    }
                }
                if best <= cutoff {
                    return best;
                }
                continue;
            }
            let left = (index + 1, box_distance(p, &self.nodes[index + 1]));
            let right = (
                node.start as usize,
                box_distance(p, &self.nodes[node.start as usize]),
            );
            let (near, far) = if left.1 <= right.1 {
                (left, right)
            } else {
                (right, left)
            };
            stack.push(far);
            stack.push(near);
        }
        best
    }
}

/// Symmetric maximum vertex-to-surface distance in object units.
/// Empty, invalid or wholly degenerate meshes return `render.invalid_lod`.
/// Only positions and indices are read; normals are irrelevant.
pub fn lod_error(full: &MeshData, level: &MeshData) -> Result<f32, RaaError> {
    let full_grid = Grid::new(full)?;
    let level_grid = Grid::new(level)?;
    let mut error: f64 = 0.0;
    let mut stack = Vec::with_capacity(64);
    for (from, to) in [(full, &level_grid), (level, &full_grid)] {
        let mut hint = 0;
        for p in &from.positions {
            error = error.max(to.distance(p.map(f64::from), error, &mut hint, &mut stack));
        }
    }
    Ok(error.sqrt() as f32)
}

#[cfg(test)]
thread_local! {
    static TRIANGLE_VISITS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod lod_grid_tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    /// A 100 x 100 meadow: clumps of 16 leaning blades, one triangle each; every 64th blade
    /// also carries a zero-area triangle. `keep` keeps one blade in `keep`, its tip lowered.
    fn meadow(blades: usize, keep: usize) -> MeshData {
        let mut rng = Lcg(7);
        let mut mesh = MeshData::new();
        let mut centre = [0.0; 2];
        for blade in 0..blades {
            if blade % 16 == 0 {
                centre = [rng.next() * 100.0, rng.next() * 100.0];
            }
            let x = centre[0] + rng.next() * 2.0 - 1.0;
            let z = centre[1] + rng.next() * 2.0 - 1.0;
            let (height, lean) = (0.5 + rng.next(), rng.next() * 0.6 - 0.3);
            if blade % keep != 0 {
                continue;
            }
            let height = if keep > 1 { height * 0.8 } else { height };
            let base = mesh.positions.len() as u32;
            mesh.positions.extend([
                [x - 0.03, 0.0, z],
                [x + 0.03, 0.0, z],
                [x + lean, height, z + lean],
            ]);
            mesh.indices.extend([base, base + 1, base + 2]);
            if blade % 64 == 0 {
                mesh.indices.extend([base, base, base + 1]);
            }
        }
        mesh
    }

    // The metric itself: symmetric maximum vertex-to-surface distance over every triangle.
    fn brute_force(full: &MeshData, level: &MeshData) -> f32 {
        let side = |from: &MeshData, to: &MeshData| {
            let grid = Grid::new(to).unwrap();
            from.positions
                .iter()
                .map(|p| {
                    let p = p.map(f64::from);
                    grid.triangles
                        .iter()
                        .map(|t| triangle_distance(p, *t))
                        .fold(f64::INFINITY, f64::min)
                })
                .fold(0.0, f64::max)
        };
        side(full, level).max(side(level, full)).sqrt() as f32
    }

    #[test]
    fn lod_grid_errors_match_the_exhaustive_metric() {
        for (blades, keep) in [(5, 2), (64, 4), (700, 3), (1024, 8)] {
            let (full, level) = (meadow(blades, 1), meadow(blades, keep));
            let error = lod_error(&full, &level).unwrap();
            assert!(error > 0.0);
            assert_eq!(
                error,
                brute_force(&full, &level),
                "{blades} blades, 1 in {keep}"
            );
        }
    }

    #[test]
    fn lod_grid_work_per_vertex_stays_bounded_on_a_dense_meadow() {
        let per_vertex = |blades: usize| {
            let (full, level) = (meadow(blades, 1), meadow(blades, 4));
            TRIANGLE_VISITS.with(|v| v.set(0));
            lod_error(&full, &level).unwrap();
            let visits = TRIANGLE_VISITS.with(|v| v.get());
            visits as f64 / (full.positions.len() + level.positions.len()) as f64
        };
        let (sparse, dense) = (per_vertex(2048), per_vertex(32768));
        assert!(
            dense <= 3.0 && dense <= sparse * 2.0,
            "triangles tested per vertex: {sparse:.1} at 2048 blades, {dense:.1} at 32768"
        );
    }
}
