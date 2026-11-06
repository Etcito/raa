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

struct Grid {
    triangles: Vec<[Point; 3]>,
    buckets: HashMap<[i32; 3], Vec<usize>>,
    origin: Point,
    cell: f64,
    end: [i32; 3],
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
        let mut origin = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for p in &mesh.positions {
            for i in 0..3 {
                origin[i] = origin[i].min(p[i] as f64);
                max[i] = max[i].max(p[i] as f64);
            }
        }
        let diagonal = sub(max, origin);
        let cell = dot(diagonal, diagonal).sqrt() / 32.0;
        if cell == 0.0 {
            return Err(invalid("Detail mesh is degenerate"));
        }
        let mut grid = Self {
            triangles: Vec::new(),
            buckets: HashMap::new(),
            origin,
            cell,
            end: [0; 3],
        };
        grid.end = grid.coords(max);
        for indices in mesh.indices.chunks_exact(3) {
            let triangle: [Point; 3] =
                std::array::from_fn(|i| mesh.positions[indices[i] as usize].map(f64::from));
            let ab = sub(triangle[1], triangle[0]);
            let ac = sub(triangle[2], triangle[0]);
            let cross = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            // Zero-area triangles carry no surface; a wholly degenerate mesh is refused.
            if dot(cross, cross) == 0.0 {
                continue;
            }
            let lo = grid.coords(std::array::from_fn(|i| {
                triangle.iter().map(|p| p[i]).fold(f64::INFINITY, f64::min)
            }));
            let hi = grid.coords(std::array::from_fn(|i| {
                triangle
                    .iter()
                    .map(|p| p[i])
                    .fold(f64::NEG_INFINITY, f64::max)
            }));
            let index = grid.triangles.len();
            grid.triangles.push(triangle);
            for z in lo[2]..=hi[2] {
                for y in lo[1]..=hi[1] {
                    for x in lo[0]..=hi[0] {
                        grid.buckets.entry([x, y, z]).or_default().push(index);
                    }
                }
            }
        }
        if grid.triangles.is_empty() {
            return Err(invalid("Detail mesh has no nondegenerate triangle"));
        }
        Ok(grid)
    }

    fn coords(&self, p: Point) -> [i32; 3] {
        std::array::from_fn(|i| ((p[i] - self.origin[i]) / self.cell).floor() as i32)
    }

    fn distance(&self, p: Point) -> f64 {
        // Clamp the starting cell so even arbitrarily distant queries need at most 33 rings.
        let center = std::array::from_fn::<_, 3, _>(|i| self.coords(p)[i].clamp(0, self.end[i]));
        let mut best = f64::INFINITY;
        for ring in 0..=32 {
            let lo = std::array::from_fn::<_, 3, _>(|i| (center[i] - ring).max(0));
            let hi = std::array::from_fn::<_, 3, _>(|i| (center[i] + ring).min(self.end[i]));
            for z in lo[2]..=hi[2] {
                for y in lo[1]..=hi[1] {
                    for x in lo[0]..=hi[0] {
                        if (x - center[0])
                            .abs()
                            .max((y - center[1]).abs())
                            .max((z - center[2]).abs())
                            != ring
                        {
                            continue;
                        }
                        if let Some(bucket) = self.buckets.get(&[x, y, z]) {
                            for &index in bucket {
                                best = best.min(triangle_distance(p, self.triangles[index]));
                            }
                        }
                    }
                }
            }
            // Lower bound for every unsearched ring: distance to the complement
            // of the searched cell box, excluding directions beyond the mesh grid.
            let mut remaining = f64::INFINITY;
            for i in 0..3 {
                if lo[i] > 0 {
                    remaining = remaining
                        .min((p[i] - (self.origin[i] + lo[i] as f64 * self.cell)).max(0.0));
                }
                if hi[i] < self.end[i] {
                    remaining = remaining
                        .min((self.origin[i] + (hi[i] + 1) as f64 * self.cell - p[i]).max(0.0));
                }
            }
            if remaining * remaining > best {
                break;
            }
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
    for p in &full.positions {
        error = error.max(level_grid.distance(p.map(f64::from)));
    }
    for p in &level.positions {
        error = error.max(full_grid.distance(p.map(f64::from)));
    }
    Ok(error.sqrt() as f32)
}
