//! Deterministic surface Poisson disk initialization of foliage micro-triangles
//! (M7, contract `m7-compute`).
//!
//! [`initialize_foliage`] turns the retained [`FoliageSource`] of one tree into exactly
//! `count` [`MicroTriangle`] records. The candidates are the area-weighted surface samples
//! of `sample_foliage`; a candidate joins the set only when it is at least the current
//! Poisson radius away from every accepted candidate. When the requested count cannot fit
//! at that radius an explicit schedule halves the radius and tries again, so a nonempty
//! foliage source always reaches the requested count; the radius is never silently
//! replaced by independent random samples. Each micro-triangle reuses the candidate's
//! area-equivalent proxy corners and its source shading normal; `opacity` starts at 1.

use std::collections::HashMap;

use raa_core::RaaError;

use super::foliage_resources::{FoliageSource, MicroTriangle};
use super::foliage_sampling::{FoliageRng, FoliageSample, FoliageSurface, sample_foliage};

/// Candidate batches tried at one radius, each `count` samples long.
const BATCHES_PER_RADIUS: u32 = 8;
/// Factor applied to the radius when the requested count cannot fit.
const RADIUS_REDUCTION: f32 = 0.5;
/// Reductions before the separation constraint is dropped to fill the last samples.
const MAX_REDUCTIONS: u32 = 16;

fn invalid(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message: message.into(),
    }
}

/// The hexagonal-packing radius that fits `count` samples on `area` square metres.
fn spacing_radius(area: f32, count: u32) -> f32 {
    (2.0 * area / (3.0f32.sqrt() * count as f32)).sqrt()
}

/// A spatial hash of the accepted positions, keyed by one-radius cells.
struct Grid {
    cell: f32,
    cells: HashMap<[i32; 3], Vec<usize>>,
}

impl Grid {
    fn new(radius: f32) -> Self {
        Self {
            cell: radius.max(f32::MIN_POSITIVE),
            cells: HashMap::new(),
        }
    }

    fn key(&self, p: [f32; 3]) -> [i32; 3] {
        [
            (p[0] / self.cell).floor() as i32,
            (p[1] / self.cell).floor() as i32,
            (p[2] / self.cell).floor() as i32,
        ]
    }

    fn insert(&mut self, p: [f32; 3], index: usize) {
        self.cells.entry(self.key(p)).or_default().push(index);
    }

    fn refill(&mut self, points: &[[f32; 3]]) {
        self.cells.clear();
        for (index, point) in points.iter().enumerate() {
            self.insert(*point, index);
        }
    }

    fn far_enough(&self, p: [f32; 3], points: &[[f32; 3]], radius2: f32) -> bool {
        let key = self.key(p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let cell = [key[0] + dx, key[1] + dy, key[2] + dz];
                    let Some(indices) = self.cells.get(&cell) else {
                        continue;
                    };
                    for &index in indices {
                        let q = points[index];
                        let d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
                        if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < radius2 {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }
}

fn micro_triangle(sample: &FoliageSample) -> MicroTriangle {
    MicroTriangle {
        p0: sample.corners[0],
        p1: sample.corners[1],
        p2: sample.corners[2],
        normal: sample.normal,
        opacity: 1.0,
    }
}

/// Builds exactly `count` Poisson-separated micro-triangles of `source`.
///
/// A positive `count` is required. The Poisson radius starts at the hexagonal estimate
/// for the eligible area and is halved after [`BATCHES_PER_RADIUS`] candidate batches
/// fail to fill the set, so a nonempty foliage source always returns exactly `count`
/// records; a source without an eligible foliage triangle returns an empty list.
///
/// Handoff: 'Run differentiable foliage optimization' consumes `initialize_foliage`,
/// which returns the seeded, area-weighted Poisson micro-triangle seed.
pub fn initialize_foliage(
    source: &FoliageSource,
    count: u32,
    seed: u32,
) -> Result<Vec<MicroTriangle>, RaaError> {
    if count == 0 {
        return Err(invalid("foliage micro-triangle count must be positive"));
    }
    let target = count as usize;
    let mut rng = FoliageRng::new(seed);
    let mut batch = sample_foliage(source, count, rng.next_u32())?;
    if batch.is_empty() {
        return Ok(Vec::new());
    }
    let area = batch[0].represented_area * count as f32;
    let mut radius = spacing_radius(area, count);
    let mut grid = Grid::new(radius);
    let mut accepted: Vec<[f32; 3]> = Vec::new();
    accepted
        .try_reserve_exact(target)
        .map_err(|_| invalid("cannot reserve the foliage micro-triangle positions"))?;
    let mut triangles: Vec<MicroTriangle> = Vec::new();
    triangles
        .try_reserve_exact(target)
        .map_err(|_| invalid("cannot reserve the foliage micro-triangles"))?;
    let mut surface: Option<FoliageSurface> = None;
    let mut batches = BATCHES_PER_RADIUS;
    let mut reductions = 0u32;
    loop {
        for sample in batch.drain(..) {
            if accepted.len() == target {
                return Ok(triangles);
            }
            let kept =
                radius <= 0.0 || grid.far_enough(sample.position, &accepted, radius * radius);
            if kept {
                let index = accepted.len();
                accepted.push(sample.position);
                triangles.push(micro_triangle(&sample));
                if radius > 0.0 {
                    grid.insert(sample.position, index);
                }
            }
        }
        batches -= 1;
        if batches == 0 {
            reductions += 1;
            radius = if reductions > MAX_REDUCTIONS {
                0.0
            } else {
                radius * RADIUS_REDUCTION
            };
            if radius > 0.0 {
                grid = Grid::new(radius);
                grid.refill(&accepted);
            }
            batches = BATCHES_PER_RADIUS;
        }
        // Every further batch reuses one eligible-area table instead of rebuilding it.
        if surface.is_none() {
            surface = Some(FoliageSurface::new(source)?);
        }
        let table = surface
            .as_ref()
            .ok_or_else(|| invalid("the foliage eligible-area table is missing"))?;
        batch = table.sample(source, count, rng.next_u32())?;
        if batch.is_empty() {
            return Ok(Vec::new());
        }
    }
}
