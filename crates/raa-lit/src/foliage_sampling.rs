//! Area-weighted stratified foliage surface sampling (M7, contract `m7-sampling`).
//!
//! [`sample_foliage`] turns the retained [`FoliageSource`] of one tree into `count`
//! surface samples. Each stratum owns `totalArea / count` square metres, picks one
//! source triangle through the cumulative eligible area and one uniform barycentric
//! point. [`FoliageSample::normal`] is the normalized interpolation of the exact
//! source shading normals; [`FoliageSample::corners`] form an equilateral proxy
//! triangle around the sample whose total area equals the eligible source area.
//!
//! [`FoliageRng`] is the seeded 32-bit stream: no clock and no operating-system
//! randomness. Sampling is deterministic for a given source, count and seed.

use std::mem::size_of;

use raa_core::RaaError;

use super::foliage_resources::{FoliageSource, SourceTriangle};

/// One area-weighted surface sample of a foliage source.
///
/// `source_triangle` indexes [`FoliageSource::triangles`], `barycentric` weights its
/// positions and source normals, `position` and `corners` are source-root metres and
/// `represented_area` is the square metres this stratum represents.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FoliageSample {
    pub source_triangle: u32,
    pub barycentric: [f32; 3],
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub corners: [[f32; 3]; 3],
    pub represented_area: f32,
}

/// Area of the unit-circumradius equilateral proxy template (`3 * sqrt(3) / 4`).
const PROXY_AREA: f32 = 1.299_038;
/// Full turn in radians.
const TAU: f32 = std::f32::consts::TAU;

/// A deterministic 32-bit seed stream.
///
/// The sequence is the low-bias 32-bit integer hash of the incremented state (the
/// constants of `splitmix32`). It is independent of any clock or random source.
#[derive(Clone, Debug)]
pub struct FoliageRng {
    state: u32,
}

impl FoliageRng {
    /// Starts the stream at `seed`.
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    /// Returns the next 32-bit value of the stream.
    pub fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(0x9e37_79b9);
        let mut z = self.state;
        z = (z ^ (z >> 16)).wrapping_mul(0x85eb_ca6b);
        z = (z ^ (z >> 13)).wrapping_mul(0xc2b2_ae35);
        z ^ (z >> 16)
    }

    /// Returns the next value in `[0, 1)` with 24 random bits.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

fn invalid(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message: message.into(),
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn unit(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len = dot(v, v).sqrt();
    if len > f32::MIN_POSITIVE && len.is_finite() {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        fallback
    }
}

fn geometric_normal(t: &SourceTriangle) -> [f32; 3] {
    let n = cross(
        sub(t.positions[1], t.positions[0]),
        sub(t.positions[2], t.positions[0]),
    );
    unit(n, [0.0, 1.0, 0.0])
}

fn area(t: &SourceTriangle) -> Option<f32> {
    if !t.foliage {
        return None;
    }
    let n = cross(
        sub(t.positions[1], t.positions[0]),
        sub(t.positions[2], t.positions[0]),
    );
    let value = 0.5 * dot(n, n).sqrt();
    if value.is_finite() && value > 0.0 {
        Some(value)
    } else {
        None
    }
}

fn interpolate(values: [[f32; 3]; 3], weights: [f32; 3]) -> [f32; 3] {
    [
        weights[0] * values[0][0] + weights[1] * values[1][0] + weights[2] * values[2][0],
        weights[0] * values[0][1] + weights[1] * values[1][1] + weights[2] * values[2][1],
        weights[0] * values[0][2] + weights[1] * values[1][2] + weights[2] * values[2][2],
    ]
}

fn tangent_basis(normal: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let helper = if normal[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let tangent = unit(cross(helper, normal), [1.0, 0.0, 0.0]);
    (tangent, cross(normal, tangent))
}

/// Samples the eligible surface of `source` into `count` strata.
///
/// Only `foliage` triangles with a positive finite area participate. A positive
/// `count` is required; an allocation that cannot be represented or reserved returns
/// `foliage.invalid`, and no sample is silently dropped. A source without an eligible
/// triangle returns an empty list.
///
/// Handoff: 'Assemble stochastic foliage and its volume' reads `sample_foliage`
/// samples (source triangle, position, normal, corners and represented area) as the
/// stochastic foliage surface and its proxy volume.
pub fn sample_foliage(
    source: &FoliageSource,
    count: u32,
    seed: u32,
) -> Result<Vec<FoliageSample>, RaaError> {
    if count == 0 {
        return Err(invalid("foliage sample count must be positive"));
    }
    let mut eligible = Vec::new();
    let mut cumulative = Vec::new();
    let mut total = 0.0f64;
    for (index, triangle) in source.triangles.iter().enumerate() {
        if let Some(value) = area(triangle) {
            total += value as f64;
            eligible.push(index as u32);
            cumulative.push(total);
        }
    }
    if eligible.is_empty() || !total.is_finite() || total <= 0.0 {
        return Ok(Vec::new());
    }
    let count_usize = count as usize;
    let bytes = count_usize
        .checked_mul(size_of::<FoliageSample>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| invalid("foliage sample count overflows the address space"))?;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(count_usize)
        .map_err(|_| invalid(format!("cannot reserve {bytes} bytes of foliage samples")))?;
    let represented_area = (total / count as f64) as f32;
    let scale = (total / (count as f64 * PROXY_AREA as f64)).sqrt() as f32;

    let mut rng = FoliageRng::new(seed);
    for stratum in 0..count {
        let offset = rng.next_f32() as f64;
        let target = (stratum as f64 + offset) / count as f64 * total;
        let slot = cumulative
            .partition_point(|bound| *bound <= target)
            .min(cumulative.len() - 1);
        let index = eligible[slot];
        let triangle = &source.triangles[index as usize];

        let root = rng.next_f32().sqrt();
        let second = rng.next_f32();
        let weights = [1.0 - root, root * (1.0 - second), root * second];
        let position = interpolate(triangle.positions, weights);
        let normal = unit(
            interpolate(triangle.normals, weights),
            geometric_normal(triangle),
        );

        let (tangent, bitangent) = tangent_basis(normal);
        let azimuth = rng.next_f32() * TAU;
        let mut corners = [[0.0f32; 3]; 3];
        for (vertex, corner) in corners.iter_mut().enumerate() {
            let (sin, cos) = (azimuth + vertex as f32 * (TAU / 3.0)).sin_cos();
            for axis in 0..3 {
                let radial = cos * tangent[axis] + sin * bitangent[axis];
                corner[axis] = position[axis] + scale * radial;
            }
        }

        samples.push(FoliageSample {
            source_triangle: index,
            barycentric: weights,
            position,
            normal,
            corners,
            represented_area,
        });
    }
    Ok(samples)
}
