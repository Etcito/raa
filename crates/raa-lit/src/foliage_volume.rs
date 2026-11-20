//! Native foliage volume accumulation (M7, contract `m7-volume`).
//!
//! [`FoliageVolumeAccumulator`] bins area-weighted unlit linear albedo and the
//! nondirectional canopy optical depth into the source-root bounding box at 16 or 32
//! voxels per axis. [`FoliageVolumeAccumulator::add`] clamps each position to the box
//! (the maximum boundary included), accumulates in deterministic call order and never
//! bakes a vertex colour or a light into the albedo. [`FoliageVolumeAccumulator::finish`]
//! is the one finalization path of both native methods: RGB is the area-weighted mean
//! albedo, A is `1 - exp(-area / meanVoxelFaceArea)`, and the bytes are RGBA8 in x-fastest
//! order `4 * (x + resolution * (y + resolution * z))`.
//!
//! The payload is a native D3 volume, never a logical slice stack of a 2D atlas. Both
//! the stochastic assembler and the differentiable optimizer hand the `finish` bytes to
//! the shared `FoliageResources::initialize` upload.

use raa_core::RaaError;

fn invalid(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message: message.into(),
    }
}

fn finite3(v: [f32; 3]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn quantize(value: f64) -> u8 {
    (value * 255.0).round().clamp(0.0, 255.0) as u8
}

#[derive(Clone, Copy, Debug, Default)]
struct Cell {
    area: f64,
    weighted: [f64; 3],
}

/// The source-root voxel accumulator of one native foliage generation.
///
/// `new` refuses `foliage.invalid` for non-finite or degenerate bounds and a resolution
/// other than 16 or 32. `add` refuses non-finite positions or albedos, an albedo outside
/// `0..=1` and a negative or non-finite area. `finish` returns exactly `4 * resolution^3`
/// bytes and leaves an untouched cell at zero.
///
/// Handoff: 'Assemble stochastic foliage and its volume' calls `FoliageVolumeAccumulator`
/// (`new`, `add`, `finish`) during its sample traversal; today this type accumulates the
/// area-weighted unlit albedo and the canopy optical depth and `finish` returns the
/// x-fastest RGBA8 payload that the native volume upload consumes. The earlier handoff of
/// the same downstream card for the sampled surface is
/// `crates/raa-lit/src/foliage_sampling.rs:150` `sample_foliage`.
///
/// Handoff: 'Run differentiable foliage optimization' calls `FoliageVolumeAccumulator`
/// over its micro-triangle traversal and uploads the same `finish` payload through that
/// shared native texture path.
#[derive(Debug)]
pub struct FoliageVolumeAccumulator {
    min: [f32; 3],
    extent: [f64; 3],
    resolution: u32,
    mean_face_area: f64,
    cells: Vec<Cell>,
}

impl FoliageVolumeAccumulator {
    /// Starts an accumulator over the source-root box `min..max`.
    ///
    /// The box must be finite and strictly increasing on every axis. `meanVoxelFaceArea`
    /// is the mean of the three distinct faces of one voxel,
    /// `(vx*vy + vy*vz + vz*vx) / 3` for the voxel sizes `v*`.
    pub fn new(
        min: [f32; 3],
        max: [f32; 3],
        resolution: u32,
    ) -> Result<FoliageVolumeAccumulator, RaaError> {
        if !finite3(min) || !finite3(max) {
            return Err(invalid("foliage volume bounds must be finite"));
        }
        let mut extent = [0.0f64; 3];
        for (axis, size) in extent.iter_mut().enumerate() {
            if max[axis] <= min[axis] {
                return Err(invalid(format!(
                    "foliage volume bounds are degenerate on axis {axis}: {} >= {}",
                    min[axis], max[axis]
                )));
            }
            *size = f64::from(max[axis]) - f64::from(min[axis]);
        }
        if resolution != 16 && resolution != 32 {
            return Err(invalid(format!(
                "foliage volume resolution {resolution} must be 16 or 32"
            )));
        }
        let voxel = extent.map(|size| size / f64::from(resolution));
        let mean_face_area =
            (voxel[0] * voxel[1] + voxel[1] * voxel[2] + voxel[2] * voxel[0]) / 3.0;
        let count = resolution as usize * resolution as usize * resolution as usize;
        Ok(FoliageVolumeAccumulator {
            min,
            extent,
            resolution,
            mean_face_area,
            cells: vec![Cell::default(); count],
        })
    }

    /// Adds one unlit surface sample with its represented square metres.
    ///
    /// The position clamps to the box, including the maximum boundary. Contributions land
    /// in call order so a fixed sequence reproduces the same bytes.
    pub fn add(
        &mut self,
        position: [f32; 3],
        albedo: [f32; 3],
        represented_area: f32,
    ) -> Result<(), RaaError> {
        if !finite3(position) {
            return Err(invalid("foliage volume position must be finite"));
        }
        if !finite3(albedo) {
            return Err(invalid("foliage volume albedo must be finite"));
        }
        if albedo.iter().any(|channel| !(0.0..=1.0).contains(channel)) {
            return Err(invalid("foliage volume albedo must be linear RGB in 0..=1"));
        }
        if !represented_area.is_finite() || represented_area < 0.0 {
            return Err(invalid(
                "foliage volume represented area must be finite and non-negative",
            ));
        }
        let index = self.voxel_index(position);
        let area = f64::from(represented_area);
        let cell = &mut self.cells[index];
        cell.area += area;
        for (axis, channel) in albedo.iter().enumerate() {
            cell.weighted[axis] += f64::from(*channel) * area;
        }
        Ok(())
    }

    /// Finalizes the volume into the native RGBA8 payload.
    ///
    /// An occupied cell holds the area-weighted mean albedo and
    /// `1 - exp(-area / meanVoxelFaceArea)`; an empty cell is four zero bytes.
    pub fn finish(self) -> Vec<u8> {
        let mut out = vec![0u8; self.cells.len() * 4];
        for (index, cell) in self.cells.iter().enumerate() {
            if cell.area <= 0.0 {
                continue;
            }
            let base = index * 4;
            for axis in 0..3 {
                out[base + axis] = quantize(cell.weighted[axis] / cell.area);
            }
            out[base + 3] = quantize(1.0 - (-cell.area / self.mean_face_area).exp());
        }
        out
    }

    fn voxel_index(&self, position: [f32; 3]) -> usize {
        let resolution = f64::from(self.resolution);
        let last = resolution - 1.0;
        let mut index = 0usize;
        let mut stride = 1usize;
        for (axis, coordinate) in position.iter().enumerate() {
            let normalized =
                (f64::from(*coordinate) - f64::from(self.min[axis])) / self.extent[axis];
            let voxel = (normalized * resolution).floor().clamp(0.0, last) as usize;
            index += voxel * stride;
            stride *= self.resolution as usize;
        }
        index
    }
}
