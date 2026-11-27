//! Differentiable foliage optimization (M7, contracts `m7-compute` and `m7-optimize`).
//!
//! [`FoliageCompute::new`] allocates the generation-owned triangle, gradient, counted-bin,
//! pixel and params buffers next to the retained reference views, and
//! [`FoliageCompute::finish`] copies the terminal triangle records back exactly once and
//! freezes them through `freeze_terminal` before anything is published. [`optimize_foliage`]
//! validates the material settings and the 64..=128 rendered references, seeds the
//! micro-triangles with `initialize_foliage` and runs bins, raster, gradients and update in
//! that order; an overflowing counted bin grows and re-runs the bins stage before the raster
//! consumer reads it, never truncating a triangle list. The result keeps the source wood in
//! source order, publishes only the opaque terminal triangles with their independent normals
//! and builds the shared native volume from the sampler's strata and the unlit source albedo.
//! [`FoliageOptimizeCoordinator`] is the CPU state machine the asynchronous run drives and the
//! subject tests exercise; it refuses a consumer or a publication out of order.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use raa_core::{GpuContext, RaaError};
use raa_geom::MeshData;

use crate::LitHandle;
use crate::foliage_appearance::sample_foliage_albedo;
use crate::foliage_bins::{encode_bins, needs_bin_retry, tiles_per_axis};
use crate::foliage_gradients::encode_gradients;
use crate::foliage_poisson::initialize_foliage;
use crate::foliage_raster::{PIXEL_RECORD_STRIDE, encode_raster};
use crate::foliage_resources::{FoliageCompute, FoliageReferences, FoliageSource, MicroTriangle};
use crate::foliage_sampling::{FoliageSample, sample_foliage};
use crate::foliage_stochastic::FoliageMeshBake;
use crate::foliage_update::{
    TRIANGLE_RECORD_STRIDE, encode_update, freeze_terminal, triangle_to_record,
};
use crate::foliage_volume::FoliageVolumeAccumulator;

/// Smallest accepted reference view count of one optimization (`64`).
pub const MIN_REFERENCE_VIEWS: u32 = 64;
/// Largest accepted reference view count of one optimization (`128`).
pub const MAX_REFERENCE_VIEWS: u32 = 128;
/// Triangle capacity of a freshly allocated counted bin.
pub const INITIAL_BIN_CAPACITY: u32 = 32;
/// Soft-coverage temperature of the optimizer, in reference pixels.
pub const OPTIMIZE_TEMPERATURE: f32 = 1.0;
/// Learning rate of one bounded optimizer step.
pub const OPTIMIZE_LEARNING_RATE: f32 = 0.01;
/// Opacity the terminal generation anneals onto and publishes.
pub const OPACITY_TARGET: f32 = 1.0;
/// Bytes of the optimizer params uniform.
pub const FOLIAGE_OPTIMIZE_PARAMS_SIZE: usize = 64;
/// Tolerance of the terminal unit-normal check.
pub const NORMAL_LENGTH_TOLERANCE: f32 = 1e-3;

fn invalid(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message: message.into(),
    }
}

fn bake_failed(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: "foliage.bake_failed",
        message: message.into(),
    }
}

/// The validated material settings of one differentiable foliage bake.
///
/// `count`, `resolution`, `iterations` and the three weights carry the `m7-material-settings`
/// defaults (`4096`, `32`, `64`, `1`, `1`, `0.01`); the CALLER validates the authored values
/// before calling [`optimize_foliage`], which refuses an impossible configuration again.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FoliageOptimizeSettings {
    pub count: u32,
    pub seed: u32,
    pub resolution: u32,
    pub iterations: u32,
    pub silhouette_weight: f32,
    pub normal_weight: f32,
    pub edge_weight: f32,
}

impl FoliageOptimizeSettings {
    /// The `m7-material-settings` defaults.
    pub const DEFAULT: Self = Self {
        count: 4096,
        seed: 0,
        resolution: 32,
        iterations: 64,
        silhouette_weight: 1.0,
        normal_weight: 1.0,
        edge_weight: 0.01,
    };

    /// Refuses `foliage.invalid` for a zero count or iteration budget, a resolution other
    /// than 16 or 32 and a non-finite, negative or all-zero weight set.
    pub fn validate(&self) -> Result<(), RaaError> {
        if self.count == 0 {
            return Err(invalid("foliage triangle count must be positive"));
        }
        if self.resolution != 16 && self.resolution != 32 {
            return Err(invalid(format!(
                "foliage resolution {} must be 16 or 32",
                self.resolution
            )));
        }
        if self.iterations == 0 {
            return Err(invalid("foliage iteration count must be positive"));
        }
        for (name, weight) in [
            ("silhouette", self.silhouette_weight),
            ("normal", self.normal_weight),
            ("edge", self.edge_weight),
        ] {
            if !weight.is_finite() || weight < 0.0 {
                return Err(invalid(format!(
                    "foliage {name} weight {weight} must be finite and non-negative"
                )));
            }
        }
        if self.silhouette_weight <= 0.0 && self.normal_weight <= 0.0 && self.edge_weight <= 0.0 {
            return Err(invalid("at least one foliage weight must be positive"));
        }
        Ok(())
    }
}

/// Refuses a reference view count outside the native optimization range `64..=128`.
pub fn validate_reference_count(views: u32) -> Result<(), RaaError> {
    if (MIN_REFERENCE_VIEWS..=MAX_REFERENCE_VIEWS).contains(&views) {
        Ok(())
    } else {
        Err(invalid(format!(
            "foliage optimization views {views} must be within {MIN_REFERENCE_VIEWS}..={MAX_REFERENCE_VIEWS}"
        )))
    }
}

/// The counted-bin triangle capacity of a fresh allocation: [`INITIAL_BIN_CAPACITY`], never
/// above the generation's triangle count.
pub fn initial_bin_capacity(count: u32) -> u32 {
    INITIAL_BIN_CAPACITY.min(count.max(1))
}

/// The 64-byte optimizer uniform of contract `m7-compute`: `count@0`, `views@4`,
/// `resolution@8`, `iteration@12`, `silhouetteWeight@16`, `normalWeight@20`, `edgeWeight@24`,
/// `temperature@28`, `learningRate@32`, `opacityTarget@36`, `binCapacity@40`,
/// `generation@44`, then sixteen zero bytes.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FoliageOptimizeParams {
    pub count: u32,
    pub views: u32,
    pub resolution: u32,
    pub iteration: u32,
    pub silhouette_weight: f32,
    pub normal_weight: f32,
    pub edge_weight: f32,
    pub temperature: f32,
    pub learning_rate: f32,
    pub opacity_target: f32,
    pub capacity: u32,
    pub generation: u32,
}

impl FoliageOptimizeParams {
    /// The little-endian 64-byte record in the shared `m7-compute` layout.
    pub fn to_bytes(self) -> [u8; FOLIAGE_OPTIMIZE_PARAMS_SIZE] {
        let mut out = [0u8; FOLIAGE_OPTIMIZE_PARAMS_SIZE];
        let u = |b: &mut [u8; FOLIAGE_OPTIMIZE_PARAMS_SIZE], o: usize, v: u32| {
            b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        };
        let f = |b: &mut [u8; FOLIAGE_OPTIMIZE_PARAMS_SIZE], o: usize, v: f32| {
            b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        };
        u(&mut out, 0, self.count);
        u(&mut out, 4, self.views);
        u(&mut out, 8, self.resolution);
        u(&mut out, 12, self.iteration);
        f(&mut out, 16, self.silhouette_weight);
        f(&mut out, 20, self.normal_weight);
        f(&mut out, 24, self.edge_weight);
        f(&mut out, 28, self.temperature);
        f(&mut out, 32, self.learning_rate);
        f(&mut out, 36, self.opacity_target);
        u(&mut out, 40, self.capacity);
        u(&mut out, 44, self.generation);
        out
    }
}

/// The public stage of one differentiable optimizer generation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FoliageOptimizeState {
    /// Settings, seed, references and buffers are ready; no kernel was encoded.
    Prepared,
    /// The bins stage must be encoded (or re-encoded with a larger capacity).
    Bins,
    /// The bins stage was submitted and its counts are being read.
    AwaitingBins,
    /// The counts fit; the raster stage may run.
    Raster,
    /// The raster stage was encoded; the gradient stage may run.
    Gradients,
    /// The gradient stage was encoded; the update stage may run.
    Update,
    /// Every iteration finished; only the terminal readback and publication remain.
    Finalizing,
    /// The opaque terminal generation was published.
    Published,
    /// The run was cancelled before publication.
    Cancelled,
}

/// What the counted-bin overflow check asks the optimizer to do next.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FoliageBinOutcome {
    /// A tile held more triangles than the capacity: grow and re-run the bins stage.
    Retry,
    /// Every tile fits: the raster consumer may read the bins.
    Raster,
}

/// The ordered state machine of one differentiable optimizer run.
///
/// The async [`optimize_foliage`] drives exactly these transitions around the GPU stages;
/// the subject tests drive them without a device. A consumer encoded out of order and a
/// publication before `Finalizing` return `foliage.invalid`; cancellation from any
/// non-published stage returns [`FoliageOptimizeState::Cancelled`] and forbids publication;
/// [`FoliageOptimizeCoordinator::publish`] refuses a non-finite, non-unit or non-opaque
/// terminal triangle with `foliage.bake_failed`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoliageOptimizeCoordinator {
    state: FoliageOptimizeState,
    iteration: u32,
    iterations: u32,
    capacity: u32,
}

impl FoliageOptimizeCoordinator {
    /// Starts in [`FoliageOptimizeState::Prepared`] with a positive iteration budget and
    /// counted-bin capacity; anything else returns `foliage.invalid`.
    pub fn new(iterations: u32, capacity: u32) -> Result<Self, RaaError> {
        if iterations == 0 {
            return Err(invalid("foliage optimizer needs at least one iteration"));
        }
        if capacity == 0 {
            return Err(invalid("foliage optimizer bin capacity must be positive"));
        }
        Ok(Self {
            state: FoliageOptimizeState::Prepared,
            iteration: 0,
            iterations,
            capacity,
        })
    }

    /// The current public stage.
    pub fn state(&self) -> FoliageOptimizeState {
        self.state
    }

    /// The zero-based iteration the next bins stage belongs to.
    pub fn iteration(&self) -> u32 {
        self.iteration
    }

    /// The positive iteration budget of the run.
    pub fn iterations(&self) -> u32 {
        self.iterations
    }

    /// The counted-bin triangle capacity the next bins stage writes.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    fn require(&self, expected: FoliageOptimizeState) -> Result<(), RaaError> {
        if self.state == expected {
            Ok(())
        } else {
            Err(invalid(format!(
                "foliage optimizer is in {:?}, expected {expected:?}",
                self.state
            )))
        }
    }

    /// `Prepared -> Bins`.
    pub fn begin(&mut self) -> Result<(), RaaError> {
        self.require(FoliageOptimizeState::Prepared)?;
        self.state = FoliageOptimizeState::Bins;
        Ok(())
    }

    /// `Bins -> AwaitingBins`, after the bins submission.
    pub fn bins_encoded(&mut self) -> Result<(), RaaError> {
        self.require(FoliageOptimizeState::Bins)?;
        self.state = FoliageOptimizeState::AwaitingBins;
        Ok(())
    }

    /// Consumes the submitted tile counts: an overflow grows the capacity and returns to
    /// `Bins` (never to a consumer), a fitting set releases the raster stage.
    pub fn bins_completed(&mut self, counts: &[u32]) -> Result<FoliageBinOutcome, RaaError> {
        self.require(FoliageOptimizeState::AwaitingBins)?;
        if needs_bin_retry(counts, self.capacity) {
            let largest = counts.iter().copied().max().unwrap_or(0);
            let grown = self.capacity.saturating_mul(2).max(largest);
            if grown <= self.capacity {
                return Err(invalid("foliage bin capacity cannot grow further"));
            }
            self.capacity = grown;
            self.state = FoliageOptimizeState::Bins;
            return Ok(FoliageBinOutcome::Retry);
        }
        self.state = FoliageOptimizeState::Raster;
        Ok(FoliageBinOutcome::Raster)
    }

    /// `Raster -> Gradients`, after the raster submission.
    pub fn raster_encoded(&mut self) -> Result<(), RaaError> {
        self.require(FoliageOptimizeState::Raster)?;
        self.state = FoliageOptimizeState::Gradients;
        Ok(())
    }

    /// `Gradients -> Update`, after the gradient submission.
    pub fn gradients_encoded(&mut self) -> Result<(), RaaError> {
        self.require(FoliageOptimizeState::Gradients)?;
        self.state = FoliageOptimizeState::Update;
        Ok(())
    }

    /// `Update -> Bins` of the next iteration, or `Update -> Finalizing` when the budget is
    /// spent.
    pub fn update_encoded(&mut self) -> Result<(), RaaError> {
        self.require(FoliageOptimizeState::Update)?;
        if self.iteration + 1 < self.iterations {
            self.iteration += 1;
            self.state = FoliageOptimizeState::Bins;
        } else {
            self.state = FoliageOptimizeState::Finalizing;
        }
        Ok(())
    }

    /// Cancels a non-published run; repeated cancellation is idempotent.
    pub fn cancel(&mut self) -> Result<(), RaaError> {
        if self.state == FoliageOptimizeState::Published {
            return Err(invalid(
                "a published foliage optimization cannot be cancelled",
            ));
        }
        self.state = FoliageOptimizeState::Cancelled;
        Ok(())
    }

    /// Publishes the terminal micro-triangles from `Finalizing`.
    ///
    /// Every triangle must be finite, exactly opaque and carry a unit normal; a violation is
    /// `foliage.bake_failed`, so a partial or numerical generation is never published.
    pub fn publish(&mut self, triangles: &[MicroTriangle]) -> Result<(), RaaError> {
        self.require(FoliageOptimizeState::Finalizing)?;
        for (index, triangle) in triangles.iter().enumerate() {
            let finite = triangle
                .p0
                .iter()
                .chain(triangle.p1.iter())
                .chain(triangle.p2.iter())
                .chain(triangle.normal.iter())
                .all(|value| value.is_finite())
                && triangle.opacity.is_finite();
            if !finite {
                return Err(bake_failed(format!(
                    "terminal foliage micro-triangle {index} is not finite"
                )));
            }
            if triangle.opacity != OPACITY_TARGET {
                return Err(bake_failed(format!(
                    "terminal foliage micro-triangle {index} opacity {} is not opaque",
                    triangle.opacity
                )));
            }
            let length = (triangle.normal[0] * triangle.normal[0]
                + triangle.normal[1] * triangle.normal[1]
                + triangle.normal[2] * triangle.normal[2])
                .sqrt();
            if (length - 1.0).abs() > NORMAL_LENGTH_TOLERANCE {
                return Err(bake_failed(format!(
                    "terminal foliage micro-triangle {index} normal length {length} is not unit"
                )));
            }
        }
        self.state = FoliageOptimizeState::Published;
        Ok(())
    }
}

fn push_triangle(
    mesh: &mut MeshData,
    positions: [[f32; 3]; 3],
    normals: [[f32; 3]; 3],
) -> Result<(), RaaError> {
    let base = u32::try_from(mesh.positions.len())
        .map_err(|_| invalid("foliage mesh exceeds the 32-bit vertex index range"))?;
    mesh.positions.extend_from_slice(&positions);
    mesh.normals.extend_from_slice(&normals);
    mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
    Ok(())
}

/// Assembles the published bake of a terminal optimizer generation.
///
/// The wood triangles of `source` stay in source order; `terminal` contributes one opaque
/// triangle per entry with its independent normal, never a winding-derived one. `samples`
/// and `albedo` hold the same `sample_foliage` strata and unlit linear RGB values the native
/// volume accumulates; an empty sample list emits the zeroed `4 * resolution^3` payload, so
/// empty eligible foliage preserves the wood and a zero volume.
pub fn assemble_optimized_foliage(
    source: &FoliageSource,
    terminal: &[MicroTriangle],
    samples: &[FoliageSample],
    albedo: &[[f32; 3]],
    resolution: u32,
) -> Result<FoliageMeshBake, RaaError> {
    if samples.len() != albedo.len() {
        return Err(invalid(format!(
            "foliage albedo length {} does not match sample count {}",
            albedo.len(),
            samples.len()
        )));
    }
    if resolution != 16 && resolution != 32 {
        return Err(invalid(format!(
            "foliage volume resolution {resolution} must be 16 or 32"
        )));
    }
    let mut mesh = MeshData::new();
    let wood_count = source.triangles.iter().filter(|t| !t.foliage).count();
    crate::foliage_stochastic::reserve_triangles(
        &mut mesh,
        wood_count.saturating_add(terminal.len()),
    )?;
    let mut wood = 0u32;
    for triangle in &source.triangles {
        if triangle.foliage {
            continue;
        }
        push_triangle(&mut mesh, triangle.positions, triangle.normals)?;
        wood = wood
            .checked_add(1)
            .ok_or_else(|| invalid("foliage wood triangle count overflows"))?;
    }
    let mut foliage = 0u32;
    for (index, triangle) in terminal.iter().enumerate() {
        let finite = triangle
            .p0
            .iter()
            .chain(triangle.p1.iter())
            .chain(triangle.p2.iter())
            .chain(triangle.normal.iter())
            .all(|value| value.is_finite());
        if !finite || triangle.opacity != OPACITY_TARGET {
            return Err(bake_failed(format!(
                "terminal foliage micro-triangle {index} is not finite and opaque"
            )));
        }
        push_triangle(
            &mut mesh,
            [triangle.p0, triangle.p1, triangle.p2],
            [triangle.normal; 3],
        )?;
        foliage = foliage
            .checked_add(1)
            .ok_or_else(|| invalid("foliage triangle count overflows"))?;
    }
    let volume = if samples.is_empty() {
        vec![0u8; 4 * resolution as usize * resolution as usize * resolution as usize]
    } else {
        let mut accumulator = FoliageVolumeAccumulator::new(source.min, source.max, resolution)?;
        for (sample, albedo) in samples.iter().zip(albedo) {
            accumulator.add(sample.position, *albedo, sample.represented_area)?;
        }
        accumulator.finish()
    };
    Ok(FoliageMeshBake {
        mesh,
        volume,
        foliage_triangles: foliage,
        wood_triangles: wood,
    })
}

fn bin_tiles(resolution: u32, views: u32) -> Result<u32, RaaError> {
    if resolution == 0 {
        return Err(invalid("foliage bin resolution must be positive"));
    }
    let side = tiles_per_axis(resolution);
    side.checked_mul(side)
        .and_then(|count| count.checked_mul(views))
        .ok_or_else(|| invalid("foliage bin tile count overflows"))
}

fn check_storage(gpu: &GpuContext, bytes: u64, label: &str) -> Result<(), RaaError> {
    let limits = gpu.device().limits();
    let max = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size);
    if bytes == 0 || bytes > max {
        return Err(invalid(format!(
            "foliage {label} needs {bytes} bytes; the device binds at most {max}"
        )));
    }
    Ok(())
}

fn validate_references(references: &FoliageReferences) -> Result<(u32, u32), RaaError> {
    let views = u32::try_from(references.views.len())
        .map_err(|_| invalid("foliage reference view count overflows"))?;
    if views == 0 || views > 128 {
        return Err(invalid(format!(
            "foliage compute needs 1..=128 reference views, got {views}"
        )));
    }
    let width = references.normal_coverage.width();
    let height = references.normal_coverage.height();
    let layers = references.normal_coverage.depth_or_array_layers();
    if width == 0 || width != height || layers != views {
        return Err(invalid(format!(
            "foliage references must be square with {views} layer(s)"
        )));
    }
    for (name, texture, format) in [
        (
            "normal_coverage",
            &references.normal_coverage,
            wgpu::TextureFormat::Rgba16Float,
        ),
        ("depth", &references.depth, wgpu::TextureFormat::Rgba32Float),
        (
            "albedo_coverage",
            &references.albedo_coverage,
            wgpu::TextureFormat::Rgba8Unorm,
        ),
    ] {
        if texture.format() != format
            || texture.dimension() != wgpu::TextureDimension::D2
            || texture.mip_level_count() != 1
            || texture.width() != width
            || texture.height() != height
            || texture.depth_or_array_layers() != views
        {
            return Err(invalid(format!(
                "foliage {name} reference must be {width}x{height} {format:?} with {views} layer(s)"
            )));
        }
    }
    for (matrix, near_far) in &references.views {
        if matrix
            .iter()
            .chain(near_far.iter())
            .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "foliage view matrices and near/far values must be finite",
            ));
        }
    }
    Ok((views, width))
}

fn validate_triangles(triangles: &[MicroTriangle]) -> Result<(), RaaError> {
    for (index, triangle) in triangles.iter().enumerate() {
        let finite = triangle
            .p0
            .iter()
            .chain(triangle.p1.iter())
            .chain(triangle.p2.iter())
            .chain(triangle.normal.iter())
            .all(|value| value.is_finite());
        if !finite || !triangle.opacity.is_finite() {
            return Err(invalid(format!(
                "foliage micro-triangle {index} is not finite"
            )));
        }
        if !(0.0..=1.0).contains(&triangle.opacity) {
            return Err(invalid(format!(
                "foliage micro-triangle {index} opacity {} must be within 0..=1",
                triangle.opacity
            )));
        }
    }
    Ok(())
}

impl FoliageCompute {
    /// Allocates one optimizer generation over `triangles` and the retained `references`.
    ///
    /// The triangle and gradient buffers hold one five-lane 80-byte record per
    /// micro-triangle, the counted bins hold `16x16` pixel tiles with
    /// [`initial_bin_capacity`] triangle slots each, and the pixel buffer holds one 32-byte
    /// record per view pixel. The params uniform starts with the `m7-material-settings`
    /// weights and iteration zero. References must be square D2 arrays with one mip and the
    /// same layer count as the view table; a non-finite triangle, an empty generation or an
    /// allocation above the device storage limit returns `foliage.invalid` before the
    /// generation exists.
    pub fn new(
        gpu: &GpuContext,
        triangles: &[MicroTriangle],
        references: FoliageReferences,
        generation: u32,
    ) -> Result<FoliageCompute, RaaError> {
        gpu.ensure_live()?;
        let count = u32::try_from(triangles.len())
            .map_err(|_| invalid("foliage micro-triangle count overflows"))?;
        if count == 0 {
            return Err(invalid("foliage compute needs at least one micro-triangle"));
        }
        validate_triangles(triangles)?;
        let (views, resolution) = validate_references(&references)?;
        let device = gpu.device();
        let record_bytes = u64::from(count) * TRIANGLE_RECORD_STRIDE as u64;
        check_storage(gpu, record_bytes, "triangle and gradient buffers")?;
        let capacity = initial_bin_capacity(count);
        let tiles = bin_tiles(resolution, views)?;
        let bin_bytes = u64::from(tiles)
            .checked_mul(u64::from(capacity + 1))
            .and_then(|words| words.checked_mul(4))
            .ok_or_else(|| invalid("foliage bin buffer size overflows"))?;
        check_storage(gpu, bin_bytes, "bin buffer")?;
        let pixel_bytes = u64::from(resolution)
            .checked_mul(u64::from(resolution))
            .and_then(|pixels| pixels.checked_mul(u64::from(views)))
            .and_then(|pixels| pixels.checked_mul(PIXEL_RECORD_STRIDE as u64))
            .ok_or_else(|| invalid("foliage pixel buffer size overflows"))?;
        check_storage(gpu, pixel_bytes, "pixel buffer")?;
        let storage = wgpu::BufferUsages::STORAGE;
        let triangles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa.lit.foliage.optimize.triangles"),
            size: record_bytes,
            usage: storage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let gradients = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa.lit.foliage.optimize.gradients"),
            size: record_bytes,
            usage: storage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bins = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa.lit.foliage.optimize.bins"),
            size: bin_bytes,
            usage: storage | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let pixels = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa.lit.foliage.optimize.pixels"),
            size: pixel_bytes,
            usage: storage,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa.lit.foliage.optimize.params"),
            size: FOLIAGE_OPTIMIZE_PARAMS_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let defaults = FoliageOptimizeSettings::DEFAULT;
        let params_record = FoliageOptimizeParams {
            count,
            views,
            resolution,
            iteration: 0,
            silhouette_weight: defaults.silhouette_weight,
            normal_weight: defaults.normal_weight,
            edge_weight: defaults.edge_weight,
            temperature: OPTIMIZE_TEMPERATURE,
            learning_rate: OPTIMIZE_LEARNING_RATE,
            opacity_target: OPACITY_TARGET,
            capacity,
            generation,
        };
        gpu.queue()
            .write_buffer(&params, 0, &params_record.to_bytes());
        let record_len = triangles.len() * TRIANGLE_RECORD_STRIDE;
        let mut records = Vec::new();
        records.try_reserve_exact(record_len).map_err(|_| {
            invalid(format!(
                "cannot reserve {record_len} bytes of foliage triangle records"
            ))
        })?;
        for triangle in triangles {
            records.extend_from_slice(&triangle_to_record(*triangle));
        }
        gpu.queue().write_buffer(&triangles_buffer, 0, &records);
        Ok(FoliageCompute {
            triangles: triangles_buffer,
            gradients,
            bins,
            pixels,
            references,
            params,
            count,
            views,
            resolution,
            generation,
        })
    }

    /// Reads the terminal triangle records back exactly once and returns them frozen.
    ///
    /// The returned future owns the generation until the copy completes, submits one
    /// buffer-to-buffer copy and only then maps its staging buffer, so readiness is never
    /// claimed before queue completion. [`freeze_terminal`] normalizes every retained normal,
    /// writes the exact opaque opacity and refuses a non-finite record with
    /// `foliage.bake_failed`; a lost device or a failed map returns `foliage.bake_failed` and
    /// never a partial generation.
    pub fn finish(
        self,
        gpu: &GpuContext,
    ) -> impl Future<Output = Result<Vec<MicroTriangle>, RaaError>> + 'static {
        let gpu = gpu.clone();
        async move {
            if let Err(error) = gpu.ensure_live() {
                return Err(bake_failed(format!(
                    "foliage terminal readback unavailable: {}",
                    error.message()
                )));
            }
            let bytes = u64::from(self.count) * TRIANGLE_RECORD_STRIDE as u64;
            let staging = gpu.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("raa.lit.foliage.optimize.readback"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder =
                gpu.device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("raa.lit.foliage.optimize.finish"),
                    });
            encoder.copy_buffer_to_buffer(&self.triangles, 0, &staging, 0, bytes);
            gpu.queue().submit([encoder.finish()]);
            let mapped = BufferRead::new(&gpu, staging).await?;
            let triangles = freeze_terminal(&mapped, self.count)?;
            drop(self);
            Ok(triangles)
        }
    }
}

#[derive(Default)]
struct MapState {
    done: bool,
    error: Option<String>,
    waker: Option<Waker>,
}

/// One immutable staging readback: the map callback sets the state and the poll waits for
/// queue completion on a native host.
struct BufferRead {
    buffer: wgpu::Buffer,
    /// `None` copies the whole mapped range; `Some(stride)` keeps only the first 32-bit
    /// word of every `stride`-byte record (see [`read_record_heads`]).
    stride: Option<u64>,
    #[cfg(not(target_arch = "wasm32"))]
    gpu: GpuContext,
    state: Arc<Mutex<MapState>>,
}

impl BufferRead {
    #[cfg(not(target_arch = "wasm32"))]
    fn new(gpu: &GpuContext, buffer: wgpu::Buffer) -> BufferRead {
        let state = Arc::new(Mutex::new(MapState::default()));
        let signal = Arc::clone(&state);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let mut state = signal.lock().unwrap_or_else(|error| error.into_inner());
                state.done = true;
                state.error = result.err().map(|error| error.to_string());
                if let Some(waker) = state.waker.take() {
                    waker.wake();
                }
            });
        BufferRead {
            buffer,
            stride: None,
            gpu: gpu.clone(),
            state,
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn new(_gpu: &GpuContext, buffer: wgpu::Buffer) -> BufferRead {
        let state = Arc::new(Mutex::new(MapState::default()));
        let signal = Arc::clone(&state);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let mut state = signal.lock().unwrap_or_else(|error| error.into_inner());
                state.done = true;
                state.error = result.err().map(|error| error.to_string());
                if let Some(waker) = state.waker.take() {
                    waker.wake();
                }
            });
        BufferRead {
            buffer,
            stride: None,
            state,
        }
    }
}

impl BufferRead {
    /// Reads only the first word of every `stride`-byte record of the mapped buffer.
    fn record_heads(mut self, stride: u64) -> BufferRead {
        self.stride = Some(stride);
        self
    }
}

/// Bytes of one mapped window a chunked readback copies into the heap at a time.
const READBACK_WINDOW: u64 = 1 << 20;

/// The first little-endian 32-bit word of every `stride`-byte record of the mapped
/// `buffer`, as bytes. It walks the buffer in whole-record windows of about
/// [`READBACK_WINDOW`] bytes, so the heap never holds the full mapped range (a mapped range
/// is a heap copy on the web backend); the result is reserved fallibly.
fn read_record_heads(buffer: &wgpu::Buffer, stride: u64) -> Result<Vec<u8>, RaaError> {
    let size = buffer.size();
    if stride < 4 || !size.is_multiple_of(stride) {
        return Err(bake_failed(format!(
            "foliage readback of {size} bytes is not whole {stride}-byte records"
        )));
    }
    let records = size / stride;
    let mut heads = Vec::new();
    heads
        .try_reserve_exact(records as usize * 4)
        .map_err(|_| bake_failed(format!("cannot reserve {} readback bytes", records * 4)))?;
    // Window offsets stay multiples of the 8-byte map alignment: an even record count
    // per window when the stride is only 4-aligned.
    let mut per_window = (READBACK_WINDOW / stride).max(2);
    if !stride.is_multiple_of(8) {
        per_window -= per_window % 2;
    }
    let mut first = 0u64;
    while first < records {
        let last = (first + per_window).min(records);
        let mapped = buffer
            .slice(first * stride..last * stride)
            .get_mapped_range()
            .map_err(|error| bake_failed(format!("foliage mapped range unavailable: {error}")))?;
        for record in mapped.chunks_exact(stride as usize) {
            heads.extend_from_slice(&record[..4]);
        }
        first = last;
    }
    Ok(heads)
}

/// A fallible heap copy of the whole mapped `buffer`.
fn read_all(buffer: &wgpu::Buffer) -> Result<Vec<u8>, RaaError> {
    let mapped = buffer
        .slice(..)
        .get_mapped_range()
        .map_err(|error| bake_failed(format!("foliage mapped range unavailable: {error}")))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(mapped.len())
        .map_err(|_| bake_failed(format!("cannot reserve {} readback bytes", mapped.len())))?;
    bytes.extend_from_slice(&mapped);
    Ok(bytes)
}

impl Future for BufferRead {
    type Output = Result<Vec<u8>, RaaError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let mut state = this.state.lock().unwrap_or_else(|error| error.into_inner());
        if !state.done {
            state.waker = Some(cx.waker().clone());
            drop(state);
            #[cfg(not(target_arch = "wasm32"))]
            let _ = this.gpu.device().poll(wgpu::PollType::wait_indefinitely());
            state = this.state.lock().unwrap_or_else(|error| error.into_inner());
            if !state.done {
                return Poll::Pending;
            }
        }
        let error = state.error.take();
        drop(state);
        if let Some(error) = error {
            return Poll::Ready(Err(bake_failed(format!(
                "foliage readback failed: {error}"
            ))));
        }
        let bytes = match this.stride {
            Some(stride) => read_record_heads(&this.buffer, stride),
            None => read_all(&this.buffer),
        };
        this.buffer.unmap();
        Poll::Ready(bytes)
    }
}

fn write_params(
    gpu: &GpuContext,
    work: &FoliageCompute,
    settings: &FoliageOptimizeSettings,
    iteration: u32,
    capacity: u32,
) {
    let params = FoliageOptimizeParams {
        count: work.count,
        views: work.views,
        resolution: work.resolution,
        iteration,
        silhouette_weight: settings.silhouette_weight,
        normal_weight: settings.normal_weight,
        edge_weight: settings.edge_weight,
        temperature: OPTIMIZE_TEMPERATURE,
        learning_rate: OPTIMIZE_LEARNING_RATE,
        opacity_target: OPACITY_TARGET,
        capacity,
        generation: work.generation,
    };
    gpu.queue()
        .write_buffer(&work.params, 0, &params.to_bytes());
}

fn resize_bins(gpu: &GpuContext, work: &mut FoliageCompute, capacity: u32) -> Result<(), RaaError> {
    let tiles = bin_tiles(work.resolution, work.views)?;
    let slots = capacity
        .checked_add(1)
        .ok_or_else(|| invalid("foliage bin capacity overflows"))?;
    let bytes = u64::from(tiles)
        .checked_mul(u64::from(slots))
        .and_then(|words| words.checked_mul(4))
        .ok_or_else(|| invalid("foliage bin buffer size overflows"))?;
    check_storage(gpu, bytes, "bin buffer")?;
    work.bins = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("raa.lit.foliage.optimize.bins"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    Ok(())
}

fn read_bin_counts(
    gpu: &GpuContext,
    bins: &wgpu::Buffer,
    capacity: u32,
    tiles: u32,
) -> impl Future<Output = Result<Vec<u32>, RaaError>> + 'static {
    let gpu = gpu.clone();
    let bins = bins.clone();
    let stride = (capacity as u64 + 1) * 4;
    let size = stride * u64::from(tiles);
    async move {
        if size == 0 {
            return Ok(Vec::new());
        }
        let staging = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa.lit.foliage.optimize.bin_counts"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("raa.lit.foliage.optimize.bin_counts"),
            });
        encoder.copy_buffer_to_buffer(&bins, 0, &staging, 0, size);
        gpu.queue().submit([encoder.finish()]);
        // Only each tile's count word reaches the heap, never the full bin records.
        let heads = BufferRead::new(&gpu, staging).record_heads(stride).await?;
        let mut counts = Vec::new();
        counts
            .try_reserve_exact(heads.len() / 4)
            .map_err(|_| bake_failed(format!("cannot reserve {} bin counts", heads.len() / 4)))?;
        counts.extend(
            heads
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_le_bytes(*word)),
        );
        Ok(counts)
    }
}

type AlbedoFuture = Pin<Box<dyn Future<Output = Result<Vec<[f32; 3]>, RaaError>> + 'static>>;

/// A host yield: a future that resolves on a later turn of the host event loop (in a
/// browser, a later task, so timers, heartbeats and frames of the page run before it).
pub type FoliageHostYield = fn() -> Pin<Box<dyn Future<Output = ()> + 'static>>;

thread_local! {
    static HOST_YIELD: std::cell::Cell<Option<FoliageHostYield>> =
        const { std::cell::Cell::new(None) };
}

/// Installs (`Some`) or clears (`None`) the host yield [`optimize_foliage`] awaits between
/// its optimization steps and after each GPU readback. raa-js installs a zero-delay timeout
/// so the page heartbeat keeps running during the whole bake; a native host installs none.
pub fn set_foliage_host_yield(host_yield: Option<FoliageHostYield>) {
    HOST_YIELD.with(|cell| cell.set(host_yield));
}

/// Resolves once every submission made before its creation has completed on the GPU; a
/// native host polls the device, a browser resumes in the task of the work-done promise.
struct WorkDone {
    #[cfg(not(target_arch = "wasm32"))]
    gpu: GpuContext,
    state: Arc<Mutex<MapState>>,
}

fn work_done(gpu: &GpuContext) -> WorkDone {
    let state = Arc::new(Mutex::new(MapState::default()));
    let signal = Arc::clone(&state);
    gpu.queue().on_submitted_work_done(move || {
        let mut state = signal.lock().unwrap_or_else(|error| error.into_inner());
        state.done = true;
        if let Some(waker) = state.waker.take() {
            waker.wake();
        }
    });
    WorkDone {
        #[cfg(not(target_arch = "wasm32"))]
        gpu: gpu.clone(),
        state,
    }
}

impl Future for WorkDone {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        let mut state = this.state.lock().unwrap_or_else(|error| error.into_inner());
        if !state.done {
            state.waker = Some(cx.waker().clone());
            drop(state);
            #[cfg(not(target_arch = "wasm32"))]
            let _ = this.gpu.device().poll(wgpu::PollType::wait_indefinitely());
            state = this.state.lock().unwrap_or_else(|error| error.into_inner());
            if !state.done {
                return Poll::Pending;
            }
        }
        Poll::Ready(())
    }
}

/// One yield between optimization steps: waits until the submitted GPU work is done, so at
/// most one step is queued and the browser resumes the bake in a later task, then awaits
/// the installed host yield. Changes no setting, iteration or result.
async fn yield_step(gpu: &GpuContext) {
    work_done(gpu).await;
    let host_yield = HOST_YIELD.with(|cell| cell.get());
    if let Some(host_yield) = host_yield {
        host_yield().await;
    }
}

struct OptimizeJob {
    gpu: GpuContext,
    source: FoliageSource,
    samples: Vec<FoliageSample>,
    albedo: AlbedoFuture,
    work: FoliageCompute,
    coordinator: FoliageOptimizeCoordinator,
    settings: FoliageOptimizeSettings,
}

enum Prepared {
    Empty {
        source: FoliageSource,
        resolution: u32,
    },
    Job(Box<OptimizeJob>),
}

/// Stage of an optimizer run: settings, references, seeding, sampling and allocation.
pub const STAGE_PREPARE: &str = "preparation";
/// Stage of an optimizer run: the unlit source albedo of the volume samples.
pub const STAGE_ALBEDO: &str = "albedo sampling";
/// Stage of an optimizer run: the counted-bin submission of one iteration.
pub const STAGE_BINS: &str = "bins";
/// Stage of an optimizer run: the readback of the counted-bin tile counts.
pub const STAGE_BIN_READBACK: &str = "bin readback";
/// Stage of an optimizer run: the counted-bin overflow check and regrowth.
pub const STAGE_BIN_GROWTH: &str = "bin growth";
/// Stage of an optimizer run: the raster, gradient and update submission of one iteration.
pub const STAGE_ITERATION: &str = "raster, gradients and update";
/// Stage of an optimizer run: the single readback of the terminal triangles.
pub const STAGE_TERMINAL_READBACK: &str = "terminal readback";
/// Stage of an optimizer run: the opaque publication check of the terminal triangles.
pub const STAGE_PUBLICATION: &str = "publication";
/// Stage of an optimizer run: the wood-preserving proxy mesh and the native volume.
pub const STAGE_ASSEMBLY: &str = "assembly";

/// The live progress of one optimizer run: its source generation, the stage it is in and
/// the zero-based iteration of its `iterations` budget.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FoliageOptimizeProgress {
    pub generation: u32,
    pub stage: &'static str,
    pub iteration: u32,
    pub iterations: u32,
}

struct ProgressSlot {
    ticket: u64,
    progress: FoliageOptimizeProgress,
}

static PROGRESS: Mutex<Vec<ProgressSlot>> = Mutex::new(Vec::new());
static NEXT_TICKET: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn with_progress<R>(f: impl FnOnce(&mut Vec<ProgressSlot>) -> R) -> R {
    let mut slots = PROGRESS.lock().unwrap_or_else(|error| error.into_inner());
    f(&mut slots)
}

/// The live progress of the newest unsettled optimizer run of `generation`, or `None`
/// when no run of that generation is in flight (never started, finished or failed).
pub fn foliage_optimize_progress(generation: u32) -> Option<FoliageOptimizeProgress> {
    with_progress(|slots| {
        slots
            .iter()
            .rev()
            .find(|slot| slot.progress.generation == generation)
            .map(|slot| slot.progress)
    })
}

/// Lists one optimizer run in the progress registry from its registration until it is
/// dropped, which happens when the run settles (ready or error) or is abandoned.
#[derive(Debug)]
pub struct FoliageOptimizeProgressTicket {
    ticket: u64,
    generation: u32,
    iterations: u32,
}

impl FoliageOptimizeProgressTicket {
    /// Registers a run of `generation` with an `iterations` budget in [`STAGE_PREPARE`].
    pub fn register(generation: u32, iterations: u32) -> Self {
        let ticket = NEXT_TICKET.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        with_progress(|slots| {
            slots.push(ProgressSlot {
                ticket,
                progress: FoliageOptimizeProgress {
                    generation,
                    stage: STAGE_PREPARE,
                    iteration: 0,
                    iterations,
                },
            })
        });
        Self {
            ticket,
            generation,
            iterations,
        }
    }

    /// The source generation of the run.
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// The iteration budget the run was registered with.
    pub fn iterations(&self) -> u32 {
        self.iterations
    }

    /// Publishes that the run entered `stage` during zero-based `iteration`.
    pub fn set(&self, stage: &'static str, iteration: u32) {
        with_progress(|slots| {
            if let Some(slot) = slots.iter_mut().find(|slot| slot.ticket == self.ticket) {
                slot.progress.stage = stage;
                slot.progress.iteration = iteration;
            }
        });
    }
}

impl Drop for FoliageOptimizeProgressTicket {
    fn drop(&mut self) {
        with_progress(|slots| slots.retain(|slot| slot.ticket != self.ticket));
    }
}

/// Names the failing `stage` and zero-based `iteration` in `error`: the message reads
/// `foliage optimizer failed at stage '<stage>' (iteration <i> of <n>): <cause>`. The code
/// of a `foliage.*` refusal is kept; any other error becomes `foliage.bake_failed`.
pub fn optimize_stage_error(
    error: RaaError,
    stage: &'static str,
    iteration: u32,
    iterations: u32,
) -> RaaError {
    let code = match &error {
        RaaError::FrameGraph { code, .. } => code,
        _ => "foliage.bake_failed",
    };
    RaaError::FrameGraph {
        code,
        message: format!(
            "foliage optimizer failed at stage '{stage}' (iteration {iteration} of {iterations}): {}",
            error.message()
        ),
    }
}

/// The device side of one optimizer run that [`drive_foliage_optimizer`] sequences. The
/// renderer implements it with the GPU stages; the subject tests implement it on the CPU.
pub trait FoliageOptimizeSteps {
    /// Writes the params of `iteration` with `capacity` and submits the counted bins.
    fn submit_bins(&mut self, iteration: u32, capacity: u32) -> Result<(), RaaError>;
    /// Reads back the per-tile triangle counts of the last bins submission.
    fn bin_counts(&mut self, capacity: u32) -> impl Future<Output = Result<Vec<u32>, RaaError>>;
    /// Reallocates the counted bins for the grown `capacity`.
    fn grow_bins(&mut self, capacity: u32) -> Result<(), RaaError>;
    /// Encodes and submits raster, gradients and update, advancing `coordinator` through
    /// `raster_encoded`, `gradients_encoded` and `update_encoded`.
    fn submit_iteration(
        &mut self,
        coordinator: &mut FoliageOptimizeCoordinator,
    ) -> Result<(), RaaError>;
    /// Reads the terminal triangles back once, frozen opaque.
    fn terminal(&mut self) -> impl Future<Output = Result<Vec<MicroTriangle>, RaaError>>;
    /// Waits for the submitted work and lets the host run (the step yield).
    fn settle(&mut self) -> impl Future<Output = ()>;
}

/// Drives one optimizer run from `Prepared` to `Published` through `steps` and returns the
/// published terminal triangles.
///
/// Every iteration submits the bins, reads their counts, grows and re-runs them on an
/// overflow, then submits raster, gradients and update; the terminal readback is checked by
/// [`FoliageOptimizeCoordinator::publish`]. Before each stage the run's `progress` ticket
/// is set to that stage and iteration, so a waiting caller sees it advance, and every error
/// is returned through [`optimize_stage_error`] naming the stage that failed. The loop ends
/// after the coordinator's finite iteration budget or at the first error; a bin capacity
/// that cannot grow is an error, never another retry.
pub async fn drive_foliage_optimizer<S: FoliageOptimizeSteps>(
    steps: &mut S,
    coordinator: &mut FoliageOptimizeCoordinator,
    progress: &FoliageOptimizeProgressTicket,
) -> Result<Vec<MicroTriangle>, RaaError> {
    let iterations = coordinator.iterations();
    let at = |stage: &'static str, iteration: u32| {
        progress.set(stage, iteration);
        move |error: RaaError| optimize_stage_error(error, stage, iteration, iterations)
    };
    coordinator.begin().map_err(at(STAGE_BINS, 0))?;
    while coordinator.state() != FoliageOptimizeState::Finalizing {
        let iteration = coordinator.iteration();
        loop {
            let fail = at(STAGE_BINS, iteration);
            steps
                .submit_bins(iteration, coordinator.capacity())
                .map_err(fail)?;
            coordinator.bins_encoded().map_err(fail)?;
            let fail = at(STAGE_BIN_READBACK, iteration);
            let counts = steps
                .bin_counts(coordinator.capacity())
                .await
                .map_err(fail)?;
            steps.settle().await;
            let fail = at(STAGE_BIN_GROWTH, iteration);
            match coordinator.bins_completed(&counts).map_err(fail)? {
                FoliageBinOutcome::Retry => {
                    steps.grow_bins(coordinator.capacity()).map_err(fail)?;
                }
                FoliageBinOutcome::Raster => break,
            }
        }
        // The stage is set before the step runs (a method-call argument is evaluated after
        // its receiver, so `at(..)` inside `map_err` would report the stage too late).
        let fail = at(STAGE_ITERATION, iteration);
        steps.submit_iteration(coordinator).map_err(fail)?;
        steps.settle().await;
    }
    let last = iterations - 1;
    let fail = at(STAGE_TERMINAL_READBACK, last);
    let triangles = steps.terminal().await.map_err(fail)?;
    steps.settle().await;
    let fail = at(STAGE_PUBLICATION, last);
    coordinator.publish(&triangles).map_err(fail)?;
    Ok(triangles)
}

/// The GPU implementation of [`FoliageOptimizeSteps`] over one generation's buffers.
struct GpuSteps {
    gpu: GpuContext,
    work: Option<FoliageCompute>,
    settings: FoliageOptimizeSettings,
    tiles: u32,
}

impl GpuSteps {
    fn work(&self) -> Result<&FoliageCompute, RaaError> {
        self.work
            .as_ref()
            .ok_or_else(|| invalid("foliage optimizer buffers were already read back"))
    }
}

impl FoliageOptimizeSteps for GpuSteps {
    fn submit_bins(&mut self, iteration: u32, capacity: u32) -> Result<(), RaaError> {
        let work = self.work()?;
        write_params(&self.gpu, work, &self.settings, iteration, capacity);
        let mut encoder =
            self.gpu
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("raa.lit.foliage.optimize.bins"),
                });
        encode_bins(&self.gpu, &mut encoder, work)?;
        self.gpu.queue().submit([encoder.finish()]);
        Ok(())
    }

    fn bin_counts(&mut self, capacity: u32) -> impl Future<Output = Result<Vec<u32>, RaaError>> {
        let read = self
            .work()
            .map(|work| read_bin_counts(&self.gpu, &work.bins, capacity, self.tiles));
        async move { read?.await }
    }

    fn grow_bins(&mut self, capacity: u32) -> Result<(), RaaError> {
        let work = self
            .work
            .as_mut()
            .ok_or_else(|| invalid("foliage optimizer buffers were already read back"))?;
        resize_bins(&self.gpu, work, capacity)
    }

    fn submit_iteration(
        &mut self,
        coordinator: &mut FoliageOptimizeCoordinator,
    ) -> Result<(), RaaError> {
        let work = self.work()?;
        let mut encoder =
            self.gpu
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("raa.lit.foliage.optimize.iteration"),
                });
        encode_raster(&self.gpu, &mut encoder, work)?;
        coordinator.raster_encoded()?;
        encode_gradients(&self.gpu, &mut encoder, work)?;
        coordinator.gradients_encoded()?;
        encode_update(&self.gpu, &mut encoder, work)?;
        self.gpu.queue().submit([encoder.finish()]);
        coordinator.update_encoded()
    }

    fn terminal(&mut self) -> impl Future<Output = Result<Vec<MicroTriangle>, RaaError>> {
        let work = self.work.take();
        let gpu = self.gpu.clone();
        async move {
            match work {
                Some(work) => work.finish(&gpu).await,
                None => Err(invalid("foliage optimizer buffers were already read back")),
            }
        }
    }

    fn settle(&mut self) -> impl Future<Output = ()> {
        yield_step(&self.gpu)
    }
}

impl OptimizeJob {
    async fn run(
        self,
        progress: &FoliageOptimizeProgressTicket,
    ) -> Result<FoliageMeshBake, RaaError> {
        let OptimizeJob {
            gpu,
            source,
            samples,
            albedo,
            work,
            mut coordinator,
            settings,
        } = self;
        let iterations = coordinator.iterations();
        progress.set(STAGE_ALBEDO, 0);
        let albedo = albedo
            .await
            .map_err(|error| optimize_stage_error(error, STAGE_ALBEDO, 0, iterations))?;
        yield_step(&gpu).await;
        let tiles = bin_tiles(work.resolution, work.views)
            .map_err(|error| optimize_stage_error(error, STAGE_PREPARE, 0, iterations))?;
        let mut steps = GpuSteps {
            gpu,
            work: Some(work),
            settings,
            tiles,
        };
        let triangles = drive_foliage_optimizer(&mut steps, &mut coordinator, progress).await?;
        let last = iterations - 1;
        progress.set(STAGE_ASSEMBLY, last);
        assemble_optimized_foliage(&source, &triangles, &samples, &albedo, settings.resolution)
            .map_err(|error| optimize_stage_error(error, STAGE_ASSEMBLY, last, iterations))
    }
}

fn prepare(
    gpu: &GpuContext,
    lit: &LitHandle,
    source: FoliageSource,
    references: FoliageReferences,
    settings: FoliageOptimizeSettings,
) -> Result<Prepared, RaaError> {
    settings.validate()?;
    let views = u32::try_from(references.views.len())
        .map_err(|_| invalid("foliage reference view count overflows"))?;
    validate_reference_count(views)?;
    let triangles = initialize_foliage(&source, settings.count, settings.seed)?;
    if triangles.is_empty() {
        return Ok(Prepared::Empty {
            source,
            resolution: settings.resolution,
        });
    }
    let samples = sample_foliage(&source, settings.count, settings.seed)?;
    if samples.is_empty() {
        return Ok(Prepared::Empty {
            source,
            resolution: settings.resolution,
        });
    }
    let work = FoliageCompute::new(gpu, &triangles, references, source.generation)?;
    let capacity = initial_bin_capacity(work.count);
    let coordinator = FoliageOptimizeCoordinator::new(settings.iterations, capacity)?;
    let albedo = Box::pin(sample_foliage_albedo(gpu, lit, &source, &samples));
    Ok(Prepared::Job(Box::new(OptimizeJob {
        gpu: gpu.clone(),
        source,
        samples,
        albedo,
        work,
        coordinator,
        settings,
    })))
}

/// Runs the differentiable foliage optimizer of one tree generation.
///
/// The settings are validated against `m7-material-settings`, the reference table must hold
/// 64..=128 rendered views and the micro-triangles are seeded by `initialize_foliage` and
/// run through bins, raster, gradients and update for every iteration. An overflowing
/// counted bin grows and re-runs the bins stage before the raster consumer reads it. The
/// terminal readback is frozen opaque through `freeze_terminal`, the wood of `source` is
/// retained in source order and only the final opaque mesh with independent normals and the
/// native source-albedo/occlusion volume is returned; empty eligible foliage yields the wood
/// and a zeroed volume without a zero-workgroup dispatch. A numerical or GPU failure rejects
/// `foliage.bake_failed` and never a partial ready asset. The returned future snapshots its
/// inputs and holds no renderer borrow. It yields to the host event loop (GPU work done,
/// then the [`set_foliage_host_yield`] hook) after the albedo samples, after every bin
/// readback, after every iteration and after the terminal readback.
///
/// From its call until it settles, the run is listed in [`foliage_optimize_progress`] under
/// `source.generation` with its current stage and iteration, and every error names the
/// stage that failed ([`optimize_stage_error`]); the iteration loop is
/// [`drive_foliage_optimizer`].
///
/// Handoff: 'Publish completed foliage assets in the renderer' (card #9135) calls
/// `optimize_foliage`, which today validates the m7 settings and the 64..=128 references,
/// seeds the micro-triangles, runs the ordered optimizer iterations with bin-overflow
/// retries, freezes the terminal opaque generation and assembles the wood-preserving mesh
/// with the native source-albedo/occlusion volume.
pub fn optimize_foliage(
    gpu: &GpuContext,
    lit: &LitHandle,
    source: FoliageSource,
    references: FoliageReferences,
    settings: FoliageOptimizeSettings,
) -> impl Future<Output = Result<FoliageMeshBake, RaaError>> + 'static {
    let progress = FoliageOptimizeProgressTicket::register(source.generation, settings.iterations);
    let prepared = prepare(gpu, lit, source, references, settings)
        .map_err(|error| optimize_stage_error(error, STAGE_PREPARE, 0, settings.iterations));
    async move {
        match prepared? {
            Prepared::Empty { source, resolution } => {
                progress.set(STAGE_ASSEMBLY, 0);
                assemble_optimized_foliage(&source, &[], &[], &[], resolution).map_err(|error| {
                    optimize_stage_error(error, STAGE_ASSEMBLY, 0, settings.iterations)
                })
            }
            Prepared::Job(job) => job.run(&progress).await,
        }
    }
}
