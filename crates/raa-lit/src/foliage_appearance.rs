//! Unlit source-material albedo evaluation for native foliage baking (M7, contract
//! `m7-appearance`).
//!
//! [`sample_foliage_albedo`] replays the actual source material of a foliage generation at
//! the native positions of [`FoliageSample`]s. A dedicated compute pipeline evaluates
//! `material_main` with the sample's source-root position, source normal and barycentric
//! source UV on the renderer's own material record and material-data resources, and writes
//! one linear unlit RGB value per sample. No directional light, GI, exposure or shadow term
//! participates: the value is the material output's `base_color`. Only the sampled albedo
//! storage buffer is read back, one entry per sample in sample order.
//!
//! Group 0 is the pair returned by [`LitHandle::material_compute_group`] (bindings 0-7 at
//! compute visibility, binding 8 left out); group 1 binding 0 is the read-only packed
//! sample input and binding 1 the writable rgba-f32 albedo output. These bindings never
//! enter a fragment layout.

use std::future::Future;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use raa_core::{GpuContext, MaterialId, RaaError};
use raa_shader::Composer;
use wgpu::util::DeviceExt;

use super::LitHandle;
use super::foliage_resources::{FoliageSource, SourceTriangle};
use super::foliage_sampling::FoliageSample;

/// Bytes of one packed compute sample input: four `vec4<f32>` lanes and one `vec4<u32>`.
pub const SAMPLE_INPUT_STRIDE: usize = 80;
/// Bytes of one albedo output: linear RGB and a zero W lane.
pub const SAMPLE_OUTPUT_STRIDE: usize = 16;
/// Threads of one appearance compute workgroup.
pub const APPEARANCE_WORKGROUP: u32 = 64;

/// Marker of the material half of [`raa_shader::PRELUDE_WGSL`].
const MATERIAL_SECTION: &str = "// SurfaceMaterialGpu.header.x models";
/// Marker where the prelude turns to area-light and shadow helpers that read group 1.
const AREA_SECTION: &str = "// S11 area lights";

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

/// Number of 64-thread workgroups one dispatch of `samples` needs.
pub fn dispatch_size(samples: u32) -> u32 {
    samples.div_ceil(APPEARANCE_WORKGROUP)
}

/// The dispatch of `samples` when it fits `limit` workgroups, or `None` when it does not.
pub fn checked_dispatch(samples: u32, limit: u32) -> Option<u32> {
    let groups = dispatch_size(samples);
    (groups <= limit).then_some(groups)
}

/// The barycentric interpolation of a source triangle's preserved UVs.
pub fn sample_uv(triangle: &SourceTriangle, barycentric: [f32; 3]) -> [f32; 2] {
    let mix = |axis: usize| {
        barycentric[0] * triangle.uv[0][axis]
            + barycentric[1] * triangle.uv[1][axis]
            + barycentric[2] * triangle.uv[2][axis]
    };
    [mix(0), mix(1)]
}

fn write_f32(out: &mut [u8], offset: usize, values: &[f32]) {
    for (i, value) in values.iter().enumerate() {
        out[offset + 4 * i..offset + 4 * i + 4].copy_from_slice(&value.to_le_bytes());
    }
}

fn write_u32(out: &mut [u8], offset: usize, values: &[u32]) {
    for (i, value) in values.iter().enumerate() {
        out[offset + 4 * i..offset + 4 * i + 4].copy_from_slice(&value.to_le_bytes());
    }
}

fn f32_at(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// Packs one sample into its 80-byte compute record: source-root position at 0, source
/// normal at 16, barycentric source UV at 32, barycentric weights at 48 and
/// `source_triangle, material, 0, 0` at 64. Refuses `foliage.invalid` for a source triangle
/// the source does not hold or non-finite coordinates before anything is submitted.
pub fn pack_sample(
    source: &FoliageSource,
    sample: &FoliageSample,
) -> Result<[u8; SAMPLE_INPUT_STRIDE], RaaError> {
    let triangle = source
        .triangles
        .get(sample.source_triangle as usize)
        .ok_or_else(|| {
            invalid(format!(
                "foliage sample names source triangle {} of {}",
                sample.source_triangle,
                source.triangles.len()
            ))
        })?;
    if sample
        .position
        .iter()
        .chain(sample.normal.iter())
        .chain(sample.barycentric.iter())
        .any(|value| !value.is_finite())
    {
        return Err(invalid("foliage sample coordinates must be finite"));
    }
    let uv = sample_uv(triangle, sample.barycentric);
    let mut out = [0u8; SAMPLE_INPUT_STRIDE];
    write_f32(&mut out, 0, &sample.position);
    write_f32(&mut out, 16, &sample.normal);
    write_f32(&mut out, 32, &uv);
    write_f32(&mut out, 48, &sample.barycentric);
    write_u32(
        &mut out,
        64,
        &[sample.source_triangle, triangle.material, 0, 0],
    );
    Ok(out)
}

/// Packs `samples` in order into the owned compute input records. An empty list packs to an
/// empty buffer without validating `source`; a refused sample returns `foliage.invalid`.
pub fn pack_samples(
    source: &FoliageSource,
    samples: &[FoliageSample],
) -> Result<Vec<u8>, RaaError> {
    let bytes = samples.len().saturating_mul(SAMPLE_INPUT_STRIDE);
    let mut out = Vec::new();
    out.try_reserve_exact(bytes).map_err(|_| {
        invalid(format!(
            "cannot reserve {bytes} bytes of foliage sample inputs"
        ))
    })?;
    for sample in samples {
        out.extend_from_slice(&pack_sample(source, sample)?);
    }
    Ok(out)
}

/// Composes the compute module of one source material function: the group-0 half of the
/// prelude, its material declarations (through the material-data and atlas helpers),
/// [`raa_shader::LIB_WGSL`], `material_main` and the appearance entry point of
/// `wgsl/foliage_appearance.wgsl`. The area-light, shadow and GI half of the prelude is
/// left out because it reads group 1, which the appearance pipeline owns. The composed
/// module is validated with naga; a material that does not compose returns
/// `foliage.bake_failed`.
pub fn compose_appearance(material_wgsl: &str) -> Result<String, RaaError> {
    let prelude = raa_shader::PRELUDE_WGSL;
    let head_end = prelude
        .find("@group(1)")
        .ok_or_else(|| bake_failed("foliage appearance prelude has no group 1 marker"))?;
    let material_start = prelude
        .find(MATERIAL_SECTION)
        .ok_or_else(|| bake_failed("foliage appearance prelude has no material section"))?;
    let area_start = prelude
        .find(AREA_SECTION)
        .ok_or_else(|| bake_failed("foliage appearance prelude has no area-light section"))?;
    let mut src = String::new();
    for line in prelude[..head_end].lines() {
        if !line.contains("@group(0) @binding(8)") {
            src.push_str(line);
            src.push('\n');
        }
    }
    src.push_str(&prelude[material_start..area_start]);
    src.push('\n');
    // The material section's reflection class calls the near-mirror policy, which the prelude
    // places outside the head and material sections.
    if !src.contains("fn raa_near_mirror(") {
        src.push_str(raa_shader::NEAR_MIRROR_WGSL);
        src.push('\n');
    }
    // ... and the integration's material helpers (DDGI sky term) test feature bits through
    // `raa_feature_enabled`, generated with the shared constants after the prelude (group 0 only).
    if !src.contains("fn raa_feature_enabled(") {
        let start = prelude
            .find("const FRAME_GPU_SIZE: u32")
            .ok_or_else(|| bake_failed("foliage appearance prelude has no shared constants"))?;
        let feature = prelude[start..]
            .find("fn raa_feature_enabled(")
            .map(|at| start + at)
            .ok_or_else(|| bake_failed("foliage appearance prelude has no feature test"))?;
        let end = prelude[feature..]
            .find("\n}\n")
            .map(|at| feature + at + 3)
            .ok_or_else(|| bake_failed("foliage appearance feature test is not closed"))?;
        src.push_str(&prelude[start..end]);
        src.push('\n');
    }
    src.push_str(raa_shader::LIB_WGSL);
    src.push('\n');
    src.push_str(material_wgsl);
    src.push('\n');
    src.push_str(include_str!("wgsl/foliage_appearance.wgsl"));
    Composer::validate(&src).map_err(|error| {
        bake_failed(format!(
            "foliage appearance shader {}: {}",
            error.code, error.message
        ))
    })?;
    Ok(src)
}

fn group1_entries() -> [wgpu::BindGroupLayoutEntry; 2] {
    let compute = wgpu::ShaderStages::COMPUTE;
    let storage = |binding: u32, read_only: bool, size: usize| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: compute,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(size as u64),
        },
        count: None,
    };
    [
        storage(0, true, SAMPLE_INPUT_STRIDE),
        storage(1, false, SAMPLE_OUTPUT_STRIDE),
    ]
}

#[derive(Default)]
struct MapState {
    done: bool,
    error: Option<String>,
    waker: Option<Waker>,
}

/// One mapped readback of an appearance output staging buffer.
struct Readback {
    buffer: wgpu::Buffer,
    #[cfg(not(target_arch = "wasm32"))]
    gpu: GpuContext,
    state: Arc<Mutex<MapState>>,
}

impl Readback {
    #[cfg(not(target_arch = "wasm32"))]
    fn new(gpu: &GpuContext, buffer: wgpu::Buffer) -> Readback {
        let state = Arc::new(Mutex::new(MapState::default()));
        let signal = Arc::clone(&state);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let mut s = signal.lock().unwrap_or_else(|e| e.into_inner());
                s.done = true;
                s.error = result.err().map(|e| e.to_string());
                if let Some(waker) = s.waker.take() {
                    waker.wake();
                }
            });
        Readback {
            buffer,
            gpu: gpu.clone(),
            state,
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn new(_gpu: &GpuContext, buffer: wgpu::Buffer) -> Readback {
        let state = Arc::new(Mutex::new(MapState::default()));
        let signal = Arc::clone(&state);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let mut s = signal.lock().unwrap_or_else(|e| e.into_inner());
                s.done = true;
                s.error = result.err().map(|e| e.to_string());
                if let Some(waker) = s.waker.take() {
                    waker.wake();
                }
            });
        Readback { buffer, state }
    }
}

impl Future for Readback {
    type Output = Result<Vec<u8>, RaaError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let mut state = this.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.done {
            state.waker = Some(cx.waker().clone());
            drop(state);
            #[cfg(not(target_arch = "wasm32"))]
            let _ = this.gpu.device().poll(wgpu::PollType::wait_indefinitely());
            state = this.state.lock().unwrap_or_else(|e| e.into_inner());
            if !state.done {
                return Poll::Pending;
            }
        }
        let error = state.error.take();
        drop(state);
        if let Some(error) = error {
            return Poll::Ready(Err(RaaError::Gpu(format!(
                "foliage appearance readback failed: {error}"
            ))));
        }
        let bytes = {
            let slice = this.buffer.slice(..);
            let mapped = match slice.get_mapped_range() {
                Ok(mapped) => mapped,
                Err(error) => {
                    return Poll::Ready(Err(RaaError::Gpu(format!(
                        "foliage appearance mapped range unavailable: {error}"
                    ))));
                }
            };
            mapped.to_vec()
        };
        this.buffer.unmap();
        Poll::Ready(Ok(bytes))
    }
}

/// The owned submission of one appearance call: the snapshot material source, the group-0
/// pair, the packed sample records and the renderer's GPU context.
struct AppearanceJob {
    gpu: GpuContext,
    layout: wgpu::BindGroupLayout,
    group0: wgpu::BindGroup,
    material_wgsl: String,
    inputs: Vec<u8>,
    count: u32,
}

impl AppearanceJob {
    async fn run(self) -> Result<Vec<[f32; 3]>, RaaError> {
        self.gpu.ensure_live()?;
        let device = self.gpu.device();
        let composed = compose_appearance(&self.material_wgsl)?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("foliage.appearance"),
            source: wgpu::ShaderSource::Wgsl(composed.into()),
        });
        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("foliage.appearance.samples"),
            entries: &group1_entries(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("foliage.appearance"),
            bind_group_layouts: &[Some(&self.layout), Some(&group1_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("foliage.appearance"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("sample_albedo"),
            compilation_options: Default::default(),
            cache: None,
        });
        let inputs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("foliage.appearance.samples"),
            contents: &self.inputs,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_size = u64::from(self.count) * SAMPLE_OUTPUT_STRIDE as u64;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("foliage.appearance.albedo"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("foliage.appearance.readback"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("foliage.appearance.samples"),
            layout: &group1_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inputs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("foliage.appearance"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("foliage.appearance"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &self.group0, &[]);
            pass.set_bind_group(1, &group1, &[]);
            pass.dispatch_workgroups(dispatch_size(self.count), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output_size);
        self.gpu.queue().submit([encoder.finish()]);
        let bytes = Readback::new(&self.gpu, staging).await?;
        let mut albedo = Vec::with_capacity(self.count as usize);
        for index in 0..self.count as usize {
            let offset = index * SAMPLE_OUTPUT_STRIDE;
            albedo.push([
                f32_at(&bytes, offset),
                f32_at(&bytes, offset + 4),
                f32_at(&bytes, offset + 8),
            ]);
        }
        Ok(albedo)
    }
}

fn prepare(
    gpu: &GpuContext,
    lit: &LitHandle,
    source: &FoliageSource,
    samples: &[FoliageSample],
) -> Result<Option<AppearanceJob>, RaaError> {
    if samples.is_empty() {
        return Ok(None);
    }
    let count = u32::try_from(samples.len())
        .map_err(|_| invalid("foliage appearance sample count exceeds u32"))?;
    let limit = gpu.limits().max_compute_workgroups_per_dimension;
    if checked_dispatch(count, limit).is_none() {
        return Err(invalid(format!(
            "foliage appearance dispatch {} exceeds the limit {limit}",
            dispatch_size(count)
        )));
    }
    let inputs = pack_samples(source, samples)?;
    let triangle = source
        .triangles
        .get(samples[0].source_triangle as usize)
        .ok_or_else(|| invalid("foliage sample names a missing source triangle"))?;
    let material_fn = lit.material_fn_of(MaterialId(triangle.material));
    let material_wgsl = lit.material_fn_source(material_fn).ok_or_else(|| {
        bake_failed(format!(
            "foliage source material function {} is not registered",
            material_fn.0
        ))
    })?;
    let (layout, group0) = lit.material_compute_group(gpu);
    Ok(Some(AppearanceJob {
        gpu: gpu.clone(),
        layout,
        group0,
        material_wgsl,
        inputs,
        count,
    }))
}

/// Evaluates the unlit linear albedo of the source material at every sample's native
/// position, source normal and barycentric source UV, one entry per sample in sample order.
///
/// The submission inputs are snapshotted before this function returns: the material source
/// from [`LitHandle::material_fn_source`] and the group-0 pair from
/// [`LitHandle::material_compute_group`], plus the packed sample records. The returned
/// future holds no borrow of the renderer. No directional light, GI, exposure or shadow
/// contributes. Empty samples return `[]`; a material function that is unknown or does not
/// compose, and any invalid input, returns `foliage.bake_failed` or `foliage.invalid`
/// before a submission; the private buffers are released when the submission completes.
///
/// Handoff: 'Assemble stochastic foliage and its volume' and 'Run differentiable foliage
/// optimization' call `sample_foliage_albedo` for the albedo of their native generation and
/// hand the returned linear RGB to `FoliageVolumeAccumulator::add`.
pub fn sample_foliage_albedo(
    gpu: &GpuContext,
    lit: &LitHandle,
    source: &FoliageSource,
    samples: &[FoliageSample],
) -> impl Future<Output = Result<Vec<[f32; 3]>, RaaError>> + 'static {
    let prepared = prepare(gpu, lit, source, samples);
    async move {
        match prepared? {
            None => Ok(Vec::new()),
            Some(job) => job.run().await,
        }
    }
}
