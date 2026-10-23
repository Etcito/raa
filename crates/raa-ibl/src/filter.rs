//! GPU filtering of the sky cube (IBL value): source mips 1..5 by box downsample
//! (`wgsl/downsample.wgsl`), mip 0 copied into the IBL-owned output cube, output mips 1..5
//! by GGX importance sampling with PDF-based source mip (`wgsl/prefilter.wgsl`), the SH9
//! projection by compute reduction (`wgsl/sh_project.wgsl`), then the 9 coefficients copied
//! into a ring of 3 staging buffers for asynchronous readback.
//!
//! Everything is allocated by `Filter::new`; a `record` only creates bind groups when the
//! source texture changes (cached for the last source). Readback: `record` copies into the
//! next free staging buffer (or sets the retry flag), the caller submits, then
//! `after_submit` maps it. The map callback stores the coefficients with their sequence
//! number and unmaps. On wasm the browser drives map callbacks; native callers poll the
//! device (`Device::poll`). The shared slot is an `Arc<Mutex<..>>` rather than an
//! `Rc<RefCell<..>>` because wgpu requires a `Send` map callback on native.

#![allow(
    dead_code,
    reason = "used by IblHandle::filter / apply_pending (later task)"
)]

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use raa_core::{GpuContext, RaaError};

/// SKY_ENV face size in px.
pub(crate) const ENV_SIZE: u32 = 256;
/// SKY_ENV mip count (256..8).
pub(crate) const ENV_MIPS: u32 = 6;
/// HDR_FORMAT.
pub(crate) const ENV_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Usage the source cube must have.
pub(crate) const SOURCE_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING
    .union(wgpu::TextureUsages::STORAGE_BINDING)
    .union(wgpu::TextureUsages::COPY_SRC);
/// GGX importance samples per prefiltered texel (IBL value).
pub(crate) const PREFILTER_SAMPLES: u32 = 1024;
/// Number of staging buffers of the readback ring.
pub(crate) const STAGING_RING: usize = 3;
/// SH coefficients: 9 x vec4<f32> (EnvSh.sh layout).
pub(crate) const SH_BYTES: u64 = 9 * 16;

/// Workgroup edge of downsample, prefilter and sh_partial (`@workgroup_size(8, 8, 1)`).
const WORKGROUP: u32 = 8;
/// Source mip the SH projection reads (64 px faces).
const SH_MIP: u32 = 2;
/// sh_partial workgroups: (64 / 8) x (64 / 8) x 6 faces; one 144 B record each.
const SH_PARTIALS: u64 =
    ((ENV_SIZE >> SH_MIP) / WORKGROUP * ((ENV_SIZE >> SH_MIP) / WORKGROUP) * 6) as u64;
/// Filtered mips 1..5 (downsample and prefilter).
const FILTERED: usize = (ENV_MIPS - 1) as usize;

pub(crate) const DOWNSAMPLE_WGSL: &str = include_str!("wgsl/downsample.wgsl");
pub(crate) const PREFILTER_WGSL: &str = include_str!("wgsl/prefilter.wgsl");
pub(crate) const SH_PROJECT_WGSL: &str = include_str!("wgsl/sh_project.wgsl");

/// SH9 irradiance coefficients (rgb, w = 0), the shape `LightingResources::set_env` takes.
pub(crate) type Sh9 = [[f32; 4]; 9];

/// State of one staging buffer of the ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Free,
    /// A copy with this sequence number was recorded; mapped by the next `after_submit`.
    Written(u64),
    /// map_async is pending for this sequence number.
    Mapping(u64),
}

/// Device-free ring and sequence bookkeeping of the readback (`T` = the read value).
#[derive(Debug)]
pub(crate) struct Readback<T> {
    slots: [Slot; STAGING_RING],
    /// Next slot tried by `reserve` (round robin).
    cursor: usize,
    /// Sequence number of the next reservation (starts at 1).
    next_seq: u64,
    /// Sequence number of the newest stored result (0 = none yet).
    applied_seq: u64,
    newest: Option<T>,
    retry: bool,
}

impl<T> Readback<T> {
    pub(crate) fn new() -> Readback<T> {
        Readback {
            slots: [Slot::Free; STAGING_RING],
            cursor: 0,
            next_seq: 1,
            applied_seq: 0,
            newest: None,
            retry: false,
        }
    }

    /// Reserves the next free slot for a copy and gives it the next sequence number. With
    /// every slot busy it returns None and sets the retry flag.
    pub(crate) fn reserve(&mut self) -> Option<usize> {
        for k in 0..STAGING_RING {
            let i = (self.cursor + k) % STAGING_RING;
            if self.slots[i] == Slot::Free {
                self.slots[i] = Slot::Written(self.next_seq);
                self.next_seq += 1;
                self.cursor = (i + 1) % STAGING_RING;
                return Some(i);
            }
        }
        self.retry = true;
        None
    }

    /// Marks every slot written since the last call as mapping; entry i is the sequence
    /// number to map slot i with.
    pub(crate) fn start_mapping(&mut self) -> [Option<u64>; STAGING_RING] {
        let mut out = [None; STAGING_RING];
        for (slot, o) in self.slots.iter_mut().zip(out.iter_mut()) {
            if let Slot::Written(seq) = *slot {
                *slot = Slot::Mapping(seq);
                *o = Some(seq);
            }
        }
        out
    }

    /// Map callback of `slot` for `seq`: frees the slot and stores `value` (None = map
    /// failed) unless a newer result was already stored.
    pub(crate) fn complete(&mut self, slot: usize, seq: u64, value: Option<T>) {
        if let Some(s) = self.slots.get_mut(slot)
            && *s == Slot::Mapping(seq)
        {
            *s = Slot::Free;
        }
        if let Some(v) = value
            && seq > self.applied_seq
        {
            self.applied_seq = seq;
            self.newest = Some(v);
        }
    }

    /// The newest completed result, once.
    pub(crate) fn take(&mut self) -> Option<T> {
        self.newest.take()
    }

    /// Whether a copy was skipped since the last call (all staging buffers busy).
    pub(crate) fn take_retry(&mut self) -> bool {
        std::mem::take(&mut self.retry)
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Decodes 9 little-endian vec4<f32>.
fn decode_sh(bytes: &[u8]) -> Option<Sh9> {
    if bytes.len() < SH_BYTES as usize {
        return None;
    }
    let mut sh = [[0.0; 4]; 9];
    for (c, chunk) in sh.iter_mut().flatten().zip(bytes.chunks_exact(4)) {
        *c = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Some(sh)
}

/// Bind groups over one source cube (cached for the last source texture).
struct SourceBinds {
    /// The source texture (wgpu resources compare by identity).
    texture: wgpu::Texture,
    /// downsample[k - 1]: source mip k - 1 read, source mip k storage (k = 1..5).
    downsample: [wgpu::BindGroup; FILTERED],
    /// prefilter[m - 1]: whole source cube, sampler, output mip m storage, params of m.
    prefilter: [wgpu::BindGroup; FILTERED],
    /// Source mip 2, partial sums, coefficients.
    sh: wgpu::BindGroup,
}

/// The IBL filter: pipelines, output cube and readback ring, built once per GpuContext.
pub(crate) struct Filter {
    downsample: wgpu::ComputePipeline,
    prefilter: wgpu::ComputePipeline,
    sh_partial: wgpu::ComputePipeline,
    sh_final: wgpu::ComputePipeline,
    downsample_layout: wgpu::BindGroupLayout,
    prefilter_layout: wgpu::BindGroupLayout,
    sh_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// Output cube (SKY_ENV, TEXTURE_BINDING | STORAGE_BINDING | COPY_DST).
    output: wgpu::Texture,
    /// Cube view of every output mip (S10 binding 3).
    output_view: wgpu::TextureView,
    /// D2Array storage views of output mips 1..5.
    output_mips: [wgpu::TextureView; FILTERED],
    /// PrefilterParams of mips 1..5 (constant, written at creation).
    params: [wgpu::Buffer; FILTERED],
    /// One ShPartial record per sh_partial workgroup.
    partials: wgpu::Buffer,
    /// The 9 coefficients written by sh_final (STORAGE | COPY_SRC).
    coefficients: wgpu::Buffer,
    /// MAP_READ | COPY_DST, SH_BYTES each.
    staging: [wgpu::Buffer; STAGING_RING],
    readback: Arc<Mutex<Readback<Sh9>>>,
    source: Option<SourceBinds>,
    /// `ibl.invalid_source` refusal of the last `record`, None after a valid one.
    invalid_source: Option<RaaError>,
}

fn texture_entry(
    binding: u32,
    dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: dimension,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: ENV_FORMAT,
            view_dimension: wgpu::TextureViewDimension::D2Array,
        },
        count: None,
    }
}

fn buffer_entry(
    binding: u32,
    ty: wgpu::BufferBindingType,
    size: u64,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        count: None,
    }
}

fn storage_rw() -> wgpu::BufferBindingType {
    wgpu::BufferBindingType::Storage { read_only: false }
}

fn layout(
    gpu: &GpuContext,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    gpu.device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries,
        })
}

fn compute_pipeline(
    gpu: &GpuContext,
    label: &str,
    bind_layout: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
    entry: &str,
) -> wgpu::ComputePipeline {
    let device = gpu.device();
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(bind_layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn shader(gpu: &GpuContext, label: &str, source: &str) -> wgpu::ShaderModule {
    gpu.device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
}

/// D2Array view of one mip of a cube texture (the 6 faces as layers).
fn mip_view(texture: &wgpu::Texture, mip: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("raa-ibl.filter.mip"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        base_mip_level: mip,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(6),
        ..Default::default()
    })
}

/// Cube view of every mip.
fn cube_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("raa-ibl.filter.cube"),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    })
}

/// Workgroups along one edge of a `256 >> mip` px face.
fn groups(mip: u32) -> u32 {
    (ENV_SIZE >> mip).div_ceil(WORKGROUP)
}

/// Why `texture` is not a valid source cube (None when it is).
fn source_mismatch(texture: &wgpu::Texture) -> Option<String> {
    let size = texture.size();
    let ok = texture.dimension() == wgpu::TextureDimension::D2
        && (size.width, size.height, size.depth_or_array_layers) == (ENV_SIZE, ENV_SIZE, 6)
        && texture.format() == ENV_FORMAT
        && texture.mip_level_count() == ENV_MIPS
        && texture.sample_count() == 1
        && texture.usage().contains(SOURCE_USAGE);
    (!ok).then(|| {
        format!(
            "the source cube must be {ENV_SIZE}x{ENV_SIZE}x6 {ENV_FORMAT:?} with {ENV_MIPS} mips, \
             1 sample and usage {SOURCE_USAGE:?}; got {}x{}x{} {:?} {:?} with {} mips, {} \
             samples and usage {:?}",
            size.width,
            size.height,
            size.depth_or_array_layers,
            texture.dimension(),
            texture.format(),
            texture.mip_level_count(),
            texture.sample_count(),
            texture.usage()
        )
    })
}

impl Filter {
    /// Creates the pipelines, the output cube and every buffer of the filter.
    pub(crate) fn new(gpu: &GpuContext) -> Filter {
        use wgpu::TextureViewDimension as Dim;
        let device = gpu.device();
        let downsample_layout = layout(
            gpu,
            "raa-ibl.filter.downsample",
            &[texture_entry(0, Dim::D2Array), storage_texture_entry(1)],
        );
        let prefilter_layout = layout(
            gpu,
            "raa-ibl.filter.prefilter",
            &[
                texture_entry(0, Dim::Cube),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                storage_texture_entry(2),
                buffer_entry(3, wgpu::BufferBindingType::Uniform, 16),
            ],
        );
        let sh_layout = layout(
            gpu,
            "raa-ibl.filter.sh",
            &[
                texture_entry(0, Dim::D2Array),
                buffer_entry(1, storage_rw(), SH_PARTIALS * SH_BYTES),
                buffer_entry(2, storage_rw(), SH_BYTES),
            ],
        );
        let downsample_module = shader(gpu, "raa-ibl.filter.downsample", DOWNSAMPLE_WGSL);
        let prefilter_module = shader(gpu, "raa-ibl.filter.prefilter", PREFILTER_WGSL);
        let sh_module = shader(gpu, "raa-ibl.filter.sh", SH_PROJECT_WGSL);
        let downsample = compute_pipeline(
            gpu,
            "raa-ibl.filter.downsample",
            &downsample_layout,
            &downsample_module,
            "downsample",
        );
        let prefilter = compute_pipeline(
            gpu,
            "raa-ibl.filter.prefilter",
            &prefilter_layout,
            &prefilter_module,
            "prefilter",
        );
        let sh_partial = compute_pipeline(
            gpu,
            "raa-ibl.filter.sh_partial",
            &sh_layout,
            &sh_module,
            "sh_partial",
        );
        let sh_final = compute_pipeline(
            gpu,
            "raa-ibl.filter.sh_final",
            &sh_layout,
            &sh_module,
            "sh_final",
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raa-ibl.filter.linear_clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let output = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("raa-ibl.filter.output"),
            size: wgpu::Extent3d {
                width: ENV_SIZE,
                height: ENV_SIZE,
                depth_or_array_layers: 6,
            },
            mip_level_count: ENV_MIPS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ENV_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let output_view = cube_view(&output);
        let output_mips = std::array::from_fn(|i| mip_view(&output, i as u32 + 1));
        let params = std::array::from_fn(|i| {
            let mip = i as u32 + 1;
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("raa-ibl.filter.prefilter_params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let roughness = mip as f32 / (ENV_MIPS - 1) as f32;
            let mut bytes = [0u8; 16];
            bytes[0..4].copy_from_slice(&mip.to_le_bytes());
            bytes[4..8].copy_from_slice(&roughness.to_le_bytes());
            bytes[8..12].copy_from_slice(&(ENV_SIZE >> mip).to_le_bytes());
            bytes[12..16].copy_from_slice(&PREFILTER_SAMPLES.to_le_bytes());
            gpu.queue().write_buffer(&buffer, 0, &bytes);
            buffer
        });
        let partials = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-ibl.filter.sh_partials"),
            size: SH_PARTIALS * SH_BYTES,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let coefficients = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-ibl.filter.sh"),
            size: SH_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = std::array::from_fn(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("raa-ibl.filter.sh_staging"),
                size: SH_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        Filter {
            downsample,
            prefilter,
            sh_partial,
            sh_final,
            downsample_layout,
            prefilter_layout,
            sh_layout,
            sampler,
            output,
            output_view,
            output_mips,
            params,
            partials,
            coefficients,
            staging,
            readback: Arc::new(Mutex::new(Readback::new())),
            source: None,
            invalid_source: None,
        }
    }

    /// The output cube texture.
    pub(crate) fn output(&self) -> &wgpu::Texture {
        &self.output
    }

    /// Cube view of the output (every mip), for `LightingResources::set_env`.
    pub(crate) fn output_view(&self) -> &wgpu::TextureView {
        &self.output_view
    }

    /// The `ibl.invalid_source` refusal of the last `record` (None after a valid source).
    pub(crate) fn invalid_source(&self) -> Option<&RaaError> {
        self.invalid_source.as_ref()
    }

    /// Caches the bind groups over `source`, rebuilt only when the source texture changes.
    fn ensure_source(&mut self, gpu: &GpuContext, source: &wgpu::Texture) {
        if self.source.as_ref().is_some_and(|s| s.texture == *source) {
            return;
        }
        let device = gpu.device();
        let downsample = std::array::from_fn(|i| {
            let k = i as u32 + 1;
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raa-ibl.filter.downsample"),
                layout: &self.downsample_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&mip_view(source, k - 1)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&mip_view(source, k)),
                    },
                ],
            })
        });
        let cube = cube_view(source);
        let prefilter = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raa-ibl.filter.prefilter"),
                layout: &self.prefilter_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&cube),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.output_mips[i]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.params[i].as_entire_binding(),
                    },
                ],
            })
        });
        let sh = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-ibl.filter.sh"),
            layout: &self.sh_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&mip_view(source, SH_MIP)),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.partials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.coefficients.as_entire_binding(),
                },
            ],
        });
        self.source = Some(SourceBinds {
            texture: source.clone(),
            downsample,
            prefilter,
            sh,
        });
    }

    /// Records the whole filter of `source` into `encoder`: source mips 1..5 by downsample,
    /// source mip 0 copied to the output, output mips 1..5 by GGX prefilter, sh_partial and
    /// sh_final, then the coefficients copied into the next free staging buffer (none free:
    /// the retry flag is set). Invariant: a source that is not SKY_ENV with SOURCE_USAGE is
    /// skipped (nothing recorded) and leaves `ibl.invalid_source` in `invalid_source()`.
    pub(crate) fn record(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Texture,
    ) {
        if let Some(message) = source_mismatch(source) {
            self.invalid_source = Some(RaaError::FrameGraph {
                code: "ibl.invalid_source",
                message,
            });
            return;
        }
        self.invalid_source = None;
        self.ensure_source(gpu, source);
        // Invariant: ensure_source just set it.
        let Some(binds) = self.source.as_ref() else {
            return;
        };
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa-ibl.filter.downsample"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.downsample);
            for (k, bind_group) in (1..ENV_MIPS).zip(&binds.downsample) {
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(groups(k), groups(k), 6);
            }
        }
        encoder.copy_texture_to_texture(
            source.as_image_copy(),
            self.output.as_image_copy(),
            wgpu::Extent3d {
                width: ENV_SIZE,
                height: ENV_SIZE,
                depth_or_array_layers: 6,
            },
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa-ibl.filter.prefilter_sh"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.prefilter);
            for (m, bind_group) in (1..ENV_MIPS).zip(&binds.prefilter) {
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(groups(m), groups(m), 6);
            }
            pass.set_bind_group(0, &binds.sh, &[]);
            pass.set_pipeline(&self.sh_partial);
            pass.dispatch_workgroups(groups(SH_MIP), groups(SH_MIP), 6);
            pass.set_pipeline(&self.sh_final);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let reserved = lock(&self.readback).reserve();
        if let Some(staging) = reserved.and_then(|i| self.staging.get(i)) {
            encoder.copy_buffer_to_buffer(&self.coefficients, 0, staging, 0, SH_BYTES);
        }
    }

    /// Call after submitting the encoder of `record`: maps the staging buffers written
    /// since the last call. Each map callback stores the coefficients with their sequence
    /// number (older than the newest stored: dropped) and unmaps. On wasm the browser drives
    /// the callbacks; native callers poll the device.
    pub(crate) fn after_submit(&mut self) {
        let mapping = lock(&self.readback).start_mapping();
        for (slot, (seq, buffer)) in mapping.into_iter().zip(&self.staging).enumerate() {
            let Some(seq) = seq else {
                continue;
            };
            let readback = Arc::clone(&self.readback);
            let mapped = buffer.clone();
            buffer.map_async(wgpu::MapMode::Read, .., move |result| {
                let value = result.ok().and_then(|()| {
                    mapped
                        .get_mapped_range(..)
                        .ok()
                        .and_then(|view| decode_sh(&view))
                });
                // Unmapping an unmapped buffer (failed map) is a no-op.
                mapped.unmap();
                lock(&readback).complete(slot, seq, value);
            });
        }
    }

    /// The newest completed SH9, once.
    pub(crate) fn take_sh(&self) -> Option<Sh9> {
        lock(&self.readback).take()
    }

    /// Whether a readback was skipped since the last call (all staging buffers busy), so
    /// the caller filters again.
    pub(crate) fn take_retry(&self) -> bool {
        lock(&self.readback).take_retry()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readback_reserves_round_robin_and_sets_retry_when_full() {
        let mut r = Readback::<u32>::new();
        assert_eq!(r.reserve(), Some(0));
        assert_eq!(r.reserve(), Some(1));
        assert_eq!(r.reserve(), Some(2));
        assert!(!r.take_retry());
        assert_eq!(r.reserve(), None);
        assert!(r.take_retry());
        assert!(!r.take_retry(), "take_retry clears the flag");
        assert_eq!(
            r.start_mapping(),
            [Some(1), Some(2), Some(3)],
            "slots carry increasing sequence numbers"
        );
        assert_eq!(r.start_mapping(), [None; STAGING_RING], "mapped once");
        assert_eq!(r.reserve(), None, "mapping slots stay busy");
        r.complete(1, 2, Some(20));
        assert_eq!(r.reserve(), Some(1), "a completed slot is free again");
        assert_eq!(r.start_mapping(), [None, Some(4), None]);
    }

    #[test]
    fn readback_keeps_only_the_newest_result_and_gives_it_once() {
        let mut r = Readback::<u32>::new();
        for _ in 0..3 {
            r.reserve();
        }
        r.start_mapping();
        assert_eq!(r.take(), None);
        r.complete(1, 2, Some(20));
        r.complete(0, 1, Some(10));
        assert_eq!(r.take(), Some(20), "the older result 1 is dropped");
        assert_eq!(r.take(), None, "given once");
        r.complete(2, 3, None);
        assert_eq!(r.take(), None, "a failed map stores nothing");
        assert_eq!(r.start_mapping(), [None; STAGING_RING]);
        assert_eq!(r.reserve(), Some(0), "the failed slot is free again");
        r.start_mapping();
        r.complete(0, 4, Some(40));
        assert_eq!(r.take(), Some(40));
    }

    #[test]
    fn decode_sh_reads_nine_little_endian_vec4() {
        let mut bytes = [0u8; SH_BYTES as usize];
        for (i, chunk) in bytes.chunks_exact_mut(4).enumerate() {
            chunk.copy_from_slice(&(i as f32).to_le_bytes());
        }
        let sh = decode_sh(&bytes).expect("144 bytes decode");
        assert_eq!(sh[0], [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(sh[8], [32.0, 33.0, 34.0, 35.0]);
        assert_eq!(decode_sh(&bytes[..140]), None);
    }

    #[test]
    fn sh_partials_cover_mip_two() {
        assert_eq!(SH_PARTIALS, 8 * 8 * 6);
        assert_eq!(groups(SH_MIP), 8);
        assert_eq!(groups(ENV_MIPS - 1), 1);
    }

    /// Source cube filled with a constant: the SH reads back as that colour (L00 =
    /// c x 0.282095 x 4 pi, the others ~0), and an invalid source is refused. The GPU
    /// context is leaked like in lut.rs.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn filter_constant_cube_reads_back_constant_sh() {
        let Some(gpu) = raa_core::testing::software_gpu() else {
            raa_core::testing::skip(
                "filter_constant_cube_reads_back_constant_sh",
                "software_gpu returned None; see the software_gpu line above",
            );
            return;
        };
        let scope = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let mut filter = Filter::new(&gpu);
        let source = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("test.source"),
            size: wgpu::Extent3d {
                width: ENV_SIZE,
                height: ENV_SIZE,
                depth_or_array_layers: 6,
            },
            mip_level_count: ENV_MIPS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ENV_FORMAT,
            usage: SOURCE_USAGE | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // f16 (1.0, 0.5, 0.25, 1.0).
        let texel: [u16; 4] = [0x3C00, 0x3800, 0x3400, 0x3C00];
        let face: Vec<u8> = (0..ENV_SIZE * ENV_SIZE)
            .flat_map(|_| texel.iter().flat_map(|h| h.to_le_bytes()))
            .collect();
        for layer in 0..6 {
            gpu.queue().write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &source,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &face,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(ENV_SIZE * 8),
                    rows_per_image: Some(ENV_SIZE),
                },
                wgpu::Extent3d {
                    width: ENV_SIZE,
                    height: ENV_SIZE,
                    depth_or_array_layers: 1,
                },
            );
        }
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        filter.record(&gpu, &mut encoder, &source);
        assert!(filter.invalid_source().is_none());
        gpu.queue().submit([encoder.finish()]);
        filter.after_submit();
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        let error = raa_core::testing::block_on(scope.pop());
        assert!(error.is_none(), "validation error: {error:?}");
        let sh = filter.take_sh().expect("SH read back after the poll");
        assert!(filter.take_sh().is_none(), "given once");
        assert!(!filter.take_retry());
        let l00 = 0.282_095 * 4.0 * std::f32::consts::PI;
        for (c, want) in [1.0, 0.5, 0.25].into_iter().enumerate() {
            assert!(
                (sh[0][c] - want * l00).abs() <= 0.01 * want * l00,
                "L00[{c}] = {}, want {}",
                sh[0][c],
                want * l00
            );
            for (k, coef) in sh.iter().enumerate().skip(1) {
                assert!(coef[c].abs() <= 1e-3, "L[{k}][{c}] = {}", coef[c]);
            }
        }

        let small = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("test.small"),
            size: wgpu::Extent3d {
                width: 128,
                height: 128,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ENV_FORMAT,
            usage: SOURCE_USAGE,
            view_formats: &[],
        });
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        filter.record(&gpu, &mut encoder, &small);
        let refusal = filter.invalid_source().expect("small cube refused");
        assert_eq!(refusal.code(), "ibl.invalid_source");
        assert!(refusal.message().contains("128x128x6"), "{refusal}");
        drop(filter);
        std::mem::forget(gpu);
    }
}
