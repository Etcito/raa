//! GPU filtering of the sky cube (IBL value): source mips 1..5 by box downsample
//! (`wgsl/downsample.wgsl`), mip 0 copied into the IBL-owned output cube, output mips 1..5
//! by GGX importance sampling with PDF-based source mip (`wgsl/prefilter.wgsl`), the SH9
//! projection by compute reduction (`wgsl/sh_project.wgsl`), then the 9 coefficients copied
//! into a ring of 3 staging buffers for asynchronous readback.
//!
//! Everything is allocated by `Filter::new`; a `record` only creates bind groups when the
//! source texture changes (cached for the last source). Readback: a successful `record`
//! reserves a free staging buffer and tags it with its publication generation and the
//! current invalidation epoch, copies into it (or, with every buffer busy, sets the retry
//! flag), the caller submits, then `after_submit` maps it. The map callback first matches
//! and frees its original slot/sequence, then reports the tagged value to the
//! [`PublicationGate`], which publishes only current-epoch, non-superseded completions. On
//! wasm the browser drives map callbacks; native callers poll the device (`Device::poll`).
//! The shared slot is an `Arc<Mutex<..>>` rather than an `Rc<RefCell<..>>` because wgpu
//! requires a `Send` map callback on native; the gate is separate so invalidation never
//! needs the `RefCell`-borrowed filter state.

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

/// Identity of one filter request: the publication generation it obtained and the
/// invalidation epoch that was current when it was recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RequestTag {
    pub(crate) generation: u64,
    pub(crate) epoch: u64,
}

/// State of one staging buffer of the ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Free,
    /// A copy with this sequence number was recorded; mapped by the next `after_submit`.
    Written(u64),
    /// map_async is pending for this sequence number.
    Mapping(u64),
}

/// Result of one map callback. The slot was matched and freed before this is built; the
/// gate decides whether the tagged request may still publish.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Completion<T> {
    /// The slot no longer holds this request (it was reused): nothing was freed.
    Stale,
    /// The request's copy was unmapped and freed but its map/decode failed.
    Failed(RequestTag),
    /// The request's copy was unmapped, freed and is newer than every accepted one.
    Accepted { tag: RequestTag, value: T },
    /// The request's copy was unmapped and freed but a newer accepted one exists.
    Superseded(RequestTag),
}

/// Device-free ring, tag and sequence bookkeeping of the SH9 readback.
#[derive(Debug)]
pub(crate) struct Readback {
    slots: [Slot; STAGING_RING],
    /// Tag of the request occupying each slot (valid while the slot is not free).
    tags: [RequestTag; STAGING_RING],
    /// Next slot tried by `reserve` (round robin).
    cursor: usize,
    /// Sequence number of the next reservation (starts at 1).
    next_seq: u64,
    /// Sequence number of the newest accepted completion (0 = none yet).
    applied_seq: u64,
    retry: bool,
}

impl Readback {
    pub(crate) fn new() -> Readback {
        Readback {
            slots: [Slot::Free; STAGING_RING],
            tags: [RequestTag {
                generation: 0,
                epoch: 0,
            }; STAGING_RING],
            cursor: 0,
            next_seq: 1,
            applied_seq: 0,
            retry: false,
        }
    }

    /// Whether at least one staging buffer can be reserved.
    pub(crate) fn has_free(&self) -> bool {
        self.slots.contains(&Slot::Free)
    }

    /// Requests another filter because every staging buffer is busy.
    pub(crate) fn set_retry(&mut self) {
        self.retry = true;
    }

    /// Reserves the next free slot for a copy tagged `tag` and gives it the next sequence
    /// number. With every slot busy it returns None and sets the retry flag.
    pub(crate) fn reserve(&mut self, tag: RequestTag) -> Option<(usize, u64)> {
        for k in 0..STAGING_RING {
            let i = (self.cursor + k) % STAGING_RING;
            if self.slots[i] == Slot::Free {
                self.slots[i] = Slot::Written(self.next_seq);
                self.tags[i] = tag;
                let seq = self.next_seq;
                self.next_seq += 1;
                self.cursor = (i + 1) % STAGING_RING;
                return Some((i, seq));
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

    /// Map callback of `slot` for `seq`: first matches and frees that exact slot/sequence,
    /// then classifies `value` (None = map/decode failed). A callback whose sequence no
    /// longer occupies its slot frees nothing (`Stale`).
    pub(crate) fn complete(
        &mut self,
        slot: usize,
        seq: u64,
        value: Option<Sh9>,
    ) -> Completion<Sh9> {
        let tag = match self.slots.get(slot) {
            Some(Slot::Mapping(mapped)) if *mapped == seq => {
                self.slots[slot] = Slot::Free;
                self.tags[slot]
            }
            _ => return Completion::Stale,
        };
        match value {
            Some(value) if seq > self.applied_seq => {
                self.applied_seq = seq;
                Completion::Accepted { tag, value }
            }
            Some(_) => Completion::Superseded(tag),
            None => Completion::Failed(tag),
        }
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

/// What the publication getters observe without advancing any work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PublicationSnapshot {
    pub(crate) requested_generation: u64,
    pub(crate) published_generation: Option<u64>,
    pub(crate) pending: bool,
}

#[derive(Debug)]
struct GateState {
    /// Invalidation epoch of the current environment; every cancellation bumps it.
    epoch: u64,
    /// Monotonic count of successfully recorded filter requests.
    requested_generation: u64,
    published_generation: Option<u64>,
    /// The SH9 last given to the lighting, None while the environment is cleared.
    published_sh: Option<Sh9>,
    /// Current-epoch requests whose readback has not completed yet.
    outstanding: usize,
    /// A completion accepted by the readback, waiting for `apply_pending`.
    ready: Option<(u64, Sh9)>,
    /// Attributable `ibl.readback` failure, cleared by publication or invalidation.
    error: Option<RaaError>,
}

/// Cancellation and publication gate shared with the map callbacks. It owns its
/// `Arc<Mutex<..>>` independently of the `RefCell`-borrowed filter state, so
/// `IblHandle::invalidate` always cancels in-flight work and every callback checks the
/// current epoch before it can publish.
#[derive(Clone, Debug)]
pub(crate) struct PublicationGate {
    inner: Arc<Mutex<GateState>>,
}

impl PublicationGate {
    pub(crate) fn new() -> PublicationGate {
        PublicationGate {
            inner: Arc::new(Mutex::new(GateState {
                epoch: 0,
                requested_generation: 0,
                published_generation: None,
                published_sh: None,
                outstanding: 0,
                ready: None,
                error: None,
            })),
        }
    }

    /// Obtains the generation of a successfully recorded filter request in the current
    /// epoch. The request stays pending until its completion is published.
    pub(crate) fn begin_request(&self) -> RequestTag {
        let mut state = lock(&self.inner);
        state.requested_generation = state.requested_generation.saturating_add(1);
        state.outstanding = state.outstanding.saturating_add(1);
        RequestTag {
            generation: state.requested_generation,
            epoch: state.epoch,
        }
    }

    /// Drops a request whose staging reservation failed, without a publication.
    pub(crate) fn abandon(&self, tag: RequestTag) {
        let mut state = lock(&self.inner);
        if state.epoch == tag.epoch {
            state.outstanding = state.outstanding.saturating_sub(1);
        }
    }

    /// Records a readback completion. The readback already matched, unmapped and freed the
    /// slot; this publishes only when the request epoch is still current and the readback
    /// accepted the value as newer than every earlier completion.
    pub(crate) fn complete(&self, completion: Completion<Sh9>) {
        let mut state = lock(&self.inner);
        match completion {
            Completion::Stale => {}
            Completion::Superseded(tag) => {
                if state.epoch == tag.epoch {
                    state.outstanding = state.outstanding.saturating_sub(1);
                }
            }
            Completion::Failed(tag) => {
                if state.epoch == tag.epoch {
                    state.outstanding = state.outstanding.saturating_sub(1);
                    state.error = Some(RaaError::FrameGraph {
                        code: "ibl.readback",
                        message: format!(
                            "the environment SH9 readback of requested generation {} failed to map",
                            tag.generation
                        ),
                    });
                }
            }
            Completion::Accepted { tag, value } => {
                if state.epoch == tag.epoch {
                    state.outstanding = state.outstanding.saturating_sub(1);
                    let superseded = state
                        .ready
                        .is_some_and(|(ready, _)| ready >= tag.generation);
                    if !superseded {
                        state.ready = Some((tag.generation, value));
                    }
                }
            }
        }
    }

    /// Cancels every request of the current epoch and clears the published environment and
    /// its error. The requested-generation counter is preserved.
    pub(crate) fn invalidate(&self) {
        let mut state = lock(&self.inner);
        state.epoch = state.epoch.saturating_add(1);
        state.published_generation = None;
        state.published_sh = None;
        state.outstanding = 0;
        state.ready = None;
        state.error = None;
    }

    /// Publishes the completed accepted environment once and records its generation.
    pub(crate) fn take_publication(&self) -> Option<(u64, Sh9)> {
        let mut state = lock(&self.inner);
        let (generation, sh) = state.ready.take()?;
        state.published_generation = Some(generation);
        state.published_sh = Some(sh);
        state.error = None;
        Some((generation, sh))
    }

    /// The SH9 last given to the lighting, None while the environment is cleared.
    pub(crate) fn published_sh(&self) -> Option<Sh9> {
        lock(&self.inner).published_sh
    }

    /// The attributable readback failure of the current epoch.
    pub(crate) fn error(&self) -> Option<RaaError> {
        lock(&self.inner).error.clone()
    }

    /// A consistent copy of the lifecycle observations.
    pub(crate) fn snapshot(&self) -> PublicationSnapshot {
        let state = lock(&self.inner);
        PublicationSnapshot {
            requested_generation: state.requested_generation,
            published_generation: state.published_generation,
            pending: state.outstanding > 0 || state.ready.is_some(),
        }
    }
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
    readback: Arc<Mutex<Readback>>,
    source: Option<SourceBinds>,
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
        }
    }

    /// Cube view of the output (every mip), for `LightingResources::set_env`.
    pub(crate) fn output_view(&self) -> &wgpu::TextureView {
        &self.output_view
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
    /// sh_final, then the coefficients copied into the next free staging buffer tagged with
    /// the next generation of `gate` (none free: the retry flag is set and no generation is
    /// invented). A source that is not SKY_ENV with SOURCE_USAGE records nothing, cancels
    /// `gate` and returns `ibl.invalid_source`.
    pub(crate) fn record(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Texture,
        gate: &PublicationGate,
    ) -> Result<(), RaaError> {
        if let Some(message) = source_mismatch(source) {
            let error = RaaError::FrameGraph {
                code: "ibl.invalid_source",
                message,
            };
            gate.invalidate();
            return Err(error);
        }
        self.ensure_source(gpu, source);
        // Invariant: ensure_source just set it.
        let Some(binds) = self.source.as_ref() else {
            return Ok(());
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
        let mut readback = lock(&self.readback);
        if !readback.has_free() {
            readback.set_retry();
            return Ok(());
        }
        let tag = gate.begin_request();
        let Some((slot, _seq)) = readback.reserve(tag) else {
            drop(readback);
            gate.abandon(tag);
            return Ok(());
        };
        drop(readback);
        if let Some(staging) = self.staging.get(slot) {
            encoder.copy_buffer_to_buffer(&self.coefficients, 0, staging, 0, SH_BYTES);
        }
        Ok(())
    }

    /// Call after submitting the encoder of `record`: maps the staging buffers written
    /// since the last call. Each map callback first matches and frees its slot/sequence,
    /// then reports the decoded coefficients to `gate`, which publishes only a current-epoch
    /// completion newer than every earlier one. On wasm the browser drives the callbacks;
    /// native callers poll the device.
    pub(crate) fn after_submit(&self, gate: &PublicationGate) {
        let mapping = lock(&self.readback).start_mapping();
        for (slot, (seq, buffer)) in mapping.into_iter().zip(&self.staging).enumerate() {
            let Some(seq) = seq else {
                continue;
            };
            let readback = Arc::clone(&self.readback);
            let callback_gate = gate.clone();
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
                let completion = lock(&readback).complete(slot, seq, value);
                callback_gate.complete(completion);
            });
        }
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

    fn tag(generation: u64) -> RequestTag {
        RequestTag {
            generation,
            epoch: 0,
        }
    }

    fn uniform_sh(value: f32) -> Sh9 {
        let mut sh = [[0.0; 4]; 9];
        sh[0] = [value, value, value, 0.0];
        sh
    }

    fn snapshot(requested: u64, published: Option<u64>, pending: bool) -> PublicationSnapshot {
        PublicationSnapshot {
            requested_generation: requested,
            published_generation: published,
            pending,
        }
    }

    /// Drives `Readback` and `PublicationGate` like `Filter::record` and its map callbacks.
    struct Harness {
        gate: PublicationGate,
        readback: Readback,
    }

    impl Harness {
        fn new() -> Harness {
            Harness {
                gate: PublicationGate::new(),
                readback: Readback::new(),
            }
        }

        /// The reservation and generation allocation of a successful `Filter::record`.
        fn request(&mut self) -> Option<(usize, u64, RequestTag)> {
            if !self.readback.has_free() {
                self.readback.set_retry();
                return None;
            }
            let tag = self.gate.begin_request();
            match self.readback.reserve(tag) {
                Some((slot, seq)) => Some((slot, seq, tag)),
                None => {
                    self.gate.abandon(tag);
                    None
                }
            }
        }

        fn mapping(&mut self) -> [Option<u64>; STAGING_RING] {
            self.readback.start_mapping()
        }

        /// One map callback: the readback frees the slot, then the gate decides.
        fn complete(&mut self, slot: usize, seq: u64, value: Option<Sh9>) {
            let completion = self.readback.complete(slot, seq, value);
            self.gate.complete(completion);
        }
    }

    #[test]
    fn readback_reserves_round_robin_and_sets_retry_when_full() {
        let mut r = Readback::new();
        assert_eq!(r.reserve(tag(1)), Some((0, 1)));
        assert_eq!(r.reserve(tag(2)), Some((1, 2)));
        assert_eq!(r.reserve(tag(3)), Some((2, 3)));
        assert!(!r.take_retry());
        assert_eq!(r.reserve(tag(4)), None);
        assert!(r.take_retry());
        assert!(!r.take_retry(), "take_retry clears the flag");
        assert_eq!(
            r.start_mapping(),
            [Some(1), Some(2), Some(3)],
            "slots carry increasing sequence numbers"
        );
        assert_eq!(r.start_mapping(), [None; STAGING_RING], "mapped once");
        assert_eq!(r.reserve(tag(4)), None, "mapping slots stay busy");
        assert_eq!(
            r.complete(1, 2, Some(uniform_sh(2.0))),
            Completion::Accepted {
                tag: tag(2),
                value: uniform_sh(2.0)
            }
        );
        assert_eq!(
            r.reserve(tag(4)),
            Some((1, 4)),
            "a completed slot is free again"
        );
        assert_eq!(r.start_mapping(), [None, Some(4), None]);
    }

    #[test]
    fn readback_keeps_only_the_newest_completion_and_frees_failed_slots() {
        let mut r = Readback::new();
        for generation in 1..=3 {
            r.reserve(tag(generation));
        }
        r.start_mapping();
        assert_eq!(
            r.complete(1, 2, Some(uniform_sh(2.0))),
            Completion::Accepted {
                tag: tag(2),
                value: uniform_sh(2.0)
            }
        );
        assert_eq!(
            r.complete(0, 1, Some(uniform_sh(1.0))),
            Completion::Superseded(tag(1)),
            "an older sequence is dropped"
        );
        assert_eq!(
            r.complete(2, 3, None),
            Completion::Failed(tag(3)),
            "a failed map frees the slot without a value"
        );
        assert_eq!(r.start_mapping(), [None; STAGING_RING]);
        assert_eq!(
            r.reserve(tag(4)),
            Some((0, 4)),
            "the failed slot is free again"
        );
    }

    #[test]
    fn initial_publication_is_empty_and_a_request_stays_pending_until_applied() {
        let mut h = Harness::new();
        assert_eq!(h.gate.snapshot(), snapshot(0, None, false));
        assert_eq!(h.gate.error(), None);
        assert_eq!(h.gate.published_sh(), None);

        let (slot, seq, _) = h.request().expect("one free slot");
        assert_eq!(h.gate.snapshot(), snapshot(1, None, true));
        assert_eq!(h.mapping(), [Some(seq), None, None]);
        h.complete(slot, seq, Some(uniform_sh(2.0)));
        assert_eq!(
            h.gate.snapshot(),
            snapshot(1, None, true),
            "a completed readback is not published by the getter"
        );
        let (generation, sh) = h.gate.take_publication().expect("publish once");
        assert_eq!(generation, 1);
        assert_eq!(sh, uniform_sh(2.0));
        assert_eq!(h.gate.snapshot(), snapshot(1, Some(1), false));
        assert_eq!(h.gate.published_sh(), Some(uniform_sh(2.0)));
        assert_eq!(h.gate.take_publication(), None, "published once");
    }

    #[test]
    fn failed_readback_is_attributable_and_distinct_from_pending_work() {
        let mut h = Harness::new();
        let (slot, seq, _) = h.request().expect("one free slot");
        h.mapping();
        h.complete(slot, seq, None);
        assert_eq!(h.gate.snapshot(), snapshot(1, None, false));
        let error = h.gate.error().expect("a readback error");
        assert_eq!(error.code(), "ibl.readback");
        assert!(
            error.message().contains("generation 1"),
            "attributable: {error}"
        );
        assert_eq!(h.gate.published_sh(), None);

        let (slot, seq, _) = h.request().expect("retry request");
        assert_eq!(h.gate.snapshot(), snapshot(2, None, true));
        assert_eq!(
            h.gate.error().map(|e| e.code()),
            Some("ibl.readback"),
            "the failure stays visible while the retry is pending"
        );
        h.mapping();
        h.complete(slot, seq, Some(uniform_sh(3.0)));
        h.gate.take_publication().expect("publication");
        assert_eq!(h.gate.snapshot(), snapshot(2, Some(2), false));
        assert_eq!(h.gate.error(), None, "publication clears the error");
    }

    #[test]
    fn earlier_completion_publishes_while_a_newer_request_is_outstanding() {
        let mut h = Harness::new();
        let (slot1, seq1, _) = h.request().expect("first slot");
        let (slot2, seq2, _) = h.request().expect("second slot");
        h.mapping();
        h.complete(slot1, seq1, Some(uniform_sh(1.0)));
        let (generation, _) = h.gate.take_publication().expect("earlier publish");
        assert_eq!(generation, 1);
        assert_eq!(h.gate.snapshot(), snapshot(2, Some(1), true));
        h.complete(slot2, seq2, Some(uniform_sh(2.0)));
        let (generation, _) = h.gate.take_publication().expect("newer publish");
        assert_eq!(generation, 2);
        assert_eq!(h.gate.snapshot(), snapshot(2, Some(2), false));
    }

    #[test]
    fn superseded_completions_never_downgrade_the_publication() {
        let mut h = Harness::new();
        let (slot1, seq1, _) = h.request().expect("first slot");
        let (slot2, seq2, _) = h.request().expect("second slot");
        h.mapping();
        h.complete(slot2, seq2, Some(uniform_sh(2.0)));
        h.gate.take_publication().expect("publish the newer value");
        h.complete(slot1, seq1, Some(uniform_sh(1.0)));
        assert_eq!(
            h.gate.published_sh(),
            Some(uniform_sh(2.0)),
            "the older completion cannot downgrade the environment"
        );
        assert_eq!(h.gate.snapshot(), snapshot(2, Some(2), false));
    }

    #[test]
    fn invalidation_clears_publication_and_keeps_the_request_counter() {
        let mut h = Harness::new();
        let (slot, seq, _) = h.request().expect("one free slot");
        h.mapping();
        h.complete(slot, seq, Some(uniform_sh(2.0)));
        h.gate.take_publication().expect("publish");
        assert_eq!(h.gate.snapshot(), snapshot(1, Some(1), false));

        h.gate.invalidate();
        assert_eq!(
            h.gate.snapshot(),
            snapshot(1, None, false),
            "requested generation survives the invalidation"
        );
        assert_eq!(h.gate.published_sh(), None);
        assert_eq!(h.gate.error(), None);

        let (_, _, request) = h.request().expect("a new epoch request");
        assert_eq!(request.generation, 2);
        assert_eq!(request.epoch, 1);
        assert_eq!(h.gate.snapshot(), snapshot(2, None, true));
    }

    #[test]
    fn callbacks_after_invalidation_or_source_replacement_cannot_restore_cleared_state() {
        let mut h = Harness::new();
        // Source A: request generation 1.
        let (old_slot, old_seq, _) = h.request().expect("source A slot");
        h.mapping();
        assert_eq!(h.gate.snapshot(), snapshot(1, None, true));

        // The pass disables the environment or replaces the model/source.
        h.gate.invalidate();
        assert_eq!(h.gate.snapshot(), snapshot(1, None, false));

        // Source B: request generation 2 in the new epoch.
        let (new_slot, new_seq, _) = h.request().expect("source B slot");
        assert_eq!(h.gate.snapshot(), snapshot(2, None, true));
        h.mapping();

        // The old callback lands after the invalidation: it frees its slot, publishes nothing.
        h.complete(old_slot, old_seq, Some(uniform_sh(1.0)));
        assert_eq!(h.gate.snapshot(), snapshot(2, None, true));
        assert_eq!(h.gate.published_sh(), None);
        assert_eq!(h.gate.error(), None);

        // Only the replaced source's completion publishes.
        h.complete(new_slot, new_seq, Some(uniform_sh(2.0)));
        let (generation, sh) = h.gate.take_publication().expect("new publication");
        assert_eq!(generation, 2);
        assert_eq!(sh, uniform_sh(2.0));
        assert_eq!(h.gate.snapshot(), snapshot(2, Some(2), false));
    }

    #[test]
    fn invalidation_keeps_outstanding_staging_slots_until_their_callbacks_free_them() {
        let mut h = Harness::new();
        let mut requests = Vec::new();
        for _ in 0..STAGING_RING {
            requests.push(h.request().expect("a free slot"));
        }
        h.mapping();
        h.gate.invalidate();
        assert_eq!(
            h.gate.snapshot(),
            snapshot(STAGING_RING as u64, None, false)
        );
        assert!(h.request().is_none(), "mapping slots are not reused early");
        assert!(h.readback.take_retry(), "the skip asks for a retry");

        let (slot, seq, _) = requests[1];
        h.complete(slot, seq, Some(uniform_sh(1.0)));
        assert_eq!(
            h.gate.published_sh(),
            None,
            "a cancelled callback publishes nothing"
        );
        assert!(h.request().is_some(), "its slot is free for a new request");
    }

    #[test]
    fn a_late_callback_for_a_reused_slot_frees_nothing() {
        let mut h = Harness::new();
        let (slot, seq, _) = h.request().expect("first request");
        h.mapping();
        h.gate.invalidate();
        h.complete(slot, seq, Some(uniform_sh(1.0)));
        let (slot2, seq2, _) = h.request().expect("second request");
        h.mapping();
        assert_eq!(
            h.readback.complete(slot2, seq, Some(uniform_sh(9.0))),
            Completion::Stale,
            "an old sequence does not match the reused slot"
        );
        assert!(
            h.readback.complete(slot2, seq2, Some(uniform_sh(2.0)))
                == Completion::Accepted {
                    tag: RequestTag {
                        generation: 2,
                        epoch: 1,
                    },
                    value: uniform_sh(2.0),
                },
            "the real completion still matches"
        );
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
        let gate = PublicationGate::new();
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
        filter
            .record(&gpu, &mut encoder, &source, &gate)
            .expect("valid source");
        assert_eq!(
            gate.snapshot(),
            snapshot(1, None, true),
            "the recorded request is pending"
        );
        gpu.queue().submit([encoder.finish()]);
        filter.after_submit(&gate);
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        let error = raa_core::testing::block_on(scope.pop());
        assert!(error.is_none(), "validation error: {error:?}");
        let (generation, sh) = gate
            .take_publication()
            .expect("SH read back after the poll");
        assert_eq!(generation, 1);
        assert_eq!(gate.take_publication(), None, "published once");
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
        let error = filter
            .record(&gpu, &mut encoder, &small, &gate)
            .expect_err("small cube refused");
        assert_eq!(error.code(), "ibl.invalid_source");
        assert!(error.message().contains("128x128x6"), "{error}");
        assert_eq!(
            gate.snapshot(),
            snapshot(1, None, false),
            "an invalid source invalidates the publication"
        );
        assert_eq!(gate.published_sh(), None);
        drop(filter);
        std::mem::forget(gpu);
    }
}
