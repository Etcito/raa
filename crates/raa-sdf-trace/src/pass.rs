//! Trace scene state and the section 67 render targets.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::GpuContext;
use raa_core::RaaError;
use raa_core::wgpu;
use raa_pipeline::{FeaturePass, Pipeline, PrepareCtx, RecordCtx, Targets};

use crate::bvh::{self, Bvh};
use crate::program::{INSTANCE_COUNT, INSTANCE_WORDS, Instance, TraceError, TraceProgram, parse};
use crate::wgsl::{VOLUME_COARSE_TILE, composite_module, volume_mip_layout, volume_module};

pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// The trace pass runs at the full size of the render target; a zero axis collapses to one pixel.
pub fn target_size(width: u32, height: u32) -> (u32, u32) {
    (width.max(1), height.max(1))
}

#[derive(Clone, Debug)]
pub struct TraceTargets {
    pub color: wgpu::Texture,
    pub color_view: wgpu::TextureView,
    pub depth: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    /// Coarse cone prepass size: one texel per `VOLUME_COARSE_TILE` full pixels.
    pub coarse_size: (u32, u32),
    /// Sampled coarse result the full trace resumes from (binding 3).
    pub coarse: wgpu::Texture,
    pub coarse_view: wgpu::TextureView,
    /// Coarse output written by `sdf_coarse` (binding 4), copied to `coarse`.
    pub coarse_out: wgpu::Texture,
    pub coarse_out_view: wgpu::TextureView,
    pub size: (u32, u32),
}

impl TraceTargets {
    pub fn new(gpu: &GpuContext, width: u32, height: u32) -> TraceTargets {
        let size = target_size(width, height);
        let make = |label: &str, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
            let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        let storage = wgpu::TextureUsages::STORAGE_BINDING
            .union(wgpu::TextureUsages::TEXTURE_BINDING)
            .union(wgpu::TextureUsages::COPY_SRC);
        let (color, color_view) = make("raa-sdf-trace.color", COLOR_FORMAT, storage);
        let (depth, depth_view) = make("raa-sdf-trace.depth", DEPTH_FORMAT, storage);
        let coarse_size = (
            size.0.div_ceil(VOLUME_COARSE_TILE),
            size.1.div_ceil(VOLUME_COARSE_TILE),
        );
        let coarse_texture = |label: &str, usage: wgpu::TextureUsages| {
            let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: coarse_size.0,
                    height: coarse_size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        let (coarse, coarse_view) = coarse_texture(
            "raa-sdf-trace.coarse",
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        let (coarse_out, coarse_out_view) = coarse_texture(
            "raa-sdf-trace.coarse-out",
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        );
        TraceTargets {
            color,
            color_view,
            depth,
            depth_view,
            coarse_size,
            coarse,
            coarse_view,
            coarse_out,
            coarse_out_view,
            size,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TraceStats {
    pub sources: usize,
    pub instances: usize,
    pub bvh_nodes: usize,
    pub width: u32,
    pub height: u32,
    pub placeholders: usize,
}

#[derive(Clone, Debug)]
pub struct TraceScene {
    pub program: Option<TraceProgram>,
    pub bvh: Bvh,
    /// The frame's 512-byte S10 `FrameGpu`: camera matrices, sun, ambient and viewport.
    pub frame: [u8; FRAME_BYTES],
    targets: Option<TraceTargets>,
    frame_buffer: Option<wgpu::Buffer>,
    volume: Option<TraceVolume>,
    volume_revision: Option<u64>,
    target_revision: u64,
    revision: u64,
    enabled: bool,
}

impl Default for TraceScene {
    fn default() -> TraceScene {
        TraceScene {
            program: None,
            bvh: Bvh::default(),
            frame: [0; FRAME_BYTES],
            targets: None,
            frame_buffer: None,
            volume: None,
            volume_revision: None,
            target_revision: 0,
            revision: 0,
            enabled: false,
        }
    }
}

impl TraceScene {
    pub fn set_program(&mut self, bytes: &[u8]) -> Result<(), TraceError> {
        let program = parse(bytes)?;
        self.bvh = bvh::build(&program);
        self.program = Some(program);
        self.volume = None;
        self.revision += 1;
        Ok(())
    }

    pub fn update_instances(&mut self, words: &[f32]) -> Result<(), TraceError> {
        let Some(current) = self.program.as_ref() else {
            return Err(refusal(
                INSTANCE_COUNT,
                "instances",
                "the trace scene has no program",
            ));
        };
        if !words.len().is_multiple_of(INSTANCE_WORDS)
            || words.len() / INSTANCE_WORDS != current.instances.len()
        {
            return Err(refusal(
                INSTANCE_COUNT,
                "instances",
                format!(
                    "{} words is not {} records of {INSTANCE_WORDS} words",
                    words.len(),
                    current.instances.len()
                ),
            ));
        }
        let mut next = current.clone();
        for (index, record) in words.chunks_exact(INSTANCE_WORDS).enumerate() {
            let raw: Vec<u32> = record.iter().map(|value| value.to_bits()).collect();
            let instance = read_instance(&raw, current.instances[index].bend);
            next.instances[index] = instance;
        }
        crate::program::validate_instances(&next)?;
        self.bvh = bvh::build(&next);
        self.program = Some(next);
        self.volume = None;
        self.revision += 1;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.program = None;
        self.bvh = Bvh::default();
        self.volume = None;
        self.revision += 1;
    }

    pub fn stats(&self, size: (u32, u32)) -> TraceStats {
        let program = self.program.as_ref();
        TraceStats {
            sources: program.map_or(0, |p| p.sources.len()),
            instances: program.map_or(0, |p| p.instances.len()),
            bvh_nodes: self.bvh.nodes.len(),
            width: size.0,
            height: size.1,
            placeholders: program
                .map_or(0, |p| p.sources.iter().filter(|s| s.placeholder()).count()),
        }
    }

    /// World volume bounds derived from the current BVH; `None` without instances.
    pub fn volume_layout(&self) -> Option<VolumeLayout> {
        let program = self.program.as_ref()?;
        if program.instances.is_empty() {
            return None;
        }
        VolumeLayout::enclose(self.bvh.nodes.first()?.min, self.bvh.nodes.first()?.max)
    }

    /// Build (or rebuild) the volume before its consumers read it. Program, instance and
    /// bounds changes bump the revision, so a stale volume is never returned. A scene
    /// without instances keeps a valid dummy volume (frozen ABI).
    pub fn volume(&mut self, gpu: &GpuContext) -> Option<&TraceVolume> {
        if self.volume_revision == Some(self.revision) {
            return self.volume.as_ref();
        }
        let layout = self.volume_layout().unwrap_or_else(VolumeLayout::dummy);
        let revision = abi_revision(self.revision);
        self.volume = Some(TraceVolume::new(gpu, layout, revision));
        self.volume_revision = Some(self.revision);
        self.volume.as_ref()
    }
}

fn abi_revision(revision: u64) -> u32 {
    (revision % 0x00ff_ffffu64).max(1) as u32
}

fn refusal(code: &'static str, path: &str, message: impl Into<String>) -> TraceError {
    TraceError {
        code,
        path: path.to_string(),
        message: message.into(),
    }
}

fn affine(words: &[u32], base: usize) -> [[f32; 4]; 3] {
    core::array::from_fn(|row| {
        core::array::from_fn(|column| f32::from_bits(words[base + row * 4 + column]))
    })
}

fn read_instance(words: &[u32], bend: Option<[f64; 8]>) -> Instance {
    Instance {
        world_from_local: affine(words, 0),
        local_from_world: affine(words, 12),
        dist_scale: f32::from_bits(words[24]),
        source: words[25],
        material: words[26] as i32,
        key: words[27],
        deform_kind: words[28],
        bend_angle: f32::from_bits(words[29]),
        bend_length: f32::from_bits(words[30]),
        axes: words[31],
        bend,
    }
}

/// Section 67 world volume: five mip levels over an R8Snorm 3D texture whose longest base
/// axis is at most 1024 voxels, rounded up to a multiple of 16.
pub const VOLUME_MIPS: u32 = 5;
pub const VOLUME_MAX_AXIS: u32 = 1024;
pub const VOLUME_AXIS_ALIGN: u32 = 16;
pub const VOLUME_ROW_ALIGN: u32 = 256;
pub const VOLUME_UNIFORM_BYTES: u64 = 64;
/// Default WebGPU `max_storage_buffer_binding_size`; the stamped buffer is bound whole.
pub const VOLUME_MAX_BUFFER_BYTES: u64 = 128 << 20;

fn align_up(value: u32, align: u32) -> u32 {
    value.div_ceil(align) * align
}

/// World bounds, isotropic voxel spacing and signed-distance range of the volume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeLayout {
    pub origin: [f32; 3],
    pub spacing: f32,
    pub base: [u32; 3],
    pub range: f32,
}

/// One mip's dimensions plus its buffer-to-texture copy layout (bytes and u32 words).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VolumeMip {
    pub level: u32,
    pub size: [u32; 3],
    pub bytes_per_row: u32,
    pub rows_per_image: u32,
    pub pitch_words: u32,
    pub dispatch: [u32; 3],
}

impl VolumeLayout {
    /// Enclose `min..max` (the instance BVH root). Empty, non-finite or degenerate bounds
    /// own no volume; `range` covers the padded diagonal so no interior distance saturates.
    pub fn enclose(min: [f32; 3], max: [f32; 3]) -> Option<VolumeLayout> {
        if !min.iter().chain(max.iter()).all(|value| value.is_finite()) {
            return None;
        }
        let mut extent = [0.0f32; 3];
        let mut longest = 0.0f32;
        for axis in 0..3 {
            if max[axis] <= min[axis] {
                return None;
            }
            extent[axis] = max[axis] - min[axis];
            longest = longest.max(extent[axis]);
        }
        let mut spacing = longest / VOLUME_MAX_AXIS as f32;
        let (base, range) = loop {
            let mut base = [VOLUME_AXIS_ALIGN; 3];
            for axis in 0..3 {
                let cells = (extent[axis] / spacing).ceil() as u32;
                base[axis] =
                    align_up(cells.max(VOLUME_AXIS_ALIGN), VOLUME_AXIS_ALIGN).min(VOLUME_MAX_AXIS);
            }
            let range = spacing
                * (0..3)
                    .map(|axis| (base[axis] * base[axis]) as f32)
                    .sum::<f32>()
                    .sqrt();
            let bytes = u64::from(align_up(base[0], VOLUME_ROW_ALIGN))
                * u64::from(base[1])
                * u64::from(base[2]);
            if bytes <= VOLUME_MAX_BUFFER_BYTES {
                break (base, range);
            }
            // Fit the default max_storage_buffer_binding_size by coarsening, never by clipping.
            spacing *= 2.0;
        };
        Some(VolumeLayout {
            origin: min,
            spacing,
            base,
            range,
        })
    }

    /// A valid empty-scene volume: 16^3 cells, one voxel covering 1/16 m.
    pub fn dummy() -> VolumeLayout {
        let base = [VOLUME_AXIS_ALIGN; 3];
        let spacing = 1.0 / VOLUME_AXIS_ALIGN as f32;
        let range = spacing
            * (0..3)
                .map(|axis| (base[axis] * base[axis]) as f32)
                .sum::<f32>()
                .sqrt();
        VolumeLayout {
            origin: [-0.5; 3],
            spacing,
            base,
            range,
        }
    }

    pub fn mip(&self, level: u32) -> VolumeMip {
        let size = [0, 1, 2].map(|axis| (self.base[axis] >> level).max(1));
        let bytes_per_row = align_up(size[0], VOLUME_ROW_ALIGN);
        VolumeMip {
            level,
            size,
            bytes_per_row,
            rows_per_image: size[1],
            pitch_words: bytes_per_row / 4,
            dispatch: [size[0].div_ceil(32), size[1].div_ceil(8), size[2]],
        }
    }

    /// The packed stamp buffer for the base mip; each coarser level reuses its prefix.
    pub fn buffer_bytes(&self) -> u64 {
        let mip = self.mip(0);
        u64::from(mip.pitch_words) * u64::from(mip.size[1]) * u64::from(mip.size[2]) * 4
    }

    /// Absolute evaluator and coordinate roundoff allowance of the frozen ABI (`encoding.y`).
    pub fn roundoff(&self) -> f32 {
        (self.range * 1e-5).max(1e-6)
    }

    /// The 64-byte `TraceVolume` uniform: origin/spacing, base shape, encoding and build.
    /// `build.x` is the mip level and `build.y` the row pitch in u32 words. The revision
    /// fields (`shape.w`, `build.z`) stay zero until [`uniform_revision`] sets both.
    ///
    /// [`uniform_revision`]: VolumeLayout::uniform_revision
    pub fn uniform(&self, level: u32) -> [u8; VOLUME_UNIFORM_BYTES as usize] {
        let mut words = [0u32; 16];
        for axis in 0..3 {
            words[axis] = self.origin[axis].to_bits();
        }
        words[3] = self.spacing.to_bits();
        words[4..7].copy_from_slice(&self.base);
        words[8] = self.range.to_bits();
        words[9] = self.roundoff().to_bits();
        words[12] = level;
        words[13] = self.mip(level).pitch_words;
        let mut bytes = [0u8; VOLUME_UNIFORM_BYTES as usize];
        for (index, word) in words.iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// One dispatch's immutable uniform snapshot: the layout of `level` stamped with the
    /// revision consumers must match (`shape.w` and `build.z`).
    pub fn uniform_revision(
        &self,
        level: u32,
        revision: u32,
    ) -> [u8; VOLUME_UNIFORM_BYTES as usize] {
        let mut bytes = self.uniform(level);
        bytes[28..32].copy_from_slice(&revision.to_le_bytes());
        bytes[56..60].copy_from_slice(&revision.to_le_bytes());
        bytes
    }
}

/// The scene's R8Snorm world volume and the packed buffer compute stamps into before each
/// mip is copied into the texture. Frame scheduling binds and fills it.
#[derive(Clone, Debug)]
pub struct TraceVolume {
    pub layout: VolumeLayout,
    /// The frozen-ABI revision stamped into every uniform and checked by the shaders.
    pub revision: u32,
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub buffer: wgpu::Buffer,
    /// Immutable per-mip snapshot used by `sdf_stamp` / `sdf_mip`.
    pub mip_uniforms: [wgpu::Buffer; VOLUME_MIPS as usize],
    /// Immutable snapshot used by `sdf_coarse`.
    pub coarse_uniform: wgpu::Buffer,
    /// Immutable snapshot used by `sdf_trace`.
    pub uniform: wgpu::Buffer,
}

impl TraceVolume {
    pub fn new(gpu: &GpuContext, layout: VolumeLayout, revision: u32) -> TraceVolume {
        let device = gpu.device();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("raa-sdf-trace.volume"),
            size: wgpu::Extent3d {
                width: layout.base[0],
                height: layout.base[1],
                depth_or_array_layers: layout.base[2],
            },
            mip_level_count: VOLUME_MIPS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R8Snorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("raa-sdf-trace.volume"),
            format: Some(wgpu::TextureFormat::R8Snorm),
            dimension: Some(wgpu::TextureViewDimension::D3),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(VOLUME_MIPS),
            base_array_layer: 0,
            array_layer_count: None,
        });
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-sdf-trace.volume.stamp"),
            size: layout.buffer_bytes(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let uniform = |label: &str, level: u32| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: VOLUME_UNIFORM_BYTES,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue().write_buffer(&buffer, 0, &layout.uniform(level));
            buffer
        };
        let mip_uniforms =
            core::array::from_fn(|level| uniform("raa-sdf-trace.volume.mip", level as u32));
        TraceVolume {
            layout,
            revision,
            texture,
            view,
            buffer,
            mip_uniforms,
            coarse_uniform: uniform("raa-sdf-trace.volume.coarse", 0),
            uniform: uniform("raa-sdf-trace.volume.trace", 0),
        }
    }

    /// Publishes `revision` to every uniform snapshot once mip 0..4 were stamped and copied.
    /// Until then `shape.w`/`build.z` stay zero and the shaders use exact BVH tracing.
    pub fn publish(&self, gpu: &GpuContext) {
        let queue = gpu.queue();
        for (level, buffer) in self.mip_uniforms.iter().enumerate() {
            queue.write_buffer(
                buffer,
                0,
                &self.layout.uniform_revision(level as u32, self.revision),
            );
        }
        let snapshot = self.layout.uniform_revision(0, self.revision);
        queue.write_buffer(&self.coarse_uniform, 0, &snapshot);
        queue.write_buffer(&self.uniform, 0, &snapshot);
    }
}
// ---- frame passes (S12) ----
pub const TRACE_NAME: &str = "trace.sdf";
pub const TRACE_ORDER: u32 = 55;
pub const COMPOSITE_NAME: &str = "trace.composite";
pub const COMPOSITE_ORDER: u32 = 560;
pub const FRAME_BYTES: usize = 512;
const WORKGROUP: u32 = 8;
/// The installed scene: mutate it through this clone.
pub type TraceHandle = Rc<RefCell<TraceScene>>;
pub struct TraceFeature;
impl TraceFeature {
    pub fn install(gpu: &GpuContext, pipeline: &mut Pipeline) -> Result<TraceHandle, RaaError> {
        if pipeline
            .passes()
            .iter()
            .any(|info| info.name == TRACE_NAME || info.name == COMPOSITE_NAME)
        {
            return Err(RaaError::FrameGraph {
                code: "pipeline.duplicate_pass",
                message: "a trace pass is already installed".into(),
            });
        }
        let frame = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-sdf-trace.frame"),
            size: FRAME_BYTES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scene = Rc::new(RefCell::new(TraceScene {
            frame_buffer: Some(frame),
            enabled: true,
            ..TraceScene::default()
        }));
        pipeline.add(Box::new(TracePass::new(Rc::clone(&scene))))?;
        let format = pipeline.surface_format();
        pipeline.add(Box::new(CompositePass::new(gpu, format, Rc::clone(&scene))))?;
        Ok(scene)
    }
}
const MIN_WORDS: [usize; 8] = [1, 1, 12, 32, 8, 8, 1, 8];
fn f32_words<'a>(values: impl Iterator<Item = &'a f32>) -> impl Iterator<Item = u32> {
    values.map(|value| value.to_bits())
}
fn word_buffer(gpu: &GpuContext, words: &[u32], min: usize) -> wgpu::Buffer {
    let size = words.len().max(min).max(1) as u64 * 4;
    let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("raa-sdf-trace.section"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !words.is_empty() {
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        gpu.queue().write_buffer(&buffer, 0, &bytes);
    }
    buffer
}
fn empty_program() -> TraceProgram {
    TraceProgram {
        sources: Vec::new(),
        code: Vec::new(),
        data: Vec::new(),
        materials: Vec::new(),
        instances: Vec::new(),
    }
}
fn sections(program: Option<&TraceProgram>, bvh: &Bvh) -> [Vec<u32>; 8] {
    let mut out: [Vec<u32>; 8] = Default::default();
    if let Some(program) = program {
        out[0] = program.code.clone();
        out[1] = f32_words(program.data.iter()).collect();
        for source in &program.sources {
            out[2].extend([
                source.code_start,
                source.code_len,
                source.data_start,
                source.data_len,
            ]);
            out[2].extend(f32_words(source.min.iter()));
            out[2].extend(f32_words(source.max.iter()));
            out[2].extend([source.lipschitz.to_bits(), source.flags]);
        }
        for instance in &program.instances {
            out[3].extend(f32_words(instance.world_from_local.iter().flatten()));
            out[3].extend(f32_words(instance.local_from_world.iter().flatten()));
            out[3].extend([
                instance.dist_scale.to_bits(),
                instance.source,
                instance.material as u32,
                instance.key,
                instance.deform_kind,
                instance.bend_angle.to_bits(),
                instance.bend_length.to_bits(),
                instance.axes,
            ]);
            match instance.bend {
                Some(bend) if instance.deform_kind == 1 => {
                    out[4].extend(bend.iter().map(|value| (*value as f32).to_bits()));
                }
                _ => out[4].extend([0; 8]),
            }
        }
        for material in &program.materials {
            out[7].extend(f32_words(material.color.iter()));
            out[7].extend([
                material.roughness.to_bits(),
                material.metallic.to_bits(),
                0,
                0,
                0,
            ]);
        }
    }
    for node in &bvh.nodes {
        out[5].extend(f32_words(node.min.iter()));
        out[5].push(node.left);
        out[5].extend(f32_words(node.max.iter()));
        out[5].push(node.count);
    }
    out[6] = bvh.order.clone();
    out
}
struct VolumePipelines {
    stamp: wgpu::ComputePipeline,
    mip: wgpu::ComputePipeline,
    coarse: wgpu::ComputePipeline,
    trace: wgpu::ComputePipeline,
    frame: wgpu::BindGroupLayout,
    volume: wgpu::BindGroupLayout,
}

const READ_ONLY: wgpu::BufferBindingType = wgpu::BufferBindingType::Storage { read_only: true };
const READ_WRITE: wgpu::BufferBindingType = wgpu::BufferBindingType::Storage { read_only: false };

fn buffer_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(
    binding: u32,
    dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: dimension,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

impl VolumePipelines {
    /// Group 0 is exactly `trace_module`'s layout; group 1 is the frozen volume ABI.
    fn new(gpu: &GpuContext, module: &wgpu::ShaderModule) -> VolumePipelines {
        let device = gpu.device();
        let frame_entries: Vec<wgpu::BindGroupLayoutEntry> = (0..8)
            .map(|binding| buffer_entry(binding, READ_ONLY))
            .chain([
                buffer_entry(8, wgpu::BufferBindingType::Uniform),
                storage_entry(9, COLOR_FORMAT),
                storage_entry(10, DEPTH_FORMAT),
            ])
            .collect();
        let frame = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-sdf-trace.frame-layout"),
            entries: &frame_entries,
        });
        let volume_entries = [
            buffer_entry(0, wgpu::BufferBindingType::Uniform),
            texture_entry(1, wgpu::TextureViewDimension::D3),
            buffer_entry(2, READ_WRITE),
            texture_entry(3, wgpu::TextureViewDimension::D2),
            storage_entry(4, wgpu::TextureFormat::Rgba32Float),
        ];
        let volume = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-sdf-trace.volume-layout"),
            entries: &volume_entries,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raa-sdf-trace.pipeline-layout"),
            bind_group_layouts: &[Some(&frame), Some(&volume)],
            immediate_size: 0,
        });
        let compute = |label: &str, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        VolumePipelines {
            stamp: compute("raa-sdf-trace.stamp", "sdf_stamp"),
            mip: compute("raa-sdf-trace.mip", "sdf_mip"),
            coarse: compute("raa-sdf-trace.coarse", "sdf_coarse"),
            trace: compute("raa-sdf-trace.trace", "sdf_trace"),
            frame,
            volume,
        }
    }
}

struct VolumeGroups {
    frame: wgpu::BindGroup,
    mips: [wgpu::BindGroup; VOLUME_MIPS as usize],
    coarse: wgpu::BindGroup,
    trace: wgpu::BindGroup,
}

impl VolumeGroups {
    fn new(
        gpu: &GpuContext,
        pipelines: &VolumePipelines,
        buffers: &[wgpu::Buffer; 8],
        frame_buffer: &wgpu::Buffer,
        targets: &TraceTargets,
        volume: &TraceVolume,
    ) -> VolumeGroups {
        let device = gpu.device();
        let frame_entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .chain([
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: frame_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&targets.color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(&targets.depth_view),
                },
            ])
            .collect();
        let frame = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-sdf-trace.frame-group"),
            layout: &pipelines.frame,
            entries: &frame_entries,
        });
        let volume_group = |uniform: &wgpu::Buffer| {
            let entries = [
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&volume.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: volume.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&targets.coarse_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&targets.coarse_out_view),
                },
            ];
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raa-sdf-trace.volume-group"),
                layout: &pipelines.volume,
                entries: &entries,
            })
        };
        VolumeGroups {
            frame,
            mips: core::array::from_fn(|level| volume_group(&volume.mip_uniforms[level])),
            coarse: volume_group(&volume.coarse_uniform),
            trace: volume_group(&volume.uniform),
        }
    }
}

struct TracePass {
    scene: TraceHandle,
    pipelines: Option<VolumePipelines>,
    count: Option<usize>,
    buffers: Option<[wgpu::Buffer; 8]>,
    uploaded: Option<u64>,
    groups: Option<VolumeGroups>,
    groups_for: Option<(u64, u64)>,
    /// Scene revision whose volume was stamped and published in an earlier frame.
    stamped: Option<u64>,
}

impl TracePass {
    fn new(scene: TraceHandle) -> TracePass {
        TracePass {
            scene,
            pipelines: None,
            count: None,
            buffers: None,
            uploaded: None,
            groups: None,
            groups_for: None,
            stamped: None,
        }
    }
}

impl FeaturePass for TracePass {
    fn name(&self) -> &str {
        TRACE_NAME
    }
    fn order(&self) -> u32 {
        TRACE_ORDER
    }
    fn set_enabled(&mut self, enabled: bool, _lighting: &mut raa_pipeline::LightingResources) {
        self.scene.borrow_mut().enabled = enabled;
    }
    fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
        let size = targets.size();
        let mut scene = self.scene.borrow_mut();
        scene.targets = Some(TraceTargets::new(gpu, size.0, size.1));
        scene.target_revision += 1;
        drop(scene);
        self.groups = None;
        self.groups_for = None;
    }
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let (revision, count) = {
            let scene = self.scene.borrow();
            (
                scene.revision,
                scene
                    .program
                    .as_ref()
                    .map_or(0, |program| program.instances.len()),
            )
        };
        if self.count != Some(count) {
            let source = self
                .scene
                .borrow()
                .program
                .as_ref()
                .map_or_else(|| volume_module(&empty_program()), volume_module);
            let module = ctx
                .gpu
                .device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(TRACE_NAME),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            self.pipelines = Some(VolumePipelines::new(ctx.gpu, &module));
            self.count = Some(count);
            self.stamped = None;
            self.groups = None;
            self.groups_for = None;
        }
        if self.uploaded != Some(revision) {
            let scene = self.scene.borrow();
            let words = sections(scene.program.as_ref(), &scene.bvh);
            self.buffers = Some(std::array::from_fn(|index| {
                word_buffer(ctx.gpu, &words[index], MIN_WORDS[index])
            }));
            self.uploaded = Some(revision);
            self.groups = None;
            self.groups_for = None;
        }
        {
            let mut scene = self.scene.borrow_mut();
            if let Some(frame) = scene.frame_buffer.as_ref() {
                ctx.gpu.queue().write_buffer(frame, 0, &scene.frame);
            }
            scene.volume(ctx.gpu);
        }
        let scene = self.scene.borrow();
        let key = Some((scene.revision, scene.target_revision));
        if self.groups_for != key
            && let (Some(pipelines), Some(buffers), Some(targets), Some(frame), Some(volume)) = (
                self.pipelines.as_ref(),
                self.buffers.as_ref(),
                scene.targets.as_ref(),
                scene.frame_buffer.as_ref(),
                scene.volume.as_ref(),
            )
        {
            self.groups = Some(VolumeGroups::new(
                ctx.gpu, pipelines, buffers, frame, targets, volume,
            ));
            self.groups_for = key;
        }
        Ok(())
    }
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let scene = self.scene.borrow();
        if !scene.enabled {
            return Ok(());
        }
        let (Some(targets), Some(pipelines), Some(groups), Some(volume)) = (
            scene.targets.as_ref(),
            self.pipelines.as_ref(),
            self.groups.as_ref(),
            scene.volume.as_ref(),
        ) else {
            return Ok(());
        };
        if self.stamped != Some(scene.revision) {
            let mips: Option<Vec<_>> = (0..VOLUME_MIPS)
                .map(|level| volume_mip_layout(volume.layout.base, level))
                .collect();
            let Some(mips) = mips else {
                return Ok(());
            };
            for (level, mip) in mips.iter().enumerate() {
                let level = level as u32;
                let mut pass = ctx
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(if level == 0 {
                            "raa-sdf-trace.stamp"
                        } else {
                            "raa-sdf-trace.mip"
                        }),
                        timestamp_writes: None,
                    });
                pass.set_pipeline(if level == 0 {
                    &pipelines.stamp
                } else {
                    &pipelines.mip
                });
                pass.set_bind_group(0, &groups.frame, &[]);
                pass.set_bind_group(1, &groups.mips[level as usize], &[]);
                pass.dispatch_workgroups(mip.workgroups[0], mip.workgroups[1], mip.workgroups[2]);
                drop(pass);
                ctx.encoder.copy_buffer_to_texture(
                    wgpu::TexelCopyBufferInfo {
                        buffer: &volume.buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(mip.bytes_per_row),
                            rows_per_image: Some(mip.rows_per_image),
                        },
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &volume.texture,
                        mip_level: level,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: mip.dimensions[0],
                        height: mip.dimensions[1],
                        depth_or_array_layers: mip.dimensions[2],
                    },
                );
            }
            volume.publish(ctx.gpu);
            self.stamped = Some(scene.revision);
        }
        let workgroups = |(w, h): (u32, u32)| (w.div_ceil(WORKGROUP), h.div_ceil(WORKGROUP));
        let (coarse_x, coarse_y) = workgroups(targets.coarse_size);
        {
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("raa-sdf-trace.coarse"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&pipelines.coarse);
            pass.set_bind_group(0, &groups.frame, &[]);
            pass.set_bind_group(1, &groups.coarse, &[]);
            pass.dispatch_workgroups(coarse_x, coarse_y, 1);
        }
        ctx.encoder.copy_texture_to_texture(
            targets.coarse_out.as_image_copy(),
            targets.coarse.as_image_copy(),
            wgpu::Extent3d {
                width: targets.coarse_size.0,
                height: targets.coarse_size.1,
                depth_or_array_layers: 1,
            },
        );
        {
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("raa-sdf-trace.trace"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&pipelines.trace);
            pass.set_bind_group(0, &groups.frame, &[]);
            pass.set_bind_group(1, &groups.trace, &[]);
            let (width, height) = workgroups(targets.size);
            pass.dispatch_workgroups(width, height, 1);
        }
        Ok(())
    }
}
struct CompositePass {
    scene: TraceHandle,
    pipeline: wgpu::RenderPipeline,
    frame_group: wgpu::BindGroup,
    group: Option<wgpu::BindGroup>,
    group_revision: Option<u64>,
}
/// Explicit layouts of `trace.composite`: the frame uniform and the two unfilterable
/// trace textures (r32float is not a filterable format).
fn composite_layouts(gpu: &GpuContext) -> (wgpu::BindGroupLayout, wgpu::PipelineLayout) {
    let device = gpu.device();
    let frame = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(COMPOSITE_NAME),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(FRAME_BYTES as u64),
            },
            count: None,
        }],
    });
    let texture = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let textures = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(COMPOSITE_NAME),
        entries: &[texture(0), texture(1)],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(COMPOSITE_NAME),
        bind_group_layouts: &[Some(&frame), Some(&textures)],
        immediate_size: 0,
    });
    (frame, layout)
}

impl CompositePass {
    fn new(gpu: &GpuContext, format: wgpu::TextureFormat, scene: TraceHandle) -> CompositePass {
        let (frame_layout, layout) = composite_layouts(gpu);
        let module = gpu
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(COMPOSITE_NAME),
                source: wgpu::ShaderSource::Wgsl(composite_module().into()),
            });
        let pipeline = gpu
            .device()
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(COMPOSITE_NAME),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        let frame_group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(COMPOSITE_NAME),
            layout: &frame_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: scene
                    .borrow()
                    .frame_buffer
                    .as_ref()
                    .expect("frame buffer")
                    .as_entire_binding(),
            }],
        });
        CompositePass {
            scene,
            pipeline,
            frame_group,
            group: None,
            group_revision: None,
        }
    }
}
impl FeaturePass for CompositePass {
    fn name(&self) -> &str {
        COMPOSITE_NAME
    }
    fn order(&self) -> u32 {
        COMPOSITE_ORDER
    }
    fn output_overlay(&self) -> bool {
        true
    }
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let scene = self.scene.borrow();
        if !scene.enabled || scene.program.is_none() {
            return Ok(());
        }
        let Some(targets) = scene.targets.as_ref() else {
            return Ok(());
        };
        if self.group_revision != Some(scene.target_revision) {
            let layout = self.pipeline.get_bind_group_layout(1);
            self.group = Some(
                ctx.gpu
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(COMPOSITE_NAME),
                        layout: &layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&targets.color_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&targets.depth_view),
                            },
                        ],
                    }),
            );
            self.group_revision = Some(scene.target_revision);
        }
        let Some(group) = self.group.as_ref() else {
            return Ok(());
        };
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(COMPOSITE_NAME),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.backbuffer,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.targets.depth(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.frame_group, &[]);
        pass.set_bind_group(1, group, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}
