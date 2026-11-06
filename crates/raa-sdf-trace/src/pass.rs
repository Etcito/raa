//! Trace scene state and the section 67 render targets.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::GpuContext;
use raa_core::RaaError;
use raa_core::wgpu;
use raa_pipeline::{FeaturePass, Pipeline, PrepareCtx, RecordCtx, Targets};

use crate::bvh::{self, Bvh};
use crate::program::{INSTANCE_COUNT, INSTANCE_WORDS, Instance, TraceError, TraceProgram, parse};
use crate::wgsl::{composite_module, trace_module};

pub const MAX_TARGET: u32 = 1024;
pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// `ceil(w*f)` x `ceil(h*f)` with `f = min(1, MAX_TARGET / max(w, h))`.
pub fn target_size(width: u32, height: u32) -> (u32, u32) {
    let longest = width.max(height);
    if longest == 0 {
        return (1, 1);
    }
    if longest <= MAX_TARGET {
        return (width.max(1), height.max(1));
    }
    let scale = f64::from(MAX_TARGET) / f64::from(longest);
    (
        ((f64::from(width) * scale).ceil() as u32).max(1),
        ((f64::from(height) * scale).ceil() as u32).max(1),
    )
}

#[derive(Clone, Debug)]
pub struct TraceTargets {
    pub color: wgpu::Texture,
    pub color_view: wgpu::TextureView,
    pub depth: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    pub size: (u32, u32),
}

impl TraceTargets {
    pub fn new(gpu: &GpuContext, width: u32, height: u32) -> TraceTargets {
        let size = target_size(width, height);
        let make = |label: &str, format: wgpu::TextureFormat| {
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
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    .union(wgpu::TextureUsages::TEXTURE_BINDING)
                    .union(wgpu::TextureUsages::COPY_SRC),
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        let (color, color_view) = make("raa-sdf-trace.color", COLOR_FORMAT);
        let (depth, depth_view) = make("raa-sdf-trace.depth", DEPTH_FORMAT);
        TraceTargets {
            color,
            color_view,
            depth,
            depth_view,
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
        self.revision += 1;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.program = None;
        self.bvh = Bvh::default();
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
struct TracePass {
    scene: TraceHandle,
    pipeline: Option<wgpu::ComputePipeline>,
    count: Option<usize>,
    buffers: Option<[wgpu::Buffer; 8]>,
    uploaded: Option<u64>,
    group: Option<wgpu::BindGroup>,
}
impl TracePass {
    fn new(scene: TraceHandle) -> TracePass {
        TracePass {
            scene,
            pipeline: None,
            count: None,
            buffers: None,
            uploaded: None,
            group: None,
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
        self.group = None;
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
                .map_or_else(|| trace_module(&empty_program()), trace_module);
            let module = ctx
                .gpu
                .device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(TRACE_NAME),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            self.pipeline = Some(ctx.gpu.device().create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some(TRACE_NAME),
                    layout: None,
                    module: &module,
                    entry_point: Some("sdf_trace"),
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));
            self.count = Some(count);
            self.group = None;
        }
        if self.uploaded != Some(revision) {
            let scene = self.scene.borrow();
            let words = sections(scene.program.as_ref(), &scene.bvh);
            self.buffers = Some(std::array::from_fn(|index| {
                word_buffer(ctx.gpu, &words[index], MIN_WORDS[index])
            }));
            self.uploaded = Some(revision);
            self.group = None;
        }
        let scene = self.scene.borrow();
        if let Some(frame) = scene.frame_buffer.as_ref() {
            ctx.gpu.queue().write_buffer(frame, 0, &scene.frame);
        }
        if self.group.is_none()
            && let (Some(buffers), Some(pipeline), Some(targets), Some(frame)) = (
                self.buffers.as_ref(),
                self.pipeline.as_ref(),
                scene.targets.as_ref(),
                scene.frame_buffer.as_ref(),
            )
        {
            let mut entries: Vec<wgpu::BindGroupEntry> = buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            entries.push(wgpu::BindGroupEntry {
                binding: 8,
                resource: frame.as_entire_binding(),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::TextureView(&targets.color_view),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 10,
                resource: wgpu::BindingResource::TextureView(&targets.depth_view),
            });
            self.group = Some(
                ctx.gpu
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(TRACE_NAME),
                        layout: &pipeline.get_bind_group_layout(0),
                        entries: &entries,
                    }),
            );
        }
        Ok(())
    }
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let scene = self.scene.borrow();
        let (Some(targets), Some(pipeline), Some(group)) = (
            scene.targets.as_ref(),
            self.pipeline.as_ref(),
            self.group.as_ref(),
        ) else {
            return Ok(());
        };
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(TRACE_NAME),
                timestamp_writes: None,
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(
            targets.size.0.div_ceil(WORKGROUP),
            targets.size.1.div_ceil(WORKGROUP),
            1,
        );
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
