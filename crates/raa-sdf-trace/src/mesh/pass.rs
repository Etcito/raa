//! Frame pass. Only the compute lanes rejected by the screen gate enter BVH traversal.

use std::{cell::RefCell, rc::Rc, sync::mpsc::sync_channel};

use raa_core::{GpuContext, RaaError};
use raa_pipeline::{FeaturePass, HDR_FORMAT, ImageDesc, ImageDim, Pipeline, RecordCtx};

use super::{
    COUNTER_COUNT, MeshTraceScene, MeshTraceStats, PathMode, REFERENCE_SPP, hybrid::HybridGpu,
};
use super::geometry::TRIANGLE_BYTES;
use crate::wgsl::mesh_trace_module;

pub const MESH_TRACE_NAME: &str = "trace.mesh";
const ORDER: u32 = 66;
const COUNTER_BYTES: u64 = (COUNTER_COUNT * 4) as u64;
const CONFIDENCE: &str = "ssr.confidence";
const COMPOSITE_WGSL: &str = include_str!("../wgsl/mesh_composite.wgsl");

pub type MeshTraceHandle = Rc<RefCell<MeshTraceScene>>;

pub struct MeshTraceFeature;

impl MeshTraceFeature {
    pub fn install(pipeline: &mut Pipeline) -> Result<MeshTraceHandle, RaaError> {
        let state = MeshTraceHandle::default();
        pipeline.add(Box::new(MeshTracePass {
            state: state.clone(),
            gpu: None,
            hybrid: None,
            targets: None,
        }))?;
        pipeline.set_enabled(MESH_TRACE_NAME, false)?;
        Ok(state)
    }
}

struct MeshTracePass {
    state: MeshTraceHandle,
    gpu: Option<Pipelines>,
    hybrid: Option<HybridGpu>,
    targets: Option<Images>,
}

impl FeaturePass for MeshTracePass {
    fn name(&self) -> &str {
        MESH_TRACE_NAME
    }
    fn order(&self) -> u32 {
        ORDER
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let mut scene = self.state.borrow_mut();
        let settings = scene.settings;
        if !settings.active() {
            return Ok(());
        }
        scene.poll();
        let Some(frame) = scene.frame.as_ref() else {
            return Ok(());
        };
        let device = ctx.gpu.device();
        let gpu = self.gpu.get_or_insert_with(|| Pipelines::new(device));
        if self
            .targets
            .as_ref()
            .is_none_or(|images| images.size != ctx.size)
        {
            self.targets = Some(Images::new(device, ctx.size));
        }
        let images = self.targets.as_mut().expect("mesh ray targets");
        let screen = ctx.resources.get(CONFIDENCE, "trace");
        let hiz = ctx.resources.get(CONFIDENCE, "hiz");
        if !settings.reference && (screen.is_none() || hiz.is_none()) {
            return Err(RaaError::FrameGraph {
                code: "trace.forward_order",
                message: "hybrid reflections need the current confidence pass before trace.mesh"
                    .into(),
            });
        }
        let source = ctx.resources.get("forward", "color");
        if source.is_none() && !settings.reference {
            return Err(RaaError::FrameGraph {
                code: "trace.forward_order",
                message: "hybrid reflections need forward HDR before trace.mesh".into(),
            });
        }
        gpu.triangles.upload(
            ctx.gpu,
            scene.geometry.revision,
            &scene.geometry.triangle_bytes,
        );
        gpu.nodes
            .upload(ctx.gpu, scene.geometry.revision, &scene.geometry.node_bytes);
        gpu.materials
            .upload(ctx.gpu, scene.material_revision, &scene.material_bytes);
        let history_valid = images.previous.is_some_and(|(index, epoch)| {
            index.checked_add(1) == Some(ctx.frame) && scene.epoch == epoch
        });
        let path_mode = if settings.reference {
            PathMode::Off
        } else {
            settings.path_mode
        };
        let filtered = ctx.targets.hdr();
        let params = [
            scene.geometry.count(),
            settings.reflection_bounces,
            settings.path_bounces,
            u32::from(settings.reference),
            settings.confidence_threshold.to_bits(),
            u32::from(settings.reflections),
            u32::from(history_valid),
            REFERENCE_SPP,
            scene.sky_diffuse.to_bits(),
            path_mode as u32,
            settings.diffuse_output as u32,
            ctx.frame as u32,
            0, // Reserved for the direct-light sampler.
            settings.lighting_output.code(),
            0,
            0,
        ];
        let queue = ctx.gpu.queue();
        let bytes: Vec<_> = params.into_iter().flat_map(u32::to_le_bytes).collect();
        queue.write_buffer(&gpu.options, 0, &bytes);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(MESH_TRACE_NAME),
            layout: &gpu.compute_layout,
            entries: &[
                buffer_entry(0, frame),
                buffer_entry(1, &gpu.options),
                buffer_entry(2, &gpu.nodes.buffer),
                buffer_entry(3, &gpu.triangles.buffer),
                buffer_entry(4, &gpu.materials.buffer),
                buffer_entry(5, &gpu.counters),
                image_entry(6, ctx.targets.depth()),
                image_entry(7, ctx.targets.normal()),
                image_entry(8, ctx.targets.motion()),
                image_entry(
                    9,
                    screen
                        .as_ref()
                        .map_or(&images.previous_depth, |image| &image.view),
                ),
                image_entry(10, &images.previous_depth),
                image_entry(
                    11,
                    source.as_ref().map_or(filtered.src(), |image| &image.view),
                ),
                image_entry(12, &images.radiance),
                buffer_entry(13, ctx.lighting.local_lights_buffer()),
                image_entry(14, ctx.lighting.env_specular_view()),
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: wgpu::BindingResource::Sampler(ctx.lighting.linear_clamp()),
                },
                image_entry(16, ctx.lighting.gi_volume_view()),
                buffer_entry(17, ctx.lighting.gi_volume_params_buffer()),
            ],
        });
        ctx.encoder.clear_buffer(&gpu.counters, 0, None);
        let (width, height) = images.trace_size;
        if path_mode != PathMode::Off {
            let ao = ctx.lighting.slot_view(7).expect("lighting AO slot");
            self.hybrid
                .get_or_insert_with(|| HybridGpu::new(device, &gpu.compute_layout))
                .record(
                    ctx.gpu,
                    ctx.encoder,
                    ctx.frame,
                    &group,
                    &ao,
                    &mut scene.path,
                    path_mode,
                    images.trace_size,
                );
        } else {
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(MESH_TRACE_NAME),
                    ..Default::default()
                });
            pass.set_pipeline(&gpu.trace);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        if let Some(hiz) = hiz {
            ctx.encoder.copy_texture_to_texture(
                hiz.view.texture().as_image_copy(),
                images.previous_depth.texture().as_image_copy(),
                extent(ctx.size),
            );
            images.previous = Some((ctx.frame, scene.epoch));
        }
        let composite_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trace.mesh.resolve"),
            layout: &gpu.composite.get_bind_group_layout(0),
            entries: &[
                image_entry(0, filtered.src()),
                image_entry(1, &images.radiance),
                image_entry(2, ctx.targets.motion()),
                buffer_entry(3, &gpu.options),
            ],
        });
        {
            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("trace.mesh.resolve"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: filtered.dst(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&gpu.composite);
            pass.set_bind_group(0, &composite_group, &[]);
            pass.draw(0..3, 0..1);
        }
        filtered.swap();
        ctx.resources.publish(
            MESH_TRACE_NAME,
            "color",
            filtered.src(),
            ImageDesc::new(ImageDim::D2, HDR_FORMAT, ctx.size.0, ctx.size.1, 1, 1),
        );

        if scene.pending.is_none() {
            let (send, receive) = sync_channel(1);
            ctx.encoder
                .copy_buffer_to_buffer(&gpu.counters, 0, &gpu.readback, 0, COUNTER_BYTES);
            let readback = gpu.readback.clone();
            let frame = ctx.frame;
            ctx.encoder.map_buffer_on_submit(
                &gpu.readback,
                wgpu::MapMode::Read,
                ..,
                move |loaded| {
                    let report = loaded.ok().and_then(|()| {
                        let counts = readback.get_mapped_range(..).ok().map(|bytes| {
                            std::array::from_fn(|i| {
                                u32::from_le_bytes(
                                    bytes[i * 4..i * 4 + 4].try_into().expect("one counter"),
                                )
                            })
                        });
                        readback.unmap();
                        counts.map(|counts| MeshTraceStats {
                            frame,
                            size: (width, height),
                            settings,
                            counts,
                        })
                    });
                    let _ = send.send(report);
                },
            );
            scene.pending = Some(receive);
        }
        Ok(())
    }
}

struct Images {
    size: (u32, u32),
    trace_size: (u32, u32),
    radiance: wgpu::TextureView,
    previous_depth: wgpu::TextureView,
    previous: Option<(u64, u64)>,
}

fn extent(size: (u32, u32)) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: size.0,
        height: size.1,
        depth_or_array_layers: 1,
    }
}

fn image(
    device: &wgpu::Device,
    label: &'static str,
    size: (u32, u32),
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: extent(size),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: usage | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

impl Images {
    fn new(device: &wgpu::Device, size: (u32, u32)) -> Self {
        let trace_size = (size.0.div_ceil(2), size.1.div_ceil(2));
        Self {
            size,
            trace_size,
            previous: None,
            radiance: image(
                device,
                "trace.mesh.radiance",
                trace_size,
                HDR_FORMAT,
                wgpu::TextureUsages::STORAGE_BINDING,
            ),
            previous_depth: image(
                device,
                "trace.mesh.previous_depth",
                size,
                wgpu::TextureFormat::R32Float,
                wgpu::TextureUsages::COPY_DST,
            ),
        }
    }
}

struct DataBuffer {
    buffer: wgpu::Buffer,
    revision: Option<u64>,
    label: &'static str,
}

impl DataBuffer {
    fn new(device: &wgpu::Device, label: &'static str, min: u64) -> Self {
        let buffer = buffer(
            device,
            label,
            min,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        Self {
            buffer,
            revision: None,
            label,
        }
    }

    fn upload(&mut self, gpu: &GpuContext, revision: u64, bytes: &[u8]) {
        if self.revision == Some(revision) {
            return;
        }
        if bytes.len() as u64 > self.buffer.size() {
            let size = bytes.len().next_power_of_two() as u64;
            self.buffer = buffer(
                gpu.device(),
                self.label,
                size,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
        }
        if !bytes.is_empty() {
            gpu.queue().write_buffer(&self.buffer, 0, bytes);
        }
        self.revision = Some(revision);
    }
}

pub(super) fn buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

pub(super) fn buffer_entry(binding: u32, value: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: value.as_entire_binding(),
    }
}

pub(super) fn image_entry(binding: u32, value: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(value),
    }
}

struct Pipelines {
    compute_layout: wgpu::BindGroupLayout,
    trace: wgpu::ComputePipeline,
    composite: wgpu::RenderPipeline,
    options: wgpu::Buffer,
    counters: wgpu::Buffer,
    readback: wgpu::Buffer,
    nodes: DataBuffer,
    triangles: DataBuffer,
    materials: DataBuffer,
}

impl Pipelines {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(MESH_TRACE_NAME),
            source: wgpu::ShaderSource::Wgsl(mesh_trace_module().into()),
        });
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            ty,
            visibility: wgpu::ShaderStages::COMPUTE,
            count: None,
        };
        let buf = |ty| wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let tex = |sample_type, view_dimension| wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled: false,
        };
        use wgpu::{BufferBindingType as B, TextureSampleType as T, TextureViewDimension as D};
        let read = buf(B::Storage { read_only: true });
        let unfiltered = tex(T::Float { filterable: false }, D::D2);
        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(MESH_TRACE_NAME),
            entries: &[
                entry(0, buf(B::Uniform)),
                entry(1, buf(B::Uniform)),
                entry(2, read),
                entry(3, read),
                entry(4, read),
                entry(5, buf(B::Storage { read_only: false })),
                entry(6, tex(T::Depth, D::D2)),
                entry(7, unfiltered),
                entry(8, unfiltered),
                entry(9, unfiltered),
                entry(10, unfiltered),
                entry(11, unfiltered),
                entry(
                    12,
                    wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: HDR_FORMAT,
                        view_dimension: D::D2,
                    },
                ),
                entry(13, buf(B::Uniform)),
                entry(14, tex(T::Float { filterable: true }, D::Cube)),
                entry(
                    15,
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
                entry(16, tex(T::Float { filterable: true }, D::D3)),
                entry(17, buf(B::Uniform)),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(MESH_TRACE_NAME),
            bind_group_layouts: &[Some(&compute_layout)],
            immediate_size: 0,
        });
        let trace = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(MESH_TRACE_NAME),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("trace_mesh"),
            compilation_options: Default::default(),
            cache: None,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("trace.mesh.resolve"),
            source: wgpu::ShaderSource::Wgsl(COMPOSITE_WGSL.into()),
        });
        let composite = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("trace.mesh.resolve"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(HDR_FORMAT.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            compute_layout,
            trace,
            composite,
            options: buffer(
                device,
                "trace.mesh.options",
                64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            counters: buffer(
                device,
                "trace.mesh.counters",
                COUNTER_BYTES,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            ),
            readback: buffer(
                device,
                "trace.mesh.readback",
                COUNTER_BYTES,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            nodes: DataBuffer::new(device, "trace.mesh.nodes", 32),
            triangles: DataBuffer::new(device, "trace.mesh.triangles", TRIANGLE_BYTES),
            materials: DataBuffer::new(device, "trace.mesh.materials", 48),
        }
    }
}
