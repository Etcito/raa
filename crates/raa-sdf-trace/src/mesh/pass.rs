//! Frame pass. Only the compute lanes rejected by the screen gate enter BVH traversal.

use std::{cell::RefCell, rc::Rc, sync::mpsc::sync_channel};

use raa_core::{GpuContext, RaaError};
use raa_pipeline::{FeaturePass, HDR_FORMAT, ImageDesc, ImageDim, Pipeline, RecordCtx};

use super::geometry::TRIANGLE_BYTES;
use super::{
    COUNTER_COUNT, MeshTraceScene, MeshTraceSettings, MeshTraceStats, PathMode, REFERENCE_SPP,
    TraceReport, hybrid::HybridGpu,
};
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
        Self::install_legacy(pipeline, state.clone())?;
        Ok(state)
    }

    /// Install the legacy pass on the same scene used by graph passes and spherical GI.
    pub fn install_legacy(pipeline: &mut Pipeline, state: MeshTraceHandle) -> Result<(), RaaError> {
        if pipeline.entry_enabled(MESH_TRACE_NAME).is_none() {
            pipeline.add(Box::new(MeshTracePass::new(
                MESH_TRACE_NAME.into(),
                state,
                None,
            )))?;
            pipeline.set_enabled(MESH_TRACE_NAME, false)?;
        }
        Ok(())
    }

    pub fn install_graph(
        pipeline: &mut Pipeline,
        entry: String,
        state: MeshTraceHandle,
        options: GraphMeshTrace,
    ) -> Result<(), RaaError> {
        options.settings.validate()?;
        let reports = Rc::new(RefCell::new(TraceReport::default()));
        let mut pass = MeshTracePass::new(entry.clone(), state.clone(), Some(options));
        pass.graph_report = Some(reports.clone());
        pipeline.add(Box::new(pass))?;
        state.borrow_mut().graph_reports.insert(entry, reports);
        Ok(())
    }
}

/// Per-instance graph options. Graph traces share scene buffers and model weights.
#[derive(Clone, Copy)]
pub struct GraphMeshTrace {
    pub settings: MeshTraceSettings,
    pub reference_sample: (u32, u32),
}

struct MeshTracePass {
    entry: String,
    graph: Option<GraphMeshTrace>,
    graph_report: Option<Rc<RefCell<TraceReport>>>,
    state: MeshTraceHandle,
    gpu: Option<Pipelines>,
    hybrid: Option<HybridGpu>,
    targets: Option<Images>,
}

impl MeshTracePass {
    fn new(entry: String, state: MeshTraceHandle, graph: Option<GraphMeshTrace>) -> Self {
        Self {
            entry,
            graph,
            graph_report: None,
            state,
            gpu: None,
            hybrid: None,
            targets: None,
        }
    }
}

impl Drop for MeshTracePass {
    fn drop(&mut self) {
        if let Some(report) = &self.graph_report {
            let mut scene = self.state.borrow_mut();
            if scene
                .graph_reports
                .get(&self.entry)
                .is_some_and(|active| Rc::ptr_eq(active, report))
            {
                scene.graph_reports.remove(&self.entry);
                scene.release_gpu_geometry_if_idle();
            }
        }
    }
}

impl FeaturePass for MeshTracePass {
    fn name(&self) -> &str {
        &self.entry
    }
    fn order(&self) -> u32 {
        ORDER
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let mut scene = self.state.borrow_mut();
        let graph = self.graph;
        let settings = graph.map_or(scene.settings, |options| options.settings);
        if !settings.active() {
            return Ok(());
        }
        let [nodes, triangles, materials] = scene.geometry_buffers(ctx.gpu);
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
            self.targets = Some(Images::new(
                device,
                ctx.size,
                graph.is_some() && settings.reference,
            ));
        }
        let images = self.targets.as_mut().expect("mesh ray targets");
        let input = |port| ctx.resources.input(&self.entry, port);
        let screen = if graph.is_some() {
            input("screenTrace")
        } else {
            ctx.resources.get(CONFIDENCE, "trace")
        };
        let hiz = if graph.is_some() {
            input("hiz")
        } else {
            ctx.resources.get(CONFIDENCE, "hiz")
        };
        let source = if graph.is_some() {
            input("screenColor")
        } else {
            ctx.resources.get("forward", "color")
        };
        if !settings.reference && (screen.is_none() || hiz.is_none() || source.is_none()) {
            return Err(RaaError::FrameGraph {
                code: "trace.forward_order",
                message: format!(
                    "{} requires current screen trace, Hi-Z, and source HDR",
                    self.entry
                ),
            });
        }
        let depth = ctx.targets.depth_input(ctx.resources, &self.entry);
        let normal = ctx.targets.normal_input(ctx.resources, &self.entry);
        let motion = ctx.targets.motion_input(ctx.resources, &self.entry);
        if graph.is_some()
            && !settings.reference
            && [&screen, &hiz, &source].into_iter().any(|image| {
                image
                    .as_ref()
                    .is_some_and(|image| (image.desc.width, image.desc.height) != ctx.size)
            })
        {
            return Err(RaaError::FrameGraph {
                code: "trace.input_extent",
                message: format!("{} requires scene-sized inputs", self.entry),
            });
        }
        let history_valid = images.previous.is_some_and(|(index, epoch)| {
            index.checked_add(1) == Some(ctx.frame) && scene.epoch == epoch
        });
        let path_mode = if settings.reference {
            PathMode::Off
        } else {
            settings.path_mode
        };
        let reference_sample = graph
            .map(|options| options.reference_sample)
            .or(scene.reference_sample)
            .filter(|_| settings.reference);
        let (reference_seed, reference_spp) =
            reference_sample.unwrap_or((ctx.frame as u32, REFERENCE_SPP));
        let filtered = ctx.targets.hdr();
        let params = [
            scene.geometry.count(),
            settings.reflection_bounces,
            settings.path_bounces,
            u32::from(settings.reference),
            settings.confidence_threshold.to_bits(),
            u32::from(settings.reflections),
            u32::from(history_valid),
            reference_spp,
            scene.sky_diffuse.to_bits(),
            path_mode as u32,
            settings.diffuse_output as u32,
            reference_seed,
            settings.panel_sampling as u32,
            settings.lighting_output.code(),
            u32::from(reference_sample.is_some()),
            u32::from(graph.is_some()),
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
                buffer_entry(2, &nodes),
                buffer_entry(3, &triangles),
                buffer_entry(4, &materials),
                buffer_entry(5, &gpu.counters),
                image_entry(6, &depth),
                image_entry(7, &normal),
                image_entry(8, &motion),
                image_entry(
                    9,
                    screen
                        .as_ref()
                        .map_or(&images.previous_depth, |image| &image.view),
                ),
                image_entry(10, &images.previous_depth),
                image_entry(
                    11,
                    source
                        .as_ref()
                        .map_or(&images.previous_depth, |image| &image.view),
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
                image_entry(18, &images.terminal_samples),
                image_entry(19, &images.terminal_surfaces),
                image_entry(20, &images.terminal_weights),
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
                    graph.is_none(),
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
        ctx.resources.publish(
            &self.entry,
            "radiance",
            &images.radiance,
            ImageDesc::new(ImageDim::D2, HDR_FORMAT, width, height, 1, 1),
        );
        // An unlinked graph color output has no assigned HDR destination.
        if graph.is_none() || filtered.is_bound() {
            if graph.is_some() && settings.reference {
                ctx.encoder.copy_texture_to_texture(
                    images.radiance.texture().as_image_copy(),
                    filtered.dst_texture().as_image_copy(),
                    extent(ctx.size),
                );
            } else {
                let composite_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("trace.mesh.resolve"),
                    layout: &gpu.composite.get_bind_group_layout(0),
                    entries: &[
                        image_entry(
                            0,
                            source
                                .as_ref()
                                .filter(|_| graph.is_some() && path_mode != PathMode::Off)
                                .map_or(filtered.src(), |image| &image.view),
                        ),
                        image_entry(1, &images.radiance),
                        image_entry(2, &motion),
                        buffer_entry(3, &gpu.options),
                        image_entry(4, &images.terminal_samples),
                        image_entry(5, &images.terminal_surfaces),
                        image_entry(6, &images.terminal_weights),
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
            }
            filtered.swap();
            ctx.resources.publish(
                &self.entry,
                "color",
                filtered.src(),
                ImageDesc::new(ImageDim::D2, HDR_FORMAT, ctx.size.0, ctx.size.1, 1, 1),
            );
        }
        let mut graph_report = self.graph_report.as_ref().map(|report| report.borrow_mut());
        let report = graph_report.as_deref_mut().unwrap_or(&mut scene.report);
        report.poll(settings);
        if report.pending.is_none() {
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
                            reference_spp: if settings.reference { reference_spp } else { 0 },
                            counts,
                        })
                    });
                    let _ = send.send(report);
                },
            );
            report.pending = Some(receive);
        }
        Ok(())
    }
}

struct Images {
    size: (u32, u32),
    trace_size: (u32, u32),
    radiance: wgpu::TextureView,
    terminal_samples: wgpu::TextureView,
    terminal_surfaces: wgpu::TextureView,
    terminal_weights: wgpu::TextureView,
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
    fn new(device: &wgpu::Device, size: (u32, u32), full_resolution: bool) -> Self {
        let trace_size = if full_resolution {
            size
        } else {
            (size.0.div_ceil(2), size.1.div_ceil(2))
        };
        let terminal_size = if full_resolution { (1, 1) } else { trace_size };
        let history_size = if full_resolution { (1, 1) } else { size };
        Self {
            size,
            trace_size,
            previous: None,
            radiance: image(
                device,
                "trace.mesh.radiance",
                trace_size,
                HDR_FORMAT,
                wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            ),
            terminal_samples: image(
                device,
                "trace.mesh.terminal_samples",
                terminal_size,
                HDR_FORMAT,
                wgpu::TextureUsages::STORAGE_BINDING,
            ),
            terminal_surfaces: image(
                device,
                "trace.mesh.terminal_surfaces",
                terminal_size,
                wgpu::TextureFormat::Rgba32Uint,
                wgpu::TextureUsages::STORAGE_BINDING,
            ),
            terminal_weights: image(
                device,
                "trace.mesh.terminal_weights",
                terminal_size,
                HDR_FORMAT,
                wgpu::TextureUsages::STORAGE_BINDING,
            ),
            previous_depth: image(
                device,
                "trace.mesh.previous_depth",
                history_size,
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

pub(super) struct GeometryBuffers {
    nodes: DataBuffer,
    triangles: DataBuffer,
    materials: DataBuffer,
}

impl GeometryBuffers {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            nodes: DataBuffer::new(device, "trace.mesh.nodes", 32),
            triangles: DataBuffer::new(device, "trace.mesh.triangles", TRIANGLE_BYTES),
            materials: DataBuffer::new(device, "trace.mesh.materials", 48),
        }
    }

    pub(super) fn upload(
        &mut self,
        gpu: &GpuContext,
        geometry: &super::Geometry,
        material_revision: u64,
        material_bytes: &[u8],
    ) -> [wgpu::Buffer; 3] {
        self.nodes
            .upload(gpu, geometry.revision, &geometry.node_bytes);
        self.triangles
            .upload(gpu, geometry.revision, &geometry.triangle_bytes);
        self.materials
            .upload(gpu, material_revision, material_bytes);
        [
            self.nodes.buffer.clone(),
            self.triangles.buffer.clone(),
            self.materials.buffer.clone(),
        ]
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
        let storage_image = |format| wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: D::D2,
        };
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
                entry(12, storage_image(HDR_FORMAT)),
                entry(13, buf(B::Uniform)),
                entry(14, tex(T::Float { filterable: true }, D::Cube)),
                entry(
                    15,
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
                entry(16, tex(T::Float { filterable: true }, D::D3)),
                entry(17, buf(B::Uniform)),
                entry(18, storage_image(HDR_FORMAT)),
                entry(19, storage_image(wgpu::TextureFormat::Rgba32Uint)),
                entry(20, storage_image(HDR_FORMAT)),
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
        }
    }
}
