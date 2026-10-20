//! The forward pass: pipeline, GPU buffers and per-mesh draws shared with the renderer.

use std::cell::RefCell;
use std::num::NonZeroU64;
use std::rc::Rc;

use raa_core::{
    BACKBUFFER, CLEAR_COLOR, GpuContext, MeshId, PassBuilder, RaaError, RecordContext, RenderPass,
    ResourceId,
};

use crate::prep::{FRAME_UNIFORM_SIZE, INSTANCE_STRIDE};
use crate::{DrawRange, RendererOptions};

/// WGSL of the forward pass: `raa_material::WGSL_SHADING` followed by `forward.wgsl`.
pub const FORWARD_WGSL: &str = concat!(
    include_str!("../../raa-material/src/shading.wgsl"),
    include_str!("forward.wgsl")
);

/// Byte size of one S1 vertex.
const VERTEX_STRIDE: u64 = 24;
/// Byte size of one S1 MaterialGpu.
const MATERIAL_STRIDE: u64 = 48;

const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 8] = wgpu::vertex_attr_array![
    2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4,
    6 => Float32x4, 7 => Float32x4, 8 => Float32x4,
    9 => Uint32
];

/// One uploaded mesh.
pub(crate) struct GpuMesh {
    pub(crate) vertex: wgpu::Buffer,
    pub(crate) index: wgpu::Buffer,
    pub(crate) index_count: u32,
}

/// GPU state shared by the renderer and its [`ForwardPass`] (single-threaded WebGPU objects).
#[allow(dead_code)] // buffers and the mesh table are written by the renderer
pub(crate) struct ForwardState {
    pub(crate) pipeline: wgpu::RenderPipeline,
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) uniform_buffer: wgpu::Buffer,
    pub(crate) instance_buffer: wgpu::Buffer,
    pub(crate) material_buffer: wgpu::Buffer,
    /// Indexed by `MeshId.0`; `None` for removed or never uploaded meshes.
    pub(crate) meshes: Vec<Option<GpuMesh>>,
    /// Draws of the current frame (capacity reused).
    pub(crate) draws: Vec<DrawRange>,
    /// Depth target, set when the pass is installed in a frame graph.
    pub(crate) depth: Option<ResourceId>,
}

#[allow(dead_code)] // used by the renderer
impl ForwardState {
    /// Creates the pipeline for `surface_format` and the uniform, instance and material
    /// buffers sized from `opts`.
    pub(crate) fn new(
        gpu: &GpuContext,
        surface_format: wgpu::TextureFormat,
        opts: RendererOptions,
    ) -> Result<ForwardState, RaaError> {
        let device = gpu.device();
        let instance_size = u64::from(opts.max_instances.max(1)) * INSTANCE_STRIDE as u64;
        let material_size = u64::from(opts.max_materials.max(1)) * MATERIAL_STRIDE;
        let limits = device.limits();
        if instance_size > limits.max_buffer_size {
            return Err(RaaError::Resource(format!(
                "render.too_many_instances: {} instances need {instance_size} B, limit {}",
                opts.max_instances, limits.max_buffer_size
            )));
        }
        if material_size > limits.max_storage_buffer_binding_size {
            return Err(RaaError::Resource(format!(
                "render.too_many_materials: {} materials need {material_size} B, limit {}",
                opts.max_materials, limits.max_storage_buffer_binding_size
            )));
        }

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa.forward.frame"),
            size: FRAME_UNIFORM_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa.forward.instances"),
            size: instance_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let material_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa.forward.materials"),
            size: material_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa.forward.layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(FRAME_UNIFORM_SIZE as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(MATERIAL_STRIDE),
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa.forward.bind_group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: material_buffer.as_entire_binding(),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raa.forward.pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa.forward.wgsl"),
            source: wgpu::ShaderSource::Wgsl(FORWARD_WGSL.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("raa.forward.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: VERTEX_STRIDE,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &VERTEX_ATTRIBUTES,
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: INSTANCE_STRIDE as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &INSTANCE_ATTRIBUTES,
                    }),
                ],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Ok(ForwardState {
            pipeline,
            bind_group,
            uniform_buffer,
            instance_buffer,
            material_buffer,
            meshes: Vec::new(),
            draws: Vec::with_capacity(opts.max_instances as usize),
            depth: None,
        })
    }
}

/// Frame-graph pass drawing every [`DrawRange`] of the shared state with depth testing.
pub struct ForwardPass {
    state: Rc<RefCell<ForwardState>>,
    depth: ResourceId,
}

#[allow(dead_code)] // created by the renderer's install
impl ForwardPass {
    /// Creates the pass writing BACKBUFFER and `depth`, and records `depth` in the state.
    pub(crate) fn new(state: Rc<RefCell<ForwardState>>, depth: ResourceId) -> ForwardPass {
        state.borrow_mut().depth = Some(depth);
        ForwardPass { state, depth }
    }
}

impl RenderPass for ForwardPass {
    fn name(&self) -> &str {
        "forward"
    }

    fn declare(&self, builder: &mut PassBuilder) {
        builder.write(BACKBUFFER);
        builder.write(self.depth);
    }

    fn record(&mut self, ctx: &mut RecordContext<'_>) -> Result<(), RaaError> {
        let color_view = ctx.view(BACKBUFFER)?;
        let depth_view = ctx.view(self.depth)?;
        let color_load = if ctx.is_first_write(BACKBUFFER) {
            wgpu::LoadOp::Clear(CLEAR_COLOR)
        } else {
            wgpu::LoadOp::Load
        };
        let depth_load = if ctx.is_first_write(self.depth) {
            wgpu::LoadOp::Clear(1.0)
        } else {
            wgpu::LoadOp::Load
        };
        let state = self.state.borrow();
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("forward"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: color_load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: depth_load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&state.pipeline);
        pass.set_bind_group(0, &state.bind_group, &[]);
        pass.set_vertex_buffer(1, state.instance_buffer.slice(..));
        for draw in &state.draws {
            let mesh = mesh_of(&state.meshes, draw.mesh)?;
            pass.set_vertex_buffer(0, mesh.vertex.slice(..));
            pass.set_index_buffer(mesh.index.slice(..), wgpu::IndexFormat::Uint32);
            let first = draw.first_instance;
            let end = first.saturating_add(draw.instance_count);
            pass.draw_indexed(0..mesh.index_count, 0, first..end);
        }
        Ok(())
    }
}

/// Looks up an uploaded mesh; unknown or removed ids give `render.unknown_mesh`.
fn mesh_of(meshes: &[Option<GpuMesh>], id: MeshId) -> Result<&GpuMesh, RaaError> {
    meshes
        .get(id.0 as usize)
        .and_then(Option::as_ref)
        .ok_or_else(|| {
            RaaError::Resource(format!(
                "render.unknown_mesh: mesh {} is not uploaded",
                id.0
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_wgsl_starts_with_material_shading() {
        assert!(FORWARD_WGSL.starts_with(raa_material::WGSL_SHADING));
        assert!(FORWARD_WGSL.contains("fn vs_main"));
        assert!(FORWARD_WGSL.contains("fn fs_main"));
    }

    #[test]
    fn unknown_mesh_is_a_resource_refusal() {
        let meshes: Vec<Option<GpuMesh>> = vec![None];
        for id in [MeshId(0), MeshId(7)] {
            let err = mesh_of(&meshes, id)
                .err()
                .expect("unknown mesh must be refused");
            assert_eq!(err.code(), "core.resource");
            assert!(err.to_string().contains("render.unknown_mesh"), "{err}");
        }
    }
}
