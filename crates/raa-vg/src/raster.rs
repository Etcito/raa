//! The S32 raster module of RAA's virtualized geometry: the hardware rasteriser's WGSL
//! constants and module builder, the per-pixel [`VisibilityBuffer`] and the hardware
//! [`Rasterizer`] that draws a cull's output into it. The software rasteriser, the cascade
//! depth path and their frame passes land with their own cards. Adapted from solstice (MIT),
//! Copyright (c) 2026 SamG-Coder: `crates/solstice-render/shaders/vg/raster_common.wgsl`,
//! `raster_hw.wgsl`, `raster_hw_fragment.wgsl`, `vis_write.wgsl` and the hardware parts of
//! `crates/solstice-render/src/vg/raster.rs:8-110` (the `VisibilityBuffer`, `new`, `bindings`
//! and `record`), revision 7e465b5. The reversed-Z visibility write becomes the standard-Z
//! `raa_depth_key` integer compare.

use crate::culling::{CullWork, PASS_VIEW_WGSL, ViewSlots};
use crate::scene::VgScene;
use raa_core::{GpuContext, wgpu};

/// S32 `raster_common.wgsl`: the group-2 scene bindings (0 `PassView`, 1 instances, 2 clusters,
/// 3 vertices, 4 triangles, 5 visible, 6 `CullStateRead`) and the `encode_id` and
/// `local_vertex` helpers shared by both rasterisers. The builder concatenates it after
/// `raa_shader::DEPTH_WGSL` and [`PASS_VIEW_WGSL`].
pub const RASTER_COMMON_WGSL: &str = include_str!("wgsl/raster_common.wgsl");

/// S32 `vis_write.wgsl`: group-2 bindings 7 (`vis_depth`) and 8 (`vis_id`), both
/// `array<atomic<u32>>`, and `write_visibility`, which keeps the larger `raa_depth_key` of
/// the fragment depth with the integer compare-exchange loop of the source, then stores the id.
pub const VIS_WRITE_WGSL: &str = include_str!("wgsl/vis_write.wgsl");

const RASTER_HW_WGSL: &str = include_str!("wgsl/raster_hw.wgsl");

const RASTER_HW_FRAGMENT_WGSL: &str = include_str!("wgsl/raster_hw_fragment.wgsl");

/// The hardware raster module: `raa_shader::DEPTH_WGSL` + [`PASS_VIEW_WGSL`] +
/// [`RASTER_COMMON_WGSL`] + [`VIS_WRITE_WGSL`] + `raster_hw.wgsl` + `raster_hw_fragment.wgsl`,
/// with entry points `vs_visibility`, `vs_depth` (vertex) and `fs_visibility` (fragment) at
/// group 2 bindings 0-8.
pub fn raster_hw_module_wgsl() -> String {
    [
        raa_shader::DEPTH_WGSL,
        PASS_VIEW_WGSL,
        RASTER_COMMON_WGSL,
        VIS_WRITE_WGSL,
        RASTER_HW_WGSL,
        RASTER_HW_FRAGMENT_WGSL,
    ]
    .concat()
}

/// Format of the colour attachment that exists only to satisfy WebGPU's rule that a render
/// pass have at least one attachment. Nothing is ever written to it.
const DUMMY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// The stages that see the read-write visibility bindings 7-8. WebGPU refuses writable
/// storage in the vertex stage without `VERTEX_WRITABLE_STORAGE`, so they are fragment and
/// compute only; the software rasteriser reaches them from a compute pass.
const VIS_STORAGE_STAGES: wgpu::ShaderStages =
    wgpu::ShaderStages::FRAGMENT.union(wgpu::ShaderStages::COMPUTE);

/// One buffer of `size` bytes with `usage`, labelled with `label`.
fn buffer(gpu: &GpuContext, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

/// A bind group entry naming the whole of `buffer`.
fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// Per-pixel depth key and triangle ID, as storage buffers both rasterisers can write, plus
/// the dummy colour target the hardware pass draws into.
pub struct VisibilityBuffer {
    depth: wgpu::Buffer,
    ids: wgpu::Buffer,
    dummy_target: wgpu::TextureView,
    size: (u32, u32),
}

impl VisibilityBuffer {
    /// Two STORAGE | COPY_DST | COPY_SRC buffers of `w * h` u32 words, one for the depth key
    /// and one for the id, and a `w` by `h` R8Unorm render target for the hardware pass.
    /// `COPY_SRC` exists so tests can read the buffers back.
    pub fn new(gpu: &GpuContext, size: (u32, u32)) -> VisibilityBuffer {
        let bytes = u64::from(size.0) * u64::from(size.1) * 4;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let depth = buffer(gpu, "raa-vg visibility depth", bytes, usage);
        let ids = buffer(gpu, "raa-vg visibility ids", bytes, usage);
        let dummy_target = gpu
            .device()
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("raa-vg visibility dummy target"),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DUMMY_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        VisibilityBuffer {
            depth,
            ids,
            dummy_target,
            size,
        }
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    pub fn depth(&self) -> &wgpu::Buffer {
        &self.depth
    }

    pub fn ids(&self) -> &wgpu::Buffer {
        &self.ids
    }

    /// Resets every pixel to "no geometry". Depth key 0 is the far plane or empty, and the
    /// resolve reads the depth before the id, so the ids need no clearing.
    pub fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.depth, 0, None);
    }

    /// The two storage buffers and the one-byte-per-pixel dummy target.
    pub fn allocated_bytes(&self) -> u64 {
        self.depth.size() + self.ids.size() + u64::from(self.size.0) * u64::from(self.size.1)
    }
}

/// One group-2 bind group of a raster, plus the `PassView` slot its dynamic offset selects.
pub struct RasterBindings {
    group: wgpu::BindGroup,
    slot: u32,
}

/// What a hardware raster draws into. The `Cascade` variant lands with the cascade card.
pub enum RasterTarget<'a> {
    Visibility(&'a VisibilityBuffer),
}

/// The hardware visibility rasteriser: its group-2 layout, the `vs_visibility` +
/// `fs_visibility` render pipeline and the 16-byte placeholder bound in place of a scene
/// buffer that does not exist yet. It owns no frame state.
pub struct Rasterizer {
    layout: wgpu::BindGroupLayout,
    hardware: wgpu::RenderPipeline,
    placeholder: wgpu::Buffer,
}

impl Rasterizer {
    /// The group-2 layout (0-6 visible to vertex, fragment and compute; 7-8 read-write storage
    /// visible to fragment and compute only), the hardware visibility pipeline and the
    /// placeholder buffer.
    pub fn new(gpu: &GpuContext) -> Rasterizer {
        let device = gpu.device();
        let scene_stages = wgpu::ShaderStages::VERTEX_FRAGMENT.union(wgpu::ShaderStages::COMPUTE);
        let storage = |read_only: bool| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let entry = |binding: u32, visibility: wgpu::ShaderStages, ty: wgpu::BindingType| {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility,
                ty,
                count: None,
            }
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-vg raster"),
            entries: &[
                entry(
                    0,
                    scene_stages,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(
                            crate::culling::PASS_VIEW_SIZE as u64,
                        ),
                    },
                ),
                entry(1, scene_stages, storage(true)),
                entry(2, scene_stages, storage(true)),
                entry(3, scene_stages, storage(true)),
                entry(4, scene_stages, storage(true)),
                entry(5, scene_stages, storage(true)),
                entry(6, scene_stages, storage(true)),
                entry(7, VIS_STORAGE_STAGES, storage(false)),
                entry(8, VIS_STORAGE_STAGES, storage(false)),
            ],
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-vg hardware raster"),
            source: wgpu::ShaderSource::Wgsl(raster_hw_module_wgsl().into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raa-vg hardware raster"),
            bind_group_layouts: &[None, None, Some(&layout)],
            immediate_size: 0,
        });
        let hardware = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("raa-vg hardware raster"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_visibility"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                // Facing is decided per instance in `raster_hw.wgsl`.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_visibility"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: DUMMY_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::empty(),
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Rasterizer {
            layout,
            hardware,
            placeholder: buffer(
                gpu,
                "raa-vg raster placeholder",
                16,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
        }
    }

    /// Binds a scene, a culling run's output and a visibility buffer at group 2. A scene
    /// buffer that has not been uploaded is replaced by the placeholder, so the group is
    /// always valid.
    pub fn bindings(
        &self,
        gpu: &GpuContext,
        slots: &ViewSlots,
        slot: u32,
        scene: &VgScene,
        work: &CullWork,
        vis: &VisibilityBuffer,
    ) -> RasterBindings {
        let instances = scene.instances().unwrap_or(&self.placeholder);
        let clusters = scene.clusters().unwrap_or(&self.placeholder);
        let vertices = scene.vertices().unwrap_or(&self.placeholder);
        let triangles = scene.triangles().unwrap_or(&self.placeholder);
        let group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-vg raster"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: slots.binding(),
                },
                bind_entry(1, instances),
                bind_entry(2, clusters),
                bind_entry(3, vertices),
                bind_entry(4, triangles),
                bind_entry(5, work.visible()),
                bind_entry(6, work.state()),
                bind_entry(7, vis.depth()),
                bind_entry(8, vis.ids()),
            ],
        });
        RasterBindings { group, slot }
    }

    /// Draws what the most recent cull of `work` selected into `target`.
    ///
    /// Handoff: `Rasterizer::bindings(&self, gpu: &GpuContext, slots: &ViewSlots, slot: u32,
    /// scene: &VgScene, work: &CullWork, vis: &VisibilityBuffer) -> RasterBindings` and
    /// `Rasterizer::record(&self, encoder: &mut wgpu::CommandEncoder, bindings: &RasterBindings,
    /// work: &CullWork, target: RasterTarget<'_>)` (crates/raa-vg/src/raster.rs) are exposed to
    /// card 'raa-vg: software rasterisation of small clusters in the frame', which attaches the
    /// software dispatch by extending this method (same group-2 bindings, `CullWork::args()` at
    /// `software_args_offset()`) and records the frame's raster passes from `frame_passes.rs`.
    /// State: the group-2 `RasterBindings` bind group with its dynamic `PassView` slot and the
    /// `CullWork` indirect arguments; no keys.
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &RasterBindings,
        work: &CullWork,
        target: RasterTarget<'_>,
    ) {
        match target {
            RasterTarget::Visibility(vis) => {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("raa-vg hardware raster"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &vis.dummy_target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Discard,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.hardware);
                pass.set_bind_group(2, &bindings.group, &[ViewSlots::offset(bindings.slot)]);
                pass.draw_indirect(work.args(), CullWork::draw_args_offset());
            }
        }
    }

    /// The GPU memory the rasteriser owns: the 16-byte placeholder buffer.
    pub fn allocated_bytes(&self) -> u64 {
        self.placeholder.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::culling::CullPipelines;
    use raa_core::testing::{block_on, skip, software_gpu};

    #[test]
    fn visibility_buffer_and_rasterizer_create_without_validation_error() {
        let test = "visibility_buffer_and_rasterizer_create_without_validation_error";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let rasterizer = Rasterizer::new(&gpu);
        let vis = VisibilityBuffer::new(&gpu, (64, 64));
        let pipelines = CullPipelines::new(&gpu);
        let work = CullWork::new(&gpu, &pipelines, "raster test", 1024, 1024);
        let slots = ViewSlots::new(&gpu);
        let scene = VgScene::new();
        let _bindings = rasterizer.bindings(&gpu, &slots, 0, &scene, &work, &vis);
        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        vis.clear(&mut encoder);
        gpu.queue().submit(Some(encoder.finish()));
        let _ = gpu.device().poll(wgpu::PollType::wait_indefinitely());
        let error = block_on(errors.pop());
        assert!(error.is_none(), "raster validation: {error:?}");
        assert_eq!(vis.size(), (64, 64));
        assert_eq!(vis.allocated_bytes(), 2 * 64 * 64 * 4 + 64 * 64);
        assert_eq!(rasterizer.allocated_bytes(), 16);
        // Writable storage visible to the vertex stage would have failed the scope without
        // VERTEX_WRITABLE_STORAGE; the fragment writes 7-8 and the software pass uses compute.
        assert_eq!(
            VIS_STORAGE_STAGES,
            wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE
        );
        assert!(!VIS_STORAGE_STAGES.contains(wgpu::ShaderStages::VERTEX));
    }
}
