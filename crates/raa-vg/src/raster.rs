//! The S32 raster module of RAA's virtualized geometry: the hardware and software rasterisers'
//! WGSL constants and module builders, the per-pixel [`VisibilityBuffer`], the stamped
//! [`DepthBuffer`] of the shadow cascades and the [`Rasterizer`] that draws a cull's output
//! into either. The frame passes land with their own cards. Adapted from solstice (MIT),
//! Copyright (c) 2026 SamG-Coder: `crates/solstice-render/shaders/vg/raster_common.wgsl`,
//! `raster_hw.wgsl`, `raster_hw_fragment.wgsl`, `raster_sw.wgsl`, `raster_sw_body.wgsl`,
//! `vis_write.wgsl` and `crates/solstice-render/src/vg/raster.rs:8-110` (the
//! `VisibilityBuffer`, `new`, `bindings` and `record`) and `:14-24` and `:240-300` (the
//! `DepthBuffer`, `DEPTH_STAMPS`, `depth_buffer`, `record_software_depth` and `merge_depth`),
//! revision 7e465b5. The reversed-Z visibility write becomes the standard-Z `raa_depth_key`
//! integer compare.

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

const RASTER_SW_WGSL: &str = include_str!("wgsl/raster_sw.wgsl");

/// S32 `raster_sw_body.wgsl`: the software rasteriser of one small cluster per workgroup. Its
/// `GROUP_SIZE` is the workgroup size every module that includes it shares. The including
/// module supplies `write_visibility` and `depth_bias`, so the visibility variant keys the
/// depth through `raa_depth_key` and a cascade depth variant can stamp it instead.
pub const RASTER_SW_BODY_WGSL: &str = include_str!("wgsl/raster_sw_body.wgsl");

const DEPTH_STAMP_WGSL: &str = include_str!("wgsl/depth_stamp.wgsl");

const RASTER_SW_DEPTH_WGSL: &str = include_str!("wgsl/raster_sw_depth.wgsl");

const SHADOW_MERGE_WGSL: &str = include_str!("wgsl/shadow_merge.wgsl");

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

/// The software raster module: `raa_shader::DEPTH_WGSL` + [`PASS_VIEW_WGSL`] +
/// [`RASTER_COMMON_WGSL`] + [`VIS_WRITE_WGSL`] + `raster_sw.wgsl` + [`RASTER_SW_BODY_WGSL`],
/// with the compute entry point `main` over the same group-2 bindings 0-8 as the hardware pass
/// (all nine in one compute stage). The cluster of workgroup `w` is
/// `visible[state.sw_base + w]`, and back faces are dropped unless `instance_two_sided`.
///
/// Handoff: `raster_sw_module_wgsl() -> String` exported by crates/raa-vg/src/raster.rs is
/// exposed to card 'raa-vg: software rasterisation of small clusters in the frame', which
/// builds the compute pipeline (entry `main`) on [`Rasterizer`]'s group-2 layout and dispatches
/// it at `CullWork::software_args_offset()` next to the hardware draw. State/keys: the
/// `CullWork` indirect software extent and the `CullStateRead.sw_base`/`sw_count` window into
/// `CullWork::visible()`, whose entries hold the cluster word; the per-pixel key is written by
/// `write_visibility` (`raa_depth_key`), so no CPU-side key registry.
pub fn raster_sw_module_wgsl() -> String {
    [
        raa_shader::DEPTH_WGSL,
        PASS_VIEW_WGSL,
        RASTER_COMMON_WGSL,
        VIS_WRITE_WGSL,
        RASTER_SW_WGSL,
        RASTER_SW_BODY_WGSL,
    ]
    .concat()
}

/// The cascade software depth module: `raa_shader::DEPTH_WGSL` + [`PASS_VIEW_WGSL`] +
/// [`RASTER_COMMON_WGSL`] + `depth_stamp.wgsl` + `raster_sw_depth.wgsl` +
/// [`RASTER_SW_BODY_WGSL`], with the compute entry point `main` over group 2 bindings 0-6 as
/// the visibility software pass plus 7 `shadow_depth: array<atomic<u32>>`, which
/// `atomicMax`es the stamped word of `view.stamp`.
///
/// Handoff: `raster_sw_depth_module_wgsl() -> String` (crates/raa-vg/src/raster.rs) is exposed
/// to card 'raa-vg: cascade raster GPU path (DepthBuffer, RasterTarget::Cascade, software
/// depth, merge)', which creates its compute pipeline (entry `main`) and dispatches it at
/// `CullWork::software_args_offset()` with the view's group-2 bindings and its `DepthBuffer`
/// at binding 7. State/keys: the `CullWork` software extent and the stamped words
/// ([`DEPTH_STAMPS`]), which that card clears only when the stamps start a new round.
pub fn raster_sw_depth_module_wgsl() -> String {
    [
        raa_shader::DEPTH_WGSL,
        PASS_VIEW_WGSL,
        RASTER_COMMON_WGSL,
        DEPTH_STAMP_WGSL,
        RASTER_SW_DEPTH_WGSL,
        RASTER_SW_BODY_WGSL,
    ]
    .concat()
}

/// The cascade merge module: `raa_shader::DEPTH_WGSL` + [`PASS_VIEW_WGSL`] +
/// `depth_stamp.wgsl` + `shadow_merge.wgsl`, without [`RASTER_COMMON_WGSL`], so group 2 is 0
/// `shadow_depth` read-only storage and 1 `PassView` uniform with a dynamic offset; `vs_main`
/// (vertex) is the fullscreen triangle and `fs_main` (fragment) writes `frag_depth` from the
/// words whose stamp is `view.stamp`, discarding the others.
///
/// Handoff: `shadow_merge_module_wgsl() -> String` (crates/raa-vg/src/raster.rs) is exposed to
/// card 'raa-vg: cascade raster GPU path (DepthBuffer, RasterTarget::Cascade, software depth,
/// merge)', whose `Rasterizer::merge_depth` creates the fullscreen pipeline over the cascade's
/// existing depth attachment; card 'raa-vg: shadow.rs cascade views and cascade recording' then
/// records it after the software depth pass. State/keys: `view.viewport` sizes the pixel index,
/// `view.stamp` selects the words this drawing stamped, and `stamped_depth` restores their
/// standard-Z depth.
pub fn shadow_merge_module_wgsl() -> String {
    [
        raa_shader::DEPTH_WGSL,
        PASS_VIEW_WGSL,
        DEPTH_STAMP_WGSL,
        SHADOW_MERGE_WGSL,
    ]
    .concat()
}

/// Where a drawing's stamp sits in a stamped cascade depth word: the top byte, above the top
/// 24 bits of `raa_depth_key`. Twin of `DEPTH_STAMP_SHIFT` in `depth_stamp.wgsl`.
pub const DEPTH_STAMP_SHIFT: u32 = 24;

/// The mask of the 24-bit depth key a stamped word carries. Twin of `DEPTH_STAMP_RANGE` in
/// `depth_stamp.wgsl`.
pub const DEPTH_STAMP_RANGE: u32 = 16777215;

/// Stamps a drawing can carry: a byte, with 0 meaning never drawn.
///
/// Handoff: `DEPTH_STAMPS` (crates/raa-vg/src/raster.rs) is exposed to card 'raa-vg: cascade
/// raster GPU path (DepthBuffer, RasterTarget::Cascade, software depth, merge)', whose
/// `DepthBuffer` allocates its stamped words and clears them only when the stamps wrap. State:
/// [`DEPTH_STAMP_SHIFT`], [`DEPTH_STAMP_RANGE`] and the CPU twins [`stamped`], [`stamp_of`] and
/// [`stamped_depth`] follow the same layout as `depth_stamp.wgsl`.
pub const DEPTH_STAMPS: u32 = 255;

/// A stamped cascade depth word: the stamp in the top byte over the top 24 bits of the
/// standard-Z depth key, so a later stamp beats every earlier word and, under one stamp, a
/// nearer depth is larger. Twin of `stamped` in `depth_stamp.wgsl`.
pub fn stamped(stamp: u32, depth: f32) -> u32 {
    let key = (1.0 - depth.clamp(0.0, 1.0)).to_bits();
    (stamp << DEPTH_STAMP_SHIFT) | (key >> 8)
}

/// The drawing stamp a word carries. Twin of `stamp_of` in `depth_stamp.wgsl`.
pub fn stamp_of(word: u32) -> u32 {
    word >> DEPTH_STAMP_SHIFT
}

/// The standard-Z depth a stamped word holds. Twin of `stamped_depth` in `depth_stamp.wgsl`.
pub fn stamped_depth(word: u32) -> f32 {
    1.0 - f32::from_bits((word & DEPTH_STAMP_RANGE) << 8)
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

/// One cascade's worth of stamped software-rasterized depth: `resolution^2` u32 words, each
/// `stamp << 24 | raa_depth_key(depth) >> 8`, so the nearest sample of the latest drawing wins
/// and an earlier drawing's words lose to it. Unlike a [`VisibilityBuffer`] there is no id
/// buffer, and the buffer is cleared only when the stamps start a new round.
pub struct DepthBuffer {
    buffer: wgpu::Buffer,
    resolution: u32,
}

impl DepthBuffer {
    /// A STORAGE | COPY_DST buffer of `resolution^2` u32 words.
    pub fn new(gpu: &GpuContext, resolution: u32) -> DepthBuffer {
        let bytes = u64::from(resolution) * u64::from(resolution) * 4;
        DepthBuffer {
            buffer: buffer(
                gpu,
                "raa-vg cascade depth",
                bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            resolution,
        }
    }

    pub fn resolution(&self) -> u32 {
        self.resolution
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Resets every pixel to "never drawn"; call it when the stamps start a new round.
    pub fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.buffer, 0, None);
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.buffer.size()
    }
}

/// One group-2 bind group of a raster, plus the `PassView` slot its dynamic offset selects.
pub struct RasterBindings {
    group: wgpu::BindGroup,
    slot: u32,
}

/// What a hardware raster draws into: a visibility buffer, or one layer of a cascade depth map
/// whose software depth [`DepthBuffer`] carries `stamp`.
pub enum RasterTarget<'a> {
    Visibility(&'a VisibilityBuffer),
    Cascade {
        layer: &'a wgpu::TextureView,
        depth: &'a DepthBuffer,
        stamp: u32,
    },
}

/// The rasterisers: the visibility pass (`vs_visibility` + `fs_visibility`), the software
/// visibility compute pipeline (entry `main`), the depth-only cascade path (`vs_depth`, the
/// software depth compute pipeline and the fullscreen `shadow_merge` merge pipeline) and the
/// 16-byte placeholder bound in place of a scene buffer that does not exist yet. It owns no
/// frame state.
pub struct Rasterizer {
    layout: wgpu::BindGroupLayout,
    depth_layout: wgpu::BindGroupLayout,
    merge_layout: wgpu::BindGroupLayout,
    hardware: wgpu::RenderPipeline,
    software: wgpu::ComputePipeline,
    hardware_depth: wgpu::RenderPipeline,
    software_depth: wgpu::ComputePipeline,
    merge: wgpu::RenderPipeline,
    placeholder: wgpu::Buffer,
}

impl Rasterizer {
    /// The visibility group-2 layout (0-6 visible to vertex, fragment and compute; 7-8
    /// read-write storage visible to fragment and compute only), the shared depth layout
    /// (0-6 as `raster_common` visible to vertex and compute, 7 the stamped depth write
    /// visible to compute only), the merge layout (0 read-only stamped depth, 1 `PassView`),
    /// the five pipelines and the placeholder buffer.
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

        let software_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-vg software raster"),
            source: wgpu::ShaderSource::Wgsl(raster_sw_module_wgsl().into()),
        });
        let software = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raa-vg software raster"),
            layout: Some(&pipeline_layout),
            module: &software_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Cascades have no `VisibilityBuffer`, so their group 2 is the shared depth layout:
        // the scene and cull bindings 0-6 plus the stamped depth buffer at 7. Only the
        // software depth pass reads 7, and only a compute stage may write storage.
        let cascade_stages = wgpu::ShaderStages::VERTEX.union(wgpu::ShaderStages::COMPUTE);
        let pass_view = || wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: wgpu::BufferSize::new(crate::culling::PASS_VIEW_SIZE as u64),
        };
        let depth_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-vg cascade depth"),
            entries: &[
                entry(0, cascade_stages, pass_view()),
                entry(1, cascade_stages, storage(true)),
                entry(2, cascade_stages, storage(true)),
                entry(3, cascade_stages, storage(true)),
                entry(4, cascade_stages, storage(true)),
                entry(5, cascade_stages, storage(true)),
                entry(6, cascade_stages, storage(true)),
                entry(7, wgpu::ShaderStages::COMPUTE, storage(false)),
            ],
        });
        let merge_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-vg cascade merge"),
            entries: &[
                entry(0, wgpu::ShaderStages::FRAGMENT, storage(true)),
                entry(1, wgpu::ShaderStages::FRAGMENT, pass_view()),
            ],
        });
        let depth_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("raa-vg cascade depth"),
                bind_group_layouts: &[None, None, Some(&depth_layout)],
                immediate_size: 0,
            });
        let hardware_depth = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("raa-vg cascade hardware raster"),
            layout: Some(&depth_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_depth"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                // Facing is decided per instance in `raster_hw.wgsl`, as for visibility.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            // No fragment: the depth is all a cascade needs from the hardware list.
            fragment: None,
            multiview_mask: None,
            cache: None,
        });

        let software_depth_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-vg cascade software depth"),
            source: wgpu::ShaderSource::Wgsl(raster_sw_depth_module_wgsl().into()),
        });
        let software_depth = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raa-vg cascade software depth"),
            layout: Some(&depth_pipeline_layout),
            module: &software_depth_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let merge_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-vg cascade merge"),
            source: wgpu::ShaderSource::Wgsl(shadow_merge_module_wgsl().into()),
        });
        let merge_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("raa-vg cascade merge"),
                bind_group_layouts: &[None, None, Some(&merge_layout)],
                immediate_size: 0,
            });
        let merge = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("raa-vg cascade merge"),
            layout: Some(&merge_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &merge_module,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            // The depth is the only output; the shader writes `frag_depth` per pixel.
            fragment: Some(wgpu::FragmentState {
                module: &merge_module,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[],
            }),
            multiview_mask: None,
            cache: None,
        });

        Rasterizer {
            layout,
            depth_layout,
            merge_layout,
            hardware,
            software,
            hardware_depth,
            software_depth,
            merge,
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

    /// Binds a scene, a culling run's output and one cascade's stamped depth buffer at group 2
    /// over the shared depth layout. A scene buffer that has not been uploaded is replaced by
    /// the placeholder, so the group is always valid.
    ///
    /// Handoff: `Rasterizer::depth_bindings`, `Rasterizer::merge_bindings`,
    /// `Rasterizer::record_software_depth`, `Rasterizer::merge_depth` and `DepthBuffer::new`
    /// (crates/raa-vg/src/raster.rs) are exposed to card 'raa-vg: shadow.rs cascade views and
    /// cascade recording', whose `CascadeWork` builds these two bind groups per cascade and
    /// records the hardware depth draw, the software depth dispatch and the merge over the
    /// cascade layer. State: the cascade `PassView` slot of [`ViewSlots`], the `CullWork`
    /// indirect extents at `CullWork::draw_args_offset()`, `software_args_offset()` and
    /// `merge_args_offset()`, and the [`DepthBuffer`] it binds at 7. The software depth module
    /// itself is the upstream handoff at `raster_sw_depth_module_wgsl`; this method only
    /// supplies its bind group.
    pub fn depth_bindings(
        &self,
        gpu: &GpuContext,
        slots: &ViewSlots,
        slot: u32,
        scene: &VgScene,
        work: &CullWork,
        depth: &DepthBuffer,
    ) -> RasterBindings {
        let instances = scene.instances().unwrap_or(&self.placeholder);
        let clusters = scene.clusters().unwrap_or(&self.placeholder);
        let vertices = scene.vertices().unwrap_or(&self.placeholder);
        let triangles = scene.triangles().unwrap_or(&self.placeholder);
        let group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-vg cascade depth"),
            layout: &self.depth_layout,
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
                bind_entry(7, depth.buffer()),
            ],
        });
        RasterBindings { group, slot }
    }

    /// Binds one cascade's stamped depth buffer and `PassView` at group 2 over the merge
    /// layout: 0 read-only depth, 1 the `PassView` uniform at `slot`.
    pub fn merge_bindings(
        &self,
        gpu: &GpuContext,
        slots: &ViewSlots,
        slot: u32,
        depth: &DepthBuffer,
    ) -> RasterBindings {
        let group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-vg cascade merge"),
            layout: &self.merge_layout,
            entries: &[
                bind_entry(0, depth.buffer()),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: slots.binding(),
                },
            ],
        });
        RasterBindings { group, slot }
    }

    /// Draws what the most recent cull of `work` selected into `target`: the hardware list as
    /// one indirect draw, then the software rasteriser of the small clusters as one indirect
    /// compute dispatch with the same group 2 at `CullWork::software_args_offset()`.
    ///
    /// Handoff: `Rasterizer::record(&self, encoder: &mut wgpu::CommandEncoder, bindings:
    /// &RasterBindings, work: &CullWork, target: RasterTarget<'_>) -> ()` and
    /// `Rasterizer::bindings(&self, gpu: &GpuContext, slots: &ViewSlots, slot: u32, scene:
    /// &VgScene, work: &CullWork, vis: &VisibilityBuffer) -> RasterBindings`
    /// (crates/raa-vg/src/raster.rs) are exposed to card 'raa-vg: software rasterisation of small
    /// clusters in the frame', whose software dispatch this method now records behind the same
    /// group 2, and to `frame_passes.rs`, which builds the bindings and calls `record`. State:
    /// the group-2 `RasterBindings` bind group with its dynamic `PassView` slot and the `CullWork`
    /// indirect arguments; no keys.
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &RasterBindings,
        work: &CullWork,
        target: RasterTarget<'_>,
    ) {
        match target {
            RasterTarget::Visibility(vis) => {
                {
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
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("raa-vg software raster"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.software);
                pass.set_bind_group(2, &bindings.group, &[ViewSlots::offset(bindings.slot)]);
                pass.dispatch_workgroups_indirect(work.args(), CullWork::software_args_offset());
            }
            RasterTarget::Cascade { layer, .. } => {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("raa-vg cascade raster"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: layer,
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
                pass.set_pipeline(&self.hardware_depth);
                pass.set_bind_group(2, &bindings.group, &[ViewSlots::offset(bindings.slot)]);
                pass.draw_indirect(work.args(), CullWork::draw_args_offset());
            }
        }
    }

    /// Rasterizes the clusters `work` sent to software as depth alone, under the view's stamp
    /// in the bound [`DepthBuffer`], at `CullWork::software_args_offset()`.
    pub fn record_software_depth(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &RasterBindings,
        work: &CullWork,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raa-vg cascade software depth"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.software_depth);
        pass.set_bind_group(2, &bindings.group, &[ViewSlots::offset(bindings.slot)]);
        pass.dispatch_workgroups_indirect(work.args(), CullWork::software_args_offset());
    }

    /// Lays the bound [`DepthBuffer`]'s stamped words over one cascade `layer` as a fullscreen
    /// draw, at `CullWork::merge_args_offset()`; the depth test keeps only the nearer samples.
    pub fn merge_depth(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &RasterBindings,
        work: &CullWork,
        layer: &wgpu::TextureView,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("raa-vg cascade merge"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: layer,
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
        pass.set_pipeline(&self.merge);
        pass.set_bind_group(2, &bindings.group, &[ViewSlots::offset(bindings.slot)]);
        pass.draw_indirect(work.args(), CullWork::merge_args_offset());
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
        // `Rasterizer::new` created every pipeline in that scope, including the visibility
        // pair whose modules carry these entry points.
        let module = wgpu::naga::front::wgsl::parse_str(&raster_hw_module_wgsl())
            .expect("hardware raster module");
        for name in ["vs_visibility", "fs_visibility"] {
            assert!(module.entry_points.iter().any(|entry| entry.name == name));
        }
        let module = wgpu::naga::front::wgsl::parse_str(&raster_sw_module_wgsl())
            .expect("software raster module");
        assert!(module.entry_points.iter().any(|entry| entry.name == "main"));
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

    /// The cascade depth path on a software adapter: `Rasterizer::new` creates all five
    /// pipelines without a validation error, a 256 [`DepthBuffer`] is `256 * 256 * 4` bytes,
    /// and the hardware depth draw, the software depth dispatch and the merge record into one
    /// layer of a 4-layer cascade map without a validation error.
    ///
    /// Inputs, all initialised here because no other pass of the frame exists: the cascade map
    /// is a 4-layer 256x256 `Depth32Float` array and its target layer is cleared to 1.0 (far),
    /// as `shadow.csm` leaves it; the software depth buffer is cleared to never-drawn; the bind
    /// groups come from an uploaded one-triangle [`VgScene`], never from the 16-byte
    /// placeholders; the cascade [`PassView`] sits at `CASCADE_BASE` with an orthographic
    /// identity projection over the 256 m map. The [`CullWork`] is empty, so the indirect draws
    /// and the software dispatch enqueue no work.
    #[test]
    fn cascade_depth_path_records_without_validation_error() {
        use crate::culling::{CASCADE_BASE, PassView};
        use crate::scene::{INSTANCE_CASTS_SHADOW, MeshKeys, VgInstanceRecord, hierarchy_word};
        use raa_cluster::{BuildOptions, ClusterStore, OwnedClusterInput, bake};
        use raa_core::MeshId;
        use raa_math::{Mat4, Vec3};

        let test = "cascade_depth_path_records_without_validation_error";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let device = gpu.device();

        let input = OwnedClusterInput::single_triangle();
        let mesh = bake(input.as_input(), &BuildOptions::default()).expect("the triangle bakes");
        let quant = [
            mesh.quant_origin[0],
            mesh.quant_origin[1],
            mesh.quant_origin[2],
            mesh.quant_step,
        ];
        let uv_quant = [
            mesh.uv_origin[0],
            mesh.uv_origin[1],
            mesh.uv_step[0],
            mesh.uv_step[1],
        ];
        let bounds = [
            mesh.bounds_center[0],
            mesh.bounds_center[1],
            mesh.bounds_center[2],
            mesh.bounds_radius,
        ];
        let mut store = ClusterStore::new();
        store.attach(1, vec![mesh]);
        let mut keys = MeshKeys::default();
        keys.set(1, &[MeshId(0)]);
        let mut scene = VgScene::new();
        assert!(scene.upload_pool(&gpu, &store, &keys), "the pool uploads");
        let range = scene.section_range(1, 0).expect("the section is packed");
        let (_, hierarchy, root) = scene.section_header(1, 0).expect("the header exists");
        scene.upload_instances(
            &gpu,
            &[VgInstanceRecord {
                model: Mat4::IDENTITY,
                previous_rows: [[0.0; 4]; 3],
                quant,
                uv_quant,
                ranges: [
                    range.node_base,
                    range.cluster_base,
                    range.vertex_base,
                    range.triangle_base,
                ],
                ids: [0, 0, INSTANCE_CASTS_SHADOW, hierarchy_word(hierarchy, root)],
                bounds,
            }],
        );
        assert_eq!(scene.instance_count(), 1);

        let slots = ViewSlots::new(&gpu);
        let view = PassView::orthographic(Mat4::IDENTITY, Vec3::ZERO, [256.0, 256.0], 1.0, 1.0);
        slots.write(&gpu, CASCADE_BASE, &view);

        let map = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("raa-vg cascade test map"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 4,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let layer = map.create_view(&wgpu::TextureViewDescriptor {
            label: Some("raa-vg cascade test layer"),
            format: None,
            dimension: Some(wgpu::TextureViewDimension::D2),
            usage: None,
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 1,
            array_layer_count: Some(1),
        });

        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let rasterizer = Rasterizer::new(&gpu);
        let depth = DepthBuffer::new(&gpu, 256);
        assert_eq!(depth.resolution(), 256);
        assert_eq!(depth.allocated_bytes(), 256 * 256 * 4);
        let pipelines = CullPipelines::new(&gpu);
        let work = CullWork::new(&gpu, &pipelines, "raa-vg cascade test", 1024, 1024);
        let depth_bindings =
            rasterizer.depth_bindings(&gpu, &slots, CASCADE_BASE, &scene, &work, &depth);
        let merge_bindings = rasterizer.merge_bindings(&gpu, &slots, CASCADE_BASE, &depth);

        let mut encoder = device.create_command_encoder(&Default::default());
        depth.clear(&mut encoder);
        {
            let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("raa-vg cascade test clear"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &layer,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        rasterizer.record(
            &mut encoder,
            &depth_bindings,
            &work,
            RasterTarget::Cascade {
                layer: &layer,
                depth: &depth,
                stamp: 1,
            },
        );
        rasterizer.record_software_depth(&mut encoder, &depth_bindings, &work);
        rasterizer.merge_depth(&mut encoder, &merge_bindings, &work, &layer);
        gpu.queue().submit(Some(encoder.finish()));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        let error = block_on(errors.pop());
        assert!(error.is_none(), "cascade validation: {error:?}");
    }
}
