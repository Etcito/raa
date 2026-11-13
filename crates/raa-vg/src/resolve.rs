//! The S32 resolve module of RAA's virtualized geometry: the WGSL module builder for
//! `vg.resolve`, the [`VgSurface`] outputs, the 320-byte `VgFrameGpu` writer, the [`Resolve`]
//! compute pipeline and the surface bind helpers of `vg.gbuffer` and `vg.shade`. Adapted from
//! solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-render/shaders/vg/resolve.wgsl` and the surface, frame and resolve parts of
//! `crates/solstice-render/src/vg/resolve.rs`, revision 7e465b5. The wetness include, the
//! waterline bindings, the material texture tiers and the emissive output are removed; the
//! resolve binds one group at index 2 with the eight read-only buffers and four write storage
//! textures of contract `vg-resolve-module`.

use crate::culling::{CullWork, PASS_VIEW_WGSL};
use crate::raster::VisibilityBuffer;
use crate::scene::VgScene;
use raa_core::{GpuContext, wgpu};
use raa_math::Mat4;

const RESOLVE_WGSL: &str = include_str!("wgsl/resolve.wgsl");

/// The resolve module: `raa_shader::DEPTH_WGSL` + [`PASS_VIEW_WGSL`] + `resolve.wgsl`, with
/// entry point `main` at `@workgroup_size(8, 8)` and group-2 bindings 0-12 (a 320 B `VgFrame`
/// uniform, the eight read-only scene and visibility buffers, then the four write storage
/// textures). Handoff to the `VgSurface`/`Resolve` card: build the compute pipeline from
/// `main` and bind that list.
pub fn resolve_module_wgsl() -> String {
    [raa_shader::DEPTH_WGSL, PASS_VIEW_WGSL, RESOLVE_WGSL].concat()
}

/// Bytes of one `VgFrameGpu` uniform record.
pub const VG_FRAME_GPU_SIZE: usize = 320;

/// The CPU twin of `VgFrame` in `wgsl/resolve.wgsl`: four matrices, the viewport and camera
/// vectors, and the debug-view and history words of [`write_vg_frame`].
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct VgFrameParams {
    pub view_proj: Mat4,
    pub unjittered_view_proj: Mat4,
    pub previous_view_proj: Mat4,
    pub inv_view_proj: Mat4,
    pub viewport: [f32; 4],
    pub camera_jitter: [f32; 4],
    pub camera_pos: [f32; 4],
    pub debug_view: u32,
    pub history_valid: bool,
}

/// Writes the S32 `VgFrameGpu` record little-endian: `view_proj` @0, `unjittered_view_proj`
/// @64, `previous_view_proj` @128, `inv_view_proj` @192, `viewport` @256, `camera_jitter`
/// @272, `camera_pos` @288 and `flags` @304 = (debug view, history valid, 0, 0).
pub fn write_vg_frame(out: &mut [u8; VG_FRAME_GPU_SIZE], params: &VgFrameParams) {
    params.view_proj.write_le_bytes(&mut out[0..64]);
    params
        .unjittered_view_proj
        .write_le_bytes(&mut out[64..128]);
    params.previous_view_proj.write_le_bytes(&mut out[128..192]);
    params.inv_view_proj.write_le_bytes(&mut out[192..256]);
    put_floats(out, 256, &params.viewport);
    put_floats(out, 272, &params.camera_jitter);
    put_floats(out, 288, &params.camera_pos);
    put_u32(out, 304, params.debug_view);
    put_u32(out, 308, u32::from(params.history_valid));
    put_u32(out, 312, 0);
    put_u32(out, 316, 0);
}

fn put_floats(out: &mut [u8], at: usize, values: &[f32; 4]) {
    for (k, value) in values.iter().enumerate() {
        let start = at + 4 * k;
        out[start..start + 4].copy_from_slice(&value.to_le_bytes());
    }
}

fn put_u32(out: &mut [u8], at: usize, value: u32) {
    out[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

const IDS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const UV_MOTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
const ALBEDO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// One storage texture of `format` with a texture view, at least one texel per axis.
fn surface_view(
    gpu: &GpuContext,
    label: &str,
    size: (u32, u32),
    format: wgpu::TextureFormat,
) -> wgpu::TextureView {
    gpu.device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size.0.max(1),
                height: size.1.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

/// The four S32 resolve outputs of one surface size: the instance/material/debug/depth words,
/// the world normal with its facing sign, the uv and S12 motion, and the base or debug colour.
/// They are write storage textures to `vg.resolve` and read textures to `vg.gbuffer` and
/// `vg.shade`; a resize creates a new surface and drops this one.
pub struct VgSurface {
    ids: wgpu::TextureView,
    normal: wgpu::TextureView,
    uv_motion: wgpu::TextureView,
    albedo: wgpu::TextureView,
    size: (u32, u32),
}

impl VgSurface {
    pub fn new(gpu: &GpuContext, size: (u32, u32)) -> VgSurface {
        VgSurface {
            ids: surface_view(gpu, "raa-vg surface ids", size, IDS_FORMAT),
            normal: surface_view(gpu, "raa-vg surface normal", size, NORMAL_FORMAT),
            uv_motion: surface_view(gpu, "raa-vg surface uv motion", size, UV_MOTION_FORMAT),
            albedo: surface_view(gpu, "raa-vg surface albedo", size, ALBEDO_FORMAT),
            size,
        }
    }

    pub fn ids(&self) -> &wgpu::TextureView {
        &self.ids
    }

    pub fn normal(&self) -> &wgpu::TextureView {
        &self.normal
    }

    pub fn uv_motion(&self) -> &wgpu::TextureView {
        &self.uv_motion
    }

    pub fn albedo(&self) -> &wgpu::TextureView {
        &self.albedo
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// The four textures: 16 B of ids, 8 B of normal, 16 B of uv/motion and 4 B of albedo per
    /// pixel.
    pub fn allocated_bytes(&self) -> u64 {
        u64::from(self.size.0) * u64::from(self.size.1) * (16 + 8 + 16 + 4)
    }
}

/// Bytes of one `VgShadeGpu` record (material function id, debug view, 0, 0).
const SHADE_SIZE: u64 = 16;
/// Bytes of the placeholder bound for a scene buffer that does not exist yet.
const PLACEHOLDER_SIZE: u64 = 16;

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

/// The resolve compute pipeline and its group-2 layout, which matches bindings 0-12 of
/// `resolve_module_wgsl`: 0 `VgFrame` uniform, 1-8 read-only storage buffers and 9-12 write
/// storage textures. The 16-byte placeholder stands in for a scene buffer the scene has not
/// uploaded, so the group is always valid.
pub struct Resolve {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    placeholder: wgpu::Buffer,
}

impl Resolve {
    pub fn new(gpu: &GpuContext) -> Resolve {
        let device = gpu.device();
        let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let storage = |read_only: bool| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let texture = |binding: u32, format: wgpu::TextureFormat| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-vg resolve"),
            entries: &[
                entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(VG_FRAME_GPU_SIZE as u64),
                    },
                ),
                entry(1, storage(true)),
                entry(2, storage(true)),
                entry(3, storage(true)),
                entry(4, storage(true)),
                entry(5, storage(true)),
                entry(6, storage(true)),
                entry(7, storage(true)),
                entry(8, storage(true)),
                texture(9, IDS_FORMAT),
                texture(10, NORMAL_FORMAT),
                texture(11, UV_MOTION_FORMAT),
                texture(12, ALBEDO_FORMAT),
            ],
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-vg resolve"),
            source: wgpu::ShaderSource::Wgsl(resolve_module_wgsl().into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raa-vg resolve"),
            bind_group_layouts: &[None, None, Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raa-vg resolve"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Resolve {
            layout,
            pipeline,
            placeholder: buffer(
                gpu,
                "raa-vg resolve placeholder",
                PLACEHOLDER_SIZE,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
        }
    }

    /// Binds the frame, the visibility buffers, a culling run's visible list, the scene pool,
    /// the S10 material records and the surface at group 2. A missing pool buffer is replaced
    /// so the group is always valid: the 16-byte placeholder serves the vertex and triangle
    /// arrays, while `materials` (always at least one 256-byte record) serves an absent
    /// instance or cluster pool, whose runtime-array element the shader sizes above 16 bytes.
    ///
    /// Handoff: `Resolve::bindings(&self, gpu: &GpuContext, frame: &wgpu::Buffer, vis:
    /// &VisibilityBuffer, work: &CullWork, scene: &VgScene, materials: &wgpu::Buffer, surface:
    /// &VgSurface) -> wgpu::BindGroup` and [`Resolve::record`] are exposed to card 'raa-vg:
    /// frame state and the vg.cull, vg.raster, vg.resolve passes (main view, first phase,
    /// hardware)', which writes the `VgFrameGpu` record, binds the scene's material buffer and
    /// records one dispatch per frame. State: the group-2 bind group and the (w, h) dispatch
    /// size; no keys.
    #[allow(clippy::too_many_arguments)]
    pub fn bindings(
        &self,
        gpu: &GpuContext,
        frame: &wgpu::Buffer,
        vis: &VisibilityBuffer,
        work: &CullWork,
        scene: &VgScene,
        materials: &wgpu::Buffer,
        surface: &VgSurface,
    ) -> wgpu::BindGroup {
        let placeholder = &self.placeholder;
        let instances = scene.instances().unwrap_or(materials);
        let clusters = scene.clusters().unwrap_or(materials);
        let vertices = scene.vertices().unwrap_or(placeholder);
        let triangles = scene.triangles().unwrap_or(placeholder);
        gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-vg resolve"),
            layout: &self.layout,
            entries: &[
                bind_entry(0, frame),
                bind_entry(1, vis.depth()),
                bind_entry(2, vis.ids()),
                bind_entry(3, work.visible()),
                bind_entry(4, instances),
                bind_entry(5, clusters),
                bind_entry(6, vertices),
                bind_entry(7, triangles),
                bind_entry(8, materials),
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(surface.ids()),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(surface.normal()),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(surface.uv_motion()),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView(surface.albedo()),
                },
            ],
        })
    }

    /// Dispatches one 8x8 resolve over `size`.
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        group: &wgpu::BindGroup,
        size: (u32, u32),
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raa-vg resolve"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(2, group, &[]);
        pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
    }

    /// The GPU memory the resolve owns: its 16-byte placeholder buffer.
    pub fn allocated_bytes(&self) -> u64 {
        self.placeholder.size()
    }
}

/// Group-2 layout of `VG_SURFACE_WGSL`, read by the fragment stage of `vg.gbuffer` and
/// `vg.shade`: 0 the ids texture, 1-3 the float surface textures (never filtered) and 4 the
/// dynamically offset 16-byte `VgShadeGpu` uniform.
///
/// Handoff: `surface_layout(gpu: &GpuContext) -> wgpu::BindGroupLayout` and
/// [`surface_bind_group`] (crates/raa-vg/src/resolve.rs) are exposed to cards 'raa-vg:
/// vg.gbuffer merge into depth and G-buffer' and 'raa-vg: vg.shade forward contributor and
/// VgShadeGpu slots', which bind the surface and select a `VgShadeGpu` slot with the dynamic
/// offset. State: the surface texture views and the shade slot buffer; no keys.
pub fn surface_layout(gpu: &GpuContext) -> wgpu::BindGroupLayout {
    let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty,
        count: None,
    };
    let texture = |binding: u32, sample_type: wgpu::TextureSampleType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    gpu.device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-vg surface"),
            entries: &[
                texture(0, wgpu::TextureSampleType::Uint),
                texture(1, wgpu::TextureSampleType::Float { filterable: false }),
                texture(2, wgpu::TextureSampleType::Float { filterable: false }),
                texture(3, wgpu::TextureSampleType::Float { filterable: false }),
                entry(
                    4,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(SHADE_SIZE),
                    },
                ),
            ],
        })
}

/// Binds the four surface textures and the first 16-byte `VgShadeGpu` record at group 2 for a
/// layout from [`surface_layout`]. The caller adds its slot stride as the dynamic offset.
pub fn surface_bind_group(
    gpu: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    surface: &VgSurface,
    shade: &wgpu::Buffer,
) -> wgpu::BindGroup {
    gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raa-vg surface"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(surface.ids()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(surface.normal()),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(surface.uv_motion()),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(surface.albedo()),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: shade,
                    offset: 0,
                    size: wgpu::BufferSize::new(SHADE_SIZE),
                }),
            },
        ],
    })
}
