//! GPU resources of the `dof` pass: [`DofPipelines`] (created once at install) and
//! [`DofTargets`] (surface-sized, recreated on resize).
//!
//! Every bind group layout lists the bindings its entry points statically use (both
//! gathers share one, reading both prefiltered layers), so no pass binds a texture it also
//! renders into. The prefilter and composite groups read hdr.src(), so they exist once per
//! texture of the HDR chain (A and B); record picks the one matching `hdr.src_texture()`.
#![allow(dead_code)]

use std::num::NonZeroU64;

use raa_core::GpuContext;
use raa_core::wgpu;
use raa_pipeline::{HDR_FORMAT, Targets};

use crate::shaders::{
    CS_AUTOFOCUS, CS_COC, CS_DILATE, CS_TILES, DOF_COC_WGSL, DOF_GATHER_WGSL, FS_COMPOSITE,
    FS_GATHER_FAR, FS_GATHER_NEAR, FS_PREFILTER, PARAMS_SIZE, TILE_SIZE, VS_MAIN,
};

/// Format of the full-res signed CoC.
const COC_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
/// Format of the tile min/max CoC and of the dilated tiles.
const TILE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;
/// Initial autofocus distance in metres (the `DofDesc` default focus).
const INITIAL_FOCUS_M: f32 = 10.0;

/// Sizes of the pass's targets for a surface: every one at least 1x1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DofSizes {
    /// Full res (CoC, composite).
    pub full: (u32, u32),
    /// `ceil(full / TILE_SIZE)`: tiles and dilated tiles (also the `cs_tiles` dispatch).
    pub tiles: (u32, u32),
    /// `ceil(full / 2)`: prefiltered and gathered layers.
    pub half: (u32, u32),
}

pub(crate) fn dof_sizes(size: (u32, u32)) -> DofSizes {
    let full = (size.0.max(1), size.1.max(1));
    DofSizes {
        full,
        tiles: (full.0.div_ceil(TILE_SIZE), full.1.div_ceil(TILE_SIZE)),
        half: (full.0.div_ceil(2), full.1.div_ceil(2)),
    }
}

fn entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty,
        count: None,
    }
}

fn texture_type(sample_type: wgpu::TextureSampleType) -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type,
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    }
}

fn storage_type(format: wgpu::TextureFormat) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        access: wgpu::StorageTextureAccess::WriteOnly,
        format,
        view_dimension: wgpu::TextureViewDimension::D2,
    }
}

const PARAMS: wgpu::BindingType = wgpu::BindingType::Buffer {
    ty: wgpu::BufferBindingType::Uniform,
    has_dynamic_offset: false,
    min_binding_size: NonZeroU64::new(PARAMS_SIZE),
};
const FOCUS: wgpu::BindingType = wgpu::BindingType::Buffer {
    ty: wgpu::BufferBindingType::Storage { read_only: false },
    has_dynamic_offset: false,
    min_binding_size: NonZeroU64::new(4),
};
const FLOAT: wgpu::TextureSampleType = wgpu::TextureSampleType::Float { filterable: true };
const UNFILTERABLE: wgpu::TextureSampleType = wgpu::TextureSampleType::Float { filterable: false };
const DEPTH: wgpu::TextureSampleType = wgpu::TextureSampleType::Depth;

fn layout(
    device: &wgpu::Device,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    })
}

fn view_entry(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

fn compute(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
    entry: &str,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// Fullscreen-triangle render pipeline of `entry` into `targets` HDR_FORMAT targets.
fn render(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
    entry: &str,
    targets: usize,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    let target = Some(wgpu::ColorTargetState {
        format: HDR_FORMAT,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    });
    let color_targets = vec![target; targets];
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(entry),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(VS_MAIN),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &color_targets,
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// A single-mip 2D texture and its default view.
fn texture(
    device: &wgpu::Device,
    label: &str,
    size: (u32, u32),
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
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
}

/// Pipelines, layouts, sampler, `DofParams` uniform and autofocus buffer, created once.
pub(crate) struct DofPipelines {
    pub autofocus: wgpu::ComputePipeline,
    pub coc: wgpu::ComputePipeline,
    pub tiles: wgpu::ComputePipeline,
    pub dilate: wgpu::ComputePipeline,
    /// MRT: target 0 near, 1 far (both half res).
    pub prefilter: wgpu::RenderPipeline,
    pub gather_near: wgpu::RenderPipeline,
    pub gather_far: wgpu::RenderPipeline,
    pub composite: wgpu::RenderPipeline,
    autofocus_layout: wgpu::BindGroupLayout,
    coc_layout: wgpu::BindGroupLayout,
    tiles_layout: wgpu::BindGroupLayout,
    dilate_layout: wgpu::BindGroupLayout,
    prefilter_layout: wgpu::BindGroupLayout,
    /// Shared by both gathers.
    gather_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    /// Linear clamp (the shaders currently use textureLoad only).
    pub linear: wgpu::Sampler,
    /// `DofParams` (PARAMS_SIZE bytes), written each frame by the pass.
    pub params: wgpu::Buffer,
    /// The autofocus distance in metres (one f32), kept on the GPU across frames.
    pub focus: wgpu::Buffer,
}

impl DofPipelines {
    pub(crate) fn new(gpu: &GpuContext) -> DofPipelines {
        let device = gpu.device();
        let coc_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-dof.coc"),
            source: wgpu::ShaderSource::Wgsl(DOF_COC_WGSL.into()),
        });
        let gather_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-dof.gather"),
            source: wgpu::ShaderSource::Wgsl(DOF_GATHER_WGSL.into()),
        });
        let cs = wgpu::ShaderStages::COMPUTE;
        let fs = wgpu::ShaderStages::FRAGMENT;
        let autofocus_layout = layout(
            device,
            "raa-dof.autofocus",
            &[
                entry(0, cs, PARAMS),
                entry(1, cs, texture_type(DEPTH)),
                entry(2, cs, FOCUS),
            ],
        );
        let coc_layout = layout(
            device,
            "raa-dof.coc",
            &[
                entry(0, cs, PARAMS),
                entry(1, cs, texture_type(DEPTH)),
                entry(2, cs, FOCUS),
                entry(3, cs, storage_type(COC_FORMAT)),
            ],
        );
        let tiles_layout = layout(
            device,
            "raa-dof.tiles",
            &[
                entry(4, cs, texture_type(UNFILTERABLE)),
                entry(5, cs, storage_type(TILE_FORMAT)),
            ],
        );
        let dilate_layout = layout(
            device,
            "raa-dof.dilate",
            &[
                entry(6, cs, texture_type(UNFILTERABLE)),
                entry(7, cs, storage_type(TILE_FORMAT)),
            ],
        );
        let prefilter_layout = layout(
            device,
            "raa-dof.prefilter",
            &[
                entry(1, fs, texture_type(FLOAT)),
                entry(2, fs, texture_type(UNFILTERABLE)),
            ],
        );
        // both gathers read both prefiltered layers (`dof_load_layer` picks one by side)
        let gather_layout = layout(
            device,
            "raa-dof.gather",
            &[
                entry(2, fs, texture_type(UNFILTERABLE)),
                entry(3, fs, texture_type(UNFILTERABLE)),
                entry(5, fs, texture_type(FLOAT)),
                entry(6, fs, texture_type(FLOAT)),
            ],
        );
        let composite_layout = layout(
            device,
            "raa-dof.composite",
            &[
                entry(0, fs, PARAMS),
                entry(1, fs, texture_type(FLOAT)),
                entry(2, fs, texture_type(UNFILTERABLE)),
                entry(4, fs, texture_type(DEPTH)),
                entry(5, fs, texture_type(FLOAT)),
                entry(6, fs, texture_type(FLOAT)),
            ],
        );
        let focus = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-dof.focus"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue()
            .write_buffer(&focus, 0, &INITIAL_FOCUS_M.to_le_bytes());
        DofPipelines {
            autofocus: compute(device, &coc_module, &autofocus_layout, CS_AUTOFOCUS),
            coc: compute(device, &coc_module, &coc_layout, CS_COC),
            tiles: compute(device, &coc_module, &tiles_layout, CS_TILES),
            dilate: compute(device, &coc_module, &dilate_layout, CS_DILATE),
            prefilter: render(device, &gather_module, &prefilter_layout, FS_PREFILTER, 2),
            gather_near: render(device, &gather_module, &gather_layout, FS_GATHER_NEAR, 1),
            gather_far: render(device, &gather_module, &gather_layout, FS_GATHER_FAR, 1),
            composite: render(device, &gather_module, &composite_layout, FS_COMPOSITE, 1),
            autofocus_layout,
            coc_layout,
            tiles_layout,
            dilate_layout,
            prefilter_layout,
            gather_layout,
            composite_layout,
            linear: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("raa-dof.linear_clamp"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("raa-dof.params"),
                size: PARAMS_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            focus,
        }
    }
}

/// The groups that read hdr.src(), for one texture of the HDR chain.
pub(crate) struct DofHdrGroups {
    pub texture: wgpu::Texture,
    pub prefilter: wgpu::BindGroup,
    pub composite: wgpu::BindGroup,
}

/// Surface-sized textures, views and bind groups of the pass, recreated on resize.
pub(crate) struct DofTargets {
    pub sizes: DofSizes,
    _coc: wgpu::Texture,
    _tiles: wgpu::Texture,
    _dilated: wgpu::Texture,
    _near_prefilter: wgpu::Texture,
    _far_prefilter: wgpu::Texture,
    _near_gather: wgpu::Texture,
    _far_gather: wgpu::Texture,
    /// Render targets of the prefilter (MRT 0 and 1) and of the two gathers.
    pub near_prefilter_view: wgpu::TextureView,
    pub far_prefilter_view: wgpu::TextureView,
    pub near_gather_view: wgpu::TextureView,
    pub far_gather_view: wgpu::TextureView,
    pub autofocus_group: wgpu::BindGroup,
    pub coc_group: wgpu::BindGroup,
    pub tiles_group: wgpu::BindGroup,
    pub dilate_group: wgpu::BindGroup,
    /// Both gathers (prefiltered near at 5, far at 6).
    pub gather_group: wgpu::BindGroup,
    /// One entry per texture of the HDR chain (A and B).
    pub hdr: [DofHdrGroups; 2],
}

impl DofTargets {
    pub(crate) fn new(gpu: &GpuContext, p: &DofPipelines, targets: &Targets) -> DofTargets {
        let device = gpu.device();
        let sizes = dof_sizes(targets.size());
        let storage = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
        let attachment =
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let (coc, coc_view) = texture(device, "raa-dof.coc", sizes.full, COC_FORMAT, storage);
        let (tiles, tiles_view) =
            texture(device, "raa-dof.tiles", sizes.tiles, TILE_FORMAT, storage);
        let (dilated, dilated_view) =
            texture(device, "raa-dof.dilated", sizes.tiles, TILE_FORMAT, storage);
        let half = |label: &str| texture(device, label, sizes.half, HDR_FORMAT, attachment);
        let (near_prefilter, near_prefilter_view) = half("raa-dof.near_prefilter");
        let (far_prefilter, far_prefilter_view) = half("raa-dof.far_prefilter");
        let (near_gather, near_gather_view) = half("raa-dof.near_gather");
        let (far_gather, far_gather_view) = half("raa-dof.far_gather");

        let params = || wgpu::BindGroupEntry {
            binding: 0,
            resource: p.params.as_entire_binding(),
        };
        let focus = || wgpu::BindGroupEntry {
            binding: 2,
            resource: p.focus.as_entire_binding(),
        };
        let group =
            |label: &str, layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindGroupEntry<'_>]| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout,
                    entries,
                })
            };
        let depth = targets.depth();
        let autofocus_group = group(
            "raa-dof.autofocus",
            &p.autofocus_layout,
            &[params(), view_entry(1, depth), focus()],
        );
        let coc_group = group(
            "raa-dof.coc",
            &p.coc_layout,
            &[
                params(),
                view_entry(1, depth),
                focus(),
                view_entry(3, &coc_view),
            ],
        );
        let tiles_group = group(
            "raa-dof.tiles",
            &p.tiles_layout,
            &[view_entry(4, &coc_view), view_entry(5, &tiles_view)],
        );
        let dilate_group = group(
            "raa-dof.dilate",
            &p.dilate_layout,
            &[view_entry(6, &tiles_view), view_entry(7, &dilated_view)],
        );
        let gather_group = group(
            "raa-dof.gather",
            &p.gather_layout,
            &[
                view_entry(2, &coc_view),
                view_entry(3, &dilated_view),
                view_entry(5, &near_prefilter_view),
                view_entry(6, &far_prefilter_view),
            ],
        );
        let hdr_groups = |texture: &wgpu::Texture, view: &wgpu::TextureView| DofHdrGroups {
            texture: texture.clone(),
            prefilter: group(
                "raa-dof.prefilter",
                &p.prefilter_layout,
                &[view_entry(1, view), view_entry(2, &coc_view)],
            ),
            composite: group(
                "raa-dof.composite",
                &p.composite_layout,
                &[
                    params(),
                    view_entry(1, view),
                    view_entry(2, &coc_view),
                    view_entry(4, depth),
                    view_entry(5, &near_gather_view),
                    view_entry(6, &far_gather_view),
                ],
            ),
        };
        let hdr = targets.hdr();
        let hdr = [
            hdr_groups(hdr.src_texture(), hdr.src()),
            hdr_groups(hdr.dst_texture(), hdr.dst()),
        ];

        DofTargets {
            sizes,
            _coc: coc,
            _tiles: tiles,
            _dilated: dilated,
            _near_prefilter: near_prefilter,
            _far_prefilter: far_prefilter,
            _near_gather: near_gather,
            _far_gather: far_gather,
            near_prefilter_view,
            far_prefilter_view,
            near_gather_view,
            far_gather_view,
            autofocus_group,
            coc_group,
            tiles_group,
            dilate_group,
            gather_group,
            hdr,
        }
    }

    /// The groups reading `src` (hdr.src_texture()), or None if the chain was recreated
    /// since the last resize.
    pub(crate) fn hdr_groups(&self, src: &wgpu::Texture) -> Option<&DofHdrGroups> {
        self.hdr.iter().find(|g| g.texture == *src)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_of_720p_and_of_a_1x1_target() {
        let s = dof_sizes((1280, 720));
        assert_eq!(s.full, (1280, 720));
        assert_eq!(s.tiles, (80, 45));
        assert_eq!(s.half, (640, 360));
        let s = dof_sizes((1, 1));
        assert_eq!((s.full, s.tiles, s.half), ((1, 1), (1, 1), (1, 1)));
        let s = dof_sizes((0, 0));
        assert_eq!((s.full, s.tiles, s.half), ((1, 1), (1, 1), (1, 1)));
        assert_eq!(dof_sizes((17, 3)).tiles, (2, 1));
        assert_eq!(dof_sizes((17, 3)).half, (9, 2));
    }
}
