//! GPU resources of the `ssr` pass: [`SsrPipelines`] (created once, with the lighting
//! layout) and [`SsrTargets`] (surface-sized, recreated on resize).
//!
//! Hi-Z: one r32float texture with `hiz::mip_count` levels; level 0 is written by
//! `cs_hiz_copy` from `depth`, level n + 1 by `cs_hiz_reduce` from level n, each with its
//! `HizParams` in its own 256-byte slot of one uniform buffer. HDR mip chain: HDR_FORMAT
//! down to 1x1; mip 0 receives a copy of hdr.src() (so it is COPY_DST too) and mip k + 1 is
//! drawn by `fs_downsample` from mip k. The trace and composite bind groups only read the
//! mip chain (never hdr.src() itself), so one of each serves both textures of the HDR chain.
#![allow(dead_code)]

use std::num::NonZeroU64;

use raa_core::GpuContext;
use raa_core::wgpu;
use raa_pipeline::{HDR_FORMAT, Targets};

use crate::hiz;
use crate::shaders::{
    CS_HIZ_COPY, CS_HIZ_REDUCE, FS_COMPOSITE, FS_DOWNSAMPLE, FS_TRACE, HIZ_PARAMS_SIZE,
    HIZ_WORKGROUP, SSR_HIZ_WGSL, SSR_PARAMS_SIZE, SSR_TRACE_WGSL, VS_FULLSCREEN,
};

/// Byte stride of one level's `HizParams` (the largest min_uniform_buffer_offset_alignment
/// WebGPU allows, so a multiple of every device's).
const SLOT_STRIDE: u64 = 256;
/// Format of the Hi-Z pyramid.
const HIZ_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
/// Format of the trace result (hit uv, confidence, cone mip).
pub(crate) const TRACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// One Hi-Z dispatch: level 0 is the copy (src = dst = depth size), level n + 1 the reduce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HizLevel {
    pub src: (u32, u32),
    pub dst: (u32, u32),
    /// Workgroups along x and y: `ceil(dst / HIZ_WORKGROUP)`.
    pub groups: (u32, u32),
}

/// The dispatches building the pyramid of a `size` depth target, in order.
pub(crate) fn hiz_levels(size: (u32, u32)) -> Vec<HizLevel> {
    let groups = |(w, h): (u32, u32)| (w.div_ceil(HIZ_WORKGROUP), h.div_ceil(HIZ_WORKGROUP));
    let size = (size.0.max(1), size.1.max(1));
    let count = hiz::mip_count(size.0, size.1);
    let mut levels = Vec::with_capacity(count as usize);
    let mut src = size;
    levels.push(HizLevel {
        src,
        dst: src,
        groups: groups(src),
    });
    for _ in 1..count {
        let dst = hiz::next_size(src.0, src.1);
        levels.push(HizLevel {
            src,
            dst,
            groups: groups(dst),
        });
        src = dst;
    }
    levels
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

fn uniform_type(size: u64) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: NonZeroU64::new(size),
    }
}

fn texture_type(sample_type: wgpu::TextureSampleType) -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type,
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    }
}

const FLOAT: wgpu::TextureSampleType = wgpu::TextureSampleType::Float { filterable: true };
const UNFILTERABLE: wgpu::TextureSampleType = wgpu::TextureSampleType::Float { filterable: false };
const FILTERING: wgpu::BindingType =
    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering);

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

/// Fullscreen-triangle render pipeline of `entry` into one `format` target.
fn render(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layouts: &[Option<&wgpu::BindGroupLayout>],
    entry: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry),
        bind_group_layouts: layouts,
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(entry),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(VS_FULLSCREEN),
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
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn texture(
    device: &wgpu::Device,
    label: &str,
    size: (u32, u32),
    mips: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

/// One single-mip view per level of `texture`.
fn mip_views(texture: &wgpu::Texture, label: &str) -> Vec<wgpu::TextureView> {
    (0..texture.mip_level_count())
        .map(|level| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(label),
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            })
        })
        .collect()
}

/// Pipelines, layouts, sampler and `SsrParams` uniform, created once at install.
pub(crate) struct SsrPipelines {
    pub hiz_copy: wgpu::ComputePipeline,
    pub hiz_reduce: wgpu::ComputePipeline,
    pub downsample: wgpu::RenderPipeline,
    /// Layout [group 0 = `ssr_layout`, group 1 = the lighting layout] for both.
    pub trace: wgpu::RenderPipeline,
    pub composite: wgpu::RenderPipeline,
    copy_layout: wgpu::BindGroupLayout,
    reduce_layout: wgpu::BindGroupLayout,
    down_layout: wgpu::BindGroupLayout,
    ssr_layout: wgpu::BindGroupLayout,
    /// Linear clamp, for the HDR mip chain.
    linear: wgpu::Sampler,
    /// `SsrParams` (SSR_PARAMS_SIZE bytes), written each frame by the pass.
    pub params: wgpu::Buffer,
    /// 1x1 TRACE_FORMAT texture bound at binding 6 while fs_trace renders the trace.
    _dummy: wgpu::Texture,
    dummy_view: wgpu::TextureView,
}

impl SsrPipelines {
    pub(crate) fn new(gpu: &GpuContext, lighting: &wgpu::BindGroupLayout) -> SsrPipelines {
        let device = gpu.device();
        let hiz_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-ssr.hiz"),
            source: wgpu::ShaderSource::Wgsl(SSR_HIZ_WGSL.into()),
        });
        let trace_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-ssr.trace"),
            source: wgpu::ShaderSource::Wgsl(SSR_TRACE_WGSL.into()),
        });
        let cs = wgpu::ShaderStages::COMPUTE;
        let fs = wgpu::ShaderStages::FRAGMENT;
        let storage = wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: HIZ_FORMAT,
            view_dimension: wgpu::TextureViewDimension::D2,
        };
        let params = entry(0, cs, uniform_type(HIZ_PARAMS_SIZE));
        let copy_layout = layout(
            device,
            "raa-ssr.hiz_copy",
            &[
                params,
                entry(1, cs, texture_type(wgpu::TextureSampleType::Depth)),
                entry(3, cs, storage),
            ],
        );
        let reduce_layout = layout(
            device,
            "raa-ssr.hiz_reduce",
            &[
                params,
                entry(2, cs, texture_type(UNFILTERABLE)),
                entry(3, cs, storage),
            ],
        );
        let down_layout = layout(
            device,
            "raa-ssr.downsample",
            &[entry(4, fs, texture_type(FLOAT)), entry(5, fs, FILTERING)],
        );
        let ssr_layout = layout(
            device,
            "raa-ssr.ssr",
            &[
                entry(0, fs, uniform_type(SSR_PARAMS_SIZE)),
                entry(1, fs, texture_type(wgpu::TextureSampleType::Depth)),
                entry(2, fs, texture_type(FLOAT)),
                entry(3, fs, texture_type(FLOAT)),
                entry(4, fs, texture_type(UNFILTERABLE)),
                entry(5, fs, texture_type(FLOAT)),
                entry(6, fs, texture_type(FLOAT)),
                entry(7, fs, FILTERING),
            ],
        );
        let ssr_layouts = [Some(&ssr_layout), Some(lighting)];
        let dummy = texture(
            device,
            "raa-ssr.dummy",
            (1, 1),
            1,
            TRACE_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let dummy_view = dummy.create_view(&wgpu::TextureViewDescriptor::default());
        SsrPipelines {
            hiz_copy: compute(device, &hiz_module, &copy_layout, CS_HIZ_COPY),
            hiz_reduce: compute(device, &hiz_module, &reduce_layout, CS_HIZ_REDUCE),
            downsample: render(
                device,
                &hiz_module,
                &[Some(&down_layout)],
                FS_DOWNSAMPLE,
                HDR_FORMAT,
            ),
            trace: render(device, &trace_module, &ssr_layouts, FS_TRACE, TRACE_FORMAT),
            composite: render(
                device,
                &trace_module,
                &ssr_layouts,
                FS_COMPOSITE,
                HDR_FORMAT,
            ),
            copy_layout,
            reduce_layout,
            down_layout,
            ssr_layout,
            linear: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("raa-ssr.linear_clamp"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            }),
            params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("raa-ssr.params"),
                size: SSR_PARAMS_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            _dummy: dummy,
            dummy_view,
        }
    }
}

/// Surface-sized textures, views and bind groups of the pass, recreated on resize.
pub(crate) struct SsrTargets {
    pub size: (u32, u32),
    /// Dispatch k writes Hi-Z level k with `hiz_groups[k]` (0 = copy, else reduce).
    pub hiz_levels: Vec<HizLevel>,
    pub hiz_groups: Vec<wgpu::BindGroup>,
    _hiz: wgpu::Texture,
    _hiz_views: Vec<wgpu::TextureView>,
    /// `HizParams` of level k at `k * SLOT_STRIDE`.
    _hiz_params: wgpu::Buffer,
    /// HDR mip chain; mip 0 is the copy destination of hdr.src().
    pub hdr_mips: wgpu::Texture,
    /// Single-mip views: draw k of the downsample renders into `hdr_views[k + 1]` with
    /// `down_groups[k]` (which reads mip k).
    pub hdr_views: Vec<wgpu::TextureView>,
    pub down_groups: Vec<wgpu::BindGroup>,
    _trace: wgpu::Texture,
    pub trace_view: wgpu::TextureView,
    /// Group 0 of fs_trace (dummy at binding 6) and of fs_composite (trace at 6).
    pub trace_group: wgpu::BindGroup,
    pub composite_group: wgpu::BindGroup,
}

impl SsrTargets {
    pub(crate) fn new(gpu: &GpuContext, p: &SsrPipelines, targets: &Targets) -> SsrTargets {
        let device = gpu.device();
        let size = targets.size();
        let size = (size.0.max(1), size.1.max(1));
        let mips = hiz::mip_count(size.0, size.1);
        let levels = hiz_levels(size);

        let usage = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
        let hiz_tex = texture(device, "raa-ssr.hiz", size, mips, HIZ_FORMAT, usage);
        let hiz_views = mip_views(&hiz_tex, "raa-ssr.hiz_level");
        let hiz_full = hiz_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let hiz_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-ssr.hiz_params"),
            size: SLOT_STRIDE * levels.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut bytes = vec![0u8; (SLOT_STRIDE as usize) * levels.len()];
        for (k, l) in levels.iter().enumerate() {
            let at = k * SLOT_STRIDE as usize;
            for (i, v) in [l.src.0, l.src.1, l.dst.0, l.dst.1].into_iter().enumerate() {
                bytes[at + 4 * i..at + 4 * i + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        gpu.queue().write_buffer(&hiz_params, 0, &bytes);
        let hiz_groups = (0..levels.len())
            .map(|k| {
                let params = wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &hiz_params,
                        offset: k as u64 * SLOT_STRIDE,
                        size: NonZeroU64::new(HIZ_PARAMS_SIZE),
                    }),
                };
                let (label, layout, src) = if k == 0 {
                    (
                        "raa-ssr.hiz_copy",
                        &p.copy_layout,
                        view_entry(1, targets.depth()),
                    )
                } else {
                    (
                        "raa-ssr.hiz_reduce",
                        &p.reduce_layout,
                        view_entry(2, &hiz_views[k - 1]),
                    )
                };
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout,
                    entries: &[params, src, view_entry(3, &hiz_views[k])],
                })
            })
            .collect();

        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST;
        let hdr_mips = texture(device, "raa-ssr.hdr_mips", size, mips, HDR_FORMAT, usage);
        let hdr_views = mip_views(&hdr_mips, "raa-ssr.hdr_mip");
        let hdr_full = hdr_mips.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = wgpu::BindGroupEntry {
            binding: 5,
            resource: wgpu::BindingResource::Sampler(&p.linear),
        };
        let down_groups = hdr_views[..hdr_views.len() - 1]
            .iter()
            .map(|src| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("raa-ssr.downsample"),
                    layout: &p.down_layout,
                    entries: &[view_entry(4, src), sampler.clone()],
                })
            })
            .collect();

        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let trace = texture(device, "raa-ssr.trace", size, 1, TRACE_FORMAT, usage);
        let trace_view = trace.create_view(&wgpu::TextureViewDescriptor::default());
        let ssr_group = |label: &str, at6: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &p.ssr_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: p.params.as_entire_binding(),
                    },
                    view_entry(1, targets.depth()),
                    view_entry(2, targets.normal()),
                    view_entry(3, targets.material()),
                    view_entry(4, &hiz_full),
                    view_entry(5, &hdr_full),
                    view_entry(6, at6),
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::Sampler(&p.linear),
                    },
                ],
            })
        };
        let trace_group = ssr_group("raa-ssr.trace", &p.dummy_view);
        let composite_group = ssr_group("raa-ssr.composite", &trace_view);

        SsrTargets {
            size,
            hiz_levels: levels,
            hiz_groups,
            _hiz: hiz_tex,
            _hiz_views: hiz_views,
            _hiz_params: hiz_params,
            hdr_mips,
            hdr_views,
            down_groups,
            _trace: trace,
            trace_view,
            trace_group,
            composite_group,
        }
    }

    /// Mip count of the Hi-Z pyramid and of the HDR mip chain (both down to 1x1).
    pub(crate) fn mip_count(&self) -> u32 {
        self.hiz_levels.len() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hiz_dispatches_cover_every_level_of_720p() {
        let levels = hiz_levels((1280, 720));
        assert_eq!(levels.len() as u32, hiz::mip_count(1280, 720));
        assert_eq!(levels.len(), 11);
        assert_eq!(levels[0].src, (1280, 720));
        assert_eq!(levels[0].dst, (1280, 720));
        for (k, l) in levels.iter().enumerate() {
            if k > 0 {
                assert_eq!(l.src, levels[k - 1].dst, "level {k} reads the previous one");
                assert_eq!(l.dst, hiz::next_size(l.src.0, l.src.1), "level {k} size");
            }
            assert_eq!(l.groups.0, l.dst.0.div_ceil(8), "level {k} x groups");
            assert_eq!(l.groups.1, l.dst.1.div_ceil(8), "level {k} y groups");
            assert!(l.groups.0 * 8 >= l.dst.0 && (l.groups.0 - 1) * 8 < l.dst.0);
        }
        assert_eq!(levels.last().map(|l| l.dst), Some((1, 1)));
        assert_eq!(levels[1].groups, (80, 45));
        assert_eq!(levels[2].groups, (40, 23));
    }

    #[test]
    fn hiz_params_fit_their_slot() {
        assert!(HIZ_PARAMS_SIZE <= SLOT_STRIDE);
    }
}
