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
    HIZ_WORKGROUP, SSR_HIZ_WGSL, SSR_PARAMS_SIZE, VS_FULLSCREEN, ssr_trace_module_wgsl,
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
    formats: &[wgpu::TextureFormat],
    constants: &[(&str, f64)],
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
            compilation_options: wgpu::PipelineCompilationOptions {
                constants,
                ..Default::default()
            },
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants,
                ..Default::default()
            },
            targets: &formats
                .iter()
                .map(|&format| {
                    Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })
                })
                .collect::<Vec<_>>(),
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
    /// Layout [group 0 = `trace_layout`, group 1 = the lighting layout] (fs_trace).
    pub trace: wgpu::RenderPipeline,
    pub hit: wgpu::RenderPipeline,
    pub incident: wgpu::RenderPipeline,
    pub replace: wgpu::RenderPipeline,
    replace_layout: wgpu::BindGroupLayout,
    /// Layout [group 0 = `composite_layout`, group 1 = the lighting layout] (fs_composite).
    pub composite: wgpu::RenderPipeline,
    copy_layout: wgpu::BindGroupLayout,
    reduce_layout: wgpu::BindGroupLayout,
    down_layout: wgpu::BindGroupLayout,
    /// Group 0 of fs_trace: bindings 0, 1, 2 and 4 (3 FRAGMENT sampled textures). fs_trace has
    /// no binding 6, so no dummy texture is bound at it.
    trace_layout: wgpu::BindGroupLayout,
    /// Group 0 of fs_composite: bindings 0, 1, 2, 3, 5, 6 and 7 (5 FRAGMENT sampled textures
    /// and one filtering sampler).
    composite_layout: wgpu::BindGroupLayout,
    /// Linear clamp, for the HDR mip chain.
    linear: wgpu::Sampler,
    /// The trace module and the lighting layout, kept to rebuild `trace` and `composite`.
    trace_module: wgpu::ShaderModule,
    lighting: wgpu::BindGroupLayout,
    /// `SsrParams` (SSR_PARAMS_SIZE bytes), written each frame by the pass.
    pub params: wgpu::Buffer,
}

impl SsrPipelines {
    /// `constants`: the trace module's overrides (`tunables::OVERRIDE_NAMES` with values).
    pub(crate) fn new(
        gpu: &GpuContext,
        lighting: &wgpu::BindGroupLayout,
        constants: &[(&str, f64)],
    ) -> SsrPipelines {
        let device = gpu.device();
        let hiz_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-ssr.hiz"),
            source: wgpu::ShaderSource::Wgsl(SSR_HIZ_WGSL.into()),
        });
        let trace_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-ssr.trace"),
            source: wgpu::ShaderSource::Wgsl(ssr_trace_module_wgsl().into()),
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
        // Each entry point gets only the bindings it statically uses (ssr_trace.wgsl:35-44), so
        // its FRAGMENT sampled-texture count stays within the 16-texture limit with the full
        // group 1 (11 sampled textures): fs_trace 3 + 11 = 14, fs_composite 5 + 11 = 16.
        let trace_layout = layout(
            device,
            "raa-ssr.trace",
            &[
                entry(0, fs, uniform_type(SSR_PARAMS_SIZE)),
                entry(1, fs, texture_type(wgpu::TextureSampleType::Depth)),
                entry(2, fs, texture_type(FLOAT)),
                entry(4, fs, texture_type(UNFILTERABLE)),
            ],
        );
        let composite_layout = layout(
            device,
            "raa-ssr.composite",
            &[
                entry(0, fs, uniform_type(SSR_PARAMS_SIZE)),
                entry(1, fs, texture_type(wgpu::TextureSampleType::Depth)),
                entry(2, fs, texture_type(FLOAT)),
                entry(3, fs, texture_type(FLOAT)),
                entry(5, fs, texture_type(FLOAT)),
                entry(6, fs, texture_type(FLOAT)),
                entry(7, fs, FILTERING),
            ],
        );
        let replace_layout = layout(
            device,
            "raa-ssr.replace",
            &[
                entry(0, fs, uniform_type(SSR_PARAMS_SIZE)),
                entry(1, fs, texture_type(wgpu::TextureSampleType::Depth)),
                entry(2, fs, texture_type(FLOAT)),
                entry(3, fs, texture_type(FLOAT)),
                entry(5, fs, texture_type(FLOAT)),
                entry(6, fs, texture_type(UNFILTERABLE)),
            ],
        );
        let (trace, hit, composite) = trace_pipelines(
            device,
            &trace_module,
            [&trace_layout, &composite_layout, lighting],
            constants,
        );
        SsrPipelines {
            hiz_copy: compute(device, &hiz_module, &copy_layout, CS_HIZ_COPY),
            hiz_reduce: compute(device, &hiz_module, &reduce_layout, CS_HIZ_REDUCE),
            downsample: render(
                device,
                &hiz_module,
                &[Some(&down_layout)],
                FS_DOWNSAMPLE,
                &[HDR_FORMAT],
                &[],
            ),
            hit,
            incident: render(
                device,
                &trace_module,
                &[Some(&composite_layout)],
                "fs_incident",
                &[wgpu::TextureFormat::Rgba32Float],
                &[],
            ),
            replace: render(
                device,
                &trace_module,
                &[Some(&replace_layout), Some(lighting)],
                "fs_replace",
                &[HDR_FORMAT, HDR_FORMAT, wgpu::TextureFormat::Rgba32Float],
                &[],
            ),
            replace_layout,
            trace,
            composite,
            trace_module,
            lighting: lighting.clone(),
            copy_layout,
            reduce_layout,
            down_layout,
            trace_layout,
            composite_layout,
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
        }
    }
}

impl SsrPipelines {
    /// Rebuilds `trace` and `composite` (the pipelines of the trace module) with `constants`.
    pub(crate) fn rebuild_trace(&mut self, device: &wgpu::Device, constants: &[(&str, f64)]) {
        let layouts = [&self.trace_layout, &self.composite_layout, &self.lighting];
        let (trace, hit, _) = trace_pipelines(device, &self.trace_module, layouts, constants);
        self.trace = trace;
        self.hit = hit;
    }
}

/// `fs_trace` and `fs_composite` pipelines; `layouts` = trace, composite, lighting.
fn trace_pipelines(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layouts: [&wgpu::BindGroupLayout; 3],
    constants: &[(&str, f64)],
) -> (
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
) {
    let [trace, composite, lighting] = layouts;
    (
        render(
            device,
            module,
            &[Some(trace), Some(lighting)],
            FS_TRACE,
            &[TRACE_FORMAT],
            constants,
        ),
        render(
            device,
            module,
            &[Some(trace), Some(lighting)],
            "fs_hit",
            &[TRACE_FORMAT, wgpu::TextureFormat::Rgba32Float],
            constants,
        ),
        render(
            device,
            module,
            &[Some(composite), Some(lighting)],
            FS_COMPOSITE,
            &[HDR_FORMAT],
            &[],
        ),
    )
}

/// The `(depth, normal, material)` views the Hi-Z copy, trace and composite groups of an
/// [`SsrTargets`] were built from. `prepare` rebuilds only the groups whose input differs.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SsrViewKey {
    pub depth: wgpu::TextureView,
    pub normal: wgpu::TextureView,
    pub material: wgpu::TextureView,
}

pub(crate) fn replace_group(
    device: &wgpu::Device,
    p: &SsrPipelines,
    hdr: &wgpu::TextureView,
    incident: &wgpu::TextureView,
    views: &SsrViewKey,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raa-ssr.replace"),
        layout: &p.replace_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p.params.as_entire_binding(),
            },
            view_entry(1, &views.depth),
            view_entry(2, &views.normal),
            view_entry(3, &views.material),
            view_entry(5, hdr),
            view_entry(6, incident),
        ],
    })
}

pub(crate) fn diagnostic_view(
    device: &wgpu::Device,
    label: &str,
    size: (u32, u32),
    format: wgpu::TextureFormat,
) -> wgpu::TextureView {
    texture(
        device,
        label,
        size,
        1,
        format,
        wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
    )
    .create_view(&wgpu::TextureViewDescriptor::default())
}

/// Bind group 0 of `fs_trace`: params, depth, normal and the full Hi-Z view.
fn trace_group(
    device: &wgpu::Device,
    p: &SsrPipelines,
    hiz: &wgpu::TextureView,
    views: &SsrViewKey,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raa-ssr.trace"),
        layout: &p.trace_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p.params.as_entire_binding(),
            },
            view_entry(1, &views.depth),
            view_entry(2, &views.normal),
            view_entry(4, hiz),
        ],
    })
}

/// Bind group 0 of `fs_composite`: params, depth, normal, material, the HDR chain and the
/// trace view.
fn composite_group(
    device: &wgpu::Device,
    p: &SsrPipelines,
    hdr: &wgpu::TextureView,
    trace: &wgpu::TextureView,
    views: &SsrViewKey,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raa-ssr.composite"),
        layout: &p.composite_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p.params.as_entire_binding(),
            },
            view_entry(1, &views.depth),
            view_entry(2, &views.normal),
            view_entry(3, &views.material),
            view_entry(5, hdr),
            view_entry(6, trace),
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::Sampler(&p.linear),
            },
        ],
    })
}

/// The Hi-Z level-0 group (`cs_hiz_copy`): `params` slot 0, `depth` sampling and `dst`.
fn hiz_copy_group(
    device: &wgpu::Device,
    p: &SsrPipelines,
    params: &wgpu::Buffer,
    depth: &wgpu::TextureView,
    dst: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raa-ssr.hiz_copy"),
        layout: &p.copy_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: params,
                    offset: 0,
                    size: NonZeroU64::new(HIZ_PARAMS_SIZE),
                }),
            },
            view_entry(1, depth),
            view_entry(3, dst),
        ],
    })
}

/// Hi-Z texture and dispatch resources, independently reusable by the `ssr` pass and stages.
pub(crate) struct HizTargets {
    pub size: (u32, u32),
    pub levels: Vec<HizLevel>,
    pub groups: Vec<wgpu::BindGroup>,
    _texture: wgpu::Texture,
    pub full_view: wgpu::TextureView,
    views: Vec<wgpu::TextureView>,
    params: wgpu::Buffer,
    depth: wgpu::TextureView,
}

impl HizTargets {
    pub(crate) fn new(
        gpu: &GpuContext,
        p: &SsrPipelines,
        size: (u32, u32),
        depth: &wgpu::TextureView,
    ) -> Self {
        let device = gpu.device();
        let size = (size.0.max(1), size.1.max(1));
        let levels = hiz_levels(size);
        let usage = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
        let texture = texture(
            device,
            "raa-ssr.hiz",
            size,
            levels.len() as u32,
            HIZ_FORMAT,
            usage,
        );
        let views = mip_views(&texture, "raa-ssr.hiz_level");
        let full_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-ssr.hiz_params"),
            size: SLOT_STRIDE * levels.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut bytes = vec![0; SLOT_STRIDE as usize * levels.len()];
        for (k, level) in levels.iter().enumerate() {
            for (i, value) in [level.src.0, level.src.1, level.dst.0, level.dst.1]
                .into_iter()
                .enumerate()
            {
                bytes[k * SLOT_STRIDE as usize + 4 * i..k * SLOT_STRIDE as usize + 4 * i + 4]
                    .copy_from_slice(&value.to_le_bytes());
            }
        }
        gpu.queue().write_buffer(&params, 0, &bytes);
        let groups = (0..levels.len())
            .map(|k| {
                if k == 0 {
                    return hiz_copy_group(device, p, &params, depth, &views[0]);
                }
                let params_entry = wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params,
                        offset: k as u64 * SLOT_STRIDE,
                        size: NonZeroU64::new(HIZ_PARAMS_SIZE),
                    }),
                };
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("raa-ssr.hiz_reduce"),
                    layout: &p.reduce_layout,
                    entries: &[
                        params_entry,
                        view_entry(2, &views[k - 1]),
                        view_entry(3, &views[k]),
                    ],
                })
            })
            .collect();
        Self {
            size,
            levels,
            groups,
            _texture: texture,
            full_view,
            views,
            params,
            depth: depth.clone(),
        }
    }

    pub(crate) fn set_depth(
        &mut self,
        device: &wgpu::Device,
        p: &SsrPipelines,
        depth: &wgpu::TextureView,
    ) {
        if self.depth == *depth {
            return;
        }
        self.groups[0] = hiz_copy_group(device, p, &self.params, depth, &self.views[0]);
        self.depth = depth.clone();
    }
}

/// Trace texture and group, independent of the pass's HDR mip chain.
pub(crate) struct TraceTarget {
    _texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub group: wgpu::BindGroup,
    views: SsrViewKey,
    hiz: wgpu::TextureView,
}

impl TraceTarget {
    pub(crate) fn new(
        device: &wgpu::Device,
        p: &SsrPipelines,
        size: (u32, u32),
        hiz: &wgpu::TextureView,
        views: SsrViewKey,
    ) -> Self {
        let texture = texture(
            device,
            "raa-ssr.trace",
            size,
            1,
            TRACE_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let group = trace_group(device, p, hiz, &views);
        Self {
            _texture: texture,
            view,
            group,
            views,
            hiz: hiz.clone(),
        }
    }

    pub(crate) fn set_views(
        &mut self,
        device: &wgpu::Device,
        p: &SsrPipelines,
        hiz: &wgpu::TextureView,
        views: SsrViewKey,
    ) {
        if self.hiz != *hiz || self.views.depth != views.depth || self.views.normal != views.normal
        {
            self.group = trace_group(device, p, hiz, &views);
        }
        self.views = views;
        self.hiz = hiz.clone();
    }
}

/// Surface-sized resources of the whole pass, composed from the shared Hi-Z and trace builders.
pub(crate) struct SsrTargets {
    pub size: (u32, u32),
    pub(crate) views: SsrViewKey,
    pub hiz: HizTargets,
    pub composite: CompositeTargets,
    pub trace: TraceTarget,
}

impl std::ops::Deref for SsrTargets {
    type Target = CompositeTargets;

    fn deref(&self) -> &Self::Target {
        &self.composite
    }
}

/// HDR mip chain and composite group, keyed by its five sampled input views.
pub(crate) struct CompositeTargets {
    pub size: (u32, u32),
    pub hdr_mips: wgpu::Texture,
    pub hdr_full: wgpu::TextureView,
    pub hdr_views: Vec<wgpu::TextureView>,
    pub down_groups: Vec<wgpu::BindGroup>,
    pub composite_group: wgpu::BindGroup,
    views: SsrViewKey,
    trace: wgpu::TextureView,
}

impl CompositeTargets {
    pub(crate) fn new(
        device: &wgpu::Device,
        p: &SsrPipelines,
        size: (u32, u32),
        mips: u32,
        trace: &wgpu::TextureView,
        views: SsrViewKey,
    ) -> Self {
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
        let composite_group = composite_group(device, p, &hdr_full, trace, &views);
        Self {
            size,
            hdr_mips,
            hdr_full,
            hdr_views,
            down_groups,
            composite_group,
            views,
            trace: trace.clone(),
        }
    }

    pub(crate) fn set_views(
        &mut self,
        device: &wgpu::Device,
        p: &SsrPipelines,
        trace: &wgpu::TextureView,
        views: SsrViewKey,
    ) {
        if self.trace != *trace || self.views != views {
            self.composite_group = composite_group(device, p, &self.hdr_full, trace, &views);
            self.trace = trace.clone();
            self.views = views;
        }
    }
}

impl SsrTargets {
    pub(crate) fn new(gpu: &GpuContext, p: &SsrPipelines, targets: &Targets) -> SsrTargets {
        let device = gpu.device();
        let size = targets.size();
        let size = (size.0.max(1), size.1.max(1));
        let views = SsrViewKey {
            depth: targets.depth().clone(),
            normal: targets.normal().clone(),
            material: targets.material().clone(),
        };
        let hiz = HizTargets::new(gpu, p, size, &views.depth);
        let trace = TraceTarget::new(device, p, size, &hiz.full_view, views.clone());
        let mips = hiz.levels.len() as u32;
        let composite = CompositeTargets::new(device, p, size, mips, &trace.view, views.clone());
        SsrTargets {
            size,
            views,
            hiz,
            composite,
            trace,
        }
    }

    pub(crate) fn set_views(&mut self, device: &wgpu::Device, p: &SsrPipelines, views: SsrViewKey) {
        let depth = self.views.depth != views.depth;
        let normal = self.views.normal != views.normal;
        let material = self.views.material != views.material;
        if !(depth || normal || material) {
            return;
        }
        if depth {
            self.hiz.set_depth(device, p, &views.depth);
        }
        if depth || normal {
            self.trace
                .set_views(device, p, &self.hiz.full_view, views.clone());
        }
        if depth || normal || material {
            self.composite
                .set_views(device, p, &self.trace.view, views.clone());
        }
        self.views = views;
    }

    pub(crate) fn mip_count(&self) -> u32 {
        self.hiz.levels.len() as u32
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
        const { assert!(HIZ_PARAMS_SIZE <= SLOT_STRIDE) };
    }

    /// A bound `depth` input enters `SsrTargets::views`; with none bound the key is the
    /// depth target the groups were resized for. The test needs no submit: it inspects the
    /// key `prepare` builds through `Targets::depth_input` and its siblings.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn bound_depth_input_keys_the_groups_and_unbound_uses_the_target() {
        use raa_core::testing::{skip, software_gpu};
        use raa_pipeline::{ImageDesc, ImageDim, LightingResources, PublishedImage, ResourceTable};

        let test = "bound_depth_input_keys_the_groups_and_unbound_uses_the_target";
        let Some(gpu) = software_gpu() else {
            skip(test, "no software adapter");
            return;
        };
        let size = (16, 16);
        let targets = Targets::new(&gpu, size);
        let lighting = LightingResources::new(&gpu);
        let pipelines = SsrPipelines::new(&gpu, lighting.opaque_layout(), &[]);
        let mut ssr = SsrTargets::new(&gpu, &pipelines, &targets);
        let resources = ResourceTable::new();
        let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("test.ssr.depth_input"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let bound = texture.create_view(&Default::default());
        resources.bind_input(
            "ssr",
            "depth",
            PublishedImage {
                view: bound.clone(),
                desc: ImageDesc::new(
                    ImageDim::D2,
                    wgpu::TextureFormat::Depth32Float,
                    size.0,
                    size.1,
                    1,
                    1,
                ),
            },
        );
        let views = SsrViewKey {
            depth: targets.depth_input(&resources, "ssr"),
            normal: targets.normal_input(&resources, "ssr"),
            material: targets.material_input(&resources, "ssr"),
        };
        ssr.set_views(gpu.device(), &pipelines, views);
        assert!(
            ssr.views.depth == bound,
            "the bound depth view is the cached key"
        );
        resources.clear();
        let views = SsrViewKey {
            depth: targets.depth_input(&resources, "ssr"),
            normal: targets.normal_input(&resources, "ssr"),
            material: targets.material_input(&resources, "ssr"),
        };
        ssr.set_views(gpu.device(), &pipelines, views);
        assert!(
            ssr.views.depth == *targets.depth(),
            "unbound falls back to the depth target"
        );
    }
}
