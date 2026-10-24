//! The `opaque.copy` pass (S12, order 70): `wgsl/copy.wgsl`.
//!
//! On resize the pass creates `opaque_color` (HDR_FORMAT, the full mip chain of the target
//! size, one view per mip plus a full view) and `opaque_depth` (R32Float), and the bind groups
//! of every draw over them. While a Transparent draw exists it installs both full views as the
//! group-1 `opaque_color` / `opaque_depth` resources (S10 bindings 9 and 10) and, each frame,
//! copies hdr.src() into mip 0, downsamples every further mip from the previous one (2x2
//! filtered) and writes the prepass depth into `opaque_depth`. The HDR chain is not swapped.
//!
//! The lighting resources are set or cleared only when that state changes (or the views were
//! recreated by a resize), so the group-1 bind group is not rebuilt every frame.

use raa_core::wgpu;
use raa_core::{GpuContext, RaaError};
use raa_lit::{DrawClass, LitHandle};
use raa_pipeline::{FeaturePass, HDR_FORMAT, LightingResources, PrepareCtx, RecordCtx, Targets};

/// S12 name of the pass.
pub(crate) const NAME: &str = "opaque.copy";
/// S12 order of the pass (after the opaque HDR passes, before `transparent.oit`).
pub(crate) const ORDER: u32 = 70;
/// Format of the opaque depth copy (textureLoad only, S10 binding 10).
const DEPTH_COPY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// Render module (no PRELUDE), group 0 only: 0 source mip and 1 linear clamp sampler
/// (`fs_downsample`), 2 the prepass depth (`fs_depth`). Every fragment entry is drawn with
/// the fullscreen triangle of [`VS_MAIN`] (3 vertices).
pub(crate) const COPY_WGSL: &str = include_str!("wgsl/copy.wgsl");
/// Fullscreen triangle vertex entry.
const VS_MAIN: &str = "vs_main";
/// 2x2 filtered downsample of the previous mip.
const FS_DOWNSAMPLE: &str = "fs_downsample";
/// Prepass depth into R32Float.
const FS_DEPTH: &str = "fs_depth";

/// Mips of the full chain of a `w` x `h` texture: floor(log2(max(w, h))) + 1.
pub(crate) fn mip_count(w: u32, h: u32) -> u32 {
    32 - w.max(h).max(1).leading_zeros()
}

fn layout_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
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

fn view_entry(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
    entry: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
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

/// One fullscreen draw into `target` (every texel written, so nothing is loaded).
fn draw(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    group: &wgpu::BindGroup,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("raa-transmission.opaque_copy"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.draw(0..3, 0..1);
}

/// The surface-sized copies and the bind group of every draw over them.
struct Copies {
    size: (u32, u32),
    color: wgpu::Texture,
    /// Every mip (the group-1 `opaque_color`).
    color_view: wgpu::TextureView,
    /// One single-mip view per mip (render target of mip k, source of mip k + 1).
    mip_views: Vec<wgpu::TextureView>,
    /// Group k reads mip k (downsample into mip k + 1).
    down_groups: Vec<wgpu::BindGroup>,
    /// Kept alive for `depth_view`.
    _depth: wgpu::Texture,
    /// The group-1 `opaque_depth`, also the target of `fs_depth`.
    depth_view: wgpu::TextureView,
    /// Reads the prepass depth target.
    depth_group: wgpu::BindGroup,
    /// Both textures of the HDR chain the copies were sized for (the copy source is one).
    hdr: [wgpu::Texture; 2],
}

/// The `opaque.copy` render pass.
#[allow(dead_code, reason = "installed by TransmissionFeature::install")]
pub(crate) struct OpaqueCopyPass {
    lit: LitHandle,
    down_layout: wgpu::BindGroupLayout,
    depth_layout: wgpu::BindGroupLayout,
    down: wgpu::RenderPipeline,
    depth: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    copies: Option<Copies>,
    /// True while the copy views are the lighting resources' opaque copy.
    installed: bool,
    /// True when a resize recreated the views after they were installed.
    stale: bool,
}

impl OpaqueCopyPass {
    /// Builds the two render pipelines (`fs_downsample` into HDR_FORMAT, `fs_depth` into
    /// R32Float) and the linear clamp sampler; the copies are created by the first resize.
    #[allow(dead_code, reason = "installed by TransmissionFeature::install")]
    pub(crate) fn new(gpu: &GpuContext, lit: LitHandle) -> Self {
        let device = gpu.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-transmission.copy"),
            source: wgpu::ShaderSource::Wgsl(COPY_WGSL.into()),
        });
        let down_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-transmission.copy.down_layout"),
            entries: &[
                layout_entry(
                    0,
                    texture_type(wgpu::TextureSampleType::Float { filterable: true }),
                ),
                layout_entry(
                    1,
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
            ],
        });
        let depth_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-transmission.copy.depth_layout"),
            entries: &[layout_entry(
                2,
                texture_type(wgpu::TextureSampleType::Depth),
            )],
        });
        let down = create_pipeline(device, &module, &down_layout, FS_DOWNSAMPLE, HDR_FORMAT);
        let depth = create_pipeline(device, &module, &depth_layout, FS_DEPTH, DEPTH_COPY_FORMAT);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raa-transmission.copy.linear_clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        OpaqueCopyPass {
            lit,
            down_layout,
            depth_layout,
            down,
            depth,
            sampler,
            copies: None,
            installed: false,
            stale: false,
        }
    }
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
            width: size.0.max(1),
            height: size.1.max(1),
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

impl FeaturePass for OpaqueCopyPass {
    fn name(&self) -> &str {
        NAME
    }

    fn order(&self) -> u32 {
        ORDER
    }

    /// Recreates `opaque_color` (HDR_FORMAT, full mip chain, TEXTURE_BINDING |
    /// RENDER_ATTACHMENT | COPY_DST, a full view and one view per mip), `opaque_depth`
    /// (R32Float, TEXTURE_BINDING | RENDER_ATTACHMENT) and the bind groups of every draw.
    /// Installed views are replaced at the next prepare.
    fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
        let device = gpu.device();
        let size = targets.size();
        let size = (size.0.max(1), size.1.max(1));
        let mips = mip_count(size.0, size.1);
        let color = texture(
            device,
            "raa-transmission.opaque_color",
            size,
            mips,
            HDR_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST,
        );
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let mip_views: Vec<wgpu::TextureView> = (0..mips)
            .map(|level| {
                color.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("raa-transmission.opaque_color.mip"),
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let down_groups = mip_views
            .iter()
            .take(mip_views.len().saturating_sub(1))
            .map(|src| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("raa-transmission.copy.down"),
                    layout: &self.down_layout,
                    entries: &[
                        view_entry(0, src),
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                })
            })
            .collect();
        let depth = texture(
            device,
            "raa-transmission.opaque_depth",
            size,
            1,
            DEPTH_COPY_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-transmission.copy.depth"),
            layout: &self.depth_layout,
            entries: &[view_entry(2, targets.depth())],
        });
        let hdr = targets.hdr();
        self.copies = Some(Copies {
            size,
            color,
            color_view,
            mip_views,
            down_groups,
            _depth: depth,
            depth_view,
            depth_group,
            hdr: [hdr.src_texture().clone(), hdr.dst_texture().clone()],
        });
        self.stale = self.installed;
    }

    /// Clears the opaque copy from the lighting resources (dummies) when disabled.
    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        if !enabled {
            lighting.clear_opaque_copy();
            self.installed = false;
            self.stale = false;
        }
    }

    /// Installs the copy while a Transparent draw exists and clears it otherwise, calling a
    /// setter only on a state change (or after a resize replaced the installed views).
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let wanted = !self.lit.draws(DrawClass::Transparent).is_empty();
        match (&self.copies, wanted) {
            (Some(c), true) if !self.installed || self.stale => {
                ctx.lighting
                    .set_opaque_copy(c.color_view.clone(), c.depth_view.clone());
                self.installed = true;
                self.stale = false;
            }
            (_, false) if self.installed => {
                ctx.lighting.clear_opaque_copy();
                self.installed = false;
                self.stale = false;
            }
            _ => {}
        }
        Ok(())
    }

    /// While installed: copies hdr.src() into mip 0, downsamples each further mip from the
    /// previous one and writes the prepass depth into `opaque_depth`. No HDR swap.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if !self.installed {
            return Ok(());
        }
        let Some(c) = self.copies.as_ref() else {
            return Err(RaaError::Resource(
                "opaque.copy: record ran before a resize created the copies".to_owned(),
            ));
        };
        let src = ctx.targets.hdr().src_texture();
        if !c.hdr.contains(src) {
            return Err(RaaError::Resource(
                "opaque.copy: the HDR source is not a texture the pass was resized for".to_owned(),
            ));
        }
        let encoder = &mut *ctx.encoder;
        encoder.copy_texture_to_texture(
            src.as_image_copy(),
            c.color.as_image_copy(),
            wgpu::Extent3d {
                width: c.size.0,
                height: c.size.1,
                depth_or_array_layers: 1,
            },
        );
        for (target, group) in c.mip_views.iter().skip(1).zip(&c.down_groups) {
            draw(encoder, target, &self.down, group);
        }
        draw(encoder, &c.depth_view, &self.depth, &c.depth_group);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_shader::Composer;

    #[test]
    fn copy_shader_validates() {
        if let Err(e) = Composer::validate(COPY_WGSL) {
            panic!("{e}");
        }
    }

    #[test]
    fn copy_shader_defines_its_entry_points() {
        for entry in [VS_MAIN, FS_DOWNSAMPLE, FS_DEPTH] {
            assert!(COPY_WGSL.contains(&format!("fn {entry}(")), "{entry}");
        }
    }

    #[test]
    fn mip_count_is_the_full_chain() {
        assert_eq!(mip_count(1280, 720), 11);
        assert_eq!(mip_count(960, 540), 10);
        assert_eq!(mip_count(1024, 1), 11);
        assert_eq!(mip_count(1, 1), 1);
        assert_eq!(mip_count(0, 0), 1);
    }
}
