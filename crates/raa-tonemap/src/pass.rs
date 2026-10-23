//! The `tonemap` feature pass (S12, order 200, required): a fullscreen triangle that reads
//! the HDR chain's source, the depth target and the env uniform and writes the backbuffer.

use std::collections::HashMap;

use raa_core::{GpuContext, RaaError};
use raa_pipeline::{FeaturePass, Pipeline, PrepareCtx, RecordCtx, Targets};

use crate::{ToneMapping, TonemapControl};

/// WGSL of the display transform (vertex `vs_main`, fragment `fs_main`; group 0: params,
/// hdr source, depth, EnvGpu).
pub const TONEMAP_WGSL: &str = include_str!("tonemap.wgsl");

/// Name of the pass in the pipeline (S12).
pub const TONEMAP_NAME: &str = "tonemap";
/// Default order of the pass (S12): after every other pass.
pub const TONEMAP_ORDER: u32 = 200;

/// Size in bytes of the pass's own uniform: `exposure: vec4<f32>` and `flags: vec4<u32>`.
const PARAMS_SIZE: usize = 32;
/// Size in bytes of the S10 `EnvGpu` uniform.
const ENV_SIZE: u64 = 256;

/// The bind group of the last frame with the textures it was built for.
struct Bound {
    hdr: wgpu::Texture,
    depth: wgpu::Texture,
    group: wgpu::BindGroup,
}

/// Fullscreen tone mapping pass; created with its [`TonemapControl`] by [`TonemapPass::new`].
pub struct TonemapPass {
    control: TonemapControl,
    surface_is_srgb: bool,
    module: wgpu::ShaderModule,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    params: wgpu::Buffer,
    env: wgpu::Buffer,
    /// One render pipeline per backbuffer format.
    pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    bound: Option<Bound>,
}

fn params_bytes(settings: &ToneMapping, surface_is_srgb: bool) -> [u8; PARAMS_SIZE] {
    let mut out = [0u8; PARAMS_SIZE];
    let words = [
        settings.exposure_scale().to_bits(),
        settings.exposure_ev.to_bits(),
        0,
        0,
        settings.mapper.gpu_index(),
        settings.look.gpu_index(),
        u32::from(settings.dither),
        u32::from(surface_is_srgb),
    ];
    for (chunk, word) in out.chunks_exact_mut(4).zip(words) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    out
}

fn layout_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty,
        count: None,
    }
}

fn uniform(min_size: u64) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: wgpu::BufferSize::new(min_size),
    }
}

impl TonemapPass {
    /// Builds the pass for `pipeline`'s surface and returns it with the control that edits its
    /// settings (TONEMAP_DEFAULT until set). The caller adds the pass to the pipeline.
    pub fn new(gpu: &GpuContext, pipeline: &Pipeline) -> (TonemapPass, TonemapControl) {
        let device = gpu.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa.tonemap.wgsl"),
            source: wgpu::ShaderSource::Wgsl(TONEMAP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa.tonemap.layout"),
            entries: &[
                layout_entry(0, uniform(PARAMS_SIZE as u64)),
                layout_entry(
                    1,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                layout_entry(
                    2,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                layout_entry(3, uniform(ENV_SIZE)),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raa.tonemap.pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let buffer = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let control = TonemapControl::default();
        let mut pass = TonemapPass {
            control: control.clone(),
            surface_is_srgb: pipeline.surface_is_srgb(),
            module,
            layout,
            pipeline_layout,
            params: buffer("raa.tonemap.params", PARAMS_SIZE as u64),
            env: buffer("raa.tonemap.env", ENV_SIZE),
            pipelines: HashMap::new(),
            bound: None,
        };
        pass.ensure_pipeline(gpu, pipeline.surface_format());
        (pass, control)
    }

    fn ensure_pipeline(&mut self, gpu: &GpuContext, format: wgpu::TextureFormat) {
        if self.pipelines.contains_key(&format) {
            return;
        }
        let pipeline = gpu
            .device()
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("raa.tonemap.pipeline"),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &self.module,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.module,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        self.pipelines.insert(format, pipeline);
    }

    fn ensure_bind_group(&mut self, gpu: &GpuContext, targets: &Targets) {
        let hdr = targets.hdr().src_texture();
        let depth = targets.depth_texture();
        if let Some(bound) = &self.bound
            && bound.hdr == *hdr
            && bound.depth == *depth
        {
            return;
        }
        let group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa.tonemap.bind_group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(targets.hdr().src()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(targets.depth()),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.env.as_entire_binding(),
                },
            ],
        });
        self.bound = Some(Bound {
            hdr: hdr.clone(),
            depth: depth.clone(),
            group,
        });
    }
}

impl FeaturePass for TonemapPass {
    fn name(&self) -> &str {
        TONEMAP_NAME
    }

    fn order(&self) -> u32 {
        TONEMAP_ORDER
    }

    fn required(&self) -> bool {
        true
    }

    fn resize(&mut self, _gpu: &GpuContext, _targets: &Targets) {
        self.bound = None;
    }

    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let queue = ctx.gpu.queue();
        let settings = self.control.get();
        queue.write_buffer(
            &self.params,
            0,
            &params_bytes(&settings, self.surface_is_srgb),
        );
        queue.write_buffer(&self.env, 0, &ctx.lighting.env().to_bytes());
        Ok(())
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        self.ensure_pipeline(ctx.gpu, ctx.backbuffer_format);
        self.ensure_bind_group(ctx.gpu, ctx.targets);
        let (Some(pipeline), Some(bound)) = (
            self.pipelines.get(&ctx.backbuffer_format),
            self.bound.as_ref(),
        ) else {
            return Err(RaaError::Resource(
                "tonemap: pipeline or bind group missing at record".to_owned(),
            ));
        };
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("raa.tonemap"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.backbuffer,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bound.group, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Look, ToneMapper};

    #[test]
    fn params_bytes_pack_exposure_mapper_look_dither_and_srgb() {
        let s = ToneMapping {
            mapper: ToneMapper::Aces,
            exposure_ev: 2.0,
            look: Look::Golden,
            dither: false,
        };
        let b = params_bytes(&s, true);
        let word =
            |i: usize| u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        assert_eq!(f32::from_bits(word(0)), 4.0);
        assert_eq!(f32::from_bits(word(1)), 2.0);
        assert_eq!([word(4), word(5), word(6), word(7)], [1, 2, 0, 1]);
    }
}
