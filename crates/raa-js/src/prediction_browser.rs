use raa_core::{GpuContext, RaaError};
use raa_pipeline::{EnvGpu, Pipeline};
use raa_tonemap::{ToneMapping, TonemapControl, TonemapPass};

use crate::{PredictionRefresh, PredictionRenderer};

pub(crate) const REFRESH_SHADER: &str = include_str!("prediction_browser.wgsl");

pub(crate) struct BrowserPrediction {
    refresh: wgpu::ComputePipeline,
    display: TonemapPass,
    settings: TonemapControl,
}

impl BrowserPrediction {
    pub(crate) fn new(gpu: &GpuContext, pipeline: &Pipeline) -> Self {
        let module = gpu
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("browser prediction repair"),
                source: wgpu::ShaderSource::Wgsl(
                    format!("{}\n{REFRESH_SHADER}", crate::PREDICTION_REFRESH_WGSL).into(),
                ),
            });
        use wgpu::{BindingType as B, ShaderStages as S, TextureViewDimension as D};
        let entries = [
            B::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            B::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: D::D2,
                multisampled: false,
            },
            B::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: D::D2,
            },
        ]
        .into_iter()
        .enumerate()
        .map(|(binding, ty)| wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: S::COMPUTE,
            ty,
            count: None,
        })
        .collect::<Vec<_>>();
        let group_layout =
            gpu.device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("browser prediction repair"),
                    entries: &entries,
                });
        let layout = gpu
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("browser prediction repair"),
                bind_group_layouts: &[Some(&group_layout)],
                immediate_size: 0,
            });
        let refresh = gpu
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("browser prediction repair"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("refresh"),
                compilation_options: Default::default(),
                cache: None,
            });
        let (display, settings) = TonemapPass::new(gpu, pipeline);
        Self {
            refresh,
            display,
            settings,
        }
    }

    pub(crate) fn frame<'a>(
        &'a mut self,
        settings: ToneMapping,
        env: EnvGpu,
        depth: &'a wgpu::Texture,
        samples: &'a wgpu::Texture,
    ) -> Result<impl PredictionRenderer + 'a, RaaError> {
        self.settings.set(settings)?;
        Ok(BrowserFrame {
            state: self,
            env,
            depth,
            samples,
        })
    }
}

struct BrowserFrame<'a> {
    state: &'a mut BrowserPrediction,
    env: EnvGpu,
    depth: &'a wgpu::Texture,
    samples: &'a wgpu::Texture,
}

impl PredictionRenderer for BrowserFrame<'_> {
    fn refresh_pipeline(
        &mut self,
        gpu: &GpuContext,
        target: PredictionRefresh<'_>,
    ) -> Result<(wgpu::ComputePipeline, Vec<wgpu::BindGroup>), RaaError> {
        use wgpu::BindingResource::TextureView;
        let pipeline = &self.state.refresh;
        let samples = self.samples.create_view(&Default::default());
        let group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("browser prediction refresh"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: target.work.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: TextureView(&samples),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: TextureView(target.hdr),
                },
            ],
        });
        Ok((pipeline.clone(), vec![group]))
    }

    fn display(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        hdr: &wgpu::TextureView,
        output: &wgpu::TextureView,
    ) -> Result<(), RaaError> {
        self.state.display.record_external(
            gpu,
            encoder,
            hdr,
            self.depth.create_view(&Default::default()),
            &self.env,
            output,
        );
        Ok(())
    }
}

pub(crate) fn upload(
    gpu: &GpuContext,
    size: wgpu::Extent3d,
    label: &'static str,
    format: wgpu::TextureFormat,
    channels: u32,
    bytes: &[u8],
) -> wgpu::Texture {
    let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    gpu.queue().write_texture(
        texture.as_image_copy(),
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size.width * channels * 4),
            rows_per_image: Some(size.height),
        },
        size,
    );
    texture
}
