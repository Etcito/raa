//! Restricted same-eye, fully covered static-scene presentation.

use raa_core::{GpuContext, RaaError};
use raa_math::{Mat4, Vec4};
use raa_pipeline::EnvGpu;
use raa_scene::Camera;
use raa_tonemap::{ToneMapping, TonemapSource};

use crate::prediction_browser::BrowserPrediction;
use crate::prediction_model::{GeneratedInputs, GeneratedModel};

pub(crate) const WGSL: &str = include_str!("prediction_camera.wgsl");

pub(crate) struct CameraWarp {
    pub target_to_source: Mat4,
    pub source_to_target: Mat4,
}

impl CameraWarp {
    pub(crate) fn new(source: Camera, target: Camera) -> Option<Self> {
        if source.eye != target.eye
            || source.near != target.near
            || source.far != target.far
            || source.aspect != target.aspect
            || source.ortho_height.is_some()
            || target.ortho_height.is_some()
        {
            return None;
        }
        let source_vp = source.projection() * source.view();
        let target_vp = target.projection() * target.view();
        let target_to_source = source_vp * target_vp.inverse()?;
        let source_to_target = target_vp * source_vp.inverse()?;
        for x in [-1.0, 1.0] {
            for y in [-1.0, 1.0] {
                let q = target_to_source.mul_vec4(Vec4::new(x, y, 0.5, 1.0));
                if !(q.w.is_finite()
                    && q.w > 0.0
                    && q.x.abs() <= q.w * 1.00001
                    && q.y.abs() <= q.w * 1.00001)
                {
                    return None;
                }
            }
        }
        Some(Self {
            target_to_source,
            source_to_target,
        })
    }
}

pub(crate) struct CameraPrediction {
    warp: wgpu::ComputePipeline,
    no_refresh: wgpu::TextureView,
    no_changes: wgpu::TextureView,
}

impl CameraPrediction {
    pub(crate) fn new(gpu: &GpuContext) -> Self {
        let device = gpu.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prediction.camera.optics"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let warp = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prediction.camera.optics"),
            layout: None,
            module: &module,
            entry_point: Some("reproject_optics"),
            compilation_options: Default::default(),
            cache: None,
        });
        let empty = |label, format| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        Self {
            warp,
            no_refresh: empty(
                "prediction.camera.no-refresh",
                wgpu::TextureFormat::Rgba16Float,
            ),
            no_changes: empty(
                "prediction.camera.no-lighting-changes",
                wgpu::TextureFormat::R32Uint,
            ),
        }
    }

    pub(crate) fn record(
        &self,
        gpu: &GpuContext,
        source: &TonemapSource,
        warp: CameraWarp,
        output: &wgpu::Texture,
        browser: &mut BrowserPrediction,
        tone: ToneMapping,
        env: &EnvGpu,
        learned: Option<(&GeneratedModel, u64)>,
    ) -> Result<(wgpu::CommandBuffer, wgpu::Buffer, bool), RaaError> {
        use wgpu::util::DeviceExt;
        use wgpu::{BindingResource, BufferUsages as B, TextureFormat as F, TextureUsages as U};
        let size = output.size();
        if source.hdr.size() != size || source.depth.size() != size {
            return Err(RaaError::Unsupported(
                "camera prediction requires native retained HDR and depth".into(),
            ));
        }
        let device = gpu.device();
        let view = |label, format| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: U::TEXTURE_BINDING | U::STORAGE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let hdr = view("prediction.camera.hdr", F::Rgba16Float);
        let depth = view("prediction.camera.depth", F::R32Float);
        let motion = view("prediction.camera.motion", F::Rgba32Float);
        let old_hdr = source.hdr.create_view(&Default::default());
        let old_depth = source.depth.create_view(&Default::default());
        let matrix = warp
            .target_to_source
            .cols
            .into_iter()
            .chain(warp.source_to_target.cols)
            .flatten()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let parameters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("prediction.camera.matrices"),
            usage: B::UNIFORM,
            contents: &matrix,
        });
        let buffer = |label, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 4,
                usage,
                mapped_at_creation: false,
            })
        };
        let incomplete = buffer(
            "prediction.camera.incomplete",
            B::STORAGE | B::COPY_SRC | B::COPY_DST,
        );
        let validation = buffer("prediction.camera.validation", B::MAP_READ | B::COPY_DST);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("prediction.camera"),
        });
        encoder.clear_buffer(&incomplete, 0, None);
        let entries = [
            parameters.as_entire_binding(),
            BindingResource::TextureView(&old_hdr),
            BindingResource::TextureView(&old_depth),
            BindingResource::TextureView(&hdr),
            BindingResource::TextureView(&depth),
            incomplete.as_entire_binding(),
            BindingResource::TextureView(&motion),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, resource)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource,
        })
        .collect::<Vec<_>>();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prediction.camera"),
            layout: &self.warp.get_bind_group_layout(0),
            entries: &entries,
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prediction.camera"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.warp);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(size.width.div_ceil(8), size.height.div_ceil(8), 1);
        }
        let refined = learned.and_then(|(model, source_age)| {
            model.record(
                gpu,
                &mut encoder,
                GeneratedInputs {
                    warped: &hdr,
                    refreshed: &self.no_refresh,
                    resolved: &hdr,
                    depth: &depth,
                    motion: &motion,
                    lighting: &self.no_changes,
                    source_age,
                },
            )
        });
        browser.display_camera(
            gpu,
            &mut encoder,
            tone,
            env,
            refined.as_ref().unwrap_or(&hdr),
            depth,
            output,
        )?;
        encoder.copy_buffer_to_buffer(&incomplete, 0, &validation, 0, 4);
        Ok((encoder.finish(), validation, refined.is_some()))
    }
}
