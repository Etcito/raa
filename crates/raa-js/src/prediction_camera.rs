//! Restricted same-eye, fully covered static-scene presentation.

use raa_core::{GpuContext, RaaError};
use raa_math::{Mat4, Vec4};
use raa_pipeline::EnvGpu;
use raa_scene::Camera;
use raa_tonemap::{ToneMapping, TonemapSource};

use crate::prediction_browser::BrowserPrediction;
use crate::prediction_model::GeneratedModel;

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

const CAMERA_MODEL: &str = include_str!("prediction_camera_model.wgsl");

struct CameraScratch {
    size: wgpu::Extent3d,
    hdr: wgpu::TextureView,
    depth: wgpu::TextureView,
    model: Option<(wgpu::Buffer, wgpu::TextureView)>,
}

pub(crate) struct CameraPrediction {
    warp: wgpu::ComputePipeline,
    fused: wgpu::ComputePipeline,
    residual: wgpu::ComputePipeline,
    scratch: Option<CameraScratch>,
    optical: wgpu::Buffer,
    age: wgpu::Buffer,
    incomplete: wgpu::Buffer,
}

impl CameraPrediction {
    pub(crate) fn new(gpu: &GpuContext) -> Self {
        let device = gpu.device();
        let source = format!(
            "{WGSL}\n{}\n{CAMERA_MODEL}",
            crate::prediction_model::WGSL.replace("@group(0)", "@group(1)")
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prediction.camera.optics"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        use wgpu::BufferUsages as B;
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        Self {
            warp: pipeline("reproject_optics"),
            fused: pipeline("reproject_model_stem"),
            residual: pipeline("reproject_model_residual"),
            scratch: None,
            optical: buffer("prediction.camera.matrices", 128, B::UNIFORM | B::COPY_DST),
            age: buffer("prediction.camera.age", 4, B::UNIFORM | B::COPY_DST),
            incomplete: buffer(
                "prediction.camera.incomplete",
                4,
                B::STORAGE | B::COPY_SRC | B::COPY_DST,
            ),
        }
    }

    pub(crate) fn prepare_extent(&mut self, gpu: &GpuContext, size: wgpu::Extent3d, learned: bool) {
        let device = gpu.device();
        use wgpu::{BufferUsages as B, TextureFormat as F, TextureUsages as U};
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
        if self.scratch.as_ref().is_none_or(|s| s.size != size) {
            self.scratch = Some(CameraScratch {
                size,
                hdr: view("prediction.camera.hdr", F::Rgba16Float),
                depth: view("prediction.camera.depth", F::R32Float),
                model: None,
            });
        }
        let scratch = self.scratch.as_mut().expect("prepared camera scratch");
        let bytes = u64::from(size.width) * u64::from(size.height) * 48;
        let limits = device.limits();
        if learned
            && scratch.model.is_none()
            && bytes <= 128 * 1024 * 1024
            && bytes
                <= u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size)
        {
            let stem = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("prediction.camera.features"),
                size: bytes,
                usage: B::STORAGE,
                mapped_at_creation: false,
            });
            scratch.model = Some((stem, view("prediction.camera.refined", F::Rgba16Float)));
        }
    }

    pub(crate) fn record(
        &mut self,
        gpu: &GpuContext,
        source: &TonemapSource,
        warp: CameraWarp,
        output: &wgpu::Texture,
        browser: &mut BrowserPrediction,
        tone: ToneMapping,
        env: &EnvGpu,
        learned: Option<(&GeneratedModel, u64)>,
    ) -> Result<(wgpu::CommandBuffer, wgpu::Buffer, bool), RaaError> {
        use wgpu::{BindingResource as R, BufferUsages as B};
        let size = output.size();
        if source.hdr.size() != size || source.depth.size() != size {
            return Err(RaaError::Unsupported(
                "camera prediction requires native retained HDR and depth".into(),
            ));
        }
        let learned = learned.filter(|(_, age)| *age <= 8);
        self.prepare_extent(gpu, size, learned.is_some());
        let scratch = self.scratch.as_ref().expect("prepared camera scratch");
        let learned = learned.zip(scratch.model.as_ref());
        let device = gpu.device();
        let old_hdr = source.hdr.create_view(&Default::default());
        let old_depth = source.depth.create_view(&Default::default());
        let mut matrix = [0u8; 128];
        for (word, value) in matrix.chunks_exact_mut(4).zip(
            warp.target_to_source
                .cols
                .into_iter()
                .chain(warp.source_to_target.cols)
                .flatten(),
        ) {
            word.copy_from_slice(&value.to_le_bytes());
        }
        gpu.queue().write_buffer(&self.optical, 0, &matrix);
        let validation = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prediction.camera.validation"),
            size: 4,
            usage: B::MAP_READ | B::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("prediction.camera"),
        });
        encoder.clear_buffer(&self.incomplete, 0, None);
        let group = |pipeline: &wgpu::ComputePipeline, index, bindings: &[(u32, R<'_>)]| {
            let entries = bindings
                .iter()
                .map(|(binding, resource)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: resource.clone(),
                })
                .collect::<Vec<_>>();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("prediction.camera"),
                layout: &pipeline.get_bind_group_layout(index),
                entries: &entries,
            })
        };
        let run = |encoder: &mut wgpu::CommandEncoder,
                   pipeline: &wgpu::ComputePipeline,
                   bindings: &[&wgpu::BindGroup]| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prediction.camera"),
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            for (i, binding) in bindings.iter().enumerate() {
                pass.set_bind_group(i as u32, *binding, &[]);
            }
            pass.dispatch_workgroups(size.width.div_ceil(8), size.height.div_ceil(8), 1);
        };
        let optics = [
            (0, self.optical.as_entire_binding()),
            (1, R::TextureView(&old_hdr)),
            (2, R::TextureView(&old_depth)),
            (3, R::TextureView(&scratch.hdr)),
            (4, R::TextureView(&scratch.depth)),
            (5, self.incomplete.as_entire_binding()),
        ];
        let mut refined = &scratch.hdr;
        if let Some(((model, age), (features, result))) = learned {
            gpu.queue()
                .write_buffer(&self.age, 0, &(age as f32 / 8.0).to_le_bytes());
            let warp = group(&self.fused, 0, &optics);
            let stem = group(
                &self.fused,
                1,
                &[
                    (0, model.weights().as_entire_binding()),
                    (1, features.as_entire_binding()),
                    (7, self.age.as_entire_binding()),
                ],
            );
            run(&mut encoder, &self.fused, &[&warp, &stem]);
            let empty = group(&self.residual, 0, &[]);
            let decode = group(
                &self.residual,
                1,
                &[
                    (0, model.weights().as_entire_binding()),
                    (1, features.as_entire_binding()),
                    (8, R::TextureView(&scratch.hdr)),
                    (9, R::TextureView(result)),
                ],
            );
            run(&mut encoder, &self.residual, &[&empty, &decode]);
            refined = result;
        } else {
            let warp = group(&self.warp, 0, &optics);
            run(&mut encoder, &self.warp, &[&warp]);
        }
        browser.display_camera(
            gpu,
            &mut encoder,
            tone,
            env,
            refined,
            scratch.depth.clone(),
            output,
        )?;
        encoder.copy_buffer_to_buffer(&self.incomplete, 0, &validation, 0, 4);
        Ok((encoder.finish(), validation, learned.is_some()))
    }
}
