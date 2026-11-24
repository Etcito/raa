use raa_core::{GpuContext, RaaError};

use super::pass::{buffer, buffer_entry, image_entry};
use crate::wgsl::mesh_hybrid_module;

const LABEL: &str = "trace.mesh.hybrid_path";
const WEIGHT_BYTES: usize = 131 * 4;
const PIXEL_BYTES: u64 = 8 * 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum PathMode {
    #[default]
    Off = 0,
    Short = 1,
    Traced = 2,
    Learned = 3,
}

#[derive(Clone)]
pub(super) struct PathCapture {
    pub seed: u32,
    pub result: Option<((u32, u32), wgpu::Buffer)>,
}

#[derive(Default)]
pub(super) struct PathState {
    pub(super) revision: u64,
    pub(super) weights: Option<Vec<u8>>,
    pub(super) capture: Option<PathCapture>,
}

impl PathState {
    pub(super) fn load(&mut self, bytes: &[u8]) -> Result<(), RaaError> {
        if bytes.len() != WEIGHT_BYTES
            || bytes.chunks_exact(4).any(|word| {
                !f32::from_le_bytes(word.try_into().expect("one model lane")).is_finite()
            })
        {
            return Err(RaaError::FrameGraph {
                code: "trace.invalid_path_model",
                message: "expected 131 finite little-endian specular weight lanes".into(),
            });
        }
        self.weights = Some(bytes.to_vec());
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }
}

pub(super) struct HybridGpu {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    options: wgpu::Buffer,
    weights: wgpu::Buffer,
    dummy: wgpu::Buffer,
    uploaded: Option<u64>,
}

impl HybridGpu {
    pub(super) fn new(device: &wgpu::Device, scene_layout: &wgpu::BindGroupLayout) -> Self {
        let binding = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        use wgpu::{BufferBindingType as B, BufferUsages as U};
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(LABEL),
            entries: &[
                binding(0, B::Uniform),
                binding(1, B::Storage { read_only: true }),
                binding(2, B::Storage { read_only: false }),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(LABEL),
            bind_group_layouts: &[Some(scene_layout), Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(LABEL),
            source: wgpu::ShaderSource::Wgsl(mesh_hybrid_module().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(LABEL),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("trace_hybrid_path"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            layout,
            pipeline,
            options: buffer(device, LABEL, 16, U::UNIFORM | U::COPY_DST),
            weights: buffer(device, LABEL, WEIGHT_BYTES as u64, U::STORAGE | U::COPY_DST),
            dummy: buffer(device, LABEL, 4, U::STORAGE),
            uploaded: None,
        }
    }

    pub(super) fn record(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        frame: u64,
        scene: &wgpu::BindGroup,
        ao: &wgpu::TextureView,
        state: &mut PathState,
        mode: PathMode,
        size: (u32, u32),
        capture_enabled: bool,
    ) {
        use wgpu::BufferUsages as U;
        let device = gpu.device();
        let queue = gpu.queue();
        if self.uploaded != Some(state.revision) {
            if let Some(weights) = &state.weights {
                queue.write_buffer(&self.weights, 0, weights);
            }
            self.uploaded = Some(state.revision);
        }
        let capture_bytes = u64::from(size.0) * u64::from(size.1) * PIXEL_BYTES;
        let request = state.capture.as_mut().filter(|_| capture_enabled);
        let seed = request.as_ref().map_or(frame as u32, |req| req.seed);
        let sample_buffer = request
            .as_ref()
            .map(|_| buffer(device, LABEL, capture_bytes, U::STORAGE | U::COPY_SRC));
        let params = [mode as u32, seed, u32::from(request.is_some()), 1];
        let bytes: Vec<_> = params.into_iter().flat_map(u32::to_le_bytes).collect();
        queue.write_buffer(&self.options, 0, &bytes);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(LABEL),
            layout: &self.layout,
            entries: &[
                buffer_entry(0, &self.options),
                buffer_entry(1, &self.weights),
                buffer_entry(2, sample_buffer.as_ref().unwrap_or(&self.dummy)),
                image_entry(3, ao),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(LABEL),
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, scene, &[]);
            pass.set_bind_group(1, &group, &[]);
            pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
        }
        if let (Some(request), Some(sample_buffer)) = (request, sample_buffer) {
            let readback = buffer(device, LABEL, capture_bytes, U::MAP_READ | U::COPY_DST);
            encoder.copy_buffer_to_buffer(&sample_buffer, 0, &readback, 0, capture_bytes);
            request.result = Some((size, readback));
        }
    }
}
