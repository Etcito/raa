//! Native WebGPU prediction recorder. All work is encoded outside the scene graph.

use raa_core::{GpuContext, RaaError};
use wgpu::util::DeviceExt;

const WARP: &str = include_str!("prediction_warp.wgsl");
/// WGSL declarations shared with the target-only shading provider.
pub const PREDICTION_REFRESH_WGSL: &str = include_str!("prediction_refresh.wgsl");
const WORK_HEADER_BYTES: u64 = 20;
const INCOMPLETE_OFFSET: u64 = 16;

/// Same-device textures for a single clean, unjittered native-size view. The provider
/// must mark uncertain correspondence, unsupported shading and deformation for repair.
pub struct PredictionGuides<'a> {
    pub target_depth: &'a wgpu::Texture,
    /// RGBA32Float. UV displacement XY, rejection Z, expected source depth W.
    pub target_motion: &'a wgpu::Texture,
    /// R32Uint. Nonzero where target lighting needs refresh.
    pub lighting_changed: &'a wgpu::Texture,
    pub depth_tolerance: f32,
    pub refresh_limit: u32,
}

/// The GPU-built list of target pixels requiring shading. The kernel uses 64 threads
/// per workgroup, shades one pixel per thread, and must guard the tail with min(work.count, capacity). Write RGBA16Float linear HDR
/// with alpha 1 only when the exact target pixel can be shaded.
pub struct PredictionRefresh<'a> {
    pub work: &'a wgpu::Buffer,
    pub hdr: &'a wgpu::TextureView,
    pub capacity: u32,
}

/// A renderer-owned, stateless target shading and display provider.
pub trait PredictionRenderer {
    /// Bind the provider's shading kernel to the work list and target HDR. The warp
    /// recorder dispatches it indirectly with the bounded work count.
    fn refresh_pipeline(
        &mut self,
        gpu: &GpuContext,
        target: PredictionRefresh<'_>,
    ) -> Result<(wgpu::ComputePipeline, Vec<wgpu::BindGroup>), RaaError>;

    /// Apply the active display transform to resolved linear HDR. Optional inference
    /// also belongs here, with this HDR and the provider's target guides.
    fn display(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        hdr: &wgpu::TextureView,
        output: &wgpu::TextureView,
    ) -> Result<(), RaaError>;
}

pub(crate) struct PredictionWarp {
    warp: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
}

impl PredictionWarp {
    pub(crate) fn new(gpu: &GpuContext) -> Self {
        use wgpu::{BufferBindingType as B, TextureSampleType as T};
        let device = gpu.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa prediction warp"),
            source: wgpu::ShaderSource::Wgsl(format!("{}\n{WARP}", raa_taa::REPROJECT_WGSL).into()),
        });
        let float = T::Float { filterable: false };
        let work = buffer(B::Storage { read_only: false });
        let pipeline = |name, bindings: &[(u32, wgpu::BindingType)]| {
            let entries: Vec<_> = bindings
                .iter()
                .map(|&(binding, ty)| wgpu::BindGroupLayoutEntry {
                    binding,
                    ty,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                })
                .collect();
            let group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(name),
                entries: &entries,
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(name),
                bind_group_layouts: &[Some(&group)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            warp: pipeline(
                "warp",
                &[
                    (0, buffer(B::Uniform)),
                    (1, sampled(float)),
                    (2, sampled(T::Depth)),
                    (3, sampled(float)),
                    (4, sampled(T::Depth)),
                    (5, sampled(T::Uint)),
                    (6, storage_texture()),
                    (7, work),
                ],
            ),
            resolve: pipeline(
                "resolve",
                &[
                    (7, work),
                    (8, sampled(float)),
                    (9, sampled(float)),
                    (10, storage_texture()),
                ],
            ),
        }
    }

    pub(crate) fn record(
        &self,
        gpu: &GpuContext,
        source: &raa_tonemap::TonemapSource,
        guides: PredictionGuides<'_>,
        output: &wgpu::Texture,
        renderer: &mut impl PredictionRenderer,
    ) -> Result<(wgpu::CommandBuffer, wgpu::Buffer), RaaError> {
        use wgpu::{
            BindingResource as R, BufferUsages as B, TextureFormat as F, TextureUsages as T,
        };
        let device = gpu.device();
        let size = output.size();
        let limit = device.limits();
        let work_bytes = WORK_HEADER_BYTES + 4 * u64::from(guides.refresh_limit.max(1));
        let groups = [size.width.div_ceil(8), size.height.div_ceil(8)];
        for (texture, format) in [
            (&source.hdr, F::Rgba16Float),
            (&source.depth, F::Depth32Float),
            (guides.target_depth, F::Depth32Float),
            (guides.target_motion, F::Rgba32Float),
            (guides.lighting_changed, F::R32Uint),
        ] {
            if texture.size() != size
                || texture.dimension() != wgpu::TextureDimension::D2
                || texture.sample_count() != 1
                || texture.format() != format
                || !texture.usage().contains(T::TEXTURE_BINDING)
            {
                return Err(unsupported(
                    "prediction guides must match native output and their declared formats",
                ));
            }
        }
        if !(0.0..1.0).contains(&guides.depth_tolerance)
            || u64::from(size.width) * u64::from(size.height) >= u64::from(u32::MAX)
            || u64::from(guides.refresh_limit) > u64::from(size.width) * u64::from(size.height)
            || groups
                .into_iter()
                .any(|count| count > limit.max_compute_workgroups_per_dimension)
            || guides.refresh_limit.div_ceil(64) > limit.max_compute_workgroups_per_dimension
            || work_bytes > u64::from(limit.max_storage_buffer_binding_size)
            || work_bytes > limit.max_buffer_size
        {
            return Err(unsupported(
                "prediction budget or depth tolerance exceeds limits",
            ));
        }
        let texture = |label| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: F::Rgba16Float,
                usage: T::TEXTURE_BINDING | T::STORAGE_BINDING,
                view_formats: &[],
            })
        };
        let view = |texture: &wgpu::Texture| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        let warped = view(&texture("raa prediction warped HDR"));
        let refreshed = view(&texture("raa prediction refreshed HDR"));
        let resolved = view(&texture("raa prediction resolved HDR"));
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("raa prediction parameters"),
            usage: B::UNIFORM,
            contents: &[
                guides.depth_tolerance.to_bits().to_le_bytes(),
                guides.refresh_limit.to_le_bytes(),
            ]
            .concat(),
        });
        let make_buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        // Keep one spare element for the shader's runtime-array minimum binding size.
        let work = make_buffer(
            "raa prediction work",
            work_bytes,
            B::STORAGE | B::COPY_SRC | B::COPY_DST,
        );
        let indirect = make_buffer("raa prediction dispatch", 12, B::INDIRECT | B::COPY_DST);
        let validation = make_buffer("raa prediction validation", 4, B::MAP_READ | B::COPY_DST);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("raa prediction"),
        });
        let mut header = [0u8; WORK_HEADER_BYTES as usize];
        header[4] = 1;
        header[8] = 1;
        let init = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("raa prediction work header"),
            usage: B::COPY_SRC,
            contents: &header,
        });
        encoder.copy_buffer_to_buffer(&init, 0, &work, 0, WORK_HEADER_BYTES);
        let source_depth = view(&source.depth);
        let source = view(&source.hdr);
        let motion = view(guides.target_motion);
        let depth = view(guides.target_depth);
        let change = view(guides.lighting_changed);
        let dispatch = |encoder: &mut wgpu::CommandEncoder,
                        pipeline: &wgpu::ComputePipeline,
                        bindings: &[(u32, R<'_>)]| {
            let entries: Vec<_> = bindings
                .iter()
                .map(|(binding, resource)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: resource.clone(),
                })
                .collect();
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raa prediction"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(groups[0], groups[1], 1);
        };
        dispatch(
            &mut encoder,
            &self.warp,
            &[
                (0, params.as_entire_binding()),
                (1, R::TextureView(&source)),
                (2, R::TextureView(&source_depth)),
                (3, R::TextureView(&motion)),
                (4, R::TextureView(&depth)),
                (5, R::TextureView(&change)),
                (6, R::TextureView(&warped)),
                (7, work.as_entire_binding()),
            ],
        );
        if guides.refresh_limit != 0 {
            encoder.copy_buffer_to_buffer(&work, 0, &indirect, 0, 12);
            let (pipeline, bindings) = renderer.refresh_pipeline(
                gpu,
                PredictionRefresh {
                    work: &work,
                    hdr: &refreshed,
                    capacity: guides.refresh_limit,
                },
            )?;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa prediction selective refresh"),
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            for (index, group) in bindings.iter().enumerate() {
                pass.set_bind_group(index as u32, group, &[]);
            }
            pass.dispatch_workgroups_indirect(&indirect, 0);
        }
        dispatch(
            &mut encoder,
            &self.resolve,
            &[
                (7, work.as_entire_binding()),
                (8, R::TextureView(&warped)),
                (9, R::TextureView(&refreshed)),
                (10, R::TextureView(&resolved)),
            ],
        );
        renderer.display(gpu, &mut encoder, &resolved, &view(output))?;
        encoder.copy_buffer_to_buffer(&work, INCOMPLETE_OFFSET, &validation, 0, 4);
        Ok((encoder.finish(), validation))
    }
}

fn buffer(ty: wgpu::BufferBindingType) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty,
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

fn sampled(sample_type: wgpu::TextureSampleType) -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type,
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    }
}

fn storage_texture() -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        access: wgpu::StorageTextureAccess::WriteOnly,
        format: wgpu::TextureFormat::Rgba16Float,
        view_dimension: wgpu::TextureViewDimension::D2,
    }
}

fn unsupported(message: &str) -> RaaError {
    RaaError::Unsupported(message.into())
}
