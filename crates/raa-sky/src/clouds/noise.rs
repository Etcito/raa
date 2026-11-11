//! One-time GPU generation of the cloud noise volumes, weather map and coverage CDF.
//!
//! Sizes, formats and dispatch counts follow Solstice
//! `crates/solstice-render/src/clouds/mod.rs:27-30, 111-128, 207-209, 234-266` at
//! revision 7e465b599c13da165948ebaa44df824c36c3ae07 (MIT, Copyright (c) 2026
//! SamG-Coder). The noise is ready only after the acknowledged submission of the
//! exact view and invocation that carried its generation.

use raa_core::{GpuContext, RaaError};

use super::shaders::{COVERAGE_ENTRIES, NOISE_ENTRIES, coverage_module_wgsl, noise_module_wgsl};

/// Dispatch counts of the five passes: shape, detail, weather, histogram, accumulate.
const DISPATCHES: [(u32, u32, u32); 5] =
    [(32, 32, 32), (8, 8, 8), (64, 64, 1), (64, 64, 1), (1, 1, 1)];
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const CDF_BYTES: u64 = 256 * 4;

/// Device-free readiness gate of the noise generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct NoiseGate {
    /// Invocation whose encoders carried the passes, with a mask of those views.
    carried: Option<(u64, u32)>,
    generated: bool,
}

impl NoiseGate {
    /// Whether the passes must be recorded for this encoder; notes the carrier.
    pub(super) fn record(&mut self, invocation: u64, view: u32) -> bool {
        if self.generated || view >= 32 {
            return false;
        }
        let mask = match self.carried {
            Some((carried, mask)) if carried == invocation => mask,
            _ => 0,
        };
        self.carried = Some((invocation, mask | (1 << view)));
        true
    }

    /// Marks the noise generated when this view and invocation carried the passes.
    pub(super) fn acknowledge(&mut self, view: u32, invocation: u64) {
        if let Some((carried, mask)) = self.carried
            && carried == invocation
            && view < 32
            && mask & (1 << view) != 0
        {
            self.generated = true;
            self.carried = None;
        }
    }

    pub(super) fn generated(&self) -> bool {
        self.generated
    }
}

/// The cloud noise and coverage generation passes with their readiness.
pub(super) struct CloudNoise {
    /// Pipelines in dispatch order with their group-2 bind groups.
    passes: Vec<(wgpu::ComputePipeline, wgpu::BindGroup)>,
    gate: NoiseGate,
}

fn texture(
    gpu: &GpuContext,
    label: &str,
    size: u32,
    depth: u32,
    dim: wgpu::TextureDimension,
) -> wgpu::TextureView {
    gpu.device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: depth,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: dim,
            format: FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

fn buffer(gpu: &GpuContext, label: &str) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: CDF_BYTES,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

fn entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

fn storage_texture(
    binding: u32,
    view_dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    entry(
        binding,
        wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: FORMAT,
            view_dimension,
        },
    )
}

fn storage_buffer(binding: u32) -> wgpu::BindGroupLayoutEntry {
    entry(
        binding,
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    )
}

/// Validates `source` on the CPU, then builds one pipeline per entry over `layout`.
fn pipelines(
    gpu: &GpuContext,
    label: &str,
    source: &str,
    entries: &[&str],
    layout: &wgpu::BindGroupLayout,
) -> Result<Vec<wgpu::ComputePipeline>, RaaError> {
    raa_shader::Composer::validate(source)
        .map_err(|error| RaaError::Gpu(format!("{label} shader: {error}")))?;
    let device = gpu.device();
    let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&empty), Some(&empty), Some(layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    Ok(entries
        .iter()
        .map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        })
        .collect())
}

impl CloudNoise {
    /// Creates the resources and the five generation passes; waits on nothing.
    pub(super) fn new(gpu: &GpuContext) -> Result<CloudNoise, RaaError> {
        use wgpu::{TextureDimension as Dim, TextureViewDimension as View};
        let device = gpu.device();
        let noise_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-sky.cloud_noise"),
            entries: &[storage_texture(0, View::D3), storage_texture(1, View::D2)],
        });
        let coverage_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-sky.cloud_coverage"),
            entries: &[
                entry(
                    0,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: View::D2,
                        multisampled: false,
                    },
                ),
                storage_buffer(1),
                storage_buffer(2),
            ],
        });
        let noise = pipelines(
            gpu,
            "raa-sky.cloud_noise",
            &noise_module_wgsl(),
            &NOISE_ENTRIES,
            &noise_layout,
        )?;
        let coverage = pipelines(
            gpu,
            "raa-sky.cloud_coverage",
            &coverage_module_wgsl(),
            &COVERAGE_ENTRIES,
            &coverage_layout,
        )?;

        let shape = texture(gpu, "raa-sky.cloud_shape", 128, 128, Dim::D3);
        let detail = texture(gpu, "raa-sky.cloud_detail", 32, 32, Dim::D3);
        let weather_map = texture(gpu, "raa-sky.cloud_weather_map", 512, 1, Dim::D2);
        let volume_placeholder = texture(gpu, "raa-sky.cloud_noise_placeholder_3d", 1, 1, Dim::D3);
        let map_placeholder = texture(gpu, "raa-sky.cloud_noise_placeholder_2d", 1, 1, Dim::D2);
        let bins = buffer(gpu, "raa-sky.cloud_coverage_bins");
        let coverage_cdf = buffer(gpu, "raa-sky.cloud_coverage_cdf");

        let group = |layout: &wgpu::BindGroupLayout, resources: &[(u32, wgpu::BindingResource)]| {
            let entries: Vec<wgpu::BindGroupEntry> = resources
                .iter()
                .map(|(binding, resource)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: resource.clone(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raa-sky.cloud_noise"),
                layout,
                entries: &entries,
            })
        };
        use wgpu::BindingResource::TextureView as Tex;
        let volume_groups = [
            group(
                &noise_layout,
                &[(0, Tex(&shape)), (1, Tex(&map_placeholder))],
            ),
            group(
                &noise_layout,
                &[(0, Tex(&detail)), (1, Tex(&map_placeholder))],
            ),
            group(
                &noise_layout,
                &[(0, Tex(&volume_placeholder)), (1, Tex(&weather_map))],
            ),
        ];
        let coverage_group = group(
            &coverage_layout,
            &[
                (0, Tex(&weather_map)),
                (1, bins.as_entire_binding()),
                (2, coverage_cdf.as_entire_binding()),
            ],
        );
        let mut passes: Vec<_> = noise.into_iter().zip(volume_groups).collect();
        passes.extend(
            coverage
                .into_iter()
                .map(|pipeline| (pipeline, coverage_group.clone())),
        );
        Ok(CloudNoise {
            passes,
            gate: NoiseGate::default(),
        })
    }

    /// Records the five passes until generation is acknowledged.
    pub(super) fn record(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        invocation: u64,
        view: u32,
    ) {
        if self.generated() || !self.gate.record(invocation, view) {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raa-sky.cloud_noise"),
            timestamp_writes: None,
        });
        for ((pipeline, group), (x, y, z)) in self.passes.iter().zip(DISPATCHES) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(2, group, &[]);
            pass.dispatch_workgroups(x, y, z);
        }
    }

    /// Acknowledges a submission of `view` in `invocation`.
    pub(super) fn acknowledge(&mut self, view: u32, invocation: u64) {
        self.gate.acknowledge(view, invocation);
    }

    pub(super) fn generated(&self) -> bool {
        self.gate.generated()
    }
}
