//! Compute passes of the M2 cloud backend (contract `P2-CLOUD-PASSES`).
//!
//! The ground shadow and its mean follow Solstice `crates/solstice-render/src/clouds/mod.rs:548-560`
//! at revision 7e465b599c13da165948ebaa44df824c36c3ae07 (MIT, Copyright (c) 2026 SamG-Coder):
//! the 256x256 map in 8x8 workgroups, then one workgroup for its mean. Every pass binds
//! only group 2; groups 0 and 1 use one empty layout.

use raa_core::{GpuContext, RaaError};

use super::super::{MAX_VIEWS, invalid};
use super::noise::CloudNoise;
use super::resources::{CloudTargets, SHADOW_SIZE};
use super::shaders::{shadow_mean_module_wgsl, shadow_module_wgsl};
use crate::atmosphere::AtmosphereResources;

/// Bind groups of one view, valid for the sky-view LUT they were built with.
struct ViewGroups {
    sky_view: wgpu::TextureView,
    shadow: wgpu::BindGroup,
    mean: wgpu::BindGroup,
}

/// The cloud pipelines with their per-view bind groups.
pub(super) struct CloudPasses {
    /// Group 2 bindings 0-10 of contract `P2-CLOUD-GPU`.
    march_layout: wgpu::BindGroupLayout,
    mean_layout: wgpu::BindGroupLayout,
    shadow: wgpu::ComputePipeline,
    mean: wgpu::ComputePipeline,
    views: [Option<ViewGroups>; MAX_VIEWS],
}

fn entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

fn texture(binding: u32, view_dimension: wgpu::TextureViewDimension) -> wgpu::BindGroupLayoutEntry {
    entry(
        binding,
        wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension,
            multisampled: false,
        },
    )
}

fn buffer(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    entry(
        binding,
        wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    )
}

fn output(binding: u32) -> wgpu::BindGroupLayoutEntry {
    entry(
        binding,
        wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba16Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
    )
}

/// Validates `source` on the CPU, then builds its pipeline over groups (empty, empty, `layout`).
fn pipeline(
    gpu: &GpuContext,
    label: &str,
    source: &str,
    entry_point: &str,
    empty: &wgpu::BindGroupLayout,
    layout: &wgpu::BindGroupLayout,
) -> Result<wgpu::ComputePipeline, RaaError> {
    raa_shader::Composer::validate(source)
        .map_err(|error| RaaError::Gpu(format!("{label} shader: {error}")))?;
    let device = gpu.device();
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(empty), Some(empty), Some(layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    Ok(
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        }),
    )
}

impl CloudPasses {
    /// Creates the layouts and the shadow and mean pipelines.
    pub(super) fn new(gpu: &GpuContext) -> Result<CloudPasses, RaaError> {
        use wgpu::BufferBindingType::{Storage, Uniform};
        use wgpu::TextureViewDimension::{D2, D3};
        let device = gpu.device();
        let filtering = wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering);
        let march_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-sky.cloud_march"),
            entries: &[
                buffer(0, Uniform),
                buffer(1, Uniform),
                texture(2, D3),
                texture(3, D3),
                texture(4, D2),
                texture(5, D2),
                texture(6, D2),
                entry(7, filtering),
                entry(8, filtering),
                buffer(9, Storage { read_only: true }),
                output(10),
            ],
        });
        let mean_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-sky.cloud_shadow_mean"),
            entries: &[texture(0, D2), output(1)],
        });
        let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-sky.cloud_empty"),
            entries: &[],
        });
        let shadow = pipeline(
            gpu,
            "raa-sky.cloud_shadow",
            &shadow_module_wgsl(),
            "raa_cloud_shadow_main",
            &empty,
            &march_layout,
        )?;
        let mean = pipeline(
            gpu,
            "raa-sky.cloud_shadow_mean",
            &shadow_mean_module_wgsl(),
            "raa_cloud_shadow_mean_main",
            &empty,
            &mean_layout,
        )?;
        Ok(CloudPasses {
            march_layout,
            mean_layout,
            shadow,
            mean,
            views: std::array::from_fn(|_| None),
        })
    }

    /// Builds the bind groups of `view` (< 4); a no-op while they exist for the same
    /// sky-view LUT. Rejects with `sky.invalid` when the view has no sky-view LUT.
    pub(super) fn bind_view(
        &mut self,
        gpu: &GpuContext,
        view: u32,
        noise: &CloudNoise,
        targets: &CloudTargets,
        atmosphere: &AtmosphereResources,
    ) -> Result<(), RaaError> {
        let slot = view as usize;
        if slot >= MAX_VIEWS {
            return Err(invalid(format!(
                "cloud view {view} is outside 0..{MAX_VIEWS}"
            )));
        }
        let Some(sky_view) = atmosphere.sky_view(view) else {
            return Err(invalid(format!("cloud view {view} has no sky-view LUT")));
        };
        if self.views[slot]
            .as_ref()
            .is_some_and(|groups| groups.sky_view == *sky_view)
        {
            return Ok(());
        }
        let shadow = self.march_bind_group(
            gpu,
            view,
            noise,
            targets,
            atmosphere,
            targets.shadow_map(),
            "raa-sky.cloud_shadow",
        )?;
        let mean = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-sky.cloud_shadow_mean"),
            layout: &self.mean_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(targets.shadow_map()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(targets.shadow_mean()),
                },
            ],
        });
        self.views[slot] = Some(ViewGroups {
            sky_view: sky_view.clone(),
            shadow,
            mean,
        });
        Ok(())
    }

    /// One bind group of `march_layout` writing `output` at binding 10.
    #[allow(clippy::too_many_arguments)]
    fn march_bind_group(
        &self,
        gpu: &GpuContext,
        view: u32,
        noise: &CloudNoise,
        targets: &CloudTargets,
        atmosphere: &AtmosphereResources,
        output: &wgpu::TextureView,
        label: &str,
    ) -> Result<wgpu::BindGroup, RaaError> {
        let Some(sky_view) = atmosphere.sky_view(view) else {
            return Err(invalid(format!("cloud view {view} has no sky-view LUT")));
        };
        use wgpu::BindingResource::{Sampler, TextureView};
        let resources = [
            targets.params(view).as_entire_binding(),
            targets.sky(view).as_entire_binding(),
            TextureView(noise.shape()),
            TextureView(noise.detail()),
            TextureView(noise.weather_map()),
            TextureView(atmosphere.transmittance()),
            TextureView(sky_view),
            Sampler(targets.noise_sampler()),
            Sampler(atmosphere.sampler()),
            noise.coverage_cdf().as_entire_binding(),
            TextureView(output),
        ];
        let entries: Vec<wgpu::BindGroupEntry> = resources
            .into_iter()
            .enumerate()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource,
            })
            .collect();
        Ok(gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.march_layout,
            entries: &entries,
        }))
    }

    /// Records the ground shadow map then its mean; nothing for a view without bind groups.
    pub(super) fn record_ground_shadow(&self, encoder: &mut wgpu::CommandEncoder, view: u32) {
        let Some(groups) = self.views.get(view as usize).and_then(Option::as_ref) else {
            return;
        };
        let workgroups = SHADOW_SIZE.div_ceil(8);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raa-sky.cloud_shadow"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.shadow);
        pass.set_bind_group(2, &groups.shadow, &[]);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
        pass.set_pipeline(&self.mean);
        pass.set_bind_group(2, &groups.mean, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Drops the bind groups of `view`.
    pub(super) fn release_view(&mut self, view: u32) {
        if let Some(slot) = self.views.get_mut(view as usize) {
            *slot = None;
        }
    }
}
