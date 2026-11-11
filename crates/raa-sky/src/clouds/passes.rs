//! Compute passes of the M2 cloud backend (contract `P2-CLOUD-PASSES`).
//!
//! The ground shadow and its mean follow Solstice `crates/solstice-render/src/clouds/mod.rs:548-560`
//! at revision 7e465b599c13da165948ebaa44df824c36c3ae07 (MIT, Copyright (c) 2026 SamG-Coder):
//! the 256x256 map in 8x8 workgroups, then one workgroup for its mean. Every pass binds
//! only group 2; groups 0 and 1 use one empty layout. The screen march follows
//! `clouds/mod.rs:579-583`: the half-resolution target in 8x8 workgroups, then a copy of
//! the result into the view's history for its next march.

use raa_core::{GpuContext, RaaError};

use super::super::{MAX_VIEWS, invalid};
use super::noise::CloudNoise;
use super::resources::{CloudTargets, PANORAMA_SIZE, SHADOW_SIZE};
use super::shaders::{
    panorama_module_wgsl, screen_module_wgsl, shadow_mean_module_wgsl, shadow_module_wgsl,
};
use crate::atmosphere::AtmosphereResources;

/// Bind groups of one view, valid for the sky-view LUT they were built with.
struct ViewGroups {
    sky_view: wgpu::TextureView,
    shadow: wgpu::BindGroup,
    mean: wgpu::BindGroup,
    panorama: wgpu::BindGroup,
}

/// The screen bind group of one view with the resource identities it was built with.
struct ScreenGroup {
    sky_view: wgpu::TextureView,
    current: wgpu::TextureView,
    depth: wgpu::TextureView,
    multisampled: bool,
    group: wgpu::BindGroup,
}

/// A screen layout (group 2 bindings 0-12) with its pipeline, for one depth sample count.
struct ScreenPipeline {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

/// The cloud pipelines with their per-view bind groups.
pub(super) struct CloudPasses {
    /// Group 2 bindings 0-10 of contract `P2-CLOUD-GPU`.
    march_layout: wgpu::BindGroupLayout,
    mean_layout: wgpu::BindGroupLayout,
    shadow: wgpu::ComputePipeline,
    mean: wgpu::ComputePipeline,
    panorama: wgpu::ComputePipeline,
    /// The screen march for a one-sample depth target.
    screen: ScreenPipeline,
    /// The screen march for a multisampled depth target.
    screen_multisampled: ScreenPipeline,
    views: [Option<ViewGroups>; MAX_VIEWS],
    screens: [Option<ScreenGroup>; MAX_VIEWS],
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

/// Group 2 bindings 0-10 of contract `P2-CLOUD-GPU`.
fn march_entries() -> [wgpu::BindGroupLayoutEntry; 11] {
    use wgpu::BufferBindingType::{Storage, Uniform};
    use wgpu::TextureViewDimension::{D2, D3};
    let filtering = wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering);
    [
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
    ]
}

/// The screen layout (bindings 0-12) and its pipeline from `screen_module_wgsl`.
fn screen_pipeline(
    gpu: &GpuContext,
    multisampled: bool,
    empty: &wgpu::BindGroupLayout,
) -> Result<ScreenPipeline, RaaError> {
    let [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10] = march_entries();
    let depth = entry(
        12,
        wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled,
        },
    );
    let history = texture(11, wgpu::TextureViewDimension::D2);
    let label = if multisampled {
        "raa-sky.cloud_screen_multisampled"
    } else {
        "raa-sky.cloud_screen"
    };
    let layout = gpu
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, history, depth],
        });
    let pipeline = pipeline(
        gpu,
        label,
        &screen_module_wgsl(multisampled),
        "raa_cloud_screen_main",
        empty,
        &layout,
    )?;
    Ok(ScreenPipeline { layout, pipeline })
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
        use wgpu::TextureViewDimension::D2;
        let device = gpu.device();
        let march_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-sky.cloud_march"),
            entries: &march_entries(),
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
        let panorama = pipeline(
            gpu,
            "raa-sky.cloud_panorama",
            &panorama_module_wgsl(),
            "raa_cloud_panorama_main",
            &empty,
            &march_layout,
        )?;
        let screen = screen_pipeline(gpu, false, &empty)?;
        let screen_multisampled = screen_pipeline(gpu, true, &empty)?;
        Ok(CloudPasses {
            march_layout,
            mean_layout,
            shadow,
            mean,
            panorama,
            screen,
            screen_multisampled,
            views: std::array::from_fn(|_| None),
            screens: std::array::from_fn(|_| None),
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
        let panorama = self.march_bind_group(
            gpu,
            view,
            noise,
            targets,
            atmosphere,
            targets.panorama(),
            "raa-sky.cloud_panorama",
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
            panorama,
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
        let entries = Self::march_entries(view, noise, targets, atmosphere, output)?;
        Ok(gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.march_layout,
            entries: &entries,
        }))
    }

    /// The bind group entries 0-10 of a cloud march writing `output`.
    fn march_entries<'a>(
        view: u32,
        noise: &'a CloudNoise,
        targets: &'a CloudTargets,
        atmosphere: &'a AtmosphereResources,
        output: &'a wgpu::TextureView,
    ) -> Result<Vec<wgpu::BindGroupEntry<'a>>, RaaError> {
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
        Ok(resources
            .into_iter()
            .enumerate()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource,
            })
            .collect())
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

    /// Records the cloud panorama (solstice `clouds/mod.rs:548-556`); nothing for a view
    /// without bind groups.
    pub(super) fn record_panorama(&self, encoder: &mut wgpu::CommandEncoder, view: u32) {
        let Some(groups) = self.views.get(view as usize).and_then(Option::as_ref) else {
            return;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raa-sky.cloud_panorama"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.panorama);
        pass.set_bind_group(2, &groups.panorama, &[]);
        pass.dispatch_workgroups(PANORAMA_SIZE.0.div_ceil(8), PANORAMA_SIZE.1.div_ceil(8), 1);
    }

    /// Records the half-resolution screen march of `view` (pipeline by the depth target's
    /// sample count), then copies its result into the view's history. The view's bind
    /// group is rebuilt only when its screen target, depth or sky-view identity changed.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_screen(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        view: u32,
        noise: &CloudNoise,
        targets: &CloudTargets,
        atmosphere: &AtmosphereResources,
        depth: &wgpu::TextureView,
    ) -> Result<(), RaaError> {
        let slot = view as usize;
        if slot >= MAX_VIEWS {
            return Err(invalid(format!(
                "cloud view {view} is outside 0..{MAX_VIEWS}"
            )));
        }
        let (Some(current), Some(history), Some(sky_view)) = (
            targets.screen_current(view),
            targets.screen_history(view),
            atmosphere.sky_view(view),
        ) else {
            return Err(invalid(format!(
                "cloud view {view} has no screen target or sky-view LUT"
            )));
        };
        let multisampled = depth.texture().sample_count() > 1;
        let stale = self.screens[slot].as_ref().is_none_or(|screen| {
            screen.current != *current || screen.depth != *depth || screen.sky_view != *sky_view
        });
        if stale {
            let layout = if multisampled {
                &self.screen_multisampled.layout
            } else {
                &self.screen.layout
            };
            let mut entries = Self::march_entries(view, noise, targets, atmosphere, current)?;
            entries.push(wgpu::BindGroupEntry {
                binding: 11,
                resource: wgpu::BindingResource::TextureView(history),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 12,
                resource: wgpu::BindingResource::TextureView(depth),
            });
            let group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raa-sky.cloud_screen"),
                layout,
                entries: &entries,
            });
            self.screens[slot] = Some(ScreenGroup {
                sky_view: sky_view.clone(),
                current: current.clone(),
                depth: depth.clone(),
                multisampled,
                group,
            });
        }
        let Some(screen) = self.screens[slot].as_ref() else {
            return Ok(());
        };
        let pipeline = if screen.multisampled {
            &self.screen_multisampled.pipeline
        } else {
            &self.screen.pipeline
        };
        let texture = current.texture();
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa-sky.cloud_screen"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(2, &screen.group, &[]);
            pass.dispatch_workgroups(texture.width().div_ceil(8), texture.height().div_ceil(8), 1);
        }
        encoder.copy_texture_to_texture(
            texture.as_image_copy(),
            history.texture().as_image_copy(),
            texture.size(),
        );
        Ok(())
    }

    /// Drops the bind groups of `view`, its screen bind group included.
    pub(super) fn release_view(&mut self, view: u32) {
        if let Some(slot) = self.views.get_mut(view as usize) {
            *slot = None;
        }
        if let Some(slot) = self.screens.get_mut(view as usize) {
            *slot = None;
        }
    }
}
