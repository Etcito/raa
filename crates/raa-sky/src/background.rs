//! The `sky.background` forward contributor (S12, order 900): a fullscreen triangle at
//! depth 1 (LessEqual, no depth write) shading every uncovered pixel with the per-pixel
//! common `raa_sky_radiance` seam along the world view ray.
//!
//! The legacy model keeps [`background_module_wgsl`]. The physical model uses
//! [`physical_background_module_wgsl`], sampling the screen-cloud result in viewport UV
//! through the private S10 group-2 bindings while the clear sky comes from the atmosphere
//! LUTs and the celestial evaluator. The contributor remains independently controllable.

use raa_core::RaaError;
use raa_lit::{ForwardContributor, LitHandle};
use raa_pipeline::{HDR_FORMAT, PrepareCtx};

use crate::atmosphere::SharedAtmosphere;
use crate::cloud_backend::SharedCloud;
use crate::desc::SkyModel;
use crate::env_pass::PhysicalSkyBinding;
use crate::shaders::{background_module_wgsl, physical_background_module_wgsl};
use crate::state::SharedSky;

/// S12 name of the contributor.
pub(crate) const BACKGROUND_NAME: &str = "sky.background";
/// S12 order of the contributor inside the forward pass.
pub(crate) const BACKGROUND_ORDER: u32 = 900;

/// Draws the procedural sky behind the opaque geometry of the forward pass.
pub(crate) struct SkyBackground {
    state: SharedSky,
    atmosphere: Option<SharedAtmosphere>,
    cloud: Option<SharedCloud>,
    binding: Option<PhysicalSkyBinding>,
    /// The legacy and physical pipelines and the msaa count they were built for.
    legacy: Option<(u32, wgpu::RenderPipeline)>,
    physical: Option<(u32, wgpu::RenderPipeline)>,
    model: SkyModel,
    physical_ready: bool,
}

impl SkyBackground {
    pub(crate) fn new(state: SharedSky) -> Self {
        SkyBackground {
            state,
            atmosphere: None,
            cloud: None,
            binding: None,
            legacy: None,
            physical: None,
            model: SkyModel::Legacy,
            physical_ready: false,
        }
    }

    /// Attaches the completed atmosphere and cloud resources of the same sky feature.
    pub(crate) fn with_resources(
        mut self,
        atmosphere: SharedAtmosphere,
        cloud: SharedCloud,
    ) -> Self {
        self.atmosphere = Some(atmosphere);
        self.cloud = Some(cloud);
        self
    }
}

/// Builds a background pipeline: groups [frame, lighting] and, for the physical model, the
/// private physical-sky group 2; no vertex buffers, HDR colour without blending,
/// Depth32Float LessEqual without depth writes, `msaa` samples.
fn build_pipeline(
    lit: &LitHandle,
    ctx: &PrepareCtx<'_>,
    msaa: u32,
    physical_layout: Option<&wgpu::BindGroupLayout>,
    wgsl: String,
) -> wgpu::RenderPipeline {
    let device = ctx.gpu.device();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("raa.sky.background"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let frame_layout = lit.frame_layout();
    let groups: Vec<Option<&wgpu::BindGroupLayout>> = match physical_layout {
        Some(physical) => vec![
            Some(&frame_layout),
            Some(ctx.lighting.layout()),
            Some(physical),
        ],
        None => vec![Some(&frame_layout), Some(ctx.lighting.layout())],
    };
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("raa.sky.background.layout"),
        bind_group_layouts: &groups,
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("raa.sky.background"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: msaa,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

impl ForwardContributor for SkyBackground {
    fn name(&self) -> &str {
        BACKGROUND_NAME
    }

    fn order(&self) -> u32 {
        BACKGROUND_ORDER
    }

    /// Builds the pipelines once per msaa count, writes the `EnvGpu` sky fields when the
    /// description changed and keeps the background mode at 1 (sky), so the S12 tonemap
    /// background rule leaves the sky alone. Only runs while the contributor is enabled.
    fn prepare(&mut self, lit: &LitHandle, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let msaa = lit.msaa();
        if self.binding.is_none() && self.atmosphere.is_some() && self.cloud.is_some() {
            self.binding = Some(PhysicalSkyBinding::new(ctx.gpu));
        }
        if self.legacy.as_ref().is_none_or(|(m, _)| *m != msaa) {
            self.legacy = Some((
                msaa,
                build_pipeline(lit, ctx, msaa, None, background_module_wgsl()),
            ));
            self.physical = None;
        }
        if self.physical.is_none()
            && let Some(binding) = &self.binding
        {
            self.physical = Some((
                msaa,
                build_pipeline(
                    lit,
                    ctx,
                    msaa,
                    Some(binding.layout()),
                    physical_background_module_wgsl(),
                ),
            ));
        }
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| RaaError::Resource(format!("{BACKGROUND_NAME}: sky state is busy")))?;
        state.background_on = true;
        if state.gpu_dirty {
            state.desc.write_env(ctx.lighting.env_mut());
            state.gpu_dirty = false;
        }
        if ctx.lighting.env().sky_params[2] != 1.0 {
            ctx.lighting.env_mut().sky_params[2] = 1.0;
        }
        self.model = state.desc.model;
        self.physical_ready = false;
        if self.model == SkyModel::Physical
            && let (Some(view), Some(binding), Some(atmosphere), Some(cloud)) = (
                state.active_view,
                self.binding.as_mut(),
                self.atmosphere.as_ref(),
                self.cloud.as_ref(),
            )
        {
            let views = cloud.try_borrow().ok().map(|backend| backend.views(view));
            let atmosphere = atmosphere.try_borrow().ok();
            if let (Some(views), Some(atmosphere)) = (views, atmosphere) {
                self.physical_ready = binding.prepare(
                    ctx.gpu,
                    &crate::env_pass::PhysicalFrame {
                        state: &state,
                        view,
                        size: ctx.size,
                        time_s: ctx.time_s,
                        atmosphere: &atmosphere,
                        views: &views,
                    },
                );
            }
        }
        Ok(())
    }

    /// Draws the fullscreen triangle; groups 0 and 1 are bound by the forward pass and, for
    /// the ready physical model, group 2 is its private sky-resource set.
    fn record(&self, pass: &mut wgpu::RenderPass<'_>, _lit: &LitHandle) -> Result<(), RaaError> {
        let physical = self.model == SkyModel::Physical && self.physical_ready;
        let pipeline = if physical {
            self.physical.as_ref()
        } else {
            self.legacy.as_ref()
        };
        let Some((_, pipeline)) = pipeline else {
            return Err(RaaError::Resource(format!(
                "{BACKGROUND_NAME}: record ran before prepare built its pipeline"
            )));
        };
        pass.set_pipeline(pipeline);
        if physical && let Some(group) = self.binding.as_ref().and_then(PhysicalSkyBinding::group) {
            pass.set_bind_group(2, group, &[]);
        }
        pass.draw(0..3, 0..1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::SkyState;

    #[test]
    fn name_and_order_follow_s12() {
        let c = SkyBackground::new(SkyState::shared());
        assert_eq!(c.name(), "sky.background");
        assert_eq!(c.order(), 900);
    }
}
