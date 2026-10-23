//! The `sky.background` forward contributor (S12, order 900): a fullscreen triangle at
//! depth 1 (LessEqual, no depth write) shading every uncovered pixel with the per-pixel
//! `raa_sky_radiance` along the world view ray.

use raa_core::RaaError;
use raa_lit::{ForwardContributor, LitHandle};
use raa_pipeline::{HDR_FORMAT, PrepareCtx};

use crate::shaders::background_module_wgsl;
use crate::state::SharedSky;

/// S12 name of the contributor.
pub(crate) const BACKGROUND_NAME: &str = "sky.background";
/// S12 order of the contributor inside the forward pass.
pub(crate) const BACKGROUND_ORDER: u32 = 900;

/// Draws the procedural sky behind the opaque geometry of the forward pass.
#[allow(dead_code)] // consumed by SkyFeature (sky.background)
pub(crate) struct SkyBackground {
    state: SharedSky,
    /// The render pipeline and the msaa count it was built for.
    pipeline: Option<(u32, wgpu::RenderPipeline)>,
}

impl SkyBackground {
    #[allow(dead_code)] // consumed by SkyFeature (sky.background)
    pub(crate) fn new(state: SharedSky) -> Self {
        SkyBackground {
            state,
            pipeline: None,
        }
    }
}

/// Builds the background pipeline: groups [frame, lighting], no vertex buffers, HDR colour
/// without blending, Depth32Float LessEqual without depth writes, `msaa` samples.
fn build_pipeline(lit: &LitHandle, ctx: &PrepareCtx<'_>, msaa: u32) -> wgpu::RenderPipeline {
    let device = ctx.gpu.device();
    let wgsl = background_module_wgsl();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("raa.sky.background"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let frame_layout = lit.frame_layout();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("raa.sky.background.layout"),
        bind_group_layouts: &[Some(&frame_layout), Some(ctx.lighting.layout())],
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

    /// Builds the pipeline once per msaa count, writes the `EnvGpu` sky fields when the
    /// description changed and keeps the background mode at 1 (sky), so the S12 tonemap
    /// background rule leaves the sky alone. Only runs while the contributor is enabled.
    fn prepare(&mut self, lit: &LitHandle, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let msaa = lit.msaa();
        if self.pipeline.as_ref().is_none_or(|(m, _)| *m != msaa) {
            self.pipeline = Some((msaa, build_pipeline(lit, ctx, msaa)));
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
        Ok(())
    }

    /// Draws the fullscreen triangle; groups 0 and 1 are bound by the forward pass.
    fn record(&self, pass: &mut wgpu::RenderPass<'_>, _lit: &LitHandle) -> Result<(), RaaError> {
        let Some((_, pipeline)) = &self.pipeline else {
            return Err(RaaError::Resource(format!(
                "{BACKGROUND_NAME}: record ran before prepare built its pipeline"
            )));
        };
        pass.set_pipeline(pipeline);
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
