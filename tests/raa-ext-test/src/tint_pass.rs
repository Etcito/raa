//! A small external render pass that overlays a fixed linear tint.

use raa_core::{
    BACKBUFFER, CLEAR_COLOR, GpuContext, PassBuilder, RaaError, RecordContext, RenderPass,
};

/// Fixed linear RGBA tint used by [`TintPass`].
pub const TINT: [f32; 4] = [0.9, 0.2, 0.6, 0.25];

const TINT_WGSL: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.9, 0.2, 0.6, 0.25);
}
"#;

/// Alpha-blended full-screen tint pass, implemented outside the engine crates.
pub struct TintPass {
    pipeline: Option<(
        raa_core::wgpu::TextureFormat,
        raa_core::wgpu::RenderPipeline,
    )>,
}

impl TintPass {
    /// Creates the CPU-only pass; its GPU pipeline is created on first record.
    pub const fn new() -> TintPass {
        Self { pipeline: None }
    }

    fn pipeline_for(
        &mut self,
        gpu: &GpuContext,
        format: raa_core::wgpu::TextureFormat,
    ) -> Result<&raa_core::wgpu::RenderPipeline, RaaError> {
        if self
            .pipeline
            .as_ref()
            .is_none_or(|(cached, _)| *cached != format)
        {
            let device = gpu.device();
            let shader = device.create_shader_module(raa_core::wgpu::ShaderModuleDescriptor {
                label: Some("ext-tint.wgsl"),
                source: raa_core::wgpu::ShaderSource::Wgsl(TINT_WGSL.into()),
            });
            let pipeline =
                device.create_render_pipeline(&raa_core::wgpu::RenderPipelineDescriptor {
                    label: Some("ext-tint.pipeline"),
                    layout: None,
                    vertex: raa_core::wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        compilation_options: raa_core::wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: raa_core::wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: raa_core::wgpu::MultisampleState::default(),
                    fragment: Some(raa_core::wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        compilation_options: raa_core::wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(raa_core::wgpu::ColorTargetState {
                            format,
                            blend: Some(raa_core::wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: raa_core::wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                });
            self.pipeline = Some((format, pipeline));
        }
        self.pipeline
            .as_ref()
            .map(|(_, pipeline)| pipeline)
            .ok_or_else(|| RaaError::Resource("ext-tint pipeline was not created".to_owned()))
    }
}

impl Default for TintPass {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderPass for TintPass {
    fn name(&self) -> &str {
        "ext-tint"
    }

    fn declare(&self, builder: &mut PassBuilder) {
        builder.read(BACKBUFFER);
        builder.write(BACKBUFFER);
    }

    fn record(&mut self, ctx: &mut RecordContext<'_>) -> Result<(), RaaError> {
        let format = ctx.format(BACKBUFFER);
        let pipeline = self.pipeline_for(ctx.gpu, format)?;
        let view = ctx.view(BACKBUFFER)?;
        let load = if ctx.is_first_write(BACKBUFFER) {
            raa_core::wgpu::LoadOp::Clear(CLEAR_COLOR)
        } else {
            raa_core::wgpu::LoadOp::Load
        };
        let mut pass = ctx
            .encoder
            .begin_render_pass(&raa_core::wgpu::RenderPassDescriptor {
                label: Some("ext-tint"),
                color_attachments: &[Some(raa_core::wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: raa_core::wgpu::Operations {
                        load,
                        store: raa_core::wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        pass.set_pipeline(pipeline);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}
