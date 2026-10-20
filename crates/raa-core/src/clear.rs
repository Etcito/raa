//! A render pass that clears one frame graph target.

use crate::graph::{PassBuilder, RecordContext, RenderPass, ResourceId};
use crate::{GpuContext, RaaError};

/// Clears a graph target to a fixed linear colour.
pub struct ClearPass {
    name: String,
    target: ResourceId,
    color: wgpu::Color,
}

impl ClearPass {
    /// Creates a clear pass, storing its name once.
    pub fn new(name: &str, target: ResourceId, color: wgpu::Color) -> Self {
        Self {
            name: name.to_owned(),
            target,
            color,
        }
    }
}

impl RenderPass for ClearPass {
    fn name(&self) -> &str {
        &self.name
    }

    fn declare(&self, builder: &mut PassBuilder) {
        builder.write(self.target);
    }

    fn record(&mut self, ctx: &mut RecordContext<'_>) -> Result<(), RaaError> {
        let view = ctx.view(self.target)?;
        let _pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&self.name),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(self.color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        Ok(())
    }

    fn resize(&mut self, _gpu: &GpuContext, _size: (u32, u32)) {}
}
