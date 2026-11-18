#![cfg(not(target_arch = "wasm32"))]

use std::{cell::RefCell, rc::Rc};

use raa_cel::{CelFeature, CelSettings};
use raa_core::testing::{skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, OffscreenTarget, RaaError, wgpu};
use raa_pipeline::{GraphFrame, Pipeline, RecordCtx};

struct Binding {
    views: [wgpu::TextureView; 2],
}

impl GraphFrame for Binding {
    fn before_record(&mut self, _: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        // The empty pipeline has no prepass: initialize depth to far and clear the
        // normal and HDR inputs before the cel shader samples them.
        {
            let _clear = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cel test inputs"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.views[0],
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: ctx.targets.normal(),
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: ctx.targets.depth(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        ctx.targets.hdr().bind(&self.views[0], &self.views[1]);
        Ok(())
    }

    fn after_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        assert_eq!(
            ctx.resources.get(pass, "color").unwrap().view,
            self.views[0]
        );
        ctx.targets.hdr().unbind();
        Ok(())
    }
}

fn texture(gpu: &GpuContext) -> wgpu::TextureView {
    gpu.device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("cel test pool"),
            size: wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

#[test]
fn instances_and_bound_chain() {
    let Some(gpu) = software_gpu() else {
        skip("cel_instance", "no software adapter");
        return;
    };
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let output = OffscreenTarget::new(&gpu, 16, 16, format);
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, format, false)));
    let installed = CelFeature::install(&gpu, &pipeline).unwrap();
    let instance = CelFeature::install_instance(&gpu, &pipeline, "b").unwrap();
    let passes = pipeline.borrow().passes();
    assert_eq!(passes.len(), 2);
    assert_eq!(passes[0].name, "cel.resolve");
    assert_eq!(passes[1].name, "cel.resolve#b");
    assert_eq!(passes[0].order, passes[1].order);
    assert!(passes[1].enabled);
    assert_eq!(
        CelFeature::install_instance(&gpu, &pipeline, "b")
            .err()
            .unwrap()
            .code(),
        "pipeline.duplicate_pass"
    );
    installed
        .set(CelSettings {
            softness: 0.3,
            ..Default::default()
        })
        .unwrap();
    instance
        .set(CelSettings {
            softness: 0.7,
            ..Default::default()
        })
        .unwrap();
    pipeline
        .borrow_mut()
        .add_graph_frame(
            "binding",
            Box::new(Binding {
                views: [texture(&gpu), texture(&gpu)],
            }),
        )
        .unwrap();
    let mut graph = FrameGraph::new(format);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().unwrap();
    for _ in 0..2 {
        gpu.queue()
            .submit([graph.execute(&gpu, output.view(), (16, 16)).unwrap()]);
    }
    let env = pipeline.borrow().lighting().env().cel;
    assert_eq!(env[1], 0.3);
    pipeline.borrow_mut().remove("cel.resolve#b").unwrap();
    assert_eq!(pipeline.borrow().lighting().env().cel, env);
}
