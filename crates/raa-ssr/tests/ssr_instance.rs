#![cfg(not(target_arch = "wasm32"))]

use std::{cell::RefCell, rc::Rc};

use raa_core::testing::{block_on, skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, OffscreenTarget, RaaError, wgpu};
use raa_pipeline::{GraphFrame, Pipeline, RecordCtx};
use raa_ssr::SsrFeature;

const SIZE: u32 = 32;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

struct Binding {
    views: [wgpu::TextureView; 2],
}

impl Binding {
    fn new(gpu: &GpuContext) -> Self {
        let views = std::array::from_fn(|_| {
            gpu.device()
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("test.ssr.chain"),
                    size: wgpu::Extent3d {
                        width: SIZE,
                        height: SIZE,
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
        });
        Self { views }
    }
}

fn clear(ctx: &mut RecordCtx<'_>, view: &wgpu::TextureView, depth: Option<&wgpu::TextureView>) {
    let _pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("test.ssr.clear"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: depth.map(|view| wgpu::RenderPassDepthStencilAttachment {
            view,
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

impl GraphFrame for Binding {
    fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if pass == "ssr" {
            // No geometry pass exists: initialise both HDR inputs and the G-buffer,
            // including far depth, before either SSR pass reads them.
            clear(ctx, &self.views[0], Some(ctx.targets.depth()));
            clear(ctx, &self.views[1], None);
            clear(ctx, ctx.targets.normal(), None);
            clear(ctx, ctx.targets.material(), None);
        }
        if pass == "ssr" || pass == "ssr#b" {
            let (src, dst) = if pass == "ssr" { (0, 1) } else { (1, 0) };
            ctx.targets.hdr().bind(&self.views[src], &self.views[dst]);
            assert!(ctx.targets.hdr().is_bound());
        }
        Ok(())
    }

    fn after_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if pass == "ssr" || pass == "ssr#b" {
            ctx.targets.hdr().unbind();
        }
        Ok(())
    }
}

#[test]
fn instances_and_bound_frames() {
    let Some(gpu) = software_gpu() else {
        skip("instances_and_bound_frames", "no software adapter");
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, FORMAT, false)));
    let _first = SsrFeature::install(&gpu, &pipeline).unwrap();
    let _second = SsrFeature::install_instance(&gpu, &pipeline, "b").unwrap();
    let passes = pipeline.borrow().passes();
    let first = passes.iter().find(|p| p.name == "ssr").unwrap();
    let second = passes.iter().find(|p| p.name == "ssr#b").unwrap();
    assert_eq!(first.order, second.order);
    assert!(second.enabled);
    assert_eq!(
        SsrFeature::install_instance(&gpu, &pipeline, "b")
            .err()
            .unwrap()
            .code(),
        "pipeline.duplicate_pass"
    );
    pipeline.borrow_mut().remove("ssr#b").unwrap();
    assert!(!pipeline.borrow().passes().iter().any(|p| p.name == "ssr#b"));
    let _second = SsrFeature::install_instance(&gpu, &pipeline, "b").unwrap();
    pipeline
        .borrow_mut()
        .add_graph_frame("binding", Box::new(Binding::new(&gpu)))
        .unwrap();
    let output = OffscreenTarget::new(&gpu, SIZE, SIZE, FORMAT);
    let mut graph = FrameGraph::new(FORMAT);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().unwrap();
    for _ in 0..2 {
        gpu.queue()
            .submit([graph.execute(&gpu, output.view(), (SIZE, SIZE)).unwrap()]);
    }
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    assert!(block_on(errors.pop()).is_none(), "GPU validation error");
}
