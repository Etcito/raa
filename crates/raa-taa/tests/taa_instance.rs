#![cfg(not(target_arch = "wasm32"))]

use std::{cell::RefCell, rc::Rc};

use raa_core::testing::{block_on, skip, software_gpu};
use raa_core::{FrameGraph, OffscreenTarget, RaaError, wgpu};
use raa_lit::{LitOptions, LitRenderer};
use raa_pipeline::{GraphFrame, Pipeline, RecordCtx};
use raa_taa::TaaFeature;

struct Binding {
    views: [wgpu::TextureView; 2],
}

impl GraphFrame for Binding {
    fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if pass == "taa" || pass == "taa#b" {
            // This empty pipeline has no prepass: initialize depth to far and motion and
            // both pool images to zero before the resolve reads them.
            let colors = [
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.views[0],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.views[1],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: ctx.targets.motion(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ];
            drop(ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("taa test clear"),
                color_attachments: &colors,
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
            }));
            ctx.targets.hdr().bind(&self.views[0], &self.views[1]);
        }
        Ok(())
    }

    fn after_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if pass == "taa" || pass == "taa#b" {
            ctx.targets.hdr().unbind();
        }
        Ok(())
    }
}

#[test]
fn instances_install_and_resolve_bound_chains_independently() {
    let Some(gpu) = software_gpu() else {
        skip(
            "instances_install_and_resolve_bound_chains_independently",
            "software_gpu returned None",
        );
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline = Rc::new(RefCell::new(Pipeline::new(
        &gpu,
        wgpu::TextureFormat::Rgba8Unorm,
        false,
    )));
    let lit = LitRenderer::new(
        &gpu,
        &pipeline,
        LitOptions {
            max_instances: 1,
            max_materials: 1,
            msaa: 1,
        },
    )
    .expect("lit");
    let frame = lit.handle().frame_buffer();
    let installed = TaaFeature::install(&gpu, &pipeline, &frame).expect("install");
    let instance = TaaFeature::install_instance(&gpu, &pipeline, &frame, "b").expect("instance");
    let rows = pipeline.borrow().passes();
    let a = rows.iter().find(|row| row.name == "taa").expect("taa");
    let b = rows.iter().find(|row| row.name == "taa#b").expect("taa#b");
    assert!(!a.enabled && b.enabled);
    assert_eq!(a.order, b.order);
    assert_eq!(
        TaaFeature::install_instance(&gpu, &pipeline, &frame, "b")
            .err()
            .expect("duplicate")
            .code(),
        "pipeline.duplicate_pass"
    );
    pipeline
        .borrow_mut()
        .set_enabled("taa", true)
        .expect("enable");
    let texture = || {
        gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("taa test pool"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
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
    };
    let views = [
        texture().create_view(&Default::default()),
        texture().create_view(&Default::default()),
    ];
    pipeline
        .borrow_mut()
        .add_graph_frame("binding", Box::new(Binding { views }))
        .expect("graph frame");
    let mut graph = FrameGraph::new(wgpu::TextureFormat::Rgba8Unorm);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().expect("compile");
    let output = OffscreenTarget::new(&gpu, 64, 64, wgpu::TextureFormat::Rgba8Unorm);
    for _ in 0..2 {
        let commands = graph
            .execute(&gpu, output.view(), output.size())
            .expect("frame");
        gpu.queue().submit([commands]);
    }
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    assert_eq!(installed.accumulated_frames(), 2);
    assert_eq!(instance.accumulated_frames(), 2);
    instance.invalidate();
    assert_eq!(instance.accumulated_frames(), 0);
    assert_eq!(installed.accumulated_frames(), 2);
    assert!(
        pipeline
            .borrow()
            .resources()
            .get("taa", "history")
            .is_some()
    );
    assert!(
        pipeline
            .borrow()
            .resources()
            .get("taa#b", "history")
            .is_some()
    );
    assert!(pipeline.borrow_mut().remove("taa#b").is_ok());
    let error = block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");
}
