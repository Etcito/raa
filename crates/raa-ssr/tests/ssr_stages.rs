#![cfg(not(target_arch = "wasm32"))]

use std::{cell::RefCell, rc::Rc};

use raa_core::testing::{skip, software_gpu};
use raa_core::{FrameGraph, OffscreenTarget, RaaError, wgpu};
use raa_pipeline::{FeaturePass, Pipeline, RecordCtx};
use raa_ssr::stages::{SsrStageKind, install_stage, stage_entry};

const SIZE: u32 = 32;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

struct Setup;

impl FeaturePass for Setup {
    fn name(&self) -> &str {
        "test.setup"
    }
    fn order(&self) -> u32 {
        30
    }
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let hdr = ctx.targets.hdr();
        let depth = ctx
            .targets
            .depth_attachment_input(ctx.resources, self.name());
        let pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test.setup"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: hdr.dst(),
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth,
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
        drop(pass);
        hdr.swap();
        Ok(())
    }
}

#[test]
fn hiz_and_trace_stage_contract() {
    let Some(gpu) = software_gpu() else {
        skip("hiz_and_trace_stage_contract", "no software adapter");
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, FORMAT, false)));
    pipeline.borrow_mut().add(Box::new(Setup)).unwrap();
    let h = install_stage(&gpu, &pipeline, SsrStageKind::Hiz, "h").unwrap();
    let t = install_stage(&gpu, &pipeline, SsrStageKind::Trace, "t").unwrap();
    let entries = pipeline
        .borrow()
        .passes()
        .into_iter()
        .map(|p| p.name)
        .collect::<Vec<_>>();
    assert!(entries.contains(&stage_entry(SsrStageKind::Hiz, "h")));
    assert!(entries.contains(&stage_entry(SsrStageKind::Trace, "t")));
    assert_eq!(
        install_stage(&gpu, &pipeline, SsrStageKind::Hiz, "h")
            .err()
            .unwrap()
            .code(),
        "pipeline.duplicate_pass"
    );

    let output = OffscreenTarget::new(&gpu, SIZE, SIZE, FORMAT);
    let mut graph = FrameGraph::new(FORMAT);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().unwrap();
    gpu.queue()
        .submit([graph.execute(&gpu, output.view(), (SIZE, SIZE)).unwrap()]);
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

    let hiz = h.published();
    assert_eq!(hiz.len(), 1);
    assert_eq!(hiz[0].0, "hiz");
    assert_eq!(hiz[0].1.format, wgpu::TextureFormat::R32Float);
    assert_eq!((hiz[0].1.width, hiz[0].1.height), (SIZE, SIZE));
    assert_eq!(hiz[0].1.mips, raa_ssr::hiz::mip_count(SIZE, SIZE));
    let trace = t.published();
    assert_eq!(trace.len(), 1);
    assert_eq!(trace[0].0, "trace");
    assert_eq!(trace[0].1.format, wgpu::TextureFormat::Rgba16Float);
    assert_eq!((trace[0].1.width, trace[0].1.height), (SIZE, SIZE));
    assert!(h.ran());
    assert!(!t.ran());
    assert!(
        raa_core::testing::block_on(errors.pop()).is_none(),
        "GPU validation error"
    );

    assert_eq!(
        t.set_values(&[("maxRoughness".into(), 1.5)])
            .unwrap_err()
            .code(),
        "post.invalid"
    );
    assert_eq!(
        t.set_values(&[("maxSteps".into(), 32.5)])
            .unwrap_err()
            .code(),
        "post.invalid"
    );
    assert!(t.set_values(&[("maxSteps".into(), 32.0)]).is_ok());
    assert_eq!(
        h.set_values(&[("anything".into(), 1.0)])
            .unwrap_err()
            .code(),
        "post.invalid"
    );
}
