//! `GradingFeature::install_instance` and the bound HDR chain (native, software GPU or skip).
#![cfg(not(target_arch = "wasm32"))]

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use raa_core::testing::{skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, OffscreenTarget, RaaError, wgpu};
use raa_grading::GradingFeature;
use raa_pipeline::{GraphFrame, Pipeline, RecordCtx};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SIZE: u32 = 32;

/// Binds the HDR chain to two pool-usage textures around the pass `pass`.
struct BindFrame {
    pass: String,
    views: [wgpu::TextureView; 2],
    bound_in_record: Rc<Cell<u32>>,
}

impl GraphFrame for BindFrame {
    fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if pass == self.pass {
            let hdr = ctx.targets.hdr();
            hdr.bind(&self.views[0], &self.views[1]);
            assert!(hdr.is_bound());
            self.bound_in_record.set(self.bound_in_record.get() + 1);
        }
        Ok(())
    }

    fn after_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if pass == self.pass {
            ctx.targets.hdr().unbind();
        }
        Ok(())
    }
}

fn pool_view(gpu: &GpuContext) -> wgpu::TextureView {
    gpu.device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("test.pool"),
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
}

#[test]
fn install_instance_adds_a_second_pass_and_removes_it() {
    let name = "install_instance_adds_a_second_pass_and_removes_it";
    let Some(gpu) = software_gpu() else {
        skip(name, "no software adapter");
        return;
    };
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, FORMAT, false)));
    GradingFeature::install(&gpu, &pipeline).unwrap();
    let handle = GradingFeature::install_instance(&gpu, &pipeline, "b").unwrap();
    {
        let passes = pipeline.borrow().passes();
        let find = |n: &str| passes.iter().find(|p| p.name == n).cloned().unwrap();
        let (a, b) = (find("grading"), find("grading#b"));
        assert_eq!(a.order, b.order);
        assert!(b.enabled);
    }
    assert_eq!(handle.amount(), 1.0);
    let err = GradingFeature::install_instance(&gpu, &pipeline, "b")
        .err()
        .unwrap();
    assert_eq!(err.code(), "pipeline.duplicate_pass");
    pipeline.borrow_mut().remove("grading#b").unwrap();
    assert!(
        pipeline
            .borrow()
            .passes()
            .iter()
            .all(|p| p.name != "grading#b")
    );
}

#[test]
fn bound_chain_frames_return_ok() {
    let name = "bound_chain_frames_return_ok";
    let Some(gpu) = software_gpu() else {
        skip(name, "no software adapter");
        return;
    };
    let target = OffscreenTarget::new(&gpu, SIZE, SIZE, FORMAT);
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, FORMAT, false)));
    GradingFeature::install(&gpu, &pipeline).unwrap();
    GradingFeature::install_instance(&gpu, &pipeline, "b").unwrap();
    let count = Rc::new(Cell::new(0));
    for pass in ["grading", "grading#b"] {
        let frame = BindFrame {
            pass: pass.to_owned(),
            views: [pool_view(&gpu), pool_view(&gpu)],
            bound_in_record: count.clone(),
        };
        pipeline
            .borrow_mut()
            .add_graph_frame(&format!("bind.{pass}"), Box::new(frame))
            .unwrap();
    }
    let mut graph = FrameGraph::new(FORMAT);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().unwrap();
    for _ in 0..2 {
        let commands = graph.execute(&gpu, target.view(), (SIZE, SIZE)).unwrap();
        gpu.queue().submit([commands]);
    }
    assert_eq!(count.get(), 4);
}
