//! `BloomFeature::install_instance` and the bound HDR chain (contracts
//! rg-install-instance and rg-hdr-chain-binding).
#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;
use std::rc::Rc;

use raa_bloom::BloomFeature;
use raa_core::testing::{skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, OffscreenTarget, RaaError, wgpu};
use raa_pipeline::{GraphFrame, Pipeline, RecordCtx};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SIZE: u32 = 32;

fn setup() -> Option<(GpuContext, Rc<RefCell<Pipeline>>)> {
    let Some(gpu) = software_gpu() else {
        skip("bloom_instance", "no software adapter");
        return None;
    };
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, FORMAT, false)));
    BloomFeature::install(&gpu, &pipeline).unwrap();
    BloomFeature::install_instance(&gpu, &pipeline, "b").unwrap();
    Some((gpu, pipeline))
}

#[test]
fn instance_is_added_refused_twice_and_removed() {
    let Some((gpu, pipeline)) = setup() else {
        return;
    };
    let passes = pipeline.borrow().passes();
    let find = |n: &str| passes.iter().find(|i| i.name == n).unwrap();
    assert_eq!(find("bloom").order, find("bloom#b").order);
    assert!(find("bloom#b").enabled);
    let err = BloomFeature::install_instance(&gpu, &pipeline, "b")
        .err()
        .unwrap();
    assert_eq!(err.code(), "pipeline.duplicate_pass");
    assert!(pipeline.borrow_mut().remove("bloom#b").is_ok());
}

/// Binds the chain to two pool-usage textures around each bloom pass record.
struct Bind {
    src: wgpu::TextureView,
    dst: wgpu::TextureView,
}

impl GraphFrame for Bind {
    fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if pass.starts_with("bloom") {
            ctx.targets.hdr().bind(&self.src, &self.dst);
            assert!(ctx.targets.hdr().is_bound());
        }
        Ok(())
    }

    fn after_record(&mut self, _pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        ctx.targets.hdr().unbind();
        Ok(())
    }
}

fn pool_view(gpu: &GpuContext) -> wgpu::TextureView {
    use wgpu::TextureUsages as U;
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
            usage: U::RENDER_ATTACHMENT
                | U::TEXTURE_BINDING
                | U::STORAGE_BINDING
                | U::COPY_SRC
                | U::COPY_DST,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

#[test]
fn two_frames_with_bound_chain_succeed() {
    let Some((gpu, pipeline)) = setup() else {
        return;
    };
    // Only the two bloom passes run; the pool textures start cleared (zero).
    let frame = Bind {
        src: pool_view(&gpu),
        dst: pool_view(&gpu),
    };
    pipeline
        .borrow_mut()
        .add_graph_frame("test.bind", Box::new(frame))
        .unwrap();
    let target = OffscreenTarget::new(&gpu, SIZE, SIZE, FORMAT);
    let mut graph = FrameGraph::new(FORMAT);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().unwrap();
    for _ in 0..2 {
        let commands = graph.execute(&gpu, target.view(), (SIZE, SIZE)).unwrap();
        gpu.queue().submit([commands]);
    }
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
}
