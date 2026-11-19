#![cfg(not(target_arch = "wasm32"))]

use std::{cell::RefCell, rc::Rc};

use raa_bloom::stages::{BloomStageKind, install_stage};
use raa_core::testing::{skip, software_gpu};
use raa_core::{FrameGraph, OffscreenTarget, wgpu};
use raa_pipeline::Pipeline;

#[test]
fn bloom_levels_install_and_publish_without_graph_inputs() {
    let Some(gpu) = software_gpu() else {
        skip("bloom_stages", "no software adapter");
        return;
    };
    let pipeline = Rc::new(RefCell::new(Pipeline::new(
        &gpu,
        wgpu::TextureFormat::Rgba8Unorm,
        false,
    )));
    let down = install_stage(&gpu, &pipeline, BloomStageKind::Down, "d1").unwrap();
    let up = install_stage(&gpu, &pipeline, BloomStageKind::Up, "u1").unwrap();
    assert_eq!(down.entry(), "bloom.down#d1");
    assert_eq!(up.entry(), "bloom.up#u1");
    assert_eq!(
        install_stage(&gpu, &pipeline, BloomStageKind::Down, "d1")
            .err()
            .unwrap()
            .code(),
        "pipeline.duplicate_pass"
    );

    let mut graph = FrameGraph::new(wgpu::TextureFormat::Rgba8Unorm);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().unwrap();
    let target = OffscreenTarget::new(&gpu, 64, 64, wgpu::TextureFormat::Rgba8Unorm);
    gpu.queue()
        .submit([graph.execute(&gpu, target.view(), (64, 64)).unwrap()]);
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let published = down.published();
    assert_eq!(published.len(), 1);
    assert_eq!(published[0].0, "level");
    assert_eq!(
        (
            published[0].1.width,
            published[0].1.height,
            published[0].1.format
        ),
        (32, 32, wgpu::TextureFormat::Rgba16Float)
    );
    assert!(up.published().is_empty());
    assert!(!down.ran());
    assert!(!up.ran());

    assert_eq!(
        down.set_values(&[("karis".into(), 0.5)])
            .err()
            .unwrap()
            .code(),
        "post.invalid"
    );
    assert!(down.set_values(&[("karis".into(), 1.0)]).is_ok());
    assert_eq!(
        up.set_values(&[("strength".into(), -1.0)])
            .err()
            .unwrap()
            .code(),
        "post.invalid"
    );
    assert_eq!(
        down.set_values(&[("strength".into(), 0.5)])
            .err()
            .unwrap()
            .code(),
        "post.invalid"
    );
}
