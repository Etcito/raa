#![cfg(not(target_arch = "wasm32"))]

use std::{cell::RefCell, rc::Rc};

use raa_core::testing::{skip, software_gpu};
use raa_core::{FrameGraph, OffscreenTarget, wgpu};
use raa_gtao::stages::{AoStageKind, install_stage, stage_entry};
use raa_pipeline::Pipeline;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

#[test]
fn trace_and_blur_install_publish_and_validate_values() {
    let Some(gpu) = software_gpu() else {
        skip(
            "trace_and_blur_install_publish_and_validate_values",
            "software_gpu returned None",
        );
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let target = OffscreenTarget::new(&gpu, 32, 32, FORMAT);
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, FORMAT, false)));
    let t = install_stage(&gpu, &pipeline, AoStageKind::Trace, "t").unwrap();
    let b = install_stage(&gpu, &pipeline, AoStageKind::Blur, "b").unwrap();
    assert_eq!(t.entry(), "ao.trace#t");
    assert_eq!(b.entry(), "ao.blur#b");
    assert_eq!(
        pipeline
            .borrow()
            .passes()
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>(),
        ["ao.trace#t", "ao.blur#b"]
    );
    assert_eq!(
        install_stage(&gpu, &pipeline, AoStageKind::Trace, "t")
            .err()
            .unwrap()
            .code(),
        "pipeline.duplicate_pass"
    );
    let mut graph = FrameGraph::new(FORMAT);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().unwrap();
    let commands = graph.execute(&gpu, target.view(), (32, 32)).unwrap();
    gpu.queue().submit([commands]);
    assert_eq!(
        t.published(),
        [(
            "raw",
            raa_pipeline::ImageDesc::new(
                raa_pipeline::ImageDim::D2,
                wgpu::TextureFormat::R32Float,
                32,
                32,
                1,
                1
            )
        )]
    );
    assert_eq!(
        b.published()
            .iter()
            .map(|(p, d)| (*p, d.format, d.width, d.height))
            .collect::<Vec<_>>(),
        [
            ("ao", wgpu::TextureFormat::Rgba16Float, 32, 32),
            ("blur", wgpu::TextureFormat::R32Float, 32, 32)
        ]
    );
    assert!(!t.ran());
    assert!(!b.ran());
    let bad = |key: &str, value| vec![(key.to_owned(), value)];
    assert_eq!(
        t.set_values(&bad("radius", 0.0)).unwrap_err().code(),
        "ao.invalid"
    );
    assert_eq!(
        t.set_values(&bad("slices", 2.5)).unwrap_err().code(),
        "ao.invalid"
    );
    t.set_values(&[("radius".into(), 1.0), ("slices".into(), 4.0)])
        .unwrap();
    assert_eq!(
        b.set_values(&bad("radius", 1.0)).unwrap_err().code(),
        "ao.invalid"
    );
    assert_eq!(
        AoStageKind::from_type_id("ao.blur"),
        Some(AoStageKind::Blur)
    );
    assert_eq!(stage_entry(AoStageKind::Trace, "x"), "ao.trace#x");
    assert!(raa_core::testing::block_on(errors.pop()).is_none());
}
