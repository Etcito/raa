//! Contract `tunable-promotion` for raa-taa: keys, defaults and the module with defaults.

use raa_core::testing::software_gpu;
use raa_pipeline::{Pipeline, validate_tunable};
use raa_shader::Composer;
use raa_taa::{TaaFeature, taa_module_source, tunables};
use std::cell::RefCell;
use std::rc::Rc;

const KEYS: [&str; 12] = [
    "filterSharpness",
    "clipTolerance",
    "blendBase",
    "blendMotionGain",
    "blendMin",
    "blendMax",
    "historySharpness",
    "blendStatic",
    "clipStatic",
    "upsampleConfidence",
    "upsampleClipScene",
    "upsampleKernel",
];
const DEFAULTS: [f64; 12] = [8.0, 1.25, 0.04, 0.03, 0.04, 0.5, 0.6, 0.015, 3.0, 3.0, 1.0, 1.0];

#[test]
fn keys_and_defaults_in_table_order() {
    let specs = tunables();
    assert_eq!(specs.len(), 12);
    for (i, spec) in specs.iter().enumerate() {
        assert_eq!(spec.key, KEYS[i]);
        assert_eq!(spec.default, DEFAULTS[i]);
        assert!(!spec.integer);
        assert_eq!(spec.rebuild, "pipeline");
        assert!(spec.min.is_none() && spec.max.is_none());
    }
}

#[test]
fn validation_refuses_unknown_and_non_finite() {
    assert!(validate_tunable(tunables(), "nope", 1.0).is_err());
    assert!(validate_tunable(tunables(), "blendMin", f64::NAN).is_err());
    assert!(validate_tunable(tunables(), "blendMin", 0.1).is_ok());
}

#[test]
fn set_tunable_refusals_carry_post_invalid() {
    let Some(gpu) = software_gpu() else {
        println!("SKIP set_tunable_refusals_carry_post_invalid: no software adapter");
        return;
    };
    let pipeline = Rc::new(RefCell::new(Pipeline::new(
        &gpu,
        raa_core::wgpu::TextureFormat::Rgba8Unorm,
        false,
    )));
    let frame = gpu
        .device()
        .create_buffer(&raa_core::wgpu::BufferDescriptor {
            label: None,
            size: 768,
            usage: raa_core::wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
    let handle = TaaFeature::install(&gpu, &pipeline, &frame).expect("install");
    for (key, value) in [("nope", 1.0), ("blendMax", f64::NAN)] {
        let err = handle.set_tunable(key, value).expect_err("refused");
        assert_eq!(err.code(), "post.invalid");
    }
    assert!(handle.set_tunable("blendMax", 0.5).is_ok());
}

#[test]
fn module_with_default_overrides_validates() {
    let source = taa_module_source();
    for name in [
        "RAA_TAA_FILTER_SHARPNESS",
        "RAA_TAA_CLIP_TOLERANCE",
        "RAA_TAA_BLEND_BASE",
        "RAA_TAA_BLEND_MOTION_GAIN",
        "RAA_TAA_BLEND_MIN",
        "RAA_TAA_BLEND_MAX",
        "RAA_TAA_HISTORY_SHARPNESS",
        "RAA_TAA_BLEND_STATIC",
        "RAA_TAA_CLIP_STATIC",
        "RAA_TAA_UPSAMPLE_CONFIDENCE",
        "RAA_TAA_UPSAMPLE_CLIP_SCENE",
        "RAA_TAA_UPSAMPLE_KERNEL",
    ] {
        assert!(source.contains(&format!("override {name}: f32")));
    }
    assert_eq!(Composer::validate(&source), Ok(()));
}
