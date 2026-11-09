//! CPU validation of the diagnostic shader and its pass wiring.
use raa_shader::Composer;

const SOURCE: &str = include_str!("../src/motion_display.rs");

fn shader() -> &'static str {
    SOURCE
        .split_once("const SHADER: &str = r#\"")
        .unwrap()
        .1
        .split_once("\"#;")
        .unwrap()
        .0
}

#[test]
fn shader_entrypoints_validate() {
    let wgsl = shader();
    assert!(wgsl.contains("@vertex fn vs_main("));
    assert!(wgsl.contains("@fragment fn fs_main("));
    Composer::validate(wgsl).expect("motion display WGSL validates");
}

#[test]
fn signed_colour_legend_and_zero() {
    let wgsl = shader();
    for expression in [
        "* 20.0",
        "positive.x + negative.y",
        "positive.y + negative.x",
        "negative.x + negative.y",
        "vec3<f32>(0.1) + 0.9 * rgb",
    ] {
        assert!(wgsl.contains(expression), "missing {expression}");
    }
    // The two axes and both signs have distinct channels, including the zero
    // written for sky, unsupported geometry, and invalid history.
    let colour = |x: f32, y: f32| {
        let x = (x * 20.0).clamp(-1.0, 1.0);
        let y = (y * 20.0).clamp(-1.0, 1.0);
        let (xp, xn, yp, yn) = (x.max(0.0), (-x).max(0.0), y.max(0.0), (-y).max(0.0));
        [xp + yn, yp + xn, xn + yn].map(|c| (0.1 + 0.9 * c).min(1.0))
    };
    assert_eq!(colour(0.0, 0.0), [0.1, 0.1, 0.1]);
    assert_eq!(colour(0.05, 0.0), [1.0, 0.1, 0.1]);
    assert_eq!(colour(-0.05, 0.0), [0.1, 1.0, 1.0]);
    assert_eq!(colour(0.0, 0.05), [0.1, 1.0, 0.1]);
    assert_eq!(colour(0.0, -0.05), [1.0, 0.1, 1.0]);
}

#[test]
fn installation_is_disabled_and_uses_output_overlay() {
    let _: fn(
        &raa_core::GpuContext,
        &std::rc::Rc<std::cell::RefCell<raa_pipeline::Pipeline>>,
    ) -> Result<(), raa_core::RaaError> = raa_lit::install_motion_display;
    assert!(SOURCE.contains("pipeline.set_enabled(\"motion.display\", false)"));
    assert!(SOURCE.contains("fn output_overlay(&self) -> bool"));
    assert!(SOURCE.contains("fn order(&self) -> u32"));
    assert!(SOURCE.contains("ctx.targets.motion()"));
    assert!(SOURCE.contains("view: ctx.backbuffer"));
}
