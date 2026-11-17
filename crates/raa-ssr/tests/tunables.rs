//! S56 tunables of raa-ssr: keys and defaults in table order, and the trace module with
//! every override at its default passes naga validation.

#[path = "../src/shaders.rs"]
mod shaders;

use raa_shader::Composer;

#[test]
fn keys_and_defaults_in_table_order() {
    let specs = raa_ssr::tunables();
    let keys: Vec<&str> = specs.iter().map(|s| s.key).collect();
    assert_eq!(keys, ["maxSteps", "refineSteps", "edgeFade"]);
    let defaults: Vec<(f64, bool)> = specs.iter().map(|s| (s.default, s.integer)).collect();
    assert_eq!(defaults, [(64.0, true), (8.0, true), (0.1, false)]);
    for s in specs {
        assert_eq!(
            (s.min, s.max, s.rebuild),
            (None, None, "pipeline"),
            "{}",
            s.key
        );
        assert!(
            s.source
                .starts_with("crates/raa-ssr/src/wgsl/ssr_trace.wgsl:")
        );
    }
}

#[test]
fn sources_name_the_override_lines() {
    let lines: Vec<&str> = shaders::SSR_TRACE_WGSL.lines().collect();
    let want = [
        "override MAX_STEPS: u32 = 64u;",
        "override REFINE_STEPS: u32 = 8u;",
        "override EDGE_FADE: f32 = 0.1;",
    ];
    for (spec, text) in raa_ssr::tunables().iter().zip(want) {
        let line: usize = spec.source.rsplit(':').next().unwrap().parse().unwrap();
        assert_eq!(lines[line - 1], text, "{}", spec.key);
    }
}

#[test]
fn trace_module_with_default_overrides_validates() {
    if let Err(e) = Composer::validate(&shaders::ssr_trace_module_wgsl()) {
        panic!("ssr_trace: {e}");
    }
}
