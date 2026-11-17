//! S56 tunables of raa-sky: the key list in table order, hand-written defaults, and the LUT
//! module with every override at its default passing naga validation.

use raa_shader::Composer;

const ATMOSPHERE_WGSL: &str = include_str!("../src/wgsl/atmosphere.wgsl");
const ATMOSPHERE_LUTS_WGSL: &str = include_str!("../src/wgsl/atmosphere_luts.wgsl");

#[test]
fn atmosphere_keys_and_defaults_follow_the_table() {
    let specs = raa_sky::tunables();
    let expected = [("skyViewSteps", 32.0, true), ("horizonFade", 0.006, false)];
    assert!(specs.len() >= expected.len());
    for (spec, (key, default, integer)) in specs.iter().zip(expected) {
        assert_eq!(spec.key, key);
        assert_eq!(spec.default, default, "{key}");
        assert_eq!(spec.integer, integer, "{key}");
        assert_eq!(spec.rebuild, "pipeline", "{key}");
        assert_eq!((spec.min, spec.max), (None, None), "{key}: unstated range");
    }
    assert_eq!(specs, raa_sky::SKY_TUNABLES);
}

#[test]
fn source_lines_hold_the_override_declarations() {
    for (key, text) in [
        ("skyViewSteps", "override RAA_SKY_VIEW_STEPS: i32 = 32;"),
        (
            "horizonFade",
            "override raa_atmosphere_horizon_fade: f32 = 0.006;",
        ),
    ] {
        let spec = raa_sky::tunables()
            .iter()
            .find(|s| s.key == key)
            .expect("key listed");
        let line: usize = spec
            .source
            .rsplit(':')
            .next()
            .and_then(|n| n.parse().ok())
            .expect("file:line");
        assert_eq!(
            ATMOSPHERE_LUTS_WGSL.lines().nth(line - 1),
            Some(text),
            "{key}"
        );
    }
}

#[test]
fn lut_module_with_default_overrides_validates() {
    let wgsl = format!("{ATMOSPHERE_WGSL}\n{ATMOSPHERE_LUTS_WGSL}\n");
    if let Err(e) = Composer::validate(&wgsl) {
        panic!("atmosphere LUT module refused: {e}");
    }
}
