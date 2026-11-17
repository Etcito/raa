//! S56: the raa-shafts tunable list and its default module.

#[allow(dead_code)]
#[path = "../src/shaders.rs"]
mod shaders;

use raa_shader::Composer;

#[test]
fn keys_and_defaults_in_table_order() {
    let specs = raa_shafts::tunables();
    let rows: Vec<(&str, f64, bool)> = specs
        .iter()
        .map(|s| (s.key, s.default, s.integer))
        .collect();
    assert_eq!(rows, [("steps", 32.0, true), ("hgG", 0.7, false)]);
    assert!(
        specs
            .iter()
            .all(|s| s.rebuild == "pipeline" && s.min.is_none() && s.max.is_none())
    );
}

#[test]
fn module_with_default_overrides_validates() {
    let wgsl = shaders::shafts_module_wgsl();
    assert!(wgsl.contains("override STEPS: u32 = 32u;"));
    assert!(wgsl.contains("override HG_G: f32 = 0.7;"));
    if let Err(e) = Composer::validate(&wgsl) {
        panic!("shafts module: {e:?}");
    }
}
