//! S56 tunables of raa-contact: the key list in table order with today's defaults, and the
//! module with its override at the default passes naga validation. set_tunable refusals are
//! covered by the inline tests of src/feature.rs (no handle without a GPU here).

#[path = "../src/shaders.rs"]
#[allow(dead_code)]
mod shaders;

use raa_shader::Composer;

#[test]
fn keys_and_defaults_in_table_order() {
    let list = raa_contact::tunables();
    let keys: Vec<&str> = list.iter().map(|s| s.key).collect();
    assert_eq!(keys, ["steps"]);
    assert_eq!(list[0].default, 24.0);
    assert!(list[0].integer);
    assert_eq!(list[0].rebuild, "pipeline");
}

#[test]
fn module_with_override_at_default_validates() {
    assert!(shaders::CONTACT_WGSL.contains("override STEPS: u32 = 24u;"));
    if let Err(e) = Composer::validate(shaders::CONTACT_WGSL) {
        panic!("{e}");
    }
}
