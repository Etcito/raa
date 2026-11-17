//! S56: raa-dof's tunable list (keys in table order, defaults) and the gather module with
//! its overrides at their defaults passes naga validation. Refusals of
//! `DofHandle::set_tunable` are tested inline in src/feature.rs (the handle needs a device).

#[path = "../src/shaders.rs"]
#[allow(dead_code)]
mod shaders;

use raa_dof::{DofDesc, tunables};
use raa_shader::Composer;

#[test]
fn keys_in_table_order_with_todays_defaults() {
    let t = tunables();
    let keys: Vec<&str> = t.iter().map(|s| s.key).collect();
    assert_eq!(keys, ["taps", "radiusPerCoc"]);
    assert_eq!(t[0].default, 48.0);
    assert!(t[0].integer);
    assert_eq!(t[1].default, 0.5);
    assert!(!t[1].integer);
    assert!(t.iter().all(|s| s.rebuild == "pipeline" && s.min.is_none()));
    assert_eq!(t[0].source, "crates/raa-dof/src/wgsl/dof_gather.wgsl:37");
    assert_eq!(t[1].source, "crates/raa-dof/src/wgsl/dof_gather.wgsl:41");
}

#[test]
fn gather_module_with_default_overrides_validates() {
    let wgsl = shaders::DOF_GATHER_WGSL;
    assert!(wgsl.contains("override TAPS: u32 = 48u;"));
    assert!(wgsl.contains("override RADIUS_PER_COC: f32 = 0.5;"));
    if let Err(e) = Composer::validate(wgsl) {
        panic!("dof_gather: {e}");
    }
    let _ = DofDesc::default();
}
