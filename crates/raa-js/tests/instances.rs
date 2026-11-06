//! m26 instances: the registry exports the instance methods; the argument, LOD option, level
//! count, hit and position helpers of `src/api/m26_instances.rs` run natively.

use std::fs;
use std::path::Path;

use raa_core::RaaError;
use raa_lit::LodParams;

#[allow(dead_code)]
#[path = "../src/api/m26_instances.rs"]
mod m26;

use m26::{
    bias_arg, handle_arg, level_count, lod_options, lod_refusal, max_arg, nearest_hits, positions,
    same_submesh_count, vec3_arg,
};

fn code_of(e: &RaaError) -> &'static str {
    match e {
        RaaError::FrameGraph { code, .. } => code,
        _ => "other",
    }
}

#[test]
fn registry_exports_the_instance_methods() {
    let path = Path::new(env!("OUT_DIR")).join("api_registry.rs");
    let reg = fs::read_to_string(&path).expect("api_registry.rs");
    assert!(reg.contains("mod m26_instances;"));
    assert!(reg.contains("m26_instances::install(ctx)?;"));
    let src = include_str!("../src/api/m26_instances.rs");
    for name in [
        "meshLodError",
        "setMeshLods",
        "clearMeshLods",
        "setInstanceLodBias",
        "setLodOptions",
        "queryInstancesRay",
        "instanceStats",
    ] {
        assert!(src.contains(&format!("js_name = {name}")), "no {name}");
    }
}

#[test]
fn bias_is_a_whole_number_from_minus_three_to_three() {
    assert_eq!(bias_arg(Some(-3.0)).unwrap(), -3);
    assert_eq!(bias_arg(Some(0.0)).unwrap(), 0);
    assert_eq!(bias_arg(Some(3.0)).unwrap(), 3);
    for bad in [Some(3.5), Some(4.0), Some(-4.0), Some(f64::NAN), None] {
        let e = bias_arg(bad).unwrap_err();
        assert_eq!(code_of(&e), "api.invalid_argument");
    }
}

#[test]
fn max_defaults_to_64_and_is_bounded() {
    assert_eq!(max_arg(None).unwrap(), 64);
    assert_eq!(max_arg(Some(1.0)).unwrap(), 1);
    assert_eq!(max_arg(Some(16384.0)).unwrap(), 16384);
    for bad in [0.0, -1.0, 2.5, 16385.0, f64::INFINITY] {
        let e = max_arg(Some(bad)).unwrap_err();
        assert_eq!(code_of(&e), "api.invalid_argument");
    }
}

#[test]
fn handles_are_whole_u32_numbers() {
    assert_eq!(handle_arg("mesh", 7.0).unwrap(), 7);
    assert!(handle_arg("mesh", -1.0).is_err());
    assert!(handle_arg("mesh", 1.5).is_err());
    assert!(handle_arg("mesh", 4294967296.0).is_err());
}

#[test]
fn vectors_are_three_finite_numbers() {
    assert_eq!(
        vec3_arg("dir", &[0.0, 1.0, -2.0]).unwrap(),
        [0.0, 1.0, -2.0]
    );
    assert!(vec3_arg("dir", &[0.0, 1.0]).is_err());
    assert!(vec3_arg("dir", &[0.0, f64::NAN, 1.0]).is_err());
    assert!(vec3_arg("dir", &[1e300, 0.0, 0.0]).is_err());
}

#[test]
fn lod_options_keep_absent_keys_and_check_ranges() {
    let base = LodParams::default();
    let (c, s) = lod_options(base, base, None, None, None, None).unwrap();
    assert_eq!(c, base);
    assert_eq!(s, base);
    let (c, s) = lod_options(base, base, Some(2.5), Some(0.4), Some(1.0), Some(-2.0)).unwrap();
    assert_eq!(c.threshold_px, 2.5);
    assert_eq!(c.hysteresis, 0.4);
    assert_eq!(c.bias, 1.0);
    assert_eq!(s.threshold_px, 1.0, "shadows always use one texel");
    assert_eq!(s.hysteresis, 0.4);
    assert_eq!(s.bias, -2.0);
    let (c, _) = lod_options(base, base, Some(2.0), None, None, None).unwrap();
    assert_eq!((c.threshold_px, c.hysteresis, c.bias), (2.0, 0.2, 0.0));
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e300] {
        let e = lod_options(base, base, Some(bad), None, None, None).unwrap_err();
        assert_eq!(code_of(&e), "api.invalid_argument");
    }
    for bad in [-0.1, 0.95, f64::NAN] {
        let e = lod_options(base, base, None, Some(bad), None, None).unwrap_err();
        assert_eq!(code_of(&e), "api.invalid_argument");
    }
    for bad in [f64::NAN, f64::INFINITY] {
        for (bias, shadow) in [(Some(bad), None), (None, Some(bad))] {
            let e = lod_options(base, base, None, None, bias, shadow).unwrap_err();
            assert_eq!(code_of(&e), "api.invalid_argument");
        }
    }
}

#[test]
fn shadow_bias_defaults_to_zero_and_keeps_its_own_value() {
    let base = LodParams::default();
    let shadow = LodParams { bias: 1.5, ..base };
    let (c, s) = lod_options(base, shadow, None, None, None, None).unwrap();
    assert_eq!(c.bias, 0.0);
    assert_eq!(s.bias, 1.5);
    let (_, s) = lod_options(base, shadow, None, None, Some(2.0), None).unwrap();
    assert_eq!(s.bias, 1.5, "bias leaves shadowBias alone");
}

#[test]
fn a_chain_has_one_to_seven_coarser_levels() {
    assert_eq!(code_of(&level_count(0).unwrap_err()), "lod.invalid");
    level_count(1).unwrap();
    level_count(7).unwrap();
    assert_eq!(code_of(&level_count(8).unwrap_err()), "lod.too_many_levels");
}

#[test]
fn levels_have_the_sub_mesh_count_of_the_full_mesh() {
    same_submesh_count(2, 2).unwrap();
    assert_eq!(
        code_of(&same_submesh_count(2, 3).unwrap_err()),
        "lod.invalid"
    );
}

#[test]
fn positions_are_xyz_triples() {
    let p = positions(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(p, vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert_eq!(
        code_of(&positions(&[1.0, 2.0]).unwrap_err()),
        "api.invalid_argument"
    );
}

#[test]
fn ray_hits_keep_the_smallest_t_per_handle_sorted_and_truncated() {
    let hits = [(3, 5.0), (1, 2.0), (3, 1.0), (2, 2.0), (1, 4.0)];
    assert_eq!(nearest_hits(&hits, 64), vec![(3, 1.0), (1, 2.0), (2, 2.0)]);
    assert_eq!(nearest_hits(&hits, 2), vec![(3, 1.0), (1, 2.0)]);
    assert!(nearest_hits(&[], 4).is_empty());
}

#[test]
fn raa_lit_refusals_map_to_lod_invalid() {
    let e = lod_refusal(RaaError::FrameGraph {
        code: "render.invalid_lod",
        message: "bad".into(),
    });
    assert_eq!(code_of(&e), "lod.invalid");
}
