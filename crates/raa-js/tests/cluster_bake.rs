//! m29 cluster bake, share 1 (S31): the pure helpers run natively. The wasm32 runtime
//! (`install`, `frame`) is not reachable here; the section validation and the state report
//! append their tests to this file in the card that owns them.

#![cfg(not(target_arch = "wasm32"))]

#[path = "../src/api/m29_cluster_bake.rs"]
mod m29;

use m29::{
    MESH_EMPTY, MESH_TOO_LARGE, MESH_VERTEX_COLORS, OptionsPatch, VG_INVALID, bake_input,
    build_options, check_budget_ms, cluster_error, stage_name, vg_code,
};
use raa_cluster::error::{EMPTY, INVALID_MESH, TOO_LARGE};
use raa_cluster::{BakeStage, BuildOptions, ClusterError};
use raa_core::RaaError;

/// The code and the message of a refusal; any other error variant fails the test.
fn refusal(e: &RaaError) -> (&'static str, &str) {
    match e {
        RaaError::FrameGraph { code, message } => (*code, message.as_str()),
        other => panic!("not a FrameGraph refusal: {other:?}"),
    }
}

fn patch(field: &str, value: f64) -> OptionsPatch {
    let mut p = OptionsPatch::default();
    match field {
        "maxClusterTriangles" => p.max_cluster_triangles = Some(value),
        "maxClusterVertices" => p.max_cluster_vertices = Some(value),
        "minReduction" => p.min_reduction = Some(value),
        "targetRatio" => p.target_ratio = Some(value),
        _ => unreachable!("{field}"),
    }
    p
}

/// One triangle: positions, normals and indices of a unit right triangle in the XY plane.
fn triangle() -> ([f32; 9], [f32; 9], [u32; 3]) {
    (
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        [0, 1, 2],
    )
}

#[test]
fn options_from_an_empty_patch_are_the_s31_defaults() {
    let o = build_options(&OptionsPatch::default()).unwrap();
    assert_eq!(o.max_cluster_triangles, 128);
    assert_eq!(o.max_cluster_vertices, 255);
    assert_eq!(o.min_reduction, 0.85);
    assert_eq!(o.target_ratio, 0.5);
    assert_eq!(o, BuildOptions::default());
}

#[test]
fn in_domain_options_are_accepted_at_their_bounds() {
    let lo = OptionsPatch {
        max_cluster_triangles: Some(8.0),
        max_cluster_vertices: Some(16.0),
        min_reduction: Some(0.5),
        target_ratio: Some(0.25),
    };
    let o = build_options(&lo).unwrap();
    assert_eq!((o.max_cluster_triangles, o.max_cluster_vertices), (8, 16));
    assert_eq!((o.min_reduction, o.target_ratio), (0.5, 0.25));
    let hi = OptionsPatch {
        max_cluster_triangles: Some(128.0),
        max_cluster_vertices: Some(255.0),
        ..OptionsPatch::default()
    };
    let o = build_options(&hi).unwrap();
    assert_eq!(
        (o.max_cluster_triangles, o.max_cluster_vertices),
        (128, 255)
    );
}

#[test]
fn each_option_domain_refuses_with_its_camel_case_field() {
    let cases = [
        ("maxClusterTriangles", 128.5),
        ("maxClusterTriangles", 7.0),
        ("maxClusterVertices", 256.0),
        ("minReduction", 1.0),
        ("targetRatio", 0.0),
        ("targetRatio", f64::NAN),
    ];
    for (field, value) in cases {
        let err = build_options(&patch(field, value)).unwrap_err();
        let (code, message) = refusal(&err);
        assert_eq!(code, VG_INVALID, "{field} = {value}");
        assert!(
            message.starts_with(&format!("options.{field}")),
            "{field} = {value}: {message}"
        );
    }
}

#[test]
fn cluster_errors_map_to_their_vg_codes_and_keep_the_path() {
    let cases = [
        (EMPTY, MESH_EMPTY),
        (TOO_LARGE, MESH_TOO_LARGE),
        (INVALID_MESH, VG_INVALID),
        ("cluster.not_finished", VG_INVALID),
    ];
    for (code, want) in cases {
        let e = ClusterError::at(code, "clusters[2]", "bad");
        let err = cluster_error(&e);
        let (got, message) = refusal(&err);
        assert_eq!(got, want, "{code}");
        assert_eq!(message, "clusters[2]: bad");
    }
}

#[test]
fn stored_codes_map_like_cluster_errors() {
    assert_eq!(vg_code(EMPTY), MESH_EMPTY);
    assert_eq!(vg_code(TOO_LARGE), MESH_TOO_LARGE);
    assert_eq!(vg_code(MESH_VERTEX_COLORS), MESH_VERTEX_COLORS);
    assert_eq!(vg_code(INVALID_MESH), VG_INVALID);
    assert_eq!(vg_code("cluster.unknown"), VG_INVALID);
}

#[test]
fn bake_input_accepts_one_triangle_and_names_each_bad_array() {
    let (p, n, i) = triangle();
    let owned = bake_input(&p, &n, None, &i).unwrap();
    assert_eq!(owned.mesh.positions.len(), 3);
    assert!(owned.uvs.is_none());

    let uvs = [0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0];
    assert!(bake_input(&p, &n, Some(&uvs), &i).unwrap().uvs.is_some());

    let short = [0.0f32, 0.0, 0.0, 1.0];
    let cases: [(&str, RaaError); 5] = [
        ("positions", bake_input(&short, &n, None, &i).unwrap_err()),
        ("normals", bake_input(&p, &n[..6], None, &i).unwrap_err()),
        ("uvs", bake_input(&p, &n, Some(&uvs[..4]), &i).unwrap_err()),
        ("indices", bake_input(&p, &n, None, &[0, 1, 3]).unwrap_err()),
        ("indices", bake_input(&p, &n, None, &[0, 1]).unwrap_err()),
    ];
    for (array, err) in &cases {
        let (code, message) = refusal(err);
        assert_eq!(code, VG_INVALID, "{array}");
        assert!(message.contains(array), "{array}: {message}");
    }
}

#[test]
fn bake_input_refuses_a_mesh_without_triangle_as_empty() {
    let (p, n, _) = triangle();
    let err = bake_input(&p, &n, None, &[]).unwrap_err();
    assert_eq!(refusal(&err).0, MESH_EMPTY);
}

#[test]
fn budget_accepts_the_range_and_refuses_outside_it() {
    assert_eq!(check_budget_ms(0.5, 0.5, 100.0, "budgetMs").unwrap(), 0.5);
    assert_eq!(
        check_budget_ms(100.0, 0.5, 100.0, "budgetMs").unwrap(),
        100.0
    );
    for ms in [0.49, 100.01, f64::NAN] {
        let err = check_budget_ms(ms, 0.5, 100.0, "budgetMs").unwrap_err();
        let (code, message) = refusal(&err);
        assert_eq!(code, VG_INVALID);
        assert!(message.starts_with("budgetMs"), "{message}");
    }
}

#[test]
fn stage_names_are_the_six_s31_strings() {
    let names = [
        (BakeStage::Weld, "weld"),
        (BakeStage::Partition, "partition"),
        (BakeStage::Level, "level"),
        (BakeStage::Hierarchy, "hierarchy"),
        (BakeStage::Pack, "pack"),
        (BakeStage::Done, "done"),
    ];
    for (stage, name) in names {
        assert_eq!(stage_name(stage), name);
    }
}
