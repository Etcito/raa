//! m29 cluster bake, share 1 (S31): the pure helpers run natively. The wasm32 runtime
//! (`install`, `frame`) is not reachable here; the section validation and the state report
//! append their tests to this file in the card that owns them.

#![cfg(not(target_arch = "wasm32"))]

#[path = "../src/api/m29_cluster_bake.rs"]
mod m29;

use m29::{
    ClusterReport, MESH_EMPTY, MESH_TOO_LARGE, MESH_VERTEX_COLORS, OptionsPatch, SectionWire,
    VG_INVALID, bake_input, build_options, check_bake_budget, check_budget_ms, check_vertex_colors,
    cluster_error, cluster_report, plan_attach, section_from_wire, stage_name, vg_code,
};
use raa_cluster::error::{EMPTY, INVALID_MESH, TOO_LARGE};
use raa_cluster::{
    BakeStage, BuildOptions, ClusterError, ClusterMesh, ClusterSource, ClusterState, ClusterStore,
    MeshClusterStatus, OwnedClusterInput,
};
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

/// A baked displaced sphere with its section arrays, as the JS side holds them.
struct Baked {
    mesh: ClusterMesh,
    vertices: Vec<u16>,
    clusters: Vec<u8>,
    nodes: Vec<u8>,
}

fn baked(subdivisions: u32) -> Baked {
    let input = OwnedClusterInput::displaced_sphere(subdivisions);
    let mesh = raa_cluster::bake(input.as_input(), &BuildOptions::default()).unwrap();
    let (clusters, nodes) = mesh.to_section_bytes();
    let vertices = mesh.vertices.iter().flat_map(|v| v.data).collect();
    Baked {
        mesh,
        vertices,
        clusters,
        nodes,
    }
}

impl Baked {
    fn wire(&self) -> SectionWire<'_> {
        let m = &self.mesh;
        SectionWire {
            version: 1.0,
            algorithm: 1.0,
            vertices: &self.vertices,
            triangles: &m.triangles,
            clusters: &self.clusters,
            nodes: &self.nodes,
            root_node: m.root_node.into(),
            hierarchy_depth: m.hierarchy_depth.into(),
            levels: m.levels.into(),
            quant_origin: m.quant_origin.map(f64::from),
            quant_step: m.quant_step.into(),
            uv_origin: m.uv_origin.map(f64::from),
            uv_step: m.uv_step.map(f64::from),
            bounds_center: m.bounds_center.map(f64::from),
            bounds_radius: m.bounds_radius.into(),
            source_triangles: m.source_triangles as f64,
        }
    }
}

/// The code and the message of a section refusal.
fn section_refusal(w: &SectionWire<'_>, at: &str) -> (&'static str, String) {
    let err = section_from_wire(w, at).unwrap_err();
    let (code, message) = refusal(&err);
    (code, message.to_string())
}

/// A refusal of `vg.invalid` whose message starts with the prefixed field name.
fn assert_refused(w: &SectionWire<'_>, at: &str, field: &str) {
    let (code, message) = section_refusal(w, at);
    assert_eq!(code, VG_INVALID, "{field}");
    assert!(
        message.starts_with(&format!("{at}{field}: ")),
        "{at}{field}: {message}"
    );
}

#[test]
fn a_baked_section_converts_back_to_an_equal_mesh() {
    let b = baked(2);
    assert_eq!(section_from_wire(&b.wire(), "").unwrap(), b.mesh);
}

#[test]
fn each_wire_rule_refuses_with_its_field_named() {
    let b = baked(2);
    let mut w = b.wire();
    w.version = 2.0;
    assert_refused(&w, "", "version");
    assert_refused(&w, "sections[1].", "version");

    let mut w = b.wire();
    w.triangles = &[];
    assert_refused(&w, "sections[1].", "triangles");

    let mut w = b.wire();
    w.clusters = &b.clusters[..47];
    assert_refused(&w, "", "clusters");
    assert_refused(&w, "sections[1].", "clusters");

    let mut w = b.wire();
    w.root_node = 1e9;
    assert_refused(&w, "", "rootNode");
    assert_refused(&w, "sections[1].", "rootNode");

    let mut w = b.wire();
    w.levels = 0.0;
    assert_refused(&w, "", "levels");
    assert_refused(&w, "sections[1].", "levels");

    let mut w = b.wire();
    w.quant_step = f64::NAN;
    assert_refused(&w, "", "quantStep");
    assert_refused(&w, "sections[1].", "quantStep");

    let mut w = b.wire();
    w.bounds_radius = -1.0;
    assert_refused(&w, "", "boundsRadius");
    assert_refused(&w, "sections[1].", "boundsRadius");

    // The complete validation: an index of 255 is outside every cluster (at most 255 vertices).
    let mut triangles = b.mesh.triangles.clone();
    triangles[0] |= 0xff;
    let mut w = b.wire();
    w.triangles = &triangles;
    let (code, message) = section_refusal(&w, "sections[1].");
    assert_eq!(code, VG_INVALID);
    assert!(
        message.starts_with("sections[1].clusters[") && message.contains(".triangle["),
        "{message}"
    );
}

#[test]
fn the_report_names_each_store_state() {
    let empty = ClusterStore::new();
    let none = cluster_report(&empty.state(1), empty.sections(1));
    assert_eq!(
        none,
        ClusterReport {
            state: "none",
            progress: 0.0,
            source: None,
            code: None,
            message: None,
            sections: 0,
            clusters: 0,
            source_triangles: 0,
            levels: 0,
        }
    );

    let baking = MeshClusterStatus {
        state: ClusterState::Baking,
        progress: 0.25,
        source: Some(ClusterSource::Runtime),
        code: None,
        message: None,
    };
    let r = cluster_report(&baking, None);
    assert_eq!(
        (r.state, r.progress, r.source),
        ("baking", 0.25, Some("runtime"))
    );
    assert_eq!((r.sections, r.clusters, r.levels), (0, 0, 0));

    let mut store = ClusterStore::new();
    let sections = vec![baked(2).mesh, baked(1).mesh];
    let clusters: u32 = sections.iter().map(|s| s.clusters.len() as u32).sum();
    let triangles: u64 = sections.iter().map(|s| s.source_triangles).sum();
    let levels = sections.iter().map(|s| s.levels).max().unwrap();
    store.attach(2, sections);
    let r = cluster_report(&store.state(2), store.sections(2));
    assert_eq!(
        (r.state, r.source, r.sections),
        ("ready", Some("attached"), 2)
    );
    assert_eq!(
        (r.clusters, r.source_triangles, r.levels),
        (clusters, triangles, levels)
    );

    store
        .start_bake(
            3,
            vec![OwnedClusterInput::single_triangle()],
            &BuildOptions::default(),
        )
        .unwrap();
    store.step_all(&mut || true);
    let r = cluster_report(&store.state(3), store.sections(3));
    assert_eq!(
        (r.state, r.source, r.sections),
        ("ready", Some("runtime"), 1)
    );

    store.refuse(4, EMPTY, "no triangle survives quantisation".to_string());
    let r = cluster_report(&store.state(4), store.sections(4));
    assert_eq!(
        (r.state, r.code, r.message.as_deref(), r.source),
        (
            "refused",
            Some(MESH_EMPTY),
            Some("no triangle survives quantisation"),
            None
        )
    );
}

#[test]
fn an_attach_with_one_invalid_section_attaches_nothing() {
    let good = baked(1).mesh;
    let results = vec![
        Ok(good.clone()),
        Err(m29::vg_invalid("sections[1].version", "must be 1")),
    ];
    let err = plan_attach(2, results).unwrap_err();
    assert_eq!(refusal(&err).0, VG_INVALID);
    // Nothing is handed to the store: a full valid list is the only Ok outcome.
    assert_eq!(
        plan_attach(2, vec![Ok(good.clone()), Ok(good)])
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn an_attach_requires_one_section_per_sub_mesh() {
    let err = plan_attach(2, vec![Ok(baked(1).mesh)]).unwrap_err();
    let (code, message) = refusal(&err);
    assert_eq!(code, VG_INVALID);
    assert!(message.starts_with("sections: "), "{message}");
}

#[test]
fn the_engine_budget_accepts_0_5_and_100_and_refuses_beyond() {
    assert_eq!(check_bake_budget(0.5).unwrap(), 0.5);
    assert_eq!(check_bake_budget(100.0).unwrap(), 100.0);
    for ms in [0.4, 101.0, f64::NAN] {
        let err = check_bake_budget(ms).unwrap_err();
        let (code, message) = refusal(&err);
        assert_eq!(code, VG_INVALID, "{ms}");
        assert!(message.starts_with("ms: "), "{ms}: {message}");
    }
}

#[test]
fn a_colour_stream_is_refused_with_the_vertex_colour_code() {
    let err = check_vertex_colors(true, "mesh").unwrap_err();
    let (code, message) = refusal(&err);
    assert_eq!(code, MESH_VERTEX_COLORS);
    assert!(message.starts_with("mesh: "), "{message}");
    assert!(check_vertex_colors(false, "mesh").is_ok());
}
