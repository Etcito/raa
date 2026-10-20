//! Conformance with the published raa.prims v0 format (raa-assets-tools, `schema/` and
//! `src/format/`): every valid published fixture parses and builds, each primitive
//! tessellates to the exact box that raa-assets-tools gives (`bounds.ts`, `localBounds`),
//! the published param names map to the same meshes as the S3 names, and a consumer
//! project's bean asset, which is not valid v0, is refused at its budget.
//! The fixtures are copies, see `tests/fixtures/README.md`.

use std::collections::HashMap;

use raa_geom::MeshData;
use raa_geom::analysis::mesh_hash;
use raa_geom::{GeometryRegistry, Quality};
use raa_loader::json::{self, JsonValue};
use raa_loader::prims::{build, parse};
use raa_loader::{
    BooleanHost, BooleanResolver, BooleanTool, LoadContext, LoadError, LoadedAsset, NoBooleans,
};

/// Bounds tolerance: each half extent of the tessellated mesh is within
/// `BOUNDS_TOL * max(1, largest half extent)` of the exact box (the azimuth and polar rings
/// of the S3 tessellation hit the axes exactly, so the measured error is far below this).
const BOUNDS_TOL: f64 = 1e-5;
/// Shortfall tolerance of the sampled kinds at LOD0 (see `tolerance`).
const SAMPLED_TOL: f64 = 0.01;

fn registry() -> GeometryRegistry {
    let mut g = GeometryRegistry::new();
    raa_prims::register_all(&mut g).expect("register_all");
    g
}

fn ctx(g: &GeometryRegistry) -> LoadContext<'_> {
    LoadContext {
        geometry: g,
        booleans: &NoBooleans,
        quality: Quality::LOD0,
        strict_kinds: true,
    }
}

/// A resolver that does not cut: it returns each host as is. raa-loader cannot depend on
/// raa-csg (which installs the real one), so the fixtures with subtract/intersect children
/// build with this; only childless leaves, which no cut touches, are compared to bounds.
struct HostOnly;

impl BooleanResolver for HostOnly {
    fn resolve(
        &self,
        host: &BooleanHost<'_>,
        _tools: &[BooleanTool<'_>],
        out: &mut MeshData,
    ) -> Result<(), LoadError> {
        out.clear();
        out.positions.extend_from_slice(&host.mesh.positions);
        out.normals.extend_from_slice(&host.mesh.normals);
        out.indices.extend_from_slice(&host.mesh.indices);
        Ok(())
    }
}

fn host_only_ctx(g: &GeometryRegistry) -> LoadContext<'_> {
    LoadContext {
        geometry: g,
        booleans: &HostOnly,
        quality: Quality::LOD0,
        strict_kinds: true,
    }
}

fn read(rel: &str) -> String {
    let path = format!("{}/tests/fixtures/{rel}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// Published valid fixtures, by file name.
fn published() -> Vec<String> {
    let dir = format!("{}/tests/fixtures/published", env!("CARGO_MANIFEST_DIR"));
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{dir}: {e}"))
        .map(|e| {
            e.expect("dir entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter(|n| n.ends_with(".raa.prims.json"))
        .collect();
    names.sort();
    assert!(names.len() >= 11, "published fixtures missing: {names:?}");
    names
}

fn field<'a>(o: &'a JsonValue, key: &str) -> &'a JsonValue {
    match o {
        JsonValue::Object(members) => members
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
            .unwrap_or_else(|| panic!("no `{key}`")),
        other => panic!("not an object: {other:?}"),
    }
}

fn num(v: &JsonValue) -> f64 {
    match v {
        JsonValue::Number(x) => *x,
        other => panic!("not a number: {other:?}"),
    }
}

fn vec3(v: &JsonValue) -> [f64; 3] {
    match v {
        JsonValue::Array(a) if a.len() == 3 => [num(&a[0]), num(&a[1]), num(&a[2])],
        other => panic!("not a vec3: {other:?}"),
    }
}

/// Half extents of the exact box of a published primitive: raa-assets-tools' `localBounds`
/// (box and roundBox size / 2; capsule and cylinder [r, height / 2, r]; cone
/// [max(radii), height / 2, max(radii)]; torus [R + r, r, R + r]; ellipsoid radii).
fn exact_half_extents(kind: &str, p: &JsonValue) -> [f64; 3] {
    match kind {
        "sphere" => [num(field(p, "radius")); 3],
        "box" | "roundBox" => vec3(field(p, "size")).map(|s| s / 2.0),
        "capsule" | "cylinder" => {
            let r = num(field(p, "radius"));
            [r, num(field(p, "height")) / 2.0, r]
        }
        "cone" => {
            let r = num(field(p, "radiusBottom")).max(num(field(p, "radiusTop")));
            [r, num(field(p, "height")) / 2.0, r]
        }
        "torus" => {
            let r = num(field(p, "majorRadius")) + num(field(p, "minorRadius"));
            [r, num(field(p, "minorRadius")), r]
        }
        "ellipsoid" => vec3(field(p, "radii")),
        other => panic!("kind `{other}` is not a published kind"),
    }
}

/// Node id -> (kind, params) of every leaf (a node without children) of a published asset.
fn leaves(doc: &JsonValue) -> HashMap<String, (String, JsonValue)> {
    fn walk(nodes: &JsonValue, out: &mut HashMap<String, (String, JsonValue)>) {
        let JsonValue::Array(list) = nodes else {
            panic!("nodes")
        };
        for n in list {
            let id = match field(n, "id") {
                JsonValue::String(s) => s.clone(),
                other => panic!("id {other:?}"),
            };
            let kind = match field(n, "kind") {
                JsonValue::String(s) => s.clone(),
                other => panic!("kind {other:?}"),
            };
            let children = match n {
                JsonValue::Object(m) => m.iter().find(|(k, _)| k == "children").map(|(_, v)| v),
                _ => None,
            };
            match children {
                Some(JsonValue::Array(c)) if !c.is_empty() => {
                    walk(&JsonValue::Array(c.clone()), out)
                }
                _ => {
                    out.insert(id, (kind, field(n, "params").clone()));
                }
            }
        }
    }
    let mut out = HashMap::new();
    walk(field(doc, "nodes"), &mut out);
    out
}

fn build_text(text: &str, g: &GeometryRegistry) -> LoadedAsset {
    let asset = parse(text).unwrap_or_else(|e| panic!("parse: {e}"));
    build(&asset, &ctx(g)).unwrap_or_else(|e| panic!("build: {e}"))
}

#[test]
fn every_published_fixture_parses_and_builds() {
    let g = registry();
    let mut with_cuts = Vec::new();
    for name in published() {
        let text = read(&format!("published/{name}"));
        let asset = parse(&text).unwrap_or_else(|e| panic!("{name}: parse: {e}"));
        match build(&asset, &ctx(&g)) {
            Ok(loaded) => assert!(!loaded.meshes.is_empty(), "{name}: no meshes"),
            // Only the boolean resolver is missing: the document itself was accepted.
            Err(e) if e.code == "boolean.unsupported" => with_cuts.push(name.clone()),
            Err(e) => panic!("{name}: build: {e}"),
        }
        let cut = build(&asset, &host_only_ctx(&g))
            .unwrap_or_else(|e| panic!("{name}: build with host-only booleans: {e}"));
        assert!(!cut.meshes.is_empty(), "{name}: no meshes");
    }
    // The fixtures with subtract or intersect children are the ones needing a resolver.
    with_cuts.sort_unstable();
    assert_eq!(
        with_cuts,
        [
            "intersect-dome.raa.prims.json",
            "two-cutters.raa.prims.json",
            "wall-with-hole.raa.prims.json"
        ],
    );
}

#[test]
fn leaf_primitives_tessellate_to_the_exact_box() {
    let g = registry();
    let mut checked = 0usize;
    let mut worst: HashMap<String, f64> = HashMap::new();
    for name in published() {
        let text = read(&format!("published/{name}"));
        let doc = json::parse(&text).unwrap_or_else(|e| panic!("{name}: json {e}"));
        let leaves = leaves(&doc);
        let asset = parse(&text).unwrap_or_else(|e| panic!("{name}: parse: {e}"));
        let loaded = build(&asset, &host_only_ctx(&g)).unwrap_or_else(|e| panic!("{name}: {e}"));
        for m in &loaded.meshes {
            let Some((kind, params)) = leaves.get(&m.node_id) else {
                continue;
            };
            let h = exact_half_extents(kind, params);
            let scale = h.iter().fold(1.0_f64, |a, &b| a.max(b));
            let tol = tolerance(kind) * scale;
            let b = m.mesh.bounds();
            let got_max = [b.max.x, b.max.y, b.max.z].map(f64::from);
            let got_min = [b.min.x, b.min.y, b.min.z].map(f64::from);
            for axis in 0..3 {
                let at = format!("{name} node {} ({kind}) axis {axis}", m.node_id);
                // The mesh never leaves the exact box, and reaches it within `tol`.
                assert!(
                    got_max[axis] <= h[axis] + BOUNDS_TOL * scale,
                    "{at}: max {} past {}",
                    got_max[axis],
                    h[axis]
                );
                assert!(
                    got_min[axis] >= -h[axis] - BOUNDS_TOL * scale,
                    "{at}: min {} past {}",
                    got_min[axis],
                    -h[axis]
                );
                assert!(
                    h[axis] - got_max[axis] <= tol,
                    "{at}: max {} short of exact {}",
                    got_max[axis],
                    h[axis]
                );
                assert!(
                    got_min[axis] + h[axis] <= tol,
                    "{at}: min {} short of exact {}",
                    got_min[axis],
                    -h[axis]
                );
            }
            checked += 1;
            let short = (0..3)
                .map(|a| (h[a] - got_max[a]).max(got_min[a] + h[a]))
                .fold(0.0_f64, f64::max)
                / scale;
            worst
                .entry(kind.clone())
                .and_modify(|w| *w = w.max(short))
                .or_insert(short);
        }
    }
    // One per childless primitive of the 11 fixtures (a parent is cut by its children, so
    // its mesh is not compared); a silent empty walk would fail this.
    assert_eq!(checked, 15, "leaf primitives checked");
    eprintln!("worst relative extent shortfall at LOD0: {worst:?}");
}

/// Tolerance of a kind's extent, relative to its largest half extent. A box and a roundBox
/// are exact up to float rounding. Every other kind is sampled around its axis or along its
/// rings, so an extent can fall short of the exact box by the gap between two samples: at
/// LOD0 the measured shortfall is at most 0.85% of the largest half extent (ellipsoid; run the
/// test with --nocapture for the table).
fn tolerance(kind: &str) -> f64 {
    match kind {
        "box" | "roundBox" => BOUNDS_TOL,
        _ => SAMPLED_TOL,
    }
}

/// One-node document of `kind` with the given params JSON, union, unit transform.
fn one_node(kind: &str, params: &str, extra: &str) -> String {
    format!(
        r#"{{"format":"raa.prims","version":0,"id":"t","name":"t","nodes":[{{"id":"a","kind":"{kind}","params":{params},"transform":{{"t":[0,0,0],"r":[0,0,0,1],"s":[1,1,1]}},"material":"m","op":"union"{extra}}}],"materials":{{"m":{{"color":[0.5,0.5,0.5],"roughness":0.5,"metallic":0,"emissive":[0,0,0]}}}},"anchors":{{}},"collision":{{"mode":"derived","colliders":[]}}}}"#
    )
}

#[test]
fn published_names_build_the_same_mesh_as_the_s3_names() {
    let g = registry();
    // (published params, S3 params) pairs describing the same primitive.
    let pairs = [
        (
            "box",
            r#"{"size":[2,1,3]}"#,
            r#"{"halfExtents":[1,0.5,1.5]}"#,
        ),
        (
            "roundBox",
            r#"{"size":[2,2,2],"radius":0.4}"#,
            r#"{"halfExtents":[1,1,1],"radius":0.4}"#,
        ),
        (
            "capsule",
            r#"{"height":1.2,"radius":0.4}"#,
            r#"{"radius":0.4,"halfHeight":0.2}"#,
        ),
        (
            "cylinder",
            r#"{"height":0.9,"radius":0.3}"#,
            r#"{"radius":0.3,"halfHeight":0.45}"#,
        ),
        (
            "cone",
            r#"{"height":1,"radiusBottom":0.4,"radiusTop":0}"#,
            r#"{"radius":0.4,"halfHeight":0.5}"#,
        ),
    ];
    for (kind, published, s3) in pairs {
        let a = build_text(&one_node(kind, published, ""), &g);
        let b = build_text(&one_node(kind, s3, ""), &g);
        assert_eq!(
            mesh_hash(&a.meshes[0].mesh),
            mesh_hash(&b.meshes[0].mesh),
            "{kind}: published and S3 params differ"
        );
    }
    // A frustum (both radii > 0) has the published bounds: max radius at the widest end.
    let f = build_text(
        &one_node(
            "cone",
            r#"{"height":0.6,"radiusBottom":0.35,"radiusTop":0.2}"#,
            "",
        ),
        &g,
    );
    let b = f.meshes[0].mesh.bounds();
    assert!((f64::from(b.max.x) - 0.35).abs() <= BOUNDS_TOL);
    assert!((f64::from(b.max.y) - 0.3).abs() <= BOUNDS_TOL);
}

#[test]
fn unknown_params_stay_refused_after_the_mapping() {
    let text = read("published/capsule-cylinder.raa.prims.json").replacen(
        r#""height": 1.2,"#,
        r#""height": 1.2, "depth": 2,"#,
        1,
    );
    let e = parse(&text).expect_err("unknown param must be refused");
    assert_eq!(
        (e.code, e.path.as_str()),
        ("kind.params", "/nodes/0/params/depth")
    );
}

#[test]
fn consumer_bean_asset_is_refused_at_its_budget_not_at_height() {
    // The bean predates raa.prims v0 and is not valid under it; its first break is the
    // numeric `budget` (v0: an object). Its capsule `height` is not what is refused.
    let e = parse(&read("consumer/bean.prims.json")).expect_err("bean is not v0");
    assert_eq!((e.code, e.path.as_str()), ("schema.type", "/budget"));
}
