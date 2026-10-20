//! Every `cases()` entry through PrimsLoader + CsgResolver at LOD0: ok cases are watertight,
//! within the budget and satisfy the volume identity within 1e-4; refusals carry their exact
//! code and a message; loading is deterministic. A roundBox host (outside the table) is refused.

use raa_csg::{
    CsgCase, CsgError, CsgResolver, cases, convex_planes, intersect_convex, subtract_convex,
};
use raa_geom::analysis::{analyze, mesh_hash};
use raa_geom::{GeometryRegistry, MeshData, Quality};
use raa_loader::prims::{NodeOp, PrimNode, PrimsLoader, parse, to_canonical_json};
use raa_loader::{AssetLoader, LoadContext, LoadError, LoadedAsset};

/// CSG_EPS rule: volume identity relative difference.
const ID_TOL: f64 = 1e-4;

fn registry() -> GeometryRegistry {
    let mut g = GeometryRegistry::new();
    raa_prims::register_all(&mut g).expect("register_all");
    g
}

fn load(geometry: &GeometryRegistry, text: &str) -> Result<LoadedAsset, LoadError> {
    let ctx = LoadContext {
        geometry,
        booleans: &CsgResolver,
        quality: Quality::LOD0,
        strict_kinds: true,
    };
    PrimsLoader.load(text.as_bytes(), &ctx)
}

fn hashes(a: &LoadedAsset) -> Vec<u64> {
    a.meshes.iter().map(|m| mesh_hash(&m.mesh)).collect()
}

/// The same asset with the host's boolean children removed.
fn plain_host_text(case: &CsgCase) -> String {
    let mut doc = parse(&case.asset).expect("case parses");
    doc.nodes[0].children.clear();
    to_canonical_json(&doc)
}

#[test]
fn ok_cases_are_watertight_within_budget() {
    let g = registry();
    let mut failures = Vec::new();
    for case in cases().iter().filter(|c| c.expected == "ok") {
        match load(&g, &case.asset) {
            Ok(a) => {
                assert!(
                    a.triangles <= a.budget,
                    "{}: {} > {}",
                    case.id,
                    a.triangles,
                    a.budget
                );
                assert!(!a.meshes.is_empty(), "{}: no mesh", case.id);
                for m in &a.meshes {
                    let r = analyze(&m.mesh);
                    assert!(
                        r.closed,
                        "{}: mesh '{}' not closed: {r:?}",
                        case.id, m.node_id
                    );
                    assert!(r.volume > 0.0, "{}: volume {}", case.id, r.volume);
                }
            }
            Err(e) => failures.push(format!("{}: {e}", case.id)),
        }
    }
    assert!(
        failures.is_empty(),
        "ok cases refused:\n{}",
        failures.join("\n")
    );
}

#[test]
fn host_outside_tool_equals_plain_host_mesh() {
    let g = registry();
    let all = cases();
    let case = all
        .iter()
        .find(|c| c.id == "box-sub-sphere-outside")
        .expect("case exists");
    let cut = load(&g, &case.asset).expect("loads");
    let plain = load(&g, &plain_host_text(case)).expect("plain host loads");
    assert_eq!(hashes(&cut), hashes(&plain));
}

#[test]
fn refusal_cases_return_expected_code_and_message() {
    let g = registry();
    for case in cases().iter().filter(|c| c.expected != "ok") {
        match load(&g, &case.asset) {
            Ok(_) => panic!("{}: loaded, expected {}", case.id, case.expected),
            Err(e) => {
                assert_eq!(e.code, case.expected, "{}: {e}", case.id);
                assert!(!e.message.is_empty(), "{}: empty message", case.id);
            }
        }
    }
}

#[test]
fn loading_twice_gives_identical_mesh_hashes() {
    let g = registry();
    for case in cases() {
        let a = load(&g, &case.asset);
        let b = load(&g, &case.asset);
        match (a, b) {
            (Ok(a), Ok(b)) => assert_eq!(hashes(&a), hashes(&b), "{}", case.id),
            (Err(a), Err(b)) => {
                assert_eq!((a.code, &a.message), (b.code, &b.message), "{}", case.id)
            }
            _ => panic!("{}: outcomes differ between loads", case.id),
        }
    }
}

/// The single mesh of a loaded asset in asset space (the host has no transform).
fn asset_mesh(a: &LoadedAsset) -> MeshData {
    assert_eq!(a.meshes.len(), 1, "{}: one mesh expected", a.id);
    let mut out = MeshData::new();
    out.append_transformed(&a.meshes[0].mesh, &a.meshes[0].transform);
    out
}

/// The tool node loaded alone as a union root, in host (= asset) space.
fn tool_mesh(g: &GeometryRegistry, case: &CsgCase, tool: &PrimNode) -> MeshData {
    let mut doc = parse(&case.asset).expect("case parses");
    let mut root = tool.clone();
    root.op = NodeOp::Union;
    doc.nodes = vec![root];
    asset_mesh(&load(g, &to_canonical_json(&doc)).expect("tool loads alone"))
}

fn csg_eps(meshes: &[&MeshData]) -> f64 {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in meshes.iter().flat_map(|m| &m.positions) {
        for k in 0..3 {
            lo[k] = lo[k].min(f64::from(p[k]));
            hi[k] = hi[k].max(f64::from(p[k]));
        }
    }
    let d2: f64 = (0..3).map(|k| (hi[k] - lo[k]).max(0.0).powi(2)).sum();
    1e-6 * d2.sqrt().max(1.0)
}

/// V(a ∩ b); an EmptyResult counts as 0.
fn int_volume(a: &MeshData, b: &MeshData) -> f64 {
    let mut out = MeshData::new();
    match intersect_convex(a, b, &mut out) {
        Ok(()) => analyze(&out).volume,
        Err(CsgError::EmptyResult) => 0.0,
        Err(e) => panic!("intersect_convex: {e}"),
    }
}

/// |V(resolved) − expected| / V(host), where expected is V(H') − Σ V(H' ∩ T) with
/// H' = H ∩ I (or H without an intersect tool), and V(H) − V(H − I) for a lone intersect.
fn identity_err(g: &GeometryRegistry, case: &CsgCase, resolved: &MeshData) -> f64 {
    let doc = parse(&case.asset).expect("case parses");
    let host = &doc.nodes[0];
    let h = asset_mesh(&load(g, &plain_host_text(case)).expect("plain host loads"));
    let mut ints = Vec::new();
    let mut subs = Vec::new();
    for t in &host.children {
        match t.op {
            NodeOp::Intersect => ints.push(tool_mesh(g, case, t)),
            _ => subs.push(tool_mesh(g, case, t)),
        }
    }
    let vh = analyze(&h).volume;
    let lhs = analyze(resolved).volume;
    let rhs = match ints.as_slice() {
        [] => vh - subs.iter().map(|t| int_volume(&h, t)).sum::<f64>(),
        [i] if subs.is_empty() => {
            let planes = convex_planes(&h, csg_eps(&[&h, i])).expect("host planes");
            let mut rest = MeshData::new();
            subtract_convex(&h, &planes, &[i], &mut rest).expect("H - I");
            vh - analyze(&rest).volume
        }
        [i] => {
            let mut hp = MeshData::new();
            intersect_convex(&h, i, &mut hp).expect("H ∩ I");
            analyze(&hp).volume - subs.iter().map(|t| int_volume(&hp, t)).sum::<f64>()
        }
        _ => panic!("{}: more than one intersect tool", case.id),
    };
    (lhs - rhs).abs() / vh
}

#[test]
fn ok_cases_satisfy_the_volume_identity() {
    let g = registry();
    for case in cases().iter().filter(|c| c.expected == "ok") {
        let a = load(&g, &case.asset).unwrap_or_else(|e| panic!("{}: {e}", case.id));
        let err = identity_err(&g, case, &asset_mesh(&a));
        assert!(
            err.is_finite() && err <= ID_TOL,
            "{}: identity error {err:e} > {ID_TOL:e}",
            case.id
        );
    }
}

#[test]
fn roundbox_host_is_refused_as_not_convex() {
    let g = registry();
    let all = cases();
    let case = all
        .iter()
        .find(|c| c.id == "box-sub-ellipsoid-scoop")
        .expect("case exists");
    let mut doc = parse(&case.asset).expect("case parses");
    let host = &mut doc.nodes[0];
    host.kind = "roundBox".to_string();
    host.params.push((
        "radius".to_string(),
        raa_loader::json::JsonValue::Number(0.15),
    ));
    let text = to_canonical_json(&doc);
    assert!(text.contains("\"roundBox\"") && text.contains("\"scoop\""));
    match load(&g, &text) {
        Ok(_) => panic!("roundBox host loaded, expected boolean.csg.host_not_convex"),
        Err(e) => {
            assert_eq!(e.code, "boolean.csg.host_not_convex", "{e}");
            assert!(!e.message.is_empty());
        }
    }
}
