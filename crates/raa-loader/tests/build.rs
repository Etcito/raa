//! Q6: the sample assets build into closed meshes within their budget, with derived or
//! overridden colliders, materials and anchors; the loader registry and boolean delegation.

use std::cell::RefCell;

use raa_geom::analysis::{analyze, mesh_hash, passes};
use raa_geom::builtin::DebugCube;
use raa_geom::{GeometryRegistry, MeshData, Quality};
use raa_loader::prims::TransformDoc;
use raa_loader::prims::samples::sample_assets;
use raa_loader::prims::{
    self, DEFAULT_TRIANGLE_BUDGET, FORMAT, NodeOp, PrimNode, PrimsAsset, PrimsLoader,
};
use raa_loader::{
    AssetLoader, BoolOp, BooleanHost, BooleanResolver, BooleanTool, ColliderShape, LoadContext,
    LoadError, LoadedAsset, LoaderRegistry, NoBooleans,
};
use raa_material::Material;
use raa_math::{Mat4, Quat, Vec3};

fn registry() -> GeometryRegistry {
    let mut g = GeometryRegistry::new();
    raa_prims::register_all(&mut g).expect("register_all");
    g
}

fn ctx<'a>(g: &'a GeometryRegistry, booleans: &'a dyn BooleanResolver) -> LoadContext<'a> {
    LoadContext {
        geometry: g,
        booleans,
        quality: Quality::LOD0,
        strict_kinds: true,
    }
}

fn text(name: &str) -> String {
    sample_assets()
        .into_iter()
        .find(|(n, _)| *n == name)
        .map(|(_, t)| t)
        .expect("sample")
}

fn parsed(name: &str) -> PrimsAsset {
    prims::parse(&text(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn built(name: &str) -> LoadedAsset {
    let g = registry();
    prims::build(&parsed(name), &ctx(&g, &NoBooleans)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Roots plus union children, counted on the parsed asset.
fn count_union(nodes: &[PrimNode]) -> usize {
    nodes
        .iter()
        .filter(|n| n.op == NodeOp::Union)
        .map(|n| 1 + count_union(&n.children))
        .sum()
}

fn translation(m: &Mat4) -> [f32; 3] {
    [m.cols[3][0], m.cols[3][1], m.cols[3][2]]
}

fn trs(d: &TransformDoc) -> Mat4 {
    let f = |v: f64| v as f32;
    Mat4::from_trs(
        Vec3::new(f(d.t[0]), f(d.t[1]), f(d.t[2])),
        Quat::from_xyzw(f(d.r[0]), f(d.r[1]), f(d.r[2]), f(d.r[3])),
        Vec3::new(f(d.s[0]), f(d.s[1]), f(d.s[2])),
    )
}

fn close3(a: [f32; 3], b: [f32; 3]) -> bool {
    a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= 1e-6)
}

#[test]
fn every_sample_mesh_passes_analysis() {
    for (name, _) in sample_assets() {
        let a = built(name);
        assert_eq!(a.format, FORMAT);
        for m in &a.meshes {
            let report = analyze(&m.mesh);
            if let Err(why) = passes(&report) {
                panic!("{name}/{}: {why}", m.node_id);
            }
        }
    }
}

#[test]
fn mesh_count_is_roots_plus_union_children() {
    for (name, _) in sample_assets() {
        let p = parsed(name);
        let a = built(name);
        assert_eq!(a.meshes.len(), count_union(&p.nodes), "{name}");
    }
}

#[test]
fn triangles_match_meshes_estimate_and_budget() {
    let g = registry();
    for (name, _) in sample_assets() {
        let p = parsed(name);
        let c = ctx(&g, &NoBooleans);
        let a = prims::build(&p, &c).expect("build");
        let sum: usize = a.meshes.iter().map(|m| m.mesh.triangle_count()).sum();
        assert_eq!(a.triangles as usize, sum, "{name}");
        assert_eq!(
            a.triangles,
            prims::estimate_triangles(&p, &c).expect("estimate"),
            "{name}"
        );
        assert_eq!(
            a.budget,
            p.triangle_budget.unwrap_or(DEFAULT_TRIANGLE_BUDGET),
            "{name}"
        );
        assert!(a.triangles <= a.budget, "{name}");
        eprintln!(
            "sample {name}: LOD0 triangles {} / budget {} (explicit triangleBudget: {})",
            a.triangles,
            a.budget,
            p.triangle_budget
                .map_or_else(|| "none".to_string(), |b| b.to_string())
        );
    }
}

/// Expected collider shape name per S4 derivation (uniform world scale in the samples).
fn expected_shape(kind: &str) -> &'static str {
    match kind {
        "sphere" => "ball",
        "box" => "cuboid",
        "roundBox" => "roundCuboid",
        "capsule" => "capsuleY",
        "cylinder" => "cylinderY",
        "cone" => "coneY",
        "ellipsoid" => "convexHull",
        other => panic!("no derived collider for {other}"),
    }
}

fn shape_name(s: &ColliderShape) -> &'static str {
    match s {
        ColliderShape::Ball { .. } => "ball",
        ColliderShape::Cuboid { .. } => "cuboid",
        ColliderShape::RoundCuboid { .. } => "roundCuboid",
        ColliderShape::CapsuleY { .. } => "capsuleY",
        ColliderShape::CylinderY { .. } => "cylinderY",
        ColliderShape::ConeY { .. } => "coneY",
        ColliderShape::ConvexHull { .. } => "convexHull",
    }
}

#[test]
fn critter_and_crate_get_one_derived_collider_per_union_node() {
    for name in ["critter", "crate"] {
        let p = parsed(name);
        assert!(p.collision.is_none(), "{name} has no override");
        let a = built(name);
        assert_eq!(a.colliders.len(), count_union(&p.nodes), "{name}");
        for (c, m) in a.colliders.iter().zip(&a.meshes) {
            assert_eq!(c.node_id.as_deref(), Some(m.node_id.as_str()), "{name}");
            assert_eq!(
                shape_name(&c.shape),
                expected_shape(&m.kind),
                "{name}/{}",
                m.node_id
            );
            if let ColliderShape::ConvexHull { points } = &c.shape {
                assert!(points.len() >= 4, "{name}/{}: hull points", m.node_id);
            }
        }
    }
}

#[test]
fn lamp_colliders_equal_its_override() {
    let p = parsed("lamp");
    let docs = p.collision.as_ref().expect("lamp override");
    let a = built("lamp");
    assert_eq!(a.colliders.len(), docs.len());
    let expected = [
        (
            "base",
            ColliderShape::CylinderY {
                half_height: 0.05,
                radius: 0.4,
            },
        ),
        (
            "pole",
            ColliderShape::CylinderY {
                half_height: 0.6,
                radius: 0.04,
            },
        ),
        ("bulb", ColliderShape::Ball { radius: 0.22 }),
    ];
    for ((c, doc), (id, shape)) in a.colliders.iter().zip(docs).zip(expected) {
        assert_eq!(doc.node.as_deref(), Some(id));
        assert_eq!(c.node_id.as_deref(), Some(id));
        assert_eq!(c.shape, shape, "{id}");
        // An override naming a node is placed by its own transform after the node's world.
        let node_world = a
            .meshes
            .iter()
            .find(|m| m.node_id == id)
            .map(|m| m.transform)
            .unwrap_or_else(|| panic!("mesh {id}"));
        let expected = node_world * trs(&doc.transform);
        assert_eq!(c.transform, expected, "{id}");
    }
}

#[test]
fn critter_anchors_and_inherited_material() {
    let a = built("critter");
    let names: Vec<&str> = a.anchors.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["head", "tail"]);
    assert!(close3(translation(&a.anchors[0].1), [0.0, 0.2, 0.3]));
    assert!(close3(translation(&a.anchors[1].1), [0.0, 0.05, -0.45]));

    let mat = |id: &str| {
        a.meshes
            .iter()
            .find(|m| m.node_id == id)
            .map(|m| m.material.as_str())
            .unwrap_or_else(|| panic!("mesh {id}"))
    };
    assert_eq!(mat("body"), "fur");
    assert_eq!(mat("eye-l"), "eye");
    // The legs declare no material and inherit the body's.
    assert_eq!(mat("leg-fl"), "fur");
    assert_eq!(mat("leg-br"), "fur");

    assert_eq!(a.materials[0].0, "_default");
    assert_eq!(a.materials[0].1, Material::default());
    let names: Vec<&str> = a.materials.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["_default", "eye", "fur"]);
}

#[test]
fn materials_start_with_default_in_every_sample() {
    for (name, _) in sample_assets() {
        assert_eq!(built(name).materials[0].0, "_default", "{name}");
    }
}

#[test]
fn two_builds_give_identical_mesh_hashes() {
    for (name, _) in sample_assets() {
        let hashes = |a: &LoadedAsset| a.meshes.iter().map(|m| mesh_hash(&m.mesh)).collect();
        let first: Vec<u64> = hashes(&built(name));
        let second: Vec<u64> = hashes(&built(name));
        assert!(!first.is_empty(), "{name}");
        assert_eq!(first, second, "{name}");
    }
}

#[test]
fn loader_registry_with_prims_loader() {
    let g = registry();
    let c = ctx(&g, &NoBooleans);
    let mut r = LoaderRegistry::new();
    r.register(Box::new(PrimsLoader)).expect("first register");
    assert_eq!(r.formats(), vec!["raa.prims"]);

    for (name, t) in sample_assets() {
        let via_registry = r
            .load(t.as_bytes(), &c)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(via_registry, built(name), "{name}");
        assert_eq!(via_registry.id, name);
    }

    let e = r
        .load(b"\x00\x01 not an asset", &c)
        .expect_err("unknown bytes");
    assert_eq!(e.code, "loader.unknown_format");
    assert!(!e.message.is_empty());

    let e = r
        .register(Box::new(PrimsLoader))
        .expect_err("second register");
    assert_eq!(e.code, "loader.duplicate_format");
    assert!(!e.message.is_empty());
    assert_eq!(r.formats(), vec!["raa.prims"]);
}

fn debug_cube_asset() -> String {
    format!(
        concat!(
            r#"{{"format":"{}","version":0,"id":"dbg","name":"Debug","#,
            r#""nodes":[{{"id":"cube","kind":"debug_cube","params":{{"half":0.5}}}}],"#,
            r#""materials":{{}},"anchors":{{}}}}"#
        ),
        FORMAT
    )
}

#[test]
fn non_strict_kinds_accept_registered_debug_cube() {
    let mut g = registry();
    g.register(Box::new(DebugCube)).expect("register DebugCube");
    let t = debug_cube_asset();

    let mut lax = ctx(&g, &NoBooleans);
    lax.strict_kinds = false;
    let a = PrimsLoader
        .load(t.as_bytes(), &lax)
        .expect("non-strict load");
    assert_eq!(a.meshes.len(), 1);
    assert_eq!(a.meshes[0].kind, "debug_cube");
    assert_eq!(a.triangles, 12);
    passes(&analyze(&a.meshes[0].mesh)).expect("debug cube closed");

    let strict = ctx(&g, &NoBooleans);
    let e = PrimsLoader
        .load(t.as_bytes(), &strict)
        .expect_err("strict load");
    assert_eq!(e.code, "kind.unknown");
    assert!(!e.message.is_empty());
}

/// Tool node id, op, bounds min and max.
type Recorded = (String, BoolOp, [f32; 3], [f32; 3]);

/// Copies the host and records the bounds of every tool it receives.
#[derive(Default)]
struct Recording {
    tools: RefCell<Vec<Recorded>>,
}

fn bounds(m: &MeshData) -> ([f32; 3], [f32; 3]) {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in &m.positions {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    (lo, hi)
}

impl BooleanResolver for Recording {
    fn resolve(
        &self,
        host: &BooleanHost<'_>,
        tools: &[BooleanTool<'_>],
        out: &mut MeshData,
    ) -> Result<(), LoadError> {
        for t in tools {
            let (lo, hi) = bounds(t.mesh);
            self.tools
                .borrow_mut()
                .push((t.node_id.to_string(), t.op, lo, hi));
        }
        out.clear();
        out.positions.extend_from_slice(&host.mesh.positions);
        out.normals.extend_from_slice(&host.mesh.normals);
        out.indices.extend_from_slice(&host.mesh.indices);
        Ok(())
    }
}

#[test]
fn boolean_resolver_receives_tool_in_host_space() {
    // The host sits at x = 5 in asset space; the tool is expressed in host local space.
    let t = format!(
        concat!(
            r#"{{"format":"{}","version":0,"id":"cut","name":"Cut","#,
            r#""nodes":[{{"id":"host","kind":"box","params":{{"halfExtents":[0.5,0.5,0.5]}},"#,
            r#""transform":{{"t":[5,0,0],"r":[0,0,0,1],"s":[1,1,1]}},"#,
            r#""children":[{{"id":"hole","kind":"box","params":{{"halfExtents":[0.2,0.2,0.6]}},"#,
            r#""transform":{{"t":[0.5,0,0],"r":[0,0,0,1],"s":[1,2,1]}},"op":"subtract"}}]}}],"#,
            r#""materials":{{}},"anchors":{{}}}}"#
        ),
        FORMAT
    );
    let g = registry();
    let rec = Recording::default();
    let a = PrimsLoader
        .load(t.as_bytes(), &ctx(&g, &rec))
        .expect("load with recording resolver");
    assert_eq!(a.meshes.len(), 1);
    assert_eq!(a.colliders.len(), 1);
    let tools = rec.tools.borrow();
    assert_eq!(tools.len(), 1);
    let (id, op, lo, hi) = &tools[0];
    assert_eq!(id, "hole");
    assert_eq!(*op, BoolOp::Subtract);
    assert!(close3(*lo, [0.3, -0.4, -0.6]), "tool min {lo:?}");
    assert!(close3(*hi, [0.7, 0.4, 0.6]), "tool max {hi:?}");

    let e = PrimsLoader
        .load(t.as_bytes(), &ctx(&g, &NoBooleans))
        .expect_err("NoBooleans");
    assert_eq!(e.code, "boolean.unsupported");
    assert_eq!(e.path, "/nodes/0/children/0");
}
