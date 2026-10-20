//! Loader integration: subtract assets build with CsgResolver and are refused with NoBooleans;
//! CSG refusal codes reach the loader with node paths.

use raa_csg::{CsgResolver, cases};
use raa_geom::analysis::{analyze, mesh_hash, passes};
use raa_geom::{GeometryRegistry, Quality};
use raa_loader::prims::PrimsLoader;
use raa_loader::prims::samples::sample_assets;
use raa_loader::{
    BooleanResolver, ColliderShape, LoadContext, LoadError, LoadedAsset, LoaderRegistry, NoBooleans,
};

fn geometry() -> GeometryRegistry {
    let mut g = GeometryRegistry::new();
    raa_prims::register_all(&mut g).expect("register_all");
    g
}

fn load(text: &str, booleans: &dyn BooleanResolver) -> Result<LoadedAsset, LoadError> {
    let g = geometry();
    let mut reg = LoaderRegistry::new();
    reg.register(Box::new(PrimsLoader)).expect("register");
    let ctx = LoadContext {
        geometry: &g,
        booleans,
        quality: Quality::LOD0,
        strict_kinds: true,
    };
    reg.load(text.as_bytes(), &ctx)
}

fn case_text(id: &str) -> String {
    cases()
        .into_iter()
        .find(|c| c.id == id)
        .unwrap_or_else(|| panic!("case {id}"))
        .asset
}

#[test]
fn subtract_asset_builds_with_csg_resolver() {
    let a = load(&case_text("box-sub-cylinder-through"), &CsgResolver).expect("builds");
    assert_eq!(a.meshes.len(), 1);
    let report = analyze(&a.meshes[0].mesh);
    passes(&report).expect("passes");
    assert!(report.closed && report.volume > 0.0);
    // Plain box [0.7, 0.6, 0.5]: 8 * 0.7 * 0.6 * 0.5.
    let box_volume = 8.0 * 0.7 * 0.6 * 0.5;
    assert!(
        report.volume < box_volume,
        "{} !< {box_volume}",
        report.volume
    );
    assert!(a.triangles <= a.budget, "{} > {}", a.triangles, a.budget);
    assert_eq!(a.colliders.len(), 1);
    assert!(matches!(a.colliders[0].shape, ColliderShape::Cuboid { .. }));
}

#[test]
fn subtract_asset_refused_without_resolver() {
    let e = load(&case_text("box-sub-cylinder-through"), &NoBooleans).expect_err("refused");
    assert_eq!(e.code, "boolean.unsupported");
    assert!(!e.message.is_empty());
    assert!(e.path.starts_with("/nodes/0/children/"), "path {}", e.path);
}

#[test]
fn csg_refusal_codes_reach_the_loader() {
    let e = load(&case_text("box-sub-torus"), &CsgResolver).expect_err("refused");
    assert_eq!(e.code, "boolean.csg.tool_not_convex");
    assert_eq!(e.path, "/nodes/0/children/0");
    assert!(!e.message.is_empty());

    let e = load(&case_text("box-sub-tool-with-children"), &CsgResolver).expect_err("refused");
    assert_eq!(e.code, "boolean.csg.tool_has_children");
    assert!(!e.message.is_empty());
}

#[test]
fn crate_sample_still_builds() {
    for (name, text) in sample_assets() {
        let a = load(&text, &CsgResolver).unwrap_or_else(|e| panic!("{name} csg: {e}"));
        let b = load(&text, &NoBooleans).unwrap_or_else(|e| panic!("{name} none: {e}"));
        let ha: Vec<u64> = a.meshes.iter().map(|m| mesh_hash(&m.mesh)).collect();
        let hb: Vec<u64> = b.meshes.iter().map(|m| mesh_hash(&m.mesh)).collect();
        assert_eq!(ha, hb, "{name}");
    }
}
