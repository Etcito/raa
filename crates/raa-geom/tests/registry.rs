use raa_geom::builtin::DebugCube;
use raa_geom::{GeometryProvider, GeometryRegistry, MeshData, ParamValue, Params, Quality};

fn half(h: f32) -> Params {
    Params::new().with("half", ParamValue::Scalar(h))
}

fn registry() -> GeometryRegistry {
    let mut r = GeometryRegistry::new();
    r.register(Box::new(DebugCube)).unwrap();
    r
}

struct Named(&'static str);

impl GeometryProvider for Named {
    fn kind(&self) -> &str {
        self.0
    }
    fn param_names(&self) -> &[&'static str] {
        &[]
    }
    fn is_convex(&self) -> bool {
        true
    }
    fn triangle_count(&self, p: &Params, q: Quality) -> Result<u32, raa_geom::GeomError> {
        DebugCube.triangle_count(p, q)
    }
    fn generate(
        &self,
        p: &Params,
        q: Quality,
        out: &mut MeshData,
    ) -> Result<(), raa_geom::GeomError> {
        DebugCube.generate(p, q, out)
    }
}

#[test]
fn registering_a_kind_twice_is_duplicate_kind() {
    let mut r = registry();
    let e = r.register(Box::new(DebugCube)).unwrap_err();
    assert_eq!(e.code(), "geom.duplicate_kind");
    assert_eq!(r.kinds(), vec!["debug_cube"]);
}

#[test]
fn generating_an_unknown_kind_is_unknown_kind() {
    let mut out = MeshData::new();
    let e = registry()
        .generate("nope", &half(1.0), Quality::LOD0, &mut out)
        .unwrap_err();
    assert_eq!(e.code(), "geom.unknown_kind");
}

#[test]
fn missing_half_is_missing_param() {
    let mut out = MeshData::new();
    let e = registry()
        .generate("debug_cube", &Params::new(), Quality::LOD0, &mut out)
        .unwrap_err();
    assert_eq!(e.code(), "geom.missing_param");
}

#[test]
fn negative_half_is_invalid_param() {
    let mut out = MeshData::new();
    let e = registry()
        .generate("debug_cube", &half(-1.0), Quality::LOD0, &mut out)
        .unwrap_err();
    assert_eq!(e.code(), "geom.invalid_param");
    assert!(e.to_string().contains("half must be > 0"));
}

#[test]
fn generate_clears_a_prefilled_mesh() {
    let mut out = MeshData::new();
    out.positions.push([9.0; 3]);
    out.normals.push([0.0, 1.0, 0.0]);
    out.indices.extend_from_slice(&[0, 0, 0]);
    registry()
        .generate("debug_cube", &half(0.5), Quality::LOD0, &mut out)
        .unwrap();
    assert_eq!(out.vertex_count(), 24);
    assert_eq!(out.triangle_count(), 12);
    assert!(!out.positions.contains(&[9.0; 3]));
    out.validate().unwrap();
}

#[test]
fn two_generations_are_byte_identical() {
    let r = registry();
    let (mut a, mut b) = (MeshData::new(), MeshData::new());
    r.generate("debug_cube", &half(0.75), Quality::LOW, &mut a)
        .unwrap();
    r.generate("debug_cube", &half(0.75), Quality::LOW, &mut b)
        .unwrap();
    let (mut ba, mut bb) = (Vec::new(), Vec::new());
    a.write_vertex_bytes(&mut ba);
    b.write_vertex_bytes(&mut bb);
    assert!(!ba.is_empty());
    assert_eq!(ba, bb);
}

#[test]
fn triangle_count_matches_generated_count() {
    let r = registry();
    let p = r.get("debug_cube").unwrap();
    for q in [Quality::LOW, Quality::LOD0, Quality::HIGH] {
        let mut out = MeshData::new();
        p.generate(&half(1.0), q, &mut out).unwrap();
        assert_eq!(p.triangle_count(&half(1.0), q).unwrap(), 12);
        assert_eq!(out.triangle_count(), 12);
    }
    assert!(p.is_convex());
    assert_eq!(p.param_names(), &["half"]);
}

#[test]
fn kinds_are_in_registration_order() {
    let mut r = GeometryRegistry::new();
    r.register(Box::new(Named("zeta"))).unwrap();
    r.register(Box::new(DebugCube)).unwrap();
    r.register(Box::new(Named("alpha"))).unwrap();
    assert_eq!(r.kinds(), vec!["zeta", "debug_cube", "alpha"]);
    assert_eq!(r.get("alpha").unwrap().kind(), "alpha");
    assert!(r.get("nope").is_none());
}
