use raa_geom::{GeometryProvider, MeshData, ParamValue, Params, Quality, analysis};
use raa_prims::{PrimKind, PrimParams, PrimProvider, generate};

#[test]
fn all_kinds_sets_qualities_pass_analysis() {
    let mut fails = Vec::new();
    for kind in PrimKind::ALL {
        for set in ['A', 'B'] {
            for q in [Quality::LOW, Quality::LOD0, Quality::HIGH] {
                let name = format!("{} {set} {}", kind.as_str(), q.segments);
                let mut m = MeshData::new();
                let p = PrimParams::reference_set(kind, set);
                if let Err(e) = generate(&p, q, &mut m) {
                    fails.push(format!("{name}: {e}"));
                } else if let Err(msg) = analysis::passes(&analysis::analyze(&m)) {
                    fails.push(format!("{name}: {msg}"));
                }
            }
        }
    }
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

fn bad_cases() -> Vec<(&'static str, PrimParams)> {
    vec![
        ("sphere radius 0", PrimParams::Sphere { radius: 0.0 }),
        (
            "box extent 0",
            PrimParams::Box {
                half_extents: [1.0, 0.0, 1.0],
            },
        ),
        (
            "roundBox radius > min extent",
            PrimParams::RoundBox {
                half_extents: [0.5, 1.0, 1.0],
                radius: 0.6,
            },
        ),
        (
            "roundBox radius negative",
            PrimParams::RoundBox {
                half_extents: [1.0; 3],
                radius: -0.1,
            },
        ),
        (
            "capsule radius 0",
            PrimParams::Capsule {
                radius: 0.0,
                half_height: 1.0,
            },
        ),
        (
            "capsule negative halfHeight",
            PrimParams::Capsule {
                radius: 1.0,
                half_height: -0.1,
            },
        ),
        (
            "cylinder halfHeight 0",
            PrimParams::Cylinder {
                radius: 1.0,
                half_height: 0.0,
            },
        ),
        (
            "cone radius 0",
            PrimParams::Cone {
                radius: 0.0,
                half_height: 1.0,
                radius_top: 0.0,
            },
        ),
        (
            "torus minor >= major",
            PrimParams::Torus {
                major_radius: 0.5,
                minor_radius: 0.5,
            },
        ),
        (
            "torus minor 0",
            PrimParams::Torus {
                major_radius: 0.5,
                minor_radius: 0.0,
            },
        ),
        (
            "ellipsoid radius 0",
            PrimParams::Ellipsoid {
                radii: [1.0, 0.0, 1.0],
            },
        ),
        ("sphere NaN", PrimParams::Sphere { radius: f32::NAN }),
    ]
}

#[test]
fn each_rule_refused_by_generate() {
    for (name, p) in bad_cases() {
        let mut m = MeshData::new();
        let e = generate(&p, Quality::LOD0, &mut m).expect_err(name);
        assert_eq!(e.code(), "geom.invalid_param", "{name}");
    }
}

#[test]
fn each_rule_refused_by_provider() {
    for (name, p) in bad_cases() {
        let prov = PrimProvider(p.kind());
        let params = p.to_params();
        let mut m = MeshData::new();
        let e = prov
            .generate(&params, Quality::LOD0, &mut m)
            .expect_err(name);
        assert_eq!(e.code(), "geom.invalid_param", "{name}");
    }
}

#[test]
fn provider_refuses_missing_and_wrong_type() {
    let prov = PrimProvider(PrimKind::Sphere);
    let mut m = MeshData::new();
    let e = prov
        .generate(&Params::new(), Quality::LOD0, &mut m)
        .unwrap_err();
    assert_eq!(e.code(), "geom.missing_param");
    let wrong = Params::new().with("radius", ParamValue::Vec3([1.0; 3]));
    let e = prov.generate(&wrong, Quality::LOD0, &mut m).unwrap_err();
    assert_eq!(e.code(), "geom.invalid_param");
    let e = prov
        .triangle_count(&Params::new(), Quality::LOD0)
        .unwrap_err();
    assert_eq!(e.code(), "geom.missing_param");
}
