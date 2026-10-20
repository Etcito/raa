//! SDF primitives turned into polygons from their parameters; implements the raa-geom provider trait.

mod boxes;
mod capsule;
mod cone;
mod cylinder;
mod ellipsoid;
mod kind;
mod params;
mod provider;
pub mod reference;
mod sphere;
mod torus;

pub use kind::PrimKind;
pub use params::PrimParams;
pub use provider::{PrimProvider, register_all};

use raa_geom::{GeomError, MeshData, Quality};

/// `Quality.segments` clamped to [3, 256] (the field is public, so it may hold any value).
fn clamped_segments(q: Quality) -> u32 {
    Quality::new(q.segments).segments
}

/// Validates `p`, clears `out`, then writes the closed, outward mesh of `p` at quality `q`
/// (tessellation per the README). An invalid `p` leaves `out` untouched.
pub fn generate(p: &PrimParams, q: Quality, out: &mut MeshData) -> Result<(), GeomError> {
    p.validate()?;
    out.clear();
    let s = clamped_segments(q);
    match *p {
        PrimParams::Sphere { radius } => sphere::generate(radius, s, out),
        PrimParams::Box { half_extents } => boxes::box_generate(half_extents, s, out),
        PrimParams::RoundBox {
            half_extents,
            radius,
        } => boxes::round_box_generate(half_extents, radius, s, out),
        PrimParams::Capsule {
            radius,
            half_height,
        } => capsule::generate(radius, half_height, s, out),
        PrimParams::Cylinder {
            radius,
            half_height,
        } => cylinder::generate(radius, half_height, s, out),
        PrimParams::Cone {
            radius,
            half_height,
            radius_top,
        } => cone::generate(radius, radius_top, half_height, s, out),
        PrimParams::Torus {
            major_radius,
            minor_radius,
        } => torus::generate(major_radius, minor_radius, s, out),
        PrimParams::Ellipsoid { radii } => ellipsoid::generate(radii, s, out),
    }
}

/// Triangle count `generate` emits for `p` at quality `q`, from the closed per-kind formula
/// in the README (no mesh is built).
pub fn triangle_count(p: &PrimParams, q: Quality) -> u32 {
    let s = clamped_segments(q);
    match *p {
        PrimParams::Sphere { radius } => sphere::triangle_count(radius, s),
        PrimParams::Box { half_extents } => boxes::box_triangle_count(half_extents, s),
        PrimParams::RoundBox {
            half_extents,
            radius,
        } => boxes::round_box_triangle_count(half_extents, radius, s),
        PrimParams::Capsule {
            radius,
            half_height,
        } => capsule::triangle_count(radius, half_height, s),
        PrimParams::Cylinder {
            radius,
            half_height,
        } => cylinder::triangle_count(radius, half_height, s),
        PrimParams::Cone {
            radius,
            half_height,
            radius_top,
        } => cone::triangle_count(radius, radius_top, half_height, s),
        PrimParams::Torus {
            major_radius,
            minor_radius,
        } => torus::triangle_count(major_radius, minor_radius, s),
        PrimParams::Ellipsoid { radii } => ellipsoid::triangle_count(radii, s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_geom::{GeometryProvider, GeometryRegistry, ParamValue, Params};

    fn invalid_cases() -> Vec<PrimParams> {
        vec![
            PrimParams::Sphere { radius: 0.0 },
            PrimParams::Box {
                half_extents: [0.5, -1.0, 0.5],
            },
            PrimParams::RoundBox {
                half_extents: [0.5, 0.5, 0.2],
                radius: 0.3,
            },
            PrimParams::Capsule {
                radius: 0.4,
                half_height: -0.1,
            },
            PrimParams::Cylinder {
                radius: f32::NAN,
                half_height: 0.5,
            },
            PrimParams::Cone {
                radius: 0.5,
                half_height: 0.0,
                radius_top: 0.0,
            },
            PrimParams::Torus {
                major_radius: 0.5,
                minor_radius: 0.5,
            },
            PrimParams::Ellipsoid {
                radii: [1.0, f32::INFINITY, 1.0],
            },
        ]
    }

    #[test]
    fn generate_refuses_invalid_params_before_tessellating() {
        let cases = invalid_cases();
        assert_eq!(cases.len(), PrimKind::ALL.len());
        for p in cases {
            let mut out = MeshData::new();
            out.positions.push([1.0, 2.0, 3.0]);
            out.normals.push([0.0, 1.0, 0.0]);
            for q in [
                Quality::LOW,
                Quality::LOD0,
                Quality::HIGH,
                Quality { segments: 0 },
            ] {
                let err = generate(&p, q, &mut out).unwrap_err();
                // A tessellator would answer geom.invalid_mesh or succeed; validation runs first.
                assert_eq!(err.code(), "geom.invalid_param", "{p:?}: {err}");
                assert_eq!(out.vertex_count(), 1, "{p:?}: out touched on refusal");
            }
        }
    }

    #[test]
    fn provider_refuses_invalid_params_with_invalid_param() {
        let mut reg = GeometryRegistry::new();
        register_all(&mut reg).unwrap();
        let mut out = MeshData::new();
        let bad = Params::new().with("radius", ParamValue::Scalar(-1.0));
        let err = reg
            .generate("sphere", &bad, Quality::LOD0, &mut out)
            .unwrap_err();
        assert_eq!(err.code(), "geom.invalid_param");
        let err = PrimProvider(PrimKind::Sphere)
            .triangle_count(&bad, Quality::LOD0)
            .unwrap_err();
        assert_eq!(err.code(), "geom.invalid_param");
    }

    #[test]
    fn register_all_gives_kinds_in_all_order() {
        let mut reg = GeometryRegistry::new();
        register_all(&mut reg).unwrap();
        let expected: Vec<&str> = PrimKind::ALL.iter().map(|k| k.as_str()).collect();
        assert_eq!(reg.kinds(), expected);
        for k in PrimKind::ALL {
            let p = reg.get(k.as_str()).unwrap();
            assert_eq!(p.param_names(), k.param_names());
            assert_eq!(p.is_convex(), k.is_convex());
        }
    }

    #[test]
    fn register_all_twice_gives_duplicate_kind() {
        let mut reg = GeometryRegistry::new();
        register_all(&mut reg).unwrap();
        let err = register_all(&mut reg).unwrap_err();
        assert_eq!(err.code(), "geom.duplicate_kind");
        assert_eq!(reg.kinds().len(), PrimKind::ALL.len());
    }

    #[test]
    fn registry_reports_unknown_kind() {
        let mut reg = GeometryRegistry::new();
        register_all(&mut reg).unwrap();
        let mut out = MeshData::new();
        let p = PrimParams::reference_set(PrimKind::Sphere, 'A').to_params();
        let err = reg
            .generate("plane", &p, Quality::LOD0, &mut out)
            .unwrap_err();
        assert_eq!(err.code(), "geom.unknown_kind");
        assert!(reg.get("Sphere").is_none());
    }
}
