//! Typed primitive parameters, validation, analytic bounds and reference sets.

use crate::PrimKind;
use raa_geom::{GeomError, ParamValue, Params};
use raa_math::{Aabb, Vec3};

/// Parameters of one primitive, one variant per kind.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum PrimParams {
    Sphere {
        radius: f32,
    },
    Box {
        half_extents: [f32; 3],
    },
    RoundBox {
        half_extents: [f32; 3],
        radius: f32,
    },
    Capsule {
        radius: f32,
        half_height: f32,
    },
    Cylinder {
        radius: f32,
        half_height: f32,
    },
    Cone {
        radius: f32,
        half_height: f32,
    },
    Torus {
        major_radius: f32,
        minor_radius: f32,
    },
    Ellipsoid {
        radii: [f32; 3],
    },
}

fn invalid(kind: PrimKind, name: &str, reason: &str) -> GeomError {
    GeomError::InvalidParam {
        kind: kind.as_str().to_string(),
        name: name.to_string(),
        reason: reason.to_string(),
    }
}

fn positive(kind: PrimKind, name: &str, v: f32) -> Result<(), GeomError> {
    if !v.is_finite() {
        return Err(invalid(kind, name, "must be finite"));
    }
    if v <= 0.0 {
        return Err(invalid(kind, name, "must be > 0"));
    }
    Ok(())
}

fn positive3(kind: PrimKind, name: &str, v: [f32; 3]) -> Result<(), GeomError> {
    v.iter().try_for_each(|&c| positive(kind, name, c))
}

impl PrimParams {
    pub fn kind(&self) -> PrimKind {
        match self {
            PrimParams::Sphere { .. } => PrimKind::Sphere,
            PrimParams::Box { .. } => PrimKind::Box,
            PrimParams::RoundBox { .. } => PrimKind::RoundBox,
            PrimParams::Capsule { .. } => PrimKind::Capsule,
            PrimParams::Cylinder { .. } => PrimKind::Cylinder,
            PrimParams::Cone { .. } => PrimKind::Cone,
            PrimParams::Torus { .. } => PrimKind::Torus,
            PrimParams::Ellipsoid { .. } => PrimKind::Ellipsoid,
        }
    }

    /// Reads `p` for `kind`, refuses names the kind does not have, then validates.
    pub fn from_params(kind: PrimKind, p: &Params) -> Result<PrimParams, GeomError> {
        let k = kind.as_str();
        let s = |n: &str| p.scalar(k, n);
        let v = |n: &str| p.vec3(k, n);
        let out = match kind {
            PrimKind::Sphere => PrimParams::Sphere {
                radius: s("radius")?,
            },
            PrimKind::Box => PrimParams::Box {
                half_extents: v("halfExtents")?,
            },
            PrimKind::RoundBox => PrimParams::RoundBox {
                half_extents: v("halfExtents")?,
                radius: s("radius")?,
            },
            PrimKind::Capsule => PrimParams::Capsule {
                radius: s("radius")?,
                half_height: s("halfHeight")?,
            },
            PrimKind::Cylinder => PrimParams::Cylinder {
                radius: s("radius")?,
                half_height: s("halfHeight")?,
            },
            PrimKind::Cone => PrimParams::Cone {
                radius: s("radius")?,
                half_height: s("halfHeight")?,
            },
            PrimKind::Torus => PrimParams::Torus {
                major_radius: s("majorRadius")?,
                minor_radius: s("minorRadius")?,
            },
            PrimKind::Ellipsoid => PrimParams::Ellipsoid { radii: v("radii")? },
        };
        if let Some(extra) = p.names().find(|n| !kind.param_names().contains(n)) {
            return Err(invalid(kind, extra, "unknown parameter for this kind"));
        }
        out.validate()?;
        Ok(out)
    }

    /// Params in the S3 key order.
    pub fn to_params(&self) -> Params {
        use ParamValue::{Scalar, Vec3 as V3};
        match *self {
            PrimParams::Sphere { radius } => Params::new().with("radius", Scalar(radius)),
            PrimParams::Box { half_extents } => Params::new().with("halfExtents", V3(half_extents)),
            PrimParams::RoundBox {
                half_extents,
                radius,
            } => Params::new()
                .with("halfExtents", V3(half_extents))
                .with("radius", Scalar(radius)),
            PrimParams::Capsule {
                radius,
                half_height,
            }
            | PrimParams::Cylinder {
                radius,
                half_height,
            }
            | PrimParams::Cone {
                radius,
                half_height,
            } => Params::new()
                .with("radius", Scalar(radius))
                .with("halfHeight", Scalar(half_height)),
            PrimParams::Torus {
                major_radius,
                minor_radius,
            } => Params::new()
                .with("majorRadius", Scalar(major_radius))
                .with("minorRadius", Scalar(minor_radius)),
            PrimParams::Ellipsoid { radii } => Params::new().with("radii", V3(radii)),
        }
    }

    /// Applies the S3 rules plus finiteness.
    pub fn validate(&self) -> Result<(), GeomError> {
        let kind = self.kind();
        match *self {
            PrimParams::Sphere { radius } => positive(kind, "radius", radius),
            PrimParams::Box { half_extents } => positive3(kind, "halfExtents", half_extents),
            PrimParams::RoundBox {
                half_extents,
                radius,
            } => {
                positive3(kind, "halfExtents", half_extents)?;
                positive(kind, "radius", radius)?;
                let min = half_extents[0].min(half_extents[1]).min(half_extents[2]);
                if radius > min {
                    return Err(invalid(kind, "radius", "must be <= min(halfExtents)"));
                }
                Ok(())
            }
            PrimParams::Capsule {
                radius,
                half_height,
            } => {
                positive(kind, "radius", radius)?;
                if !half_height.is_finite() {
                    return Err(invalid(kind, "halfHeight", "must be finite"));
                }
                if half_height < 0.0 {
                    return Err(invalid(kind, "halfHeight", "must be >= 0"));
                }
                Ok(())
            }
            PrimParams::Cylinder {
                radius,
                half_height,
            }
            | PrimParams::Cone {
                radius,
                half_height,
            } => {
                positive(kind, "radius", radius)?;
                positive(kind, "halfHeight", half_height)
            }
            PrimParams::Torus {
                major_radius,
                minor_radius,
            } => {
                positive(kind, "majorRadius", major_radius)?;
                positive(kind, "minorRadius", minor_radius)?;
                if minor_radius >= major_radius {
                    return Err(invalid(kind, "minorRadius", "must be < majorRadius"));
                }
                Ok(())
            }
            PrimParams::Ellipsoid { radii } => positive3(kind, "radii", radii),
        }
    }

    /// Analytic bounds of the exact shape.
    pub fn bounds(&self) -> Aabb {
        let h = match *self {
            PrimParams::Sphere { radius } => [radius; 3],
            PrimParams::Box { half_extents } | PrimParams::RoundBox { half_extents, .. } => {
                half_extents
            }
            PrimParams::Capsule {
                radius,
                half_height,
            } => [radius, half_height + radius, radius],
            PrimParams::Cylinder {
                radius,
                half_height,
            }
            | PrimParams::Cone {
                radius,
                half_height,
            } => [radius, half_height, radius],
            PrimParams::Torus {
                major_radius,
                minor_radius,
            } => {
                let r = major_radius + minor_radius;
                [r, minor_radius, r]
            }
            PrimParams::Ellipsoid { radii } => radii,
        };
        let h = Vec3::from_array(h);
        Aabb { min: -h, max: h }
    }

    /// The S3 reference set: 'B' gives set B, any other char gives set A.
    pub fn reference_set(kind: PrimKind, set: char) -> PrimParams {
        let b = set == 'B';
        fn pick<T>(b: bool, a: T, bb: T) -> T {
            if b { bb } else { a }
        }
        match kind {
            PrimKind::Sphere => PrimParams::Sphere {
                radius: pick(b, 0.8, 1.7),
            },
            PrimKind::Box => PrimParams::Box {
                half_extents: pick(b, [0.7, 0.6, 0.5], [1.2, 0.3, 2.0]),
            },
            PrimKind::RoundBox => PrimParams::RoundBox {
                half_extents: pick(b, [0.7, 0.6, 0.5], [1.0, 1.0, 0.4]),
                radius: pick(b, 0.15, 0.35),
            },
            PrimKind::Capsule => PrimParams::Capsule {
                radius: pick(b, 0.4, 0.25),
                half_height: pick(b, 0.5, 1.2),
            },
            PrimKind::Cylinder => PrimParams::Cylinder {
                radius: pick(b, 0.6, 1.5),
                half_height: pick(b, 0.7, 0.2),
            },
            PrimKind::Cone => PrimParams::Cone {
                radius: pick(b, 0.7, 0.3),
                half_height: pick(b, 0.8, 1.5),
            },
            PrimKind::Torus => PrimParams::Torus {
                major_radius: pick(b, 0.6, 1.5),
                minor_radius: pick(b, 0.22, 0.1),
            },
            PrimKind::Ellipsoid => PrimParams::Ellipsoid {
                radii: pick(b, [0.9, 0.6, 0.5], [0.3, 1.4, 0.8]),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn code(r: Result<PrimParams, GeomError>) -> &'static str {
        r.expect_err("expected an error").code()
    }

    #[test]
    fn params_round_trip_for_both_sets() {
        for k in PrimKind::ALL {
            for set in ['A', 'B'] {
                let p = PrimParams::reference_set(k, set);
                assert_eq!(p.kind(), k);
                assert_eq!(PrimParams::from_params(k, &p.to_params()).unwrap(), p);
                let params = p.to_params();
                assert!(params.names().eq(k.param_names().iter().copied()));
            }
        }
    }

    #[test]
    fn reference_sets_validate_and_bounds_are_sane() {
        for k in PrimKind::ALL {
            for set in ['A', 'B', 'x'] {
                let p = PrimParams::reference_set(k, set);
                p.validate().unwrap();
                let b = p.bounds();
                assert!(b.max.x > 0.0 && b.max.y > 0.0 && b.max.z > 0.0);
                assert_eq!(b.min, -b.max);
            }
        }
        assert_eq!(
            PrimParams::reference_set(PrimKind::Sphere, 'x'),
            PrimParams::reference_set(PrimKind::Sphere, 'A')
        );
        let c = PrimParams::reference_set(PrimKind::Cone, 'A').bounds();
        assert_eq!((c.max.x, c.max.y), (0.7, 0.8));
    }

    #[test]
    fn s3_rules_are_enforced() {
        let bad = [
            PrimParams::Sphere { radius: 0.0 },
            PrimParams::Sphere { radius: f32::NAN },
            PrimParams::Sphere {
                radius: f32::INFINITY,
            },
            PrimParams::Box {
                half_extents: [1.0, 0.0, 1.0],
            },
            PrimParams::RoundBox {
                half_extents: [1.0, 0.5, 1.0],
                radius: 0.6,
            },
            PrimParams::RoundBox {
                half_extents: [1.0, 0.5, 1.0],
                radius: 0.0,
            },
            PrimParams::Capsule {
                radius: 0.0,
                half_height: 1.0,
            },
            PrimParams::Capsule {
                radius: 1.0,
                half_height: -0.1,
            },
            PrimParams::Cylinder {
                radius: 1.0,
                half_height: 0.0,
            },
            PrimParams::Cone {
                radius: -1.0,
                half_height: 1.0,
            },
            PrimParams::Cone {
                radius: 1.0,
                half_height: 0.0,
            },
            PrimParams::Torus {
                major_radius: 1.0,
                minor_radius: 1.0,
            },
            PrimParams::Torus {
                major_radius: 1.0,
                minor_radius: 0.0,
            },
            PrimParams::Ellipsoid {
                radii: [1.0, -1.0, 1.0],
            },
        ];
        for p in bad {
            let e = p.validate().expect_err("should be invalid");
            assert_eq!(e.code(), "geom.invalid_param", "{p:?}");
            assert!(
                matches!(&e, GeomError::InvalidParam { kind, .. } if kind == p.kind().as_str())
            );
        }
        assert!(
            PrimParams::Capsule {
                radius: 0.5,
                half_height: 0.0
            }
            .validate()
            .is_ok()
        );
        assert!(
            PrimParams::RoundBox {
                half_extents: [1.0, 0.5, 1.0],
                radius: 0.5
            }
            .validate()
            .is_ok()
        );
    }

    #[test]
    fn from_params_refuses_bad_input() {
        let missing = Params::new();
        assert_eq!(
            code(PrimParams::from_params(PrimKind::Sphere, &missing)),
            "geom.missing_param"
        );
        let half_only = Params::new().with("radius", ParamValue::Scalar(0.1));
        assert_eq!(
            code(PrimParams::from_params(PrimKind::Capsule, &half_only)),
            "geom.missing_param"
        );
        let wrong = Params::new().with("radius", ParamValue::Vec3([1.0; 3]));
        assert_eq!(
            code(PrimParams::from_params(PrimKind::Sphere, &wrong)),
            "geom.invalid_param"
        );
        let extra = Params::new()
            .with("radius", ParamValue::Scalar(1.0))
            .with("bogus", ParamValue::Scalar(1.0));
        assert_eq!(
            code(PrimParams::from_params(PrimKind::Sphere, &extra)),
            "geom.invalid_param"
        );
        let nan = Params::new().with("radius", ParamValue::Scalar(f32::NAN));
        assert_eq!(
            code(PrimParams::from_params(PrimKind::Sphere, &nan)),
            "geom.invalid_param"
        );
    }
}
