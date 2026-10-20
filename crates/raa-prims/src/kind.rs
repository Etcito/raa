//! The eight primitive kinds and their S3 names.

/// A primitive kind.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PrimKind {
    Sphere,
    Box,
    RoundBox,
    Capsule,
    Cylinder,
    Cone,
    Torus,
    Ellipsoid,
}

impl PrimKind {
    pub const ALL: [PrimKind; 8] = [
        PrimKind::Sphere,
        PrimKind::Box,
        PrimKind::RoundBox,
        PrimKind::Capsule,
        PrimKind::Cylinder,
        PrimKind::Cone,
        PrimKind::Torus,
        PrimKind::Ellipsoid,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            PrimKind::Sphere => "sphere",
            PrimKind::Box => "box",
            PrimKind::RoundBox => "roundBox",
            PrimKind::Capsule => "capsule",
            PrimKind::Cylinder => "cylinder",
            PrimKind::Cone => "cone",
            PrimKind::Torus => "torus",
            PrimKind::Ellipsoid => "ellipsoid",
        }
    }

    pub fn parse(s: &str) -> Option<PrimKind> {
        PrimKind::ALL.into_iter().find(|k| k.as_str() == s)
    }

    pub fn is_convex(&self) -> bool {
        !matches!(self, PrimKind::Torus)
    }

    pub fn param_names(&self) -> &'static [&'static str] {
        match self {
            PrimKind::Sphere => &["radius"],
            PrimKind::Box => &["halfExtents"],
            PrimKind::RoundBox => &["halfExtents", "radius"],
            PrimKind::Capsule | PrimKind::Cylinder => &["radius", "halfHeight"],
            PrimKind::Cone => &["radius", "halfHeight", "radiusTop"],
            PrimKind::Torus => &["majorRadius", "minorRadius"],
            PrimKind::Ellipsoid => &["radii"],
        }
    }

    /// The keys every document must give: `param_names` minus the optional `radiusTop`.
    pub fn required_param_names(&self) -> &'static [&'static str] {
        match self {
            PrimKind::Cone => &["radius", "halfHeight"],
            _ => self.param_names(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_as_str_round_trip() {
        for k in PrimKind::ALL {
            assert_eq!(PrimKind::parse(k.as_str()), Some(k));
        }
        assert_eq!(PrimKind::parse("Sphere"), None);
        assert_eq!(PrimKind::parse(""), None);
    }

    #[test]
    fn convexity_and_param_names() {
        for k in PrimKind::ALL {
            assert_eq!(k.is_convex(), k != PrimKind::Torus);
            assert!(!k.param_names().is_empty());
        }
        assert_eq!(PrimKind::RoundBox.param_names(), &["halfExtents", "radius"]);
        assert_eq!(
            PrimKind::Torus.param_names(),
            &["majorRadius", "minorRadius"]
        );
    }
}
