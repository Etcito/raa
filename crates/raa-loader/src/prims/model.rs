//! Document types of prims v0 and the S3 kind tables.

use crate::json::JsonValue;
use raa_math::{Mat4, Quat, Vec3};

/// A parsed prims document.
#[derive(Debug, Clone, PartialEq)]
pub struct PrimsAsset {
    pub id: String,
    pub name: String,
    pub triangle_budget: Option<u32>,
    pub nodes: Vec<PrimNode>,
    pub materials: Vec<(String, PrimMaterial)>,
    pub anchors: Vec<(String, TransformDoc)>,
    pub collision: Option<Vec<ColliderDoc>>,
}

/// One primitive node.
#[derive(Debug, Clone, PartialEq)]
pub struct PrimNode {
    pub id: String,
    pub kind: String,
    pub params: Vec<(String, JsonValue)>,
    pub transform: Option<TransformDoc>,
    pub material: Option<String>,
    pub op: NodeOp,
    pub children: Vec<PrimNode>,
}

/// Translation, quaternion `[x, y, z, w]` and scale.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformDoc {
    pub t: [f64; 3],
    pub r: [f64; 4],
    pub s: [f64; 3],
}

/// How a child combines with its parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeOp {
    #[default]
    Union,
    Subtract,
    Intersect,
}

/// A material as written in the document (colour in sRGB).
#[derive(Debug, Clone, PartialEq)]
pub struct PrimMaterial {
    pub color: [f64; 3],
    pub roughness: f64,
    pub metallic: f64,
    pub emissive: [f64; 3],
}

/// One entry of the `collision` override.
#[derive(Debug, Clone, PartialEq)]
pub struct ColliderDoc {
    pub node: Option<String>,
    pub shape: String,
    pub params: Vec<(String, JsonValue)>,
    pub transform: TransformDoc,
}

/// Type of a parameter value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // used by parse
pub(crate) enum ParamType {
    Scalar,
    Vec3,
}

type ParamList = &'static [(&'static str, ParamType)];

/// The S3 table: kind name and its params in S3 order.
#[allow(dead_code)] // used by parse
pub(crate) const KIND_TABLE: [(&str, ParamList); 8] = [
    ("sphere", &[("radius", ParamType::Scalar)]),
    ("box", &[("halfExtents", ParamType::Vec3)]),
    (
        "roundBox",
        &[
            ("halfExtents", ParamType::Vec3),
            ("radius", ParamType::Scalar),
        ],
    ),
    (
        "capsule",
        &[
            ("radius", ParamType::Scalar),
            ("halfHeight", ParamType::Scalar),
        ],
    ),
    (
        "cylinder",
        &[
            ("radius", ParamType::Scalar),
            ("halfHeight", ParamType::Scalar),
        ],
    ),
    (
        "cone",
        &[
            ("radius", ParamType::Scalar),
            ("halfHeight", ParamType::Scalar),
        ],
    ),
    (
        "torus",
        &[
            ("majorRadius", ParamType::Scalar),
            ("minorRadius", ParamType::Scalar),
        ],
    ),
    ("ellipsoid", &[("radii", ParamType::Vec3)]),
];

/// Collider shape table: shape name and its params in document order.
#[allow(dead_code)] // used by parse and collider derivation
pub(crate) const COLLIDER_TABLE: [(&str, ParamList); 6] = [
    ("ball", &[("radius", ParamType::Scalar)]),
    ("cuboid", &[("halfExtents", ParamType::Vec3)]),
    (
        "roundCuboid",
        &[
            ("halfExtents", ParamType::Vec3),
            ("borderRadius", ParamType::Scalar),
        ],
    ),
    (
        "capsuleY",
        &[
            ("halfHeight", ParamType::Scalar),
            ("radius", ParamType::Scalar),
        ],
    ),
    (
        "cylinderY",
        &[
            ("halfHeight", ParamType::Scalar),
            ("radius", ParamType::Scalar),
        ],
    ),
    (
        "coneY",
        &[
            ("halfHeight", ParamType::Scalar),
            ("radius", ParamType::Scalar),
        ],
    ),
];

/// Params of a kind in S3 order, `None` for an unknown kind.
#[allow(dead_code)] // used by parse
pub(crate) fn kind_params(kind: &str) -> Option<ParamList> {
    KIND_TABLE.iter().find(|(k, _)| *k == kind).map(|(_, p)| *p)
}

/// Params of a collider shape, `None` for an unknown shape.
#[allow(dead_code)] // used by parse
pub(crate) fn collider_params(shape: &str) -> Option<ParamList> {
    COLLIDER_TABLE
        .iter()
        .find(|(k, _)| *k == shape)
        .map(|(_, p)| *p)
}

impl NodeOp {
    /// Document name of the op.
    #[allow(dead_code)] // used by write and build
    pub(crate) fn name(self) -> &'static str {
        match self {
            NodeOp::Union => "union",
            NodeOp::Subtract => "subtract",
            NodeOp::Intersect => "intersect",
        }
    }

    /// Op for a document name.
    #[allow(dead_code)] // used by parse
    pub(crate) fn from_name(s: &str) -> Option<NodeOp> {
        [NodeOp::Union, NodeOp::Subtract, NodeOp::Intersect]
            .into_iter()
            .find(|o| o.name() == s)
    }
}

impl TransformDoc {
    #[allow(dead_code)] // used by parse and build
    pub(crate) fn identity() -> TransformDoc {
        TransformDoc {
            t: [0.0; 3],
            r: [0.0, 0.0, 0.0, 1.0],
            s: [1.0; 3],
        }
    }

    #[allow(dead_code)] // used by build
    pub(crate) fn to_mat4(&self) -> Mat4 {
        let f = |v: f64| v as f32;
        Mat4::from_trs(
            Vec3::new(f(self.t[0]), f(self.t[1]), f(self.t[2])),
            Quat::from_xyzw(f(self.r[0]), f(self.r[1]), f(self.r[2]), f(self.r[3])),
            Vec3::new(f(self.s[0]), f(self.s[1]), f(self.s[2])),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_prims::PrimKind;

    #[test]
    fn transform_doc_matches_from_trs() {
        let q = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.7);
        let doc = TransformDoc {
            t: [1.0, -2.0, 3.5],
            r: [q.x as f64, q.y as f64, q.z as f64, q.w as f64],
            s: [1.0, 2.0, 0.5],
        };
        let expected = Mat4::from_trs(Vec3::new(1.0, -2.0, 3.5), q, Vec3::new(1.0, 2.0, 0.5));
        assert_eq!(doc.to_mat4(), expected);
        assert_eq!(TransformDoc::identity().to_mat4(), Mat4::IDENTITY);
    }

    #[test]
    fn kind_table_matches_s3_and_raa_prims() {
        assert_eq!(KIND_TABLE.len(), 8);
        for k in PrimKind::ALL {
            let names: Vec<&str> = kind_params(k.as_str())
                .unwrap_or_else(|| panic!("missing kind {}", k.as_str()))
                .iter()
                .map(|(n, _)| *n)
                .collect();
            assert_eq!(names, k.param_names(), "kind {}", k.as_str());
        }
    }

    #[test]
    fn collider_table_and_ops() {
        assert_eq!(collider_params("roundCuboid").map(<[_]>::len), Some(2));
        assert!(collider_params("torus").is_none());
        assert_eq!(NodeOp::default(), NodeOp::Union);
        assert_eq!(NodeOp::from_name("subtract"), Some(NodeOp::Subtract));
        assert_eq!(NodeOp::Intersect.name(), "intersect");
    }
}
