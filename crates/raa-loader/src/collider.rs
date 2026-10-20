//! Collider shapes of a loaded asset (S4), their derivation from union nodes and the
//! conversion of `collision` override entries.

use crate::LoadError;
use crate::json::JsonValue;
use crate::prims::ColliderDoc;
use raa_geom::MeshData;
use raa_math::{Mat4, Vec3};

/// A collision shape. Uniform scale is folded into the dimensions.
#[derive(Debug, Clone, PartialEq)]
pub enum ColliderShape {
    Ball {
        radius: f32,
    },
    Cuboid {
        half_extents: [f32; 3],
    },
    RoundCuboid {
        half_extents: [f32; 3],
        border_radius: f32,
    },
    CapsuleY {
        half_height: f32,
        radius: f32,
    },
    CylinderY {
        half_height: f32,
        radius: f32,
    },
    ConeY {
        half_height: f32,
        radius: f32,
    },
    ConvexHull {
        points: Vec<[f32; 3]>,
    },
}

/// A collider in asset space, derived from a node or given by an override.
#[derive(Debug, Clone, PartialEq)]
pub struct Collider {
    /// The node it was derived from or attached to, if any.
    pub node_id: Option<String>,
    pub shape: ColliderShape,
    pub transform: Mat4,
}

/// Relative tolerance between world column lengths for a scale to count as uniform.
const UNIFORM_SCALE_REL: f32 = 1e-5;
/// Hull points are de-duplicated after rounding to this step.
const HULL_ROUND: f64 = 1e-6;

/// Lengths of the three basis columns of `m`.
fn column_lengths(m: &Mat4) -> [f32; 3] {
    let len = |c: &[f32; 4]| Vec3::new(c[0], c[1], c[2]).length();
    [len(&m.cols[0]), len(&m.cols[1]), len(&m.cols[2])]
}

/// The uniform scale of `m`, `None` when the column lengths differ by more than 1e-5 relative.
fn uniform_scale(m: &Mat4) -> Option<f32> {
    let l = column_lengths(m);
    let max = l[0].max(l[1]).max(l[2]);
    let min = l[0].min(l[1]).min(l[2]);
    if !(min.is_finite() && max.is_finite()) || min <= 0.0 {
        return None;
    }
    if max - min <= UNIFORM_SCALE_REL * max {
        Some((l[0] + l[1] + l[2]) / 3.0)
    } else {
        None
    }
}

fn param<'a>(params: &'a [(String, JsonValue)], name: &str) -> Option<&'a JsonValue> {
    params.iter().find(|(k, _)| k == name).map(|(_, v)| v)
}

fn finite(x: f64) -> Option<f32> {
    let f = x as f32;
    f.is_finite().then_some(f)
}

fn scalar_of(params: &[(String, JsonValue)], name: &str) -> Option<f32> {
    match param(params, name)? {
        JsonValue::Number(x) => finite(*x),
        _ => None,
    }
}

fn vec3_of(params: &[(String, JsonValue)], name: &str) -> Option<[f32; 3]> {
    match param(params, name)? {
        JsonValue::Array(a) => match a.as_slice() {
            [
                JsonValue::Number(x),
                JsonValue::Number(y),
                JsonValue::Number(z),
            ] => Some([finite(*x)?, finite(*y)?, finite(*z)?]),
            _ => None,
        },
        _ => None,
    }
}

/// Analytic shape of an S3 kind in node-local units, `None` for kinds that need a hull.
/// `Some(Err(name))` names the first bad param.
fn analytic_shape(
    kind: &str,
    params: &[(String, JsonValue)],
) -> Option<Result<ColliderShape, &'static str>> {
    let s = |n: &'static str| scalar_of(params, n).ok_or(n);
    let v = |n: &'static str| vec3_of(params, n).ok_or(n);
    let shape = match kind {
        "sphere" => s("radius").map(|radius| ColliderShape::Ball { radius }),
        "box" => v("halfExtents").map(|half_extents| ColliderShape::Cuboid { half_extents }),
        "roundBox" => v("halfExtents").and_then(|half_extents| {
            s("radius").map(|border_radius| ColliderShape::RoundCuboid {
                half_extents,
                border_radius,
            })
        }),
        "capsule" => s("radius").and_then(|radius| {
            s("halfHeight").map(|half_height| ColliderShape::CapsuleY {
                half_height,
                radius,
            })
        }),
        "cylinder" => s("radius").and_then(|radius| {
            s("halfHeight").map(|half_height| ColliderShape::CylinderY {
                half_height,
                radius,
            })
        }),
        "cone" => s("radius").and_then(|radius| {
            s("halfHeight").map(|half_height| ColliderShape::ConeY {
                half_height,
                radius,
            })
        }),
        _ => return None,
    };
    Some(shape)
}

/// Multiplies every dimension of an analytic shape by `k`.
fn scale_shape(shape: ColliderShape, k: f32) -> ColliderShape {
    let v = |a: [f32; 3]| [a[0] * k, a[1] * k, a[2] * k];
    match shape {
        ColliderShape::Ball { radius } => ColliderShape::Ball { radius: radius * k },
        ColliderShape::Cuboid { half_extents } => ColliderShape::Cuboid {
            half_extents: v(half_extents),
        },
        ColliderShape::RoundCuboid {
            half_extents,
            border_radius,
        } => ColliderShape::RoundCuboid {
            half_extents: v(half_extents),
            border_radius: border_radius * k,
        },
        ColliderShape::CapsuleY {
            half_height,
            radius,
        } => ColliderShape::CapsuleY {
            half_height: half_height * k,
            radius: radius * k,
        },
        ColliderShape::CylinderY {
            half_height,
            radius,
        } => ColliderShape::CylinderY {
            half_height: half_height * k,
            radius: radius * k,
        },
        ColliderShape::ConeY {
            half_height,
            radius,
        } => ColliderShape::ConeY {
            half_height: half_height * k,
            radius: radius * k,
        },
        ColliderShape::ConvexHull { points } => ColliderShape::ConvexHull { points },
    }
}

/// Mesh positions transformed by `world`, rounded to 1e-6, de-duplicated and sorted.
fn hull_points(mesh: &MeshData, world: &Mat4) -> Vec<[f32; 3]> {
    let q = |x: f32| (f64::from(x) / HULL_ROUND).round() as i64;
    let mut keys: Vec<[i64; 3]> = mesh
        .positions
        .iter()
        .map(|p| {
            let w = world.transform_point(Vec3::from_array(*p));
            [q(w.x), q(w.y), q(w.z)]
        })
        .collect();
    keys.sort_unstable();
    keys.dedup();
    let f = |k: i64| (k as f64 * HULL_ROUND) as f32;
    keys.into_iter()
        .map(|k| [f(k[0]), f(k[1]), f(k[2])])
        .collect()
}

/// Derives the collider of one union node (S4).
///
/// `params` are the node's params in S3 order, `world` its asset-space matrix, `pointer` its
/// JSON pointer, and `low_mesh` yields the node's LOW-quality mesh; it is only called when a
/// convex hull is needed.
#[allow(dead_code)] // used by the prims build step
pub(crate) fn derive_collider(
    node_id: &str,
    kind: &str,
    params: &[(String, JsonValue)],
    world: &Mat4,
    pointer: &str,
    low_mesh: impl FnOnce() -> Result<MeshData, LoadError>,
) -> Result<Collider, LoadError> {
    if kind == "torus" {
        return Err(LoadError::new(
            "collision.torus_needs_override",
            pointer,
            format!("node '{node_id}' is a torus: give a collision override for the asset"),
        ));
    }
    if let (Some(scale), Some(shape)) = (uniform_scale(world), analytic_shape(kind, params)) {
        let shape = shape.map_err(|name| {
            LoadError::new(
                "kind.params",
                format!("{pointer}/params/{name}"),
                format!("{kind} needs a valid '{name}' to derive its collider"),
            )
        })?;
        let unscale = Mat4::from_scale(Vec3::splat(1.0 / scale));
        return Ok(Collider {
            node_id: Some(node_id.to_string()),
            shape: scale_shape(shape, scale),
            transform: world.mul(&unscale),
        });
    }
    let mesh = low_mesh()?;
    let points = hull_points(&mesh, world);
    if points.is_empty() {
        return Err(LoadError::new(
            "collision.shape",
            pointer,
            format!("node '{node_id}' has no mesh points for its convex hull"),
        ));
    }
    Ok(Collider {
        node_id: Some(node_id.to_string()),
        shape: ColliderShape::ConvexHull { points },
        transform: Mat4::IDENTITY,
    })
}

fn shape_error(pointer: &str, suffix: &str, message: String) -> LoadError {
    LoadError::new("collision.shape", format!("{pointer}{suffix}"), message)
}

/// Converts one `collision` override entry to a collider (S4).
///
/// `pointer` is the entry's JSON pointer; `node_world` gives the asset-space matrix of a node
/// id. The transform is `doc.transform` composed after the node's world when `node` is set, in
/// asset space otherwise. Unknown shapes, bad params and unknown nodes give `collision.shape`.
#[allow(dead_code)] // used by the prims build step
pub(crate) fn collider_from_doc(
    doc: &ColliderDoc,
    pointer: &str,
    node_world: impl Fn(&str) -> Option<Mat4>,
) -> Result<Collider, LoadError> {
    let names: &[&str] = match doc.shape.as_str() {
        "ball" => &["radius"],
        "cuboid" => &["halfExtents"],
        "roundCuboid" => &["halfExtents", "borderRadius"],
        "capsuleY" | "cylinderY" | "coneY" => &["halfHeight", "radius"],
        other => {
            return Err(shape_error(
                pointer,
                "/shape",
                format!("unknown collider shape '{other}'"),
            ));
        }
    };
    if let Some((k, _)) = doc
        .params
        .iter()
        .find(|(k, _)| !names.contains(&k.as_str()))
    {
        return Err(shape_error(
            pointer,
            &format!("/params/{k}"),
            format!("{} has no param '{k}'", doc.shape),
        ));
    }
    let shape_name = doc.shape.as_str();
    let bad = |name: &str, rule: &str| {
        shape_error(
            pointer,
            &format!("/params/{name}"),
            format!("{shape_name} '{name}' must be {rule}"),
        )
    };
    let pos = |name: &'static str| -> Result<f32, LoadError> {
        scalar_of(&doc.params, name)
            .filter(|x| *x > 0.0)
            .ok_or_else(|| bad(name, "a number > 0"))
    };
    let nonneg = |name: &'static str| -> Result<f32, LoadError> {
        scalar_of(&doc.params, name)
            .filter(|x| *x >= 0.0)
            .ok_or_else(|| bad(name, "a number >= 0"))
    };
    let ext = |name: &'static str| -> Result<[f32; 3], LoadError> {
        vec3_of(&doc.params, name)
            .filter(|v| v.iter().all(|x| *x > 0.0))
            .ok_or_else(|| bad(name, "3 numbers > 0"))
    };
    let shape = match shape_name {
        "ball" => ColliderShape::Ball {
            radius: pos("radius")?,
        },
        "cuboid" => ColliderShape::Cuboid {
            half_extents: ext("halfExtents")?,
        },
        "roundCuboid" => {
            let half_extents = ext("halfExtents")?;
            let border_radius = pos("borderRadius")?;
            let min = half_extents[0].min(half_extents[1]).min(half_extents[2]);
            if border_radius > min {
                return Err(bad("borderRadius", "<= the smallest half extent"));
            }
            ColliderShape::RoundCuboid {
                half_extents,
                border_radius,
            }
        }
        "capsuleY" => ColliderShape::CapsuleY {
            half_height: nonneg("halfHeight")?,
            radius: pos("radius")?,
        },
        "cylinderY" => ColliderShape::CylinderY {
            half_height: pos("halfHeight")?,
            radius: pos("radius")?,
        },
        "coneY" => ColliderShape::ConeY {
            half_height: pos("halfHeight")?,
            radius: pos("radius")?,
        },
        other => {
            return Err(shape_error(
                pointer,
                "/shape",
                format!("unknown collider shape '{other}'"),
            ));
        }
    };
    let local = doc.transform.to_mat4();
    let transform = match &doc.node {
        Some(id) => {
            let world = node_world(id).ok_or_else(|| {
                shape_error(
                    pointer,
                    "/node",
                    format!("collider references unknown node '{id}'"),
                )
            })?;
            world.mul(&local)
        }
        None => local,
    };
    Ok(Collider {
        node_id: doc.node.clone(),
        shape,
        transform,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prims::TransformDoc;
    use raa_geom::Quality;
    use raa_math::Quat;
    use raa_prims::PrimParams;

    fn num(x: f64) -> JsonValue {
        JsonValue::Number(x)
    }

    fn vec(a: [f64; 3]) -> JsonValue {
        JsonValue::Array(a.iter().map(|x| num(*x)).collect())
    }

    fn low(p: PrimParams) -> impl FnOnce() -> Result<MeshData, LoadError> {
        move || {
            let mut m = MeshData::new();
            raa_prims::generate(&p, Quality::LOW, &mut m).expect("valid test params");
            Ok(m)
        }
    }

    fn no_mesh() -> Result<MeshData, LoadError> {
        panic!("mesh must not be requested")
    }

    fn assert_rigid(m: &Mat4) {
        for l in column_lengths(m) {
            assert!((l - 1.0).abs() < 1e-5, "column length {l}");
        }
    }

    #[test]
    fn uniform_sphere_folds_scale_into_ball() {
        let q = Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0).normalize_or_zero(), 0.4);
        let world = Mat4::from_trs(Vec3::new(1.0, 2.0, 3.0), q, Vec3::splat(2.0));
        let params = vec![("radius".to_string(), num(0.5))];
        let c = derive_collider("s", "sphere", &params, &world, "/nodes/0", no_mesh).unwrap();
        match c.shape {
            ColliderShape::Ball { radius } => assert!((radius - 1.0).abs() < 1e-6),
            other => panic!("expected ball, got {other:?}"),
        }
        assert_rigid(&c.transform);
        assert_eq!(c.transform.cols[3], [1.0, 2.0, 3.0, 1.0]);
        assert_eq!(c.node_id.as_deref(), Some("s"));
    }

    #[test]
    fn non_uniform_box_gives_hull_within_bounds() {
        let world = Mat4::from_trs(
            Vec3::new(0.5, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::new(1.0, 2.0, 1.0),
        );
        let he = [0.7f32, 0.6, 0.5];
        let params = vec![("halfExtents".to_string(), vec([0.7, 0.6, 0.5]))];
        let p = PrimParams::Box { half_extents: he };
        let c = derive_collider("b", "box", &params, &world, "/nodes/0", low(p)).unwrap();
        assert_eq!(c.transform, Mat4::IDENTITY);
        let ColliderShape::ConvexHull { points } = c.shape else {
            panic!("expected hull")
        };
        assert!(!points.is_empty());
        let (lo, hi) = ([-0.2f32, -1.2, -0.5], [1.2f32, 1.2, 0.5]);
        for pt in &points {
            for i in 0..3 {
                assert!(pt[i] >= lo[i] - 1e-5 && pt[i] <= hi[i] + 1e-5, "{pt:?}");
            }
        }
        let mut sorted = points.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted.dedup();
        assert_eq!(sorted, points);
    }

    #[test]
    fn ellipsoid_gives_hull() {
        let params = vec![("radii".to_string(), vec([0.9, 0.6, 0.5]))];
        let p = PrimParams::Ellipsoid {
            radii: [0.9, 0.6, 0.5],
        };
        let c = derive_collider(
            "e",
            "ellipsoid",
            &params,
            &Mat4::IDENTITY,
            "/nodes/1",
            low(p),
        )
        .unwrap();
        assert!(matches!(c.shape, ColliderShape::ConvexHull { .. }));
        assert_eq!(c.transform, Mat4::IDENTITY);
    }

    #[test]
    fn torus_is_refused_at_node_pointer() {
        let params = vec![
            ("majorRadius".to_string(), num(0.6)),
            ("minorRadius".to_string(), num(0.2)),
        ];
        let e = derive_collider(
            "t",
            "torus",
            &params,
            &Mat4::IDENTITY,
            "/nodes/0/children/1",
            no_mesh,
        )
        .unwrap_err();
        assert_eq!(e.code, "collision.torus_needs_override");
        assert_eq!(e.path, "/nodes/0/children/1");
        assert!(!e.message.is_empty());
    }

    #[test]
    fn override_with_node_uses_node_world() {
        let node_world = Mat4::from_translation(Vec3::new(0.0, 3.0, 0.0));
        let doc = ColliderDoc {
            node: Some("body".to_string()),
            shape: "ball".to_string(),
            params: vec![("radius".to_string(), num(0.25))],
            transform: TransformDoc {
                t: [1.0, 0.0, 0.0],
                ..TransformDoc::identity()
            },
        };
        let lookup = |id: &str| (id == "body").then_some(node_world);
        let c = collider_from_doc(&doc, "/collision/colliders/0", lookup).unwrap();
        assert_eq!(c.shape, ColliderShape::Ball { radius: 0.25 });
        assert_eq!(c.transform.cols[3], [1.0, 3.0, 0.0, 1.0]);
        assert_eq!(c.node_id.as_deref(), Some("body"));

        let free = ColliderDoc {
            node: None,
            ..doc.clone()
        };
        let c = collider_from_doc(&free, "/collision/colliders/0", lookup).unwrap();
        assert_eq!(c.transform.cols[3], [1.0, 0.0, 0.0, 1.0]);

        let missing = ColliderDoc {
            node: Some("nope".to_string()),
            ..doc
        };
        let e = collider_from_doc(&missing, "/c", lookup).unwrap_err();
        assert_eq!(e.code, "collision.shape");
    }

    #[test]
    fn capsule_y_maps_s3_names() {
        let params = vec![
            ("radius".to_string(), num(0.4)),
            ("halfHeight".to_string(), num(0.5)),
        ];
        let c = derive_collider(
            "c",
            "capsule",
            &params,
            &Mat4::IDENTITY,
            "/nodes/0",
            no_mesh,
        )
        .unwrap();
        assert_eq!(
            c.shape,
            ColliderShape::CapsuleY {
                half_height: 0.5,
                radius: 0.4
            }
        );
        assert_eq!(c.transform, Mat4::IDENTITY);

        let doc = ColliderDoc {
            node: None,
            shape: "capsuleY".to_string(),
            params: vec![
                ("halfHeight".to_string(), num(0.5)),
                ("radius".to_string(), num(0.4)),
            ],
            transform: TransformDoc::identity(),
        };
        let c = collider_from_doc(&doc, "/c", |_| None).unwrap();
        assert_eq!(
            c.shape,
            ColliderShape::CapsuleY {
                half_height: 0.5,
                radius: 0.4
            }
        );
    }

    #[test]
    fn bad_override_params_give_collision_shape() {
        let base = ColliderDoc {
            node: None,
            shape: "roundCuboid".to_string(),
            params: vec![
                ("halfExtents".to_string(), vec([0.5, 0.5, 0.5])),
                ("borderRadius".to_string(), num(0.8)),
            ],
            transform: TransformDoc::identity(),
        };
        let e = collider_from_doc(&base, "/c", |_| None).unwrap_err();
        assert_eq!(
            (e.code, e.path.as_str()),
            ("collision.shape", "/c/params/borderRadius")
        );
        let unknown = ColliderDoc {
            shape: "torus".to_string(),
            ..base.clone()
        };
        assert_eq!(
            collider_from_doc(&unknown, "/c", |_| None)
                .unwrap_err()
                .code,
            "collision.shape"
        );
        let missing = ColliderDoc {
            params: vec![],
            ..base
        };
        assert_eq!(
            collider_from_doc(&missing, "/c", |_| None)
                .unwrap_err()
                .code,
            "collision.shape"
        );
    }
}
