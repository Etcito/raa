//! The fixed CSG case table (S3 set A sizes, hard-coded).

use raa_loader::json::JsonValue;
use raa_loader::prims::{
    NodeOp, PrimMaterial, PrimNode, PrimsAsset, TransformDoc, to_canonical_json,
};

/// One boolean case: an asset text and its expected outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct CsgCase {
    pub id: &'static str,
    /// `"ok"` or a `csg.*` code.
    pub expected: &'static str,
    /// Canonical prims v0 text.
    pub asset: String,
}

/// LOD0 spheres and tori exceed the default 1500, so every case states its own budget.
const BUDGET: u32 = 8000;
const IDENTITY: [f64; 4] = [0.0, 0.0, 0.0, 1.0];
const SQRT_HALF: f64 = std::f64::consts::FRAC_1_SQRT_2;

fn num(x: f64) -> JsonValue {
    JsonValue::Number(x)
}

fn vec3(v: [f64; 3]) -> JsonValue {
    JsonValue::Array(v.iter().map(|x| num(*x)).collect())
}

fn tf(t: [f64; 3], r: [f64; 4], s: [f64; 3]) -> Option<TransformDoc> {
    Some(TransformDoc { t, r, s })
}

fn at(t: [f64; 3]) -> Option<TransformDoc> {
    tf(t, IDENTITY, [1.0; 3])
}

/// Rotation of `deg` degrees about Y as `[x, y, z, w]`.
fn rot_y(deg: f64) -> [f64; 4] {
    let h = deg.to_radians() * 0.5;
    [0.0, h.sin(), 0.0, h.cos()]
}

fn node(
    id: &str,
    kind: &str,
    params: Vec<(&str, JsonValue)>,
    transform: Option<TransformDoc>,
    op: NodeOp,
    material: &str,
    children: Vec<PrimNode>,
) -> PrimNode {
    PrimNode {
        id: id.to_string(),
        kind: kind.to_string(),
        params: params
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
        transform,
        material: Some(material.to_string()),
        op,
        children,
    }
}

fn sphere(r: f64) -> (&'static str, Vec<(&'static str, JsonValue)>) {
    ("sphere", vec![("radius", num(r))])
}

fn boxk(h: [f64; 3]) -> (&'static str, Vec<(&'static str, JsonValue)>) {
    ("box", vec![("halfExtents", vec3(h))])
}

fn cylinder(r: f64, hh: f64) -> (&'static str, Vec<(&'static str, JsonValue)>) {
    (
        "cylinder",
        vec![("radius", num(r)), ("halfHeight", num(hh))],
    )
}

fn torus() -> (&'static str, Vec<(&'static str, JsonValue)>) {
    (
        "torus",
        vec![("majorRadius", num(0.6)), ("minorRadius", num(0.22))],
    )
}

type Shape = (&'static str, Vec<(&'static str, JsonValue)>);

const PALETTE: [[f64; 3]; 20] = [
    [0.85, 0.30, 0.25],
    [0.25, 0.60, 0.85],
    [0.35, 0.75, 0.35],
    [0.90, 0.70, 0.20],
    [0.65, 0.40, 0.80],
    [0.20, 0.75, 0.70],
    [0.90, 0.50, 0.15],
    [0.55, 0.55, 0.90],
    [0.75, 0.85, 0.30],
    [0.90, 0.40, 0.65],
    [0.45, 0.70, 0.55],
    [0.70, 0.30, 0.30],
    [0.30, 0.45, 0.70],
    [0.60, 0.60, 0.25],
    [0.80, 0.55, 0.45],
    [0.40, 0.30, 0.65],
    [0.30, 0.65, 0.50],
    [0.85, 0.85, 0.40],
    [0.50, 0.35, 0.25],
    [0.60, 0.75, 0.85],
];

struct Tool {
    id: &'static str,
    shape: Shape,
    op: NodeOp,
    transform: Option<TransformDoc>,
    children: Vec<PrimNode>,
}

fn tool(id: &'static str, shape: Shape, op: NodeOp, transform: Option<TransformDoc>) -> Tool {
    Tool {
        id,
        shape,
        op,
        transform,
        children: Vec::new(),
    }
}

fn sub(id: &'static str, shape: Shape, transform: Option<TransformDoc>) -> Tool {
    tool(id, shape, NodeOp::Subtract, transform)
}

fn int(id: &'static str, shape: Shape, transform: Option<TransformDoc>) -> Tool {
    tool(id, shape, NodeOp::Intersect, transform)
}

fn build(index: usize, id: &str, host: Shape, tools: Vec<Tool>) -> String {
    let children = tools
        .into_iter()
        .map(|t| {
            node(
                t.id,
                t.shape.0,
                t.shape.1,
                t.transform,
                t.op,
                "tool",
                t.children,
            )
        })
        .collect();
    let root = node(
        "host",
        host.0,
        host.1,
        None,
        NodeOp::Union,
        "body",
        children,
    );
    let color = PALETTE[index % PALETTE.len()];
    let tool_color = [0.15, 0.15, 0.18];
    let asset = PrimsAsset {
        id: id.to_string(),
        name: id.to_string(),
        triangle_budget: Some(BUDGET),
        nodes: vec![root],
        materials: vec![
            (
                "body".into(),
                PrimMaterial {
                    color,
                    roughness: 0.55,
                    metallic: 0.0,
                    emissive: [0.0; 3],
                },
            ),
            (
                "tool".into(),
                PrimMaterial {
                    color: tool_color,
                    roughness: 0.8,
                    metallic: 0.0,
                    emissive: [0.0; 3],
                },
            ),
        ],
        anchors: Vec::new(),
        collision: None,
    };
    to_canonical_json(&asset)
}

/// The 20 cases in contract order: 11 ok cases, then 9 refusals.
pub fn cases() -> Vec<CsgCase> {
    let host_box = || boxk([0.7, 0.6, 0.5]);
    let host_sphere = || sphere(0.8);
    let host_cyl = || cylinder(0.6, 0.7);
    let thin_cyl = || tf([0.113, 0.0, 0.043], IDENTITY, [0.4, 1.0, 0.4]);
    let table: Vec<(&'static str, &'static str, Shape, Vec<Tool>)> = vec![
        (
            "box-sub-sphere-dent",
            "ok",
            host_box(),
            vec![sub("dent", sphere(0.4), at([0.013, 0.613, 0.007]))],
        ),
        (
            "box-sub-cylinder-through",
            "ok",
            host_box(),
            vec![sub("hole", cylinder(0.6, 0.7), thin_cyl())],
        ),
        (
            "sphere-sub-box-corner",
            "ok",
            host_sphere(),
            vec![sub("bite", host_box(), at([0.613, 0.507, 0.391]))],
        ),
        (
            "cylinder-sub-capsule-slot",
            "ok",
            host_cyl(),
            vec![sub(
                "slot",
                (
                    "capsule",
                    vec![("radius", num(0.4)), ("halfHeight", num(0.5))],
                ),
                tf(
                    [0.011, 0.613, 0.007],
                    [0.0, 0.0, SQRT_HALF, SQRT_HALF],
                    [0.6, 1.0, 0.6],
                ),
            )],
        ),
        (
            "box-sub-ellipsoid-scoop",
            "ok",
            host_box(),
            vec![sub(
                "scoop",
                ("ellipsoid", vec![("radii", vec3([0.9, 0.6, 0.5]))]),
                tf([0.713, 0.011, 0.007], IDENTITY, [0.5; 3]),
            )],
        ),
        (
            "box-int-sphere",
            "ok",
            host_box(),
            vec![int("ball", host_sphere(), at([0.013, -0.011, 0.017]))],
        ),
        (
            "sphere-int-cylinder",
            "ok",
            host_sphere(),
            vec![int("core", host_cyl(), at([0.013, 0.011, -0.017]))],
        ),
        (
            "cone-int-box",
            "ok",
            ("cone", vec![("radius", num(0.7)), ("halfHeight", num(0.8))]),
            vec![int(
                "block",
                host_box(),
                tf([0.013, 0.021, 0.017], rot_y(30.0), [1.0; 3]),
            )],
        ),
        (
            "box-sub-two-spheres",
            "ok",
            host_box(),
            vec![
                sub("dent-a", sphere(0.3), at([-0.311, 0.413, 0.011])),
                sub("dent-b", sphere(0.3), at([0.329, 0.371, -0.017])),
            ],
        ),
        (
            "box-int-sphere-sub-cylinder",
            "ok",
            host_box(),
            vec![
                int("ball", host_sphere(), at([0.013, -0.011, 0.017])),
                sub("hole", cylinder(0.6, 0.7), thin_cyl()),
            ],
        ),
        (
            "box-sub-sphere-outside",
            "ok",
            host_box(),
            vec![sub("far", host_sphere(), at([3.0, 0.0, 0.0]))],
        ),
        (
            "torus-host-sub-sphere",
            "boolean.csg.host_not_convex",
            torus(),
            vec![sub("bite", sphere(0.3), at([0.613, 0.0, 0.0]))],
        ),
        (
            "box-sub-torus",
            "boolean.csg.tool_not_convex",
            host_box(),
            vec![sub("ring", torus(), at([0.013, 0.011, 0.0]))],
        ),
        (
            "box-int-torus",
            "boolean.csg.tool_not_convex",
            host_box(),
            vec![int("ring", torus(), at([0.013, 0.011, 0.0]))],
        ),
        (
            "box-sub-box-coplanar",
            "boolean.csg.coplanar",
            host_box(),
            vec![sub("flush", boxk([0.3, 0.3, 0.3]), at([0.0, 0.3, 0.0]))],
        ),
        (
            "box-int-sphere-disjoint",
            "boolean.csg.empty_result",
            host_box(),
            vec![int("far", host_sphere(), at([3.0, 0.0, 0.0]))],
        ),
        (
            "sphere-sub-box-containing",
            "boolean.csg.empty_result",
            host_sphere(),
            vec![sub("huge", host_box(), tf([0.0; 3], IDENTITY, [2.0; 3]))],
        ),
        (
            "box-sub-overlapping-spheres",
            "boolean.csg.overlapping_tools",
            host_box(),
            vec![
                sub("dent-a", sphere(0.3), at([-0.113, 0.413, 0.011])),
                sub("dent-b", sphere(0.3), at([0.113, 0.371, -0.017])),
            ],
        ),
        (
            "box-sub-tool-with-children",
            "boolean.csg.tool_has_children",
            host_box(),
            vec![{
                let mut t = sub("parent", sphere(0.3), at([0.013, 0.413, 0.011]));
                t.children.push(node(
                    "child",
                    "sphere",
                    vec![("radius", num(0.1))],
                    at([0.2, 0.0, 0.0]),
                    NodeOp::Union,
                    "tool",
                    Vec::new(),
                ));
                t
            }],
        ),
        (
            "box-two-intersects",
            "boolean.csg.multiple_intersect",
            host_box(),
            vec![
                int("ball-a", host_sphere(), at([0.013, -0.011, 0.017])),
                int("ball-b", sphere(0.75), at([-0.021, 0.017, 0.009])),
            ],
        ),
    ];
    table
        .into_iter()
        .enumerate()
        .map(|(i, (id, expected, host, tools))| CsgCase {
            id,
            expected,
            asset: build(i, id, host, tools),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_loader::prims::parse;

    const ORDER: [&str; 20] = [
        "box-sub-sphere-dent",
        "box-sub-cylinder-through",
        "sphere-sub-box-corner",
        "cylinder-sub-capsule-slot",
        "box-sub-ellipsoid-scoop",
        "box-int-sphere",
        "sphere-int-cylinder",
        "cone-int-box",
        "box-sub-two-spheres",
        "box-int-sphere-sub-cylinder",
        "box-sub-sphere-outside",
        "torus-host-sub-sphere",
        "box-sub-torus",
        "box-int-torus",
        "box-sub-box-coplanar",
        "box-int-sphere-disjoint",
        "sphere-sub-box-containing",
        "box-sub-overlapping-spheres",
        "box-sub-tool-with-children",
        "box-two-intersects",
    ];

    #[test]
    fn table_has_20_cases_in_contract_order() {
        let all = cases();
        let ids: Vec<&str> = all.iter().map(|c| c.id).collect();
        assert_eq!(ids, ORDER);
        assert_eq!(all.iter().filter(|c| c.expected == "ok").count(), 11);
        assert!(all[..11].iter().all(|c| c.expected == "ok"));
    }

    #[test]
    fn every_text_parses_and_is_canonical() {
        for c in cases() {
            let parsed = parse(&c.asset).unwrap_or_else(|e| panic!("{}: {e}", c.id));
            assert_eq!(
                to_canonical_json(&parsed),
                c.asset,
                "{} not canonical",
                c.id
            );
        }
    }
}
