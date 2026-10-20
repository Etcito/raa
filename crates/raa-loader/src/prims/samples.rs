//! Code-built sample assets and the invalid-case refusal table.

use super::{
    ColliderDoc, NodeOp, PrimMaterial, PrimNode, PrimsAsset, TransformDoc, to_canonical_json,
};
use crate::json::JsonValue;

fn num(x: f64) -> JsonValue {
    JsonValue::Number(x)
}

fn vec3(v: [f64; 3]) -> JsonValue {
    JsonValue::Array(v.iter().map(|x| num(*x)).collect())
}

fn tf(t: [f64; 3]) -> Option<TransformDoc> {
    Some(TransformDoc {
        t,
        r: [0.0, 0.0, 0.0, 1.0],
        s: [1.0; 3],
    })
}

fn tf_full(t: [f64; 3], r: [f64; 4], s: [f64; 3]) -> TransformDoc {
    TransformDoc { t, r, s }
}

fn node(
    id: &str,
    kind: &str,
    params: Vec<(&str, JsonValue)>,
    transform: Option<TransformDoc>,
    material: Option<&str>,
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
        material: material.map(str::to_string),
        op: NodeOp::Union,
        children,
    }
}

fn sphere(id: &str, r: f64, t: [f64; 3], m: Option<&str>) -> PrimNode {
    node(id, "sphere", vec![("radius", num(r))], tf(t), m, vec![])
}

fn capsule(id: &str, r: f64, hh: f64, t: [f64; 3]) -> PrimNode {
    node(
        id,
        "capsule",
        vec![("radius", num(r)), ("halfHeight", num(hh))],
        tf(t),
        None,
        vec![],
    )
}

fn material(color: [f64; 3], roughness: f64, metallic: f64) -> PrimMaterial {
    PrimMaterial {
        color,
        roughness,
        metallic,
        emissive: [0.0; 3],
    }
}

fn collider(node: &str, shape: &str, params: Vec<(&str, JsonValue)>, t: [f64; 3]) -> ColliderDoc {
    ColliderDoc {
        node: Some(node.to_string()),
        shape: shape.to_string(),
        params: params
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
        transform: tf_full(t, [0.0, 0.0, 0.0, 1.0], [1.0; 3]),
    }
}

fn lamp() -> PrimsAsset {
    let base = node(
        "base",
        "cylinder",
        vec![("radius", num(0.4)), ("halfHeight", num(0.05))],
        None,
        Some("brass"),
        vec![
            node(
                "pole",
                "cylinder",
                vec![("radius", num(0.04)), ("halfHeight", num(0.6))],
                tf([0.0, 0.65, 0.0]),
                None,
                vec![],
            ),
            sphere("bulb", 0.22, [0.0, 1.45, 0.0], Some("glass")),
            node(
                "shade",
                "cone",
                vec![("radius", num(0.45)), ("halfHeight", num(0.25))],
                tf([0.0, 1.75, 0.0]),
                Some("shade"),
                vec![],
            ),
            node(
                "ring",
                "torus",
                vec![("majorRadius", num(0.3)), ("minorRadius", num(0.03))],
                tf([0.0, 1.05, 0.0]),
                Some("brass"),
                vec![],
            ),
        ],
    );
    PrimsAsset {
        id: "lamp".into(),
        name: "Table lamp".into(),
        // Fixed LOD0 counts: cylinders 160 x2, sphere 1520, cone 80, torus 1920 = 3840 > 1500,
        // so the smallest explicit budget is 3840.
        triangle_budget: Some(3840),
        segments: None,
        nodes: vec![base],
        materials: vec![
            ("brass".into(), material([0.80, 0.60, 0.20], 0.35, 0.9)),
            ("glass".into(), material([1.00, 0.95, 0.70], 0.2, 0.0)),
            ("shade".into(), material([0.85, 0.15, 0.15], 0.8, 0.0)),
        ],
        anchors: vec![(
            "top".into(),
            tf_full([0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0; 3]),
        )],
        collision: Some(vec![
            collider(
                "base",
                "cylinderY",
                vec![("halfHeight", num(0.05)), ("radius", num(0.4))],
                [0.0; 3],
            ),
            collider(
                "pole",
                "cylinderY",
                vec![("halfHeight", num(0.6)), ("radius", num(0.04))],
                [0.0, 0.65, 0.0],
            ),
            collider(
                "bulb",
                "ball",
                vec![("radius", num(0.22))],
                [0.0, 1.45, 0.0],
            ),
        ]),
    }
}

fn critter() -> PrimsAsset {
    let eye = |id: &str, x: f64| sphere(id, 0.05, [x, 0.12, 0.28], Some("eye"));
    let limb = |id: &str, x: f64, z: f64| capsule(id, 0.07, 0.1, [x, -0.38, z]);
    let body = node(
        "body",
        "ellipsoid",
        vec![("radii", vec3([0.45, 0.3, 0.3]))],
        None,
        Some("fur"),
        vec![
            eye("eye-l", -0.12),
            eye("eye-r", 0.12),
            limb("leg-fl", -0.2, 0.15),
            limb("leg-fr", 0.2, 0.15),
            limb("leg-bl", -0.2, -0.15),
            limb("leg-br", 0.2, -0.15),
        ],
    );
    PrimsAsset {
        id: "critter".into(),
        name: "Critter".into(),
        // Fixed LOD0 counts: ellipsoid 1520, 2 spheres 1520 each, 4 capsules 1280 each = 9680
        // > 1500, so the smallest explicit budget is 9680.
        triangle_budget: Some(9680),
        segments: None,
        nodes: vec![body],
        materials: vec![
            ("fur".into(), material([0.55, 0.35, 0.20], 0.9, 0.0)),
            ("eye".into(), material([0.05, 0.05, 0.08], 0.3, 0.0)),
        ],
        anchors: vec![
            (
                "tail".into(),
                tf_full([0.0, 0.05, -0.45], [0.0, 0.0, 0.0, 1.0], [1.0; 3]),
            ),
            (
                "head".into(),
                tf_full([0.0, 0.2, 0.3], [0.0, 0.0, 0.0, 1.0], [1.0; 3]),
            ),
        ],
        collision: None,
    }
}

fn crate_asset() -> PrimsAsset {
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let plank = |id: &str, y: f64| {
        node(
            id,
            "box",
            vec![("halfExtents", vec3([0.52, 0.05, 0.08]))],
            tf([0.0, y, 0.0]),
            Some("plank"),
            vec![],
        )
    };
    let slat = PrimNode {
        transform: Some(tf_full([0.0, 0.0, 0.52], [0.0, 0.0, h, h], [1.0; 3])),
        ..node(
            "slat",
            "box",
            vec![("halfExtents", vec3([0.52, 0.05, 0.05]))],
            None,
            Some("plank"),
            vec![],
        )
    };
    let body = node(
        "body",
        "roundBox",
        vec![
            ("halfExtents", vec3([0.5, 0.5, 0.5])),
            ("radius", num(0.05)),
        ],
        None,
        Some("wood"),
        vec![plank("plank-top", 0.5), plank("plank-bottom", -0.5), slat],
    );
    PrimsAsset {
        id: "crate".into(),
        name: "Wooden crate".into(),
        triangle_budget: None,
        segments: None,
        nodes: vec![body],
        materials: vec![
            ("plank".into(), material([0.45, 0.30, 0.15], 0.85, 0.0)),
            ("wood".into(), material([0.65, 0.45, 0.25], 0.8, 0.0)),
        ],
        anchors: vec![],
        collision: None,
    }
}

pub(crate) fn sample_assets_raw() -> Vec<(&'static str, PrimsAsset)> {
    vec![
        ("lamp", lamp()),
        ("critter", critter()),
        ("crate", crate_asset()),
    ]
}

/// The three canonical sample texts: `lamp`, `critter`, `crate`.
pub fn sample_assets() -> Vec<(&'static str, String)> {
    sample_assets_raw()
        .into_iter()
        .map(|(n, a)| (n, to_canonical_json(&a)))
        .collect()
}

const MAT: &str = r#"{"color":[0.5,0.5,0.5],"roughness":0.5,"metallic":0,"emissive":[0,0,0]}"#;

fn nd(id: &str, kind: &str, params: &str, rest: &str) -> String {
    format!(r#"{{"id":"{id}","kind":"{kind}","params":{params}{rest}}}"#)
}

fn good() -> String {
    nd("a", "sphere", r#"{"radius":1}"#, r#","material":"m""#)
}

fn doc(extra: &str, nodes: &str) -> String {
    format!(
        r#"{{"format":"raa.prims","version":0,"id":"t","name":"t",{extra}"nodes":{nodes},"materials":{{"m":{MAT}}},"anchors":{{}}}}"#
    )
}

fn one(node: &str) -> String {
    doc("", &format!("[{node}]"))
}

fn sphere_with(rest: &str) -> String {
    one(&nd("a", "sphere", r#"{"radius":1}"#, rest))
}

fn sample(name: &str) -> PrimsAsset {
    sample_assets_raw()
        .into_iter()
        .find(|(n, _)| *n == name)
        .map(|(_, a)| a)
        .unwrap_or_else(lamp)
}

/// The refusal table: label, text, expected S5 code.
pub fn invalid_cases() -> Vec<(&'static str, String, &'static str)> {
    let mut c: Vec<(&'static str, String, &'static str)> = Vec::new();
    let nested = "[".repeat(40) + &"]".repeat(40);

    c.push(("json-truncated-object", "{".into(), "json.syntax"));
    c.push((
        "json-trailing-comma",
        doc("", "[],").replace("[],", "[,]"),
        "json.syntax",
    ));
    c.push((
        "json-duplicate-top-key",
        r#"{"format":"raa.prims","format":"raa.prims"}"#.into(),
        "json.duplicate_key",
    ));
    c.push((
        "json-too-deep",
        format!(r#"{{"format":{nested}}}"#),
        "json.depth",
    ));
    c.push((
        "format-unknown",
        doc("", "[]").replace("raa.prims", "other.thing"),
        "format.unknown",
    ));
    c.push((
        "format-version-1",
        doc("", "[]").replace(r#""version":0"#, r#""version":1"#),
        "format.version",
    ));
    c.push((
        "schema-missing-nodes",
        r#"{"format":"raa.prims","version":0,"id":"t","name":"t","materials":{},"anchors":{}}"#
            .into(),
        "schema.missing_field",
    ));
    c.push((
        "schema-missing-node-kind",
        one(r#"{"id":"a","params":{"radius":1}}"#),
        "schema.missing_field",
    ));
    c.push((
        "schema-unknown-node-key",
        sphere_with(r#","colour":"m""#),
        "schema.unknown_field",
    ));
    c.push((
        "schema-unknown-top-key",
        doc(r#""extra":1,"#, "[]"),
        "schema.unknown_field",
    ));
    c.push((
        "schema-id-not-string",
        one(r#"{"id":5,"kind":"sphere","params":{"radius":1}}"#),
        "schema.type",
    ));
    c.push((
        "value-radius-overflow-1e999",
        one(&nd("a", "sphere", r#"{"radius":1e999}"#, "")),
        "value.not_finite",
    ));
    c.push((
        "value-radius-zero",
        one(&nd("a", "sphere", r#"{"radius":0}"#, "")),
        "value.range",
    ));
    c.push((
        "value-roundbox-radius-too-big",
        one(&nd(
            "a",
            "roundBox",
            r#"{"halfExtents":[1,1,1],"radius":2}"#,
            "",
        )),
        "value.range",
    ));
    c.push((
        "value-torus-minor-ge-major",
        one(&nd(
            "a",
            "torus",
            r#"{"majorRadius":1,"minorRadius":1}"#,
            "",
        )),
        "value.range",
    ));
    c.push((
        "value-quaternion-half-norm",
        sphere_with(r#","transform":{"t":[0,0,0],"r":[0,0,0,0.5],"s":[1,1,1]}"#),
        "value.range",
    ));
    c.push((
        "value-scale-component-zero",
        sphere_with(r#","transform":{"t":[0,0,0],"r":[0,0,0,1],"s":[1,0,1]}"#),
        "value.range",
    ));
    c.push((
        "value-triangle-budget-zero",
        doc(r#""triangleBudget":0,"#, &format!("[{}]", good())),
        "value.range",
    ));
    c.push((
        "id-invalid-space",
        one(&nd("a b", "sphere", r#"{"radius":1}"#, "")),
        "id.invalid",
    ));
    c.push((
        "id-duplicate-roots",
        doc("", &format!("[{},{}]", good(), good())),
        "id.duplicate",
    ));
    c.push((
        "material-unknown-reference",
        sphere_with(r#","material":"nope""#),
        "material.unknown",
    ));
    c.push((
        "kind-pyramid",
        one(&nd("a", "pyramid", r#"{"radius":1}"#, "")),
        "kind.unknown",
    ));
    c.push((
        "kind-params-missing",
        one(&nd("a", "capsule", r#"{"radius":1}"#, "")),
        "kind.params",
    ));
    c.push((
        "kind-params-extra",
        one(&nd("a", "sphere", r#"{"radius":1,"height":2}"#, "")),
        "kind.params",
    ));
    c.push((
        "kind-params-wrong-shape",
        one(&nd("a", "box", r#"{"halfExtents":1}"#, "")),
        "kind.params",
    ));

    let mut many = sample("critter");
    many.nodes = (0..65)
        .map(|i| sphere(&format!("n{i}"), 0.1, [0.0; 3], None))
        .collect();
    c.push(("cap-nodes-65-roots", to_canonical_json(&many), "cap.nodes"));

    let mut deep = sample("critter");
    let mut chain = sphere("l9", 0.1, [0.0; 3], None);
    for level in (0..9).rev() {
        chain = node(
            &format!("l{level}"),
            "sphere",
            vec![("radius", num(0.1))],
            None,
            None,
            vec![chain],
        );
    }
    deep.nodes = vec![chain];
    c.push(("cap-depth-level-9", to_canonical_json(&deep), "cap.depth"));

    let mut budget = sample("crate");
    budget.triangle_budget = Some(10);
    c.push(("cap-budget-10", to_canonical_json(&budget), "cap.budget"));

    let mut lamp_no_col = sample("lamp");
    lamp_no_col.collision = None;
    c.push((
        "collision-lamp-torus-no-override",
        to_canonical_json(&lamp_no_col),
        "collision.torus_needs_override",
    ));

    let mut shape = sample("lamp");
    if let Some(cols) = shape.collision.as_mut() {
        cols[0].shape = "pyramidY".into();
    }
    c.push((
        "collision-unknown-shape",
        to_canonical_json(&shape),
        "collision.shape",
    ));

    let mut bad_params = sample("lamp");
    if let Some(cols) = bad_params.collision.as_mut() {
        cols[2].params = vec![("radius".into(), num(-1.0))];
    }
    c.push((
        "collision-bad-params",
        to_canonical_json(&bad_params),
        "collision.shape",
    ));

    let mut bad_ref = sample("lamp");
    if let Some(cols) = bad_ref.collision.as_mut() {
        cols[1].node = Some("ghost".into());
    }
    c.push((
        "collision-unknown-node-reference",
        to_canonical_json(&bad_ref),
        "collision.shape",
    ));

    let mut sub = sample("crate");
    let mut tool = node(
        "hole",
        "box",
        vec![("halfExtents", vec3([0.2, 0.2, 0.6]))],
        None,
        None,
        vec![],
    );
    tool.op = NodeOp::Subtract;
    sub.nodes[0].children.push(tool);
    c.push((
        "boolean-crate-subtract-no-resolver",
        to_canonical_json(&sub),
        "boolean.unsupported",
    ));
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prims::DEFAULT_TRIANGLE_BUDGET;
    use raa_geom::{GeometryRegistry, ParamValue, Params, Quality};
    use std::collections::BTreeSet;

    fn params_of(n: &PrimNode) -> Params {
        let mut p = Params::new();
        for (k, v) in &n.params {
            match v {
                JsonValue::Number(x) => p.set(k, ParamValue::Scalar(*x as f32)),
                JsonValue::Array(a) if a.len() == 3 => {
                    let f = |i: usize| match &a[i] {
                        JsonValue::Number(x) => *x as f32,
                        _ => 0.0,
                    };
                    p.set(k, ParamValue::Vec3([f(0), f(1), f(2)]));
                }
                _ => panic!("bad param"),
            }
        }
        p
    }

    fn sum(reg: &GeometryRegistry, nodes: &[PrimNode]) -> u32 {
        nodes
            .iter()
            .map(|n| {
                let prov = reg.get(&n.kind).expect("kind registered");
                prov.triangle_count(&params_of(n), Quality::LOD0)
                    .expect("count")
                    + sum(reg, &n.children)
            })
            .sum()
    }

    #[test]
    fn lod0_triangle_sums_fit_budgets() {
        let mut reg = GeometryRegistry::new();
        raa_prims::register_all(&mut reg).expect("register");
        for (name, a) in sample_assets_raw() {
            let total = sum(&reg, &a.nodes);
            let budget = a.triangle_budget.unwrap_or(DEFAULT_TRIANGLE_BUDGET);
            println!("sample {name}: LOD0 triangles {total}, budget {budget}");
            assert!(total <= budget, "{name}: {total} > {budget}");
        }
    }

    #[test]
    fn sample_order_and_determinism() {
        let a = sample_assets();
        assert_eq!(
            a.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
            ["lamp", "critter", "crate"]
        );
        assert_eq!(a, sample_assets());
        assert!(a.iter().all(|(_, t)| t.ends_with("}\n")));
    }

    #[test]
    fn every_s5_code_is_covered_and_labels_unique() {
        let cases = invalid_cases();
        let codes: BTreeSet<&str> = cases.iter().map(|c| c.2).collect();
        for code in [
            "json.syntax",
            "json.duplicate_key",
            "json.depth",
            "format.unknown",
            "format.version",
            "schema.missing_field",
            "schema.unknown_field",
            "schema.type",
            "value.not_finite",
            "value.range",
            "id.invalid",
            "id.duplicate",
            "material.unknown",
            "kind.unknown",
            "kind.params",
            "cap.nodes",
            "cap.depth",
            "cap.budget",
            "collision.torus_needs_override",
            "collision.shape",
            "boolean.unsupported",
        ] {
            assert!(codes.contains(code), "no case for {code}");
        }
        let labels: BTreeSet<&str> = cases.iter().map(|c| c.0).collect();
        assert_eq!(labels.len(), cases.len());
        assert!(cases.iter().all(|c| {
            !c.0.is_empty()
                && c.0
                    .chars()
                    .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-')
        }));
        assert_eq!(cases, invalid_cases());
    }

    #[test]
    fn samples_have_unit_quaternions() {
        for (_, a) in sample_assets_raw() {
            fn walk(n: &PrimNode) {
                if let Some(t) = &n.transform {
                    let norm: f64 = t.r.iter().map(|x| x * x).sum::<f64>().sqrt();
                    assert!((norm - 1.0).abs() < 1e-9);
                }
                n.children.iter().for_each(walk);
            }
            a.nodes.iter().for_each(walk);
        }
    }
}
