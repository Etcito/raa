//! S4 canonical text of a prims document.

use super::model::{ColliderDoc, NodeOp, PrimMaterial, PrimNode, PrimsAsset, TransformDoc};
use super::{FORMAT, VERSION};
use crate::json::{JsonValue, write_pretty};

/// Canonical text (S4): `JSON.stringify(value, null, 2)` in canonical key order plus one LF.
pub fn to_canonical_json(a: &PrimsAsset) -> String {
    let mut top: Vec<(String, JsonValue)> = vec![
        key("format", JsonValue::String(FORMAT.to_string())),
        key("version", JsonValue::Number(f64::from(VERSION))),
        key("id", JsonValue::String(a.id.clone())),
        key("name", JsonValue::String(a.name.clone())),
    ];
    if let Some(b) = a.triangle_budget {
        top.push(key("triangleBudget", JsonValue::Number(f64::from(b))));
    }
    top.push(key(
        "nodes",
        JsonValue::Array(a.nodes.iter().map(node).collect()),
    ));

    let mut materials: Vec<&(String, PrimMaterial)> = a.materials.iter().collect();
    materials.sort_by(|x, y| x.0.as_bytes().cmp(y.0.as_bytes()));
    top.push(key(
        "materials",
        JsonValue::Object(
            materials
                .into_iter()
                .map(|(k, m)| (k.clone(), material(m)))
                .collect(),
        ),
    ));

    let mut anchors: Vec<&(String, TransformDoc)> = a.anchors.iter().collect();
    anchors.sort_by(|x, y| x.0.as_bytes().cmp(y.0.as_bytes()));
    top.push(key(
        "anchors",
        JsonValue::Object(
            anchors
                .into_iter()
                .map(|(k, t)| (k.clone(), transform(t)))
                .collect(),
        ),
    ));

    if let Some(colliders) = &a.collision {
        top.push(key(
            "collision",
            JsonValue::Object(vec![key(
                "colliders",
                JsonValue::Array(colliders.iter().map(collider).collect()),
            )]),
        ));
    }

    let mut out = String::new();
    write_pretty(&JsonValue::Object(top), &mut out);
    out.push('\n');
    out
}

fn key(k: &str, v: JsonValue) -> (String, JsonValue) {
    (k.to_string(), v)
}

fn numbers(xs: &[f64]) -> JsonValue {
    JsonValue::Array(xs.iter().map(|&x| JsonValue::Number(x)).collect())
}

fn transform(t: &TransformDoc) -> JsonValue {
    JsonValue::Object(vec![
        key("t", numbers(&t.t)),
        key("r", numbers(&t.r)),
        key("s", numbers(&t.s)),
    ])
}

fn material(m: &PrimMaterial) -> JsonValue {
    JsonValue::Object(vec![
        key("color", numbers(&m.color)),
        key("roughness", JsonValue::Number(m.roughness)),
        key("metallic", JsonValue::Number(m.metallic)),
        key("emissive", numbers(&m.emissive)),
    ])
}

fn node(n: &PrimNode) -> JsonValue {
    let mut o = vec![
        key("id", JsonValue::String(n.id.clone())),
        key("kind", JsonValue::String(n.kind.clone())),
        key("params", JsonValue::Object(n.params.clone())),
    ];
    if let Some(t) = &n.transform {
        o.push(key("transform", transform(t)));
    }
    if let Some(m) = &n.material {
        o.push(key("material", JsonValue::String(m.clone())));
    }
    if n.op != NodeOp::Union {
        o.push(key("op", JsonValue::String(n.op.name().to_string())));
    }
    if !n.children.is_empty() {
        o.push(key(
            "children",
            JsonValue::Array(n.children.iter().map(node).collect()),
        ));
    }
    JsonValue::Object(o)
}

fn collider(c: &ColliderDoc) -> JsonValue {
    let mut o = Vec::with_capacity(4);
    if let Some(n) = &c.node {
        o.push(key("node", JsonValue::String(n.clone())));
    }
    o.push(key("shape", JsonValue::String(c.shape.clone())));
    o.push(key("params", JsonValue::Object(c.params.clone())));
    o.push(key("transform", transform(&c.transform)));
    JsonValue::Object(o)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> TransformDoc {
        TransformDoc {
            t: [0.0; 3],
            r: [0.0, 0.0, 0.0, 1.0],
            s: [1.0; 3],
        }
    }

    fn mat(c: f64) -> PrimMaterial {
        PrimMaterial {
            color: [c, c, c],
            roughness: 0.5,
            metallic: 0.0,
            emissive: [0.0; 3],
        }
    }

    fn leaf(id: &str, kind: &str, params: Vec<(String, JsonValue)>) -> PrimNode {
        PrimNode {
            id: id.to_string(),
            kind: kind.to_string(),
            params,
            transform: None,
            material: None,
            op: NodeOp::Union,
            children: Vec::new(),
        }
    }

    fn minimal() -> PrimsAsset {
        PrimsAsset {
            id: "a".to_string(),
            name: "A".to_string(),
            triangle_budget: None,
            nodes: vec![leaf(
                "n",
                "sphere",
                vec![key("radius", JsonValue::Number(1.0))],
            )],
            materials: Vec::new(),
            anchors: Vec::new(),
            collision: None,
        }
    }

    #[test]
    fn hand_built_asset_gives_exact_text() {
        let mut tool = leaf(
            "hole",
            "cylinder",
            vec![
                key("radius", JsonValue::Number(0.25)),
                key("halfHeight", JsonValue::Number(1.0)),
            ],
        );
        tool.op = NodeOp::Subtract;
        let mut root = leaf(
            "body",
            "box",
            vec![key("halfExtents", numbers(&[0.5, 0.5, 0.5]))],
        );
        root.transform = Some(TransformDoc {
            t: [0.0, 1.5, 0.0],
            r: [0.0, 0.0, 0.0, 1.0],
            s: [1.0; 3],
        });
        root.material = Some("red".to_string());
        root.children = vec![tool];
        let a = PrimsAsset {
            id: "thing".to_string(),
            name: "Thing \"1\"".to_string(),
            triangle_budget: Some(800),
            nodes: vec![root],
            materials: vec![(
                "red".to_string(),
                PrimMaterial {
                    color: [1.0, 0.0, 0.0],
                    roughness: 0.6,
                    metallic: 0.0,
                    emissive: [0.0; 3],
                },
            )],
            anchors: vec![("top".to_string(), identity())],
            collision: Some(vec![ColliderDoc {
                node: Some("body".to_string()),
                shape: "cuboid".to_string(),
                params: vec![key("halfExtents", numbers(&[0.5, 0.5, 0.5]))],
                transform: identity(),
            }]),
        };
        let expected = r#"{
  "format": "raa.prims",
  "version": 0,
  "id": "thing",
  "name": "Thing \"1\"",
  "triangleBudget": 800,
  "nodes": [
    {
      "id": "body",
      "kind": "box",
      "params": {
        "halfExtents": [
          0.5,
          0.5,
          0.5
        ]
      },
      "transform": {
        "t": [
          0,
          1.5,
          0
        ],
        "r": [
          0,
          0,
          0,
          1
        ],
        "s": [
          1,
          1,
          1
        ]
      },
      "material": "red",
      "children": [
        {
          "id": "hole",
          "kind": "cylinder",
          "params": {
            "radius": 0.25,
            "halfHeight": 1
          },
          "op": "subtract"
        }
      ]
    }
  ],
  "materials": {
    "red": {
      "color": [
        1,
        0,
        0
      ],
      "roughness": 0.6,
      "metallic": 0,
      "emissive": [
        0,
        0,
        0
      ]
    }
  },
  "anchors": {
    "top": {
      "t": [
        0,
        0,
        0
      ],
      "r": [
        0,
        0,
        0,
        1
      ],
      "s": [
        1,
        1,
        1
      ]
    }
  },
  "collision": {
    "colliders": [
      {
        "node": "body",
        "shape": "cuboid",
        "params": {
          "halfExtents": [
            0.5,
            0.5,
            0.5
          ]
        },
        "transform": {
          "t": [
            0,
            0,
            0
          ],
          "r": [
            0,
            0,
            0,
            1
          ],
          "s": [
            1,
            1,
            1
          ]
        }
      }
    ]
  }
}
"#;
        assert_eq!(to_canonical_json(&a), expected);
    }

    #[test]
    fn materials_and_anchors_are_sorted_bytewise() {
        let mut a = minimal();
        a.materials = vec![
            ("b".into(), mat(0.2)),
            ("B".into(), mat(0.1)),
            ("a".into(), mat(0.3)),
        ];
        a.anchors = vec![
            ("z".into(), identity()),
            ("_x".into(), identity()),
            ("Z".into(), identity()),
        ];
        let text = to_canonical_json(&a);
        let pos = |s: &str| text.find(s).unwrap();
        assert!(pos("\"B\": {") < pos("\"a\": {") && pos("\"a\": {") < pos("\"b\": {"));
        assert!(pos("\"Z\": {") < pos("\"_x\": {") && pos("\"_x\": {") < pos("\"z\": {"));
        // The input is untouched.
        assert_eq!(a.materials[0].0, "b");
    }

    #[test]
    fn defaults_are_omitted() {
        let text = to_canonical_json(&minimal());
        let expected = "{\n  \"format\": \"raa.prims\",\n  \"version\": 0,\n  \"id\": \"a\",\n  \"name\": \"A\",\n  \"nodes\": [\n    {\n      \"id\": \"n\",\n      \"kind\": \"sphere\",\n      \"params\": {\n        \"radius\": 1\n      }\n    }\n  ],\n  \"materials\": {},\n  \"anchors\": {}\n}\n";
        assert_eq!(text, expected);
        for absent in [
            "triangleBudget",
            "transform",
            "\"op\"",
            "children",
            "collision",
            "material\"",
        ] {
            assert!(!text.contains(absent), "{absent} should be omitted");
        }
    }

    #[test]
    fn numbers_follow_js_rules() {
        let mut a = minimal();
        a.nodes[0].params = vec![
            key("radius", JsonValue::Number(-0.0)),
            key("tiny", JsonValue::Number(1e-7)),
        ];
        let text = to_canonical_json(&a);
        assert!(text.contains("\"radius\": 0,\n"));
        assert!(text.contains("\"tiny\": 1e-7\n"));
    }

    #[test]
    fn output_is_deterministic() {
        let a = minimal();
        assert_eq!(to_canonical_json(&a), to_canonical_json(&a));
    }
}
