//! Parse-stage tests of the `raa.render-graph` document format (S50).

use raa_graph::catalog::{GraphFormat, PortType, SizeSpec};
use raa_graph::error::{Issue, Severity, Stage};
use raa_graph::format::{
    ComputeDecl, ComputePortDecl, GraphDoc, LinkDoc, NodeDoc, PortRef, graph_from_json_value,
    graph_to_json_value, parse_graph, write_graph,
};
use raa_loader::json::JsonValue;

fn refused(text: &str) -> Vec<Issue> {
    parse_graph(text).expect_err("document must be refused")
}

fn codes(issues: &[Issue]) -> Vec<&'static str> {
    issues.iter().map(|issue| issue.code).collect()
}

#[test]
fn invalid_json_is_graph_json_with_its_code_and_position() {
    let issues = refused("{\"format\": \"raa.render-graph\",");
    assert_eq!(codes(&issues), ["graph.json"]);
    assert!(
        issues[0].message.contains("json.syntax"),
        "message: {}",
        issues[0].message
    );
    assert!(
        issues[0].message.contains('@'),
        "message: {}",
        issues[0].message
    );
    assert_eq!(
        (issues[0].stage, issues[0].severity),
        (Stage::Parse, Severity::Error)
    );
}

#[test]
fn oversize_text_and_name_are_graph_too_large() {
    let issues = refused(&" ".repeat(1_048_577));
    assert_eq!(codes(&issues), ["graph.too_large"]);
    assert!(issues[0].message.contains("1048576"));

    let name = "n".repeat(65);
    let text = format!(
        "{{\"format\":\"raa.render-graph\",\"version\":1,\"name\":\"{name}\",\"nodes\":[],\"links\":[]}}"
    );
    let issues = refused(&text);
    assert_eq!(codes(&issues), ["graph.too_large"]);
    assert!(issues[0].message.contains("/name"));
}

#[test]
fn wrong_or_missing_format_is_graph_format() {
    let wrong = refused(r#"{"format":"raa.other","version":1,"name":"x","nodes":[],"links":[]}"#);
    assert_eq!(codes(&wrong), ["graph.format"]);
    let missing = refused(r#"{"version":1,"name":"x","nodes":[],"links":[]}"#);
    assert_eq!(codes(&missing), ["graph.format"]);
}

#[test]
fn wrong_version_is_graph_version() {
    let issues =
        refused(r#"{"format":"raa.render-graph","version":2,"name":"x","nodes":[],"links":[]}"#);
    assert_eq!(codes(&issues), ["graph.version"]);
    assert!(issues[0].message.contains("version"));
}

#[test]
fn unknown_node_key_is_graph_schema_with_its_pointer() {
    let issues = refused(
        r#"{"format":"raa.render-graph","version":1,"name":"x","nodes":[{"id":"a","type":"pass.forward","foo":1}],"links":[]}"#,
    );
    assert_eq!(codes(&issues), ["graph.schema"]);
    assert!(
        issues[0].message.contains("/nodes/0/foo"),
        "message: {}",
        issues[0].message
    );
}

#[test]
fn two_schema_errors_are_both_reported() {
    let issues = refused(
        r#"{"format":"raa.render-graph","version":1,"name":"x","nodes":[{"id":"a","type":"pass.forward","extra":true}],"links":[{"from":"a","to":"b.c"}]}"#,
    );
    assert_eq!(codes(&issues), ["graph.schema", "graph.schema"]);
    assert!(
        issues[0].message.contains("/nodes/0/extra"),
        "message: {}",
        issues[0].message
    );
    assert!(
        issues[1].message.contains("/links/0/from"),
        "message: {}",
        issues[1].message
    );
}

#[test]
fn valid_document_returns_the_expected_fields() {
    let text = r#"{
      "format": "raa.render-graph",
      "version": 1,
      "name": "Valid",
      "nodes": [
        {"id": "a", "type": "pass.forward", "v": 2, "params": {"msaa": 4, "label": "main"}},
        {"id": "blur", "type": "compute.wgsl", "params": {"gain": 0.5},
         "inputs": [{"key": "src", "type": "image2d", "format": "rgba16float"}],
         "outputs": [{"key": "dst", "type": "image2d", "format": "rgba16float",
                      "size": {"width": 640, "height": 360}}],
         "math": {"nodes": []}, "wgsl": "fn rg_main() {}"}
      ],
      "links": [
        {"from": "a.color", "to": "blur.src"},
        {"from": "blur.dst", "to": "a.color", "history": true},
        {"from": "a.color.x", "to": "blur.src.y"}
      ],
      "editor": {"positions": {"a": [1, 2]}}
    }"#;
    let doc = parse_graph(text).expect("valid document");
    assert_eq!(doc.name, "Valid");
    assert_eq!(
        doc.nodes[0],
        NodeDoc {
            id: "a".to_string(),
            type_id: "pass.forward".to_string(),
            version: Some(2),
            params: vec![
                ("msaa".to_string(), JsonValue::Number(4.0)),
                ("label".to_string(), JsonValue::String("main".to_string())),
            ],
            compute: None,
        }
    );
    assert_eq!(
        doc.nodes[1],
        NodeDoc {
            id: "blur".to_string(),
            type_id: "compute.wgsl".to_string(),
            version: None,
            params: vec![("gain".to_string(), JsonValue::Number(0.5))],
            compute: Some(ComputeDecl {
                inputs: vec![ComputePortDecl {
                    key: "src".to_string(),
                    ty: PortType::Image2d,
                    format: Some(GraphFormat::Rgba16Float),
                    size: None,
                    extent: None,
                }],
                outputs: vec![ComputePortDecl {
                    key: "dst".to_string(),
                    ty: PortType::Image2d,
                    format: Some(GraphFormat::Rgba16Float),
                    size: Some(SizeSpec::Fixed {
                        width: 640,
                        height: 360,
                    }),
                    extent: None,
                }],
                math: JsonValue::Object(vec![("nodes".to_string(), JsonValue::Array(Vec::new()),)]),
                wgsl: "fn rg_main() {}".to_string(),
            }),
        }
    );
    assert_eq!(
        doc.links[0],
        LinkDoc {
            from: PortRef {
                node: "a".to_string(),
                port: "color".to_string(),
            },
            to: PortRef {
                node: "blur".to_string(),
                port: "src".to_string(),
            },
            history: false,
        }
    );
    assert!(doc.links[1].history);
    assert_eq!(doc.links[2].from.port, "color.x");
    assert_eq!(doc.links[2].to.port, "src.y");
    assert_eq!(
        doc.editor,
        Some(JsonValue::Object(vec![(
            "positions".to_string(),
            JsonValue::Object(vec![(
                "a".to_string(),
                JsonValue::Array(vec![JsonValue::Number(1.0), JsonValue::Number(2.0)]),
            )]),
        )]))
    );
}

#[test]
fn graph_from_json_value_matches_parse_graph() {
    let text = r#"{"format":"raa.render-graph","version":1,"name":"x","nodes":[],"links":[]}"#;
    let value = raa_loader::json::parse(text).expect("json");
    assert_eq!(
        graph_from_json_value(&value),
        Ok(GraphDoc {
            name: "x".to_string(),
            nodes: Vec::new(),
            links: Vec::new(),
            editor: None,
        })
    );
}

const CANONICAL: &str = r#"{
  "format": "raa.render-graph",
  "version": 1,
  "name": "Canonical",
  "nodes": [
    {
      "id": "a",
      "type": "pass.forward",
      "v": 2,
      "params": {
        "msaa": 4,
        "label": "main",
        "enabled": true,
        "radius": 0.25
      }
    },
    {
      "id": "blur",
      "type": "compute.wgsl",
      "params": {
        "gain": 0.5,
        "count": 3
      },
      "inputs": [
        {
          "key": "src",
          "type": "image2d",
          "format": "rgba16float"
        }
      ],
      "outputs": [
        {
          "key": "dst",
          "type": "image2d",
          "format": "rgba16float",
          "size": {
            "width": 640,
            "height": 360
          }
        }
      ],
      "math": {
        "nodes": [],
        "note": "stored as given"
      },
      "wgsl": "fn rg_main() {}"
    }
  ],
  "links": [
    {
      "from": "a.color",
      "to": "blur.src",
      "history": true
    },
    {
      "from": "blur.dst",
      "to": "a.color"
    }
  ],
  "editor": {
    "positions": {
      "a": [
        1,
        2
      ]
    },
    "comments": []
  }
}
"#;

#[test]
fn canonical_document_round_trips_byte_identical() {
    let doc = parse_graph(CANONICAL).expect("canonical document");
    assert_eq!(write_graph(&doc), CANONICAL);
    let JsonValue::Object(fields) = graph_to_json_value(&doc) else {
        panic!("graph_to_json_value must return an object");
    };
    let keys: Vec<&str> = fields.iter().map(|(key, _)| key.as_str()).collect();
    assert_eq!(
        keys,
        ["format", "version", "name", "nodes", "links", "editor"]
    );
}

const NON_CANONICAL: &str = r#"
{
    "links":   [
        { "to": "blur.src", "history": false, "from": "a.color" },
        { "to": "a.color",   "from": "blur.dst" }
    ],
    "name": "Loose",
    "nodes": [
        { "type": "compute.wgsl", "wgsl": "fn rg_main() {}", "math": {},
          "outputs": [], "inputs": [], "params": { "gain": 2 }, "id": "blur" },
        { "type": "pass.forward", "id": "a" }
    ],
    "version": 1,
    "format": "raa.render-graph"
}
"#;

#[test]
fn non_canonical_text_is_canonicalised_idempotently() {
    let once = write_graph(&parse_graph(NON_CANONICAL).expect("non-canonical document"));
    assert_ne!(once, NON_CANONICAL);
    assert!(
        once.starts_with(
            "{\n  \"format\": \"raa.render-graph\",\n  \"version\": 1,\n  \"name\": \"Loose\",\n  \"nodes\": [\n"
        ),
        "canonical text: {once}"
    );
    assert!(!once.contains("\"history\""), "canonical text: {once}");
    assert!(once.ends_with("}\n"), "canonical text: {once}");
    assert_eq!(
        write_graph(&parse_graph(&once).expect("canonical text parses")),
        once
    );
}

const SIMPLE_FORWARD: &str = r#"{
  "format": "raa.render-graph",
  "version": 1,
  "name": "Simple forward",
  "nodes": [
    {
      "id": "shadowCsm",
      "type": "pass.shadow.csm"
    },
    {
      "id": "prepass",
      "type": "pass.prepass"
    },
    {
      "id": "lightTiles",
      "type": "pass.light.tiles"
    },
    {
      "id": "forward",
      "type": "pass.forward"
    },
    {
      "id": "skyBackground",
      "type": "pass.sky.background"
    },
    {
      "id": "tonemap",
      "type": "pass.tonemap",
      "params": {
        "mapper": "agx",
        "exposureEv": 0
      }
    },
    {
      "id": "out",
      "type": "output.display"
    }
  ],
  "links": [
    {
      "from": "prepass.depth",
      "to": "lightTiles.depth"
    },
    {
      "from": "shadowCsm.shadowMap",
      "to": "forward.shadowMap"
    },
    {
      "from": "lightTiles.lightTiles",
      "to": "forward.lightTiles"
    },
    {
      "from": "forward.color",
      "to": "tonemap.color"
    },
    {
      "from": "tonemap.color",
      "to": "out.color"
    }
  ]
}
"#;

#[test]
fn s50_simple_forward_example_round_trips() {
    let doc = parse_graph(SIMPLE_FORWARD).expect("simple forward document");
    assert_eq!(write_graph(&doc), SIMPLE_FORWARD);
}
