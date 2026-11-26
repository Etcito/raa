//! m32 render graph, share 1/2 (S54): the pure helpers run natively. The wasm32 runtime
//! (`install`, `frame`) is not reachable here; the issue, result and statistics JSON shapes and
//! the owned-setter order are checked against S52/S54.

#![cfg(not(target_arch = "wasm32"))]

#[path = "../src/api/m32_render_graph.rs"]
mod m32;

use m32::{json_to_issue_entries, owned_setters, result_to_json, stats_to_json};
use raa_graph::Issue;
use raa_graph::error::{Severity, Stage};
use raa_graph::exec::{NodeStats, ResourceStat, SetResult};
use raa_loader::json::JsonValue;

fn object(v: &JsonValue) -> &[(String, JsonValue)] {
    match v {
        JsonValue::Object(entries) => entries,
        other => panic!("expected an object, got {other:?}"),
    }
}

fn keys(v: &JsonValue) -> Vec<&str> {
    object(v).iter().map(|(k, _)| k.as_str()).collect()
}

fn get<'a>(v: &'a JsonValue, key: &str) -> &'a JsonValue {
    object(v)
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, value)| value)
        .unwrap_or_else(|| panic!("missing key {key}"))
}

fn array(v: &JsonValue) -> &[JsonValue] {
    match v {
        JsonValue::Array(items) => items,
        other => panic!("expected an array, got {other:?}"),
    }
}

fn string(v: &JsonValue) -> &str {
    match v {
        JsonValue::String(s) => s,
        other => panic!("expected a string, got {other:?}"),
    }
}

fn number(v: &JsonValue) -> f64 {
    match v {
        JsonValue::Number(n) => *n,
        other => panic!("expected a number, got {other:?}"),
    }
}

fn sample_issue() -> Issue {
    Issue {
        code: "graph.param_range",
        severity: Severity::Error,
        stage: Stage::Structure,
        node: Some("ao".to_string()),
        port: None,
        param: Some("radius".to_string()),
        link: None,
        message: "radius is above its maximum".to_string(),
        fix: "lower radius".to_string(),
    }
}

#[test]
fn issue_json_names_node_and_param_only() {
    let json = json_to_issue_entries(&[sample_issue()]);
    let items = array(&json);
    assert_eq!(items.len(), 1);
    let issue = &items[0];
    assert_eq!(
        keys(issue),
        ["code", "severity", "nodeId", "param", "message", "fix"]
    );
    assert_eq!(string(get(issue, "code")), "graph.param_range");
    assert_eq!(string(get(issue, "severity")), "error");
    assert_eq!(string(get(issue, "nodeId")), "ao");
    assert_eq!(string(get(issue, "param")), "radius");
    assert!(!keys(issue).contains(&"port"));
    assert!(!keys(issue).contains(&"link"));
}

#[test]
fn result_json_ok_and_refused() {
    let ok = result_to_json(&SetResult {
        ok: true,
        issues: Vec::new(),
        rebuilt: 3,
    });
    assert_eq!(keys(&ok), ["ok", "issues", "rebuilt"]);
    assert!(matches!(get(&ok, "ok"), JsonValue::Bool(true)));
    assert!(array(get(&ok, "issues")).is_empty());
    assert_eq!(number(get(&ok, "rebuilt")), 3.0);

    let refused = result_to_json(&SetResult {
        ok: false,
        issues: vec![sample_issue()],
        rebuilt: 0,
    });
    assert!(matches!(get(&refused, "ok"), JsonValue::Bool(false)));
    assert_eq!(array(get(&refused, "issues")).len(), 1);
    assert_eq!(number(get(&refused, "rebuilt")), 0.0);
}

#[test]
fn stats_json_null_frame_and_two_nodes() {
    let empty = stats_to_json(None, &[]);
    assert_eq!(keys(&empty), ["frameIndex", "nodes"]);
    assert!(matches!(get(&empty, "frameIndex"), JsonValue::Null));
    assert!(array(get(&empty, "nodes")).is_empty());

    let nodes = [
        NodeStats {
            node: "forward".to_string(),
            type_id: "pass.forward".to_string(),
            ran: true,
            resources: vec![ResourceStat {
                port: "color".to_string(),
                width: 1280,
                height: 720,
                layers: 1,
                mips: 1,
                format: "rgba16float".to_string(),
                bytes: 7_372_800,
            }],
            history_valid: None,
        },
        NodeStats {
            node: "out".to_string(),
            type_id: "output.display".to_string(),
            ran: false,
            resources: Vec::new(),
            history_valid: None,
        },
    ];
    let json = stats_to_json(Some(7), &nodes);
    assert_eq!(number(get(&json, "frameIndex")), 7.0);
    let items = array(get(&json, "nodes"));
    assert_eq!(items.len(), 2);
    assert_eq!(keys(&items[0]), ["id", "type", "ran", "resources"]);
    assert_eq!(string(get(&items[0], "id")), "forward");
    assert_eq!(string(get(&items[0], "type")), "pass.forward");
    assert!(matches!(get(&items[0], "ran"), JsonValue::Bool(true)));
    let resources = array(get(&items[0], "resources"));
    assert_eq!(resources.len(), 1);
    assert_eq!(
        keys(&resources[0]),
        [
            "port", "width", "height", "layers", "mips", "format", "bytes"
        ]
    );
    assert_eq!(string(get(&resources[0], "port")), "color");
    assert_eq!(number(get(&resources[0], "width")), 1280.0);
    assert_eq!(number(get(&resources[0], "height")), 720.0);
    assert_eq!(number(get(&resources[0], "bytes")), 7_372_800.0);
    assert_eq!(string(get(&items[1], "id")), "out");
    assert!(matches!(get(&items[1], "ran"), JsonValue::Bool(false)));
}

#[test]
fn owned_setters_are_in_api_order() {
    let names = owned_setters();
    assert_eq!(names.len(), 21);
    assert_eq!(
        names,
        [
            "setPassEnabled",
            "setPassOrder",
            "setToneMapping",
            "setShadows",
            "setAo",
            "setContactShadows",
            "setBloom",
            "setTransparency",
            "setSsr",
            "setDof",
            "setFog",
            "setSunShafts",
            "setGrading",
            "applyEnvironment",
            "setCel",
            "setVirtualGeometry",
            "setAreaLightShading",
            "setAreaLightShadows",
            "setGlobalIllumination",
            "setGiQuality",
            "setSphericalGi",
        ]
    );
    assert_eq!(names.first().copied(), Some("setPassEnabled"));
    assert_eq!(names.last().copied(), Some("setSphericalGi"));
}
