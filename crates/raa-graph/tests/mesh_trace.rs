use raa_graph::catalog::{GraphFormat, ParamValue, SizeSpec};
use raa_graph::plan::InputSource;
use serde_json::{Value, json};

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/geometry-reflections.json")).unwrap()
}

#[test]
fn geometry_reflections_depend_on_ssr_and_own_the_hdr_output() {
    let plan = raa_graph::plan_from_json(&fixture().to_string()).unwrap();
    let mirrors = plan.nodes.iter().find(|node| node.id == "mirrors").unwrap();
    for (key, producer, port) in [
        ("color", "ssr", "color"),
        ("screenColor", "forward", "color"),
        ("screenTrace", "screen", "trace"),
        ("hiz", "hiz", "hiz"),
        ("motion", "prepass", "motion"),
    ] {
        let input = mirrors
            .inputs
            .iter()
            .find(|input| input.port == key)
            .unwrap();
        assert_eq!(
            input.source,
            InputSource::Link {
                node: producer.into(),
                port: port.into(),
                history: false
            }
        );
        let source = plan.nodes.iter().find(|node| node.id == producer).unwrap();
        assert!(source.order_index < mirrors.order_index);
    }
    let color = plan
        .resources
        .iter()
        .find(|resource| resource.producer.node == "mirrors")
        .unwrap();
    assert_eq!(color.producer.port, "color");
    assert_eq!(color.format, Some(GraphFormat::Rgba16Float));
    assert_eq!(color.size, SizeSpec::Scale(1.0));

    let mut missing = fixture();
    missing["links"]
        .as_array_mut()
        .unwrap()
        .retain(|link| link["to"] != "mirrors.screenTrace");
    assert!(raa_graph::plan_from_json(&missing.to_string()).is_err());
}

#[test]
fn rewire_display_to_reference_or_fitted_trace() {
    let mut doc = fixture();
    let nodes = doc["nodes"].as_array_mut().unwrap();
    nodes.push(json!({
        "id": "reference", "type": "mesh.reference",
        "params": { "lightingOutput": "bounce", "lightingDepth": 3, "samples": 2, "seed": 12 }
    }));
    let links = doc["links"].as_array_mut().unwrap();
    links.push(json!({ "from": "forward.color", "to": "reference.color" }));
    links
        .iter_mut()
        .find(|link| link["to"] == "tonemap.color")
        .unwrap()["from"] = json!("reference.color");
    let plan = raa_graph::plan_from_json(&doc.to_string()).unwrap();
    assert!(plan.nodes.iter().any(|node| node.id == "forward"));
    assert!(!plan.nodes.iter().any(|node| node.id == "mirrors"));
    let reference = plan
        .nodes
        .iter()
        .find(|node| node.id == "reference")
        .unwrap();
    assert!(
        reference
            .params
            .contains(&("lightingDepth".into(), ParamValue::Int(3)))
    );

    let mut doc = fixture();
    doc["nodes"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|node| node["id"] == "mirrors")
        .unwrap()["type"] = json!("mesh.fitted-specular");
    doc["links"]
        .as_array_mut()
        .unwrap()
        .retain(|link| link["to"] != "mirrors.color");
    let plan = raa_graph::plan_from_json(&doc.to_string()).unwrap();
    assert!(
        plan.nodes
            .iter()
            .any(|node| node.type_id == "mesh.fitted-specular")
    );
    assert!(!plan.nodes.iter().any(|node| node.id == "ssr"));
}
