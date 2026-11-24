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

fn grids_fixture(kind: &str) -> Value {
    let mut doc = fixture();
    doc["nodes"].as_array_mut().unwrap().push(json!({
        "id": "sphere", "type": "pass.gi.spherical"
    }));
    doc["nodes"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|node| node["id"] == "mirrors")
        .unwrap()["type"] = json!(kind);
    if kind == "mesh.fitted-specular" {
        doc["links"]
            .as_array_mut()
            .unwrap()
            .retain(|link| link["to"] != "mirrors.color");
    }
    doc
}

fn grid_source<'a>(plan: &'a raa_graph::GraphPlan, node: &str, port: &str) -> &'a InputSource {
    &plan
        .nodes
        .iter()
        .find(|candidate| candidate.id == node)
        .unwrap()
        .inputs
        .iter()
        .find(|input| input.port == port)
        .unwrap()
        .source
}

fn explicit_grid(node: &str, port: &str) -> InputSource {
    InputSource::Link {
        node: node.into(),
        port: port.into(),
        history: false,
    }
}

#[test]
fn inherited_and_explicit_mesh_grids_keep_order_and_disconnect_independently_of_forward() {
    for kind in ["mesh.reflection", "mesh.fitted-specular"] {
        let mut doc = grids_fixture(kind);
        let plan = raa_graph::plan_from_json(&doc.to_string()).unwrap();
        for (producer, output, input, ty) in [
            (
                "sphere",
                "lookup",
                "sphericalLookup",
                raa_graph::catalog::PortType::Buffer,
            ),
            (
                "volume",
                "volumeAtlas",
                "volumeAtlas",
                raa_graph::catalog::PortType::Image3d,
            ),
        ] {
            assert_eq!(
                grid_source(&plan, "mirrors", input),
                &InputSource::Default {
                    node: producer.into(),
                    port: output.into()
                }
            );
            let consumer = plan.nodes.iter().find(|node| node.id == "mirrors").unwrap();
            let resource = plan
                .resources
                .iter()
                .find(|resource| {
                    resource.producer.node == producer && resource.producer.port == output
                })
                .unwrap();
            assert_eq!(resource.ty, ty);
            assert!(resource.first_use < consumer.order_index);
            assert_eq!(resource.last_use, consumer.order_index);
            assert!(!resource.history);
        }

        let links = doc["links"].as_array_mut().unwrap();
        links.push(json!({"from": "sphere.lookup", "to": "mirrors.sphericalLookup"}));
        let plan = raa_graph::plan_from_json(&doc.to_string()).unwrap();
        assert_eq!(
            grid_source(&plan, "mirrors", "sphericalLookup"),
            &explicit_grid("sphere", "lookup")
        );
        assert_eq!(
            grid_source(&plan, "mirrors", "volumeAtlas"),
            &InputSource::Absent
        );

        let links = doc["links"].as_array_mut().unwrap();
        links.push(json!({"from": "volume.volumeAtlas", "to": "mirrors.volumeAtlas"}));
        let plan = raa_graph::plan_from_json(&doc.to_string()).unwrap();
        assert_eq!(
            grid_source(&plan, "mirrors", "sphericalLookup"),
            &explicit_grid("sphere", "lookup")
        );
        assert_eq!(
            grid_source(&plan, "mirrors", "volumeAtlas"),
            &explicit_grid("volume", "volumeAtlas")
        );

        doc["links"]
            .as_array_mut()
            .unwrap()
            .retain(|link| link["to"] != "mirrors.sphericalLookup");
        let plan = raa_graph::plan_from_json(&doc.to_string()).unwrap();
        assert_eq!(
            grid_source(&plan, "mirrors", "sphericalLookup"),
            &InputSource::Absent
        );
        assert_eq!(
            grid_source(&plan, "mirrors", "volumeAtlas"),
            &explicit_grid("volume", "volumeAtlas")
        );

        doc["nodes"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|node| node["id"] == "mirrors")
            .unwrap()["params"] = json!({ "inheritGi": false });
        doc["links"]
            .as_array_mut()
            .unwrap()
            .retain(|link| link["to"] != "mirrors.volumeAtlas");
        let plan = raa_graph::plan_from_json(&doc.to_string()).unwrap();
        assert_eq!(
            grid_source(&plan, "mirrors", "sphericalLookup"),
            &InputSource::Absent
        );
        assert_eq!(
            grid_source(&plan, "mirrors", "volumeAtlas"),
            &InputSource::Absent
        );
        assert_eq!(
            grid_source(&plan, "forward", "sphericalLookup"),
            &InputSource::Default {
                node: "sphere".into(),
                port: "lookup".into(),
            }
        );

        doc["links"]
            .as_array_mut()
            .unwrap()
            .push(json!({"from": "sphere.lookup", "to": "mirrors.sphericalLookup"}));
        let plan = raa_graph::plan_from_json(&doc.to_string()).unwrap();
        assert_eq!(
            grid_source(&plan, "mirrors", "sphericalLookup"),
            &explicit_grid("sphere", "lookup")
        );
    }
}

#[test]
fn mesh_grid_inputs_reject_unrelated_buffers_wrong_images_and_history() {
    for (from, to, history, expected) in [
        (
            "forward.color",
            "mirrors.volumeAtlas",
            false,
            "graph.link_type",
        ),
        (
            "volume.volumeAtlas",
            "mirrors.volumeAtlas",
            true,
            "graph.history",
        ),
        (
            "screen.trace",
            "mirrors.sphericalLookup",
            false,
            "graph.link_type",
        ),
        (
            "sphere.lookup",
            "mirrors.sphericalLookup",
            true,
            "graph.history",
        ),
    ] {
        let mut doc = grids_fixture("mesh.fitted-specular");
        doc["links"]
            .as_array_mut()
            .unwrap()
            .push(json!({ "from": from, "to": to, "history": history }));
        let issues = raa_graph::plan_from_json(&doc.to_string()).unwrap_err();
        assert!(
            issues.iter().any(|issue| issue.code == expected),
            "{from} -> {to}: {issues:?}"
        );
    }
    let mut doc = grids_fixture("mesh.reflection");
    doc["nodes"]
        .as_array_mut()
        .unwrap()
        .push(json!({ "id": "gtao", "type": "pass.gi.screen" }));
    doc["links"]
        .as_array_mut()
        .unwrap()
        .push(json!({ "from": "gtao.probesRaw", "to": "mirrors.sphericalLookup" }));
    let issues = raa_graph::plan_from_json(&doc.to_string()).unwrap_err();
    assert!(
        issues.iter().any(|issue| issue.code == "graph.fixed_wiring"
            && issue.port.as_deref() == Some("sphericalLookup")),
        "{issues:?}"
    );
}

#[test]
fn reference_has_no_terminal_grid_input() {
    let catalog = raa_graph::Catalog::build();
    let reference = catalog.node("mesh.reference").unwrap();
    assert!(
        reference
            .inputs
            .iter()
            .all(|port| !matches!(port.key, "sphericalLookup" | "volumeAtlas"))
    );
}
