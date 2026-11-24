use raa_graph::catalog::{Catalog, PortType};
use raa_graph::format::{parse_graph, write_graph};
use raa_graph::plan::{InputSource, plan_from_json};
use raa_loader::json::JsonValue;

const SPHERICAL: &str = include_str!("../src/frozen/spherical_gi.json");
const FULL_GI: &str = include_str!("../graphs/full-gi.json");

#[test]
fn spherical_lookup_has_typed_fixed_wiring_and_runs_before_forward() {
    let catalog = Catalog::build();
    let spherical = catalog.node("pass.gi.spherical").unwrap();
    let forward = catalog.node("pass.forward").unwrap();
    let output = spherical
        .outputs
        .iter()
        .find(|port| port.key == "lookup")
        .unwrap();
    let input = forward
        .inputs
        .iter()
        .find(|port| port.key == "sphericalLookup")
        .unwrap();
    assert_eq!(output.ty, PortType::Buffer);
    assert_eq!(input.ty, output.ty);
    assert!(!input.generic);

    let plan = plan_from_json(SPHERICAL).unwrap();
    let producer = plan
        .nodes
        .iter()
        .find(|node| node.id == "spherical")
        .unwrap();
    let consumer = plan.nodes.iter().find(|node| node.id == "forward").unwrap();
    assert!(producer.order_index < consumer.order_index);
    assert!(consumer.inputs.iter().any(|input| {
        input.port == "sphericalLookup"
            && matches!(&input.source, InputSource::Link { node, port, .. } if node == "spherical" && port == "lookup")
    }));
    assert_eq!(raa_graph::frozen_plan("spherical_gi"), Some(plan));
}

#[test]
fn spherical_radius_and_private_buffer_are_validated_before_apply() {
    let mut doc = parse_graph(SPHERICAL).unwrap();
    let spherical = doc
        .nodes
        .iter_mut()
        .find(|node| node.id == "spherical")
        .unwrap();
    spherical
        .params
        .push(("clipStart".into(), JsonValue::Number(12.0)));
    let invalid_radius = plan_from_json(&write_graph(&doc)).unwrap_err();
    assert!(invalid_radius.iter().any(|issue| {
        issue.code == "graph.param_range"
            && issue.node.as_deref() == Some("spherical")
            && issue.param.as_deref() == Some("zMax")
    }));

    let mut doc = parse_graph(SPHERICAL).unwrap();
    let lookup = doc
        .links
        .iter_mut()
        .find(|link| link.from.port == "lookup")
        .unwrap();
    lookup.to.port = "lightTiles".into();
    let invalid_link = plan_from_json(&write_graph(&doc)).unwrap_err();
    assert!(invalid_link.iter().any(|issue| {
        issue.code == "graph.fixed_wiring"
            && issue
                .message
                .contains("private to the forward lighting bind group")
    }));
}

#[test]
fn volume_probes_and_restir_keep_the_same_forward_resource_link() {
    for mode in ["probes", "restir"] {
        let mut doc = parse_graph(FULL_GI).unwrap();
        let screen = doc
            .nodes
            .iter_mut()
            .find(|node| node.id == "giScreen")
            .unwrap();
        screen
            .params
            .iter_mut()
            .find(|(key, _)| key == "indirect")
            .unwrap()
            .1 = JsonValue::String(mode.into());
        let plan = plan_from_json(&write_graph(&doc)).unwrap();
        let screen = plan
            .nodes
            .iter()
            .find(|node| node.id == "giScreen")
            .unwrap();
        assert!(
            screen
                .inputs
                .iter()
                .any(|input| input.port == "volumeAtlas")
        );
        let forward = plan.nodes.iter().find(|node| node.id == "forward").unwrap();
        assert!(forward.inputs.iter().any(|input| input.port == "gi"));
    }
}
