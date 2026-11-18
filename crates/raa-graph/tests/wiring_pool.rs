//! Pooling (S57) and history resets of the wiring extension: `pool_resources` and `history_resets`
//! of `exec_ext/wiring/plan_fns.rs`, compiled into this test crate. Expected values are derived by
//! hand in each test.

#[allow(dead_code)]
#[path = "../src/exec_ext/wiring/plan_fns.rs"]
mod plan_fns;

use plan_fns::{history_resets, pool_resources};
use raa_graph::catalog::{GraphFormat, ParamValue, PortType, SizeSpec};
use raa_graph::format::{GraphDoc, LinkDoc, PortRef, parse_graph, write_graph};
use raa_graph::plan::{
    GraphPlan, PlanHistory, PlanNode, PlanPortRef, PlanResource, plan_from_json,
};
use raa_graph::registry;

/// A colour resource of the given lifetime: an Rgba16Float surface-sized image, one mip, one layer.
fn resource(id: u32, first_use: u32, last_use: u32) -> PlanResource {
    PlanResource {
        id,
        producer: PlanPortRef {
            node: "producer".to_owned(),
            port: "color".to_owned(),
        },
        ty: PortType::Image2d,
        format: Some(GraphFormat::Rgba16Float),
        size: SizeSpec::Scale(1.0),
        mips: 1,
        layers: 1,
        first_use,
        last_use,
        history: false,
    }
}

#[test]
fn resources_share_slots_first_fit_by_id_and_answer_in_input_order() {
    let r1 = resource(1, 0, 2);
    let r2 = resource(2, 2, 3);
    let r3 = resource(3, 3, 5);
    let r4 = PlanResource {
        format: Some(GraphFormat::R8Unorm),
        ..resource(4, 1, 4)
    };
    let r5 = PlanResource {
        history: true,
        ..resource(5, 4, 6)
    };
    // By id: r1 -> 0; r2 -> 1 (it touches r1 at 2); r3 -> 0 (r1 ends at 2, before 3);
    // r4 -> 2 (other format); r5 -> 3 (history, so it skips the free slots 0 and 1).
    let input = [r3, r1, r5, r2, r4];
    assert_eq!(pool_resources(&input), vec![0, 0, 3, 1, 2]);
}

#[test]
fn an_empty_input_gives_an_empty_result() {
    assert!(pool_resources(&[]).is_empty());
}

/// The Full GI starter as a document.
fn full_gi() -> GraphDoc {
    let (_, _, json) = registry::STARTERS
        .iter()
        .find(|(name, _, _)| *name == "Full GI")
        .expect("the Full GI starter exists");
    parse_graph(json).expect("the Full GI starter parses")
}

fn port(text: &str) -> PortRef {
    let (node, port) = text.split_once('.').expect("a port reference has a dot");
    PortRef {
        node: node.to_owned(),
        port: port.to_owned(),
    }
}

fn planned(doc: &GraphDoc) -> GraphPlan {
    plan_from_json(&write_graph(doc)).unwrap_or_else(|issues| panic!("{issues:?}"))
}

/// Full GI planned with the history link `taa.history` -> `taa.history`.
fn full_gi_with_history() -> GraphPlan {
    let mut doc = full_gi();
    doc.links.push(LinkDoc {
        from: port("taa.history"),
        to: port("taa.history"),
        history: true,
    });
    planned(&doc)
}

/// Full GI planned without any history link.
fn full_gi_without_history() -> GraphPlan {
    planned(&full_gi())
}

/// The plan with the TAA node's `blendBase` set to `value`.
fn with_blend_base(mut plan: GraphPlan, value: f64) -> GraphPlan {
    let taa: &mut PlanNode = plan
        .nodes
        .iter_mut()
        .find(|node| node.id == "taa")
        .expect("the Full GI starter has a taa node");
    let param = taa
        .params
        .iter_mut()
        .find(|(key, _)| key == "blendBase")
        .expect("pass.taa has the blendBase parameter");
    param.1 = ParamValue::Number(value);
    plan
}

#[test]
fn a_history_restarts_when_there_is_no_previous_plan() {
    let next = full_gi_with_history();
    assert_eq!(history_resets(None, &next), vec!["taa.history"]);
}

#[test]
fn the_same_plan_twice_keeps_every_history() {
    let previous = full_gi_with_history();
    let next = full_gi_with_history();
    assert!(history_resets(Some(&previous), &next).is_empty());
}

#[test]
fn a_changed_producer_parameter_restarts_its_history() {
    let previous = with_blend_base(full_gi_with_history(), 0.06);
    let next = with_blend_base(full_gi_with_history(), 0.3);
    assert_eq!(history_resets(Some(&previous), &next), vec!["taa.history"]);
}

#[test]
fn a_previous_plan_without_the_history_link_restarts_it() {
    let previous = full_gi_without_history();
    let next = full_gi_with_history();
    assert_eq!(history_resets(Some(&previous), &next), vec!["taa.history"]);
}

#[test]
fn a_removed_history_link_has_nothing_to_restart() {
    let previous = full_gi_with_history();
    let next = full_gi_without_history();
    assert!(history_resets(Some(&previous), &next).is_empty());
}
