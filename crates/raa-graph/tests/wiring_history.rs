//! Device-free checks of the executor-owned history wiring (`history_refusals`, `history_resets`).
#[allow(dead_code)]
#[path = "../src/exec_ext/wiring/plan_fns.rs"]
mod plan_fns;
use raa_graph::Catalog;
use raa_graph::plan::{
    GraphPlan, PlanHistory, PlanNode, PlanPortRef, PlanResource, plan_from_json,
};

/// Full GI with one extra link inserted at the front of `links`.
fn full_gi_with(link: &str) -> GraphPlan {
    let text = include_str!("../graphs/full-gi.json").replacen(
        "\"links\": [",
        &format!("\"links\": [{link},"),
        1,
    );
    plan_from_json(&text).unwrap()
}

#[test]
fn taa_own_history_is_accepted() {
    let plan = full_gi_with(r#"{"from": "taa.history", "to": "taa.history", "history": true}"#);
    assert_eq!(plan_fns::history_refusals(&plan, &Catalog::build()), vec![]);
    assert_eq!(
        plan_fns::history_resets(Some(&plan), &plan),
        Vec::<String>::new()
    );
}

#[test]
fn gi_history_output_is_refused() {
    let plan =
        full_gi_with(r#"{"from": "giHistory.history", "to": "taa.history", "history": true}"#);
    assert_eq!(
        plan_fns::history_refusals(&plan, &Catalog::build()),
        vec![("taa".to_string(), "history".to_string())]
    );
}

#[test]
fn plain_link_into_history_input_is_refused() {
    let plan = full_gi_with(r#"{"from": "fog.color", "to": "taa.history"}"#);
    assert_eq!(
        plan_fns::history_refusals(&plan, &Catalog::build()),
        vec![("taa".to_string(), "history".to_string())]
    );
}
