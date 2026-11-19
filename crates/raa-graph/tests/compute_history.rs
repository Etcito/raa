//! Compute history link planning.

#[allow(dead_code)]
#[path = "../src/exec_ext/compute/plan.rs"]
mod plan;
#[allow(dead_code)]
#[path = "../src/exec_ext/wiring/plan_fns.rs"]
mod plan_fns;

use plan::history_inputs;
use raa_graph::catalog::{Catalog, GraphFormat, ParamValue, PortType, SizeSpec};
use raa_graph::plan::{
    GraphPlan, InputSource, PlanHistory, PlanNode, PlanPortRef, PlanResource, plan_from_json,
};

#[test]
fn compute_history_input_is_planned_as_a_history_pair() {
    let text = r#"{"format":"raa.render-graph","version":1,"name":"history","nodes":[{"id":"prepass","type":"pass.prepass"},{"id":"forward","type":"pass.forward"},{"id":"tonemap","type":"pass.tonemap"},{"id":"out","type":"output.display"},{"id":"acc","type":"compute.wgsl","inputs":[{"key":"src","type":"image2d"},{"key":"history","type":"image2d"}],"outputs":[{"key":"dst","type":"image2d","format":"rgba16float"}],"math":null,"wgsl":"fn rg_main(p: vec2<u32>) {}"},{"id":"show","type":"view.image"}],"links":[{"from":"prepass.normal","to":"acc.src"},{"from":"acc.dst","to":"acc.history","history":true},{"from":"acc.dst","to":"show.src"},{"from":"show.color","to":"tonemap.color"},{"from":"tonemap.color","to":"out.color"}]}"#;
    let graph: GraphPlan = plan_from_json(text).expect("self history graph is valid");
    assert!(
        graph
            .histories
            .iter()
            .any(|history| history.consumer.node == "acc" && history.consumer.port == "history")
    );
    assert_eq!(
        history_inputs(&graph),
        vec![("acc".into(), "history".into(), "acc".into(), "dst".into())]
    );
}
