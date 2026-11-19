//! Compute node pipeline ordering and source mapping.
#[allow(dead_code)]
#[path = "../src/exec_ext/wiring/plan_fns.rs"]
mod plan_fns;

use plan_fns::{history_source, link_order, node_entries};
use raa_graph::Catalog;
use raa_graph::plan::{
    GraphPlan, PlanHistory, PlanNode, PlanPortRef, PlanResource, plan_from_json,
};
use raa_pipeline::{PassInfo, PassRole};

fn calc_document() -> String {
    r#"{"format":"raa.render-graph","version":1,"name":"calc","nodes":[{"id":"prepass","type":"pass.prepass"},{"id":"forward","type":"pass.forward"},{"id":"tonemap","type":"pass.tonemap"},{"id":"out","type":"output.display"},{"id":"calc","type":"compute.wgsl","inputs":[{"key":"src","type":"image2d"}],"outputs":[{"key":"dst","type":"image2d","format":"rgba16float"}],"math":null,"wgsl":"fn rg_main(p: vec2<u32>) {}"},{"id":"view","type":"view.image"}],"links":[{"from":"prepass.normal","to":"calc.src"},{"from":"calc.dst","to":"view.src"},{"from":"view.color","to":"tonemap.color"},{"from":"tonemap.color","to":"out.color"}]}"#.to_owned()
}

fn pass(name: &str, order: u32) -> PassInfo {
    PassInfo {
        name: name.into(),
        role: PassRole::Pass,
        enabled: true,
        order,
        required: false,
    }
}

#[test]
fn compute_entries_follow_their_non_history_sources() {
    let catalog = Catalog::build();
    let calc = plan_from_json(&calc_document()).expect("calc document plans");
    assert!(node_entries(&calc, &catalog).contains(&("calc".into(), "compute#calc".into())));
    assert!(history_source("compute.wgsl", "dst"));

    let stage_doc = r#"{"format":"raa.render-graph","version":1,"name":"stage compute","nodes":[{"id":"pre","type":"pass.prepass"},{"id":"acc","type":"gi.accumulate"},{"id":"den","type":"compute.wgsl","inputs":[{"key":"filter","type":"image2d"}],"outputs":[{"key":"dst","type":"image2d","format":"rgba16float"}],"math":null,"wgsl":"fn rg_main(p: vec2<u32>) {}"},{"id":"res","type":"gi.resolve"},{"id":"forward","type":"pass.forward"},{"id":"tonemap","type":"pass.tonemap"},{"id":"out","type":"output.display"},{"id":"pyr","type":"gi.pyramid"},{"id":"probes","type":"gi.probes"}],"links":[{"from":"pre.depth","to":"pyr.depth"},{"from":"pyr.pyramid","to":"probes.pyramid"},{"from":"pre.normal","to":"probes.normal"},{"from":"probes.probesFiltered","to":"acc.probes"},{"from":"acc.filter","to":"den.filter"},{"from":"den.dst","to":"res.irradiance"},{"from":"res.gi","to":"forward.gi"},{"from":"forward.color","to":"tonemap.color"},{"from":"tonemap.color","to":"out.color"}]}"#;
    let stage = plan_from_json(stage_doc).expect("stage compute plan");
    let entries = node_entries(&stage, &catalog);
    let installed = [
        "prepass",
        "gi.accumulate#acc",
        "gi.resolve#res",
        "compute#den",
        "forward",
        "tonemap",
    ]
    .iter()
    .enumerate()
    .map(|(i, name)| pass(name, i as u32))
    .collect::<Vec<_>>();
    assert_eq!(
        link_order(&stage, &entries, &installed),
        [
            "prepass",
            "gi.accumulate#acc",
            "compute#den",
            "gi.resolve#res",
            "forward",
            "tonemap"
        ]
    );
}
