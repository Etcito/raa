#[allow(dead_code)]
#[path = "../src/exec_ext/wiring/plan_fns.rs"]
mod plan_fns;

use raa_graph::Catalog;
use raa_graph::plan::{
    GraphPlan, InputSource, PlanHistory, PlanNode, PlanPortRef, PlanResource, plan_from_json,
};
use raa_pipeline::{PassInfo, PassRole};

fn full_gi_stages() -> &'static str {
    raa_graph::registry::STARTERS
        .iter()
        .find(|(_, stem, _)| *stem == "full-gi-stages")
        .map(|(_, _, json)| *json)
        .expect("generated Full GI stages starter")
}

fn starter_plan() -> GraphPlan {
    plan_from_json(full_gi_stages()).expect("Full GI stages plans")
}

fn node<'a>(plan: &'a GraphPlan, id: &str) -> &'a PlanNode {
    plan.nodes.iter().find(|n| n.id == id).unwrap()
}

fn order(plan: &GraphPlan, id: &str) -> u32 {
    node(plan, id).order_index
}

fn compact_graph(spatial: bool, atrous: &[u32]) -> String {
    let mut nodes = String::from(
        r#"{"id":"prepass","type":"pass.prepass"},
        {"id":"giVolume","type":"pass.gi.volume"},
        {"id":"giPyramid","type":"gi.pyramid"},
        {"id":"giProbes","type":"gi.probes"},
        {"id":"giAccumulate","type":"gi.accumulate"},
        {"id":"giResolve","type":"gi.resolve"},
        {"id":"forward","type":"pass.forward"},
        {"id":"tonemap","type":"pass.tonemap"},
        {"id":"out","type":"output.display"}"#,
    );
    for (i, step) in atrous.iter().enumerate() {
        nodes.push_str(&format!(
            r#",{{"id":"filter{i}","type":"gi.atrous","params":{{"step":{step}}}}}"#
        ));
    }
    let mut links = String::from(
        r#"{"from":"giPyramid.pyramid","to":"giProbes.pyramid"},
        {"from":"giProbes.probesFiltered","to":"giAccumulate.probes"},
        {"from":"giAccumulate.output","to":"giAccumulate.history","history":true},
        {"from":"giAccumulate.meta","to":"giAccumulate.historyMeta","history":true}"#,
    );
    let mut previous = "giAccumulate.filter".to_string();
    for (i, _) in atrous.iter().enumerate() {
        links.push_str(&format!(r#",{{"from":"{previous}","to":"filter{i}.src"}}"#));
        previous = format!("filter{i}.dst");
    }
    links.push_str(&format!(
        r#",{{"from":"{previous}","to":"giResolve.irradiance"}}"#
    ));
    if spatial {
        links.push_str(
            r#",{"from":"giPyramid.pyramid","to":"restirInitial.pyramid"},
            {"from":"restirInitial.reservoirs","to":"restirTemporal.initial"},
            {"from":"restirTemporal.irradiance","to":"giAccumulate.irradiance"}"#,
        );
        nodes.push_str(
            r#",{"id":"restirInitial","type":"gi.restir.initial"},
            {"id":"restirTemporal","type":"gi.restir.temporal"}"#,
        );
    }
    links.push_str(
        r#",{"from":"giResolve.gi","to":"forward.gi"},
        {"from":"forward.color","to":"tonemap.color"},
        {"from":"tonemap.color","to":"out.color"}"#,
    );
    format!(
        r#"{{"format":"raa.render-graph","version":1,"name":"GI stage test","nodes":[{nodes}],"links":[{links}]}}"#
    )
}

#[test]
fn stage_catalog_and_history_inputs() {
    let catalog = Catalog::build();
    let ids = [
        "gi.pyramid",
        "gi.probes",
        "gi.restir.initial",
        "gi.restir.temporal",
        "gi.restir.spatial",
        "gi.accumulate",
        "gi.atrous",
        "gi.resolve",
        "area.trace",
        "area.temporal",
        "area.spatial",
        "area.restir.initial",
        "area.restir.temporal",
        "area.restir.spatial",
        "area.mask",
        "ssgi.trace",
        "ssgi.resolve",
    ];
    let stages: Vec<_> = catalog
        .nodes
        .iter()
        .filter(|n| ids.contains(&n.id))
        .collect();
    assert_eq!(stages.len(), 17);
    assert_eq!(stages.iter().filter(|n| n.family == "gi_stages").count(), 8);
    assert_eq!(
        stages.iter().filter(|n| n.family == "area_stages").count(),
        7
    );
    assert_eq!(
        stages.iter().filter(|n| n.family == "ssgi_stages").count(),
        2
    );
    for spec in &stages {
        assert_eq!(spec.pass, None, "{}", spec.id);
        assert!(spec.multi_instance, "{}", spec.id);
    }
    let mut histories: Vec<_> = stages
        .iter()
        .flat_map(|n| &n.inputs)
        .filter(|p| p.history)
        .map(|p| p.key)
        .collect();
    histories.sort();
    assert_eq!(
        histories,
        [
            "depthHistory",
            "history",
            "history",
            "history",
            "historyMeta"
        ]
    );
}

#[test]
fn full_gi_stages_plan_and_history_wiring() {
    let plan = starter_plan();
    assert!(!plan.nodes.iter().any(|n| {
        ["pass.gi.screen", "pass.gi.area-shadow", "pass.gi.ssgi"].contains(&n.type_id.as_str())
    }));
    let gi_order = [
        "giPyramid",
        "giProbes",
        "giAccumulate",
        "giAtrous1",
        "giAtrous2",
        "giAtrous3",
        "giAtrous4",
        "giResolve",
        "ssgiResolve",
        "forward",
    ];
    assert!(
        gi_order
            .windows(2)
            .all(|w| order(&plan, w[0]) < order(&plan, w[1]))
    );
    let area_order = [
        "giPyramid",
        "areaTrace",
        "areaTemporal",
        "areaSpatial",
        "areaMask",
        "forward",
    ];
    assert!(
        area_order
            .windows(2)
            .all(|w| order(&plan, w[0]) < order(&plan, w[1]))
    );
    assert_eq!(
        node(&plan, "forward")
            .inputs
            .iter()
            .find(|i| i.port == "gi")
            .unwrap()
            .source,
        InputSource::Link {
            node: "ssgiResolve".into(),
            port: "ssgi".into(),
            history: false
        }
    );
    assert_eq!(
        node(&plan, "forward")
            .inputs
            .iter()
            .find(|i| i.port == "areaShadow")
            .unwrap()
            .source,
        InputSource::Link {
            node: "areaMask".into(),
            port: "areaShadow".into(),
            history: false
        }
    );
    let consumers: Vec<_> = plan
        .histories
        .iter()
        .map(|h| (h.consumer.node.as_str(), h.consumer.port.as_str()))
        .collect();
    assert_eq!(
        consumers,
        [
            ("giAccumulate", "history"),
            ("giAccumulate", "historyMeta"),
            ("areaTemporal", "history"),
            ("areaTemporal", "depthHistory"),
            ("ssgiResolve", "history")
        ]
    );
    assert!(plan_fns::history_refusals(&plan, &Catalog::build()).is_empty());
    let changed = full_gi_stages().replacen(
        "\"from\": \"giAccumulate.output\",\n      \"to\": \"giAccumulate.history\"",
        "\"from\": \"giAtrous1.dst\",\n      \"to\": \"giAccumulate.history\"",
        1,
    );
    let refused = plan_from_json(&changed).unwrap();
    assert_eq!(
        plan_fns::history_refusals(&refused, &Catalog::build()),
        vec![("giAccumulate".into(), "history".into())]
    );

    let entries = plan_fns::node_entries(&plan, &Catalog::build());
    let names = [
        "prepass",
        "gi.pyramid#giPyramid",
        "gi.probes#giProbes",
        "gi.accumulate#giAccumulate",
        "gi.atrous#giAtrous1",
        "gi.atrous#giAtrous2",
        "gi.atrous#giAtrous3",
        "gi.atrous#giAtrous4",
        "gi.resolve#giResolve",
        "forward",
        "tonemap",
    ];
    let installed: Vec<_> = names
        .iter()
        .enumerate()
        .map(|(i, name)| PassInfo {
            name: (*name).into(),
            role: PassRole::Pass,
            enabled: true,
            order: if i == 0 {
                30
            } else if i == 9 {
                60
            } else if i == 10 {
                200
            } else {
                42
            },
            required: false,
        })
        .collect();
    assert_eq!(
        plan_fns::link_order(&plan, &entries, &installed),
        names.map(str::to_string)
    );
}

#[test]
fn spatial_stage_can_be_skipped() {
    let plan = plan_from_json(&compact_graph(true, &[])).unwrap();
    assert!(!plan.nodes.iter().any(|n| n.type_id == "gi.restir.spatial"));
    assert_eq!(
        node(&plan, "giAccumulate")
            .inputs
            .iter()
            .find(|i| i.port == "irradiance")
            .unwrap()
            .source,
        InputSource::Link {
            node: "restirTemporal".into(),
            port: "irradiance".into(),
            history: false
        }
    );
    assert!(order(&plan, "restirTemporal") < order(&plan, "giAccumulate"));
}

#[test]
fn two_and_four_atrous_stages_form_ordered_chains() {
    for steps in [&[1, 2][..], &[1, 2, 4, 8][..]] {
        let plan = plan_from_json(&compact_graph(false, steps)).unwrap();
        let filters: Vec<_> = plan
            .nodes
            .iter()
            .filter(|n| n.type_id == "gi.atrous")
            .collect();
        assert_eq!(filters.len(), steps.len());
        assert!(
            filters
                .windows(2)
                .all(|w| w[0].order_index < w[1].order_index)
        );
        let actual: Vec<_> = filters
            .iter()
            .map(|n| {
                n.params
                    .iter()
                    .find(|(k, _)| k == "step")
                    .unwrap()
                    .1
                    .clone()
            })
            .collect();
        assert_eq!(
            actual,
            steps
                .iter()
                .map(|v| raa_graph::catalog::ParamValue::Int(i64::from(*v)))
                .collect::<Vec<_>>()
        );
        let last = filters.last().unwrap();
        let input = node(&plan, "giResolve")
            .inputs
            .iter()
            .find(|i| i.port == "irradiance")
            .unwrap();
        assert_eq!(
            input.source,
            InputSource::Link {
                node: last.id.clone(),
                port: "dst".into(),
                history: false
            }
        );
    }
}
