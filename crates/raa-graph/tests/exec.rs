//! Pipeline instructions of a plan (S53 level 1): the enabled entries and the pass order, from a
//! hand-built plan and installed list. The catalog is the baseline one (`Catalog::build`).

use raa_graph::catalog::Catalog;
use raa_graph::exec::{PipelineInstructions, pipeline_instructions};
use raa_graph::plan::{GraphPlan, PlanNode, PlanPortRef};
use raa_pipeline::{PassInfo, PassRole};

/// A plan whose nodes have the given types, in plan order.
fn plan(types: &[&str]) -> GraphPlan {
    let nodes = types
        .iter()
        .enumerate()
        .map(|(index, type_id)| PlanNode {
            id: format!("n{index}"),
            type_id: type_id.to_string(),
            type_version: 1,
            order_index: index as u32,
            params: Vec::new(),
            inputs: Vec::new(),
            compute: None,
        })
        .collect();
    GraphPlan {
        name: "test".to_string(),
        nodes,
        links: Vec::new(),
        resources: Vec::new(),
        histories: Vec::new(),
        output: PlanPortRef {
            node: "out".to_string(),
            port: "color".to_string(),
        },
    }
}

fn entry(name: &str, role: PassRole, order: u32) -> PassInfo {
    PassInfo {
        name: name.to_string(),
        role,
        enabled: true,
        order,
        required: false,
    }
}

/// The installed entries of a graph with the S12 default orders: passes, then the contributors
/// of `forward`.
fn installed() -> Vec<PassInfo> {
    vec![
        entry("shadow.csm", PassRole::Pass, 20),
        entry("prepass", PassRole::Pass, 30),
        entry("light.tiles", PassRole::Pass, 35),
        entry("ao.gtao", PassRole::Pass, 40),
        entry("forward", PassRole::Pass, 60),
        entry("ssr", PassRole::Pass, 65),
        entry("fog", PassRole::Pass, 68),
        entry("tonemap", PassRole::Pass, 200),
        entry("fur.shells", PassRole::Contributor, 500),
        entry("sky.background", PassRole::Contributor, 900),
    ]
}

fn enabled(instructions: &PipelineInstructions) -> Vec<(&str, bool)> {
    instructions
        .enable
        .iter()
        .map(|(name, on)| (name.as_str(), *on))
        .collect()
}

#[test]
fn enabled_marks_every_installed_entry_the_plan_drives_in_installed_order() {
    let catalog = Catalog::build();
    let plan = plan(&[
        "pass.prepass",
        "pass.light.tiles",
        "pass.forward",
        "pass.ssr",
        "pass.tonemap",
        "output.display",
    ]);
    let instructions = pipeline_instructions(&plan, &catalog, &installed());
    assert_eq!(
        enabled(&instructions),
        [
            ("shadow.csm", false),
            ("prepass", true),
            ("light.tiles", true),
            ("ao.gtao", false),
            ("forward", true),
            ("ssr", true),
            ("fog", false),
            ("tonemap", true),
            ("fur.shells", false),
            ("sky.background", false),
        ]
    );
}

#[test]
fn an_absent_installed_pass_is_inserted_before_the_first_planned_pass_of_greater_order() {
    let catalog = Catalog::build();
    let plan = plan(&[
        "pass.prepass",
        "pass.light.tiles",
        "pass.forward",
        "pass.tonemap",
    ]);
    let instructions = pipeline_instructions(&plan, &catalog, &installed());
    // shadow.csm (20) before prepass (30); ao.gtao (40) before forward (60); ssr (65) and fog
    // (68) before tonemap (200). Each installed pass appears exactly once.
    assert_eq!(
        instructions.order,
        [
            "shadow.csm",
            "prepass",
            "light.tiles",
            "ao.gtao",
            "forward",
            "ssr",
            "fog",
            "tonemap",
        ]
    );
}

#[test]
fn a_contributor_is_enabled_only_when_the_plan_has_its_node_and_is_not_ordered() {
    let catalog = Catalog::build();
    let plan = plan(&[
        "pass.prepass",
        "pass.forward",
        "pass.tonemap",
        "pass.sky.background",
    ]);
    let instructions = pipeline_instructions(&plan, &catalog, &installed());
    let on = enabled(&instructions);
    assert!(on.contains(&("sky.background", true)));
    assert!(on.contains(&("fur.shells", false)));
    assert!(
        !instructions
            .order
            .iter()
            .any(|name| name == "sky.background")
    );
    assert!(!instructions.order.iter().any(|name| name == "fur.shells"));
}

#[test]
fn an_all_present_plan_keeps_the_plan_order_even_when_it_differs_from_the_default() {
    let catalog = Catalog::build();
    let installed = vec![
        entry("prepass", PassRole::Pass, 30),
        entry("light.tiles", PassRole::Pass, 35),
        entry("forward", PassRole::Pass, 60),
        entry("ssr", PassRole::Pass, 65),
        entry("fog", PassRole::Pass, 68),
        entry("tonemap", PassRole::Pass, 200),
    ];
    // fog is planned before ssr, against the default order: the plan order wins.
    let plan = plan(&[
        "pass.prepass",
        "pass.light.tiles",
        "pass.forward",
        "pass.fog",
        "pass.ssr",
        "pass.tonemap",
    ]);
    let instructions = pipeline_instructions(&plan, &catalog, &installed);
    assert!(instructions.enable.iter().all(|(_, on)| *on));
    assert_eq!(
        instructions.order,
        ["prepass", "light.tiles", "forward", "fog", "ssr", "tonemap"]
    );
}
