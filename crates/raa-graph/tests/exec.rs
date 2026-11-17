//! Pipeline instructions of a plan (S53 level 1): the enabled entries and the pass order, from a
//! hand-built plan and installed list. The catalog is the baseline one (`Catalog::build`). The
//! device-free executor helpers, `settings_targets` and `external_changes`, are tested at the end.

use raa_graph::catalog::{Catalog, FeatureId, ParamValue};
use raa_graph::error::Stage;
use raa_graph::exec::{
    GraphExecutor, PipelineInstructions, SettingsSnapshot, default_values, export_snapshot,
    external_changes, pipeline_instructions, settings_targets,
};
use raa_graph::format::{parse_graph, write_graph};
use raa_graph::plan::{GraphPlan, PlanNode, PlanPortRef, plan_from_json};
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

/// A valid graph: the required passes, the colour chain and one GTAO node whose parameters are the
/// JSON object `ao_params`.
fn graph_text(ao_params: &str) -> String {
    format!(
        r#"{{"format":"raa.render-graph","version":1,"name":"exec test","nodes":[
{{"id":"prepass","type":"pass.prepass"}},
{{"id":"ao","type":"pass.ao.gtao","params":{ao_params}}},
{{"id":"forward","type":"pass.forward"}},
{{"id":"tonemap","type":"pass.tonemap"}},
{{"id":"out","type":"output.display"}}],
"links":[{{"from":"forward.color","to":"tonemap.color"}},{{"from":"tonemap.color","to":"out.color"}}]}}"#
    )
}

/// The configuration before a graph: four passes enabled in default order, and the Gtao defaults.
fn before_graph() -> SettingsSnapshot {
    let passes = ["prepass", "ao.gtao", "forward", "tonemap"];
    SettingsSnapshot {
        passes: passes.iter().map(|name| (name.to_string(), true)).collect(),
        order: passes.iter().map(|name| name.to_string()).collect(),
        values: vec![(FeatureId::Gtao, default_values(&FeatureId::Gtao))],
    }
}

/// The Gtao value of `key` in `targets`.
fn gtao(targets: &SettingsSnapshot, key: &str) -> ParamValue {
    let (_, values) = targets
        .values
        .iter()
        .find(|(feature, _)| *feature == FeatureId::Gtao)
        .expect("Gtao values");
    values
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.clone())
        .expect("Gtao key")
}

#[test]
fn settings_targets_fills_absent_params_with_defaults_and_keeps_written_ones() {
    let catalog = Catalog::build();
    let plan = plan_from_json(&graph_text(r#"{"intensity":0.5}"#)).expect("the graph plans");
    let targets = settings_targets(&plan, &catalog, &before_graph()).expect("the targets resolve");
    assert_eq!(gtao(&targets, "intensity"), ParamValue::Number(0.5));
    for (key, default) in default_values(&FeatureId::Gtao) {
        if key != "intensity" {
            assert_eq!(gtao(&targets, &key), default, "default of {key}");
        }
    }
    assert!(targets.passes.contains(&("ao.gtao".to_owned(), true)));
}

#[test]
fn a_refused_gtao_radius_is_a_param_range_issue_on_its_node() {
    let catalog = Catalog::build();
    // The catalog bounds the radius, so the plan is built with it and the feature refuses it.
    let mut plan = plan_from_json(&graph_text("{}")).expect("the graph plans");
    for node in plan
        .nodes
        .iter_mut()
        .filter(|node| node.type_id == "pass.ao.gtao")
    {
        for (key, value) in &mut node.params {
            if key.as_str() == "radius" {
                *value = ParamValue::Number(25.0);
            }
        }
    }
    let issues = settings_targets(&plan, &catalog, &before_graph())
        .expect_err("a radius of 25 is outside (0, 10]");
    assert_eq!(issues.len(), 1);
    assert_eq!(issues[0].code, "graph.param_range");
    assert_eq!(issues[0].stage, Stage::Apply);
    assert_eq!(issues[0].node.as_deref(), Some("ao"));
    assert_eq!(issues[0].param.as_deref(), Some("radius"));
}

#[test]
fn external_changes_names_the_pipeline_and_the_changed_feature() {
    let catalog = Catalog::build();
    let plan = plan_from_json(&graph_text("{}")).expect("the graph plans");
    let applied = settings_targets(&plan, &catalog, &before_graph()).expect("the targets resolve");
    assert!(external_changes(&applied, &applied).is_empty());

    let mut observed = applied.clone();
    observed.order.swap(0, 1);
    for (feature, values) in &mut observed.values {
        if *feature == FeatureId::Gtao {
            for (key, value) in values.iter_mut() {
                if key.as_str() == "radius" {
                    *value = ParamValue::Number(2.0);
                }
            }
        }
    }
    assert_eq!(
        external_changes(&applied, &observed),
        vec![FeatureId::Pipeline, FeatureId::Gtao]
    );
}

/// Sets the value of `key` in one feature's values, which `default_values` gives for every key.
fn set_value(values: &mut [(String, ParamValue)], key: &str, value: ParamValue) {
    let (_, slot) = values
        .iter_mut()
        .find(|(name, _)| name == key)
        .expect("a snapshot key");
    *slot = value;
}

/// The Setting values of the export tests: the defaults of every feature with keys, with the AO
/// radius at 1.5. `full` removes Fog's values and turns Gi's presence keys on (Gi's volume and SSGI
/// passes are enabled).
fn round_trip_values(full: bool) -> Vec<(FeatureId, Vec<(String, ParamValue)>)> {
    let features = [
        FeatureId::Tonemap,
        FeatureId::Shadow,
        FeatureId::Vg,
        FeatureId::GiQuality,
        FeatureId::Gi,
        FeatureId::AreaShadows,
        FeatureId::AreaShading,
        FeatureId::Gtao,
        FeatureId::Contact,
        FeatureId::Ssr,
        FeatureId::Fog,
        FeatureId::Shafts,
        FeatureId::Oit,
        FeatureId::Dof,
        FeatureId::Bloom,
        FeatureId::Grading,
        FeatureId::Cel,
    ];
    let mut values = Vec::new();
    for feature in features {
        if full && feature == FeatureId::Fog {
            continue;
        }
        let mut entry = default_values(&feature);
        match feature {
            FeatureId::Gtao => set_value(&mut entry, "radius", ParamValue::Number(1.5)),
            FeatureId::Gi if full => {
                set_value(&mut entry, "enabled", ParamValue::Bool(true));
                set_value(&mut entry, "screenSpace", ParamValue::Bool(true));
            }
            _ => {}
        }
        values.push((feature, entry));
    }
    values
}

/// An all-enabled snapshot in default order. `full` adds the side-effect entries `sky.atmosphere`,
/// `sky.cloud-screen`, `gi.volume` and `gi.ssgi`, and `fog`; `sky.background` is a contributor.
fn round_trip_snapshot(full: bool) -> SettingsSnapshot {
    let order: Vec<&str> = if full {
        vec![
            "sky.atmosphere",
            "shadow.csm",
            "gi.volume",
            "prepass",
            "light.tiles",
            "gi.ssgi",
            "sky.cloud-screen",
            "forward",
            "ssr",
            "fog",
            "tonemap",
        ]
    } else {
        vec![
            "shadow.csm",
            "prepass",
            "light.tiles",
            "forward",
            "ssr",
            "tonemap",
        ]
    };
    let mut passes: Vec<(String, bool)> =
        order.iter().map(|name| (name.to_string(), true)).collect();
    passes.push(("sky.background".to_owned(), true));
    SettingsSnapshot {
        passes,
        order: order.iter().map(|name| name.to_string()).collect(),
        values: round_trip_values(full),
    }
}

#[test]
fn the_export_writes_parses_and_plans() {
    let doc = export_snapshot(&Catalog::build(), &round_trip_snapshot(false), 4);
    let text = write_graph(&doc);
    parse_graph(&text).expect("the export parses");
    plan_from_json(&text).expect("the export plans");
}

#[test]
fn settings_targets_of_the_exported_plan_is_the_snapshot() {
    let catalog = Catalog::build();
    let snapshot = round_trip_snapshot(false);
    let doc = export_snapshot(&catalog, &snapshot, 4);
    let plan = plan_from_json(&write_graph(&doc)).expect("the export plans");
    let targets = settings_targets(&plan, &catalog, &snapshot).expect("the targets resolve");
    assert_eq!(targets, snapshot);
}

#[test]
fn side_effect_passes_round_trip_and_an_absent_feature_keeps_its_catalog_defaults() {
    let catalog = Catalog::build();
    let snapshot = round_trip_snapshot(true);
    let doc = export_snapshot(&catalog, &snapshot, 4);
    let fog = doc
        .nodes
        .iter()
        .find(|node| node.id == "fog")
        .expect("fog is enabled");
    for param in &catalog.node("pass.fog").expect("pass.fog").params {
        let written = fog
            .params
            .iter()
            .find(|(key, _)| key == param.key)
            .map(|(_, value)| value);
        assert_eq!(written, Some(&param.default.to_json()), "{}", param.key);
    }

    let plan = plan_from_json(&write_graph(&doc)).expect("the export plans");
    for id in ["skyAtmosphere", "skyCloudScreen", "giSsgi"] {
        assert!(
            plan.nodes.iter().any(|node| node.id == id),
            "{id} is planned"
        );
    }
    for type_id in [
        "pass.sky.atmosphere",
        "pass.sky.cloud-screen",
        "pass.gi.ssgi",
    ] {
        assert!(
            catalog.node(type_id).is_some_and(|spec| spec.side_effect),
            "{type_id} is a side-effect pass"
        );
    }
    let targets = settings_targets(&plan, &catalog, &snapshot).expect("the targets resolve");
    assert_eq!(targets, snapshot);
}

#[test]
fn node_ids_are_lower_camel_case_and_the_colour_chain_runs_forward_to_out() {
    let doc = export_snapshot(&Catalog::build(), &round_trip_snapshot(false), 4);
    let ids: Vec<&str> = doc.nodes.iter().map(|node| node.id.as_str()).collect();
    assert_eq!(
        ids,
        [
            "shadowCsm",
            "prepass",
            "lightTiles",
            "forward",
            "skyBackground",
            "ssr",
            "tonemap",
            "out",
        ]
    );
    let links: Vec<String> = doc
        .links
        .iter()
        .map(|link| {
            format!(
                "{}.{} -> {}.{}",
                link.from.node, link.from.port, link.to.node, link.to.port
            )
        })
        .collect();
    assert_eq!(
        links,
        [
            "forward.color -> ssr.color",
            "ssr.color -> tonemap.color",
            "tonemap.color -> out.color",
        ]
    );
}

#[test]
fn owned_setters_has_the_twenty_s54_setters_starting_with_set_pass_enabled() {
    let setters = GraphExecutor::owned_setters();
    assert_eq!(setters.len(), 20);
    assert_eq!(setters.first().copied(), Some("setPassEnabled"));
}

/// The value of `key` of `feature` in `targets`.
fn target(targets: &SettingsSnapshot, feature: &FeatureId, key: &str) -> ParamValue {
    targets
        .values
        .iter()
        .find(|(candidate, _)| candidate == feature)
        .and_then(|(_, values)| values.iter().find(|(name, _)| name == key))
        .map(|(_, value)| value.clone())
        .unwrap_or_else(|| panic!("no {feature:?} value {key}"))
}

/// P5-Q2: applying the Simple forward starter enables the cascade shadow pass and sends the shadow
/// node's parameters to the Shadow feature (the defaults, and a written value in place of its
/// default); the absent `shadow.contact` and `ao.gtao` passes are disabled and the Contact feature
/// keeps its values.
#[test]
fn simple_forward_apply_enables_the_sun_shadow_and_forwards_its_parameters() {
    let catalog = Catalog::build();
    let (_, _, json) = raa_graph::registry::STARTERS
        .iter()
        .find(|&&(_, stem, _)| stem == "simple-forward")
        .expect("the Simple forward starter");
    let passes = [
        "shadow.csm",
        "prepass",
        "light.tiles",
        "ao.gtao",
        "shadow.contact",
        "forward",
        "tonemap",
    ];
    let mut before = SettingsSnapshot {
        passes: passes
            .iter()
            .map(|name| (name.to_string(), false))
            .collect(),
        order: passes.iter().map(|name| name.to_string()).collect(),
        values: vec![
            (FeatureId::Shadow, default_values(&FeatureId::Shadow)),
            (FeatureId::Contact, default_values(&FeatureId::Contact)),
        ],
    };
    before.passes.push(("sky.background".to_owned(), false));
    let plan = plan_from_json(json).expect("Simple forward plans");
    let targets = settings_targets(&plan, &catalog, &before).expect("the targets resolve");
    for name in ["shadow.csm", "forward", "sky.background"] {
        assert!(
            targets.passes.contains(&(name.to_owned(), true)),
            "{name} enabled: {:?}",
            targets.passes
        );
    }
    for name in ["ao.gtao", "shadow.contact"] {
        assert!(
            targets.passes.contains(&(name.to_owned(), false)),
            "{name} disabled: {:?}",
            targets.passes
        );
    }
    for feature in [FeatureId::Shadow, FeatureId::Contact] {
        for (key, default) in default_values(&feature) {
            assert_eq!(
                target(&targets, &feature, &key),
                default,
                "{feature:?} {key}"
            );
        }
    }

    // A written shadow parameter reaches the Shadow feature in place of its default.
    let mut plan = plan;
    for node in plan
        .nodes
        .iter_mut()
        .filter(|node| node.type_id == "pass.shadow.csm")
    {
        for (key, value) in &mut node.params {
            match key.as_str() {
                "distance" => *value = ParamValue::Number(25.0),
                "softness" => *value = ParamValue::Number(0.5),
                _ => {}
            }
        }
    }
    let targets = settings_targets(&plan, &catalog, &before).expect("the targets resolve");
    assert_eq!(
        target(&targets, &FeatureId::Shadow, "distance"),
        ParamValue::Number(25.0)
    );
    assert_eq!(
        target(&targets, &FeatureId::Shadow, "softness"),
        ParamValue::Number(0.5)
    );
}
