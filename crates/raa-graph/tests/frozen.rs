//! Frozen plans reconstruct their committed source and stay current with the node catalog.

use raa_graph::catalog::Catalog;
use raa_graph::plan::{GraphPlan, plan_from_json};
use raa_graph::registry;

/// Assert every frozen node was generated against the catalog's current type version.
fn assert_plan_versions_current(name: &str, plan: &GraphPlan, catalog: &Catalog) {
    for node in &plan.nodes {
        let spec = catalog.node(&node.type_id).unwrap_or_else(|| {
            panic!(
                "frozen plan {name:?} has unknown node type {:?}",
                node.type_id
            )
        });
        assert_eq!(
            node.type_version, spec.version,
            "frozen plan {name:?} has a stale version for {}",
            node.type_id
        );
    }
}

#[test]
fn frozen_plans_match_source_and_current_versions() {
    let catalog = Catalog::build();
    let frozen = registry::frozen_all();
    assert!(
        frozen.iter().any(|(name, _, _)| *name == "simple_forward"),
        "simple_forward must be in the frozen registry"
    );

    for (name, make_plan, source_json) in frozen {
        let frozen_plan = make_plan();
        let source_plan = plan_from_json(source_json)
            .unwrap_or_else(|issues| panic!("frozen source {name:?} does not plan: {issues:?}"));
        assert_eq!(
            frozen_plan, source_plan,
            "frozen plan {name:?} differs from its source"
        );
        assert_plan_versions_current(name, &frozen_plan, &catalog);
    }
}

#[test]
fn frozen_area_shadow_signals_match_editable_plans() {
    use raa_graph::catalog::SignalEncoding;
    let mut area_signals = 0;
    for (_, make_plan, source_json) in registry::frozen_all() {
        let frozen = make_plan();
        let editable = plan_from_json(source_json).unwrap();
        for (frozen_node, editable_node) in frozen.nodes.iter().zip(&editable.nodes) {
            for input in &frozen_node.inputs {
                let expected = editable.input_signal(editable_node, &input.port);
                assert_eq!(frozen.input_signal(frozen_node, &input.port), expected);
                if let Some(signal) = expected {
                    assert_eq!(signal.encoding, SignalEncoding::PackedVisibility);
                    assert_eq!(signal.version, 1);
                    area_signals += 1;
                }
            }
        }
    }
    assert!(area_signals > 0);
}

#[test]
fn frozen_registry_has_the_shipped_graphs() {
    let catalog = Catalog::build();
    let frozen = registry::frozen_all();
    let names: Vec<_> = frozen.iter().map(|(name, _, _)| *name).collect();
    assert_eq!(names, ["debug", "full_gi", "simple_forward", "spherical_gi"]);

    for (name, make_plan, source_json) in frozen {
        let frozen_plan = make_plan();
        let source_plan = plan_from_json(source_json)
            .unwrap_or_else(|issues| panic!("frozen source {name:?} does not plan: {issues:?}"));
        assert_eq!(
            frozen_plan, source_plan,
            "frozen plan {name:?} differs from its saved source"
        );
        assert_plan_versions_current(name, &frozen_plan, &catalog);
    }
}

// The device-free selection of the m33 API: the Engine binding itself is wasm32-only, but the
// unknown/stale decision is plain Rust, so this test includes the file by path instead of adding
// a dependency from raa-graph to raa-js.
#[cfg(not(target_arch = "wasm32"))]
#[path = "../../raa-js/src/api/m33_frozen_render_graph.rs"]
mod m33;

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn frozen_api_selection() {
    use m33::select_frozen;
    use raa_graph::format::parse_graph;
    use raa_graph::validate::canonical_doc;

    let catalog = Catalog::build();
    let (_, make_plan, source_json) = registry::frozen_all()
        .into_iter()
        .find(|(name, _, _)| *name == "simple_forward")
        .expect("simple_forward must be in the frozen registry");

    let selected = select_frozen("simple_forward", &catalog).expect("the starter selects");
    assert_eq!(
        selected.plan,
        make_plan(),
        "the selected plan is the literal frozen plan"
    );
    let source_doc = parse_graph(source_json).expect("the saved source parses");
    assert_eq!(
        selected.doc,
        canonical_doc(&source_doc, &catalog),
        "the selected document is the canonical saved source"
    );

    let unknown = select_frozen("does_not_exist", &catalog).expect_err("an absent name refuses");
    assert_eq!(unknown.len(), 1);
    assert_eq!(unknown[0].code, "graph.frozen_unknown");

    let mut changed = Catalog::build();
    let node_id = selected.plan.nodes[0].id.clone();
    let type_id = selected.plan.nodes[0].type_id.clone();
    let spec = changed
        .nodes
        .iter_mut()
        .find(|spec| spec.id == type_id.as_str())
        .expect("the plan's type is in the catalog");
    spec.version += 1;
    let stale = select_frozen("simple_forward", &changed)
        .expect_err("a catalog version change refuses the literal");
    assert_eq!(stale[0].code, "graph.frozen_stale");
    assert_eq!(stale[0].node.as_deref(), Some(node_id.as_str()));

    let again = select_frozen("simple_forward", &Catalog::build())
        .expect("the unchanged catalog still selects");
    assert_eq!(
        again.plan, selected.plan,
        "a refused selection never upgrades the literal"
    );
}

#[test]
fn changed_node_version_fails_the_catalog_version_assertion() {
    let catalog = Catalog::build();
    let (name, make_plan, _) = registry::frozen_all()
        .into_iter()
        .find(|(name, _, _)| *name == "simple_forward")
        .expect("simple_forward must be in the frozen registry");
    let mut changed = make_plan();
    let node = changed.nodes.first_mut().expect("simple_forward has nodes");
    node.type_version += 1;

    let assertion = std::panic::catch_unwind(|| {
        assert_plan_versions_current(name, &changed, &catalog);
    });
    assert!(
        assertion.is_err(),
        "the catalog version assertion must reject an artificially changed frozen node version"
    );
}
