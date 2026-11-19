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
