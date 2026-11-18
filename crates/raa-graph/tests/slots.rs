//! The `slots` family (contract slots-family, slots-two-ao-doc): the catalog flags, a second AO
//! branch that feeds the forward `ao` slot and plans, any image of the slot's type and format
//! accepted, a wrong format refused on forward, and the fixed slot set checked against the
//! fragment sampled-texture limit.

use raa_graph::catalog::{Catalog, GraphFormat, ParamValue, PortType};
use raa_graph::error::{Issue, Severity, Stage};
use raa_graph::format::parse_graph;
use raa_graph::plan::{InputSource, plan_from_json};
use raa_graph::registry;
use raa_graph::validate::{GraphLimits, ValidateCtx, validate};

/// The nine forward-class slot inputs that take any image of their type.
const FORWARD_SLOTS: [&str; 9] = [
    "shadowMap",
    "envSpecular",
    "ao",
    "contact",
    "gi",
    "areaShadow",
    "cloudTransmittance",
    "opaqueColor",
    "opaqueDepth",
];

/// The G-buffer inputs that take a bound image (contract slots-family, row 2).
const GBUFFER: [(&str, &[&str]); 13] = [
    ("pass.ao.gtao", &["depth", "normal"]),
    ("pass.shadow.contact", &["depth"]),
    ("pass.light.tiles", &["depth"]),
    ("pass.ssr", &["depth", "normal", "material"]),
    ("pass.taa", &["depth", "motion"]),
    ("pass.gi.screen", &["depth", "normal", "motion"]),
    ("pass.gi.ssgi", &["depth", "normal", "motion"]),
    ("pass.gi.area-shadow", &["depth", "normal", "motion"]),
    ("pass.dof", &["depth"]),
    ("pass.sunshafts", &["depth"]),
    ("pass.transparent.oit", &["depth"]),
    ("pass.sky.cloud-screen", &["depth"]),
    ("pass.tonemap", &["depth"]),
];

/// The G-buffer inputs that stay fixed, so a link from another source is refused.
const KEEP_FIXED: [(&str, &str); 7] = [
    ("pass.fog", "depth"),
    ("pass.opaque.copy", "depth"),
    ("pass.particles.blend", "depth"),
    ("pass.cel.resolve", "depth"),
    ("pass.cel.resolve", "normal"),
    ("pass.trace.sdf", "depth"),
    ("pass.gi.screen", "material"),
];

/// Runs `check` with the context of tests/plan.rs `plan_at`: every pass installed, one viewport,
/// msaa 4.
fn with_ctx<T>(limits: GraphLimits, check: impl FnOnce(&ValidateCtx<'_>) -> T) -> T {
    let catalog = Catalog::build();
    let installed = |_: &str| true;
    let engine_options = [("msaa", ParamValue::Int(4))];
    let ctx = ValidateCtx {
        catalog: &catalog,
        limits,
        installed: &installed,
        viewports: 1,
        engine_options: &engine_options,
    };
    check(&ctx)
}

/// The issues of the document `text` under `limits`; the text must parse.
fn issues_of(text: &str, limits: GraphLimits) -> Vec<Issue> {
    let doc = parse_graph(text).unwrap_or_else(|issues| panic!("the fixture parses: {issues:?}"));
    with_ctx(limits, |ctx| validate(&doc, ctx))
}

/// The `slots-two-ao-doc` text. `@AO@` is the source of the forward `ao` link and `@EXTRA@` an
/// optional node line placed after aoWide.
const TWO_AO: &str = r#"{"format": "raa.render-graph", "version": 1, "name": "Simple forward two AO branches", "nodes": [
{"id": "skyAtmosphere", "type": "pass.sky.atmosphere"},
{"id": "skyCloudLighting", "type": "pass.sky.cloud-lighting"},
{"id": "skyEnv", "type": "pass.sky.env"},
{"id": "shadowCsm", "type": "pass.shadow.csm"},
{"id": "prepass", "type": "pass.prepass"},
{"id": "lightTiles", "type": "pass.light.tiles"},
{"id": "aoNarrow", "type": "pass.ao.gtao", "params": {"radius": 0.3, "intensity": 1, "power": 2}},
{"id": "aoWide", "type": "pass.ao.gtao", "params": {"radius": 2, "intensity": 1, "power": 2}},
@EXTRA@
{"id": "forward", "type": "pass.forward"},
{"id": "skyBackground", "type": "pass.sky.background"},
{"id": "tonemap", "type": "pass.tonemap", "params": {"mapper": "agx", "exposureEv": 0}},
{"id": "vNarrow", "type": "view.image", "params": {"channels": "r", "label": "ao 0.3 m"}},
{"id": "vWide", "type": "view.image", "params": {"channels": "r", "label": "ao 2 m"}},
{"id": "aoNarrowVsWide", "type": "view.compare", "params": {"mode": "split", "position": 0.5}},
{"id": "vFinal", "type": "view.image", "params": {"label": "final, forward ao from 2 m"}},
{"id": "grid", "type": "view.grid", "params": {"columns": 2}},
{"id": "out", "type": "output.display"}
], "links": [
{"from": "prepass.depth", "to": "lightTiles.depth"},
{"from": "shadowCsm.shadowMap", "to": "forward.shadowMap"},
{"from": "lightTiles.lightTiles", "to": "forward.lightTiles"},
{"from": "@AO@", "to": "forward.ao"},
{"from": "forward.color", "to": "tonemap.color"},
{"from": "aoNarrow.ao", "to": "vNarrow.src"},
{"from": "aoWide.ao", "to": "vWide.src"},
{"from": "vNarrow.color", "to": "aoNarrowVsWide.a"},
{"from": "vWide.color", "to": "aoNarrowVsWide.b"},
{"from": "tonemap.color", "to": "vFinal.src"},
{"from": "aoNarrowVsWide.color", "to": "grid.t0"},
{"from": "vFinal.color", "to": "grid.t1"},
{"from": "grid.color", "to": "out.color"}
]}"#;

/// The two-AO document with the forward `ao` link from `ao_from` and an optional extra node line.
fn two_ao(ao_from: &str, extra: &str) -> String {
    TWO_AO.replace("@AO@", ao_from).replace("@EXTRA@", extra)
}

/// The number of issues with `code`.
fn count(issues: &[Issue], code: &str) -> usize {
    issues.iter().filter(|issue| issue.code == code).count()
}

#[test]
fn slot_gbuffer_and_overlay_inputs_carry_the_contract_flags() {
    let catalog = Catalog::build();
    let generic = |node: &str, key: &str| {
        let spec = catalog
            .node(node)
            .unwrap_or_else(|| panic!("no node type {node}"));
        spec.inputs
            .iter()
            .find(|port| port.key == key)
            .unwrap_or_else(|| panic!("{node} has no input {key}"))
            .generic
    };
    for key in FORWARD_SLOTS {
        assert!(generic("pass.forward", key), "forward {key} is generic");
    }
    assert!(!generic("pass.forward", "lightTiles"));
    for (node, keys) in GBUFFER {
        for key in keys {
            assert!(generic(node, key), "{node} {key} is generic");
        }
    }
    for (node, key) in KEEP_FIXED {
        assert!(!generic(node, key), "{node} {key} stays fixed");
    }
    assert!(catalog.node("pass.ao.gtao").unwrap().multi_instance);
    assert!(catalog.node("pass.shadow.contact").unwrap().multi_instance);
    assert!(!catalog.node("pass.prepass").unwrap().multi_instance);

    let overlay = catalog.node("pass.overlay").expect("pass.overlay exists");
    let depth = overlay
        .inputs
        .iter()
        .find(|port| port.key == "depth")
        .expect("pass.overlay has a depth input");
    assert_eq!(depth.ty, PortType::Depth2d);
    assert_eq!(depth.format, Some(GraphFormat::Depth32Float));
    assert_eq!(
        depth.default_source,
        ["pass.prepass.depth", "pass.vg.gbuffer.depth"]
    );
    assert!(!depth.absent_ok && depth.generic && depth.slot.is_none());
}

#[test]
fn a_second_ao_branch_validates_and_feeds_the_forward_ao_slot() {
    let text = two_ao("aoWide.ao", "");
    let issues = issues_of(&text, GraphLimits::CORE);
    assert!(
        !issues.iter().any(|issue| issue.severity == Severity::Error),
        "{issues:?}"
    );
    let plan = plan_from_json(&text).unwrap_or_else(|issues| panic!("it plans: {issues:?}"));
    assert!(plan.nodes.iter().any(|node| node.id == "aoNarrow"));
    assert!(plan.nodes.iter().any(|node| node.id == "aoWide"));
    let forward = plan
        .nodes
        .iter()
        .find(|node| node.id == "forward")
        .expect("forward is planned");
    let ao = forward
        .inputs
        .iter()
        .find(|input| input.port == "ao")
        .expect("forward has an ao input");
    assert_eq!(
        ao.source,
        InputSource::Link {
            node: "aoWide".into(),
            port: "ao".into(),
            history: false,
        }
    );
}

#[test]
fn any_image_of_the_slot_type_and_format_feeds_the_forward_slot() {
    let extra = r#"{"id": "shadowContact", "type": "pass.shadow.contact"},"#;
    let issues = issues_of(&two_ao("shadowContact.contact", extra), GraphLimits::CORE);
    assert!(
        !issues.iter().any(|issue| issue.severity == Severity::Error),
        "{issues:?}"
    );
}

#[test]
fn a_format_the_slot_cannot_take_is_a_link_type_on_forward() {
    let issues = issues_of(&two_ao("aoWide.raw", ""), GraphLimits::CORE);
    assert_eq!(count(&issues, "graph.link_type"), 1, "{issues:?}");
    let link = issues
        .iter()
        .find(|issue| issue.code == "graph.link_type")
        .expect("one link_type issue");
    assert_eq!(link.node.as_deref(), Some("forward"));
    assert_eq!(link.port.as_deref(), Some("ao"));
}

#[test]
fn a_fixed_slot_set_above_the_fragment_limit_is_a_budget_issue() {
    let (_, _, simple) = registry::STARTERS
        .iter()
        .copied()
        .find(|&(name, _, _)| name == "Simple forward")
        .expect("the Simple forward starter exists");
    let limits = GraphLimits {
        sampled_textures: 9,
        ..GraphLimits::CORE
    };
    let issues = issues_of(simple, limits);
    let budget: Vec<&Issue> = issues
        .iter()
        .filter(|issue| issue.code == "graph.budget" && issue.message.contains("fixed slot set"))
        .collect();
    assert_eq!(budget.len(), 1, "{issues:?}");
    assert_eq!(budget[0].node.as_deref(), Some("forward"));
    assert_eq!(budget[0].stage, Stage::Budget);
    assert!(budget[0].message.contains("of 10 sampled textures"));
    assert!(
        budget[0]
            .fix
            .contains("Raise maxSampledTexturesPerShaderStage to 10")
    );
}

#[test]
fn no_starter_keeps_a_slot_set_above_the_core_limit() {
    for &(name, stem, json) in registry::STARTERS {
        let issues = issues_of(json, GraphLimits::CORE);
        assert!(
            issues
                .iter()
                .all(|issue| !issue.message.contains("fixed slot set")),
            "{name} ({stem}): {issues:?}"
        );
    }
}
