//! Catalog and device-free plan contracts for the post-stage node families.
#[allow(dead_code)]
#[path = "../src/exec_ext/post_stages/plan.rs"]
mod plan;
#[allow(dead_code)]
#[path = "../src/exec_ext/wiring/plan_fns.rs"]
mod plan_fns;

use plan::{
    POST_STAGE_TYPES, bloom_composites, karis_nodes, stage_bindings, stage_nodes, stage_values,
    view_aliases,
};
use raa_graph::Catalog;
use raa_graph::catalog::{Category, GraphFormat, NodeTypeSpec, ParamBinding, ParamValue};
use raa_graph::plan::{
    GraphPlan, InputSource, PlanHistory, PlanNode, PlanPortRef, PlanResource, plan_from_json,
};

fn graph(nodes: Vec<String>, links: Vec<String>) -> String {
    format!(
        r#"{{"format":"raa.render-graph","version":1,"name":"stages","nodes":[{}],"links":[{}]}}"#,
        nodes.join(","),
        links.join(",")
    )
}
fn node(id: &str, ty: &str) -> String {
    format!(r#"{{"id":"{id}","type":"{ty}"}}"#)
}
fn link(from: &str, to: &str) -> String {
    format!(r#"{{"from":"{from}","to":"{to}"}}"#)
}
fn base_nodes() -> Vec<String> {
    [
        ("prepass", "pass.prepass"),
        ("forward", "pass.forward"),
        ("tonemap", "pass.tonemap"),
        ("out", "output.display"),
    ]
    .into_iter()
    .map(|(id, ty)| node(id, ty))
    .collect()
}
fn base_links() -> Vec<String> {
    vec![
        link("forward.color", "tonemap.color"),
        link("tonemap.color", "out.color"),
    ]
}
fn planned(text: &str) -> GraphPlan {
    plan_from_json(text).expect("graph plans")
}

#[test]
fn stage_types() {
    let catalog = Catalog::build();
    assert_eq!(
        POST_STAGE_TYPES,
        [
            "ao.trace",
            "ao.blur",
            "ssr.hiz",
            "ssr.trace",
            "ssr.composite",
            "bloom.down",
            "bloom.up"
        ]
    );
    for (id, family, category, order, inputs, outputs) in [
        (
            "ao.trace",
            "ao_stages",
            Category::Lighting,
            40,
            vec!["depth", "normal"],
            vec!["raw"],
        ),
        (
            "ao.blur",
            "ao_stages",
            Category::Lighting,
            40,
            vec!["src", "depth"],
            vec!["ao", "blur"],
        ),
        (
            "ssr.hiz",
            "ssr_stages",
            Category::Post,
            65,
            vec!["depth"],
            vec!["hiz"],
        ),
        (
            "ssr.trace",
            "ssr_stages",
            Category::Post,
            65,
            vec!["depth", "normal", "hiz"],
            vec!["trace"],
        ),
        (
            "ssr.composite",
            "ssr_stages",
            Category::Post,
            65,
            vec!["color", "depth", "normal", "material", "trace"],
            vec!["color"],
        ),
        (
            "bloom.down",
            "bloom_stages",
            Category::Post,
            120,
            vec!["src"],
            vec!["level"],
        ),
        (
            "bloom.up",
            "bloom_stages",
            Category::Post,
            120,
            vec!["src", "base", "color"],
            vec!["level", "color"],
        ),
    ] {
        let spec = catalog.node(id).unwrap();
        assert_eq!(
            (
                spec.family,
                &spec.category,
                spec.default_order,
                spec.multi_instance,
                spec.pass
            ),
            (family, &category, order, true, None),
            "{id}"
        );
        assert_eq!(
            spec.inputs.iter().map(|p| p.key).collect::<Vec<_>>(),
            inputs,
            "{id}"
        );
        assert_eq!(
            spec.outputs.iter().map(|p| p.key).collect::<Vec<_>>(),
            outputs,
            "{id}"
        );
        assert!(spec.inputs.iter().all(|p| p.generic
            && !p.history
            && p.slot.is_none()
            && p.mips == 1
            && p.layers == 1
            && p.size == raa_graph::catalog::SizeSpec::Scale(1.0)));
        assert!(spec.outputs.iter().all(|p| !p.required
            && p.default_source.is_empty()
            && !p.generic
            && (p.mips == 1 || (id == "ssr.hiz" && p.key == "hiz"))
            && p.layers == 1
            && (p.size == raa_graph::catalog::SizeSpec::Scale(1.0)
                || (id == "bloom.down"
                    && p.key == "level"
                    && p.size == raa_graph::catalog::SizeSpec::Input("src"))
                || (id == "bloom.up"
                    && p.key == "level"
                    && p.size == raa_graph::catalog::SizeSpec::Input("base")))));
        for input in &spec.inputs {
            match input.key {
                "depth" => assert_eq!(
                    (
                        input.ty.clone(),
                        input.format.clone(),
                        input.required,
                        input.absent_ok,
                        input.default_source
                    ),
                    (
                        raa_graph::catalog::PortType::Depth2d,
                        Some(GraphFormat::Depth32Float),
                        false,
                        false,
                        &["pass.prepass.depth", "pass.vg.gbuffer.depth"][..]
                    ),
                    "{id}.depth"
                ),
                "normal" => assert_eq!(
                    (
                        input.ty.clone(),
                        input.format.clone(),
                        input.required,
                        input.absent_ok,
                        input.default_source
                    ),
                    (
                        raa_graph::catalog::PortType::Image2d,
                        Some(GraphFormat::Rgba16Float),
                        false,
                        false,
                        &["pass.prepass.normal", "pass.vg.gbuffer.normal"][..]
                    ),
                    "{id}.normal"
                ),
                "material" => assert_eq!(
                    (
                        input.ty.clone(),
                        input.format.clone(),
                        input.required,
                        input.absent_ok,
                        input.default_source
                    ),
                    (
                        raa_graph::catalog::PortType::Image2d,
                        Some(GraphFormat::Rgba8Unorm),
                        false,
                        false,
                        &["pass.prepass.material", "pass.vg.gbuffer.material"][..]
                    ),
                    "{id}.material"
                ),
                "src" | "base" | "hiz" | "trace" | "color" if input.required => {
                    assert!(
                        !input.absent_ok && input.default_source.is_empty(),
                        "{id}.{}",
                        input.key
                    )
                }
                "color" => assert!(
                    input.absent_ok && input.default_source.is_empty(),
                    "{id}.color"
                ),
                _ => assert!(!input.required && !input.absent_ok, "{id}.{}", input.key),
            }
        }
        assert!(
            spec.params
                .iter()
                .all(|p| !p.primary && p.binding == ParamBinding::Node)
        );
    }
    assert_eq!(
        catalog.node("ao.trace").unwrap().outputs[0].format,
        Some(GraphFormat::R32Float)
    );
    assert_eq!(
        catalog.node("ssr.trace").unwrap().outputs[0].format,
        Some(GraphFormat::Rgba16Float)
    );
}

#[test]
fn ao_graph_without_blur_plans() {
    let mut nodes = base_nodes();
    nodes.extend([
        node("aoTrace", "ao.trace"),
        r#"{"id":"aoView","type":"view.image","params":{"channels":"r"}}"#.into(),
    ]);
    let mut links = base_links();
    links.retain(|l| l != &link("tonemap.color", "out.color"));
    links.extend([
        link("aoTrace.raw", "aoView.src"),
        link("aoView.color", "out.color"),
    ]);
    let p = planned(&graph(nodes, links));
    assert!(p.nodes.iter().any(|n| n.id == "aoTrace"));
    assert!(!p.nodes.iter().any(|n| n.type_id == "ao.blur"));
    let raw = p
        .resources
        .iter()
        .find(|r| r.producer.node == "aoTrace" && r.producer.port == "raw")
        .unwrap();
    assert_eq!(raw.format, Some(GraphFormat::R32Float));
    assert_eq!(
        stage_nodes(&p),
        [(
            "aoTrace".into(),
            "ao.trace".into(),
            "ao.trace#aoTrace".into()
        )]
    );
    assert_eq!(
        view_aliases(&p),
        [("ao.trace#aoTrace".into(), "ao.trace".into())]
    );
    let entries = plan_fns::node_entries(&p, &Catalog::build());
    assert!(stage_bindings(&p, &Catalog::build(), &entries).contains(&(
        "ao.trace#aoTrace".into(),
        "depth",
        "prepass".into(),
        "depth".into()
    )));
}

fn bloom_graph(levels: usize) -> String {
    let mut nodes = base_nodes();
    nodes.extend((1..=levels).map(|i| node(&format!("bloomDown{i}"), "bloom.down")));
    nodes.extend((1..levels).map(|i| node(&format!("bloomUp{i}"), "bloom.up")));
    let mut links = base_links();
    links.push(link("bloomUp1.color", "tonemap.color"));
    links.retain(|l| l != &link("forward.color", "tonemap.color"));
    links.push(link("forward.color", "bloomDown1.src"));
    for i in 2..=levels {
        links.push(link(
            &format!("bloomDown{}.level", i - 1),
            &format!("bloomDown{i}.src"),
        ));
    }
    for i in 1..levels {
        let source = if i == levels - 1 {
            format!("bloomDown{levels}.level")
        } else {
            format!("bloomUp{}.level", i + 1)
        };
        links.push(link(&source, &format!("bloomUp{i}.src")));
        links.push(link(
            &format!("bloomDown{i}.level"),
            &format!("bloomUp{i}.base"),
        ));
    }
    links.push(link("forward.color", "bloomUp1.color"));
    graph(nodes, links)
}

#[test]
fn two_and_five_bloom_levels() {
    for levels in [2, 5] {
        let p = planned(&bloom_graph(levels));
        assert_eq!(
            p.nodes.iter().filter(|n| n.type_id == "bloom.down").count(),
            levels
        );
        assert_eq!(
            p.nodes.iter().filter(|n| n.type_id == "bloom.up").count(),
            levels - 1
        );
        assert_eq!(karis_nodes(&p), ["bloomDown1"]);
        let entries = plan_fns::node_entries(&p, &Catalog::build());
        assert_eq!(
            bloom_composites(&p, &entries),
            [("bloom.up#bloomUp1".into(), "forward".into(), "color".into())]
        );
        let bindings = stage_bindings(&p, &Catalog::build(), &entries);
        let expected = if levels == 2 {
            "bloom.down#bloomDown2"
        } else {
            "bloom.up#bloomUp2"
        };
        assert!(bindings.contains(&(
            "bloom.up#bloomUp1".into(),
            "src",
            expected.into(),
            "level".into()
        )));
        if levels == 2 {
            assert!(bindings.contains(&(
                "bloom.up#bloomUp1".into(),
                "base",
                "bloom.down#bloomDown1".into(),
                "level".into()
            )));
        }
    }
}

#[test]
fn ssr_stage_chain_plans() {
    let mut nodes = base_nodes();
    nodes.extend([
        node("ssrHiz", "ssr.hiz"),
        node("ssrTrace", "ssr.trace"),
        node("ssrComposite", "ssr.composite"),
    ]);
    let mut links = base_links();
    links.retain(|l| l != &link("forward.color", "tonemap.color"));
    links.extend([
        link("forward.color", "ssrComposite.color"),
        link("ssrComposite.color", "tonemap.color"),
        link("ssrHiz.hiz", "ssrTrace.hiz"),
        link("ssrTrace.trace", "ssrComposite.trace"),
    ]);
    let p = planned(&graph(nodes, links));
    let catalog = Catalog::build();
    let chain = plan_fns::chain_resources(&p, &catalog);
    assert!(
        chain
            .iter()
            .any(|r| r.producer.node == "ssrComposite" && r.producer.port == "color")
    );
    let entries = plan_fns::node_entries(&p, &catalog);
    let bindings = stage_bindings(&p, &catalog, &entries);
    assert!(bindings.contains(&(
        "ssr.trace#ssrTrace".into(),
        "hiz",
        "ssr.hiz#ssrHiz".into(),
        "hiz".into()
    )));
    assert!(!bindings.iter().any(|(_, port, _, _)| *port == "color"));
}

#[test]
fn stage_values_maps_node_params() {
    let p = planned(&graph(
        {
            let mut n = base_nodes();
            n.push(node("aoTrace", "ao.trace"));
            n.push(r#"{"id":"aoView","type":"view.image","params":{"channels":"r"}}"#.into());
            n
        },
        vec![
            link("forward.color", "tonemap.color"),
            link("aoTrace.raw", "aoView.src"),
            link("aoView.color", "out.color"),
        ],
    ));
    let node = p.nodes.iter().find(|n| n.id == "aoTrace").unwrap();
    let values = stage_values(node, Catalog::build().node("ao.trace").unwrap());
    assert_eq!(
        values
            .iter()
            .map(|(key, _)| key.as_str())
            .collect::<Vec<_>>(),
        ["radius", "intensity", "power", "slices", "steps"]
    );
    let spec = Catalog::build();
    let defaults = spec
        .node("ao.trace")
        .unwrap()
        .params
        .iter()
        .map(|p| {
            (
                p.key.to_owned(),
                match p.default {
                    ParamValue::Number(v) => v,
                    ParamValue::Int(v) => v as f64,
                    _ => 0.0,
                },
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(values, defaults);
}
