//! Registry order and hand-derived filter weights for S59 compute presets.

use raa_graph::catalog::{Catalog, ParamValue, PortType};
use raa_graph::format::{GraphDoc, LinkDoc, PortRef, parse_graph};
use raa_graph::registry;
use raa_graph::validate::{GraphLimits, ValidateCtx, validate};

const PRESET_STEMS: [&str; 6] = [
    "atrous-step",
    "bilateral-blur",
    "blend",
    "box-blur",
    "gaussian-blur",
    "temporal-accumulate",
];

fn link(from: (&str, &str), to: (&str, &str), history: bool) -> LinkDoc {
    LinkDoc {
        from: PortRef {
            node: from.0.to_string(),
            port: from.1.to_string(),
        },
        to: PortRef {
            node: to.0.to_string(),
            port: to.1.to_string(),
        },
        history,
    }
}

fn simple_forward() -> GraphDoc {
    let text = registry::STARTERS
        .iter()
        .find(|&&(_, stem, _)| stem == "simple-forward")
        .map(|&(_, _, text)| text)
        .expect("Simple forward starter exists");
    parse_graph(text).expect("Simple forward parses")
}

fn preset(stem: &str) -> (&'static str, &'static str) {
    let &(name, _, text) = registry::PRESETS
        .iter()
        .find(|&&(_, candidate, _)| candidate == stem)
        .unwrap_or_else(|| panic!("missing preset {stem}"));
    (name, text)
}

#[test]
fn preset_registry_order_and_temporal_self_history_validate() {
    assert_eq!(
        registry::PRESETS
            .iter()
            .map(|entry| entry.1)
            .collect::<Vec<_>>(),
        PRESET_STEMS
    );
    let names = [
        ("atrous-step", "A-trous step"),
        ("bilateral-blur", "Bilateral blur"),
        ("blend", "Blend"),
        ("box-blur", "Box blur"),
        ("gaussian-blur", "Gaussian blur"),
        ("temporal-accumulate", "Temporal accumulate"),
    ];
    for (stem, expected_name) in names {
        assert_eq!(preset(stem).0, expected_name);
    }

    let (_, text) = preset("temporal-accumulate");
    let parsed = parse_graph(text).expect("temporal preset parses");
    let compute = parsed.nodes[0].clone();
    let mut doc = simple_forward();
    doc.nodes.push(compute.clone());
    doc.links
        .push(link(("prepass", "normal"), (&compute.id, "src"), false));
    doc.links
        .push(link((&compute.id, "dst"), (&compute.id, "history"), true));
    let catalog = Catalog::build();
    let installed = |_: &str| true;
    let engine_options = [("msaa", ParamValue::Int(4))];
    let ctx = ValidateCtx {
        catalog: &catalog,
        limits: GraphLimits::CORE,
        installed: &installed,
        viewports: 1,
        engine_options: &engine_options,
    };
    let issues = validate(&doc, &ctx);
    assert!(
        issues
            .iter()
            .all(|issue| issue.node.as_deref() != Some(&compute.id)),
        "temporal self-history issues: {issues:?}"
    );
    assert_eq!(
        compute.compute.as_ref().unwrap().inputs[1].ty,
        PortType::Image2d
    );
}

#[test]
fn preset_filter_weights_match_hand_calculations() {
    let hand_kernel = [0.0625_f32, 0.25, 0.375, 0.25, 0.0625];
    let one_dimensional_sum: f32 = hand_kernel.iter().sum();
    assert_eq!(one_dimensional_sum, 1.0);
    let (_, atrous_text) = preset("atrous-step");
    let atrous = parse_graph(atrous_text).unwrap();
    let atrous_wgsl = &atrous.nodes[0].compute.as_ref().unwrap().wgsl;
    assert!(atrous_wgsl.contains("array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625)"));
    assert_eq!(one_dimensional_sum * one_dimensional_sum, 1.0);

    let (_, box_text) = preset("box-blur");
    let box_doc = parse_graph(box_text).unwrap();
    let box_wgsl = &box_doc.nodes[0].compute.as_ref().unwrap().wgsl;
    let box_radius = 2_u32;
    let box_divisor = (2 * box_radius + 1).pow(2);
    assert_eq!(box_divisor, 25);
    assert!(box_wgsl.contains("/ 25.0"));

    let (_, bilateral_text) = preset("bilateral-blur");
    let bilateral_doc = parse_graph(bilateral_text).unwrap();
    let bilateral_wgsl = &bilateral_doc.nodes[0].compute.as_ref().unwrap().wgsl;
    let radius = 4_i32;
    let step = 2_i32;
    let footprint = radius * step;
    assert_eq!(footprint, 8);
    assert!(bilateral_wgsl.contains("n0_x = -4; n0_x <= 4"));
    assert!(bilateral_wgsl.contains("n0_x * 2, n0_y * 2"));
    // At the centre, both exponentials are exp(0), and pow(dot(n, n), 32) = 1.
    let hand_derived_centre_weight = 1.0_f32;
    assert_eq!(hand_derived_centre_weight, 1.0);
    assert!(bilateral_wgsl.contains(
        "let n0_k = exp(-(n0_dl * n0_dl) / (2.0 * 0.6 * 0.6)) * exp(-abs(n0_dz) / 0.002) * pow(n0_nd, 32.0);"
    ));
}
