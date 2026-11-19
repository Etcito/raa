//! Device-free compute executor helper contracts.

#[allow(dead_code)]
#[path = "../src/exec_ext/compute/plan.rs"]
mod plan;
#[allow(dead_code)]
#[path = "../src/exec_ext/wiring/plan_fns.rs"]
mod plan_fns;

use plan::{compute_inputs, compute_nodes, frame_rows, output_size, pack_params, source_entry};
use raa_graph::Catalog;
use raa_graph::catalog::{GraphFormat, ParamValue, PortType, SizeSpec};
use raa_graph::plan::{
    GraphPlan, InputSource, PlanHistory, PlanNode, PlanPortRef, PlanResource, plan_from_json,
};

#[test]
fn parameter_packing_matches_wgsl_uniform_layout() {
    let packed = pack_params(&[
        ("gain".into(), ParamValue::Number(0.5)),
        ("on".into(), ParamValue::Bool(true)),
        ("tint".into(), ParamValue::Vec3([1.0, 2.0, 3.0])),
        ("uv".into(), ParamValue::Vec2([4.0, 5.0])),
    ]);
    assert_eq!(packed.len(), 48);
    for (offset, value) in [
        (0, 0.5f32),
        (16, 1.0),
        (20, 2.0),
        (24, 3.0),
        (32, 4.0),
        (36, 5.0),
    ] {
        assert_eq!(&packed[offset..offset + 4], &value.to_le_bytes());
    }
    assert_eq!(&packed[4..8], &1u32.to_le_bytes());
    for offset in [8, 12, 28, 40, 44] {
        assert_eq!(&packed[offset..offset + 4], &[0; 4]);
    }
    assert_eq!(pack_params(&[]), vec![0; 16]);
}

#[test]
fn sizes_and_frame_rows_are_little_endian() {
    assert_eq!(
        output_size(Some(&SizeSpec::Scale(0.5)), (1280, 720)),
        (640, 360)
    );
    assert_eq!(
        output_size(Some(&SizeSpec::Scale(1.0 / 3.0)), (100, 100)),
        (33, 33)
    );
    assert_eq!(output_size(None, (100, 80)), (100, 80));
    assert_eq!(
        output_size(
            Some(&SizeSpec::Fixed {
                width: 7,
                height: 9
            }),
            (100, 80)
        ),
        (7, 9)
    );
    let rows = frame_rows((640, 360), 5, 1.5);
    assert_eq!(&rows[0..4], &640u32.to_le_bytes());
    assert_eq!(&rows[4..8], &360u32.to_le_bytes());
    assert_eq!(&rows[8..12], &(1.0f32 / 640.0).to_le_bytes());
    assert_eq!(&rows[12..16], &(1.0f32 / 360.0).to_le_bytes());
    assert_eq!(&rows[16..20], &5u32.to_le_bytes());
    assert_eq!(&rows[20..24], &1.5f32.to_le_bytes());
    assert_eq!(&rows[24..32], &[0; 8]);
}

fn compute_graph() -> GraphPlan {
    let text = r#"{"format":"raa.render-graph","version":1,"name":"Simple forward","nodes":[{"id":"prepass","type":"pass.prepass"},{"id":"forward","type":"pass.forward"},{"id":"tonemap","type":"pass.tonemap"},{"id":"out","type":"output.display"},{"id":"calc","type":"compute.wgsl","inputs":[{"key":"src","type":"image2d"}],"outputs":[{"key":"dst","type":"image2d","format":"rgba16float"}],"math":null,"wgsl":"fn rg_main(p: vec2<u32>) {}"},{"id":"show","type":"view.image"}],"links":[{"from":"prepass.normal","to":"calc.src"},{"from":"calc.dst","to":"show.src"},{"from":"show.color","to":"tonemap.color"},{"from":"tonemap.color","to":"out.color"}]}"#;
    plan_from_json(text).expect("valid compute graph")
}

#[test]
fn compute_node_inputs_and_source_entries_follow_the_plan() {
    let graph = compute_graph();
    let catalog = Catalog::build();
    let nodes = compute_nodes(&graph);
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].id, "calc");
    let inputs = compute_inputs(&graph, &catalog, nodes[0]);
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0].key, "src");
    assert_eq!(inputs[0].source, Some(("prepass".into(), "normal".into())));
    assert!(!inputs[0].history);
    assert!(inputs[0].filterable);
    assert_eq!(
        source_entry(&graph, &catalog, "calc"),
        Some("compute#calc".into())
    );
    assert_eq!(
        source_entry(&graph, &catalog, "prepass"),
        Some("prepass".into())
    );
    let stages = catalog
        .starters
        .iter()
        .find(|starter| starter.file == "full-gi-stages")
        .expect("full GI stage starter");
    let mut stages = plan_from_json(stages.json).expect("valid stage graph");
    stages
        .nodes
        .iter_mut()
        .find(|node| node.type_id == "gi.accumulate")
        .expect("accumulate stage")
        .id = "acc".into();
    assert_eq!(
        source_entry(&stages, &catalog, "acc"),
        Some("gi.accumulate#acc".into())
    );
    assert_eq!(plan::compute_entry("calc"), "compute#calc");
    assert_eq!(plan::RG_FRAME_BYTES, 368);
    assert_eq!(plan::COMPUTE_ORDER, 46);
    assert_eq!(GraphFormat::Rgba16Float.name(), "rgba16float");
}
