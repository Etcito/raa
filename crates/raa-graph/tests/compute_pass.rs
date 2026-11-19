//! Installing and removing compute passes.
use raa_graph::{
    exec::{FeatureHandles, GraphExecutor, HandleSet},
    plan::plan_from_json,
};
use std::{cell::RefCell, rc::Rc};

#[test]
fn compute_pass_is_installed_and_removed() {
    let name = "compute_pass_is_installed_and_removed";
    let Some(gpu) = raa_core::testing::software_gpu() else {
        raa_core::testing::skip(name, "no software adapter");
        return;
    };
    let pipeline = Rc::new(RefCell::new(raa_pipeline::Pipeline::new(
        &gpu,
        wgpu::TextureFormat::Rgba8Unorm,
        false,
    )));
    let mut lit = raa_lit::LitRenderer::new(&gpu, &pipeline, raa_lit::LitOptions::default())
        .expect("lit renderer");
    let mut handles = HandleSet::default();
    let mut h = FeatureHandles {
        gpu: &gpu,
        pipeline: &pipeline,
        handles: &mut handles,
        viewports: 1,
        msaa: 4,
        lit: Some(&mut lit),
    };
    let text = r#"{"format":"raa.render-graph","version":1,"name":"Simple forward","nodes":[{"id":"prepass","type":"pass.prepass"},{"id":"forward","type":"pass.forward"},{"id":"tonemap","type":"pass.tonemap"},{"id":"out","type":"output.display"},{"id":"calc","type":"compute.wgsl","inputs":[{"key":"src","type":"image2d"}],"outputs":[{"key":"dst","type":"image2d","format":"rgba16float"}],"math":null,"wgsl":"fn rg_main(p: vec2<u32>) {}"},{"id":"show","type":"view.image"}],"links":[{"from":"prepass.normal","to":"calc.src"},{"from":"calc.dst","to":"show.src"},{"from":"show.color","to":"tonemap.color"},{"from":"tonemap.color","to":"out.color"}]}"#;
    let mut executor = GraphExecutor::new();
    let result = executor.set_plan(plan_from_json(text).expect("plan"), None, &mut h);
    assert!(result.ok, "{:?}", result.issues);
    let passes = pipeline.borrow().passes();
    let prepass = passes
        .iter()
        .position(|p| p.name == "prepass")
        .expect("prepass");
    let compute = passes
        .iter()
        .position(|p| p.name == "compute#calc" && p.enabled)
        .expect("compute");
    assert!(compute > prepass);
    let stats = executor.stats();
    let calc: Vec<_> = stats.iter().filter(|node| node.node == "calc").collect();
    assert_eq!(calc.len(), 1);
    assert!(!calc[0].ran);
    executor.clear_graph(&mut h).expect("clear");
    assert!(
        !pipeline
            .borrow()
            .passes()
            .iter()
            .any(|p| p.name == "compute#calc")
    );
}
