//! Device-free checks of the executor-owned history wiring (`history_refusals`, `history_resets`).
#[allow(dead_code)]
#[path = "../src/exec_ext/wiring/plan_fns.rs"]
mod plan_fns;
use raa_graph::Catalog;
use raa_graph::plan::{
    GraphPlan, PlanHistory, PlanNode, PlanPortRef, PlanResource, plan_from_json,
};

/// Full GI with one extra link inserted at the front of `links`.
fn full_gi_with(link: &str) -> GraphPlan {
    let text = include_str!("../graphs/full-gi.json").replacen(
        "\"links\": [",
        &format!("\"links\": [{link},"),
        1,
    );
    plan_from_json(&text).unwrap()
}

#[test]
fn taa_own_history_is_accepted() {
    let plan = full_gi_with(r#"{"from": "taa.history", "to": "taa.history", "history": true}"#);
    assert_eq!(plan_fns::history_refusals(&plan, &Catalog::build()), vec![]);
    assert_eq!(
        plan_fns::history_resets(Some(&plan), &plan),
        Vec::<String>::new()
    );
}

#[test]
fn gi_history_output_is_refused() {
    let plan =
        full_gi_with(r#"{"from": "giHistory.history", "to": "taa.history", "history": true}"#);
    assert_eq!(
        plan_fns::history_refusals(&plan, &Catalog::build()),
        vec![("taa".to_string(), "history".to_string())]
    );
}

#[test]
fn plain_link_into_history_input_is_refused() {
    let plan = full_gi_with(r#"{"from": "fog.color", "to": "taa.history"}"#);
    assert_eq!(
        plan_fns::history_refusals(&plan, &Catalog::build()),
        vec![("taa".to_string(), "history".to_string())]
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wired_taa_history_changes_only_after_submit() {
    use raa_core::{GpuContext, GpuOptions, testing::block_on};
    if !cfg!(target_os = "windows") {
        return;
    }
    for _ in 0..2 {
        let gpu = block_on(GpuContext::headless(&GpuOptions::default())).expect("Windows GPU");
        exercise_wired_history(&gpu);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn exercise_wired_history(gpu: &raa_core::GpuContext) {
    use std::{cell::RefCell, rc::Rc};

    use raa_core::{FrameGraph, OffscreenTarget};
    use raa_graph::{
        exec::{FeatureHandles, GraphExecutor, HandleSet},
        format::parse_graph,
        validate::{GraphLimits, ValidateCtx},
    };
    use raa_lit::{LitOptions, LitRenderer};
    use raa_pipeline::Pipeline;
    use raa_scene::{RenderList, Scene};

    let format = wgpu::TextureFormat::Rgba8Unorm;
    let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, format, false)));
    let mut lit = LitRenderer::new(
        gpu,
        &pipeline,
        LitOptions {
            msaa: 1,
            ..LitOptions::default()
        },
    )
    .unwrap();
    let taa = raa_taa::TaaFeature::install(gpu, &pipeline, &lit.handle().frame_buffer()).unwrap();
    let (tonemap_pass, tonemap) = raa_tonemap::TonemapPass::new(gpu, &pipeline.borrow());
    pipeline.borrow_mut().add(Box::new(tonemap_pass)).unwrap();
    let mut handles = HandleSet {
        taa: Some(taa.clone()),
        tonemap: Some(tonemap),
        ..HandleSet::default()
    };
    let catalog = Catalog::build();
    let options = [("msaa", raa_graph::catalog::ParamValue::Int(1))];
    let ctx = ValidateCtx {
        catalog: &catalog,
        limits: GraphLimits::CORE,
        installed: &|_| true,
        viewports: 1,
        engine_options: &options,
    };
    let document = |id: &str| {
        parse_graph(&format!(
            r#"{{
        "format":"raa.render-graph","version":1,"name":"wired history",
        "nodes":[{{"id":"prepass","type":"pass.prepass"}},
                 {{"id":"forward","type":"pass.forward"}},
                 {{"id":"{id}","type":"pass.taa"}},
                 {{"id":"tonemap","type":"pass.tonemap"}},
                 {{"id":"out","type":"output.display"}}],
        "links":[{{"from":"forward.color","to":"{id}.color"}},
                 {{"from":"{id}.history","to":"{id}.history","history":true}},
                 {{"from":"{id}.color","to":"tonemap.color"}},
                 {{"from":"tonemap.color","to":"out.color"}}]
    }}"#
        ))
        .unwrap()
    };
    let mut executor = GraphExecutor::new();
    let mut apply = |id, viewports, lit: &mut LitRenderer| {
        let mut h = FeatureHandles {
            gpu,
            pipeline: &pipeline,
            handles: &mut handles,
            viewports,
            msaa: 1,
            lit: Some(lit),
        };
        executor.set_graph(document(id), &ValidateCtx { viewports, ..ctx }, &mut h)
    };
    assert!(apply("taa", 1, &mut lit).ok);

    let mut graph = FrameGraph::new(format);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().unwrap();
    let mut output = OffscreenTarget::new(gpu, 16, 16, format);
    let mut scene = Scene::new();
    scene.camera.set_viewport(16, 16);
    let mut record =
        |lit: &mut LitRenderer, graph: &mut FrameGraph, output: &OffscreenTarget, cut| {
            if cut {
                lit.prepare_scene_view(0, true);
            }
            scene.camera.set_viewport(output.size().0, output.size().1);
            lit.prepare(
                gpu,
                &mut scene,
                &mut RenderList::default(),
                &mut Vec::new(),
                &mut Vec::new(),
            )
            .unwrap();
            graph.execute(gpu, output.view(), output.size()).unwrap()
        };
    let read = || {
        pipeline
            .borrow()
            .resources()
            .input("taa", "history")
            .map(|image| image.view)
    };
    let submit = |graph: &mut FrameGraph, commands| {
        gpu.queue().submit([commands]);
        graph.notify_submitted();
        graph.notify_submitted();
    };

    let first = record(&mut lit, &mut graph, &output, false);
    assert!(read().is_none());
    assert_eq!(taa.accumulated_frames(), 0);
    submit(&mut graph, first);
    assert_eq!(taa.accumulated_frames(), 1);

    drop(record(&mut lit, &mut graph, &output, false));
    let previous = read().expect("first submission is readable");
    assert_eq!(taa.accumulated_frames(), 1);
    let refused = apply("replacement", 2, &mut lit);
    assert!(
        !refused.ok
            && refused
                .issues
                .iter()
                .any(|issue| issue.code == "graph.multi_view")
    );
    let retry = record(&mut lit, &mut graph, &output, false);
    assert_eq!(read(), Some(previous.clone()));
    submit(&mut graph, retry);
    assert_eq!(taa.accumulated_frames(), 2);

    let later = record(&mut lit, &mut graph, &output, false);
    assert_ne!(read(), Some(previous));
    submit(&mut graph, later);
    lit.set_camera_jitter(Some(false));
    let cut = record(&mut lit, &mut graph, &output, true);
    assert!(read().is_none());
    assert_eq!(lit.handle().camera_jitter_ndc(), [0.0; 2]);
    submit(&mut graph, cut);
    assert!(apply("replacement", 1, &mut lit).ok);
    let swapped = record(&mut lit, &mut graph, &output, false);
    assert!(read().is_none());
    submit(&mut graph, swapped);

    output = OffscreenTarget::new(gpu, 17, 11, format);
    let resized = record(&mut lit, &mut graph, &output, false);
    assert!(read().is_none());
    submit(&mut graph, resized);
    assert_eq!(taa.accumulated_frames(), 1);
}
