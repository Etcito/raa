//! Applying and clearing a graph of AO, SSR and bloom stage nodes.
#![cfg(not(target_arch = "wasm32"))]

use std::{cell::RefCell, rc::Rc};

use raa_graph::catalog::{Catalog, ParamValue};
use raa_graph::exec::{FeatureHandles, GraphExecutor, HandleSet};
use raa_graph::format::parse_graph;
use raa_graph::validate::{GraphLimits, ValidateCtx};
use raa_lit::{LitOptions, LitRenderer};
use raa_pipeline::Pipeline;

const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn node(id: &str, ty: &str) -> String {
    format!(r#"{{"id":"{id}","type":"{ty}"}}"#)
}

fn node_with_params(id: &str, ty: &str, params: &str) -> String {
    format!(r#"{{"id":"{id}","type":"{ty}","params":{{{params}}}}}"#)
}

fn link(from: &str, to: &str) -> String {
    format!(r#"{{"from":"{from}","to":"{to}"}}"#)
}

/// The N = 2 bloom graph of the plan tests with the AO trace/blur and the SSR chain before it.
fn post_stage_graph(ao_radius: Option<f64>) -> String {
    let nodes = vec![
        node("prepass", "pass.prepass"),
        node("forward", "pass.forward"),
        node("tonemap", "pass.tonemap"),
        node("out", "output.display"),
        match ao_radius {
            Some(radius) => {
                node_with_params("aoTrace", "ao.trace", &format!("\"radius\":{radius}"))
            }
            None => node("aoTrace", "ao.trace"),
        },
        node("aoBlur", "ao.blur"),
        node("ssrHiz", "ssr.hiz"),
        node("ssrTrace", "ssr.trace"),
        node("ssrComposite", "ssr.composite"),
        node("bloomDown1", "bloom.down"),
        node("bloomDown2", "bloom.down"),
        node("bloomUp1", "bloom.up"),
    ];
    let links = vec![
        link("aoTrace.raw", "aoBlur.src"),
        link("aoBlur.ao", "forward.ao"),
        link("ssrHiz.hiz", "ssrTrace.hiz"),
        link("ssrTrace.trace", "ssrComposite.trace"),
        link("forward.color", "ssrComposite.color"),
        link("ssrComposite.color", "bloomDown1.src"),
        link("ssrComposite.color", "bloomUp1.color"),
        link("bloomDown1.level", "bloomDown2.src"),
        link("bloomDown2.level", "bloomUp1.src"),
        link("bloomDown1.level", "bloomUp1.base"),
        link("bloomUp1.color", "tonemap.color"),
        link("tonemap.color", "out.color"),
    ];
    format!(
        r#"{{"format":"raa.render-graph","version":1,"name":"post stages","nodes":[{}],"links":[{}]}}"#,
        nodes.join(","),
        links.join(",")
    )
}

fn pass_names(pipeline: &Rc<RefCell<Pipeline>>) -> Vec<String> {
    pipeline
        .borrow()
        .passes()
        .into_iter()
        .map(|pass| pass.name)
        .collect()
}

#[test]
fn post_stage_graphs_apply_and_clear() {
    use raa_core::testing::{skip, software_gpu};
    let name = "post_stage_graphs_apply_and_clear";
    let Some(gpu) = software_gpu() else {
        skip(name, "no software adapter");
        return;
    };
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, SURFACE, false)));
    let mut lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit renderer");
    let gtao = raa_gtao::GtaoFeature::install(&gpu, &pipeline, &mut lit).expect("GTAO feature");
    let ssr = raa_ssr::SsrFeature::install(&gpu, &pipeline).expect("SSR feature");
    let bloom = raa_bloom::BloomFeature::install(&gpu, &pipeline).expect("bloom feature");
    let mut handles = HandleSet {
        gtao: Some(gtao),
        ssr: Some(ssr),
        bloom: Some(bloom),
        ..HandleSet::default()
    };
    let mut h = FeatureHandles {
        gpu: &gpu,
        pipeline: &pipeline,
        handles: &mut handles,
        viewports: 1,
        msaa: 4,
        lit: Some(&mut lit),
    };
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
    let mut executor = GraphExecutor::new();
    let doc = parse_graph(&post_stage_graph(None)).expect("post-stage graph parses");
    let result = executor.set_graph(doc, &ctx, &mut h);
    assert!(result.ok, "refused: {:?}", result.issues);
    let stage_passes = [
        "ao.trace#aoTrace",
        "ao.blur#aoBlur",
        "ssr.hiz#ssrHiz",
        "ssr.trace#ssrTrace",
        "ssr.composite#ssrComposite",
        "bloom.down#bloomDown1",
        "bloom.down#bloomDown2",
        "bloom.up#bloomUp1",
    ];
    let applied = pass_names(&pipeline);
    for entry in stage_passes {
        assert!(
            applied.iter().any(|name| name == entry),
            "{entry}: {applied:?}"
        );
    }
    executor.clear_graph(&mut h).expect("clear graph");
    let cleared = pass_names(&pipeline);
    for entry in stage_passes {
        assert!(
            !cleared.iter().any(|name| name == entry),
            "{entry}: {cleared:?}"
        );
    }
    let doc = parse_graph(&post_stage_graph(Some(0.0))).expect("refused graph parses");
    let refused = executor.set_graph(doc, &ctx, &mut h);
    assert!(!refused.ok, "radius 0 accepted: {:?}", refused.issues);
    assert_eq!(
        pass_names(&pipeline),
        cleared,
        "a refused graph changed the pass list"
    );
}
