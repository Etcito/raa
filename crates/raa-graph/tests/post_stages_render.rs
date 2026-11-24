//! Rendering post-stage graphs on the software GPU: the AO trace/blur branch into a view, the SSR
//! Hi-Z/trace/composite chain and five bloom levels. No pixel metric, hash or reference image.

#![cfg(not(target_arch = "wasm32"))]

use std::{cell::RefCell, rc::Rc};

use raa_graph::catalog::{Catalog, ParamValue};
use raa_graph::exec::{FeatureHandles, GraphExecutor, HandleSet};
use raa_graph::format::parse_graph;
use raa_graph::validate::{GraphLimits, ValidateCtx};
use raa_lit::{LitOptions, LitRenderer};
use raa_pipeline::Pipeline;

const SIZE: (u32, u32) = (640, 360);
const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn node(id: &str, ty: &str) -> String {
    format!(r#"{{"id":"{id}","type":"{ty}"}}"#)
}

fn link(from: &str, to: &str) -> String {
    format!(r#"{{"from":"{from}","to":"{to}"}}"#)
}

fn graph(nodes: Vec<String>, links: Vec<String>) -> String {
    format!(
        r#"{{"format":"raa.render-graph","version":1,"name":"post stages","nodes":[{}],"links":[{}]}}"#,
        nodes.join(","),
        links.join(",")
    )
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

/// The AO branch: `ao.trace` feeds `ao.blur`, whose `ao` port is Rgba16Float and reaches the
/// display through a view node, because `pass.forward.ao` takes R8Unorm. `prepass`, `forward` and
/// `tonemap` are side-effect nodes, so they stay planned while the AO branch owns the output.
fn ao_graph() -> String {
    let mut nodes = base_nodes();
    nodes.extend([
        node("aoTrace", "ao.trace"),
        node("aoBlur", "ao.blur"),
        node("aoView", "view.image"),
    ]);
    let links = vec![
        link("forward.color", "tonemap.color"),
        link("aoTrace.raw", "aoBlur.src"),
        link("aoBlur.ao", "aoView.src"),
        link("aoView.color", "out.color"),
    ];
    graph(nodes, links)
}

/// The SSR chain of `tests/post_stages.rs` test (4): Hi-Z, trace, composite, then tonemap.
fn ssr_graph() -> String {
    let mut nodes = base_nodes();
    nodes.extend([
        node("ssrHiz", "ssr.hiz"),
        node("ssrTrace", "ssr.trace"),
        node("ssrComposite", "ssr.composite"),
    ]);
    let links = vec![
        link("forward.color", "ssrComposite.color"),
        link("ssrComposite.color", "tonemap.color"),
        link("ssrHiz.hiz", "ssrTrace.hiz"),
        link("ssrTrace.trace", "ssrComposite.trace"),
        link("tonemap.color", "out.color"),
    ];
    graph(nodes, links)
}

/// The same scene with the incident image exposed before environment replacement.
fn split_ssr_graph() -> String {
    let mut nodes = base_nodes();
    nodes.extend([
        node("ssrHiz", "ssr.hiz"),
        node("ssrTrace", "ssr.trace"),
        node("ssrIncident", "ssr.incident"),
        node("ssrReplace", "ssr.replace"),
    ]);
    graph(
        nodes,
        vec![
            link("ssrHiz.hiz", "ssrTrace.hiz"),
            link("ssrTrace.trace", "ssrIncident.trace"),
            link("forward.color", "ssrIncident.color"),
            link("forward.color", "ssrReplace.color"),
            link("ssrIncident.incident", "ssrReplace.incident"),
            link("ssrReplace.color", "tonemap.color"),
            link("tonemap.color", "out.color"),
        ],
    )
}

/// The N = 5 bloom graph of `tests/post_stages.rs` test (3): five down levels and four up levels.
fn bloom_graph(levels: usize) -> String {
    let mut nodes = base_nodes();
    nodes.extend((1..=levels).map(|i| node(&format!("bloomDown{i}"), "bloom.down")));
    nodes.extend((1..levels).map(|i| node(&format!("bloomUp{i}"), "bloom.up")));
    let mut links = vec![
        link("forward.color", "tonemap.color"),
        link("forward.color", "bloomDown1.src"),
    ];
    links.retain(|l| l != &link("forward.color", "tonemap.color"));
    links.push(link("bloomUp1.color", "tonemap.color"));
    links.push(link("tonemap.color", "out.color"));
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

/// Renders `frames` submitted frames of `text` with the GTAO, SSR and bloom features installed and
/// the scene camera set on their handles, over a floor and a cube. None without a software adapter.
fn render_plan_frames(name: &str, text: &str, frames: u32) -> Option<Vec<String>> {
    use raa_core::{FrameGraph, MaterialId, OffscreenTarget};
    use raa_geom::builtin::DebugCube;
    use raa_geom::{GeometryProvider, MeshData, ParamValue as GeomValue, Params, Quality};
    use raa_lit::AttributedMesh;
    use raa_math::Vec3;
    use raa_scene::{RenderList, Scene, Transform};

    let Some(gpu) = raa_core::testing::software_gpu() else {
        raa_core::testing::skip(name, "no software adapter");
        return None;
    };
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, SURFACE, false)));
    let mut lit = LitRenderer::new(
        &gpu,
        &pipeline,
        LitOptions {
            max_instances: 16,
            max_materials: 4,
            msaa: 1,
        },
    )
    .expect("lit renderer");
    let (tonemap_pass, tonemap) = {
        let p = pipeline.borrow();
        raa_tonemap::TonemapPass::new(&gpu, &p)
    };
    pipeline
        .borrow_mut()
        .add(Box::new(tonemap_pass))
        .expect("tonemap pass");
    let gtao = raa_gtao::GtaoFeature::install(&gpu, &pipeline, &mut lit).expect("GTAO feature");
    let ssr = raa_ssr::SsrFeature::install(&gpu, &pipeline).expect("SSR feature");
    let bloom = raa_bloom::BloomFeature::install(&gpu, &pipeline).expect("bloom feature");
    let mut mesh = MeshData::new();
    DebugCube
        .generate(
            &Params::new().with("half", GeomValue::Scalar(0.5)),
            Quality::LOD0,
            &mut mesh,
        )
        .expect("debug cube");
    let asset = lit
        .upload(&gpu, &AttributedMesh::from_mesh(mesh))
        .expect("upload cube");
    let mesh_id = lit.submesh_ids(asset).expect("cube submesh")[0];
    let mut scene = Scene::with_capacity(2, 2);
    scene.camera.set_viewport(SIZE.0, SIZE.1);
    scene.camera.eye = Vec3::new(0.0, 4.0, 5.0);
    for (translation, scale) in [
        (Vec3::new(0.0, 0.5, 0.0), Vec3::splat(1.0)),
        (Vec3::new(0.0, -0.05, 0.0), Vec3::new(12.0, 0.1, 12.0)),
    ] {
        let cube = scene
            .add_node(
                None,
                Transform {
                    translation,
                    scale,
                    ..Transform::IDENTITY
                },
            )
            .expect("node");
        scene
            .add_instance(cube, mesh_id, MaterialId(0))
            .expect("instance");
    }
    let proj = scene.camera.projection();
    let inv_proj = proj.inverse().expect("invertible projection");
    gtao.set_camera(&proj, &inv_proj);
    ssr.set_camera(&scene.camera.view(), &proj);
    let mut handles = HandleSet {
        gtao: Some(gtao),
        ssr: Some(ssr),
        bloom: Some(bloom),
        tonemap: Some(tonemap),
        ..HandleSet::default()
    };
    let mut executor = GraphExecutor::new();
    let catalog = Catalog::build();
    let installed = |_: &str| true;
    let engine_options = [("msaa", ParamValue::Int(1))];
    let ctx = ValidateCtx {
        catalog: &catalog,
        limits: GraphLimits::CORE,
        installed: &installed,
        viewports: 1,
        engine_options: &engine_options,
    };
    let doc = parse_graph(text).expect("post-stage graph parses");
    let applied = {
        let mut h = FeatureHandles {
            gpu: &gpu,
            pipeline: &pipeline,
            handles: &mut handles,
            viewports: 1,
            msaa: 1,
            lit: Some(&mut lit),
        };
        executor.set_graph(doc, &ctx, &mut h)
    };
    assert!(applied.ok, "{name}: refused: {:?}", applied.issues);
    let mut frame_graph = FrameGraph::new(SURFACE);
    raa_pipeline::Pipeline::install(&pipeline, &mut frame_graph);
    frame_graph.compile().expect("compile pipeline graph");
    let target = OffscreenTarget::new(&gpu, SIZE.0, SIZE.1, SURFACE);
    let mut list = RenderList::with_capacity(2);
    let (mut tags, mut slots) = (Vec::new(), Vec::new());
    for frame in 0..frames {
        {
            let mut h = FeatureHandles {
                gpu: &gpu,
                pipeline: &pipeline,
                handles: &mut handles,
                viewports: 1,
                msaa: 1,
                lit: Some(&mut lit),
            };
            executor
                .frame(&mut h)
                .unwrap_or_else(|error| panic!("{name}: frame {frame}: {error}"));
        }
        lit.prepare(&gpu, &mut scene, &mut list, &mut tags, &mut slots)
            .expect("prepare lit scene");
        let commands = frame_graph
            .execute(&gpu, target.view(), target.size())
            .expect("execute frame graph");
        gpu.queue().submit([commands]);
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll submitted frame");
    }
    Some(
        executor
            .stats()
            .into_iter()
            .filter(|stat| stat.ran)
            .map(|stat| stat.node)
            .collect(),
    )
}

fn assert_ran(name: &str, ran: &[String], nodes: &[&str]) {
    for node in nodes {
        assert!(
            ran.iter().any(|id| id == node),
            "{name}: {node} did not record; ran: {ran:?}"
        );
    }
}

#[test]
fn post_stage_graphs_render_frames_without_panic() {
    let name = "post_stage_graphs_render_frames_without_panic";
    if let Some(ran) = render_plan_frames(name, &ao_graph(), 3) {
        assert_ran(name, &ran, &["aoTrace", "aoBlur"]);
    }
    if let Some(ran) = render_plan_frames(name, &ssr_graph(), 3) {
        assert_ran(name, &ran, &["ssrHiz", "ssrTrace", "ssrComposite"]);
    }
    if let Some(ran) = render_plan_frames(name, &split_ssr_graph(), 3) {
        assert_ran(
            name,
            &ran,
            &["ssrHiz", "ssrTrace", "ssrIncident", "ssrReplace"],
        );
    }
    if let Some(ran) = render_plan_frames(name, &bloom_graph(5), 3) {
        assert_ran(
            name,
            &ran,
            &[
                "bloomDown1",
                "bloomDown2",
                "bloomDown3",
                "bloomDown4",
                "bloomDown5",
                "bloomUp1",
                "bloomUp2",
                "bloomUp3",
                "bloomUp4",
            ],
        );
    }
}
