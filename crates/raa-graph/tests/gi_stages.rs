#[allow(dead_code)]
#[path = "../src/exec_ext/wiring/plan_fns.rs"]
mod plan_fns;

use raa_graph::Catalog;
use raa_graph::plan::{
    GraphPlan, InputSource, PlanHistory, PlanNode, PlanPortRef, PlanResource, plan_from_json,
};
use raa_pipeline::{PassInfo, PassRole};

fn full_gi_stages() -> &'static str {
    raa_graph::registry::STARTERS
        .iter()
        .find(|(_, stem, _)| *stem == "full-gi-stages")
        .map(|(_, _, json)| *json)
        .expect("generated Full GI stages starter")
}

fn starter_plan() -> GraphPlan {
    plan_from_json(full_gi_stages()).expect("Full GI stages plans")
}

fn node<'a>(plan: &'a GraphPlan, id: &str) -> &'a PlanNode {
    plan.nodes.iter().find(|n| n.id == id).unwrap()
}

fn order(plan: &GraphPlan, id: &str) -> u32 {
    node(plan, id).order_index
}

fn compact_graph(spatial: bool, atrous: &[u32]) -> String {
    let mut nodes = String::from(
        r#"{"id":"prepass","type":"pass.prepass"},
        {"id":"giVolume","type":"pass.gi.volume"},
        {"id":"giPyramid","type":"gi.pyramid"},
        {"id":"giProbes","type":"gi.probes"},
        {"id":"giAccumulate","type":"gi.accumulate"},
        {"id":"giResolve","type":"gi.resolve"},
        {"id":"forward","type":"pass.forward"},
        {"id":"tonemap","type":"pass.tonemap"},
        {"id":"out","type":"output.display"}"#,
    );
    for (i, step) in atrous.iter().enumerate() {
        nodes.push_str(&format!(
            r#",{{"id":"filter{i}","type":"gi.atrous","params":{{"step":{step}}}}}"#
        ));
    }
    let mut links = String::from(
        r#"{"from":"giPyramid.pyramid","to":"giProbes.pyramid"},
        {"from":"giProbes.probesFiltered","to":"giAccumulate.probes"},
        {"from":"giAccumulate.output","to":"giAccumulate.history","history":true},
        {"from":"giAccumulate.meta","to":"giAccumulate.historyMeta","history":true}"#,
    );
    let mut previous = "giAccumulate.filter".to_string();
    for (i, _) in atrous.iter().enumerate() {
        links.push_str(&format!(r#",{{"from":"{previous}","to":"filter{i}.src"}}"#));
        previous = format!("filter{i}.dst");
    }
    links.push_str(&format!(
        r#",{{"from":"{previous}","to":"giResolve.irradiance"}}"#
    ));
    if spatial {
        links.push_str(
            r#",{"from":"giPyramid.pyramid","to":"restirInitial.pyramid"},
            {"from":"restirInitial.reservoirs","to":"restirTemporal.initial"},
            {"from":"restirTemporal.irradiance","to":"giAccumulate.irradiance"}"#,
        );
        nodes.push_str(
            r#",{"id":"restirInitial","type":"gi.restir.initial"},
            {"id":"restirTemporal","type":"gi.restir.temporal"}"#,
        );
    }
    links.push_str(
        r#",{"from":"giResolve.gi","to":"forward.gi"},
        {"from":"forward.color","to":"tonemap.color"},
        {"from":"tonemap.color","to":"out.color"}"#,
    );
    format!(
        r#"{{"format":"raa.render-graph","version":1,"name":"GI stage test","nodes":[{nodes}],"links":[{links}]}}"#
    )
}

#[test]
fn stage_catalog_and_history_inputs() {
    let catalog = Catalog::build();
    let ids = [
        "gi.pyramid",
        "gi.probes",
        "gi.restir.initial",
        "gi.restir.temporal",
        "gi.restir.spatial",
        "gi.accumulate",
        "gi.atrous",
        "gi.resolve",
        "area.trace",
        "area.temporal",
        "area.spatial",
        "area.restir.initial",
        "area.restir.temporal",
        "area.restir.spatial",
        "area.mask",
        "ssgi.trace",
        "ssgi.resolve",
    ];
    let stages: Vec<_> = catalog
        .nodes
        .iter()
        .filter(|n| ids.contains(&n.id))
        .collect();
    assert_eq!(stages.len(), 17);
    assert_eq!(stages.iter().filter(|n| n.family == "gi_stages").count(), 8);
    assert_eq!(
        stages.iter().filter(|n| n.family == "area_stages").count(),
        7
    );
    assert_eq!(
        stages.iter().filter(|n| n.family == "ssgi_stages").count(),
        2
    );
    for spec in &stages {
        assert_eq!(spec.pass, None, "{}", spec.id);
        assert!(spec.multi_instance, "{}", spec.id);
    }
    let mut histories: Vec<_> = stages
        .iter()
        .flat_map(|n| &n.inputs)
        .filter(|p| p.history)
        .map(|p| p.key)
        .collect();
    histories.sort();
    assert_eq!(
        histories,
        [
            "depthHistory",
            "history",
            "history",
            "history",
            "historyMeta"
        ]
    );
}

#[test]
fn full_gi_stages_plan_and_history_wiring() {
    let plan = starter_plan();
    assert!(!plan.nodes.iter().any(|n| {
        ["pass.gi.screen", "pass.gi.area-shadow", "pass.gi.ssgi"].contains(&n.type_id.as_str())
    }));
    let gi_order = [
        "giPyramid",
        "giProbes",
        "giAccumulate",
        "giAtrous1",
        "giAtrous2",
        "giAtrous3",
        "giAtrous4",
        "giResolve",
        "ssgiResolve",
        "forward",
    ];
    assert!(
        gi_order
            .windows(2)
            .all(|w| order(&plan, w[0]) < order(&plan, w[1]))
    );
    let area_order = [
        "giPyramid",
        "areaTrace",
        "areaTemporal",
        "areaSpatial",
        "areaMask",
        "forward",
    ];
    assert!(
        area_order
            .windows(2)
            .all(|w| order(&plan, w[0]) < order(&plan, w[1]))
    );
    assert_eq!(
        node(&plan, "forward")
            .inputs
            .iter()
            .find(|i| i.port == "gi")
            .unwrap()
            .source,
        InputSource::Link {
            node: "ssgiResolve".into(),
            port: "ssgi".into(),
            history: false
        }
    );
    assert_eq!(
        node(&plan, "forward")
            .inputs
            .iter()
            .find(|i| i.port == "areaShadow")
            .unwrap()
            .source,
        InputSource::Link {
            node: "areaMask".into(),
            port: "areaShadow".into(),
            history: false
        }
    );
    let consumers: Vec<_> = plan
        .histories
        .iter()
        .map(|h| (h.consumer.node.as_str(), h.consumer.port.as_str()))
        .collect();
    assert_eq!(
        consumers,
        [
            ("giAccumulate", "history"),
            ("giAccumulate", "historyMeta"),
            ("areaTemporal", "history"),
            ("areaTemporal", "depthHistory"),
            ("ssgiResolve", "history")
        ]
    );
    assert!(plan_fns::history_refusals(&plan, &Catalog::build()).is_empty());
    let changed = full_gi_stages().replacen(
        "\"from\": \"giAccumulate.output\",\n      \"to\": \"giAccumulate.history\"",
        "\"from\": \"giAtrous1.dst\",\n      \"to\": \"giAccumulate.history\"",
        1,
    );
    let refused = plan_from_json(&changed).unwrap();
    assert_eq!(
        plan_fns::history_refusals(&refused, &Catalog::build()),
        vec![("giAccumulate".into(), "history".into())]
    );

    let entries = plan_fns::node_entries(&plan, &Catalog::build());
    let names = [
        "prepass",
        "gi.pyramid#giPyramid",
        "gi.probes#giProbes",
        "gi.accumulate#giAccumulate",
        "gi.atrous#giAtrous1",
        "gi.atrous#giAtrous2",
        "gi.atrous#giAtrous3",
        "gi.atrous#giAtrous4",
        "gi.resolve#giResolve",
        "forward",
        "tonemap",
    ];
    let installed: Vec<_> = names
        .iter()
        .enumerate()
        .map(|(i, name)| PassInfo {
            name: (*name).into(),
            role: PassRole::Pass,
            enabled: true,
            order: if i == 0 {
                30
            } else if i == 9 {
                60
            } else if i == 10 {
                200
            } else {
                42
            },
            required: false,
        })
        .collect();
    assert_eq!(
        plan_fns::link_order(&plan, &entries, &installed),
        names.map(str::to_string)
    );
}

#[test]
fn spatial_stage_can_be_skipped() {
    let plan = plan_from_json(&compact_graph(true, &[])).unwrap();
    assert!(!plan.nodes.iter().any(|n| n.type_id == "gi.restir.spatial"));
    assert_eq!(
        node(&plan, "giAccumulate")
            .inputs
            .iter()
            .find(|i| i.port == "irradiance")
            .unwrap()
            .source,
        InputSource::Link {
            node: "restirTemporal".into(),
            port: "irradiance".into(),
            history: false
        }
    );
    assert!(order(&plan, "restirTemporal") < order(&plan, "giAccumulate"));
}

#[test]
fn two_and_four_atrous_stages_form_ordered_chains() {
    for steps in [&[1, 2][..], &[1, 2, 4, 8][..]] {
        let plan = plan_from_json(&compact_graph(false, steps)).unwrap();
        let filters: Vec<_> = plan
            .nodes
            .iter()
            .filter(|n| n.type_id == "gi.atrous")
            .collect();
        assert_eq!(filters.len(), steps.len());
        assert!(
            filters
                .windows(2)
                .all(|w| w[0].order_index < w[1].order_index)
        );
        let actual: Vec<_> = filters
            .iter()
            .map(|n| {
                n.params
                    .iter()
                    .find(|(k, _)| k == "step")
                    .unwrap()
                    .1
                    .clone()
            })
            .collect();
        assert_eq!(
            actual,
            steps
                .iter()
                .map(|v| raa_graph::catalog::ParamValue::Int(i64::from(*v)))
                .collect::<Vec<_>>()
        );
        let last = filters.last().unwrap();
        let input = node(&plan, "giResolve")
            .inputs
            .iter()
            .find(|i| i.port == "irradiance")
            .unwrap();
        assert_eq!(
            input.source,
            InputSource::Link {
                node: last.id.clone(),
                port: "dst".into(),
                history: false
            }
        );
    }
}

/// The monolithic GI passes that the stage nodes of Full GI stages replace.
const MONOLITHIC: [&str; 3] = ["gi.screen", "gi.area-shadow", "gi.ssgi"];

#[test]
fn full_gi_stages_plans_to_stage_passes_without_monolithic_gi() {
    let plan = starter_plan();
    let catalog = Catalog::build();
    let stages: Vec<_> = plan
        .nodes
        .iter()
        .filter_map(|n| {
            raa_gi::stages::GiStageKind::from_type_id(&n.type_id)
                .map(|kind| raa_gi::stages::stage_entry(kind, &n.id))
        })
        .collect();
    for entry in [
        "gi.pyramid#giPyramid",
        "gi.probes#giProbes",
        "gi.accumulate#giAccumulate",
        "gi.resolve#giResolve",
        "area.trace#areaTrace",
        "area.temporal#areaTemporal",
        "area.spatial#areaSpatial",
        "area.mask#areaMask",
        "ssgi.trace#ssgiTrace",
        "ssgi.resolve#ssgiResolve",
    ] {
        assert!(stages.iter().any(|s| s == entry), "missing stage {entry}");
    }
    for monolithic in MONOLITHIC {
        let type_id = format!("pass.{monolithic}");
        assert!(
            plan.nodes.iter().all(|n| n.type_id != type_id),
            "{type_id} is planned"
        );
    }
    // The engine installs the monolithic passes: the plan disables each of them.
    let installed: Vec<_> = [
        "prepass",
        "gi.screen",
        "gi.area-shadow",
        "gi.ssgi",
        "forward",
    ]
    .iter()
    .zip([30, 42, 43, 44, 60])
    .map(|(name, order)| PassInfo {
        name: (*name).into(),
        role: PassRole::Pass,
        enabled: true,
        order,
        required: false,
    })
    .collect();
    let instructions = raa_graph::exec::pipeline_instructions(&plan, &catalog, &installed);
    for monolithic in MONOLITHIC {
        assert!(
            instructions
                .enable
                .iter()
                .any(|(name, on)| name == monolithic && !on),
            "{monolithic} stays enabled: {:?}",
            instructions.enable
        );
    }
}

/// The executor applies Full GI stages on an engine with the lit renderer and the GI feature:
/// every stage installs and takes its values (area.mask's AreaShadows settings go to the
/// settings rank, not to the stage), and the monolithic GI passes end disabled.
#[test]
fn full_gi_stages_applies_with_stage_passes_only() {
    use raa_graph::exec::{FeatureHandles, GraphExecutor, HandleSet};
    use std::{cell::RefCell, rc::Rc};
    let name = "full_gi_stages_applies_with_stage_passes_only";
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
    let gi = raa_gi::feature::GiFeature::install(&gpu, &pipeline, &mut lit, None, None)
        .expect("GI feature");
    let mut handles = HandleSet {
        gi: Some(gi),
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
    let mut executor = GraphExecutor::new();
    let result = executor.set_plan(starter_plan(), None, &mut h);
    assert!(result.ok, "refused: {:?}", result.issues);
    let p = pipeline.borrow();
    let enabled: Vec<String> = p
        .passes()
        .into_iter()
        .filter(|pass| p.entry_enabled(&pass.name).unwrap_or(false))
        .map(|pass| pass.name)
        .collect();
    for entry in [
        "gi.pyramid#giPyramid",
        "area.mask#areaMask",
        "ssgi.resolve#ssgiResolve",
    ] {
        assert!(enabled.iter().any(|e| e == entry), "{entry}: {enabled:?}");
    }
    for monolithic in MONOLITHIC {
        assert!(
            enabled.iter().all(|e| e != monolithic),
            "{monolithic}: {enabled:?}"
        );
    }
}

/// Records the page's Full GI stage graph for three submitted frames at the per-half capture
/// size. The graph has no mesh pass that initializes these stage inputs, so the lit pipeline's
/// prepass clears depth to 1.0 (far) and clears the HDR/G-buffer targets before any draws.
#[test]
fn full_gi_stages_renders_frames_without_panic() {
    use raa_core::{FrameGraph, MaterialId, OffscreenTarget};
    use raa_geom::builtin::DebugCube;
    use raa_geom::{GeometryProvider, MeshData, ParamValue, Params, Quality};
    use raa_lit::{AttributedMesh, LitOptions, LitRenderer, LocalLight, LocalLightKind};
    use raa_scene::{RenderList, Scene, Transform};
    use std::{cell::RefCell, rc::Rc};

    const SIZE: (u32, u32) = (1280, 720);
    const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    let name = "full_gi_stages_renders_frames_without_panic";
    let Some(gpu) = raa_core::testing::software_gpu() else {
        raa_core::testing::skip(name, "no software adapter");
        return;
    };
    let pipeline = Rc::new(RefCell::new(raa_pipeline::Pipeline::new(
        &gpu, SURFACE, false,
    )));
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
    let mut mesh = MeshData::new();
    DebugCube
        .generate(
            &Params::new().with("half", ParamValue::Scalar(0.5)),
            Quality::LOD0,
            &mut mesh,
        )
        .expect("debug cube");
    let asset = lit
        .upload(&gpu, &AttributedMesh::from_mesh(mesh))
        .expect("upload cube");
    let mesh_id = lit.submesh_ids(asset).expect("cube submesh")[0];
    let mut scene = Scene::with_capacity(1, 1);
    scene.camera.set_viewport(SIZE.0, SIZE.1);
    let cube_node = scene
        .add_node(None, Transform::IDENTITY)
        .expect("cube node");
    scene
        .add_instance(cube_node, mesh_id, MaterialId(0))
        .expect("cube instance");
    // Scene::default supplies the page's directional sun; install its overhead area light too.
    lit.local_lights_mut()
        .add(LocalLight {
            kind: LocalLightKind::Area,
            position: [0.0, 3.0, 0.0],
            direction: [0.0, -1.0, 0.0],
            intensity: 6.0,
            range: 8.0,
            size: [2.0, 1.0],
            ..LocalLight::default()
        })
        .expect("area light");
    let gi = raa_gi::feature::GiFeature::install(&gpu, &pipeline, &mut lit, None, None)
        .expect("GI feature");
    let gi_frame = gi.clone();
    let mut handles = raa_graph::exec::HandleSet {
        gi: Some(gi),
        ..Default::default()
    };
    let mut executor = raa_graph::exec::GraphExecutor::new();
    let applied = {
        let mut h = raa_graph::exec::FeatureHandles {
            gpu: &gpu,
            pipeline: &pipeline,
            handles: &mut handles,
            viewports: 1,
            msaa: 1,
            lit: Some(&mut lit),
        };
        executor.set_plan(starter_plan(), None, &mut h)
    };
    assert!(applied.ok, "refused: {:?}", applied.issues);
    let mut graph = FrameGraph::new(SURFACE);
    raa_pipeline::Pipeline::install(&pipeline, &mut graph);
    graph.compile().expect("compile pipeline graph");
    let target = OffscreenTarget::new(&gpu, SIZE.0, SIZE.1, SURFACE);
    let mut list = RenderList::with_capacity(1);
    let mut tags = Vec::new();
    let mut slots = Vec::new();
    let mut recorded = Vec::new();
    for frame in 0..3 {
        {
            let mut h = raa_graph::exec::FeatureHandles {
                gpu: &gpu,
                pipeline: &pipeline,
                handles: &mut handles,
                viewports: 1,
                msaa: 1,
                lit: Some(&mut lit),
            };
            executor.frame(&mut h).expect("executor frame");
        }
        gi_frame.set_view(&scene.camera, SIZE, 1);
        gi_frame.set_sun(&scene.light);
        gi_frame.set_local_light_count(1);
        let gi_stats = gi_frame.prepare_frame(&gpu, &mut scene, &mut lit);
        assert!(gi_stats.active, "frame {frame}: GI inactive: {gi_stats:?}");
        lit.prepare(&gpu, &mut scene, &mut list, &mut tags, &mut slots)
            .expect("prepare lit scene");
        let commands = graph
            .execute(&gpu, target.view(), target.size())
            .expect("execute frame graph");
        gpu.queue().submit([commands]);
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll submitted frame");
        gi_frame.observe_submission(frame + 1, true);
        let stats = executor.stats();
        for (node_id, type_id) in [
            ("giPyramid", "gi.pyramid"),
            ("giProbes", "gi.probes"),
            ("giAccumulate", "gi.accumulate"),
            ("giAtrous1", "gi.atrous"),
            ("giAtrous2", "gi.atrous"),
            ("giAtrous3", "gi.atrous"),
            ("giAtrous4", "gi.atrous"),
            ("giResolve", "gi.resolve"),
            ("areaTrace", "area.trace"),
            ("areaTemporal", "area.temporal"),
            ("areaSpatial", "area.spatial"),
            ("areaMask", "area.mask"),
            ("ssgiTrace", "ssgi.trace"),
            ("ssgiResolve", "ssgi.resolve"),
        ] {
            let stage = stats
                .iter()
                .find(|s| s.node == node_id && s.type_id == type_id);
            assert!(
                stage.is_some(),
                "frame {frame}: missing stage stats for {node_id} ({type_id})"
            );
            assert!(
                stage.unwrap().ran,
                "frame {frame}: {node_id} did not record"
            );
            if !recorded.contains(&node_id) {
                recorded.push(node_id);
            }
        }
    }
    assert_eq!(recorded.len(), 14, "recorded stages: {recorded:?}");
}

/// Exercises all six ReSTIR stage recorders on the same submitted-frame harness as Full GI.
#[test]
fn restir_stages_render_frames_without_panic() {
    use raa_core::{FrameGraph, MaterialId, OffscreenTarget};
    use raa_geom::builtin::DebugCube;
    use raa_geom::{GeometryProvider, MeshData, ParamValue, Params, Quality};
    use raa_lit::{
        AttributedMesh, LitOptions, LitRenderer, LocalLight, LocalLightKind, StochasticMode,
        StochasticShadowSettings, install_local_lights,
    };
    use raa_scene::{RenderList, Scene, Transform};
    use std::{cell::RefCell, rc::Rc};

    const SIZE: (u32, u32) = (1280, 720);
    const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    const GRAPH: &str = r#"{"format":"raa.render-graph","version":1,"name":"ReSTIR stage frame test","nodes":[{"id":"prepass","type":"pass.prepass"},{"id":"lightTiles","type":"pass.light.tiles"},{"id":"giVolume","type":"pass.gi.volume"},{"id":"giPyramid","type":"gi.pyramid"},{"id":"giInitial","type":"gi.restir.initial"},{"id":"giTemporal","type":"gi.restir.temporal"},{"id":"giSpatial","type":"gi.restir.spatial"},{"id":"giAccumulate","type":"gi.accumulate"},{"id":"forward","type":"pass.forward"},{"id":"tonemap","type":"pass.tonemap"},{"id":"out","type":"output.display"},{"id":"areaInitial","type":"area.restir.initial"},{"id":"areaTemporal","type":"area.restir.temporal"},{"id":"areaSpatial","type":"area.restir.spatial"},{"id":"areaMask","type":"area.mask"}],"links":[{"from":"prepass.depth","to":"lightTiles.depth"},{"from":"giPyramid.pyramid","to":"giInitial.pyramid"},{"from":"giInitial.reservoirs","to":"giTemporal.initial"},{"from":"giTemporal.reservoirs","to":"giSpatial.reservoirs"},{"from":"giSpatial.irradiance","to":"giAccumulate.irradiance"},{"from":"giAccumulate.output","to":"giAccumulate.history","history":true},{"from":"giAccumulate.meta","to":"giAccumulate.historyMeta","history":true},{"from":"giAccumulate.output","to":"forward.gi"},{"from":"areaInitial.reservoirs","to":"areaTemporal.initial"},{"from":"areaTemporal.reservoirs","to":"areaSpatial.reservoirs"},{"from":"areaSpatial.raw","to":"areaMask.src"},{"from":"areaMask.areaShadow","to":"forward.areaShadow"},{"from":"forward.color","to":"tonemap.color"},{"from":"tonemap.color","to":"out.color"}]}"#;
    let name = "restir_stages_render_frames_without_panic";
    let Some(gpu) = raa_core::testing::software_gpu() else {
        raa_core::testing::skip(name, "no software adapter");
        return;
    };
    let pipeline = Rc::new(RefCell::new(raa_pipeline::Pipeline::new(
        &gpu, SURFACE, false,
    )));
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
    let mut mesh = MeshData::new();
    DebugCube
        .generate(
            &Params::new().with("half", ParamValue::Scalar(0.5)),
            Quality::LOD0,
            &mut mesh,
        )
        .expect("debug cube");
    let asset = lit
        .upload(&gpu, &AttributedMesh::from_mesh(mesh))
        .expect("upload cube");
    let mesh_id = lit.submesh_ids(asset).expect("cube submesh")[0];
    let mut scene = Scene::with_capacity(1, 1);
    scene.camera.set_viewport(SIZE.0, SIZE.1);
    let cube = scene
        .add_node(None, Transform::IDENTITY)
        .expect("cube node");
    scene
        .add_instance(cube, mesh_id, MaterialId(0))
        .expect("cube instance");
    lit.local_lights_mut()
        .add(LocalLight {
            kind: LocalLightKind::Area,
            position: [0.0, 3.0, 0.0],
            direction: [0.0, -1.0, 0.0],
            intensity: 6.0,
            range: 8.0,
            size: [2.0, 1.0],
            ..LocalLight::default()
        })
        .expect("area light");
    install_local_lights(&gpu, &pipeline, &mut lit).expect("light tiling installs");
    let gi = raa_gi::feature::GiFeature::install(&gpu, &pipeline, &mut lit, None, None)
        .expect("GI feature");
    let gi_frame = gi.clone();
    let mut handles = raa_graph::exec::HandleSet {
        gi: Some(gi),
        ..Default::default()
    };
    let mut executor = raa_graph::exec::GraphExecutor::new();
    let plan = plan_from_json(GRAPH).expect("ReSTIR graph plans");
    let applied = {
        let mut h = raa_graph::exec::FeatureHandles {
            gpu: &gpu,
            pipeline: &pipeline,
            handles: &mut handles,
            viewports: 1,
            msaa: 1,
            lit: Some(&mut lit),
        };
        executor.set_plan(plan, None, &mut h)
    };
    assert!(applied.ok, "refused: {:?}", applied.issues);
    let mut graph = FrameGraph::new(SURFACE);
    raa_pipeline::Pipeline::install(&pipeline, &mut graph);
    graph.compile().expect("compile pipeline graph");
    let target = OffscreenTarget::new(&gpu, SIZE.0, SIZE.1, SURFACE);
    let mut list = RenderList::with_capacity(1);
    let mut tags = Vec::new();
    let mut slots = Vec::new();
    let expected = [
        ("giInitial", "gi.restir.initial"),
        ("giTemporal", "gi.restir.temporal"),
        ("giSpatial", "gi.restir.spatial"),
        ("areaInitial", "area.restir.initial"),
        ("areaTemporal", "area.restir.temporal"),
        ("areaSpatial", "area.restir.spatial"),
    ];
    let mut recorded = Vec::new();
    for frame in 0..3 {
        {
            let mut h = raa_graph::exec::FeatureHandles {
                gpu: &gpu,
                pipeline: &pipeline,
                handles: &mut handles,
                viewports: 1,
                msaa: 1,
                lit: Some(&mut lit),
            };
            executor.frame(&mut h).expect("executor frame");
            lit.set_area_stochastic(StochasticShadowSettings {
                mode: StochasticMode::Restir,
                ..StochasticShadowSettings::standard()
            })
            .expect("ReSTIR area mode");
        }
        gi_frame.set_view(&scene.camera, SIZE, 1);
        gi_frame.set_sun(&scene.light);
        gi_frame.set_local_light_count(1);
        let frame_stats = gi_frame.prepare_frame(&gpu, &mut scene, &mut lit);
        assert!(
            frame_stats.active,
            "frame {frame}: GI inactive: {frame_stats:?}"
        );
        lit.prepare(&gpu, &mut scene, &mut list, &mut tags, &mut slots)
            .expect("prepare lit scene");
        assert!(
            lit.handle().area_stochastic().active > 0,
            "area slots not selected: {:?}",
            lit.handle().area_stochastic()
        );
        assert_eq!(lit.handle().area_stochastic().mode, StochasticMode::Restir);
        let commands = graph
            .execute(&gpu, target.view(), target.size())
            .expect("execute frame graph");
        gpu.queue().submit([commands]);
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll submitted frame");
        gi_frame.observe_submission(frame + 1, true);
        let stats = executor.stats();
        for (node_id, type_id) in expected {
            let stage = stats
                .iter()
                .find(|s| s.node == node_id && s.type_id == type_id);
            assert!(
                stage.is_some_and(|s| s.ran),
                "frame {frame}: {node_id} ({type_id}) not recorded"
            );
            if !recorded.contains(&node_id) {
                recorded.push(node_id);
            }
        }
    }
    assert_eq!(recorded.len(), 6, "recorded ReSTIR stages: {recorded:?}");
}
