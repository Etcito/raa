#[allow(dead_code)]
#[path = "../src/exec_ext/wiring/plan_fns.rs"]
mod plan_fns;

use raa_graph::Catalog;
use raa_graph::plan::{
    GraphPlan, InputSource, PlanHistory, PlanNode, PlanPortRef, PlanResource, plan_from_json,
};
use raa_pipeline::{PassInfo, PassRole};

// Parallel software-GPU contexts can crash this shared adapter; serialize their lifetimes.
static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
fn area_signal_encodings_and_cadence_are_checked_during_planning() {
    use raa_graph::catalog::SignalEncoding;
    use raa_graph::format::{LinkDoc, PortRef, parse_graph, write_graph};

    let baseline = parse_graph(full_gi_stages()).unwrap();
    let mask_signal = |plan: &GraphPlan| {
        plan.input_signal(node(plan, "areaMask"), "src")
            .unwrap()
            .encoding
    };
    assert_eq!(
        mask_signal(&starter_plan()),
        SignalEncoding::PackedVisibility
    );

    let mut pairs = baseline.clone();
    let mask = pairs
        .links
        .iter_mut()
        .find(|link| link.to.node == "areaMask")
        .unwrap();
    mask.from = PortRef {
        node: "areaTrace".into(),
        port: "raw".into(),
    };
    assert_eq!(
        mask_signal(&plan_from_json(&write_graph(&pairs)).unwrap()),
        SignalEncoding::StratifiedPairs
    );

    let mut restir = pairs.clone();
    let restir_nodes = parse_graph(
        r#"{
        "format":"raa.render-graph","version":1,"name":"Area ReSTIR",
        "nodes":[
            {"id":"initial","type":"area.restir.initial"},
            {"id":"reuse","type":"area.restir.temporal"},
            {"id":"aggregate","type":"area.restir.spatial"}
        ],
        "links":[
            {"from":"initial.reservoirs","to":"reuse.initial"},
            {"from":"reuse.reservoirs","to":"aggregate.reservoirs"}
        ]
    }"#,
    )
    .unwrap();
    restir.nodes.extend(restir_nodes.nodes);
    restir.links.extend(restir_nodes.links);
    let aggregate = PortRef {
        node: "aggregate".into(),
        port: "raw".into(),
    };
    restir
        .links
        .iter_mut()
        .find(|link| link.to.node == "areaMask")
        .unwrap()
        .from = aggregate.clone();
    assert_eq!(
        mask_signal(&plan_from_json(&write_graph(&restir)).unwrap()),
        SignalEncoding::RestirAggregate
    );

    let temporal_raw = PortRef {
        node: "areaTemporal".into(),
        port: "raw".into(),
    };
    for producer in [
        aggregate,
        PortRef {
            node: "areaSpatial".into(),
            port: "mask".into(),
        },
    ] {
        let mut wrong = restir.clone();
        wrong
            .links
            .iter_mut()
            .find(|link| link.to == temporal_raw)
            .unwrap()
            .from = producer;
        let issues = plan_from_json(&write_graph(&wrong)).unwrap_err();
        assert!(issues.iter().any(|issue| issue.code == "graph.link_type"
            && issue.node.as_deref() == Some("areaTemporal")
            && issue.port.as_deref() == Some("raw")
            && issue.message.contains("stratifiedPairs v1")));
    }

    let history = baseline
        .links
        .iter()
        .find(|link| link.to.node == "areaTemporal" && link.to.port == "history")
        .unwrap();
    let mut current_history = baseline.clone();
    current_history.links.retain(|link| link.to != history.to);
    current_history.links.push(LinkDoc {
        from: PortRef {
            node: "areaTrace".into(),
            port: "raw".into(),
        },
        to: history.to.clone(),
        history: false,
    });
    let issues = plan_from_json(&write_graph(&current_history)).unwrap_err();
    assert!(
        issues.iter().any(
            |issue| issue.code == "graph.link_type" && issue.port.as_deref() == Some("history")
        )
    );
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
    let _gpu_guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
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
    let _gpu_guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
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
    let _gpu_guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
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
    let before = executor.active_plan().unwrap().clone();
    let invalid = GRAPH.replace(
        r#"{"from":"areaMask.areaShadow","to":"forward.areaShadow"}"#,
        r#"{"from":"areaSpatial.raw","to":"forward.areaShadow"}"#,
    );
    assert_ne!(invalid, GRAPH);
    let catalog = Catalog::build();
    let result = executor.set_graph(
        raa_graph::format::parse_graph(&invalid).unwrap(),
        &raa_graph::validate::ValidateCtx {
            catalog: &catalog,
            limits: raa_graph::validate::GraphLimits::CORE,
            installed: &|_| true,
            viewports: 1,
            engine_options: &[],
        },
        &mut raa_graph::exec::FeatureHandles {
            gpu: &gpu,
            pipeline: &pipeline,
            handles: &mut handles,
            viewports: 1,
            msaa: 1,
            lit: Some(&mut lit),
        },
    );
    assert!(!result.ok);
    assert!(result.issues.iter().any(
        |issue| issue.code == "graph.link_type" && issue.port.as_deref() == Some("areaShadow")
    ));
    assert_eq!(executor.active_plan(), Some(&before));
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
        ("areaMask", "area.mask"),
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
    assert_eq!(
        recorded.len(),
        expected.len(),
        "recorded ReSTIR stages: {recorded:?}"
    );
}

/// Post nodes whose features these tests do not install: the colour chain skips them.
const UNINSTALLED_POST: [&str; 4] = ["ssr", "fog", "taa", "bloom"];

/// Full GI stages with the nodes in `remove` dropped (and every link touching them) and the
/// links `(from, to)` added: the same edit the gi-stages scene applies to its variants. The
/// colour chain is cut after gi.history so tonemap reads an installed pass.
fn variant_plan(remove: &[&str], add: &[(&str, &str)]) -> GraphPlan {
    use raa_graph::format::{LinkDoc, PortRef, parse_graph, write_graph};
    let mut doc = parse_graph(full_gi_stages()).expect("Full GI stages parses");
    let remove: Vec<&str> = remove.iter().chain(&UNINSTALLED_POST).copied().collect();
    let remove = remove.as_slice();
    let add: Vec<(&str, &str)> = add
        .iter()
        .copied()
        .chain([("giHistory.color", "tonemap.color")])
        .collect();
    doc.nodes.retain(|n| !remove.contains(&n.id.as_str()));
    doc.links.retain(|l| {
        !remove.contains(&l.from.node.as_str()) && !remove.contains(&l.to.node.as_str())
    });
    let port = |text: &str| {
        let (node, port) = text.split_once('.').expect("node.port");
        PortRef {
            node: node.into(),
            port: port.into(),
        }
    };
    for (from, to) in add {
        doc.links.push(LinkDoc {
            from: port(from),
            to: port(to),
            history: false,
        });
    }
    plan_from_json(&write_graph(&doc)).expect("variant plans")
}

/// What `render_plan_frames` reads back: the tone-mapped frame, the enabled pipeline entries
/// and the stage nodes that recorded on the last frame.
struct Rendered {
    pixels: Vec<u8>,
    size: (u32, u32),
    enabled: Vec<String>,
    ran: Vec<String>,
    /// The stage kinds (catalog type ids) that recorded on the last frame, under any node id.
    ran_kinds: Vec<String>,
}

/// Renders `frames` submitted frames of `plan` on the software GPU over a floor and a cube lit
/// by the default sun and an overhead area light. None when there is no software adapter.
fn render_plan_frames(name: &str, plan: GraphPlan, frames: u32) -> Option<Rendered> {
    use raa_core::{FrameGraph, MaterialId, OffscreenTarget};
    use raa_geom::builtin::DebugCube;
    use raa_geom::{GeometryProvider, MeshData, ParamValue, Params, Quality};
    use raa_lit::{AttributedMesh, LitOptions, LitRenderer, LocalLight, LocalLightKind};
    use raa_math::Vec3;
    use raa_scene::{RenderList, Scene, Transform};
    use std::{cell::RefCell, rc::Rc};

    const SIZE: (u32, u32) = (640, 360);
    const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    let _gpu_guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = raa_core::testing::software_gpu() else {
        raa_core::testing::skip(name, "no software adapter");
        return None;
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
    let (tonemap_pass, tonemap) = {
        let p = pipeline.borrow();
        raa_tonemap::TonemapPass::new(&gpu, &p)
    };
    pipeline
        .borrow_mut()
        .add(Box::new(tonemap_pass))
        .expect("tonemap pass");
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
    let mut scene = Scene::with_capacity(2, 2);
    scene.camera.set_viewport(SIZE.0, SIZE.1);
    scene.camera.eye = Vec3::new(0.0, 4.0, 5.0);
    // A dim sun keeps the floor from clipping, so the area light's shadow stays in range.
    scene.light.intensity = 0.05;
    scene.ambient.intensity = 0.02;
    for (translation, scale) in [
        (Vec3::new(0.0, 0.5, 0.0), Vec3::splat(1.0)),
        (Vec3::new(0.0, -0.05, 0.0), Vec3::new(12.0, 0.1, 12.0)),
    ] {
        let node = scene
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
            .add_instance(node, mesh_id, MaterialId(0))
            .expect("instance");
    }
    lit.local_lights_mut()
        .add(LocalLight {
            kind: LocalLightKind::Area,
            position: [-2.5, 3.0, 0.0],
            direction: [0.640_184, -0.768_221, 0.0],
            intensity: 12.0,
            range: 10.0,
            size: [3.0, 3.0],
            ..LocalLight::default()
        })
        .expect("area light");
    raa_lit::install_local_lights(&gpu, &pipeline, &mut lit).expect("light tiling installs");
    let gi = raa_gi::feature::GiFeature::install(&gpu, &pipeline, &mut lit, None, None)
        .expect("GI feature");
    let gi_frame = gi.clone();
    let mut handles = raa_graph::exec::HandleSet {
        gi: Some(gi),
        tonemap: Some(tonemap),
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
        executor.set_plan(plan, None, &mut h)
    };
    assert!(applied.ok, "refused: {:?}", applied.issues);
    let mut graph = FrameGraph::new(SURFACE);
    raa_pipeline::Pipeline::install(&pipeline, &mut graph);
    graph.compile().expect("compile pipeline graph");
    let target = OffscreenTarget::new(&gpu, SIZE.0, SIZE.1, SURFACE);
    let mut list = RenderList::with_capacity(2);
    let (mut tags, mut slots) = (Vec::new(), Vec::new());
    for frame in 0..frames {
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
        let stats = gi_frame.prepare_frame(&gpu, &mut scene, &mut lit);
        assert!(stats.active, "frame {frame}: GI inactive: {stats:?}");
        lit.prepare(&gpu, &mut scene, &mut list, &mut tags, &mut slots)
            .expect("prepare lit scene");
        let commands = graph
            .execute(&gpu, target.view(), target.size())
            .expect("execute frame graph");
        gpu.queue().submit([commands]);
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll submitted frame");
        gi_frame.observe_submission(u64::from(frame) + 1, true);
    }
    let p = pipeline.borrow();
    let enabled = p
        .passes()
        .into_iter()
        .filter(|pass| p.entry_enabled(&pass.name).unwrap_or(false))
        .map(|pass| pass.name)
        .collect();
    let ran_stats: Vec<_> = executor
        .stats()
        .into_iter()
        .filter(|s| s.ran && raa_gi::stages::GiStageKind::from_type_id(&s.type_id).is_some())
        .collect();
    let ran = ran_stats.iter().map(|s| s.node.clone()).collect();
    let ran_kinds = ran_stats.into_iter().map(|s| s.type_id).collect();
    Some(Rendered {
        pixels: target.read_rgba8(&gpu).expect("read frame"),
        size: SIZE,
        enabled,
        ran,
        ran_kinds,
    })
}

/// Variance of the luma left after subtracting each pixel's 3x3 mean over the floor rows: grain and
/// speckle raise it, while smooth shading contributes the same in every variant.
fn pixel_variance(image: &Rendered) -> f64 {
    let (w, h) = (image.size.0 as usize, image.size.1 as usize);
    let luma = |x: usize, y: usize| {
        let i = (y * w + x) * 4;
        let p = &image.pixels;
        0.2126 * f64::from(p[i]) + 0.7152 * f64::from(p[i + 1]) + 0.0722 * f64::from(p[i + 2])
    };
    let (mut sum, mut sum_sq, mut count) = (0.0, 0.0, 0.0);
    // The floor below the cube: the rows the area shadow and the SSGI bounce land on.
    for y in h * 11 / 20..h - 1 {
        for x in 1..w - 1 {
            let mut mean = 0.0;
            for dy in 0..3 {
                for dx in 0..3 {
                    mean += luma(x + dx - 1, y + dy - 1);
                }
            }
            let d = luma(x, y) - mean / 9.0;
            sum += d;
            sum_sq += d * d;
            count += 1.0;
        }
    }
    sum_sq / count - (sum / count).powi(2)
}

/// Mean luma over the same floor rows.
fn floor_mean(image: &Rendered) -> f64 {
    let (w, h) = (image.size.0 as usize, image.size.1 as usize);
    let rows = h * 11 / 20..h - 1;
    let sum: f64 = rows
        .clone()
        .flat_map(|y| (1..w - 1).map(move |x| (x, y)))
        .map(|(x, y)| {
            let p = &image.pixels[(y * w + x) * 4..];
            0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2])
        })
        .sum();
    sum / (rows.len() * (w - 2)) as f64
}

const VARIANCE_FRAMES: u32 = 6;

fn assert_family_has_no_monolithic_pass(image: &Rendered, family: &[&str]) {
    for pass in family {
        assert!(
            !image.enabled.iter().any(|e| e == pass),
            "monolithic {pass} is enabled: {:?}",
            image.enabled
        );
    }
}

/// No pass of stage `kind` is in the plan, installed in the pipeline or recorded, under any node
/// id: a pipeline entry is `<kind>#<node id>`.
fn assert_no_pass_of_kind(plan: &GraphPlan, image: &Rendered, kind: &str) {
    assert!(
        plan.nodes.iter().all(|n| n.type_id != kind),
        "the plan holds a {kind} node"
    );
    assert!(
        image
            .enabled
            .iter()
            .all(|e| e.split('#').next() != Some(kind)),
        "a {kind} pass is installed: {:?}",
        image.enabled
    );
    assert!(
        image.ran_kinds.iter().all(|k| k != kind),
        "a {kind} pass recorded: {:?}",
        image.ran_kinds
    );
}

#[test]
fn area_without_spatial_is_noisier_than_full_chain() {
    let name = "area_without_spatial_is_noisier_than_full_chain";
    let Some(full) = render_plan_frames(name, variant_plan(&[], &[]), VARIANCE_FRAMES) else {
        return;
    };
    let plan = variant_plan(
        &["areaSpatial"],
        &[("areaTemporal.filtered", "areaMask.src")],
    );
    assert!(plan.nodes.iter().all(|n| n.type_id != "area.spatial"));
    let variant = render_plan_frames(name, plan.clone(), VARIANCE_FRAMES).expect("adapter present");
    assert_no_pass_of_kind(&plan, &variant, "area.spatial");
    assert_family_has_no_monolithic_pass(&variant, &["gi.area-shadow"]);
    for node in ["areaTrace", "areaTemporal", "areaMask"] {
        assert!(variant.ran.iter().any(|n| n == node), "{node} did not run");
    }
    assert!(!variant.ran.iter().any(|n| n == "areaSpatial"));
    let (full_var, variant_var) = (pixel_variance(&full), pixel_variance(&variant));
    eprintln!("area variance: full {full_var} without spatial {variant_var}");
    assert!(
        variant_var > full_var,
        "without area.spatial the image must be noisier: {variant_var} vs {full_var}"
    );
    // area.temporal holds (S, U) pairs; read as visibility, S darkens the whole floor (or, in a
    // bright room, clamps to 1 and shows no shadow). area.mask converts them to S / U itself, an
    // unbiased estimate of the filtered visibility, so the floor keeps its brightness.
    let (full_mean, variant_mean) = (floor_mean(&full), floor_mean(&variant));
    eprintln!("area floor mean: full {full_mean} without spatial {variant_mean}");
    assert!(
        variant_mean > full_mean / 2.0,
        "the mask holds S, not visibility: floor mean {variant_mean} vs {full_mean}"
    );
}

#[test]
fn ssgi_without_resolve_is_noisier_than_full_chain() {
    let name = "ssgi_without_resolve_is_noisier_than_full_chain";
    let Some(full) = render_plan_frames(name, variant_plan(&[], &[]), VARIANCE_FRAMES) else {
        return;
    };
    let plan = variant_plan(&["ssgiResolve"], &[("ssgiTrace.raw", "forward.gi")]);
    assert!(plan.nodes.iter().all(|n| n.type_id != "ssgi.resolve"));
    let variant = render_plan_frames(name, plan.clone(), VARIANCE_FRAMES).expect("adapter present");
    assert_no_pass_of_kind(&plan, &variant, "ssgi.resolve");
    assert_family_has_no_monolithic_pass(&variant, &["gi.ssgi"]);
    assert!(variant.ran.iter().any(|n| n == "ssgiTrace"));
    assert!(!variant.ran.iter().any(|n| n == "ssgiResolve"));
    let (full_var, variant_var) = (pixel_variance(&full), pixel_variance(&variant));
    eprintln!("ssgi variance: full {full_var} without resolve {variant_var}");
    assert!(
        variant_var > full_var,
        "without ssgi.resolve the image must be noisier: {variant_var} vs {full_var}"
    );
}

#[test]
fn full_chain_records_no_monolithic_pass() {
    let name = "full_chain_records_no_monolithic_pass";
    let Some(full) = render_plan_frames(name, variant_plan(&[], &[]), VARIANCE_FRAMES) else {
        return;
    };
    assert_family_has_no_monolithic_pass(&full, &MONOLITHIC);
    for node in ["areaSpatial", "ssgiResolve"] {
        assert!(full.ran.iter().any(|n| n == node), "{node} did not run");
    }
}

/// A graph frame that records, at `before_record` of `forward`, whether a forward slot's bound
/// view is the same texture view the named stage published. The binding is applied by the
/// executor's graph frames in `before_prepare`, so the check runs where the forward pass records.
struct ForwardSlotProbe {
    slot: u32,
    source_entry: String,
    source_port: &'static str,
    matched: std::rc::Rc<std::cell::RefCell<Option<bool>>>,
}

impl raa_pipeline::GraphFrame for ForwardSlotProbe {
    fn before_record(
        &mut self,
        pass: &str,
        ctx: &mut raa_pipeline::RecordCtx<'_>,
    ) -> Result<(), raa_core::RaaError> {
        if pass != "forward" {
            return Ok(());
        }
        let slot = ctx.lighting.slot_view(self.slot);
        let source = ctx
            .resources
            .get(&self.source_entry, self.source_port)
            .map(|image| image.view);
        *self.matched.borrow_mut() = Some(matches!(
            (slot, source),
            (Some(bound), Some(published)) if bound == published
        ));
        Ok(())
    }
}

/// Renders three frames of `plan` and reports whether the forward `slot` is bound to the view the
/// `source_entry`/`source_port` stage published on the last frame. None when there is no adapter.
fn forward_slot_reads_stage(
    name: &str,
    plan: GraphPlan,
    slot: u32,
    source_entry: String,
    source_port: &'static str,
) -> Option<bool> {
    use raa_core::{FrameGraph, MaterialId, OffscreenTarget};
    use raa_geom::builtin::DebugCube;
    use raa_geom::{GeometryProvider, MeshData, ParamValue, Params, Quality};
    use raa_lit::{
        AttributedMesh, LitOptions, LitRenderer, LocalLight, LocalLightKind, StochasticMode,
        StochasticShadowSettings, install_local_lights,
    };
    use raa_math::Vec3;
    use raa_pipeline::Pipeline;
    use raa_scene::{RenderList, Scene, Transform};
    use std::{cell::RefCell, rc::Rc};

    const SIZE: (u32, u32) = (320, 180);
    const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    let _gpu_guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
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
    let mut scene = Scene::with_capacity(2, 2);
    scene.camera.set_viewport(SIZE.0, SIZE.1);
    scene.camera.eye = Vec3::new(0.0, 2.0, 4.0);
    for (translation, scale) in [
        (Vec3::new(0.0, 0.5, 0.0), Vec3::splat(1.0)),
        (Vec3::new(0.0, -0.05, 0.0), Vec3::new(8.0, 0.1, 8.0)),
    ] {
        let node = scene
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
            .add_instance(node, mesh_id, MaterialId(0))
            .expect("instance");
    }
    lit.local_lights_mut()
        .add(LocalLight {
            kind: LocalLightKind::Area,
            position: [-2.5, 3.0, 0.0],
            direction: [0.640_184, -0.768_221, 0.0],
            intensity: 12.0,
            range: 10.0,
            size: [3.0, 3.0],
            ..LocalLight::default()
        })
        .expect("area light");
    install_local_lights(&gpu, &pipeline, &mut lit).expect("light tiling installs");
    let gi = raa_gi::feature::GiFeature::install(&gpu, &pipeline, &mut lit, None, None)
        .expect("GI feature");
    let gi_frame = gi.clone();
    let mut handles = raa_graph::exec::HandleSet {
        gi: Some(gi),
        tonemap: Some(tonemap),
        ..Default::default()
    };
    let mut executor = raa_graph::exec::GraphExecutor::new();
    {
        let mut h = raa_graph::exec::FeatureHandles {
            gpu: &gpu,
            pipeline: &pipeline,
            handles: &mut handles,
            viewports: 1,
            msaa: 1,
            lit: Some(&mut lit),
        };
        let applied = executor.set_plan(plan, None, &mut h);
        assert!(applied.ok, "{name}: refused: {:?}", applied.issues);
    }
    let matched = Rc::new(RefCell::new(None));
    pipeline
        .borrow_mut()
        .add_graph_frame(
            "forward-slot-probe",
            Box::new(ForwardSlotProbe {
                slot,
                source_entry,
                source_port,
                matched: matched.clone(),
            }),
        )
        .expect("probe graph frame");
    let mut graph = FrameGraph::new(SURFACE);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().expect("compile pipeline graph");
    let target = OffscreenTarget::new(&gpu, SIZE.0, SIZE.1, SURFACE);
    let mut list = RenderList::with_capacity(2);
    let (mut tags, mut slots) = (Vec::new(), Vec::new());
    for frame in 0..3u64 {
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
                mode: StochasticMode::Stratified,
                ..StochasticShadowSettings::standard()
            })
            .expect("stratified area mode");
        }
        gi_frame.set_view(&scene.camera, SIZE, 1);
        gi_frame.set_sun(&scene.light);
        gi_frame.set_local_light_count(1);
        gi_frame.prepare_frame(&gpu, &mut scene, &mut lit);
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
    }
    *matched.borrow()
}

#[test]
fn raw_stage_graphs_bind_their_stage_output_to_forward() {
    // The scene's edits: remove the denoise stage, link the raw producer to its consumer.
    let area = variant_plan(
        &["areaSpatial"],
        &[("areaTemporal.filtered", "areaMask.src")],
    );
    assert!(area.nodes.iter().all(|n| n.type_id != "area.spatial"));
    let area_entry = raa_gi::stages::stage_entry(raa_gi::stages::GiStageKind::AreaMask, "areaMask");
    assert_eq!(
        forward_slot_reads_stage(
            "raw_stage_graphs_bind_their_stage_output_to_forward",
            area,
            20,
            area_entry,
            "areaShadow",
        ),
        Some(true),
        "the forward area-shadow slot does not read the area.mask stage output"
    );

    let ssgi = variant_plan(&["ssgiResolve"], &[("ssgiTrace.raw", "forward.gi")]);
    assert!(ssgi.nodes.iter().all(|n| n.type_id != "ssgi.resolve"));
    let ssgi_entry =
        raa_gi::stages::stage_entry(raa_gi::stages::GiStageKind::SsgiTrace, "ssgiTrace");
    assert_eq!(
        forward_slot_reads_stage(
            "raw_stage_graphs_bind_their_stage_output_to_forward",
            ssgi,
            14,
            ssgi_entry,
            "raw",
        ),
        Some(true),
        "the forward GI slot does not read the ssgi.trace raw output"
    );
}

#[test]
fn full_gi_stages_keeps_the_spatial_and_resolve_stages() {
    let full = starter_plan();
    assert!(
        full.nodes.iter().any(|n| n.type_id == "area.spatial"),
        "the complete graph must keep area.spatial"
    );
    assert!(
        full.nodes.iter().any(|n| n.type_id == "ssgi.resolve"),
        "the complete graph must keep ssgi.resolve"
    );
    let spatial = full
        .nodes
        .iter()
        .find(|n| n.type_id == "area.spatial")
        .expect("area.spatial");
    assert_eq!(
        node(&full, "areaMask")
            .inputs
            .iter()
            .find(|i| i.port == "src")
            .unwrap()
            .source,
        InputSource::Link {
            node: spatial.id.clone(),
            port: "mask".into(),
            history: false
        }
    );
    assert_eq!(
        node(&full, "forward")
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
}
