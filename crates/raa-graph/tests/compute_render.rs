//! Render the Custom denoise starter through the software adapter when available.
//!
//! This graph has no mesh pass that initializes stage inputs beyond the prepass: depth is
//! cleared to 1.0 (far), and the G-buffer and HDR targets start cleared before draws. The
//! debug cube and camera use the same view as the GI stage render test, inside the trace bounds.
#[test]
fn custom_denoise_renders_frames_without_panic() {
    use raa_core::{FrameGraph, MaterialId, OffscreenTarget};
    use raa_geom::builtin::DebugCube;
    use raa_geom::{GeometryProvider, MeshData, ParamValue, Params, Quality};
    use raa_lit::{AttributedMesh, LitOptions, LitRenderer, LocalLight, LocalLightKind};
    use raa_scene::{RenderList, Scene, Transform};
    use std::{cell::RefCell, rc::Rc};

    const SIZE: (u32, u32) = (1280, 720);
    const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    let name = "custom_denoise_renders_frames_without_panic";
    let Some(gpu) = raa_core::testing::software_gpu() else {
        raa_core::testing::skip(name, "no software adapter");
        return;
    };
    let starter = raa_graph::registry::STARTERS
        .iter()
        .find(|(_, stem, _)| *stem == "custom-denoise")
        .map(|(_, _, json)| *json)
        .expect("generated Custom denoise starter");
    let plan = raa_graph::plan::plan_from_json(starter).expect("Custom denoise starter plans");
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
    // Scene::default supplies the directional sun; add the same overhead area light as the GI test.
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
    }

    let stats = executor.stats();
    let denoise = stats
        .iter()
        .find(|node| node.node == "giDenoise" && node.type_id == "compute.wgsl")
        .expect("giDenoise stats");
    assert!(denoise.ran, "giDenoise did not record");
    let passes = pipeline.borrow().passes();
    let accumulate = passes
        .iter()
        .position(|pass| pass.name == "gi.accumulate#giAccumulate")
        .expect("GI accumulate pass");
    let denoise = passes
        .iter()
        .position(|pass| pass.name == "compute#giDenoise")
        .expect("compute denoise pass");
    let resolve = passes
        .iter()
        .position(|pass| pass.name == "gi.resolve#giResolve")
        .expect("GI resolve pass");
    assert_eq!(denoise, accumulate + 1);
    assert_eq!(resolve, denoise + 1);
}
