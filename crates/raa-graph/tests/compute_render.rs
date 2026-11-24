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

#[test]
fn explicit_compute_extents_dispatch_through_the_last_texel() {
    use raa_core::{FrameGraph, OffscreenTarget};
    use raa_graph::exec::{FeatureHandles, GraphExecutor, HandleSet};
    use raa_pipeline::Pipeline;
    use std::{cell::RefCell, rc::Rc};

    let Some(gpu) = raa_core::testing::software_gpu() else {
        raa_core::testing::skip(
            "explicit_compute_extents_dispatch_through_the_last_texel",
            "no software adapter",
        );
        return;
    };
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, format, false)));
    let mut lit =
        raa_lit::LitRenderer::new(&gpu, &pipeline, raa_lit::LitOptions::default()).unwrap();
    let mut handles = HandleSet::default();
    let mut executor = GraphExecutor::new();
    let text = r#"{
      "format":"raa.render-graph","version":1,"name":"Explicit extents",
      "nodes":[
        {"id":"prepass","type":"pass.prepass"},{"id":"forward","type":"pass.forward"},
        {"id":"tonemap","type":"pass.tonemap"},{"id":"out","type":"output.display"},
        {"id":"scene","type":"compute.wgsl","inputs":[{"key":"src","type":"image2d"}],
          "outputs":[{"key":"dst","type":"image2d","format":"rgba8unorm",
            "extent":{"basis":"scene","numerator":1,"denominator":1}}],
          "math":null,"wgsl":"fn rg_main(p: vec2<u32>) { textureStore(rg_out_dst, vec2<i32>(p), vec4<f32>(f32(p.x < (rg_frame.size.x + 1u) / 2u), f32(p.y < (rg_frame.size.y + 1u) / 2u), 0.0, 1.0)); }"},
        {"id":"trace","type":"compute.wgsl","inputs":[{"key":"src","type":"image2d"}],
          "outputs":[{"key":"dst","type":"image2d","format":"rgba8unorm",
            "extent":{"basis":"parent","input":"src","numerator":1,"denominator":2}}],
          "math":null,"wgsl":"fn rg_main(p: vec2<u32>) { let q = min(vec2<u32>((vec2<f32>(p) + 0.5) * rg_frame.inv_size * vec2<f32>(textureDimensions(rg_in_src))), textureDimensions(rg_in_src) - 1u); textureStore(rg_out_dst, vec2<i32>(p), textureLoad(rg_in_src, vec2<i32>(q), 0)); }"},
        {"id":"display","type":"compute.wgsl","inputs":[{"key":"src","type":"image2d"}],
          "outputs":[{"key":"dst","type":"image2d","format":"rgba8unorm",
            "extent":{"basis":"display","numerator":1,"denominator":1}}],
          "math":null,"wgsl":"fn rg_main(p: vec2<u32>) { let q = min(vec2<u32>((vec2<f32>(p) + 0.5) * rg_frame.inv_size * vec2<f32>(textureDimensions(rg_in_src))), textureDimensions(rg_in_src) - 1u); textureStore(rg_out_dst, vec2<i32>(p), textureLoad(rg_in_src, vec2<i32>(q), 0)); }"},
        {"id":"show","type":"view.image"}],
      "links":[{"from":"prepass.normal","to":"scene.src"},{"from":"scene.dst","to":"trace.src"},
        {"from":"trace.dst","to":"display.src"},{"from":"display.dst","to":"show.src"},
        {"from":"show.color","to":"tonemap.color"},{"from":"tonemap.color","to":"out.color"}]
    }"#;
    let mut h = FeatureHandles {
        gpu: &gpu,
        pipeline: &pipeline,
        handles: &mut handles,
        viewports: 1,
        msaa: 1,
        lit: Some(&mut lit),
    };
    let doc = raa_graph::format::parse_graph(text).unwrap();
    let serialized = raa_graph::format::graph_to_json_value(&doc);
    assert_eq!(
        raa_graph::format::graph_from_json_value(&serialized).unwrap(),
        doc
    );
    let mut graph = FrameGraph::new(format);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().unwrap();

    let native = text.replace(r#""denominator":2"#, r#""denominator":1"#);
    for (description, display, scene, trace) in [
        (Some(native.as_str()), (64, 48), (64, 48), (64, 48)),
        (Some(text), (1, 1), (1, 1), (1, 1)),
        (None, (1919, 1079), (1279, 719), (640, 360)),
    ] {
        if let Some(description) = description {
            let plan = raa_graph::plan::plan_from_json(description).unwrap();
            let result = executor.set_plan(plan, None, &mut h);
            assert!(result.ok, "{:?}", result.issues);
        }
        pipeline.borrow_mut().set_scene_size(Some(scene));
        executor.frame(&mut h).unwrap();
        let target = OffscreenTarget::new(&gpu, display.0, display.1, format);
        let commands = graph.execute(&gpu, target.view(), display).unwrap();
        gpu.queue().submit([commands]);
        let pipeline = pipeline.borrow();
        let targets = pipeline.targets().unwrap();
        assert_eq!(targets.scene_rect().size, scene);
        assert_eq!(targets.display_rect().size, display);
        let stats = executor.stats();
        for (name, size) in [("scene", scene), ("trace", trace), ("display", display)] {
            let node = stats.iter().find(|node| node.node == name).unwrap();
            assert!(node.ran, "{name}");
            assert_eq!((node.resources[0].width, node.resources[0].height), size);
            let image = pipeline
                .resources()
                .get(&format!("compute#{name}"), "dst")
                .unwrap();
            assert_compute_corners(&gpu, image.view.texture(), size);
        }
    }
}

fn assert_compute_corners(gpu: &raa_core::GpuContext, texture: &wgpu::Texture, (w, h): (u32, u32)) {
    let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    let corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)];
    for (index, (x, y)) in corners.into_iter().enumerate() {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: index as u64 * 4,
                    ..Default::default()
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }
    gpu.queue().submit([encoder.finish()]);
    let slice = buffer.slice(..);
    let (send, receive) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        send.send(result).unwrap()
    });
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    receive.recv().unwrap().unwrap();
    let bytes = slice.get_mapped_range().unwrap();
    for ((x, y), pixel) in corners.into_iter().zip(bytes.chunks_exact(4)) {
        assert_eq!(
            pixel,
            [
                if x == 0 { 255 } else { 0 },
                if y == 0 { 255 } else { 0 },
                0,
                255
            ]
        );
    }
}
