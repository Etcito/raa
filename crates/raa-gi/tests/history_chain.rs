//! `gi.history` on a bound HDR chain (contract rg-hdr-chain-binding), on a software GPU (SKIP
//! without an adapter): a test `GraphFrame` binds the chain to two pool-usage Rgba16Float
//! textures around `gi.history`; two frames run with the debug view `none` (the pass copies src
//! into dst) and two with the voxel debug view (it writes the bound dst). Shared inputs: the lit
//! prepass writes depth from the one box and forward writes the HDR chain; no other pass runs.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{skip, software_gpu};
use raa_core::{FrameGraph, OffscreenTarget, RaaError};
use raa_gi::{DebugView, GiFeature, GiSettingsPatch};
use raa_lit::{AttributedMesh, LitOptions, LitRenderer, SurfaceMaterial};
use raa_math::{Quat, Vec3};
use raa_pipeline::{GraphFrame, Pipeline, RecordCtx};
use raa_scene::{Camera, RenderList, Scene, Transform};

const SIZE: (u32, u32) = (64, 48);
const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Binds the chain to (src, dst) for `gi.history` only and counts the bound records.
struct BindFrame {
    src: wgpu::TextureView,
    dst: wgpu::TextureView,
    bound: Rc<RefCell<u32>>,
}

impl GraphFrame for BindFrame {
    fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if pass == "gi.history" {
            ctx.targets.hdr().bind(&self.src, &self.dst);
            *self.bound.borrow_mut() += u32::from(ctx.targets.hdr().is_bound());
        }
        Ok(())
    }

    fn after_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if pass == "gi.history" {
            ctx.targets.hdr().unbind();
        }
        Ok(())
    }
}

/// A 1 m cube centred at the origin (hard edges, outward normals, CCW).
fn cube() -> AttributedMesh {
    let mut mesh = AttributedMesh::from_mesh(Default::default());
    for axis in 0..3 {
        for sign in [-0.5_f32, 0.5] {
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            let mut normal = [0.0_f32; 3];
            normal[axis] = sign.signum();
            let first = mesh.base.positions.len() as u32;
            for (a, b) in [(-0.5_f32, -0.5_f32), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)] {
                let mut p = [0.0_f32; 3];
                p[axis] = sign;
                p[u] = a;
                p[v] = b;
                mesh.base.positions.push(p);
                mesh.base.normals.push(normal);
            }
            let quad = if sign > 0.0 {
                [0u32, 1, 2, 0, 2, 3]
            } else {
                [0u32, 2, 1, 0, 3, 2]
            };
            mesh.base.indices.extend(quad.map(|i| first + i));
        }
    }
    mesh.submeshes[0].index_count = mesh.base.indices.len() as u32;
    mesh
}

fn pool_texture(gpu: &raa_core::GpuContext, label: &str) -> wgpu::TextureView {
    gpu.device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: SIZE.0,
                height: SIZE.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

#[test]
fn gi_history_runs_on_a_bound_chain_with_and_without_debug_view() {
    let test = "gi_history_runs_on_a_bound_chain_with_and_without_debug_view";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, SURFACE, false)));
    let mut lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit renderer");
    let handle = GiFeature::install(&gpu, &pipeline, &mut lit, None, None).expect("install");
    let id = lit.upload(&gpu, &cube()).expect("upload");
    let mesh = lit.submesh_ids(id).expect("sub-meshes")[0];
    let material = lit
        .materials_mut()
        .add(SurfaceMaterial::default())
        .expect("material");
    let mut scene = Scene::with_capacity(1, 1);
    let node = scene
        .add_node(
            None,
            Transform::from_trs(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE),
        )
        .expect("node");
    scene.add_instance(node, mesh, material).expect("instance");
    // The camera sits 4 m from the box: both lie inside the GI volume around the camera.
    let mut camera = Camera {
        eye: Vec3::new(0.0, 1.0, 4.0),
        target: Vec3::ZERO,
        ..scene.camera
    };
    camera.set_viewport(SIZE.0, SIZE.1);
    scene.camera = camera;
    let bound = Rc::new(RefCell::new(0));
    let frame = BindFrame {
        src: pool_texture(&gpu, "history_chain.src"),
        dst: pool_texture(&gpu, "history_chain.dst"),
        bound: bound.clone(),
    };
    pipeline
        .borrow_mut()
        .add_graph_frame("test", Box::new(frame))
        .expect("graph frame");
    let mut graph = FrameGraph::new(SURFACE);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().expect("compile");
    let backbuffer = OffscreenTarget::new(&gpu, SIZE.0, SIZE.1, SURFACE);
    let (mut list, mut tags, mut slots) = (RenderList::with_capacity(1), Vec::new(), Vec::new());
    for view in [DebugView::None, DebugView::Voxels] {
        handle
            .set(GiSettingsPatch {
                enabled: Some(true),
                debug_view: Some(view),
                ..GiSettingsPatch::default()
            })
            .expect("settings");
        for _ in 0..2 {
            handle.set_view(&camera, SIZE, 1);
            handle.prepare_frame(&gpu, &mut scene, &mut lit);
            lit.prepare(&gpu, &mut scene, &mut list, &mut tags, &mut slots)
                .expect("lit prepare");
            let commands = graph
                .execute(&gpu, backbuffer.view(), SIZE)
                .expect("execute");
            gpu.queue().submit([commands]);
            gpu.device()
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("device poll");
        }
    }
    assert_eq!(
        *bound.borrow(),
        4,
        "gi.history recorded bound in every frame"
    );
}
