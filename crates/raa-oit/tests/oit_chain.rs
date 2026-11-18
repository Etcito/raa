//! `transparent.oit` with a bound HDR chain on a software GPU (SKIP without an adapter): a
//! test `GraphFrame` binds the chain to two pool-like Rgba16Float textures before the pass
//! records and unbinds after. Frame 1 draws one translucent quad (bound resolve group),
//! frame 2 hides it (no Transparent draw: the pass copies src into dst). No mesh pass writes
//! depth here: the prepass clears depth to 1.0 (far) and the HDR chain starts cleared.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, OffscreenTarget, RaaError, wgpu};
use raa_lit::{FLAG_TRANSPARENT, LitOptions, LitRenderer, SurfaceMaterial, SurfaceModel};
use raa_math::Vec3;
use raa_mesh::{AttributedMesh, SubMesh};
use raa_oit::OitFeature;
use raa_pipeline::{GraphFrame, Pipeline, RecordCtx};
use raa_scene::{RenderList, Scene, Transform};

const SIZE: (u32, u32) = (64, 64);
const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Binds (src, dst) around `transparent.oit` and records whether it saw the chain bound.
struct Bind {
    src: wgpu::TextureView,
    dst: wgpu::TextureView,
    bound: Rc<RefCell<u32>>,
}

impl GraphFrame for Bind {
    fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if pass == "transparent.oit" {
            ctx.targets.hdr().bind(&self.src, &self.dst);
            if ctx.targets.hdr().is_bound() {
                *self.bound.borrow_mut() += 1;
            }
        }
        Ok(())
    }
    fn after_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if pass == "transparent.oit" {
            ctx.targets.hdr().unbind();
        }
        Ok(())
    }
}

fn pool_view(gpu: &GpuContext) -> wgpu::TextureView {
    gpu.device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("oit_chain.pool"),
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

fn quad() -> AttributedMesh {
    let mut m = AttributedMesh::default();
    m.base.positions = vec![
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ];
    m.base.normals = vec![[0.0, 0.0, 1.0]; 4];
    m.base.indices = vec![0, 1, 2, 0, 2, 3];
    m.submeshes = vec![SubMesh {
        first_index: 0,
        index_count: 6,
        material: None,
    }];
    m
}

#[test]
fn bound_chain_frames_with_and_without_transparent_draws() {
    let test = "bound_chain_frames_with_and_without_transparent_draws";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, SURFACE, false)));
    let opts = LitOptions {
        msaa: 1,
        ..LitOptions::default()
    };
    let mut lit = LitRenderer::new(&gpu, &pipeline, opts).expect("lit renderer");
    let _oit = OitFeature::install(&gpu, &pipeline, &mut lit).expect("oit install");
    let bound = Rc::new(RefCell::new(0));
    let frame = Bind {
        src: pool_view(&gpu),
        dst: pool_view(&gpu),
        bound: bound.clone(),
    };
    pipeline
        .borrow_mut()
        .add_graph_frame("test.bind", Box::new(frame))
        .expect("graph frame");

    let mut scene = Scene::with_capacity(1, 1);
    let id = lit.upload(&gpu, &quad()).expect("upload");
    let mesh = lit.submesh_ids(id).expect("sub-meshes")[0];
    let material = lit
        .materials_mut()
        .add(SurfaceMaterial {
            model: SurfaceModel::Flat,
            base_color: [1.0, 0.0, 0.0],
            opacity: 0.5,
            flags: FLAG_TRANSPARENT,
            ..SurfaceMaterial::default()
        })
        .expect("material");
    let node = scene.add_node(None, Transform::IDENTITY).expect("node");
    let instance = scene.add_instance(node, mesh, material).expect("instance");
    scene.camera.eye = Vec3::new(0.0, 0.0, 5.0);
    scene.camera.target = Vec3::ZERO;
    scene.camera.up = Vec3::Y;
    scene.camera.set_viewport(SIZE.0, SIZE.1);

    let mut graph = FrameGraph::new(SURFACE);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().expect("compile");
    let backbuffer = OffscreenTarget::new(&gpu, SIZE.0, SIZE.1, SURFACE);
    let (mut list, mut tags, mut slots) = (RenderList::with_capacity(1), vec![], vec![]);
    for visible in [true, false] {
        scene
            .set_instance_visible(instance, visible)
            .expect("visible");
        lit.prepare(&gpu, &mut scene, &mut list, &mut tags, &mut slots)
            .expect("prepare");
        let commands = graph
            .execute(&gpu, backbuffer.view(), SIZE)
            .expect("execute");
        gpu.queue().submit([commands]);
    }
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    let error = pollster::block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");
    assert_eq!(*bound.borrow(), 2, "the chain was bound in both frames");
}
