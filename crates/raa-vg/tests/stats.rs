//! The asynchronous cull statistics end to end on a software GPU (SKIP without an adapter): the
//! real submission hook of the frame path, `VgHandle::observe_submission`, publishes the cull
//! counters of a completely submitted invocation, and later frames keep publishing newer ones.
//!
//! The rig is the one of `tests/feature.rs`: a `Pipeline` and a `LitRenderer` of msaa 1, a shared
//! `ClusterStore` with one baked displaced sphere attached on key 3, and `VgFeature::install`.
//! Inputs the passes read are set up here: no mesh pass writes depth before the vg passes, so the
//! lit prepass clears depth to 1.0 (far) and draws nothing once the one instance is virtualized;
//! the HDR and G-buffer targets start cleared. The camera sits at (0, 1, 4) looking at the origin
//! with near 0.1 and far 16, so the instance and the camera lie inside a 16 m box.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;
use std::rc::Rc;

use raa_cluster::{BuildOptions, ClusterStore, OwnedClusterInput, bake};
use raa_core::testing::{skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, MaterialId, OffscreenTarget, wgpu};
use raa_lit::{AttributedMesh, LitOptions, LitRenderer};
use raa_math::{Quat, Vec3};
use raa_pipeline::Pipeline;
use raa_scene::{Camera, RenderList, Scene, Transform};
use raa_vg::{VgFeature, VgSettingsPatch};

const SIZE: (u32, u32) = (64, 64);
const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// The cluster-store key of the baked sphere.
const KEY: u32 = 3;
/// The five S12 entries of the feature, in recording order.
const VG_PASSES: [&str; 5] = [
    "vg.cull",
    "vg.raster",
    "vg.resolve",
    "vg.gbuffer",
    "vg.shade",
];

struct Rig {
    pipeline: Rc<RefCell<Pipeline>>,
    lit: LitRenderer,
    vg: raa_vg::VgHandle,
    scene: Scene,
    camera: Camera,
    list: RenderList,
    tags: Vec<u32>,
    slots: Vec<u32>,
    graph: FrameGraph,
    backbuffer: OffscreenTarget,
}

impl Rig {
    fn new(gpu: &GpuContext) -> Rig {
        let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, SURFACE, false)));
        let opts = LitOptions {
            msaa: 1,
            ..LitOptions::default()
        };
        let mut lit = LitRenderer::new(gpu, &pipeline, opts).expect("lit renderer");
        let store = Rc::new(RefCell::new(ClusterStore::new()));
        let vg = VgFeature::install(gpu, &pipeline, &mut lit, Rc::clone(&store), None)
            .expect("the feature installs");
        let input = OwnedClusterInput::displaced_sphere(2);
        let clusters = bake(input.as_input(), &BuildOptions::default()).expect("the sphere bakes");
        store.borrow_mut().attach(KEY, vec![clusters]);
        let OwnedClusterInput { mesh, .. } = input;
        let id = lit
            .upload(gpu, &AttributedMesh::from_mesh(mesh))
            .expect("upload");
        let submeshes = lit.submesh_ids(id).expect("sub-meshes").to_vec();
        vg.set_mesh_key(KEY, &submeshes);
        let mut scene = Scene::with_capacity(1, 1);
        let node = scene
            .add_node(
                None,
                Transform::from_trs(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE),
            )
            .expect("node");
        scene
            .add_instance(node, submeshes[0], MaterialId(0))
            .expect("instance");
        let camera = Camera {
            eye: Vec3::new(0.0, 1.0, 4.0),
            target: Vec3::ZERO,
            near: 0.1,
            far: 16.0,
            ..Camera::default()
        };
        scene.camera = camera;
        scene.camera.set_viewport(SIZE.0, SIZE.1);
        let mut graph = FrameGraph::new(SURFACE);
        Pipeline::install(&pipeline, &mut graph);
        graph.compile().expect("compile");
        Rig {
            pipeline,
            lit,
            vg,
            scene,
            camera,
            list: RenderList::with_capacity(1),
            tags: Vec::with_capacity(1),
            slots: Vec::with_capacity(1),
            graph,
            backbuffer: OffscreenTarget::new(gpu, SIZE.0, SIZE.1, SURFACE),
        }
    }

    /// Enables the five entries and the feature.
    fn enable(&self) {
        let mut pipeline = self.pipeline.borrow_mut();
        for name in VG_PASSES {
            pipeline
                .set_enabled(name, true)
                .expect("the entry is installed");
        }
        self.vg
            .set(VgSettingsPatch {
                enabled: Some(true),
                ..VgSettingsPatch::default()
            })
            .expect("msaa 1 accepts enabled");
    }

    /// One complete scene invocation: prepare, execute, submit, then the submission hook. It
    /// does not wait for the readback, as the browser does not between frames.
    fn submit(&mut self, gpu: &GpuContext, invocation: u64) {
        self.vg.set_view(&self.camera, SIZE, 1);
        self.vg.prepare_frame(gpu, &mut self.scene, &mut self.lit);
        self.lit
            .prepare(
                gpu,
                &mut self.scene,
                &mut self.list,
                &mut self.tags,
                &mut self.slots,
            )
            .expect("prepare");
        let commands = self
            .graph
            .execute(gpu, self.backbuffer.view(), SIZE)
            .expect("execute");
        gpu.queue().submit([commands]);
        self.vg.observe_submission(invocation, true);
    }

    /// Delivers the outstanding map callbacks, as the browser does between frames.
    fn wait(&self, gpu: &GpuContext) {
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
    }

    /// A whole frame: submit it and wait for its readback before the next one.
    fn frame(&mut self, gpu: &GpuContext, invocation: u64) {
        self.submit(gpu, invocation);
        self.wait(gpu);
    }
}

#[test]
fn submitted_frames_publish_cluster_and_triangle_counts() {
    let test = "submitted_frames_publish_cluster_and_triangle_counts";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let mut rig = Rig::new(&gpu);
    rig.enable();
    assert_eq!(rig.vg.stats().drawn_clusters, None);

    let mut published = None;
    for invocation in 1..=6u64 {
        rig.frame(&gpu, invocation);
        let stats = rig.vg.stats();
        if stats.drawn_clusters.is_some() && stats.drawn_triangles.is_some() {
            published = Some(stats);
            break;
        }
    }
    let stats = published.expect("a completely submitted frame publishes its counters");
    assert!(
        stats.drawn_clusters.is_some_and(|clusters| clusters > 0),
        "{stats:?}"
    );
    assert!(
        stats.drawn_triangles.is_some_and(|triangles| triangles > 0),
        "{stats:?}"
    );
    assert!(stats.stats_frame_index.is_some(), "{stats:?}");
    assert!(
        !stats.pending,
        "the readback of the published frame completed"
    );
}

/// Browser timing: the map of a submitted frame resolves in the browser's own time, after the
/// submission hook returned. The getter still publishes the resolved counters without waiting
/// for another frame, and a capture is not piled on top of a still-mapped sample.
#[test]
fn a_deferred_map_is_published_without_stalling_later_frames() {
    let test = "a_deferred_map_is_published_without_stalling_later_frames";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let mut rig = Rig::new(&gpu);
    rig.enable();

    // Submitted and map requested; the browser has not called back yet.
    rig.submit(&gpu, 1);
    let stats = rig.vg.stats();
    assert!(
        stats.pending,
        "the submitted readback is still outstanding: {stats:?}"
    );
    assert_eq!(
        stats.drawn_clusters, None,
        "no counters are invented before the map resolves"
    );

    // The browser delivers the map: the getter publishes it without another submission hook.
    rig.wait(&gpu);
    let stats = rig.vg.stats();
    assert!(
        stats.drawn_clusters.is_some_and(|clusters| clusters > 0),
        "{stats:?}"
    );
    assert!(
        stats.drawn_triangles.is_some_and(|triangles| triangles > 0),
        "{stats:?}"
    );
    assert_eq!(stats.stats_frame_index, Some(1));
    assert!(!stats.pending);

    // Later frames keep publishing newer invocations with the same hook and no blocking poll.
    rig.frame(&gpu, 2);
    let stats = rig.vg.stats();
    assert!(
        stats.stats_frame_index.is_some_and(|index| index > 1),
        "{stats:?}"
    );
    assert!(stats.drawn_clusters.is_some(), "{stats:?}");
    assert!(!stats.pending);
}
