//! `VgFeature::install` and `VgHandle` on a software GPU (SKIP without an adapter): the msaa
//! refusal of `VgHandle::set`, the disabled state (no pass recorded, no vg resource), the S12
//! recording order of the five passes with one baked instance, and the multi-view fallback.
//!
//! Every test builds a `Pipeline` and a `LitRenderer` of the stated msaa, shares a `ClusterStore`
//! with the handle and installs the feature with `VgFeature::install(.., None)`. The one instance
//! is a baked displaced sphere (raa_cluster `bake`, `attach` on key 3, `set_mesh_key` to its lit
//! sub-mesh) at the origin. Inputs the passes read are set up here: no mesh pass writes depth
//! before the vg passes, so the lit prepass (which clears depth to 1.0 and draws nothing once the
//! one instance is virtualized) provides the far depth; the camera sits at (0, 1, 4) looking at
//! the origin with near 0.1 and far 16, so the instance and the camera lie inside a 16 m box.

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
use raa_vg::{VgFeature, VgHandle, VgSettingsPatch};

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

/// A pipeline with the feature installed (all five entries disabled) and one baked instance.
struct Rig {
    pipeline: Rc<RefCell<Pipeline>>,
    lit: LitRenderer,
    vg: VgHandle,
    scene: Scene,
    camera: Camera,
    list: RenderList,
    tags: Vec<u32>,
    slots: Vec<u32>,
    graph: FrameGraph,
    backbuffer: OffscreenTarget,
}

impl Rig {
    fn new(gpu: &GpuContext, msaa: u32) -> Rig {
        let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, SURFACE, false)));
        let opts = LitOptions {
            msaa,
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

    /// Enables or disables the five entries through the pipeline.
    fn set_passes(&self, enabled: bool) {
        let mut pipeline = self.pipeline.borrow_mut();
        for name in VG_PASSES {
            pipeline
                .set_enabled(name, enabled)
                .expect("the entry is installed");
        }
    }

    /// Sets the view, prepares the feature and the lit renderer; returns the triangles drawn on
    /// forward. Nothing executes.
    fn prepare(&mut self, gpu: &GpuContext, views: u8) -> u64 {
        self.vg.set_view(&self.camera, SIZE, views);
        self.vg.prepare_frame(gpu, &mut self.scene, &mut self.lit);
        self.lit
            .prepare(
                gpu,
                &mut self.scene,
                &mut self.list,
                &mut self.tags,
                &mut self.slots,
            )
            .expect("prepare")
            .triangles
    }

    /// One frame: prepare, then execute the graph into the offscreen target and wait for it.
    fn frame(&mut self, gpu: &GpuContext, views: u8) -> u64 {
        let triangles = self.prepare(gpu, views);
        let commands = self
            .graph
            .execute(gpu, self.backbuffer.view(), SIZE)
            .expect("execute");
        gpu.queue().submit([commands]);
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        triangles
    }
}

#[test]
fn msaa_four_refuses_enabled() {
    let test = "msaa_four_refuses_enabled";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let rig = Rig::new(&gpu, 4);
    let patch = VgSettingsPatch {
        enabled: Some(true),
        ..VgSettingsPatch::default()
    };
    let error = rig.vg.set(patch).expect_err("msaa 4 refuses enabled");
    assert_eq!(error.code(), "vg.invalid");
    assert_eq!(error.message(), "msaa: virtualized geometry needs msaa 1");
    assert!(!rig.vg.get().enabled, "a refused patch changes nothing");
}

#[test]
fn disabled_records_nothing_and_allocates_nothing() {
    let test = "disabled_records_nothing_and_allocates_nothing";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let mut rig = Rig::new(&gpu, 1);
    let triangles = rig.frame(&gpu, 1);
    assert!(
        triangles > 0,
        "the instance stays on forward while the feature is disabled"
    );
    assert!(rig.vg.recorded_passes().is_empty());
    assert_eq!(rig.vg.allocated_bytes(), 0);
    let stats = rig.vg.stats();
    assert!(!stats.active);
    assert_eq!(stats.inactive_reason, Some("disabled"));
    assert!(!rig.lit.frame_feature(27));
    assert_eq!(rig.lit.virtual_instances().len(), 0);
}

#[test]
fn enabled_records_in_s12_order() {
    let test = "enabled_records_in_s12_order";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let mut rig = Rig::new(&gpu, 1);
    let forward = rig.prepare(&gpu, 1);
    assert!(forward > 0, "the baseline instance draws on forward");

    rig.vg
        .set(VgSettingsPatch {
            enabled: Some(true),
            ..VgSettingsPatch::default()
        })
        .expect("msaa 1 accepts enabled");
    rig.set_passes(true);
    let triangles = rig.frame(&gpu, 1);
    assert_eq!(rig.vg.recorded_passes(), VG_PASSES);
    assert!(rig.vg.allocated_bytes() > 0);
    let stats = rig.vg.stats();
    assert!(stats.active, "{stats:?}");
    assert_eq!(stats.inactive_reason, None);
    assert_eq!(rig.lit.virtual_instances().len(), 1);
    assert_eq!(
        triangles, 0,
        "the virtualized instance leaves the forward triangle count"
    );
    assert!(triangles < forward);
}

#[test]
fn two_views_fall_back() {
    let test = "two_views_fall_back";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let mut rig = Rig::new(&gpu, 1);
    rig.vg
        .set(VgSettingsPatch {
            enabled: Some(true),
            ..VgSettingsPatch::default()
        })
        .expect("msaa 1 accepts enabled");
    rig.set_passes(true);
    rig.frame(&gpu, 1);
    assert_eq!(rig.vg.recorded_passes(), VG_PASSES);
    assert!(rig.vg.allocated_bytes() > 0);

    rig.frame(&gpu, 2);
    let stats = rig.vg.stats();
    assert!(!stats.active);
    assert_eq!(stats.inactive_reason, Some("multiView"));
    assert_eq!(rig.vg.allocated_bytes(), 0);
    assert!(rig.vg.recorded_passes().is_empty());
}
