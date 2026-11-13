//! Two-phase occlusion culling end to end on a software GPU (SKIP without an adapter): with an
//! occluder in front of several virtualized instances, the drawn-cluster count of the main view
//! is lower with `occlusion` on than with it off, once the frames the two-phase scheme needs
//! have run (one frame seeds the previous-frame pyramid, the next tests against it) and the
//! asynchronous cull statistics of such a frame are published.
//!
//! The rig is the one of `tests/stats.rs`: a `Pipeline` and a `LitRenderer` of msaa 1, a shared
//! `ClusterStore` with one baked displaced sphere attached on key 3, and `VgFeature::install`.
//! Inputs the passes read are set up here: no mesh pass writes depth before the vg passes, so
//! the lit prepass clears depth to 1.0 (far) every frame and writes only the forward occluder;
//! the HDR and G-buffer targets start cleared. Five spheres stand in a row at z = -4 behind a
//! wall at z = 2 (the same sphere mesh scaled to 9 x 5 x 0.3 m), which covers the whole view
//! from the camera at (0, 1, 8) looking down -Z (near 0.1, far 16): the camera, the wall and the
//! instances lie inside a 16 m box. One more sphere stands in front of the wall, so the
//! occluded frame still draws clusters.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;
use std::rc::Rc;

use raa_cluster::{BuildOptions, ClusterStore, OwnedClusterInput, bake};
use raa_core::testing::{block_on, skip, software_gpu};
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
/// The five S12 entries of the feature.
const VG_PASSES: [&str; 5] = [
    "vg.cull",
    "vg.raster",
    "vg.resolve",
    "vg.gbuffer",
    "vg.shade",
];
/// The JS instance-group tag of the wall.
const WALL_TAG: u32 = 100;
/// The most frames one occlusion setting may take to publish a frame that used it.
const MAX_FRAMES: u64 = 16;

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
    invocation: u64,
}

impl Rig {
    /// The hidden row, the front sphere and the wall; `forward_wall` keeps the wall on the
    /// forward path (an explicit `set_instance(WALL_TAG, Some(false))`), as the
    /// virtual-geometry page's occlusion scene does.
    fn new(gpu: &GpuContext, forward_wall: bool) -> Rig {
        let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, SURFACE, false)));
        let opts = LitOptions {
            msaa: 1,
            ..LitOptions::default()
        };
        let mut lit = LitRenderer::new(gpu, &pipeline, opts).expect("lit renderer");
        let store = Rc::new(RefCell::new(ClusterStore::new()));
        let vg = VgFeature::install(gpu, &pipeline, &mut lit, Rc::clone(&store), None)
            .expect("the feature installs");
        let input = OwnedClusterInput::displaced_sphere(3);
        let clusters = bake(input.as_input(), &BuildOptions::default()).expect("the sphere bakes");
        store.borrow_mut().attach(KEY, vec![clusters]);
        let OwnedClusterInput { mesh, .. } = input;
        let id = lit
            .upload(gpu, &AttributedMesh::from_mesh(mesh))
            .expect("upload");
        let submeshes = lit.submesh_ids(id).expect("sub-meshes").to_vec();
        vg.set_mesh_key(KEY, &submeshes);

        let mut placements: Vec<(u32, Vec3, Vec3)> = (0..5)
            .map(|i| {
                let x = -4.0 + 2.0 * i as f32;
                (i + 1, Vec3::new(x, 1.0, -4.0), Vec3::ONE)
            })
            .collect();
        placements.push((10, Vec3::new(0.0, 1.0, 5.0), Vec3::splat(0.4)));
        placements.push((WALL_TAG, Vec3::new(0.0, 1.0, 2.0), Vec3::new(9.0, 5.0, 0.3)));
        let mut scene = Scene::with_capacity(placements.len(), placements.len());
        for (tag, at, scale) in placements {
            let node = scene
                .add_node(None, Transform::from_trs(at, Quat::IDENTITY, scale))
                .expect("node");
            let instance = scene
                .add_instance(node, submeshes[0], MaterialId(0))
                .expect("instance");
            scene.set_instance_tag(instance, tag).expect("tag");
        }
        let camera = Camera {
            eye: Vec3::new(0.0, 1.0, 8.0),
            target: Vec3::new(0.0, 1.0, 0.0),
            near: 0.1,
            far: 16.0,
            ..Camera::default()
        };
        scene.camera = camera;
        scene.camera.set_viewport(SIZE.0, SIZE.1);
        let mut graph = FrameGraph::new(SURFACE);
        Pipeline::install(&pipeline, &mut graph);
        graph.compile().expect("compile");
        let rig = Rig {
            pipeline,
            lit,
            vg,
            scene,
            camera,
            list: RenderList::with_capacity(8),
            tags: Vec::with_capacity(8),
            slots: Vec::with_capacity(8),
            graph,
            backbuffer: OffscreenTarget::new(gpu, SIZE.0, SIZE.1, SURFACE),
            invocation: 0,
        };
        rig.enable();
        if forward_wall {
            rig.vg
                .set_instance(WALL_TAG, Some(false))
                .expect("the wall stays on forward");
        }
        rig
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

    /// One complete scene invocation and its submission hook, then the map callbacks.
    fn frame(&mut self, gpu: &GpuContext) {
        self.invocation += 1;
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
        self.vg.observe_submission(self.invocation, true);
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
    }

    /// Sets `occlusion`, then runs frames until the published statistics come from the third
    /// invocation after the change or later: by then the previous-frame pyramid of a frame
    /// that used the setting is seeded. Returns the drawn clusters of that frame.
    fn drawn_clusters(&mut self, gpu: &GpuContext, occlusion: bool) -> u32 {
        self.vg
            .set(VgSettingsPatch {
                occlusion: Some(occlusion),
                ..VgSettingsPatch::default()
            })
            .expect("occlusion is a valid patch");
        let changed = self.invocation;
        for _ in 0..MAX_FRAMES {
            self.frame(gpu);
            let stats = self.vg.stats();
            if let (Some(index), Some(drawn)) = (stats.stats_frame_index, stats.drawn_clusters)
                && index >= changed + 3
            {
                println!(
                    "occlusion {occlusion}: {drawn} clusters drawn at invocation {index} \
                     (virtual instances {}, forward {})",
                    stats.virtual_instances, stats.forward_instances
                );
                return drawn;
            }
        }
        panic!(
            "statistics of a frame with occlusion {occlusion} not published within {MAX_FRAMES} \
             frames: {:?}",
            self.vg.stats()
        );
    }
}

/// Runs off, on, then off again in one rig, and asserts that occlusion on draws fewer clusters
/// than either off measurement.
fn assert_occlusion_lowers_drawn_clusters(test: &str, forward_wall: bool) {
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let mut rig = Rig::new(&gpu, forward_wall);
    let off = rig.drawn_clusters(&gpu, false);
    let on = rig.drawn_clusters(&gpu, true);
    let off_again = rig.drawn_clusters(&gpu, false);
    let error = block_on(errors.pop());
    assert!(error.is_none(), "validation error: {error:?}");
    let stats = rig.vg.stats();
    assert!(stats.active, "{stats:?}");
    let expected_virtual = if forward_wall { 6 } else { 7 };
    assert_eq!(stats.virtual_instances, expected_virtual, "{stats:?}");
    assert!(on > 0, "the sphere in front of the wall stays drawn");
    assert!(
        on < off && on < off_again,
        "occlusion on {on} clusters, off {off} then {off_again}"
    );
}

#[test]
fn occlusion_lowers_drawn_clusters_behind_a_virtualized_wall() {
    assert_occlusion_lowers_drawn_clusters(
        "occlusion_lowers_drawn_clusters_behind_a_virtualized_wall",
        false,
    );
}

#[test]
fn occlusion_lowers_drawn_clusters_behind_a_forward_wall() {
    assert_occlusion_lowers_drawn_clusters(
        "occlusion_lowers_drawn_clusters_behind_a_forward_wall",
        true,
    );
}
