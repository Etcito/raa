//! The `taa` pass (S12, order 100) on top of the real lit renderer: its install state and
//! refusals, and its history accumulation, invalidation and the lit jitter consumer path.
//! The rig follows `raa-lit`'s `tests/motion.rs`; the software-GPU half is native only.

#[cfg(not(target_arch = "wasm32"))]
mod software_gpu_frame {
    use std::cell::RefCell;
    use std::rc::Rc;

    use raa_core::testing::{block_on, skip, software_gpu};
    use raa_core::{FrameGraph, GpuContext, MaterialId, MeshId, OffscreenTarget, wgpu};
    use raa_geom::builtin::DebugCube;
    use raa_geom::{GeometryProvider, MeshData, ParamValue, Params, Quality};
    use raa_lit::{AttributedMesh, LitOptions, LitRenderer};
    use raa_math::Vec3;
    use raa_pipeline::{PassRole, Pipeline};
    use raa_scene::{RenderList, Scene, Transform};
    use raa_taa::{TaaFeature, TaaHandle};

    const SIZE: (u32, u32) = (64, 64);
    const SMALL: (u32, u32) = (48, 32);
    const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    /// Eye movement per frame in metres; the cube at the origin stays inside the view.
    const STEP: f32 = 0.05;

    /// The DebugCube (half 0.5) as an attributed mesh (S10 default uv, tangent, colour).
    fn cube() -> AttributedMesh {
        let mut mesh = MeshData::new();
        let params = Params::new().with("half", ParamValue::Scalar(0.5));
        DebugCube
            .generate(&params, Quality::LOD0, &mut mesh)
            .expect("debug cube");
        AttributedMesh::from_mesh(mesh)
    }

    /// A pipeline with a lit renderer (MSAA 1), the TAA pass installed disabled from the lit
    /// frame buffer, and one cube at the origin. The prepass writes depth (cleared to 1.0,
    /// the nearer cube wins) and motion, so the pass reads initialised inputs.
    struct Rig {
        pipeline: Rc<RefCell<Pipeline>>,
        lit: LitRenderer,
        taa: TaaHandle,
        scene: Scene,
        list: RenderList,
        tags: Vec<u32>,
        slots: Vec<u32>,
        graph: FrameGraph,
        backbuffer: OffscreenTarget,
    }

    impl Rig {
        fn new(gpu: &GpuContext) -> Rig {
            let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, SURFACE, false)));
            let mut lit = LitRenderer::new(
                gpu,
                &pipeline,
                LitOptions {
                    max_instances: 16384,
                    max_materials: 256,
                    msaa: 1,
                },
            )
            .expect("lit renderer");
            let id = lit.upload(gpu, &cube()).expect("upload");
            let mesh: MeshId = lit.submesh_ids(id).expect("sub-meshes")[0];
            let taa = TaaFeature::install(gpu, &pipeline, &lit.handle().frame_buffer())
                .expect("taa install");
            let mut scene = Scene::with_capacity(1, 1);
            scene.camera.set_viewport(SIZE.0, SIZE.1);
            let node = scene.add_node(None, Transform::IDENTITY).expect("node");
            scene
                .add_instance(node, mesh, MaterialId(0))
                .expect("instance");
            let mut graph = FrameGraph::new(SURFACE);
            Pipeline::install(&pipeline, &mut graph);
            graph.compile().expect("compile");
            Rig {
                pipeline,
                lit,
                taa,
                scene,
                list: RenderList::with_capacity(1),
                tags: Vec::with_capacity(1),
                slots: Vec::with_capacity(1),
                graph,
                backbuffer: OffscreenTarget::new(gpu, SIZE.0, SIZE.1, SURFACE),
            }
        }

        fn set_enabled(&self, enabled: bool) {
            self.pipeline
                .borrow_mut()
                .set_enabled("taa", enabled)
                .expect("taa entry");
        }

        /// Moves the eye along +X by one step; the target stays on the cube.
        fn step_camera(&mut self) {
            let eye = self.scene.camera.eye;
            self.scene.camera.eye = Vec3::new(eye.x + STEP, eye.y, eye.z);
        }

        /// Sets the viewport and renders into a new target of that size.
        fn resize(&mut self, gpu: &GpuContext, size: (u32, u32)) {
            self.scene.camera.set_viewport(size.0, size.1);
            self.backbuffer = OffscreenTarget::new(gpu, size.0, size.1, SURFACE);
        }

        fn frame(&mut self, gpu: &GpuContext) {
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
                .execute(gpu, self.backbuffer.view(), self.backbuffer.size())
                .expect("execute");
            gpu.queue().submit([commands]);
            self.graph.notify_submitted();
            gpu.device()
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("device poll");
        }
    }

    #[test]
    fn taa_installs_disabled_at_order_100_and_refuses_a_second_install() {
        let test = "taa_installs_disabled_at_order_100_and_refuses_a_second_install";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let rig = Rig::new(&gpu);

        let rows = rig.pipeline.borrow().passes();
        let taa: Vec<_> = rows.iter().filter(|row| row.name == "taa").collect();
        assert_eq!(taa.len(), 1, "one taa row");
        assert_eq!(taa[0].role, PassRole::Pass, "taa is a pass");
        assert_eq!(taa[0].order, 100, "taa order");
        assert!(!taa[0].enabled, "taa is installed disabled");
        assert!(!taa[0].required, "taa is not required");

        let refused = TaaFeature::install(&gpu, &rig.pipeline, &rig.lit.handle().frame_buffer())
            .err()
            .expect("a second install is refused");
        assert_eq!(refused.code(), "pipeline.duplicate_pass");
        assert_eq!(rig.pipeline.borrow().passes(), rows, "pass list unchanged");

        {
            let _held = rig.pipeline.borrow_mut();
            let busy = TaaFeature::install(&gpu, &rig.pipeline, &rig.lit.handle().frame_buffer())
                .err()
                .expect("install while the pipeline is borrowed");
            assert_eq!(busy.code(), "pipeline.busy");
        }
    }

    #[test]
    fn taa_history_accumulates_and_invalidates_on_disable_resize_and_invalidate() {
        let test = "taa_history_accumulates_and_invalidates_on_disable_resize_and_invalidate";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let mut rig = Rig::new(&gpu);

        // Enabled, the entry is a temporal consumer: the lit frame gets its jitter and the
        // camera history of each frame is used by the resolve.
        rig.set_enabled(true);
        for _ in 0..3 {
            rig.step_camera();
            rig.frame(&gpu);
        }
        assert_eq!(rig.taa.accumulated_frames(), 3, "three frames accumulate");
        assert!(rig.taa.history_valid(), "history is valid after 3 frames");

        // Disabling invalidates the history at once.
        rig.set_enabled(false);
        assert_eq!(rig.taa.accumulated_frames(), 0, "disable drops the count");
        assert!(!rig.taa.history_valid(), "disable invalidates the history");

        // Re-enabling starts a new history: one frame commits one frame.
        rig.set_enabled(true);
        rig.frame(&gpu);
        assert_eq!(rig.taa.accumulated_frames(), 1, "re-enabled history starts");

        // A resize invalidates before this frame commits, so the count restarts at 1.
        rig.resize(&gpu, SMALL);
        rig.frame(&gpu);
        assert_eq!(
            rig.taa.accumulated_frames(),
            1,
            "resize restarts the history"
        );

        // A camera cut through the handle reads zero at once and restarts after one frame.
        rig.taa.invalidate();
        assert_eq!(
            rig.taa.accumulated_frames(),
            0,
            "invalidate reads zero at once"
        );
        assert!(!rig.taa.history_valid(), "invalidate drops validity");
        rig.frame(&gpu);
        assert_eq!(
            rig.taa.accumulated_frames(),
            1,
            "one frame after invalidate"
        );

        let error = block_on(errors.pop());
        assert!(error.is_none(), "GPU validation error: {error:?}");
    }
}
