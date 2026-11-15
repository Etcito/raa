//! The `gi.ssgi` pass (order 44) end to end on a software GPU (SKIP without an adapter): a frame
//! test of function only (no pixel metric, no reference image, no CPU/GPU comparison; S37).
//!
//! The rig is the `area_shadow_frame.rs` rig without local lights: a white 6 x 6 m floor at y = 0,
//! a white 1 m box (albedo 0.8) on the origin, a sun of intensity 3 along (-0.3, -1, -0.2) with
//! ambient 0, the camera at eye (0, 2.5, 5) looking at the origin, GI at the default 96/5, and the
//! pipeline `prepass`, `forward` and `GiFeature::install` (gi.volume, gi.screen, gi.area-shadow,
//! gi.ssgi, gi.history).
//!
//! The shared inputs come from the pipeline; the test writes none. The prepass clears depth to
//! 1.0 (far: off-object pixels are sky, which the trace skips) and clears motion to zero; `forward`
//! writes the HDR chain; `gi.history` supplies the previous lit HDR (none on the first frame, so
//! `history_valid` is 0); `gi.screen` publishes the probe output. Level 0 of the volume spans
//! x in [-9.6, 9.6], y in [-8.0, 11.2] and z in [-4.8, 14.4] m around the camera's snapped origin,
//! which encloses the camera, floor and box; SSGI samples stay within 2 m of a pixel and in the
//! viewport.
//!
//! Asserted: the pass is installed once at order 44 and enabled; frames stay valid at every
//! quality; the part allocates exactly `ssgi_bytes` (1_843_248 at 320 x 180) and drops those bytes
//! when turned off or disabled; screen space off adds nothing.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{block_on, skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, OffscreenTarget};
use raa_gi::parts::ssgi::{SSGI_ORDER, SSGI_PASS, ssgi_bytes};
use raa_gi::{GiFeature, GiHandle, GiSettingsPatch, GiStats, SsgiQuality};
use raa_lit::{AttributedMesh, LitOptions, LitRenderer, SurfaceMaterial};
use raa_math::{Quat, Vec3};
use raa_pipeline::Pipeline;
use raa_scene::{Camera, DirectionalLight, RenderList, Scene, Transform};

const SIZE: (u32, u32) = (320, 180);
const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// The linear albedo of the floor and the box.
const WHITE: [f32; 3] = [0.8, 0.8, 0.8];

fn empty() -> AttributedMesh {
    AttributedMesh::from_mesh(Default::default())
}

fn finished(mut mesh: AttributedMesh) -> AttributedMesh {
    mesh.submeshes[0].index_count = mesh.base.indices.len() as u32;
    mesh
}

/// The 6 x 6 m single-sided floor quad at y = 0, its normal +Y and its winding counter-clockwise
/// seen from above.
fn floor_quad() -> AttributedMesh {
    let mut mesh = empty();
    for corner in [
        [-3.0, 0.0, -3.0],
        [-3.0, 0.0, 3.0],
        [3.0, 0.0, 3.0],
        [3.0, 0.0, -3.0],
    ] {
        mesh.base.positions.push(corner);
        mesh.base.normals.push([0.0, 1.0, 0.0]);
    }
    mesh.base.indices.extend([0, 1, 2, 0, 2, 3]);
    finished(mesh)
}

/// A 1 m cube centred at the origin: four vertices per face (hard edges, outward normals) and 12
/// counter-clockwise triangles.
fn unit_cube() -> AttributedMesh {
    let mut mesh = empty();
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
    finished(mesh)
}

/// The lit renderer, the GI feature and the scene above, rendered into an offscreen target of
/// `SIZE`. GI is enabled with `screen_space` as given; every other GI setting stays at its default.
struct Rig {
    pipeline: Rc<RefCell<Pipeline>>,
    lit: LitRenderer,
    handle: GiHandle,
    scene: Scene,
    list: RenderList,
    tags: Vec<u32>,
    slots: Vec<u32>,
    graph: FrameGraph,
    backbuffer: OffscreenTarget,
}

impl Rig {
    fn new(gpu: &GpuContext, screen_space: bool) -> Rig {
        let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, SURFACE, false)));
        let options = LitOptions {
            msaa: 1,
            ..LitOptions::default()
        };
        let mut lit = LitRenderer::new(gpu, &pipeline, options).expect("lit renderer");
        let handle =
            GiFeature::install(gpu, &pipeline, &mut lit, None, None).expect("the feature installs");

        let white = lit
            .materials_mut()
            .add(SurfaceMaterial {
                base_color: WHITE,
                roughness: 0.9,
                ..SurfaceMaterial::default()
            })
            .expect("material");
        let mut scene = Scene::with_capacity(2, 2);
        for (mesh, transform) in [
            (floor_quad(), Transform::IDENTITY),
            (
                unit_cube(),
                Transform::from_trs(Vec3::new(0.0, 0.5, 0.0), Quat::IDENTITY, Vec3::ONE),
            ),
        ] {
            let id = lit.upload(gpu, &mesh).expect("upload");
            let submesh = lit.submesh_ids(id).expect("sub-meshes")[0];
            let node = scene.add_node(None, transform).expect("node");
            scene.add_instance(node, submesh, white).expect("instance");
        }

        let mut camera = Camera {
            eye: Vec3::new(0.0, 2.5, 5.0),
            target: Vec3::ZERO,
            ..Camera::default()
        };
        camera.set_viewport(SIZE.0, SIZE.1);
        scene.camera = camera;
        scene.ambient.intensity = 0.0;
        let sun = DirectionalLight {
            direction: Vec3::new(-0.3, -1.0, -0.2),
            color: [1.0, 1.0, 1.0],
            intensity: 3.0,
        };
        scene.light = sun;
        handle.set_sun(&sun);
        handle
            .set(GiSettingsPatch {
                enabled: Some(true),
                screen_space: Some(screen_space),
                ..GiSettingsPatch::default()
            })
            .expect("settings");

        let mut graph = FrameGraph::new(SURFACE);
        Pipeline::install(&pipeline, &mut graph);
        graph.compile().expect("compile");

        Rig {
            pipeline,
            lit,
            handle,
            scene,
            list: RenderList::with_capacity(2),
            tags: Vec::new(),
            slots: Vec::new(),
            graph,
            backbuffer: OffscreenTarget::new(gpu, SIZE.0, SIZE.1, SURFACE),
        }
    }

    /// Applies a GI settings patch; a refused patch fails the test.
    fn set(&self, patch: GiSettingsPatch) {
        self.handle.set(patch).expect("settings");
    }

    /// Enables or disables the `gi.ssgi` pass in the pipeline.
    fn set_pass(&self, enabled: bool) {
        self.pipeline
            .borrow_mut()
            .set_enabled(SSGI_PASS, enabled)
            .expect("gi.ssgi is installed");
    }

    /// Runs one frame and returns the GI stats prepared for it.
    fn frame(&mut self, gpu: &GpuContext) -> GiStats {
        self.handle.set_view(&self.scene.camera, SIZE, 1);
        let stats = self
            .handle
            .prepare_frame(gpu, &mut self.scene, &mut self.lit);
        self.lit
            .prepare(
                gpu,
                &mut self.scene,
                &mut self.list,
                &mut self.tags,
                &mut self.slots,
            )
            .expect("lit prepare");
        let commands = self
            .graph
            .execute(gpu, self.backbuffer.view(), SIZE)
            .expect("execute");
        gpu.queue().submit([commands]);
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        stats
    }

    /// Runs `count` frames inside one validation error scope that must pop empty, and returns the
    /// stats of the last frame.
    fn run(&mut self, gpu: &GpuContext, count: u32) -> GiStats {
        let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let mut stats = None;
        for _ in 0..count {
            stats = Some(self.frame(gpu));
        }
        let error = block_on(errors.pop());
        assert!(error.is_none(), "GPU validation error: {error:?}");
        stats.expect("at least one frame")
    }
}

#[test]
fn ssgi_pass_is_installed_once_at_order_44_and_enabled() {
    let test = "ssgi_pass_is_installed_once_at_order_44_and_enabled";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let rig = Rig::new(&gpu, true);
    let rows = rig.pipeline.borrow().passes();
    let ssgi: Vec<_> = rows.iter().filter(|row| row.name == SSGI_PASS).collect();
    assert_eq!(ssgi.len(), 1, "exactly one {SSGI_PASS} row");
    assert!(ssgi[0].enabled, "{SSGI_PASS} is installed enabled");
    assert_eq!(ssgi[0].order, SSGI_ORDER, "the S12 order");
}

#[test]
fn ssgi_frames_are_valid_at_every_quality_and_allocate_ssgi_bytes() {
    let test = "ssgi_frames_are_valid_at_every_quality_and_allocate_ssgi_bytes";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let mut rig = Rig::new(&gpu, true);
    let on = rig.run(&gpu, 8).allocated_bytes;
    rig.set(GiSettingsPatch {
        screen_space: Some(false),
        ..GiSettingsPatch::default()
    });
    let off = rig.run(&gpu, 2).allocated_bytes;
    assert_eq!(ssgi_bytes(SIZE), 1_843_248, "the hand value at {SIZE:?}");
    assert_eq!(
        on.checked_sub(off),
        Some(ssgi_bytes(SIZE)),
        "the SSGI bytes of {SIZE:?}: {on} against {off}"
    );

    // The quality changes no allocation: each level stays valid and keeps the Medium bytes.
    for quality in [SsgiQuality::Low, SsgiQuality::High] {
        rig.set(GiSettingsPatch {
            screen_space: Some(true),
            screen_space_quality: Some(quality),
            ..GiSettingsPatch::default()
        });
        let bytes = rig.run(&gpu, 3).allocated_bytes;
        assert_eq!(bytes, on, "{quality:?} allocates the same bytes as Medium");
    }
}

#[test]
fn disabling_gi_ssgi_drops_its_bytes_and_stays_valid() {
    let test = "disabling_gi_ssgi_drops_its_bytes_and_stays_valid";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    // The reference is the same rig with screen space off, so no SSGI bytes are allocated.
    let mut reference = Rig::new(&gpu, false);
    let off = reference.run(&gpu, 2).allocated_bytes;

    let mut rig = Rig::new(&gpu, true);
    let on = rig.run(&gpu, 4).allocated_bytes;
    assert_eq!(
        on.checked_sub(off),
        Some(ssgi_bytes(SIZE)),
        "the SSGI bytes of {SIZE:?}: {on} against {off}"
    );
    rig.set_pass(false);
    let disabled = rig.run(&gpu, 2).allocated_bytes;
    assert_eq!(disabled, off, "a disabled {SSGI_PASS} drops its bytes");

    rig.set_pass(true);
    let restored = rig.run(&gpu, 2).allocated_bytes;
    assert_eq!(restored, on, "re-enabled {SSGI_PASS} allocates them again");
}

#[test]
fn screen_space_off_adds_nothing() {
    let test = "screen_space_off_adds_nothing";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let mut rig = Rig::new(&gpu, false);
    let first = rig.run(&gpu, 3).allocated_bytes;
    assert!(first > 0, "GI is active and allocates its volume");

    rig.set(GiSettingsPatch {
        screen_space: Some(true),
        ..GiSettingsPatch::default()
    });
    rig.set_pass(false);
    let second = rig.run(&gpu, 3).allocated_bytes;
    assert_eq!(
        second, first,
        "screen space with {SSGI_PASS} disabled adds nothing"
    );
}
