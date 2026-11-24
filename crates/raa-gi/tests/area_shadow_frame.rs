//! The `gi.area-shadow` pass (S12, order 43) end to end on a software GPU (SKIP without an
//! adapter): one frame test of the mid tier of the area-light stochastic shadows. It checks
//! function only, with no pixel metric and no reference image (S37: GPU results are judged on
//! the page).
//!
//! The scene is a 6 x 6 m white floor quad at y = 0, a 1 m white box standing on it at the
//! origin, and one area light at (0, 3, 0) with direction (0, -1, 0), size [1, 1] and range 10.
//! The camera has eye (0, 2.5, 5) and target (0, 0, 0); the sun intensity is 0 and the ambient
//! is 0; GI is enabled at the default quality 96/5. The pipeline is the lit renderer's
//! (`prepass`, `forward`), the light tiling (`light.tiles`, so the forward pass sees the area
//! light) and the GI feature (`gi.volume`, `gi.screen`, `gi.history`) plus the part under test
//! (`gi.area-shadow`, order 43); nothing else is installed.
//!
//! Every shared input is set up here:
//! - depth, the S12 normal target and the S12 motion target come from the lit prepass of these
//!   two meshes: depth is cleared to 1.0 = far, so the pixels outside the floor are sky pixels
//!   that every pass skips, and motion is cleared to 0;
//! - the HDR chain is cleared and written by `forward`;
//! - the depth pyramid comes from `gi.screen`.
//!
//! Spatial extent: at the default quality 96/5, level 0 of the volume spans 96 voxels of 0.2 m
//! (19.2 m) around the camera's snapped origin. For the eye (0, 2.5, 5) that origin is
//! (-48, -40, -24) voxels, so level 0 covers x in [-9.6, 9.6], y in [-8.0, 11.2] and z in
//! [-4.8, 14.4] m: the camera, the floor (x, z in [-3, 3]), the box (x, z in [-0.5, 0.5], y in
//! [0, 1]) and the light at (0, 3, 0) all lie inside it.
//!
//! The asserted properties are functional: the pass is installed once, enabled, at order 43; 8
//! frames raise no validation error; the lit handle publishes one occupied mid-tier slot with
//! the standard limit 4; the area-shadow resources add exactly `area_shadow_bytes((320, 180), 2)`
//! on top of the GI bytes of the same rig with `count 0`; and with GI disabled nothing is
//! allocated and 2 further frames stay valid.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{block_on, skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, GpuOptions, OffscreenTarget};
use raa_gi::parts::area_shadow::{
    AREA_SHADOW_ORDER, AREA_SHADOW_PASS, area_restir_bytes, area_shadow_bytes,
};
use raa_gi::{GiFeature, GiHandle, GiSettingsPatch, GiStats};
use raa_lit::{
    AttributedMesh, LitOptions, LitRenderer, LocalLight, LocalLightKind, StochasticMode,
    StochasticShadowSettings, SurfaceMaterial, install_local_lights, table_limit,
};
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

/// The lit renderer, the GI feature (with the area-light tiling it needs) and the scene above,
/// rendered into an offscreen target of `SIZE`.
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
    stats: Option<GiStats>,
}

fn add_area_light(lit: &mut LitRenderer, x: f32) {
    lit.local_lights_mut()
        .add(LocalLight {
            kind: LocalLightKind::Area,
            position: [x, 3.0, 0.0],
            direction: [0.0, -1.0, 0.0],
            color: [1.0, 1.0, 1.0],
            intensity: 4.0,
            range: 10.0,
            size: [1.0, 1.0],
            ..LocalLight::default()
        })
        .expect("area light");
}

impl Rig {
    fn new(gpu: &GpuContext, gi_enabled: bool) -> Rig {
        let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, SURFACE, false)));
        let options = LitOptions {
            msaa: 1,
            ..LitOptions::default()
        };
        let mut lit = LitRenderer::new(gpu, &pipeline, options).expect("lit renderer");
        install_local_lights(gpu, &pipeline, &mut lit).expect("light tiling installs");
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

        add_area_light(&mut lit, 0.0);

        let mut camera = Camera {
            eye: Vec3::new(0.0, 2.5, 5.0),
            target: Vec3::ZERO,
            ..Camera::default()
        };
        camera.set_viewport(SIZE.0, SIZE.1);
        scene.camera = camera;
        scene.ambient.intensity = 0.0;
        let sun = DirectionalLight {
            direction: Vec3::new(0.0, -1.0, 0.0),
            color: [1.0, 1.0, 1.0],
            intensity: 0.0,
        };
        scene.light = sun;
        handle.set_sun(&sun);
        handle
            .set(GiSettingsPatch {
                enabled: Some(gi_enabled),
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
            stats: None,
        }
    }

    fn frame(&mut self, gpu: &GpuContext) {
        let camera = self.scene.camera;
        self.handle.set_view(&camera, SIZE, 1);
        self.handle
            .set_local_light_count(self.lit.local_lights().active_count());
        self.stats = Some(
            self.handle
                .prepare_frame(gpu, &mut self.scene, &mut self.lit),
        );
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
        self.graph.notify_submitted();
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
    }

    fn stats(&self) -> GiStats {
        self.stats.clone().expect("frames ran")
    }
}

#[test]
fn area_shadow_pass_publishes_the_mask_and_allocates_its_resources() {
    let test = "area_shadow_pass_publishes_the_mask_and_allocates_its_resources";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let mut rig = Rig::new(&gpu, true);

    // (1) One row of the S12 pass, enabled, at order 43.
    let rows = rig.pipeline.borrow().passes();
    let shadow: Vec<_> = rows
        .iter()
        .filter(|row| row.name == AREA_SHADOW_PASS)
        .collect();
    assert_eq!(shadow.len(), 1, "exactly one {AREA_SHADOW_PASS} row");
    assert!(shadow[0].enabled, "{AREA_SHADOW_PASS} is installed enabled");
    assert_eq!(shadow[0].order, AREA_SHADOW_ORDER, "the S12 order");

    // (2) Eight frames inside the validation scope; the scope must pop clean.
    for _ in 0..8 {
        rig.frame(&gpu);
    }
    let error = block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");

    // (3) The standard settings select the one area light into one mid-tier slot, limit 4.
    let frame = rig.lit.handle().area_stochastic();
    assert_eq!(frame.active, 1, "one occupied mid-tier slot");
    assert_eq!(table_limit(&frame.table), 4, "the standard limit");

    // (4) The area-shadow resources are the difference between the standard settings and none.
    let with_shadow = rig.stats().allocated_bytes;
    rig.lit
        .set_area_stochastic(StochasticShadowSettings {
            count: 0,
            ..StochasticShadowSettings::standard()
        })
        .expect("zero mid-tier lights is a valid setting");
    for _ in 0..2 {
        rig.frame(&gpu);
    }
    let without_shadow = rig.stats().allocated_bytes;
    assert!(
        with_shadow > without_shadow,
        "the area-shadow resources added no bytes: {with_shadow} against {without_shadow}"
    );
    assert_eq!(
        with_shadow - without_shadow,
        area_shadow_bytes(SIZE, 2),
        "the area-shadow bytes of {SIZE:?} and 2 history groups"
    );
}

#[test]
fn disabled_gi_allocates_nothing_and_records_no_validation_error() {
    let test = "disabled_gi_allocates_nothing_and_records_no_validation_error";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let mut rig = Rig::new(&gpu, false);
    for _ in 0..2 {
        rig.frame(&gpu);
    }
    let stats = rig.stats();
    assert!(!stats.active, "GI is inactive: {stats:?}");
    assert_eq!(stats.allocated_bytes, 0, "GI disabled allocates nothing");

    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    for _ in 0..2 {
        rig.frame(&gpu);
    }
    let error = block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");
    assert_eq!(
        rig.stats().allocated_bytes,
        0,
        "a further 2 frames allocate nothing"
    );
}

/// The ReSTIR ultra tier on the same rig and inputs as the file header states: 8 Restir frames
/// stay valid, the lit frame publishes mode Restir with one occupied slot, the Restir resources
/// add exactly `area_restir_bytes(SIZE)` over `count 0`, and the standard (Stratified) settings
/// add exactly `area_shadow_bytes(SIZE, 2)` over it.
#[test]
fn restir_mode_publishes_the_mask_and_allocates_its_resources() {
    let test = "restir_mode_publishes_the_mask_and_allocates_its_resources";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let mut rig = Rig::new(&gpu, true);
    let restir = StochasticShadowSettings {
        mode: StochasticMode::Restir,
        ..StochasticShadowSettings::standard()
    };
    rig.lit
        .set_area_stochastic(restir)
        .expect("the Restir mode is a valid setting");
    for _ in 0..8 {
        rig.frame(&gpu);
    }
    let frame = rig.lit.handle().area_stochastic();
    assert_eq!(frame.mode, StochasticMode::Restir, "the published mode");
    assert_eq!(frame.active, 1, "one occupied mid-tier slot");
    let with_restir = rig.stats().allocated_bytes;

    rig.lit
        .set_area_stochastic(StochasticShadowSettings { count: 0, ..restir })
        .expect("zero mid-tier lights is a valid setting");
    for _ in 0..2 {
        rig.frame(&gpu);
    }
    let without = rig.stats().allocated_bytes;
    assert_eq!(
        with_restir.checked_sub(without),
        Some(area_restir_bytes(SIZE)),
        "the ReSTIR bytes of {SIZE:?}: {with_restir} against {without}"
    );
    assert_eq!(area_restir_bytes(SIZE), 5_265_621);

    rig.lit
        .set_area_stochastic(StochasticShadowSettings::standard())
        .expect("the standard settings are valid");
    for _ in 0..2 {
        rig.frame(&gpu);
    }
    let stratified = rig.stats().allocated_bytes;
    assert_eq!(
        stratified.checked_sub(without),
        Some(area_shadow_bytes(SIZE, 2)),
        "the stratified bytes of {SIZE:?} and 2 groups: {stratified} against {without}"
    );
    let error = block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");
}

#[test]
#[cfg_attr(not(target_os = "windows"), ignore = "Windows GPU regression")]
fn stratified_trace_tap_keeps_weighted_pairs_for_one_four_and_eight_lights() {
    let gpu = block_on(GpuContext::headless(&GpuOptions {
        allow_software: false,
        ..GpuOptions::default()
    }))
    .expect("Windows hardware GPU");
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let mut rig = Rig::new(&gpu, true);
    let mut lights = 1;
    for count in [1, 4, 8] {
        while lights < count {
            add_area_light(&mut rig.lit, 0.12 * lights as f32);
            lights += 1;
        }
        rig.lit
            .set_area_stochastic(StochasticShadowSettings {
                count,
                ..StochasticShadowSettings::standard()
            })
            .expect("stratified slot budget");
        rig.frame(&gpu);

        let pipeline = rig.pipeline.borrow();
        let raw = pipeline
            .resources()
            .get(AREA_SHADOW_PASS, "raw")
            .expect("raw tap");
        let (width, height) = SIZE;
        let row = (width * 8).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let layers = count.div_ceil(2);
        let staging = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("area-shadow.raw-test"),
            size: u64::from(row * height * layers),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            raw.view.texture().as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: layers,
            },
        );
        gpu.queue().submit([encoder.finish()]);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, |result| result.expect("map raw pairs"));
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll raw pairs");
        let data = staging.slice(..).get_mapped_range().expect("mapped pairs");
        for slot in 0..count {
            let mut covered = false;
            let mut umbra = false;
            for y in 0..height {
                for x in 0..width {
                    let offset = ((slot / 2 * height + y) * row + x * 8 + slot % 2 * 4) as usize;
                    // Nonnegative binary16 values have the same order as their unsigned bits.
                    let visible = u16::from_le_bytes([data[offset], data[offset + 1]]);
                    let weight = u16::from_le_bytes([data[offset + 2], data[offset + 3]]);
                    assert!(
                        visible <= weight && weight < 0x7c00,
                        "invalid pair in slot {slot}"
                    );
                    covered |= weight != 0;
                    umbra |= weight != 0 && visible == 0;
                }
            }
            assert!(covered, "no receiver for slot {slot} at limit {count}");
            if count == 1 {
                assert!(umbra, "the blocker left no zero-visibility samples");
            }
        }
    }
    assert!(
        block_on(errors.pop()).is_none(),
        "area-shadow validation error"
    );
}
