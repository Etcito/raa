//! The mid-tier area-light stochastic shadows wired into `LitRenderer::prepare` (M6).
//!
//! The software-GPU test (SKIP without an adapter) only prepares frames: no pass records, so
//! no depth or HDR target is read. Every shared input comes from the renderer itself: an empty
//! pipeline holds only the required lit passes, the scene camera is set explicitly, and one
//! area light is added through `local_lights_mut`. The scene has no instances, so nothing
//! outside the light registry and the camera feeds the stochastic selection.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{skip, software_gpu};
use raa_core::{GpuContext, wgpu};
use raa_lit::{
    AreaShadowSettings, AreaShadowTier, AreaStochasticFrame, LitOptions, LitRenderer, LocalLight,
    LocalLightKind, StochasticShadowSettings, rect_distance, stochastic_weight, table_cells,
    table_limit, table_slot,
};
use raa_math::Vec3;
use raa_pipeline::Pipeline;
use raa_scene::{Camera, RenderList, Scene};

const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Camera eye and target of the frame the test prepares.
const EYE: Vec3 = Vec3::new(0.0, 2.0, 2.0);
const TARGET: Vec3 = Vec3::new(0.0, 0.0, -4.0);
/// The area light of the frame, its unit travel direction, its size and its range.
const POSITION: Vec3 = Vec3::new(0.0, 3.0, -4.0);
const DIRECTION: Vec3 = Vec3::new(0.0, -1.0, 0.0);
const SIZE: [f32; 2] = [1.0, 1.0];
const RANGE: f32 = 10.0;
/// The band of the default area-shadow settings.
const BAND: f32 = 5.0;

/// An empty pipeline, a lit renderer with the default settings, the camera of the frame and one
/// area light.
struct Rig {
    lit: LitRenderer,
    scene: Scene,
    list: RenderList,
    tags: Vec<u32>,
    slots: Vec<u32>,
    _pipeline: Rc<RefCell<Pipeline>>,
}

impl Rig {
    fn new(gpu: &GpuContext) -> Rig {
        let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, SURFACE, false)));
        let opts = LitOptions {
            msaa: 1,
            ..LitOptions::default()
        };
        let mut lit = LitRenderer::new(gpu, &pipeline, opts).expect("lit renderer");
        lit.local_lights_mut()
            .add(LocalLight {
                kind: LocalLightKind::Area,
                position: POSITION.to_array(),
                direction: DIRECTION.to_array(),
                size: SIZE,
                range: RANGE,
                ..LocalLight::default()
            })
            .expect("area light");
        let mut scene = Scene::with_capacity(0, 0);
        scene.camera = Camera {
            eye: EYE,
            target: TARGET,
            ..Camera::default()
        };
        Rig {
            lit,
            scene,
            list: RenderList::with_capacity(0),
            tags: Vec::new(),
            slots: Vec::new(),
            _pipeline: pipeline,
        }
    }

    /// Prepares one frame (no pass is recorded).
    fn prepare(&mut self, gpu: &GpuContext) {
        self.lit
            .prepare(
                gpu,
                &mut self.scene,
                &mut self.list,
                &mut self.tags,
                &mut self.slots,
            )
            .expect("prepare");
    }

    /// The frame the handle publishes for the last prepare.
    fn frame(&self) -> AreaStochasticFrame {
        self.lit.handle().area_stochastic()
    }
}

#[test]
fn area_stochastic_wires_into_prepare() {
    let test = "area_stochastic_wires_into_prepare";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let mut rig = Rig::new(&gpu);
    let standard = StochasticShadowSettings::standard();

    // (1) Defaults before the first prepare, then one mid-tier slot after it.
    assert_eq!(rig.lit.area_stochastic(), standard);
    assert_eq!(rig.lit.active_area_stochastic(), 0);
    assert_eq!(
        rig.lit.handle().area_stochastic(),
        AreaStochasticFrame::empty()
    );
    rig.prepare(&gpu);
    assert_eq!(rig.lit.active_area_stochastic(), 1);
    let frame = rig.frame();
    assert_eq!(frame.active, 1);
    let distance = rect_distance(EYE, POSITION, DIRECTION, SIZE);
    let weight = stochastic_weight(&standard, BAND, distance);
    assert_eq!(table_slot(&frame.table, 0), Some((0, weight)));
    assert_eq!(table_limit(&frame.table), 4);
    assert_eq!(table_cells(&frame.table), 4);
    assert_eq!(frame.generations[0], 1);

    // (2) The same light keeps slot 0, so its generation does not move.
    rig.prepare(&gpu);
    assert_eq!(rig.frame().generations[0], 1);

    // (3) Nine mid-tier lights are refused and leave the settings alone.
    let before = rig.lit.area_stochastic();
    let error = rig
        .lit
        .set_area_stochastic(StochasticShadowSettings {
            count: 9,
            ..standard
        })
        .expect_err("nine mid-tier lights are refused");
    assert_eq!(error.code(), "light.invalid_area_shadows");
    assert_eq!(rig.lit.area_stochastic(), before);

    // (4) Count 0 empties the table and releases slot 0.
    rig.lit
        .set_area_stochastic(StochasticShadowSettings {
            count: 0,
            ..standard
        })
        .expect("accepted settings");
    rig.prepare(&gpu);
    let frame = rig.frame();
    assert_eq!(rig.lit.active_area_stochastic(), 0);
    assert_eq!(frame.active, 0);
    assert_eq!(table_limit(&frame.table), 0);
    assert_eq!(frame.generations[0], 2);

    // (5) The mid tier only fills under Auto: a Cone tier empties the table, Auto fills it again.
    rig.lit
        .set_area_stochastic(StochasticShadowSettings {
            count: 1,
            cells: 9,
            ..standard
        })
        .expect("accepted settings");
    let cone = AreaShadowSettings {
        tier: AreaShadowTier::Cone,
        ..rig.lit.area_shadows()
    };
    rig.lit.set_area_shadows(cone).expect("accepted settings");
    rig.prepare(&gpu);
    assert_eq!(table_limit(&rig.frame().table), 0);

    let auto = AreaShadowSettings {
        tier: AreaShadowTier::Auto,
        ..rig.lit.area_shadows()
    };
    rig.lit.set_area_shadows(auto).expect("accepted settings");
    rig.prepare(&gpu);
    let frame = rig.frame();
    assert!(table_slot(&frame.table, 0).is_some());
    assert_eq!(table_cells(&frame.table), 9);
}
