//! The area-light soft-shadow tiers wired into `LitRenderer::prepare` (M4).
//!
//! The software-GPU test (SKIP without an adapter) only prepares frames: no pass records, so
//! no depth or HDR target is read. Every shared input comes from the renderer itself: an empty
//! pipeline holds only the required lit passes, the scene camera is set explicitly, and one
//! area light is added through `local_lights_mut`. The scene has no instances, so nothing
//! outside the light registry and the camera feeds the hero selection.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{skip, software_gpu};
use raa_core::{GpuContext, wgpu};
use raa_lit::{
    AreaShadowSettings, AreaShadowTier, LitOptions, LitRenderer, LocalLight, LocalLightKind,
    hero_view_proj,
};
use raa_math::Vec3;
use raa_pipeline::Pipeline;
use raa_scene::{Camera, RenderList, Scene};

const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Camera eye and target of the frame the test prepares.
const EYE: Vec3 = Vec3::new(0.0, 2.0, 2.0);
const TARGET: Vec3 = Vec3::new(0.0, 0.0, -4.0);
/// The area light of the frame, its unit travel direction and its range.
const POSITION: Vec3 = Vec3::new(0.0, 3.0, -4.0);
const DIRECTION: Vec3 = Vec3::new(0.0, -1.0, 0.0);
const RANGE: f32 = 10.0;

/// An empty pipeline, a lit renderer with the default (zero-hero) settings, the camera of the
/// frame and one area light.
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
                size: [1.0, 1.0],
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
}

#[test]
fn area_shadow_tiers_wire_into_prepare() {
    let test = "area_shadow_tiers_wire_into_prepare";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let mut rig = Rig::new(&gpu);

    // (1) Defaults: auto tier, no hero layer requested, no heroes selected.
    rig.prepare(&gpu);
    assert_eq!(rig.lit.area_shadows(), AreaShadowSettings::default());
    assert_eq!(rig.lit.active_area_heroes(), 0);

    // (2) Hero tier with one requested layer: the area light holds slot 0.
    let hero = AreaShadowSettings {
        tier: AreaShadowTier::Hero,
        hero_count: 1,
        ..AreaShadowSettings::default()
    };
    rig.lit.set_area_shadows(hero).expect("accepted settings");
    rig.prepare(&gpu);
    assert_eq!(rig.lit.active_area_heroes(), 1);
    let handle = rig.lit.handle();
    assert_eq!(handle.area_hero_capacity(), 1);
    let targets = handle.area_hero_targets();
    let expected = hero_view_proj(POSITION, DIRECTION, RANGE);
    assert_eq!(targets[0], Some(expected));
    assert!(targets[1..].iter().all(|target| target.is_none()));

    // (3) Cone tier: no hero layer, no target and zero capacity, whatever the count.
    let cone = AreaShadowSettings {
        tier: AreaShadowTier::Cone,
        hero_count: 1,
        ..AreaShadowSettings::default()
    };
    rig.lit.set_area_shadows(cone).expect("accepted settings");
    rig.prepare(&gpu);
    assert_eq!(rig.lit.active_area_heroes(), 0);
    let handle = rig.lit.handle();
    assert_eq!(handle.area_hero_capacity(), 0);
    assert!(
        handle
            .area_hero_targets()
            .iter()
            .all(|target| target.is_none())
    );

    // (4) More than four hero layers is refused and leaves the settings alone.
    let before = rig.lit.area_shadows();
    let error = rig
        .lit
        .set_area_shadows(AreaShadowSettings {
            hero_count: 5,
            ..AreaShadowSettings::default()
        })
        .expect_err("five hero layers are refused");
    assert_eq!(error.code(), "light.invalid_area_shadows");
    assert_eq!(rig.lit.area_shadows(), before);
}
