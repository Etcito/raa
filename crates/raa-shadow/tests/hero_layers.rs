//! `ShadowHandle::hero_layers` follows the area-light tier the lit renderer prepared (M4).
//!
//! The software-GPU test (SKIP without an adapter) builds one real frame: `LitRenderer::prepare`
//! writes the area-light hero targets, then the public `FrameGraph::execute` runs the
//! `shadow.csm` pass whose prepare reads them. Every shared input is set up here: the scene
//! camera, the area light (3 m above and 4 m in front of the camera), the scene/list capacity
//! and the caster view. The empty pipeline holds only the lit passes, and no mesh instance
//! exists, so the hero layer is created and cleared without a caster draw.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, OffscreenTarget, wgpu};
use raa_lit::{
    AreaShadowSettings, AreaShadowTier, LitOptions, LitRenderer, LocalLight, LocalLightKind,
};
use raa_math::Vec3;
use raa_pipeline::Pipeline;
use raa_scene::{Camera, RenderList, Scene};
use raa_shadow::{ShadowFeature, ShadowHandle};

const TEST: &str = "hero_layers_follow_the_area_shadow_tier";
const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SIZE: (u32, u32) = (32, 32);

/// Camera of the frame: eye 2 m up, 2 m back, looking at 4 m down -Z.
const EYE: Vec3 = Vec3::new(0.0, 2.0, 2.0);
const TARGET: Vec3 = Vec3::new(0.0, 0.0, -4.0);
/// The area light: 3 m above and 4 m in front of the camera, pointing down.
const POSITION: Vec3 = Vec3::new(0.0, 5.0, -2.0);
const DIRECTION: Vec3 = Vec3::new(0.0, -1.0, 0.0);
const RANGE: f32 = 10.0;

fn sun() -> Vec3 {
    Vec3::new(-0.4, -1.0, -0.3).normalize_or_zero()
}

/// A lit renderer with one area light, the installed shadow feature and a compiled frame graph.
struct Rig {
    lit: LitRenderer,
    shadow: ShadowHandle,
    scene: Scene,
    list: RenderList,
    tags: Vec<u32>,
    slots: Vec<u32>,
    pipeline: Rc<RefCell<Pipeline>>,
    graph: FrameGraph,
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
        let shadow = ShadowFeature::install(gpu, &pipeline, &mut lit).expect("shadow feature");
        let mut scene = Scene::with_capacity(1, 1);
        scene.camera = Camera {
            eye: EYE,
            target: TARGET,
            ..Camera::default()
        };
        let mut graph = FrameGraph::new(SURFACE);
        Pipeline::install(&pipeline, &mut graph);
        graph.compile().expect("compiles");
        Rig {
            lit,
            shadow,
            scene,
            list: RenderList::with_capacity(1),
            tags: Vec::with_capacity(1),
            slots: Vec::with_capacity(1),
            pipeline,
            graph,
        }
    }

    /// Prepares the lit frame, then runs the graph including the `shadow.csm` prepare/record.
    fn frame(&mut self, gpu: &GpuContext) {
        let camera = self.scene.camera;
        self.shadow.set_view(
            &camera.view(),
            camera.fov_y_rad,
            camera.aspect,
            camera.near,
            camera.far,
            sun(),
        );
        self.lit
            .prepare(
                gpu,
                &mut self.scene,
                &mut self.list,
                &mut self.tags,
                &mut self.slots,
            )
            .expect("prepare");
        let target = OffscreenTarget::new(gpu, SIZE.0, SIZE.1, SURFACE);
        let commands = self
            .graph
            .execute(gpu, target.view(), SIZE)
            .expect("frame runs");
        gpu.queue().submit([commands]);
    }
}

#[test]
fn hero_layers_follow_the_area_shadow_tier() {
    let Some(gpu) = software_gpu() else {
        skip(TEST, "software_gpu returned None");
        return;
    };
    let mut rig = Rig::new(&gpu);

    // (1) Hero tier with one requested layer: one hero layer, the cascades unchanged.
    let hero = AreaShadowSettings {
        tier: AreaShadowTier::Hero,
        hero_count: 1,
        ..AreaShadowSettings::default()
    };
    rig.lit.set_area_shadows(hero).expect("accepted settings");
    rig.frame(&gpu);
    assert_eq!(rig.shadow.hero_layers(), 1);
    let cascades = rig.shadow.cascade_targets().expect("bound after prepare");
    assert_eq!(cascades.count, 4);
    assert_eq!(cascades.layer_views.len(), 4);

    // (2) Cone tier: no hero layer whatever the count.
    let cone = AreaShadowSettings {
        tier: AreaShadowTier::Cone,
        hero_count: 1,
        ..AreaShadowSettings::default()
    };
    rig.lit.set_area_shadows(cone).expect("accepted settings");
    rig.frame(&gpu);
    assert_eq!(rig.shadow.hero_layers(), 0);
    let cascades = rig.shadow.cascade_targets().expect("bound after prepare");
    assert_eq!(cascades.count, 4);
    assert_eq!(cascades.layer_views.len(), 4);

    // (3) Disabling `shadow.csm` drops the hero layer count immediately.
    rig.lit.set_area_shadows(hero).expect("accepted settings");
    rig.frame(&gpu);
    assert_eq!(rig.shadow.hero_layers(), 1);
    rig.pipeline
        .borrow_mut()
        .set_enabled("shadow.csm", false)
        .expect("disabled");
    assert_eq!(rig.shadow.hero_layers(), 0);

    std::mem::forget(rig.graph);
    std::mem::forget(rig.pipeline);
}
