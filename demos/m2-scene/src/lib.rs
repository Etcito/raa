//! M2 scene demo: a ground slab and 1000 lit, depth-tested cubes in a jittered lattice
//! under 10 rotating parent nodes, with Flat and PbrLite materials, rendered through the
//! RAA forward renderer on WebGPU. The scene app builds for wasm32 and for host unit tests;
//! the wasm-bindgen API lives behind `cfg(target_arch = "wasm32")`. The `alloc-count` feature
//! links the counting global allocator and reports allocations per frame.

#[cfg(feature = "alloc-count")]
#[global_allocator]
static A: raa_alloc_count::CountingAlloc = raa_alloc_count::CountingAlloc;

#[cfg(any(target_arch = "wasm32", test))]
#[cfg_attr(
    not(target_arch = "wasm32"),
    allow(dead_code, reason = "host builds only run the CPU unit tests")
)]
mod app {
    use raa_core::{FrameGraph, GpuContext, MaterialId, MeshId, RaaError};
    use raa_geom::builtin::DebugCube;
    use raa_geom::{GeometryProvider, MeshData, ParamValue, Params, Quality};
    use raa_material::Material;
    use raa_math::{Quat, Vec3};
    use raa_render::{Renderer, RendererOptions};
    use raa_scene::{NodeId, OrbitController, RenderList, Scene, SceneError, Transform};
    use raa_web::{FrameCounters, FrameInput, WebApp};

    /// Cubes per lattice axis.
    pub(crate) const LATTICE: usize = 10;
    /// Cube instances in the lattice (the ground slab excluded).
    pub(crate) const CUBES: usize = LATTICE * LATTICE * LATTICE;
    /// Distance between neighbouring lattice points, in metres.
    const SPACING: f32 = 1.1;
    /// Half edge of a lattice cube (the uploaded cube has half 1.0, scaled per node).
    const CUBE_HALF: f32 = 0.3;
    /// Largest jitter of a lattice point along each axis.
    const JITTER: f32 = 0.15;
    /// Fixed seed of the jitter LCG.
    const JITTER_SEED: u32 = 0x2545_f491;
    /// Ground slab scale and height.
    const GROUND_SCALE: Vec3 = Vec3::new(14.0, 0.2, 14.0);
    const GROUND_Y: f32 = -6.5;
    /// Rotation speed about Y of each layer's parent node, in radians per millisecond.
    const LAYER_SPEEDS: [f64; LATTICE] = [
        0.000_20, -0.000_26, 0.000_32, -0.000_38, 0.000_44, -0.000_50, 0.000_56, -0.000_62,
        0.000_68, -0.000_74,
    ];
    /// Orbit yaw per millisecond.
    const ORBIT_SPEED: f32 = 0.000_05;

    /// The ground's Flat grey material.
    pub(crate) fn ground_material() -> Material {
        Material::flat([0.18, 0.18, 0.2])
    }

    /// The 8 lattice materials: 4 Flat with distinct linear colours, then 4 PbrLite with
    /// roughness 0.2/0.4/0.6/0.8 and metallic 0/0.5/0/1.
    pub(crate) fn lattice_materials() -> [Material; 8] {
        [
            Material::flat([0.80, 0.18, 0.12]),
            Material::flat([0.15, 0.65, 0.22]),
            Material::flat([0.12, 0.30, 0.85]),
            Material::flat([0.85, 0.70, 0.15]),
            Material::pbr([0.90, 0.35, 0.10], 0.2, 0.0),
            Material::pbr([0.60, 0.20, 0.75], 0.4, 0.5),
            Material::pbr([0.10, 0.70, 0.70], 0.6, 0.0),
            Material::pbr([0.95, 0.80, 0.45], 0.8, 1.0),
        ]
    }

    /// Fixed-seed integer LCG (Numerical Recipes constants) for the jitter.
    struct Lcg(u32);

    impl Lcg {
        /// Next value in `[-JITTER, JITTER]`.
        fn jitter(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // The top 24 bits are exactly representable in f32.
            let unit = (self.0 >> 8) as f32 / (1u32 << 24) as f32;
            (unit * 2.0 - 1.0) * JITTER
        }
    }

    /// Lattice coordinate of index `i` in `0..LATTICE`, centred on 0.
    fn lattice_coord(i: usize) -> f32 {
        (i as f32 - (LATTICE as f32 - 1.0) * 0.5) * SPACING
    }

    /// Builds the ground slab and the 1000-cube lattice into `scene`. The lattice hangs
    /// under one parent node per y-layer; the parents are returned bottom layer first.
    /// Lattice cube `i` gets `materials[i % 8]`. Deterministic.
    pub(crate) fn build_scene(
        scene: &mut Scene,
        cube_mesh: MeshId,
        ground_material: MaterialId,
        materials: &[MaterialId; 8],
    ) -> Result<[NodeId; LATTICE], SceneError> {
        let ground =
            Transform::from_trs(Vec3::new(0.0, GROUND_Y, 0.0), Quat::IDENTITY, GROUND_SCALE);
        let ground = scene.add_node(None, ground)?;
        scene.add_instance(ground, cube_mesh, ground_material)?;

        let mut rng = Lcg(JITTER_SEED);
        let mut parents = [ground; LATTICE];
        let mut cube = 0;
        for (y, parent_slot) in parents.iter_mut().enumerate() {
            let layer = Transform::from_trs(
                Vec3::new(0.0, lattice_coord(y), 0.0),
                Quat::IDENTITY,
                Vec3::ONE,
            );
            let parent = scene.add_node(None, layer)?;
            *parent_slot = parent;
            for z in 0..LATTICE {
                for x in 0..LATTICE {
                    let offset = Vec3::new(rng.jitter(), rng.jitter(), rng.jitter());
                    let local = Transform::from_trs(
                        Vec3::new(lattice_coord(x), 0.0, lattice_coord(z)) + offset,
                        Quat::IDENTITY,
                        Vec3::splat(CUBE_HALF),
                    );
                    let node = scene.add_node(Some(parent), local)?;
                    scene.add_instance(node, cube_mesh, materials[cube % materials.len()])?;
                    cube += 1;
                }
            }
        }
        Ok(parents)
    }

    /// Everything created by `init`.
    struct Running {
        renderer: Renderer,
        scene: Scene,
        list: RenderList,
        parents: [NodeId; LATTICE],
        orbit: OrbitController,
    }

    /// The demo application: the forward renderer over the lattice scene.
    #[derive(Default)]
    pub(crate) struct ScenesApp {
        running: Option<Running>,
    }

    fn scene_error(e: SceneError) -> RaaError {
        RaaError::Resource(format!("{}: {e}", e.code()))
    }

    impl WebApp for ScenesApp {
        fn init(
            &mut self,
            gpu: &GpuContext,
            graph: &mut FrameGraph,
            size: (u32, u32),
            surface_format: wgpu::TextureFormat,
            surface_is_srgb: bool,
        ) -> Result<(), RaaError> {
            let mut renderer = Renderer::new(
                gpu,
                surface_format,
                surface_is_srgb,
                RendererOptions::default(),
            )?;

            let mut mesh = MeshData::new();
            let params = Params::new().with("half", ParamValue::Scalar(1.0));
            DebugCube
                .generate(&params, Quality::LOD0, &mut mesh)
                .map_err(|e| RaaError::Resource(format!("{}: {e}", e.code())))?;
            let cube = renderer.upload_mesh(gpu, &mesh)?;

            let material_error = |e: raa_material::MaterialError| {
                RaaError::Resource(format!("{}: {}", e.code, e.message))
            };
            let library = renderer.materials_mut();
            let ground = library.add(ground_material()).map_err(material_error)?;
            let mut materials = [ground; 8];
            for (slot, m) in materials.iter_mut().zip(lattice_materials()) {
                *slot = library.add(m).map_err(material_error)?;
            }

            let mut scene = Scene::with_capacity(1 + LATTICE + CUBES, 1 + CUBES);
            let parents = build_scene(&mut scene, cube, ground, &materials).map_err(scene_error)?;
            renderer.install(graph);

            let mut orbit = OrbitController::new(Vec3::ZERO, 22.0);
            orbit.pitch_rad = 25.0_f32.to_radians();
            orbit.apply(&mut scene.camera);
            scene.camera.set_viewport(size.0, size.1);

            self.running = Some(Running {
                renderer,
                scene,
                list: RenderList::with_capacity(1 + CUBES),
                parents,
                orbit,
            });
            Ok(())
        }

        fn update(
            &mut self,
            gpu: &GpuContext,
            input: &FrameInput,
        ) -> Result<FrameCounters, RaaError> {
            let Some(app) = self.running.as_mut() else {
                return Err(RaaError::Resource(
                    "demo.not_initialised: update called before init".to_string(),
                ));
            };
            for (y, &parent) in app.parents.iter().enumerate() {
                let angle = (input.now_ms * LAYER_SPEEDS[y]) as f32;
                let local = Transform::from_trs(
                    Vec3::new(0.0, lattice_coord(y), 0.0),
                    Quat::from_rotation_y(angle),
                    Vec3::ONE,
                );
                app.scene.set_local(parent, local).map_err(scene_error)?;
            }
            app.orbit.rotate(input.dt_ms as f32 * ORBIT_SPEED, 0.0);
            app.orbit.apply(&mut app.scene.camera);
            let counts = app.renderer.prepare(gpu, &mut app.scene, &mut app.list)?;
            Ok(FrameCounters {
                instances: CUBES as u32,
                draw_calls: counts.draw_calls,
                triangles: counts.triangles,
            })
        }

        fn resize(&mut self, _gpu: &GpuContext, size: (u32, u32)) {
            if let Some(app) = self.running.as_mut() {
                app.scene.camera.set_viewport(size.0, size.1);
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use raa_material::ShadingModel;

        fn built() -> Scene {
            let mut scene = Scene::with_capacity(1 + LATTICE + CUBES, 1 + CUBES);
            let materials: [MaterialId; 8] = std::array::from_fn(|i| MaterialId(i as u32 + 2));
            build_scene(&mut scene, MeshId(0), MaterialId(1), &materials).expect("build");
            scene.update_world();
            scene
        }

        #[test]
        fn build_scene_has_ground_plus_1000_cubes_under_10_parents() {
            let scene = built();
            assert_eq!(scene.instance_count(), 1001);
            assert_eq!(scene.node_count(), 1 + 10 + 1000);
        }

        #[test]
        fn render_list_has_1001_items_in_one_batch() {
            let scene = built();
            let mut list = RenderList::with_capacity(1001);
            scene.build_render_list(&mut list);
            assert_eq!(list.items.len(), 1001);
            assert_eq!(list.batches.len(), 1);
            assert_eq!(list.batches[0].count, 1001);
        }

        #[test]
        fn lattice_materials_are_4_flat_and_4_pbr_lite() {
            let materials = lattice_materials();
            let flat = materials
                .iter()
                .filter(|m| m.model == ShadingModel::Flat)
                .count();
            let pbr = materials
                .iter()
                .filter(|m| m.model == ShadingModel::PbrLite)
                .count();
            assert_eq!((flat, pbr), (4, 4));
            for m in materials.iter().chain([&ground_material()]) {
                m.validate().expect("valid material");
            }
            assert_eq!(ground_material().model, ShadingModel::Flat);
        }

        #[test]
        fn two_builds_give_equal_world_matrices() {
            let (a, b) = (built(), built());
            let (mut la, mut lb) = (
                RenderList::with_capacity(1001),
                RenderList::with_capacity(1001),
            );
            a.build_render_list(&mut la);
            b.build_render_list(&mut lb);
            assert_eq!(la.items.len(), lb.items.len());
            for (x, y) in la.items.iter().zip(&lb.items) {
                assert_eq!(x.world, y.world);
                assert_eq!(x.material, y.material);
            }
        }

        #[test]
        fn jitter_stays_within_bounds() {
            let mut rng = Lcg(JITTER_SEED);
            for _ in 0..10_000 {
                let j = rng.jitter();
                assert!((-JITTER..=JITTER).contains(&j), "{j}");
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use raa_web::{RuntimeOptions, WebRuntime, error_to_js, stats_to_js};
    use wasm_bindgen::prelude::*;

    use crate::app::ScenesApp;

    /// Reads `options[key]` as a positive integer size, if present and valid.
    fn size_option(options: &JsValue, key: &str) -> Option<u32> {
        if !options.is_object() {
            return None;
        }
        let value = js_sys::Reflect::get(options, &JsValue::from_str(key))
            .ok()?
            .as_f64()?;
        // Rejects NaN, non-positive and out-of-range values before the cast.
        (1.0..=f64::from(u32::MAX))
            .contains(&value)
            .then_some(value as u32)
    }

    /// The cumulative allocation counter, linked only by `alloc-count` builds.
    #[cfg(feature = "alloc-count")]
    const ALLOC_PROBE: Option<fn() -> u64> = Some(raa_alloc_count::allocations);
    #[cfg(not(feature = "alloc-count"))]
    const ALLOC_PROBE: Option<fn() -> u64> = None;

    /// The running demo, driven from JS.
    #[wasm_bindgen]
    pub struct Demo {
        runtime: WebRuntime,
    }

    /// Starts the demo on `canvas`. `options` may carry `width` and `height` (canvas
    /// backing size in pixels); the canvas's current size is used otherwise.
    #[wasm_bindgen]
    pub async fn start(
        canvas: web_sys::HtmlCanvasElement,
        options: JsValue,
    ) -> Result<Demo, JsValue> {
        let opts = RuntimeOptions {
            width: size_option(&options, "width").unwrap_or(canvas.width()),
            height: size_option(&options, "height").unwrap_or(canvas.height()),
            alloc_probe: ALLOC_PROBE,
            ..RuntimeOptions::default()
        };
        let runtime = WebRuntime::new(canvas, Box::new(ScenesApp::default()), opts)
            .await
            .map_err(|e| error_to_js(&e))?;
        Ok(Demo { runtime })
    }

    #[wasm_bindgen]
    impl Demo {
        pub fn frame(&mut self, now_ms: f64) -> Result<(), JsValue> {
            self.runtime.frame(now_ms).map_err(|e| error_to_js(&e))
        }

        pub fn resize(&mut self, width: u32, height: u32) {
            self.runtime.resize(width, height);
        }

        pub fn stats(&self) -> JsValue {
            stats_to_js(&self.runtime.stats(), self.runtime.adapter())
        }

        pub async fn probe_non_empty(&mut self, now_ms: f64) -> Result<bool, JsValue> {
            self.runtime
                .probe_non_empty(now_ms)
                .await
                .map_err(|e| error_to_js(&e))
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use web::{Demo, start};
