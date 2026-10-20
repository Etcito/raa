//! M4 primitive assets: the three prims v0 samples, loaded through LoaderRegistry +
//! PrimsLoader, each spawned 3 times side by side on a ground slab.
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
    use raa_geom::{GeometryRegistry, MeshData, Quality};
    use raa_loader::prims::{PrimsLoader, samples};
    use raa_loader::{LoadContext, LoadError, LoadedAsset, LoaderRegistry, NoBooleans};
    use raa_material::Material;
    use raa_math::{Quat, Vec3};
    use raa_prims::PrimParams;
    use raa_render::{Renderer, RendererOptions};
    use raa_scene::{Camera, OrbitController, RenderList, Scene, Transform};
    use raa_web::{FrameCounters, FrameInput, WebApp};

    pub(crate) const SPACING: f32 = 2.5;
    pub(crate) const COPIES: usize = 3;
    /// Ground slab half extents: 9 x 0.1 x 7 m, top face at y = 0.
    pub(crate) const SLAB_HALF: [f32; 3] = [4.5, 0.05, 3.5];
    /// Auto yaw of the orbit camera, radians per second.
    const YAW_SPEED: f64 = 0.15;
    const ORBIT_TARGET: Vec3 = Vec3::new(0.0, 0.4, 0.0);
    const ORBIT_DISTANCE: f32 = 11.0;
    const ORBIT_PITCH: f32 = 0.55;

    pub(crate) fn err(code: &str, e: impl std::fmt::Display) -> RaaError {
        RaaError::Resource(format!("{code}: {e}"))
    }

    /// A load error keeps its code, path and message (`<code> at <path>: <message>`).
    fn load_err(e: LoadError) -> RaaError {
        RaaError::Resource(e.to_string())
    }

    /// Loads every sample through a LoaderRegistry with PrimsLoader at LOD0, strict
    /// kinds and NoBooleans.
    pub(crate) fn load_samples() -> Result<Vec<LoadedAsset>, RaaError> {
        let mut geometry = GeometryRegistry::new();
        raa_prims::register_all(&mut geometry).map_err(|e| err(e.code(), e))?;
        let mut loaders = LoaderRegistry::new();
        loaders.register(Box::new(PrimsLoader)).map_err(load_err)?;
        let ctx = LoadContext {
            geometry: &geometry,
            booleans: &NoBooleans,
            quality: Quality::LOD0,
            strict_kinds: true,
        };
        samples::sample_assets()
            .iter()
            .map(|(_, text)| loaders.load(text.as_bytes(), &ctx).map_err(load_err))
            .collect()
    }

    /// The asset's meshes baked into asset space, in its mesh order.
    pub(crate) fn bake(asset: &LoadedAsset) -> Vec<MeshData> {
        asset
            .meshes
            .iter()
            .map(|m| {
                let mut out = MeshData::new();
                out.append_transformed(&m.mesh, &m.transform);
                out
            })
            .collect()
    }

    /// Lowest y of the baked meshes, so the copies rest on the slab top (y = 0).
    pub(crate) fn min_y(baked: &[MeshData]) -> f32 {
        baked
            .iter()
            .map(|m| m.bounds().min.y)
            .fold(f32::INFINITY, f32::min)
    }

    /// Position of copy `col` of asset `row`: one row per asset along Z, 3 columns
    /// along X, centred on the slab.
    pub(crate) fn slot(row: usize, rows: usize, col: usize) -> Vec3 {
        let mid = |n: usize| (n as f32 - 1.0) * 0.5;
        Vec3::new(
            (col as f32 - mid(COPIES)) * SPACING,
            0.0,
            (row as f32 - mid(rows)) * SPACING,
        )
    }

    pub(crate) fn orbit() -> OrbitController {
        let mut orbit = OrbitController::new(ORBIT_TARGET, ORBIT_DISTANCE);
        orbit.rotate(0.0, ORBIT_PITCH);
        orbit
    }

    pub(crate) fn camera(aspect: f32) -> Camera {
        let mut cam = Camera {
            fov_y_rad: 45.0_f32.to_radians(),
            aspect,
            ..Camera::default()
        };
        orbit().apply(&mut cam);
        cam
    }

    pub(crate) fn slab_material() -> Material {
        Material::pbr([0.35, 0.35, 0.36], 0.8, 0.0)
    }

    struct Running {
        renderer: Renderer,
        scene: Scene,
        list: RenderList,
        orbit: OrbitController,
    }

    #[derive(Default)]
    pub(crate) struct AssetShowcase(Option<Running>);

    impl WebApp for AssetShowcase {
        fn init(
            &mut self,
            gpu: &GpuContext,
            graph: &mut FrameGraph,
            size: (u32, u32),
            surface_format: wgpu::TextureFormat,
            surface_is_srgb: bool,
        ) -> Result<(), RaaError> {
            let assets = load_samples()?;
            let mut renderer = Renderer::new(
                gpu,
                surface_format,
                surface_is_srgb,
                RendererOptions::default(),
            )?;
            let mesh_count: usize = assets.iter().map(|a| a.meshes.len()).sum();
            let nodes = 1 + assets.len() * COPIES;
            let instances = 1 + mesh_count * COPIES;
            let mut scene = Scene::with_capacity(nodes, instances);

            // Ground slab: a Box whose top face is at y = 0.
            let slab_params = PrimParams::Box {
                half_extents: SLAB_HALF,
            };
            let mut slab = MeshData::new();
            raa_prims::generate(&slab_params, Quality::LOD0, &mut slab)
                .map_err(|e| err(e.code(), e))?;
            let slab_mesh = renderer.upload_mesh(gpu, &slab)?;
            let slab_mat = (renderer.materials_mut().add(slab_material()))
                .map_err(|e| err(e.code, &e.message))?;
            let at = Vec3::new(0.0, -SLAB_HALF[1], 0.0);
            let node = (scene.add_node(None, Transform::from_trs(at, Quat::IDENTITY, Vec3::ONE)))
                .map_err(|e| err(e.code(), e))?;
            (scene.add_instance(node, slab_mesh, slab_mat)).map_err(|e| err(e.code(), e))?;

            for (row, asset) in assets.iter().enumerate() {
                let mut names: Vec<(&str, MaterialId)> = Vec::with_capacity(asset.materials.len());
                for (name, m) in &asset.materials {
                    let id =
                        (renderer.materials_mut().add(*m)).map_err(|e| err(e.code, &e.message))?;
                    names.push((name, id));
                }
                let baked = bake(asset);
                let mut parts: Vec<(MeshId, MaterialId)> = Vec::with_capacity(baked.len());
                for (mesh, loaded) in baked.iter().zip(&asset.meshes) {
                    let material = (names.iter().find(|(n, _)| *n == loaded.material))
                        .map(|&(_, id)| id)
                        .ok_or_else(|| {
                            err(
                                "material.unknown",
                                format!("{}: {}", asset.id, loaded.material),
                            )
                        })?;
                    parts.push((renderer.upload_mesh(gpu, mesh)?, material));
                }
                let lift = -min_y(&baked);
                for col in 0..COPIES {
                    let at = slot(row, assets.len(), col) + Vec3::new(0.0, lift, 0.0);
                    let t = Transform::from_trs(at, Quat::IDENTITY, Vec3::ONE);
                    let node = scene.add_node(None, t).map_err(|e| err(e.code(), e))?;
                    for &(mesh, material) in &parts {
                        (scene.add_instance(node, mesh, material)).map_err(|e| err(e.code(), e))?;
                    }
                }
            }

            renderer.install(graph);
            scene.camera = camera(size.0 as f32 / size.1.max(1) as f32);
            self.0 = Some(Running {
                renderer,
                scene,
                list: RenderList::with_capacity(instances),
                orbit: orbit(),
            });
            Ok(())
        }

        fn update(
            &mut self,
            gpu: &GpuContext,
            input: &FrameInput,
        ) -> Result<FrameCounters, RaaError> {
            let app = (self.0.as_mut())
                .ok_or_else(|| err("demo.not_initialised", "update before init"))?;
            app.orbit
                .rotate((input.dt_ms * 1e-3 * YAW_SPEED) as f32, 0.0);
            app.orbit.apply(&mut app.scene.camera);
            let counts = app.renderer.prepare(gpu, &mut app.scene, &mut app.list)?;
            Ok(FrameCounters {
                instances: counts.instances,
                draw_calls: counts.draw_calls,
                triangles: counts.triangles,
            })
        }

        fn resize(&mut self, _gpu: &GpuContext, size: (u32, u32)) {
            if let Some(app) = self.0.as_mut() {
                app.scene.camera.set_viewport(size.0, size.1);
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn samples_load_with_resolvable_materials() {
            let assets = load_samples().expect("samples load");
            assert_eq!(assets.len(), 3);
            for a in &assets {
                assert!(!a.meshes.is_empty(), "{}", a.id);
                for m in &a.meshes {
                    assert!(a.materials.iter().any(|(n, _)| *n == m.material));
                }
                let baked = bake(a);
                assert_eq!(baked.len(), a.meshes.len());
                assert!(baked.iter().all(|m| m.validate().is_ok()));
            }
        }

        #[test]
        fn copies_rest_on_the_slab_inside_its_footprint() {
            let assets = load_samples().expect("samples load");
            for (row, a) in assets.iter().enumerate() {
                let baked = bake(a);
                let lift = -min_y(&baked);
                for col in 0..COPIES {
                    let at = slot(row, assets.len(), col) + Vec3::new(0.0, lift, 0.0);
                    for m in &baked {
                        let b = m.bounds();
                        let (lo, hi) = (b.min + at, b.max + at);
                        assert!(lo.y >= -1e-5, "{}: {lo:?}", a.id);
                        assert!(lo.x >= -SLAB_HALF[0] && hi.x <= SLAB_HALF[0]);
                        assert!(lo.z >= -SLAB_HALF[2] && hi.z <= SLAB_HALF[2]);
                    }
                }
            }
        }

        #[test]
        fn all_copies_stay_in_view_while_the_camera_orbits() {
            let assets = load_samples().expect("samples load");
            let mut orbit = orbit();
            let mut cam = camera(16.0 / 9.0);
            for _ in 0..16 {
                orbit.rotate(std::f32::consts::TAU / 16.0, 0.0);
                orbit.apply(&mut cam);
                let vp = cam.view_proj();
                for (row, a) in assets.iter().enumerate() {
                    let baked = bake(a);
                    let lift = -min_y(&baked);
                    for col in 0..COPIES {
                        let at = slot(row, assets.len(), col) + Vec3::new(0.0, lift, 0.0);
                        for p in baked.iter().flat_map(|m| &m.positions) {
                            let c = vp.transform_point(Vec3::from_array(*p) + at);
                            assert!(c.x.abs() <= 1.0 && c.y.abs() <= 1.0, "{}: {c:?}", a.id);
                        }
                    }
                }
            }
        }

        #[test]
        fn slab_material_validates() {
            slab_material().validate().expect("valid");
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    //! wasm-bindgen API of the asset demo.

    use raa_web::{RuntimeOptions, WebRuntime, error_to_js, stats_to_js};
    use wasm_bindgen::prelude::*;

    use crate::app::AssetShowcase;

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
        let runtime = WebRuntime::new(canvas, Box::new(AssetShowcase::default()), opts)
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
