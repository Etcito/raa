//! M5 simple booleans: every ok case of `raa_csg::cases()`, loaded through PrimsLoader +
//! CsgResolver, shown in a grid (table order, 4 per row), each slowly rotating so the cut
//! faces and holes face the camera.
#[cfg(feature = "alloc-count")]
#[global_allocator]
static A: raa_alloc_count::CountingAlloc = raa_alloc_count::CountingAlloc;

#[cfg(any(target_arch = "wasm32", test))]
#[allow(
    dead_code,
    reason = "host builds only run the CPU unit tests; the wasm entry points live in `web`"
)]
mod app {
    use raa_core::{FrameGraph, GpuContext, MaterialId, MeshId, RaaError};
    use raa_csg::{CsgCase, CsgResolver};
    use raa_geom::{GeometryRegistry, MeshData, Quality};
    use raa_loader::prims::PrimsLoader;
    use raa_loader::{LoadContext, LoadError, LoadedAsset, LoaderRegistry};
    use raa_math::{Aabb, Mat4, Quat, Vec3};
    use raa_render::{Renderer, RendererOptions};
    use raa_scene::{Camera, NodeId, RenderList, Scene, Transform};
    use raa_web::{FrameCounters, FrameInput, WebApp};

    /// Instances per grid row.
    pub(crate) const COLUMNS: usize = 4;
    /// Slot pitch, fixed so the layout never waits for CSG results. A host test keeps it
    /// between `spacing` of the baked diagonals and 1.5 times that.
    pub(crate) const PITCH: f32 = 3.2;
    /// Minimum pitch as a multiple of the largest bbox diagonal.
    const GAP: f32 = 1.1;
    /// Spin about the node's Y axis, radians per second.
    const SPIN_SPEED: f64 = 0.35;
    /// Spin phase offset between consecutive slots, radians.
    const PHASE_STEP: f32 = 0.9;
    /// Tilt about X so the top faces and dents are visible, radians.
    const TILT: f32 = 25.0 * std::f32::consts::PI / 180.0;
    const FOV_Y_DEG: f32 = 40.0;

    pub(crate) fn err(code: &str, e: impl std::fmt::Display) -> RaaError {
        RaaError::Resource(format!("{code}: {e}"))
    }

    /// A load error keeps its code, path and message (`<code> at <path>: <message>`).
    fn load_err(e: LoadError) -> RaaError {
        RaaError::Resource(e.to_string())
    }

    /// The cheap setup: geometry and loader registries plus the ok-case list of
    /// `raa_csg::cases()` in table order. No CSG work happens until `load_case`.
    pub(crate) struct Cases {
        geometry: GeometryRegistry,
        loaders: LoaderRegistry,
        ok: Vec<CsgCase>,
    }

    impl Cases {
        pub(crate) fn new() -> Result<Cases, RaaError> {
            let mut geometry = GeometryRegistry::new();
            raa_prims::register_all(&mut geometry).map_err(|e| err(e.code(), e))?;
            let mut loaders = LoaderRegistry::new();
            loaders.register(Box::new(PrimsLoader)).map_err(load_err)?;
            let ok = (raa_csg::cases().into_iter())
                .filter(|c| c.expected == "ok")
                .collect();
            Ok(Cases {
                geometry,
                loaders,
                ok,
            })
        }

        pub(crate) fn len(&self) -> usize {
            self.ok.len()
        }

        /// Loads ok case `i` through PrimsLoader + CsgResolver at LOD0 with strict kinds.
        pub(crate) fn load_case(&self, i: usize) -> Result<(&'static str, LoadedAsset), RaaError> {
            let case = (self.ok.get(i))
                .ok_or_else(|| err("demo.case_index", format!("no ok case {i}")))?;
            let ctx = LoadContext {
                geometry: &self.geometry,
                booleans: &CsgResolver,
                quality: Quality::LOD0,
                strict_kinds: true,
            };
            let asset = (self.loaders.load(case.asset.as_bytes(), &ctx)).map_err(load_err)?;
            Ok((case.id, asset))
        }
    }

    /// Loads every ok case in table order (refusal cases are not in the list).
    pub(crate) fn load_cases() -> Result<Vec<(&'static str, LoadedAsset)>, RaaError> {
        let cases = Cases::new()?;
        (0..cases.len()).map(|i| cases.load_case(i)).collect()
    }

    /// Union bounds of the asset's meshes in asset space.
    fn asset_bounds(asset: &LoadedAsset) -> Aabb {
        let mut b = Aabb::EMPTY;
        for m in &asset.meshes {
            for p in &m.mesh.positions {
                b.include(m.transform.transform_point(Vec3::from_array(*p)));
            }
        }
        b
    }

    /// The asset's meshes baked into asset space and recentred on its bbox centre, so the
    /// node rotates the result about its own middle. Also returns the bbox diagonal.
    pub(crate) fn bake(asset: &LoadedAsset) -> (Vec<MeshData>, f32) {
        let b = asset_bounds(asset);
        let centre = (b.min + b.max) * 0.5;
        let recentre = Mat4::from_trs(-centre, Quat::IDENTITY, Vec3::ONE);
        let baked = (asset.meshes.iter())
            .map(|m| {
                let mut out = MeshData::new();
                out.append_transformed(&m.mesh, &recentre.mul(&m.transform));
                out
            })
            .collect();
        (baked, (b.max - b.min).length())
    }

    /// Minimum slot pitch: the largest bbox diagonal plus a gap, so neighbours never touch
    /// whatever their rotation.
    pub(crate) fn spacing(diagonals: &[f32]) -> f32 {
        diagonals.iter().copied().fold(0.0, f32::max) * GAP
    }

    pub(crate) fn rows(count: usize) -> usize {
        count.div_ceil(COLUMNS).max(1)
    }

    /// Centre of slot `i` of `count`: table order, left to right then top to bottom, the
    /// full 4-column grid centred on the origin in the XY plane.
    pub(crate) fn slot(i: usize, count: usize, pitch: f32) -> Vec3 {
        let mid = |n: usize| (n as f32 - 1.0) * 0.5;
        let (row, col) = (i / COLUMNS, i % COLUMNS);
        Vec3::new(
            (col as f32 - mid(COLUMNS)) * pitch,
            (mid(rows(count)) - row as f32) * pitch,
            0.0,
        )
    }

    /// A fixed camera on +Z looking at the grid centre, far enough that the whole grid
    /// (each slot's bounding sphere of radius pitch / 2) fits at `aspect`.
    pub(crate) fn camera(aspect: f32, count: usize, pitch: f32) -> Camera {
        let fov = FOV_Y_DEG.to_radians();
        let tan = (fov * 0.5).tan();
        let half_h = rows(count) as f32 * pitch * 0.5;
        let half_w = COLUMNS as f32 * pitch * 0.5;
        let dist = (half_h / tan).max(half_w / (tan * aspect.max(1e-3))) + pitch;
        Camera {
            eye: Vec3::new(0.0, 0.0, dist),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y_rad: fov,
            near: 0.1,
            far: dist * 2.0 + pitch,
            aspect,
        }
    }

    /// Node rotation of slot `i` at `now_ms`: spin about Y (phase offset per slot), then
    /// the fixed tilt about X.
    pub(crate) fn rotation(i: usize, now_ms: f64) -> Quat {
        let spin = ((now_ms * 1e-3 * SPIN_SPEED) % std::f64::consts::TAU) as f32;
        let tilt = Quat::from_axis_angle(Vec3::X, TILT);
        tilt.mul(Quat::from_rotation_y(spin + i as f32 * PHASE_STEP))
    }

    /// Adds the node of slot `i` and one instance per part.
    pub(crate) fn spawn(
        scene: &mut Scene,
        i: usize,
        at: Vec3,
        parts: &[(MeshId, MaterialId)],
    ) -> Result<NodeId, RaaError> {
        let t = Transform::from_trs(at, rotation(i, 0.0), Vec3::ONE);
        let node = scene.add_node(None, t).map_err(|e| err(e.code(), e))?;
        for &(mesh, material) in parts {
            (scene.add_instance(node, mesh, material)).map_err(|e| err(e.code(), e))?;
        }
        Ok(node)
    }

    /// Frame update: only `set_local` on the existing nodes, no allocation.
    pub(crate) fn animate(
        scene: &mut Scene,
        nodes: &[(NodeId, Vec3)],
        now_ms: f64,
    ) -> Result<(), RaaError> {
        for (i, &(node, at)) in nodes.iter().enumerate() {
            let t = Transform::from_trs(at, rotation(i, now_ms), Vec3::ONE);
            scene.set_local(node, t).map_err(|e| err(e.code(), e))?;
        }
        Ok(())
    }

    struct Running {
        renderer: Renderer,
        scene: Scene,
        list: RenderList,
        cases: Cases,
        /// Node and slot centre per loaded case, in table order (index = slot).
        nodes: Vec<(NodeId, Vec3)>,
        count: usize,
    }

    impl Running {
        /// Loads the next ok case, uploads its meshes and materials and spawns it in its
        /// fixed slot. Only runs during the loading frames.
        fn load_next(&mut self, gpu: &GpuContext) -> Result<(), RaaError> {
            let i = self.nodes.len();
            let (id, asset) = self.cases.load_case(i)?;
            let (meshes, _) = bake(&asset);
            let renderer = &mut self.renderer;
            let mut names: Vec<(&str, MaterialId)> = Vec::with_capacity(asset.materials.len());
            for (name, m) in &asset.materials {
                let id = (renderer.materials_mut().add(*m)).map_err(|e| err(e.code, &e.message))?;
                names.push((name, id));
            }
            let mut parts: Vec<(MeshId, MaterialId)> = Vec::with_capacity(meshes.len());
            for (mesh, loaded) in meshes.iter().zip(&asset.meshes) {
                let material = (names.iter().find(|(n, _)| *n == loaded.material))
                    .map(|&(_, m)| m)
                    .ok_or_else(|| err("material.unknown", format!("{id}: {}", loaded.material)))?;
                parts.push((renderer.upload_mesh(gpu, mesh)?, material));
            }
            let at = slot(i, self.count, PITCH);
            self.nodes
                .push((spawn(&mut self.scene, i, at, &parts)?, at));
            Ok(())
        }
    }

    #[derive(Default)]
    pub(crate) struct BooleanGrid(Option<Running>);

    impl WebApp for BooleanGrid {
        /// Cheap setup plus case 0 only, so frame 0 draws a real result without paying for
        /// every CSG evaluation; `update` streams in the rest, one per frame.
        fn init(
            &mut self,
            gpu: &GpuContext,
            graph: &mut FrameGraph,
            size: (u32, u32),
            surface_format: wgpu::TextureFormat,
            surface_is_srgb: bool,
        ) -> Result<(), RaaError> {
            let cases = Cases::new()?;
            let count = cases.len();
            let renderer = Renderer::new(
                gpu,
                surface_format,
                surface_is_srgb,
                RendererOptions::default(),
            )?;
            renderer.install(graph);
            let mut scene = Scene::with_capacity(count, count);
            scene.camera = camera(size.0 as f32 / size.1.max(1) as f32, count, PITCH);
            let mut app = Running {
                renderer,
                scene,
                list: RenderList::with_capacity(count),
                cases,
                nodes: Vec::with_capacity(count),
                count,
            };
            if count > 0 {
                app.load_next(gpu)?;
            }
            self.0 = Some(app);
            Ok(())
        }

        fn update(
            &mut self,
            gpu: &GpuContext,
            input: &FrameInput,
        ) -> Result<FrameCounters, RaaError> {
            let app = (self.0.as_mut())
                .ok_or_else(|| err("demo.not_initialised", "update before init"))?;
            if app.nodes.len() < app.count {
                app.load_next(gpu)?;
            }
            animate(&mut app.scene, &app.nodes, input.now_ms)?;
            let counts = app.renderer.prepare(gpu, &mut app.scene, &mut app.list)?;
            Ok(FrameCounters {
                instances: counts.instances,
                draw_calls: counts.draw_calls,
                triangles: counts.triangles,
            })
        }

        fn resize(&mut self, _gpu: &GpuContext, size: (u32, u32)) {
            if let Some(app) = self.0.as_mut() {
                let aspect = size.0 as f32 / size.1.max(1) as f32;
                app.scene.camera = camera(aspect, app.count, PITCH);
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        const ASPECTS: [f32; 2] = [16.0 / 9.0, 960.0 / 540.0];

        fn layout() -> (Vec<(&'static str, Vec<MeshData>)>, f32) {
            let cases = load_cases().expect("ok cases load");
            let baked = cases.iter().map(|(id, a)| (*id, bake(a).0)).collect();
            (baked, PITCH)
        }

        #[test]
        fn fixed_pitch_fits_every_baked_case_and_stays_tight() {
            let cases = load_cases().expect("ok cases load");
            let diagonals: Vec<f32> = cases.iter().map(|(_, a)| bake(a).1).collect();
            let min = spacing(&diagonals);
            assert!(min <= PITCH, "spacing {min} > PITCH {PITCH}");
            assert!(PITCH <= 1.5 * min, "PITCH {PITCH} > 1.5 x spacing {min}");
        }

        #[test]
        fn setup_lists_the_ok_cases_without_loading_them() {
            let cases = Cases::new().expect("setup");
            assert_eq!(cases.len(), 11);
            let (id, _) = cases.load_case(0).expect("case 0");
            assert_eq!(id, raa_csg::cases()[0].id);
            assert!(cases.load_case(11).is_err());
        }

        #[test]
        fn all_eleven_ok_cases_load_and_none_is_refused() {
            let ok: Vec<&str> = (raa_csg::cases().iter())
                .filter(|c| c.expected == "ok")
                .map(|c| c.id)
                .collect();
            assert_eq!(ok.len(), 11);
            let cases = load_cases().expect("no ok case is refused");
            let ids: Vec<&str> = cases.iter().map(|(id, _)| *id).collect();
            assert_eq!(ids, ok, "table order kept");
            for (id, a) in &cases {
                assert!(!a.meshes.is_empty(), "{id}");
                assert!(a.triangles > 0, "{id}");
                for m in &a.meshes {
                    assert!(a.materials.iter().any(|(n, _)| *n == m.material), "{id}");
                }
                let (baked, diagonal) = bake(a);
                assert!(diagonal > 0.0, "{id}");
                assert!(baked.iter().all(|m| m.validate().is_ok()), "{id}");
            }
        }

        #[test]
        fn grid_has_four_columns_and_three_rows_for_eleven() {
            assert_eq!(rows(11), 3);
            let pitch = 2.0;
            assert_eq!(slot(0, 11, pitch), Vec3::new(-3.0, 2.0, 0.0));
            assert_eq!(slot(3, 11, pitch), Vec3::new(3.0, 2.0, 0.0));
            assert_eq!(slot(10, 11, pitch), Vec3::new(1.0, -2.0, 0.0));
        }

        #[test]
        fn every_slot_centre_is_inside_the_frustum() {
            let (cases, pitch) = layout();
            let n = cases.len();
            for aspect in ASPECTS {
                let vp = camera(aspect, n, pitch).view_proj();
                for i in 0..n {
                    let c = vp.transform_point(slot(i, n, pitch));
                    assert!(
                        c.x.abs() < 1.0 && c.y.abs() < 1.0,
                        "{aspect} slot {i}: {c:?}"
                    );
                    assert!(c.z > 0.0 && c.z < 1.0, "{aspect} slot {i}: {c:?}");
                }
            }
        }

        #[test]
        fn rotating_meshes_stay_in_view_and_in_their_slot() {
            let (cases, pitch) = layout();
            let n = cases.len();
            for aspect in ASPECTS {
                let vp = camera(aspect, n, pitch).view_proj();
                for step in 0..12 {
                    let now_ms = f64::from(step) * 1500.0;
                    for (i, (id, meshes)) in cases.iter().enumerate() {
                        let at = slot(i, n, pitch);
                        let world = Mat4::from_trs(at, rotation(i, now_ms), Vec3::ONE);
                        for p in meshes.iter().flat_map(|m| &m.positions) {
                            let local = Vec3::from_array(*p);
                            assert!(local.length() <= pitch * 0.5, "{id}: {local:?}");
                            let c = vp.transform_point(world.transform_point(local));
                            assert!(c.x.abs() <= 1.0 && c.y.abs() <= 1.0, "{id}: {c:?}");
                        }
                    }
                }
            }
        }

        #[test]
        fn rotation_update_keeps_node_and_instance_counts() {
            let (cases, pitch) = layout();
            let n = cases.len();
            let mut scene = Scene::with_capacity(n, n);
            let mut nodes = Vec::with_capacity(n);
            for i in 0..n {
                let at = slot(i, n, pitch);
                let parts = [(MeshId(i as u32), MaterialId(0))];
                nodes.push((spawn(&mut scene, i, at, &parts).expect("spawn"), at));
            }
            scene.update_world();
            let before = (scene.node_count(), scene.instance_count());
            let world0 = scene.world(nodes[0].0).expect("node");
            for frame in 1..=120 {
                animate(&mut scene, &nodes, f64::from(frame) * 16.0).expect("animate");
                scene.update_world();
                assert_eq!((scene.node_count(), scene.instance_count()), before);
            }
            let world1 = scene.world(nodes[0].0).expect("node");
            assert_ne!(world0, world1, "the node rotated");
            assert_eq!(world1.transform_point(Vec3::ZERO), nodes[0].1);
        }

        #[test]
        fn slots_rotate_with_offset_phases() {
            assert_ne!(rotation(0, 0.0), rotation(1, 0.0));
            assert_ne!(rotation(0, 0.0), rotation(0, 1000.0));
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    //! wasm-bindgen API of the booleans demo.

    use raa_web::{RuntimeOptions, WebRuntime, error_to_js, stats_to_js};
    use wasm_bindgen::prelude::*;

    use crate::app::BooleanGrid;

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
        let runtime = WebRuntime::new(canvas, Box::new(BooleanGrid::default()), opts)
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
