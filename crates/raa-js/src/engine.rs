//! The `Engine` exported to JS (wasm32 only): a [`WebApp`] inside a [`WebRuntime`].
//!
//! The engine state ([`EngineState`]) lives in an `Rc<RefCell<Option<EngineState>>>` shared
//! by [`EngineApp`] (owned by the runtime) and [`Engine`] (owned by JS).
//! `WebRuntime::app_mut` returns `&mut dyn WebApp` with no downcast, so the JS methods reach
//! the state through this shared cell instead ([`Engine::with_state`]). Everything is
//! single-threaded and there is no global state.
//!
//! Rendering goes through the lit pipeline: [`EngineApp::init`] builds the [`Pipeline`],
//! the [`LitRenderer`] (prepass and MSAA forward) and the [`TonemapPass`], runs the api
//! files' `install` ([`InstallCtx`]) and adds `raa.pipeline` to the frame graph. v0
//! materials become surface materials ([`SurfaceMaterial::from_v0`]), v0 meshes attributed
//! meshes with the S10 default attributes, and v0 instances instance groups (one scene
//! instance per sub-mesh, tagged with the JS instance handle).
//!
//! Borrow rule: JS calls into the engine never re-enter (wasm-bindgen methods run to
//! completion, and nothing here calls back into JS user code), so a `RefCell` borrow taken
//! by a method never overlaps with the borrow `EngineApp::update` takes inside
//! `runtime.frame`: methods drop their borrow before calling into the runtime. The
//! pipeline is borrowed by `raa.pipeline` only while the frame graph executes.

use std::cell::RefCell;
use std::rc::Rc;

use js_sys::{Array, Float32Array, Object, Reflect, Uint32Array};
use raa_core::{DeviceRequest, FrameGraph, GpuContext, GpuOptions, RaaError};
use raa_csg::CsgResolver;
use raa_geom::{GeometryRegistry, MeshData};
use raa_lit::{LitOptions, LitRenderer, SurfaceMaterial};
use raa_loader::prims::PrimsLoader;
use raa_loader::{LoadContext, LoaderRegistry};
use raa_material::Material;
use raa_math::{Quat, Vec3};
use raa_mesh::AttributedMesh;
use raa_pipeline::Pipeline;
use raa_scene::{NodeId, OrbitController, RenderList, Scene, Transform};
use raa_tonemap::{ToneMapping, TonemapPass};
use raa_web::{
    FrameCounters, FrameInput, RuntimeOptions, WebApp, WebRuntime, error_to_js, stats_to_js,
};
use wasm_bindgen::prelude::*;

use crate::api;
use crate::convert::{
    DISPOSED, EngineOptions, HandleMap, INVALID_ARGUMENT, UNKNOWN_HANDLE, api_error, check_trs_len,
    decode_trs, geom_error, js, load_error, quality, scene_error,
};
use crate::feature_map::FeatureMap;
use crate::state::{AssetEntry, EngineState, MeshEntry, MeshOrigin, SkinFeature};

/// WEBGPU_OPTIONAL: device features requested when the adapter has them, never required.
const WEBGPU_OPTIONAL: wgpu::Features =
    wgpu::Features::FLOAT32_FILTERABLE.union(wgpu::Features::RG11B10UFLOAT_RENDERABLE);

/// What an api file's `install` receives at init, once the M1 passes (`prepass`,
/// `forward`, `tonemap`) are in the pipeline and before `raa.pipeline` joins the graph.
// The fields are read by the api files (src/api/*.rs), not by the engine itself.
#[allow(dead_code)]
pub(crate) struct InstallCtx<'a> {
    pub gpu: &'a GpuContext,
    pub surface_format: wgpu::TextureFormat,
    pub pipeline: &'a Rc<RefCell<Pipeline>>,
    pub lit: &'a mut LitRenderer,
    pub features: &'a mut FeatureMap,
}

/// The v0 helpers of the engine methods.
impl EngineState {
    /// Refuses with `render.too_many_materials` when `extra` more would exceed maxMaterials.
    fn check_material_room(&self, extra: usize) -> Result<(), RaaError> {
        let len = self.lit.materials().len();
        if len + extra > self.max_materials as usize {
            return Err(api_error(
                "render.too_many_materials",
                format!(
                    "maxMaterials is {}, {len} are in use and {extra} more were requested",
                    self.max_materials
                ),
            ));
        }
        Ok(())
    }

    /// Adds `m` as its surface material ([`SurfaceMaterial::from_v0`]).
    fn add_material(&mut self, m: &Material) -> Result<u32, RaaError> {
        let id = self.lit.materials_mut().add(SurfaceMaterial::from_v0(m))?;
        Ok(self.materials.insert(id))
    }

    /// The node behind `parent`, or None for a root.
    fn parent(&self, parent: Option<u32>) -> Result<Option<NodeId>, RaaError> {
        parent.map(|h| self.node(h)).transpose()
    }

    fn add_node(&mut self, parent: Option<NodeId>, t: Transform) -> Result<u32, RaaError> {
        let id = self
            .scene
            .add_node(parent, t)
            .map_err(|e| scene_error(&e))?;
        Ok(self.nodes.insert(id))
    }

    /// Uploads `scratch` as an attributed mesh (one sub-mesh, S10 default attributes) and
    /// returns its new v0 mesh handle. The scratch buffer keeps its capacity.
    fn upload_scratch(&mut self, gpu: &GpuContext) -> Result<u32, RaaError> {
        let mesh = AttributedMesh::from_mesh(std::mem::take(&mut self.scratch));
        let uploaded = self.lit.upload(gpu, &mesh);
        self.scratch = mesh.base;
        let lit = uploaded?;
        Ok(self.meshes.insert(MeshEntry {
            lit,
            origin: MeshOrigin::V0,
        }))
    }

    /// v0 `addInstance`: an instance group drawing every sub-mesh of `mesh` with
    /// `material`.
    fn add_instance(&mut self, node: NodeId, mesh: u32, material: u32) -> Result<u32, RaaError> {
        let (mesh, material) = (self.mesh(mesh)?, self.material(material)?);
        self.add_instance_group(node, mesh.lit, Some(material))
    }
}

type Shared = Rc<RefCell<Option<EngineState>>>;

fn js_err(e: &RaaError) -> JsValue {
    error_to_js(e)
}

fn disposed() -> JsValue {
    js_err(&api_error(DISPOSED, "the engine was disposed"))
}

fn invalid(message: String) -> RaaError {
    api_error(INVALID_ARGUMENT, message)
}

fn pipeline_busy() -> RaaError {
    api_error(
        "pipeline.busy",
        "the pipeline is borrowed (called while a frame executes)",
    )
}

/// The runtime's application: fills the shared state at init, prepares each frame.
struct EngineApp {
    state: Shared,
    opts: EngineOptions,
}

impl WebApp for EngineApp {
    /// Builds the pipeline with the lit renderer's passes and the tonemap pass (initial
    /// tone mapping from the options), runs the api files' `install`, then installs
    /// `raa.pipeline` in `graph`.
    fn init(
        &mut self,
        gpu: &GpuContext,
        graph: &mut FrameGraph,
        size: (u32, u32),
        surface_format: wgpu::TextureFormat,
        surface_is_srgb: bool,
    ) -> Result<(), RaaError> {
        let o = self.opts;
        let pipeline = Rc::new(RefCell::new(Pipeline::new(
            gpu,
            surface_format,
            surface_is_srgb,
        )));
        let mut lit = LitRenderer::new(
            gpu,
            &pipeline,
            LitOptions {
                max_instances: o.max_instances,
                max_materials: o.max_materials,
                msaa: o.msaa,
            },
        )?;
        let (pass, tonemap) = {
            let p = pipeline.try_borrow().map_err(|_| pipeline_busy())?;
            TonemapPass::new(gpu, &p)
        };
        tonemap.set(ToneMapping {
            mapper: o.tone_mapper,
            exposure_ev: o.exposure_ev,
            ..ToneMapping::default()
        })?;
        pipeline
            .try_borrow_mut()
            .map_err(|_| pipeline_busy())?
            .add(Box::new(pass))?;
        let mut features = FeatureMap::new();
        // S18: the `skin` pass runs at order 15, before every lit draw (feature bit 20).
        let skin = SkinFeature::new(gpu);
        skin.connect(&mut lit);
        pipeline
            .try_borrow_mut()
            .map_err(|_| pipeline_busy())?
            .add(Box::new(skin.pipeline_pass()))?;
        features.insert(skin);
        api::install_all(&mut InstallCtx {
            gpu,
            surface_format,
            pipeline: &pipeline,
            lit: &mut lit,
            features: &mut features,
        })?;
        Pipeline::install(&pipeline, graph);

        let max = o.max_instances as usize;
        let mut scene = Scene::with_capacity(max, max);
        scene.camera.set_viewport(size.0, size.1);
        let orbit = OrbitController::new(Vec3::ZERO, 6.0);
        let mut geometry = GeometryRegistry::new();
        raa_prims::register_all(&mut geometry).map_err(|e| geom_error(&e))?;
        let mut loaders = LoaderRegistry::new();
        loaders
            .register(Box::new(PrimsLoader))
            .map_err(|e| load_error(&e))?;
        *self.state.borrow_mut() = Some(EngineState {
            scene,
            list: RenderList::with_capacity(max),
            tags: Vec::with_capacity(max),
            orbit,
            geometry,
            loaders,
            booleans: CsgResolver,
            scratch: MeshData::new(),
            lit,
            pipeline,
            tonemap,
            features,
            max_materials: o.max_materials,
            meshes: HandleMap::new("mesh"),
            skins: Vec::new(),
            materials: HandleMap::new("material"),
            nodes: HandleMap::new("node"),
            instances: HandleMap::new("instance"),
            dirty_meshes: Vec::new(),
            asset_handles: HandleMap::new("asset"),
            assets: Vec::new(),
            scratch_nodes: Vec::new(),
            scratch_trs: Vec::new(),
        });
        Ok(())
    }

    /// Sets the pipeline time, runs the api files' `frame`, rebuilds the instance groups
    /// of meshes replaced since the last frame, then prepares the lit frame.
    fn update(&mut self, gpu: &GpuContext, input: &FrameInput) -> Result<FrameCounters, RaaError> {
        let mut guard = self.state.borrow_mut();
        let s = guard
            .as_mut()
            .ok_or_else(|| api_error(DISPOSED, "the engine was disposed"))?;
        s.pipeline
            .try_borrow_mut()
            .map_err(|_| pipeline_busy())?
            .set_time(input.now_ms);
        api::frame_all(s, gpu, input)?;
        s.rebuild_dirty_groups()?;
        s.flush_skins(gpu)?;
        let c = s.lit.prepare(gpu, &mut s.scene, &mut s.list, &mut s.tags)?;
        Ok(FrameCounters {
            instances: c.instances,
            draw_calls: c.draw_calls,
            triangles: c.triangles,
        })
    }

    fn resize(&mut self, _gpu: &GpuContext, size: (u32, u32)) {
        if let Some(s) = self.state.borrow_mut().as_mut() {
            s.scene.camera.set_viewport(size.0, size.1);
        }
    }
}

/// The RAA engine: one canvas, one scene, the HDR lit pipeline.
#[wasm_bindgen(js_name = Engine)]
pub struct Engine {
    runtime: Option<WebRuntime>,
    state: Shared,
    viewport: (u32, u32),
}

impl Engine {
    fn runtime(&mut self) -> Result<&mut WebRuntime, JsValue> {
        self.runtime.as_mut().ok_or_else(disposed)
    }

    /// Runs `f` on the state, with the GPU context. The borrow ends before returning.
    /// Used by the engine methods and the api files.
    pub(crate) fn with_state<R>(
        &self,
        f: impl FnOnce(&GpuContext, &mut EngineState) -> Result<R, RaaError>,
    ) -> Result<R, JsValue> {
        let runtime = self.runtime.as_ref().ok_or_else(disposed)?;
        let mut guard = self.state.borrow_mut();
        let s = guard.as_mut().ok_or_else(disposed)?;
        f(runtime.gpu(), s).map_err(|e| js_err(&e))
    }
}

fn finite(name: &str, x: f64) -> Result<f32, RaaError> {
    if x.is_finite() {
        Ok(x as f32)
    } else {
        Err(invalid(format!("'{name}' must be finite, got {x}")))
    }
}

fn vec3_arg(name: &str, v: &[f32]) -> Result<Vec3, RaaError> {
    match v {
        [x, y, z] if v.iter().all(|c| c.is_finite()) => Ok(Vec3::new(*x, *y, *z)),
        [_, _, _] => Err(invalid(format!("'{name}' must hold finite numbers"))),
        _ => Err(invalid(format!(
            "'{name}' must have 3 components, got {}",
            v.len()
        ))),
    }
}

/// Reads an optional vector argument of `N` finite floats.
fn opt_floats<const N: usize>(
    name: &str,
    v: Option<&[f32]>,
    default: [f32; N],
) -> Result<[f32; N], RaaError> {
    let Some(v) = v else {
        return Ok(default);
    };
    if v.len() != N {
        return Err(invalid(format!(
            "'{name}' must have {N} components, got {}",
            v.len()
        )));
    }
    if !v.iter().all(|c| c.is_finite()) {
        return Err(invalid(format!("'{name}' must hold finite numbers")));
    }
    let mut out = default;
    out.copy_from_slice(v);
    Ok(out)
}

/// t [x, y, z], r [x, y, z, w], s [x, y, z]; each defaults to the identity part.
fn trs_arg(t: Option<&[f32]>, r: Option<&[f32]>, s: Option<&[f32]>) -> Result<Transform, RaaError> {
    let t = opt_floats("t", t, [0.0; 3])?;
    let r = opt_floats("r", r, [0.0, 0.0, 0.0, 1.0])?;
    let s = opt_floats("s", s, [1.0; 3])?;
    Ok(Transform {
        translation: Vec3::from_array(t),
        rotation: Quat::from_xyzw(r[0], r[1], r[2], r[3]),
        scale: Vec3::from_array(s),
    })
}

/// Sets `key` on a fresh plain object. Defining a property on a plain `Object` created
/// here cannot throw, so the result is ignored.
fn set(obj: &Object, key: &str, value: &JsValue) {
    let _ = Reflect::set(obj, &JsValue::from_str(key), value);
}

fn number_array(values: &[u32]) -> Array {
    values.iter().map(|&v| JsValue::from(v)).collect()
}

#[wasm_bindgen(js_class = Engine)]
impl Engine {
    /// Creates the engine on `canvas` at its current width and height. `options`:
    /// `maxInstances` (16384), `maxMaterials` (256), `powerPreference`
    /// ('high-performance' or 'low-power'), `msaa` (1 or 4, default 4), `toneMapper`
    /// ('agx', 'aces' or 'linear', default 'agx'), `exposureEv` (finite, default 0).
    /// The device is requested on a core adapter with the WEBGPU_OPTIONAL features the
    /// adapter has and the adapter's limits.
    pub async fn create(
        canvas: web_sys::HtmlCanvasElement,
        options: JsValue,
    ) -> Result<Engine, JsValue> {
        let o = js::engine_options(&options).map_err(|e| js_err(&e))?;
        let state: Shared = Rc::new(RefCell::new(None));
        let app = EngineApp {
            state: Rc::clone(&state),
            opts: o,
        };
        let opts = RuntimeOptions {
            width: canvas.width(),
            height: canvas.height(),
            gpu: GpuOptions {
                power_preference: o.power_preference,
                allow_software: true,
            },
            ..RuntimeOptions::default()
        };
        let req = DeviceRequest {
            optional_features: WEBGPU_OPTIONAL,
            adapter_limits: true,
        };
        let viewport = (opts.width.max(1), opts.height.max(1));
        let runtime = WebRuntime::new_with(canvas, Box::new(app), opts, &req)
            .await
            .map_err(|e| js_err(&e))?;
        Ok(Engine {
            runtime: Some(runtime),
            state,
            viewport,
        })
    }

    /// Renders one frame at `nowMs`.
    pub fn render(&mut self, now_ms: f64) -> Result<(), JsValue> {
        self.runtime()?.frame(now_ms).map_err(|e| js_err(&e))
    }

    /// Resizes the canvas backing store (zero sizes clamp to 1); the camera aspect follows.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), JsValue> {
        let size = (width.max(1), height.max(1));
        // Reconfiguring an unchanged canvas clears its drawable in the browser.
        // Editor layout observers may report the same size on successive frames.
        if self.viewport != size {
            self.runtime()?.resize(size.0, size.1);
            self.viewport = size;
        } else {
            self.runtime()?;
        }
        Ok(())
    }

    /// Frame statistics, the adapter and the hardware flag.
    pub fn stats(&self) -> Result<JsValue, JsValue> {
        let runtime = self.runtime.as_ref().ok_or_else(disposed)?;
        Ok(stats_to_js(&runtime.stats(), runtime.adapter()))
    }

    /// Renders one frame and tells whether any pixel differs from the clear colour.
    #[wasm_bindgen(js_name = probeNonEmpty)]
    pub async fn probe_non_empty(&mut self, now_ms: f64) -> Result<bool, JsValue> {
        self.runtime()?
            .probe_non_empty(now_ms)
            .await
            .map_err(|e| js_err(&e))
    }

    /// Releases the GPU objects and the scene; every later call throws `api.disposed`.
    pub fn dispose(&mut self) {
        self.runtime = None;
        self.state.borrow_mut().take();
    }

    /// Adds a material (`color` sRGB, `roughness`, `metallic`, `emissive` linear,
    /// `model` 'flat' or 'pbr') and returns its handle.
    #[wasm_bindgen(js_name = addMaterial)]
    pub fn add_material(&mut self, desc: JsValue) -> Result<u32, JsValue> {
        self.with_state(|_, s| {
            let m = js::material_desc(&desc)?.to_material()?;
            s.check_material_room(1)?;
            s.add_material(&m)
        })
    }

    /// Replaces the material behind `id`.
    #[wasm_bindgen(js_name = setMaterial)]
    pub fn set_material(&mut self, id: u32, desc: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, s| {
            let target = s.material(id)?;
            let m = js::material_desc(&desc)?.to_material()?;
            s.lit
                .materials_mut()
                .set(target, SurfaceMaterial::from_v0(&m))
        })
    }

    /// Generates a primitive of `kind` (S3 names and params) at `segments` (LOD0 when
    /// absent), uploads it and returns its mesh handle.
    #[wasm_bindgen(js_name = primitiveMesh)]
    pub fn primitive_mesh(
        &mut self,
        kind: String,
        params: JsValue,
        segments: Option<f64>,
    ) -> Result<u32, JsValue> {
        self.with_state(|gpu, s| {
            let p = js::prim_params(&kind, &params)?;
            let q = quality(segments)?;
            s.geometry
                .generate(&kind, &p, q, &mut s.scratch)
                .map_err(|e| geom_error(&e))?;
            s.upload_scratch(gpu)
        })
    }

    /// Frees a mesh (any mesh handle: primitiveMesh, loadPrimsAsset or addMesh); its
    /// handle is never reused.
    #[wasm_bindgen(js_name = removeMesh)]
    pub fn remove_mesh(&mut self, id: u32) -> Result<(), JsValue> {
        self.with_state(|_, s| s.remove_mesh(id))
    }

    /// Sets the camera: `eye`, `target`, `up` (+Y), `fovYDeg` (60), `near` (0.1),
    /// `far` (200), `projection` (perspective/orthographic), `orthoHeight` (10 metres).
    /// `position` aliases `eye`. The aspect follows the canvas.
    #[wasm_bindgen(js_name = setCamera)]
    pub fn set_camera(&mut self, desc: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, s| {
            let d = js::camera_desc(&desc)?;
            if !(d.fov_y_deg > 0.0 && d.fov_y_deg < 180.0) {
                return Err(invalid(format!(
                    "'fovYDeg' must be in (0, 180), got {}",
                    d.fov_y_deg
                )));
            }
            let valid_near = if d.ortho_height.is_some() { d.near >= 0.0 } else { d.near > 0.0 };
            if !(valid_near && d.far > d.near) {
                return Err(invalid(format!(
                    "'near' must be > 0 (>= 0 for orthographic) and 'far' > 'near', got near {} far {}",
                    d.near, d.far
                )));
            }
            let (eye, target) = (Vec3::from_array(d.eye), Vec3::from_array(d.target));
            if (target - eye).length_squared() <= 1e-12 {
                return Err(invalid("'eye' and 'target' must differ".to_string()));
            }
            let up = Vec3::from_array(d.up);
            if up.length_squared() <= 1e-12
                || (target - eye)
                    .normalize_or_zero()
                    .cross(up.normalize_or_zero())
                    .length_squared()
                    <= 1e-12
            {
                return Err(invalid(
                    "'up' must be non-zero and not parallel to the viewing direction".into(),
                ));
            }
            let cam = &mut s.scene.camera;
            cam.eye = eye;
            cam.target = target;
            cam.up = Vec3::from_array(d.up);
            cam.fov_y_rad = d.fov_y_deg.to_radians();
            cam.ortho_height = d.ortho_height;
            cam.near = d.near;
            cam.far = d.far;
            Ok(())
        })
    }

    /// Rotates the orbit camera by the given yaw and pitch (radians, pitch clamped to
    /// +-89 degrees) and multiplies its distance by `zoomFactor` (> 0).
    pub fn orbit(
        &mut self,
        d_yaw_rad: f64,
        d_pitch_rad: f64,
        zoom_factor: f64,
    ) -> Result<(), JsValue> {
        self.with_state(|_, s| {
            let d_yaw = finite("dYawRad", d_yaw_rad)?;
            let d_pitch = finite("dPitchRad", d_pitch_rad)?;
            let zoom = finite("zoomFactor", zoom_factor)?;
            if zoom <= 0.0 {
                return Err(invalid(format!("'zoomFactor' must be > 0, got {zoom}")));
            }
            s.orbit.rotate(d_yaw, d_pitch);
            s.orbit.zoom(zoom);
            s.orbit.apply(&mut s.scene.camera);
            Ok(())
        })
    }

    /// Replaces the orbit camera: `target` [x, y, z], `distance` (> 0), yaw and pitch in
    /// radians.
    #[wasm_bindgen(js_name = setOrbit)]
    pub fn set_orbit(
        &mut self,
        target: &[f32],
        distance: f64,
        yaw_rad: f64,
        pitch_rad: f64,
    ) -> Result<(), JsValue> {
        self.with_state(|_, s| {
            let target = vec3_arg("target", target)?;
            let distance = finite("distance", distance)?;
            if distance <= 0.0 {
                return Err(invalid(format!("'distance' must be > 0, got {distance}")));
            }
            let (yaw, pitch) = (finite("yawRad", yaw_rad)?, finite("pitchRad", pitch_rad)?);
            let mut orbit = OrbitController::new(target, distance);
            orbit.rotate(yaw, pitch);
            s.orbit = orbit;
            s.orbit.apply(&mut s.scene.camera);
            Ok(())
        })
    }

    /// Sets the directional light: `direction`, linear `color`, `intensity`.
    #[wasm_bindgen(js_name = setLight)]
    pub fn set_light(&mut self, desc: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, s| {
            s.scene.light = js::light_desc(&desc)?;
            Ok(())
        })
    }

    /// Sets the ambient term: linear `color`, `intensity`.
    #[wasm_bindgen(js_name = setAmbient)]
    pub fn set_ambient(&mut self, desc: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, s| {
            s.scene.ambient = js::ambient_desc(&desc)?;
            Ok(())
        })
    }

    /// Adds a node under `parent` (null for a root) with translation `t`, rotation `r`
    /// [x, y, z, w] and scale `s` (each defaults to the identity) and returns its handle.
    #[wasm_bindgen(js_name = addNode)]
    pub fn add_node(
        &mut self,
        parent: Option<u32>,
        t: Option<Vec<f32>>,
        r: Option<Vec<f32>>,
        s: Option<Vec<f32>>,
    ) -> Result<u32, JsValue> {
        self.with_state(|_, st| {
            let parent = st.parent(parent)?;
            let local = trs_arg(t.as_deref(), r.as_deref(), s.as_deref())?;
            st.add_node(parent, local)
        })
    }

    /// Sets the local transform of `node`.
    #[wasm_bindgen(js_name = setTransform)]
    pub fn set_transform(
        &mut self,
        node: u32,
        t: &[f32],
        r: &[f32],
        s: &[f32],
    ) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            let id = st.nodes.get(node)?;
            let local = trs_arg(Some(t), Some(r), Some(s))?;
            st.scene.set_local(id, local).map_err(|e| scene_error(&e))
        })
    }

    /// Sets many local transforms: `trs` holds 10 floats per node (t3, r4 [x, y, z, w],
    /// s3). A length mismatch throws `api.length_mismatch`. Nodes before a failing entry
    /// keep their new transform.
    #[wasm_bindgen(js_name = setTransforms)]
    pub fn set_transforms(
        &mut self,
        nodes: &Uint32Array,
        trs: &Float32Array,
    ) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            let (n, f) = (nodes.length() as usize, trs.length() as usize);
            check_trs_len(n, f)?;
            // Scratch buffers grow only for a larger input, so steady-state calls do not
            // allocate.
            if st.scratch_nodes.len() < n {
                st.scratch_nodes.resize(n, 0);
            }
            if st.scratch_trs.len() < f {
                st.scratch_trs.resize(f, 0.0);
            }
            let (handles, floats) = (&mut st.scratch_nodes[..n], &mut st.scratch_trs[..f]);
            nodes.copy_to(handles);
            trs.copy_to(floats);
            for (i, &h) in handles.iter().enumerate() {
                let id = st.nodes.get(h)?;
                if !floats[i * 10..i * 10 + 10].iter().all(|c| c.is_finite()) {
                    return Err(invalid(format!(
                        "'trs' of node {i} must hold finite numbers"
                    )));
                }
                st.scene
                    .set_local(id, decode_trs(floats, i))
                    .map_err(|e| scene_error(&e))?;
            }
            Ok(())
        })
    }

    /// Removes `node`, its subtree and their instances. The handles of removed
    /// descendants and instances stay mapped to stale ids (instance groups are marked
    /// stale): using them throws `scene.stale_node` or `scene.stale_instance`.
    #[wasm_bindgen(js_name = removeNode)]
    pub fn remove_node(&mut self, node: u32) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            let id = st.nodes.remove(node)?;
            st.remove_node(id)
        })
    }

    /// Draws `mesh` with `material` at `node` and returns the instance handle (an
    /// instance group: every sub-mesh of `mesh` drawn with `material`).
    #[wasm_bindgen(js_name = addInstance)]
    pub fn add_instance(&mut self, node: u32, mesh: u32, material: u32) -> Result<u32, JsValue> {
        self.with_state(|_, st| {
            let node = st.node(node)?;
            st.add_instance(node, mesh, material)
        })
    }

    /// Removes an instance (its whole group); its handle is never reused.
    #[wasm_bindgen(js_name = removeInstance)]
    pub fn remove_instance(&mut self, id: u32) -> Result<(), JsValue> {
        self.with_state(|_, st| st.remove_instance_group(id))
    }

    /// Shows or hides an instance (its whole group).
    #[wasm_bindgen(js_name = setVisible)]
    pub fn set_visible(&mut self, id: u32, visible: bool) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            let group = st.instances.get_mut(id)?;
            group.set_visible(&mut st.scene, visible)
        })
    }

    /// Loads a prims v0 asset at `segments` (LOD0 when absent) with the 8 S3 kinds and
    /// the CSG resolver. Registers its materials and its meshes (baked into asset space)
    /// and returns `{id, name, triangles, budget, meshes, materials, anchors}`: `meshes`
    /// and `materials` are handles (materials in the asset's material order), `anchors`
    /// maps each name to 16 column-major floats. Loader errors keep their code.
    #[wasm_bindgen(js_name = loadPrimsAsset)]
    pub fn load_prims_asset(
        &mut self,
        json: String,
        segments: Option<f64>,
    ) -> Result<JsValue, JsValue> {
        self.with_state(|gpu, st| {
            let ctx = LoadContext {
                geometry: &st.geometry,
                booleans: &st.booleans,
                quality: quality(segments)?,
                strict_kinds: true,
            };
            let asset = st
                .loaders
                .load(json.as_bytes(), &ctx)
                .map_err(|e| load_error(&e))?;
            st.check_material_room(asset.materials.len())?;
            let mut materials = Vec::with_capacity(asset.materials.len());
            for (_, m) in &asset.materials {
                materials.push(st.add_material(m)?);
            }
            let mut parts = Vec::with_capacity(asset.meshes.len());
            for loaded in &asset.meshes {
                let index = asset
                    .materials
                    .iter()
                    .position(|(name, _)| *name == loaded.material)
                    .ok_or_else(|| {
                        api_error(
                            "material.unknown",
                            format!(
                                "mesh of node '{}' uses unknown material '{}'",
                                loaded.node_id, loaded.material
                            ),
                        )
                    })?;
                st.scratch.clear();
                st.scratch
                    .append_transformed(&loaded.mesh, &loaded.transform);
                let mesh = st.upload_scratch(gpu)?;
                // Invariant: `index` comes from `asset.materials`, which `materials` mirrors.
                parts.push((mesh, materials[index]));
            }
            let mesh_handles: Vec<u32> = parts.iter().map(|&(m, _)| m).collect();
            let index = st.assets.len() as u32;
            st.assets.push(AssetEntry { parts });
            let handle = st.asset_handles.insert(index);

            let anchors = Object::new();
            for (name, m) in &asset.anchors {
                set(&anchors, name, &js::mat4_to_f32array(m));
            }
            let info = Object::new();
            set(&info, "id", &JsValue::from(handle));
            set(&info, "name", &JsValue::from_str(&asset.name));
            set(&info, "triangles", &JsValue::from(asset.triangles));
            set(&info, "budget", &JsValue::from(asset.budget));
            set(&info, "meshes", &number_array(&mesh_handles));
            set(&info, "materials", &number_array(&materials));
            set(&info, "anchors", &anchors);
            Ok(JsValue::from(info))
        })
    }

    /// Spawns a loaded asset: a new root node under `parent` (a root when absent), one
    /// identity child node per asset mesh, each with one instance of that mesh and its
    /// material. Returns `{root, nodes, instances}` (all new handles).
    #[wasm_bindgen(js_name = spawnAsset)]
    pub fn spawn_asset(&mut self, asset: u32, parent: Option<u32>) -> Result<JsValue, JsValue> {
        self.with_state(|_, st| {
            let index = st.asset_handles.get(asset)? as usize;
            let parent = st.parent(parent)?;
            let parts = match st.assets.get(index) {
                Some(entry) => entry.parts.clone(),
                None => {
                    return Err(api_error(
                        UNKNOWN_HANDLE,
                        format!("unknown asset handle {asset}"),
                    ));
                }
            };
            let root_handle = st.add_node(parent, Transform::IDENTITY)?;
            let root = st.nodes.get(root_handle)?;
            let mut nodes = Vec::with_capacity(parts.len());
            let mut instances = Vec::with_capacity(parts.len());
            for (mesh, material) in parts {
                let child_handle = st.add_node(Some(root), Transform::IDENTITY)?;
                let child = st.nodes.get(child_handle)?;
                nodes.push(child_handle);
                instances.push(st.add_instance(child, mesh, material)?);
            }
            let out = Object::new();
            set(&out, "root", &JsValue::from(root_handle));
            set(&out, "nodes", &number_array(&nodes));
            set(&out, "instances", &number_array(&instances));
            Ok(JsValue::from(out))
        })
    }
}
