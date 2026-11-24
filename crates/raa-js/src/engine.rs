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
    FrameCounters, FrameInput, FrameTiming, PredictionKey, PresentTarget, RuntimeOptions, WebApp, WebRuntime, error_to_js,
    stats_to_js,
};
use wasm_bindgen::prelude::*;

use crate::api;
use crate::convert::{
    DISPOSED, EngineOptions, HandleMap, INVALID_ARGUMENT, UNKNOWN_HANDLE, api_error, check_trs_len,
    decode_trs, geom_error, js, load_error, quality, scene_error,
};
use crate::feature_map::FeatureMap;
use crate::state::{AssetEntry, EngineState, MeshEntry, MeshOrigin, SceneInvocation, SkinFeature};

/// WEBGPU_OPTIONAL: device features requested when the adapter has them, never required.
/// `timestamp-query` lets a profiler measure the GPU time of each pass (nothing in RAA needs it).
const WEBGPU_OPTIONAL: wgpu::Features = wgpu::Features::FLOAT32_FILTERABLE
    .union(wgpu::Features::RG11B10UFLOAT_RENDERABLE)
    .union(wgpu::Features::TIMESTAMP_QUERY);

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
        self.node_parents.push((id, parent));
        Ok(self.nodes.insert(id))
    }

    /// Uploads `scratch` as an attributed mesh (one sub-mesh, S10 default attributes) and
    /// returns its new v0 mesh handle. The scratch buffer keeps its capacity.
    fn upload_scratch(&mut self, gpu: &GpuContext) -> Result<u32, RaaError> {
        let mesh = AttributedMesh::from_mesh(std::mem::take(&mut self.scratch));
        let uploaded = self.lit.upload(gpu, &mesh);
        let lit = match uploaded {
            Ok(lit) => lit,
            Err(e) => {
                self.scratch = mesh.base;
                return Err(e);
            }
        };
        let handle = self.meshes.insert(MeshEntry {
            lit,
            origin: MeshOrigin::V0,
        });
        self.retain_mesh_source(handle, &mesh);
        self.scratch = mesh.base;
        Ok(handle)
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
    js_err(&disposed_error())
}

fn disposed_error() -> RaaError {
    api_error(DISPOSED, "the engine was disposed")
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
        let trace = raa_sdf_trace::TraceFeature::install(gpu, &mut pipeline.borrow_mut())?;
        features.insert(trace);
        api::install_all(&mut InstallCtx {
            gpu,
            surface_format,
            pipeline: &pipeline,
            lit: &mut lit,
            features: &mut features,
        })?;
        let state = Rc::downgrade(&self.state);
        pipeline
            .borrow_mut()
            .set_view_prepare(Box::new(move |gpu, index, size| {
                let shared = state
                    .upgrade()
                    .ok_or_else(|| api_error(DISPOSED, "the engine was disposed"))?;
                let mut guard = shared.borrow_mut();
                let st = guard
                    .as_mut()
                    .ok_or_else(|| api_error(DISPOSED, "the engine was disposed"))?;
                prepare_sky_view(st, index, size)?;
                crate::trace::prepare_view(st, index, size)?;
                st.prepare_pipeline_view(gpu, index, size)
            }));
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
            slots: Vec::with_capacity(max),
            orbit,
            geometry,
            loaders,
            booleans: CsgResolver,
            scratch: MeshData::new(),
            lit,
            pipeline,
            tonemap,
            features,
            scene_invocation: SceneInvocation::default(),
            sky_invocation_id: 0,
            max_materials: o.max_materials,
            meshes: HandleMap::new("mesh"),
            skins: Vec::new(),
            materials: HandleMap::new("material"),
            nodes: HandleMap::new("node"),
            instances: HandleMap::new("instance"),
            node_parents: Vec::with_capacity(max),
            dirty_meshes: Vec::new(),
            single_material: false,
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
        s.advance_sky_frame(input)?;
        api::frame_all(s, gpu, input)?;
        retire_sky_views(s)?;
        s.rebuild_dirty_groups()?;
        s.flush_skins(gpu)?;
        let size = s
            .features
            .get::<api::ViewsState>()
            .map_or(input.size, |views| {
                let r = views.viewports()[0];
                let x = (r.x * input.size.0 as f64).round() as u32;
                let y = (r.y * input.size.1 as f64).round() as u32;
                (
                    ((r.x + r.width) * input.size.0 as f64).round() as u32 - x,
                    ((r.y + r.height) * input.size.1 as f64).round() as u32 - y,
                )
            });
        prepare_sky_view(s, 0, size)?;
        crate::trace::prepare_view(s, 0, input.size)?;
        // Viewport 0 of the invocation; split views select their slot in the callback.
        let discontinuity = s.scene_invocation.is_discontinuous(0);
        s.lit.prepare_scene_view(0, discontinuity);
        let c = s
            .lit
            .prepare(gpu, &mut s.scene, &mut s.list, &mut s.tags, &mut s.slots)?;
        crate::mesh_trace::prepare(s, 0)?;
        Ok(FrameCounters {
            instances: c.instances,
            draw_calls: c.draw_calls,
            triangles: c.triangles,
        })
    }

    fn resize(&mut self, _gpu: &GpuContext, size: (u32, u32)) {
        if let Some(s) = self.state.borrow_mut().as_mut() {
            s.scene.camera.set_viewport(size.0, size.1);
            s.scene_invocation.discontinuity_all();
        }
    }
}

/// Mirror m07's effective camera without consuming its discontinuity or changing the scene.
fn prepare_sky_view(st: &EngineState, index: usize, size: (u32, u32)) -> Result<(), RaaError> {
    let Some(sky) = st.features.get::<raa_sky::SkyHandle>() else {
        return Ok(());
    };
    let mut camera = st.scene.camera;
    let own = st
        .features
        .get::<api::ViewsState>()
        .and_then(|v| v.camera(index));
    if let Some(c) = own {
        camera.eye = Vec3::from_array(c.eye);
        camera.target = Vec3::from_array(c.target);
        camera.up = Vec3::Y;
        camera.fov_y_rad = c.fov_y_deg.to_radians();
        camera.near = c.near;
        camera.far = c.far;
        camera.ortho_height = None;
    }
    let cut = st.scene_invocation.is_discontinuous(index);
    camera.set_viewport(size.0, size.1);
    sky.set_view(index as u32, &camera, !cut)
}

fn retire_sky_views(st: &EngineState) -> Result<(), RaaError> {
    let count = st
        .features
        .get::<api::ViewsState>()
        .map_or(1, |v| v.viewports().len());
    if let Some(sky) = st.features.get::<raa_sky::SkyHandle>() {
        for view in count..4 {
            sky.remove_view(view as u32)?;
        }
    }
    Ok(())
}

/// The RAA engine: one canvas, one scene, the HDR lit pipeline.
#[wasm_bindgen(js_name = Engine)]
pub struct Engine {
    runtime: Option<WebRuntime>,
    prediction_warp: Option<crate::prediction_warp::PredictionWarp>,
    prediction_real: Option<(u64, raa_tonemap::TonemapSource)>,
    state: Shared,
    pub(crate) viewport: (u32, u32),
}

impl Engine {
    fn runtime(&mut self) -> Result<&mut WebRuntime, JsValue> {
        let runtime = self.runtime.as_mut().ok_or_else(disposed)?;
        // Once the browser loses the device every GPU call would panic inside wgpu (the lobby
        // rebuild's skin upload did); refuse with a typed `core.gpu` instead.
        runtime.gpu().ensure_live().map_err(|e| js_err(&e))?;
        Ok(runtime)
    }

    /// Resizes the canvas backing store (zero sizes clamp to 1); the camera aspect follows.
    /// The JS `resize(cssWidth, cssHeight, dpr?)` lives in api/m21_views.rs.
    /// The material textures (`addTexture`, `removeTexture`) live in api/m22_textures.rs.
    pub(crate) fn resize_backing(&mut self, width: u32, height: u32) -> Result<(), JsValue> {
        let size = (width.max(1), height.max(1));
        // Reconfiguring an unchanged canvas clears its drawable in the browser.
        // Editor layout observers may report the same size on successive frames.
        if self.viewport != size {
            self.runtime()?.resize(size.0, size.1);
            self.viewport = size;
            self.prediction_real = None;
        } else {
            self.runtime()?;
        }
        Ok(())
    }

    /// Runs `f` on the state, with the GPU context. The borrow ends before returning.
    /// Used by the engine methods and the api files.
    pub(crate) fn with_state<R>(
        &self,
        f: impl FnOnce(&GpuContext, &mut EngineState) -> Result<R, RaaError>,
    ) -> Result<R, JsValue> {
        let runtime = self.runtime.as_ref().ok_or_else(disposed)?;
        runtime.gpu().ensure_live().map_err(|e| js_err(&e))?;
        let mut guard = self.state.borrow_mut();
        let s = guard.as_mut().ok_or_else(disposed)?;
        f(runtime.gpu(), s).map_err(|e| js_err(&e))
    }

    /// Real source ID for the native renderer preparing a prediction. The renderer
    /// must have retained HDR and guides for this same submission.
    pub fn prediction_source(&mut self, snapshot: &JsValue) -> Result<Option<u64>, JsValue> {
        let target = present_target(snapshot).map_err(|e| js_err(&e))?;
        if prediction_key(snapshot, target).is_none() {
            return Ok(None);
        }
        Ok(self.runtime()?.prediction_source(target))
    }

    /// Accept a device-side prediction from a renderer extension. The same engine
    /// snapshot is validated again at presentation, after any newer input or rollback.
    pub fn submit_prediction(
        &mut self,
        snapshot: &JsValue,
        source_id: u64,
        deadline_ms: f64,
        measured_cost_ms: f64,
        texture: wgpu::Texture,
        commands: wgpu::CommandBuffer,
    ) -> Result<bool, JsValue> {
        let target = present_target(snapshot).map_err(|e| js_err(&e))?;
        let Some(key) = prediction_key(snapshot, target) else {
            return Ok(false);
        };
        self.runtime()?
            .submit_prediction(key, source_id, deadline_ms, measured_cost_ms, texture, commands)
            .map_err(|e| js_err(&e))
    }

    /// Prepare the analytic pipeline ahead of the display deadline on this device.
    pub fn prepare_prediction_warp(&mut self) -> Result<&GpuContext, JsValue> {
        if self.prediction_warp.is_none() {
            let warp = crate::prediction_warp::PredictionWarp::new(self.runtime()?.gpu());
            self.prediction_warp = Some(warp);
        }
        Ok(self.runtime()?.gpu())
    }

    /// Backward-warp the retained real source and shade the bounded repair list.
    /// Inputs and the stateless renderer must come from the same device and source
    /// frame / target snapshot pair. Presentation revalidates the snapshot.
    pub fn submit_warped_prediction(
        &mut self,
        snapshot: &JsValue,
        source_id: u64,
        deadline_ms: f64,
        measured_cost_ms: f64,
        guides: crate::PredictionGuides<'_>,
        renderer: &mut impl crate::PredictionRenderer,
    ) -> Result<bool, JsValue> {
        let target = present_target(snapshot).map_err(|e| js_err(&e))?;
        let Some(key) = prediction_key(snapshot, target) else {
            return Ok(false);
        };
        let runtime = self.runtime.as_mut().ok_or_else(disposed)?;
        let Some(warp) = self.prediction_warp.as_ref() else {
            return Ok(false);
        };
        let Some((captured_id, source)) = &self.prediction_real else {
            return Ok(false);
        };
        if *captured_id != source_id {
            return Ok(false);
        }
        runtime
            .record_checked_prediction(key, source_id, deadline_ms, measured_cost_ms, |gpu, output| {
                warp.record(gpu, source, guides, output, renderer)
            })
            .map_err(|e| js_err(&e))
    }

    /// Opens one scene invocation: clear the observation and open the renderer's index.
    fn begin_scene_invocation(&mut self) -> Result<(), RaaError> {
        let runtime = self.runtime.as_ref().ok_or_else(disposed_error)?;
        runtime.gpu().ensure_live()?;
        let mut guard = self.state.borrow_mut();
        let s = guard.as_mut().ok_or_else(disposed_error)?;
        let mut pipeline = s.pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
        crate::state::next_sky_invocation(&mut s.sky_invocation_id)?;
        let index = s.scene_invocation.begin();
        pipeline.begin_scene_submission();
        s.lit.begin_scene_frame(index);
        Ok(())
    }

    fn render_frame(
        &mut self,
        now_ms: f64,
        timing: Option<&mut FrameTiming>,
    ) -> Result<(), RaaError> {
        self.prediction_real = None;
        self.begin_scene_invocation()?;
        let result = self
            .runtime
            .as_mut()
            .ok_or_else(disposed_error)
            .and_then(|runtime| match timing {
                Some(timing) => runtime.frame_with_timing(now_ms, timing),
                None => runtime.frame(now_ms),
            });
        self.finish_scene_invocation(result.is_ok());
        result
    }

    /// Closes it from the pipeline observation (only real submissions are acknowledged).
    fn finish_scene_invocation(&mut self, runtime_ok: bool) {
        let mut guard = self.state.borrow_mut();
        let Some(s) = guard.as_mut() else { return };
        let (submission, suspended) = s
            .pipeline
            .try_borrow()
            .map(|p| (p.scene_submission(), p.suspended_views()))
            .unwrap_or_default();
        let (acknowledge, complete) = s.scene_invocation.finish(runtime_ok, submission);
        for view in (0..4).filter(|v| acknowledge & (1 << v) != 0) {
            s.lit.submitted_scene_view(view);
        }
        if complete && (runtime_ok || acknowledge != 0) {
            for view in (0..4).filter(|v| suspended & (1 << v) != 0) {
                s.lit.suspend_scene_view(view);
            }
        }
        s.lit.finish_scene_frame(complete);
        if let Some(sky) = s.features.get::<raa_sky::SkyHandle>() {
            sky.observe_submission(s.sky_invocation_id(), acknowledge, complete);
        }
        if let Some(vg) = s.features.get::<raa_vg::VgHandle>() {
            vg.observe_submission(s.sky_invocation_id(), complete);
        }
        if let Some(gi) = s.features.get::<raa_gi::GiHandle>() {
            gi.observe_submission(s.sky_invocation_id(), complete);
        }
    }
}

/// Consume the identity from the engine RenderSnapshot / PredictionSnapshot.
fn present_target(snapshot: &JsValue) -> Result<PresentTarget, RaaError> {
    fn integer(object: &JsValue, name: &str) -> Result<u64, RaaError> {
        let value = js::required_f64(object, name)?;
        if (0.0..=9_007_199_254_740_991.0).contains(&value) && value.fract() == 0.0 {
            Ok(value as u64)
        } else {
            Err(invalid(format!("'{name}' must be a nonnegative safe integer")))
        }
    }
    let target = js_sys::Reflect::get(snapshot, &JsValue::from_str("target"))
        .map_err(|_| invalid("cannot read snapshot target".into()))?;
    let phase = js::required_f64(&target, "phase")?;
    if !(0.0..=1.0).contains(&phase) {
        return Err(invalid("snapshot phase must be in [0, 1]".into()));
    }
    Ok(PresentTarget {
        revision: integer(snapshot, "revision")?,
        tick: integer(&target, "tick")?,
        phase,
    })
}

/// Missing or unconfirmed evidence disables speculation while preserving real fallback.
fn prediction_key(snapshot: &JsValue, target: PresentTarget) -> Option<PredictionKey> {
    let field = |object: &JsValue, name: &str| {
        js_sys::Reflect::get(object, &JsValue::from_str(name)).ok()
    };
    let array = |object: &JsValue, name: &str| {
        field(object, name)?.dyn_into::<js_sys::Array>().ok()
    };
    let inputs = field(snapshot, "inputs")?;
    let cutoff_ms = js::required_f64(&inputs, "cutoffMs").ok()?;
    if cutoff_ms < 0.0 || js::required_f64(snapshot, "observedAtMs").ok()? != cutoff_ms
        || js::required_f64(&inputs, "inputTick").ok()? != target.tick.checked_sub(1)? as f64
        || array(snapshot, "discontinuities")?.length() != 0
    {
        return None;
    }
    array(snapshot, "poses")?;
    array(snapshot, "lights")?;
    let ids = array(&inputs, "inputIds")?;
    let receipts = array(&inputs, "receivedAtMs")?;
    let confirmed = array(&inputs, "confirmedMask")?;
    if ids.length() != receipts.length() || ids.length() != confirmed.length() {
        return None;
    }
    for i in 0..ids.length() {
        if confirmed.get(i).as_bool() != Some(true)
            || ids.get(i).as_string().is_none_or(|id| id.is_empty())
            || receipts.get(i).as_f64().is_none_or(|ms| !(0.0..=cutoff_ms).contains(&ms))
        {
            return None;
        }
    }
    Some(PredictionKey {
        target,
        cutoff_ms,
        snapshot_identity: js_sys::JSON::stringify(snapshot).ok()?.as_string()?,
    })
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

fn check_frame_trace_id(id: &str, key: &str) -> Result<(), RaaError> {
    if id.trim().is_empty() {
        return Err(invalid(format!("'{key}' must be non-blank")));
    }
    Ok(())
}

fn frame_trace_id(context: &JsValue, key: &str) -> Result<String, RaaError> {
    let id = js::string(context, key)?;
    check_frame_trace_id(&id, key)?;
    Ok(id)
}

fn check_frame_trace_event(
    event_id: Option<&str>,
    event_ms: Option<f64>,
    accepted_ms: Option<f64>,
    cutoff_ms: f64,
) -> Result<(), RaaError> {
    if let Some(event_id) = event_id {
        check_frame_trace_id(event_id, "eventId")?;
    }
    match (event_ms, accepted_ms) {
        (None, None) => Ok(()),
        (Some(event), Some(accepted)) if event_id.is_some() => {
            if 0.0 <= event && event <= accepted && accepted <= cutoff_ms {
                Ok(())
            } else {
                Err(invalid(
                    "key timestamps must satisfy 0 <= eventMs <= acceptedMs <= inputCutoffMs"
                        .into(),
                ))
            }
        }
        _ => Err(invalid(
            "key timestamps require 'eventId', 'eventMs' and 'acceptedMs' together".into(),
        )),
    }
}

#[wasm_bindgen(js_class = Engine)]
impl Engine {
    /// Creates the engine on `canvas` at its current width and height. `options`:
    /// `maxInstances` (16384), `maxMaterials` (256), `powerPreference`
    /// ('high-performance' or 'low-power'), `msaa` (1 or 4, default 4), `toneMapper`
    /// ('agx', 'aces' or 'linear', default 'agx'), `exposureEv` (finite, default 0).
    /// Set `allowSoftware: false` to require a hardware adapter.
    /// The device is requested on a core adapter with the WEBGPU_OPTIONAL features the
    /// adapter has and the adapter's limits.
    pub async fn create(
        canvas: web_sys::HtmlCanvasElement,
        options: JsValue,
    ) -> Result<Engine, JsValue> {
        let o = js::engine_options(&options).map_err(|e| js_err(&e))?;
        let allow_software = if options.is_null() || options.is_undefined() {
            true
        } else {
            js::optional_bool(&options, "allowSoftware")
                .map_err(|e| js_err(&e))?
                .unwrap_or(true)
        };
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
                allow_software,
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
            prediction_warp: None,
            prediction_real: None,
            state,
            viewport,
        })
    }

    /// Renders one frame at `nowMs`; one invocation, one scene-frame index.
    pub fn render(&mut self, now_ms: f64) -> Result<(), JsValue> {
        self.render_frame(now_ms, None).map_err(|e| js_err(&e))
    }

    /// Resolves when this engine's WebGPU queue has finished work submitted so far.
    #[wasm_bindgen(js_name = submittedWorkDone)]
    pub fn submitted_work_done(&mut self) -> Result<js_sys::Promise<js_sys::Undefined>, JsValue> {
        let queue = self.runtime()?.gpu().queue().as_webgpu()
            .ok_or_else(|| js_err(&RaaError::Unsupported("WebGPU queue required".into())))?;
        Ok(queue.on_submitted_work_done())
    }

    /// Whether the canvas supports the copy used by split presentation.
    #[wasm_bindgen(js_name = supportsSplitPresentation)]
    pub fn supports_split_presentation(&self) -> Result<bool, JsValue> {
        let runtime = self.runtime.as_ref().ok_or_else(disposed)?;
        Ok(runtime.supports_split_presentation())
    }

    /// Retire pending presentation images after a cut or configuration change.
    #[wasm_bindgen(js_name = resetPresentation)]
    pub fn reset_presentation(&mut self) -> Result<(), JsValue> {
        self.runtime()?.clear_retained_frames();
        self.prediction_real = None;
        Ok(())
    }

    /// Opt-in split production. Returns no ID while a real image is pending.
    #[wasm_bindgen(js_name = produceReal)]
    pub fn produce_real(&mut self, now_ms: f64, snapshot: JsValue) -> Result<Option<f64>, JsValue> {
        let target = present_target(&snapshot).map_err(|e| js_err(&e))?;
        if !now_ms.is_finite() {
            return Err(js_err(&invalid("nowMs must be finite".into())));
        }
        if !self.runtime()?.can_produce_real() {
            return Ok(None);
        }
        self.begin_scene_invocation().map_err(|e| js_err(&e))?;
        let capture = self.prediction_warp.as_ref().and_then(|_| {
            let state = self.state.borrow();
            let state = state.as_ref()?;
            let single = state.features.get::<api::ViewsState>().is_none_or(|views| {
                matches!(views.viewports(), [rect]
                    if (rect.x, rect.y, rect.width, rect.height) == (0.0, 0.0, 1.0, 1.0))
            });
            single.then(|| state.tonemap.clone())
        });
        if let Some(control) = &capture {
            control.request_source_capture();
        }
        let result = self
            .runtime
            .as_mut()
            .ok_or_else(disposed_error)
            .and_then(|runtime| runtime.produce_real(now_ms, target));
        self.finish_scene_invocation(result.is_ok());
        let source = capture.and_then(|control| control.take_source_capture());
        if let Ok(id) = result {
            let clean = self
                .state
                .borrow()
                .as_ref()
                .is_some_and(|state| state.lit.handle().camera_jitter_ndc() == [0.0; 2]);
            self.prediction_real = source.filter(|_| clean).map(|source| (id, source));
        }
        result.map(|id| Some(id as f64)).map_err(|e| js_err(&e))
    }

    /// Select a completed image from the current engine timeline. The host composites
    /// the current HUD once after selection.
    #[wasm_bindgen(js_name = presentReady)]
    pub fn present_ready(&mut self, snapshot: JsValue) -> Result<JsValue, JsValue> {
        let target = present_target(&snapshot).map_err(|e| js_err(&e))?;
        let key = prediction_key(&snapshot, target);
        let selection = self.runtime()?.present_ready(target, key.as_ref()).map_err(|e| js_err(&e))?;
        let Some((id, predicted)) = selection else {
            return Ok(JsValue::UNDEFINED);
        };
        let result = js_sys::Object::new();
        js_sys::Reflect::set(&result, &"realFrameId".into(), &JsValue::from_f64(id as f64))?;
        js_sys::Reflect::set(&result, &"predicted".into(), &JsValue::from_bool(predicted))?;
        Ok(result.into())
    }

    /// Renders one frame and returns CPU timing keyed by the caller's trial IDs and accepted input.
    /// A rendering failure returns a partial record with its error for the same IDs.
    #[wasm_bindgen(js_name = renderInstrumented)]
    pub fn render_instrumented(
        &mut self,
        now_ms: f64,
        context: JsValue,
    ) -> Result<JsValue, JsValue> {
        let runtime = self.runtime.as_ref().ok_or_else(disposed)?;
        let view_id = frame_trace_id(&context, "viewId").map_err(|e| js_err(&e))?;
        let trial_id = frame_trace_id(&context, "trialId").map_err(|e| js_err(&e))?;
        let present_id = frame_trace_id(&context, "presentId").map_err(|e| js_err(&e))?;
        let input_cutoff_ms =
            js::required_f64(&context, "inputCutoffMs").map_err(|e| js_err(&e))?;
        let event_id = js::optional_string(&context, "eventId").map_err(|e| js_err(&e))?;
        let event_ms = js::optional_f64(&context, "eventMs").map_err(|e| js_err(&e))?;
        let accepted_ms = js::optional_f64(&context, "acceptedMs").map_err(|e| js_err(&e))?;
        check_frame_trace_event(event_id.as_deref(), event_ms, accepted_ms, input_cutoff_ms)
            .map_err(|e| js_err(&e))?;
        if !now_ms.is_finite() {
            return Err(js_err(&invalid("'nowMs' must be finite".into())));
        }
        let mut timing = runtime.start_frame_timing();
        if !(0.0..=timing.render_start_ms).contains(&input_cutoff_ms) {
            return Err(js_err(&invalid(
                "'inputCutoffMs' must be a timestamp from performance.now() at or before render start"
                    .into(),
            )));
        }

        let result = self.render_frame(now_ms, Some(&mut timing));
        let trace = Object::new();
        set(&trace, "viewId", &JsValue::from_str(&view_id));
        set(&trace, "trialId", &JsValue::from_str(&trial_id));
        set(&trace, "presentId", &JsValue::from_str(&present_id));
        set(
            &trace,
            "eventId",
            &event_id.as_deref().map_or(JsValue::NULL, JsValue::from_str),
        );
        for (key, value) in [
            ("nowMs", Some(now_ms)),
            ("inputCutoffMs", Some(input_cutoff_ms)),
            ("eventMs", event_ms),
            ("acceptedMs", accepted_ms),
            ("timeOriginMs", Some(timing.time_origin_ms)),
            ("renderStartMs", Some(timing.render_start_ms)),
            ("acquireStartMs", timing.acquire_start_ms),
            ("acquireEndMs", timing.acquire_end_ms),
            ("submitStartMs", timing.submit_start_ms),
            ("submitEndMs", timing.submit_end_ms),
            ("frameIndex", timing.frame_index.map(|index| index as f64)),
            ("presentSelectedMs", timing.present_selected_ms),
            ("presentReturnedMs", timing.present_returned_ms),
        ] {
            set(&trace, key, &value.map_or(JsValue::NULL, JsValue::from_f64));
        }
        let error = result.err().map_or(JsValue::NULL, |error| {
            let detail = Object::new();
            set(&detail, "code", &JsValue::from_str(error.code()));
            set(&detail, "message", &JsValue::from_str(error.message()));
            detail.into()
        });
        set(&trace, "error", &error);
        Ok(trace.into())
    }

    /// Frame statistics, the adapter and the hardware flag.
    pub fn stats(&self) -> Result<JsValue, JsValue> {
        let runtime = self.runtime.as_ref().ok_or_else(disposed)?;
        Ok(stats_to_js(&runtime.stats(), runtime.adapter()))
    }

    /// Renders one frame and tells whether any pixel differs from the clear colour.
    #[wasm_bindgen(js_name = probeNonEmpty)]
    pub async fn probe_non_empty(&mut self, now_ms: f64) -> Result<bool, JsValue> {
        self.begin_scene_invocation().map_err(|e| js_err(&e))?;
        let result = match self.runtime() {
            Ok(rt) => rt.probe_non_empty(now_ms).await.map_err(|e| js_err(&e)),
            Err(e) => Err(e),
        };
        self.finish_scene_invocation(result.is_ok());
        result
    }

    /// Renders one single-view stratified area sample and returns tight, native-format planes.
    #[wasm_bindgen(js_name = captureAreaSample)]
    pub async fn capture_area_sample(
        &mut self,
        now_ms: f64,
        seed: u32,
        guides: bool,
    ) -> Result<JsValue, JsValue> {
        if seed >= 0x8000_0000 || !now_ms.is_finite() {
            return Err(js_err(&invalid("invalid area capture time or seed".into())));
        }
        let shared = self.with_state(|gpu, s| {
            if !gpu.summary().is_hardware() {
                return Err(RaaError::Unsupported(
                    "area capture requires hardware WebGPU".into(),
                ));
            }
            s.features
                .get::<raa_gi::GiHandle>()
                .map(|gi| gi.shared())
                .ok_or_else(|| invalid("area capture requires GI".into()))
        })?;
        shared.set_area_capture_seed(Some(seed));
        let rendered = self.render_frame(now_ms, None);
        shared.set_area_capture_seed(None);
        rendered.map_err(|e| js_err(&e))?;

        // The legacy trace owns these textures. Temporal and spatial passes only read the raw
        // trace and guides, and no subsequent render can start while this call owns the engine.
        let (size, light, generation, textures) = self.with_state(|_, s| {
            let pipeline = s.pipeline.try_borrow().map_err(|_| pipeline_busy())?;
            let submission = pipeline.scene_submission();
            if submission.record_failed
                || submission.submitted_mask != 1
                || submission.pending_mask != 0
            {
                return Err(invalid(
                    "area capture requires one submitted primary view".into(),
                ));
            }
            let frame = s.lit.handle().area_stochastic();
            let light = u32::from_le_bytes(frame.table[..4].try_into().expect("slot word"));
            if frame.active != 1
                || light == u32::MAX
                || pipeline
                    .resources()
                    .get("gi.area-shadow", "restirAtlas")
                    .is_some()
            {
                return Err(invalid(
                    "area capture requires one stratified light in slot zero".into(),
                ));
            }
            let tap = |port| {
                pipeline
                    .resources()
                    .get("gi.area-shadow", port)
                    .ok_or_else(|| invalid(format!("missing area capture tap {port}")))
            };
            let raw = tap("raw")?;
            let mut textures = vec![raw.view.texture().clone()];
            if guides {
                let depth = tap("depthHistory")?;
                let targets = pipeline
                    .targets()
                    .ok_or_else(|| invalid("missing capture guides".into()))?;
                textures.extend([
                    depth.view.texture().clone(),
                    targets.normal_texture().clone(),
                    targets.motion_texture().clone(),
                ]);
            }
            let size = [raw.desc.width, raw.desc.height];
            if textures
                .iter()
                .any(|texture| [texture.width(), texture.height()] != size)
            {
                return Err(invalid("area capture guide extents differ".into()));
            }
            Ok((size, light, frame.generations[0], textures))
        })?;
        let runtime = self.runtime.as_ref().ok_or_else(disposed)?;
        let planes = Array::new();
        for texture in textures {
            let bytes = raa_web::read_capture_texture(runtime.gpu(), &texture)
                .await
                .map_err(|e| js_err(&e))?;
            planes.push(&js_sys::Uint8Array::from(bytes.as_slice()));
        }
        let output = Object::new();
        set(&output, "size", &number_array(&size));
        set(&output, "lightIndex", &JsValue::from(light));
        set(&output, "lightGeneration", &JsValue::from(generation));
        set(&output, "planes", &planes);
        Ok(output.into())
    }

    /// Captures one specular sample from the short or traced hybrid path estimator.
    #[wasm_bindgen(js_name = captureMeshPathSample)]
    pub async fn capture_mesh_path_sample(
        &mut self,
        now_ms: f64,
        seed: u32,
    ) -> Result<JsValue, JsValue> {
        use raa_sdf_trace::mesh::{MeshTraceHandle, PathMode};

        if !now_ms.is_finite() {
            return Err(js_err(&invalid("invalid path capture time".into())));
        }
        let handle = self.with_state(|gpu, st| {
            if !gpu.summary().is_hardware() {
                return Err(RaaError::Unsupported(
                    "path capture requires hardware WebGPU".into(),
                ));
            }
            let handle = st
                .features
                .get::<MeshTraceHandle>()
                .cloned()
                .ok_or_else(|| invalid("path capture requires mesh tracing".into()))?;
            let settings = handle.borrow().settings;
            if settings.reference
                || !matches!(settings.path_mode, PathMode::Short | PathMode::Traced)
            {
                return Err(invalid(
                    "path capture requires short or traced hybrid mode".into(),
                ));
            }
            handle.borrow_mut().begin_path_capture(seed);
            Ok(handle)
        })?;
        let rendered = self.render_frame(now_ms, None);
        let result = handle.borrow_mut().end_path_capture();
        rendered.map_err(|e| js_err(&e))?;
        self.with_state(|_, st| {
            let pipeline = st.pipeline.try_borrow().map_err(|_| pipeline_busy())?;
            let submission = pipeline.scene_submission();
            if submission.record_failed
                || submission.submitted_mask != 1
                || submission.pending_mask != 0
            {
                return Err(invalid(
                    "path capture requires one submitted primary view".into(),
                ));
            }
            Ok(())
        })?;
        let ((width, height), buffer) =
            result.ok_or_else(|| js_err(&invalid("missing path capture output".into())))?;
        let bytes = raa_web::read_capture_buffer(&buffer)
            .await
            .map_err(|e| js_err(&e))?;
        let output = Object::new();
        set(&output, "size", &number_array(&[width, height]));
        set(
            &output,
            "data",
            &js_sys::Uint8Array::from(bytes.as_slice()).into(),
        );
        Ok(output.into())
    }

    /// Reads linear RGBA16F. An isolated forward view also returns its same-frame
    /// combined lighting as `combinedData`.
    #[wasm_bindgen(js_name = captureMeshLighting)]
    pub async fn capture_mesh_lighting(
        &mut self,
        now_ms: f64,
        traced: bool,
        raw: Option<bool>,
    ) -> Result<JsValue, JsValue> {
        if !now_ms.is_finite() {
            return Err(js_err(&invalid(
                "invalid mesh lighting capture time".into(),
            )));
        }
        self.render_frame(now_ms, None)
            .map_err(|error| js_err(&error))?;
        let (texture, combined) = self.with_state(|gpu, st| {
            if !gpu.summary().is_hardware() {
                return Err(RaaError::Unsupported(
                    "mesh lighting capture requires hardware WebGPU".into(),
                ));
            }
            let pipeline = st.pipeline.try_borrow().map_err(|_| pipeline_busy())?;
            let submission = pipeline.scene_submission();
            if submission.record_failed
                || submission.submitted_mask != 1
                || submission.pending_mask != 0
            {
                return Err(invalid(
                    "mesh lighting capture requires one submitted primary view".into(),
                ));
            }
            let (producer, port) = match (traced, raw.unwrap_or(false)) {
                (true, true) => ("trace.mesh", "radiance"),
                (true, false) => ("trace.mesh", "color"),
                (false, false) => ("forward", "color"),
                (false, true) => {
                    return Err(invalid("raw mesh lighting capture requires tracing".into()));
                }
            };
            let resources = pipeline.resources();
            let texture = resources.get(producer, port).ok_or_else(|| {
                invalid(format!("missing mesh lighting capture tap {producer}.{port}"))
            })?;
            let combined = if traced {
                None
            } else {
                resources
                    .get("forward", "combined-lighting")
                    .map(|image| image.view.texture().clone())
            };
            Ok((texture.view.texture().clone(), combined))
        })?;
        let runtime = self.runtime.as_ref().ok_or_else(disposed)?;
        let bytes = raa_web::read_capture_texture(runtime.gpu(), &texture)
            .await
            .map_err(|error| js_err(&error))?;
        let output = Object::new();
        set(
            &output,
            "size",
            &number_array(&[texture.width(), texture.height()]),
        );
        set(
            &output,
            "data",
            &js_sys::Uint8Array::from(bytes.as_slice()).into(),
        );
        if let Some(combined) = combined {
            let bytes = raa_web::read_capture_texture(runtime.gpu(), &combined)
                .await
                .map_err(|error| js_err(&error))?;
            set(
                &output,
                "combinedData",
                &js_sys::Uint8Array::from(bytes.as_slice()).into(),
            );
        }
        Ok(output.into())
    }

    /// Selects camera jitter independently of TAA. Null uses the default TAA-driven policy.
    #[wasm_bindgen(js_name = setCameraJitter)]
    pub fn set_camera_jitter(&mut self, enabled: Option<bool>) -> Result<(), JsValue> {
        self.with_state(|_, s| {
            s.lit.set_camera_jitter(enabled);
            Ok(())
        })
    }

    /// Releases the GPU objects and the scene; every later call throws `api.disposed`.
    pub fn dispose(&mut self) {
        self.prediction_real = None;
        self.prediction_warp = None;
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
            let before = s.scene.camera;
            let cam = &mut s.scene.camera;
            cam.eye = eye;
            cam.target = target;
            cam.up = Vec3::from_array(d.up);
            cam.fov_y_rad = d.fov_y_deg.to_radians();
            cam.ortho_height = d.ortho_height;
            cam.near = d.near;
            cam.far = d.far;
            // Only a jump is a cut: a camera that moves a step per frame keeps its history
            // (clouds, TAA, GI, motion vectors), which reprojects through the previous view.
            if s.scene.camera.is_cut_from(&before) {
                s.scene_invocation.discontinuity(0);
            }
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
            let before = s.scene.camera;
            let mut orbit = OrbitController::new(target, distance);
            orbit.rotate(yaw, pitch);
            s.orbit = orbit;
            s.orbit.apply(&mut s.scene.camera);
            if s.scene.camera.is_cut_from(&before) {
                s.scene_invocation.discontinuity(0);
            }
            Ok(())
        })
    }

    /// Sets the directional light: `direction`, linear `color`, `intensity`.
    #[wasm_bindgen(js_name = setLight)]
    pub fn set_light(&mut self, desc: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, s| {
            let light = js::light_desc(&desc)?;
            if let Some(sky) = s.features.get::<raa_sky::SkyHandle>() {
                sky.set_manual_light(light)?;
            }
            s.scene.light = light;
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

    /// Sets the local transform of `node`; local lights attached to the node's instance
    /// groups (and its descendants') are validated before the scene changes and follow the
    /// new world transforms in the same frame (`light.invalid_attachment` refuses atomically).
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
            st.set_node_local(id, local)
        })
    }

    /// Sets many local transforms: `trs` holds 10 floats per node (t3, r4 [x, y, z, w],
    /// s3). A length mismatch throws `api.length_mismatch`. Nodes before a failing entry
    /// keep their new transform; each entry is validated (attached local lights included)
    /// before it applies, so a failing entry changes neither scene nor lights.
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
            for i in 0..n {
                let h = st.scratch_nodes[i];
                let start = i * 10;
                let id = st.nodes.get(h)?;
                if !st.scratch_trs[start..start + 10]
                    .iter()
                    .all(|c| c.is_finite())
                {
                    return Err(invalid(format!(
                        "'trs' of node {i} must hold finite numbers"
                    )));
                }
                let local = decode_trs(&st.scratch_trs, i);
                st.set_node_local(id, local)?;
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

    /// Sets the opaque planar mirror receiver and its oriented world-space plane.
    /// A null receiver clears the capture.
    #[wasm_bindgen(js_name = setPlanarReflection)]
    pub fn set_planar_reflection(
        &mut self,
        instance: Option<u32>,
        plane: JsValue,
    ) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            let reflection = if let Some(instance) = instance {
                if st.instances.get_ref(instance)?.stale {
                    return Err(api_error(
                        "scene.stale_instance",
                        "the mirror receiver's node was removed",
                    ));
                }
                let plane = js::floats::<4>("plane", &plane)?;
                Some(raa_lit::PlanarReflection::new(instance, plane)?)
            } else {
                None
            };
            st.lit.set_planar_reflection(reflection);
            Ok(())
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
