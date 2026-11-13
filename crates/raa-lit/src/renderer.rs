//! [`LitRenderer`]: the owner of the lit renderer's state. It uploads meshes, holds the
//! surface materials (default material id 0, grey standard), the material functions and
//! slot providers, the class handlers, the instance extras by tag and the forward
//! contributors, and prepares each frame: mesh bounds into the scene, scene world update,
//! the tagged and slotted render list, index-based camera selection, the persistent
//! instance records and the id stream ([`IdPrep`]), `FrameGpu` and the queue writes. Every
//! frame buffer is sized at [`LitRenderer::new`] from [`LitOptions`], so a warm frame
//! allocates nothing.
//!
//! The renderer's state is shared with passes and contributors through [`LitHandle`]
//! (`Rc<RefCell<..>>`), so the accessors into it return `Ref`/`RefMut` guards.

use std::cell::{Cell, Ref, RefCell, RefMut};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use raa_core::{GpuContext, MaterialId, MeshId, RaaError};
use raa_math::{Aabb, Mat4, Vec3};
use raa_mesh::AttributedMesh;
use raa_pipeline::{Pipeline, SubmissionState};
use raa_scene::{Camera, InstanceIndex, RenderList, Scene, frustum_planes};
use raa_shader::{
    Composer, LIGHT_GPU_SIZE, MAX_LOCAL_LIGHTS, MaterialFnId, MaterialFnRegistry, SlotProvider,
};

use crate::contributor::ForwardContributor;
use crate::extras::InstanceExtra;
use crate::frame::{FRAME_GPU_SIZE, FrameParams, write_frame_gpu};
use crate::frame_history::{CameraHistory, FrameHistory, MAX_VIEWS, frame_jitter};
use crate::handle::{LitHandle, LitShared};
use crate::instance_store::InstanceStore;
use crate::library::SurfaceLibrary;
use crate::light_stats::{LocalLightStats, ReadbackCoordinator};
use crate::local_lights::LocalLights;
use crate::lod::{LodLevel, LodParams, LodRegistry, MAX_LOD_LEVELS, camera_level};
use crate::material::{DrawClass, SurfaceMaterial};
use crate::material_data::MaterialDataStats;
use crate::meshes::{LitMeshId, LitMeshInfo};
use crate::passes::install_passes;
use crate::prep::MaterialClass;
use crate::select::SelectInput;
use crate::skin_views::SkinDraw;
use crate::textures::{AtlasRect, TexelData};
use crate::tiles::{
    LIGHT_TILES_NAME, TILING_FEATURE_ID, TileInputs, light_info, publish_local_lights,
};
use crate::virtual_set::{VirtualSet, slot_member};

/// Limits and sample count of a [`LitRenderer`]; the frame buffers are sized from them.
#[derive(Copy, Clone, Debug)]
pub struct LitOptions {
    /// Instances per frame (instance stream and extras array).
    pub max_instances: u32,
    /// Materials of the surface library (material array).
    pub max_materials: u32,
    /// MSAA samples of the forward pass: 1 or 4.
    pub msaa: u32,
}

impl Default for LitOptions {
    fn default() -> Self {
        LitOptions {
            max_instances: 16384,
            max_materials: 256,
            msaa: 4,
        }
    }
}

impl LitOptions {
    /// Refuses `render.invalid_msaa` unless `msaa` is 1 or 4.
    pub(crate) fn validate(&self) -> Result<(), RaaError> {
        if self.msaa == 1 || self.msaa == 4 {
            Ok(())
        } else {
            Err(refusal(
                "render.invalid_msaa",
                format!("msaa must be 1 or 4 samples, got {}", self.msaa),
            ))
        }
    }
}

/// What one [`LitRenderer::prepare`] submitted.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameCounts {
    /// Instances written to the instance stream.
    pub instances: u32,
    /// Instanced draw calls of every class.
    pub draw_calls: u32,
    /// Triangles drawn (sub-mesh triangles x instances).
    pub triangles: u64,
}

/// Instance selection and persistent-record uploads of the last prepared frame.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct InstanceFrameStats {
    pub instances: u32,
    pub camera: u32,
    pub cascades: [u32; 4],
    /// Camera instances per level, including full resolution at index 0.
    pub lods: [u32; MAX_LOD_LEVELS],
    pub uploaded_bytes: u64,
}

pub(crate) const NO_CAMERA_LEVEL: u8 = u8::MAX;

/// No temporal-accumulation consumer exists in this milestone (S10), so every viewport
/// reports zero current and previous jitter.
const JITTER_CONSUMER: bool = false;

struct CameraLodInput<'a> {
    list: &'a RenderList,
    tags: &'a [u32],
    slots: &'a [u32],
    index: &'a InstanceIndex,
    camera: &'a Camera,
    viewport_height: u32,
    lods: &'a LodRegistry,
    bias: &'a HashMap<u32, i8>,
    options: &'a LodParams,
    skinned: &'a dyn Fn(u32) -> Option<SkinDraw>,
}

/// Compute before IdPrep so its level lookup can remain an immutable Fn.
fn camera_levels(input: &CameraLodInput<'_>, levels: &mut [u8]) {
    let focal_px = input.viewport_height as f32 / (2.0 * (input.camera.fov_y_rad * 0.5).tan());
    for ((item, &tag), &slot) in input.list.items.iter().zip(input.tags).zip(input.slots) {
        let level = &mut levels[slot as usize];
        // Posed draws bind the full mesh's rest skin records. Drop hysteresis while posed.
        if (input.skinned)(tag).is_some() {
            *level = NO_CAMERA_LEVEL;
            continue;
        }
        let Some(chain) = input.lods.chain(item.mesh) else {
            *level = NO_CAMERA_LEVEL;
            continue;
        };
        let world = &item.world;
        let scales: [f32; 3] = std::array::from_fn(|i| {
            let c = world.cols[i];
            Vec3::new(c[0], c[1], c[2]).length()
        });
        let world_scale = if scales.iter().all(|scale| scale.is_finite()) {
            scales.into_iter().fold(0.0, f32::max)
        } else {
            f32::NAN
        };
        let distance = match input.index.bounds(slot).flatten() {
            Some(bounds) => (input.camera.eye - bounds.center()).length() - bounds.diagonal() * 0.5,
            None => (input.camera.eye - world.transform_point(Vec3::ZERO)).length(),
        };
        let params = LodParams {
            bias: input.options.bias + f32::from(input.bias.get(&tag).copied().unwrap_or(0)),
            ..*input.options
        };
        let prev = (*level != NO_CAMERA_LEVEL).then_some(*level as usize);
        *level = camera_level(chain, world_scale, distance, focal_px, prev, &params) as u8;
    }
}

fn camera_mesh(lods: &LodRegistry, levels: &[u8], slot: u32, mesh: MeshId) -> MeshId {
    lods.chain(mesh)
        .and_then(|chain| chain.get(levels[slot as usize] as usize))
        .map_or(mesh, |level| level.mesh)
}

/// Whether `full` has a detail chain.
fn has_chain(lods: &LodRegistry, full: MeshId) -> bool {
    lods.chain(full).is_some()
}

/// Removes the chain of `full`, returning whether it existed.
fn clear_chain(lods: &mut LodRegistry, full: MeshId) -> bool {
    let had = has_chain(lods, full);
    lods.clear(full);
    had
}

/// Layout is (sub-mesh count, sub-mesh ordinal); index counts may change at each level.
fn register_mesh_lods(
    lods: &mut LodRegistry,
    full: MeshId,
    levels: &[LodLevel],
    layout: &dyn Fn(MeshId) -> Option<(usize, usize)>,
) -> Result<(), RaaError> {
    let full_layout = layout(full)
        .ok_or_else(|| refusal("render.invalid_lod", "Full mesh is not uploaded".into()))?;
    if levels
        .iter()
        .any(|level| layout(level.mesh) != Some(full_layout))
    {
        return Err(refusal(
            "render.invalid_lod",
            "Detail meshes must be uploaded with matching sub-mesh layouts".into(),
        ));
    }
    lods.set(full, levels)
}

/// One forward contributor with its pipeline entry flag (the forward pass records it only
/// while `enabled` is set) and its order.
pub(crate) struct ContributorSlot {
    pub(crate) order: u32,
    pub(crate) enabled: Rc<Cell<bool>>,
    pub(crate) contributor: Box<dyn ForwardContributor>,
}

/// The contributors in recording order (ascending order, then registration order), shared
/// with the forward pass.
pub(crate) type Contributors = Rc<RefCell<Vec<ContributorSlot>>>;

/// Index at which a contributor of `order` is inserted: after every slot of an order lower
/// than or equal to it.
fn insert_position(slots: &[ContributorSlot], order: u32) -> usize {
    slots.partition_point(|s| s.order <= order)
}

fn refusal(code: &'static str, message: String) -> RaaError {
    RaaError::FrameGraph { code, message }
}

pub(crate) fn pipeline_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the pipeline is borrowed (called while a frame executes)".to_string(),
    )
}

/// One scene invocation's four reused viewport histories (RAA-FRAME) and the scene-instance
/// slots each viewport staged for previous-instance history.
///
/// Handoff: `LitRenderer::begin_scene_frame(&mut self, index: u64) -> ()`,
/// `prepare_scene_view(&mut self, view: usize, discontinuity: bool) -> ()`,
/// `submitted_scene_view(&mut self, view: usize) -> ()` and `finish_scene_frame(&mut self,
/// complete: bool) -> ()` in `crates/raa-lit/src/renderer.rs` are called by the Engine
/// around one render invocation; camera state is keyed by viewport 0..3 (`FrameHistory`),
/// previous-instance state by the existing scene-instance slot (`InstanceId::index()`, one
/// per submesh) through `InstanceStore::previous_submitted`/`finish_previous_frame`.
struct SceneLifecycle {
    history: FrameHistory,
    frame_index: u64,
    open: bool,
    selected: Option<(usize, bool)>,
    current_view: usize,
    slots: [Vec<u32>; MAX_VIEWS],
    acknowledged: u8,
    commit_pending: bool,
}

impl SceneLifecycle {
    fn new() -> Self {
        SceneLifecycle {
            history: FrameHistory::new(),
            frame_index: 0,
            open: false,
            selected: None,
            current_view: 0,
            slots: std::array::from_fn(|_| Vec::new()),
            acknowledged: 0,
            commit_pending: false,
        }
    }

    fn begin(&mut self, index: u64) {
        if self.open {
            self.history.invalidate();
            self.commit_pending = true;
        }
        self.history.begin(index);
        self.frame_index = index;
        self.open = true;
        self.selected = None;
    }

    fn prepare(&mut self, scene: &Scene, size: (u32, u32), consumer: bool) -> CameraHistory {
        if !self.open {
            let index = self.frame_index;
            self.begin(index);
        }
        let (view, discontinuity) = self.selected.take().unwrap_or((0, false));
        self.current_view = view;
        let jitter = frame_jitter(self.frame_index, size, consumer);
        let (proj, view_matrix) = (scene.camera.projection(), scene.camera.view());
        let unjittered_view_proj = proj * view_matrix;
        let view_proj = if jitter == [0.0; 2] {
            unjittered_view_proj
        } else {
            let mut offset = Mat4::IDENTITY;
            offset.cols[3] = [jitter[0], jitter[1], 0.0, 1.0];
            (offset * proj) * view_matrix
        };
        self.history.prepare(
            view,
            view_proj,
            unjittered_view_proj,
            size,
            consumer,
            discontinuity,
        )
    }

    fn submitted(&mut self, view: usize) {
        if !self.open {
            return;
        }
        let view = view % MAX_VIEWS;
        self.history.submitted(view);
        self.acknowledged |= 1 << view;
    }

    fn finish(&mut self, complete: bool) {
        if !self.open {
            return;
        }
        self.history.finish(complete);
        self.open = false;
        self.selected = None;
        self.commit_pending = true;
        self.frame_index = self.frame_index.saturating_add(1);
    }

    fn flush(&mut self, store: &mut InstanceStore) {
        for view in 0..MAX_VIEWS {
            if self.acknowledged & (1 << view) != 0 {
                for &slot in &self.slots[view] {
                    store.previous_submitted(slot);
                }
            }
        }
        self.acknowledged = 0;
        if self.commit_pending {
            store.finish_previous_frame();
            self.commit_pending = false;
        }
        for slots in &mut self.slots {
            slots.clear();
        }
    }
}

/// The HDR lit renderer; see the module docs.
pub struct LitRenderer {
    opts: LitOptions,
    shared: Rc<RefCell<LitShared>>,
    pipeline: Rc<RefCell<Pipeline>>,
    /// Pass name per handled class (Opaque is always handled).
    class_handlers: HashMap<DrawClass, String>,
    /// Instance extras by scene instance tag.
    extras: HashMap<u32, InstanceExtra>,
    contributors: Contributors,
    /// Class and function per material id, rebuilt when the library changed; capacity
    /// `max_materials`.
    material_classes: Vec<MaterialClass>,
    /// The frame's `FrameGpu` bytes.
    frame_bytes: [u8; FRAME_GPU_SIZE],
    /// Per-viewport camera history and the open scene invocation (RAA-FRAME).
    scene_frame: SceneLifecycle,
    /// The renderer-owned S17 local-light registry (M2-LOCAL-STATE).
    ///
    /// Handoff: `LitRenderer::local_lights_mut(&mut self) -> &mut LocalLights` in
    /// `crates/raa-lit/src/renderer.rs` (with the read-only `local_lights`) exposes this
    /// registry. The Engine attaches it by carrying the `LitRenderer` in
    /// `crates/raa-js/src/state.rs` and calling the accessors from its object mutators:
    /// `LocalLights::validate_parent(parent: u32, world: &Mat4)` before
    /// `Scene::set_local`, then `update_parent(parent: u32, world: &Mat4)` or
    /// `remove_parent(parent: u32)`. The state is keyed by the stable `LightHandle`
    /// (`u64`) and the parent is the existing JS instance-group handle (`u32`), never a
    /// compacted submesh slot.
    local_lights: LocalLights,
    /// The order-35 tiling pass's shared inputs: the per-viewport `FrameGpu` snapshot
    /// (`light_info` = active count, tile columns, rows and `TILE_WORDS`), the dense
    /// `LocalLights` records and the invocation/view identity of M2-FRAME-LIFECYCLE.
    /// `install_local_lights` attaches the pass to this same `Rc`.
    tile_inputs: Rc<RefCell<TileInputs>>,
    /// The bounded asynchronous overflow pool shared through `TileInputs::readback`.
    readback: Arc<Mutex<ReadbackCoordinator>>,
    /// The last closed invocation's submission observation, resolved (with the GPU it
    /// needs) at the next `prepare`; readback never fences synchronously.
    pending_resolve: Option<(u64, SubmissionState, bool)>,
    /// A background that could not reach the pipeline (borrowed), applied at `prepare`.
    pending_background: Option<[f32; 3]>,
    /// Bounds last sent to the scene per uploaded sub-mesh; only changes are sent.
    sent_bounds: HashMap<MeshId, Aabb>,
    /// Sub-mesh ids of every uploaded lit mesh, refreshed on upload/update.
    submeshes: HashMap<LitMeshId, Vec<MeshId>>,
    /// Reusable frustum query result (instance slots) and its per-slot bitset.
    visible_slots: Vec<u32>,
    visible_bits: Vec<u64>,
    /// Virtualized scene instances, left out of the draw lists (P3-LIT-SEAM).
    virtual_set: VirtualSet,
    /// Logical `FrameGpu` feature ids set through [`LitRenderer::set_frame_feature`], one
    /// bit each; id 24 (tiling) and ids >= 64 are never stored.
    frame_features: u64,
}

impl LitRenderer {
    /// A renderer drawing through `pipeline`: validates `opts` (`render.invalid_msaa`),
    /// creates the group-0 buffers sized from it, installs the default slot providers
    /// (priority 0), adds the default material id 0 (grey standard) and registers the
    /// required feature passes `prepass` (order 30) and `forward` (order 60) in `pipeline`
    /// (`pipeline.duplicate_pass` when it already has them, `pipeline.busy`).
    pub fn new(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        opts: LitOptions,
    ) -> Result<Self, RaaError> {
        opts.validate()?;
        let mut shared = {
            let p = pipeline.try_borrow().map_err(|_| pipeline_busy())?;
            LitShared::new(
                gpu,
                p.lighting().layout(),
                opts.max_instances,
                opts.max_materials,
                opts.msaa,
            )?
        };
        shared.materials.add(SurfaceMaterial::default())?;
        // `ReadbackCoordinator::resolve` takes this exact shared type; on wasm32 its wgpu
        // handles are `JsValue`-backed and not `Send`, while the renderer is single-threaded.
        #[allow(clippy::arc_with_non_send_sync)]
        let readback = Arc::new(Mutex::new(ReadbackCoordinator::new()));
        let tile_inputs = Rc::new(RefCell::new(TileInputs {
            frame: [0; FRAME_GPU_SIZE],
            lights: [0; MAX_LOCAL_LIGHTS * LIGHT_GPU_SIZE],
            index: 0,
            view: 0,
            readback: Arc::clone(&readback),
            lights_dirty: false,
        }));
        let renderer = LitRenderer {
            opts,
            shared: Rc::new(RefCell::new(shared)),
            pipeline: Rc::clone(pipeline),
            class_handlers: HashMap::new(),
            extras: HashMap::new(),
            contributors: Rc::new(RefCell::new(Vec::new())),
            material_classes: Vec::with_capacity(opts.max_materials as usize),
            frame_bytes: [0; FRAME_GPU_SIZE],
            scene_frame: SceneLifecycle::new(),
            local_lights: LocalLights::new(),
            tile_inputs,
            readback,
            pending_resolve: None,
            pending_background: None,
            sent_bounds: HashMap::new(),
            submeshes: HashMap::new(),
            visible_slots: Vec::new(),
            visible_bits: Vec::new(),
            virtual_set: VirtualSet::new(),
            frame_features: 0,
        };
        install_passes(pipeline, renderer.handle(), renderer.contributors())?;
        Ok(renderer)
    }

    /// A shared view of the renderer's state for passes and contributors.
    pub fn handle(&self) -> LitHandle {
        LitHandle::new(Rc::clone(&self.shared))
    }

    /// The inputs the order-35 `light.tiles` pass reads, shared with the pass that
    /// `install_local_lights` registers.
    pub(crate) fn tile_inputs(&self) -> Rc<RefCell<TileInputs>> {
        Rc::clone(&self.tile_inputs)
    }

    /// The forward contributors in recording order (for the forward pass).
    pub(crate) fn contributors(&self) -> Contributors {
        Rc::clone(&self.contributors)
    }

    // ---- meshes ----

    /// Validates and uploads `m` (`render.invalid_mesh` carrying the `mesh.*` code); one
    /// internal [`MeshId`] is assigned per sub-mesh ([`submesh_ids`](Self::submesh_ids)).
    pub fn upload(&mut self, gpu: &GpuContext, m: &AttributedMesh) -> Result<LitMeshId, RaaError> {
        let id = self.shared.borrow_mut().meshes.upload(gpu, m)?;
        self.refresh_submeshes(id);
        Ok(id)
    }

    /// Replaces mesh `id` (`render.unknown_mesh`, `render.invalid_mesh`; a refused mesh
    /// changes nothing). Its sub-mesh ids are kept where the sub-mesh count allows.
    pub fn update(
        &mut self,
        gpu: &GpuContext,
        id: LitMeshId,
        m: &AttributedMesh,
    ) -> Result<(), RaaError> {
        self.shared.borrow_mut().meshes.update(gpu, id, m)?;
        self.refresh_submeshes(id);
        Ok(())
    }

    /// Frees mesh `id` and its sub-mesh ids (reused lowest first). Unknown ids are ignored.
    pub fn remove(&mut self, id: LitMeshId) {
        self.shared.borrow_mut().meshes.remove(id);
        self.submeshes.remove(&id);
    }

    /// Records the sub-mesh ids of `id` for the per-frame bounds sync.
    fn refresh_submeshes(&mut self, id: LitMeshId) {
        let ids = self
            .shared
            .borrow()
            .meshes
            .submesh_ids(id)
            .map(<[MeshId]>::to_vec);
        match ids {
            Some(ids) => {
                self.submeshes.insert(id, ids);
            }
            None => {
                self.submeshes.remove(&id);
            }
        }
    }

    /// The internal [`MeshId`] of each sub-mesh of `id`, in sub-mesh order.
    pub fn submesh_ids(&self, id: LitMeshId) -> Option<Ref<'_, [MeshId]>> {
        Ref::filter_map(self.shared.borrow(), |s| s.meshes.submesh_ids(id)).ok()
    }

    /// Counts, effective sub-meshes and bounds of mesh `id`.
    pub fn info(&self, id: LitMeshId) -> Option<LitMeshInfo> {
        self.shared.borrow().meshes.info(id)
    }

    /// Registers coarser uploaded sub-meshes, finest first, preserving instance materials.
    /// Refuses invalid chains or mismatched sub-mesh layouts with `render.invalid_lod`.
    /// Skinning is per instance tag and does not restrict registration.
    pub fn set_mesh_lods(&mut self, full: MeshId, levels: &[LodLevel]) -> Result<(), RaaError> {
        let mut shared = self.shared.borrow_mut();
        let s: &mut LitShared = &mut shared;
        let meshes = &s.meshes;
        register_mesh_lods(&mut s.lods, full, levels, &|mesh| {
            let submesh = meshes.submesh(mesh)?;
            let ids = meshes.submesh_ids(submesh.lit)?;
            Some((ids.len(), ids.iter().position(|&id| id == mesh)?))
        })?;
        s.levels.fill(NO_CAMERA_LEVEL);
        s.shadow_levels.fill(NO_CAMERA_LEVEL);
        Ok(())
    }

    /// Whether `full` has a detail chain.
    pub fn has_mesh_lods(&self, full: MeshId) -> bool {
        has_chain(&self.shared.borrow().lods, full)
    }

    /// Removes a detail chain and returns whether `full` had one; an unknown id is ignored
    /// and returns false.
    pub fn clear_mesh_lods(&mut self, full: MeshId) -> bool {
        let mut s = self.shared.borrow_mut();
        let had = clear_chain(&mut s.lods, full);
        s.levels.fill(NO_CAMERA_LEVEL);
        s.shadow_levels.fill(NO_CAMERA_LEVEL);
        had
    }

    /// Sets camera selection parameters; per-tag bias is added to `options.bias`.
    pub fn set_lod_options(&mut self, options: LodParams) {
        self.shared.borrow_mut().lod_options = options;
    }

    pub fn lod_options(&self) -> LodParams {
        self.shared.borrow().lod_options
    }

    /// Cascade caster selection parameters; `threshold_px` is in shadow-map texels and the
    /// per-tag bias is added to `options.bias`.
    pub fn set_shadow_lod_options(&mut self, options: LodParams) {
        self.shared.borrow_mut().shadow_lod_options = options;
    }

    pub fn shadow_lod_options(&self) -> LodParams {
        self.shared.borrow().shadow_lod_options
    }

    /// Positive bias allows coarser geometry sooner. Zero removes the tag override.
    pub fn set_instance_lod_bias(&mut self, tag: u32, bias: i8) {
        let mut s = self.shared.borrow_mut();
        if bias == 0 {
            s.lod_bias.remove(&tag);
        } else {
            s.lod_bias.insert(tag, bias);
        }
    }

    pub fn frame_stats(&self) -> InstanceFrameStats {
        self.shared.borrow().stats
    }

    // ---- materials, functions, slots ----

    /// The surface library (id 0 is the default grey standard material).
    pub fn materials(&self) -> Ref<'_, SurfaceLibrary> {
        Ref::map(self.shared.borrow(), |s| &s.materials)
    }

    /// The surface library, to add or set materials (uploaded at the next prepare).
    pub fn materials_mut(&mut self) -> RefMut<'_, SurfaceLibrary> {
        RefMut::map(self.shared.borrow_mut(), |s| &mut s.materials)
    }

    /// The material function registry (`raa.standard` is id 0).
    pub fn fns(&self) -> Ref<'_, MaterialFnRegistry> {
        Ref::map(self.shared.borrow(), |s| &s.fns)
    }

    /// The material function registry, to register built-in functions.
    pub fn fns_mut(&mut self) -> RefMut<'_, MaterialFnRegistry> {
        RefMut::map(self.shared.borrow_mut(), |s| &mut s.fns)
    }

    /// Registers user WGSL as a material function, checked with the current slot providers
    /// (S11 rules; refusals keep their `material.fn_*` / `shader.*` code).
    pub fn register_user_fn(&mut self, name: &str, wgsl: &str) -> Result<MaterialFnId, RaaError> {
        self.shared.borrow_mut().register_user_fn(name, wgsl)
    }

    /// The shader composer with its slot providers.
    pub fn composer(&self) -> Ref<'_, Composer> {
        Ref::map(self.shared.borrow(), |s| s.composer())
    }

    /// Installs a slot provider (`shader.slot_conflict` on an equal priority) and clears
    /// the pipeline cache.
    pub fn add_slot(&mut self, p: SlotProvider) -> Result<(), RaaError> {
        self.shared.borrow_mut().add_slot(p)
    }

    // ---- classes, extras, contributors, background ----

    /// Declares that pass `pass_name` draws `class` (materials of an unhandled class are
    /// refused at prepare with `material.blend_unavailable`).
    pub fn register_class_handler(&mut self, class: DrawClass, pass_name: &str) {
        self.class_handlers.insert(class, pass_name.to_string());
    }

    /// Whether `class` has a handler; Opaque always has one (the forward pass).
    pub fn class_handled(&self, class: DrawClass) -> bool {
        class == DrawClass::Opaque || self.class_handlers.contains_key(&class)
    }

    /// Sets the extras of every instance with scene tag `tag`.
    pub fn set_instance_extra(&mut self, tag: u32, x: InstanceExtra) {
        self.extras.insert(tag, x);
    }

    /// The extras of tag `tag` (S10 defaults when none were set).
    pub fn instance_extra(&self, tag: u32) -> InstanceExtra {
        self.extras.get(&tag).copied().unwrap_or_default()
    }

    /// Returns tag `tag` to the default extras.
    pub fn clear_instance_extra(&mut self, tag: u32) {
        self.extras.remove(&tag);
    }

    /// Adds a `width` x `height` material texture (rows top first, rgba) to the atlas and
    /// returns its id, sampled in WGSL by `raa_texture(id, uv)`. Ids start at 1; a removed
    /// id is given to a later texture. Float data is clamped to [0, 1] and quantized to 8
    /// bits. Refusals (nothing changes): `texture.invalid` (a side of 0 or above
    /// [`ATLAS_SIZE`](crate::ATLAS_SIZE), a data length other than width * height * 4),
    /// `texture.atlas_full` (no room or no free id), `pipeline.busy`.
    pub fn add_texture(
        &mut self,
        gpu: &GpuContext,
        width: u32,
        height: u32,
        data: TexelData<'_>,
    ) -> Result<u32, RaaError> {
        let mut s = self.shared.try_borrow_mut().map_err(|_| pipeline_busy())?;
        s.add_texture(gpu, width, height, data)
    }

    /// Removes texture `id`; its id then samples (0, 0, 0, 0). Idempotent: an unknown or
    /// removed id does nothing. Returns whether it was live.
    pub fn remove_texture(&mut self, gpu: &GpuContext, id: u32) -> Result<bool, RaaError> {
        let mut s = self.shared.try_borrow_mut().map_err(|_| pipeline_busy())?;
        Ok(s.remove_texture(gpu, id))
    }

    /// The atlas rect of live texture `id`.
    pub fn texture_rect(&self, id: u32) -> Option<AtlasRect> {
        self.shared.borrow().texture_packer().rect(id)
    }

    /// Adds a block of material data words (group 0 binding 6) and returns its base word
    /// offset, the value a surface material carries as [`SurfaceMaterial::data`] and a
    /// material function reads with `raa_material_word(raa_material_data_base(in.record) + i)`.
    /// Blocks start after the reserved zero words; a freed range is reused. Refusals
    /// (nothing changes): `material.data_invalid` (no word), `material.data_full` (the buffer
    /// would pass the device's storage binding size), `pipeline.busy`.
    ///
    /// [`SurfaceMaterial::data`]: crate::SurfaceMaterial::data
    pub fn add_material_data(&mut self, gpu: &GpuContext, words: &[u32]) -> Result<u32, RaaError> {
        let mut s = self.shared.try_borrow_mut().map_err(|_| pipeline_busy())?;
        s.add_material_data(gpu, words)
    }

    /// Overwrites the live block at `base` with as many words as it holds. Refusals
    /// (nothing changes): `material.data_invalid` (unknown base, other length),
    /// `pipeline.busy`.
    pub fn set_material_data(
        &mut self,
        gpu: &GpuContext,
        base: u32,
        words: &[u32],
    ) -> Result<(), RaaError> {
        let mut s = self.shared.try_borrow_mut().map_err(|_| pipeline_busy())?;
        s.set_material_data(gpu, base, words)
    }

    /// Frees the block at `base` (its words read zero again). Idempotent: an unknown or
    /// removed base does nothing. Returns whether it was live.
    pub fn remove_material_data(&mut self, gpu: &GpuContext, base: u32) -> Result<bool, RaaError> {
        let mut s = self.shared.try_borrow_mut().map_err(|_| pipeline_busy())?;
        Ok(s.remove_material_data(gpu, base))
    }

    /// The words of the live block at `base`, as stored.
    pub fn material_data(&self, base: u32) -> Option<Vec<u32>> {
        self.shared
            .borrow()
            .material_data()
            .block(base)
            .map(<[u32]>::to_vec)
    }

    /// Material data placement: live blocks, words they hold, end of the highest block and
    /// capacity of the GPU buffer, in words.
    pub fn material_data_stats(&self) -> MaterialDataStats {
        let s = self.shared.borrow();
        let d = s.material_data();
        MaterialDataStats {
            blocks: d.allocator().blocks(),
            used_words: d.allocator().used_words(),
            top: d.allocator().top(),
            capacity: d.capacity(),
        }
    }

    /// Adds a forward contributor and its pipeline contributor entry (enabled; the forward
    /// pass records the contributor only while the entry is enabled). Refusals:
    /// `pipeline.duplicate_pass` (nothing added), `pipeline.busy`.
    pub fn add_contributor(&mut self, c: Box<dyn ForwardContributor>) -> Result<(), RaaError> {
        let order = c.order();
        let enabled = Rc::new(Cell::new(true));
        self.pipeline
            .try_borrow_mut()
            .map_err(|_| pipeline_busy())?
            .add_contributor_entry(c.name(), order, Rc::clone(&enabled))?;
        let mut slots = self
            .contributors
            .try_borrow_mut()
            .map_err(|_| pipeline_busy())?;
        let at = insert_position(&slots, order);
        slots.insert(
            at,
            ContributorSlot {
                order,
                enabled,
                contributor: c,
            },
        );
        Ok(())
    }

    /// Sets the display-referred background colour (S12: written unchanged where nothing
    /// was drawn while no sky contributor is enabled) in the pipeline's `EnvGpu`. A colour
    /// with a non-finite component is ignored.
    pub fn set_background(&mut self, rgb: [f32; 3]) {
        if !rgb.iter().all(|c| c.is_finite()) {
            return;
        }
        match self.pipeline.try_borrow_mut() {
            Ok(mut p) => {
                p.lighting_mut().env_mut().background = [rgb[0], rgb[1], rgb[2], 0.0];
                self.pending_background = None;
            }
            Err(_) => self.pending_background = Some(rgb),
        }
    }

    // ---- local lights (M2-LOCAL-STATE) ----

    /// The S17 local-light registry owned by this renderer: stable handles, validated
    /// descriptors and instance-group attachments.
    pub fn local_lights(&self) -> &LocalLights {
        &self.local_lights
    }

    /// Mutable access for the Engine's S17 mutators and object-mutation validation; every
    /// mutation of the registry is validated by `LocalLights` before it applies.
    pub fn local_lights_mut(&mut self) -> &mut LocalLights {
        &mut self.local_lights
    }

    /// The latest completed S12 tile-overflow statistics of this renderer: registry
    /// counts, viewport 0's grid, the newest published submitted-invocation sum and its
    /// pending state. Synchronous and never a GPU fence (S17 `lightStats`).
    ///
    /// Handoff: the m28 `Engine.lightStats()` implementation calls
    /// `LitRenderer::light_stats(&self) -> LocalLightStats` in
    /// `crates/raa-lit/src/renderer.rs`; the returned fields map one-to-one to S17's
    /// `LightStats` and read the same `LightHandle` registry and overflow pool as the
    /// order-35 pass.
    pub fn light_stats(&self) -> LocalLightStats {
        let readback = self.readback.lock().unwrap_or_else(|e| e.into_inner());
        LocalLightStats::from_coordinator(
            self.local_lights.active_count(),
            self.local_lights.attached_count(),
            &readback,
        )
    }

    // ---- virtualized instances and frame features (P3-LIT-SEAM) ----

    /// The virtualized scene instances. Members keep their persistent records but are left
    /// out of every draw list (prepass, forward, cascade casters) from the next `prepare`.
    pub fn virtual_instances_mut(&mut self) -> &mut VirtualSet {
        &mut self.virtual_set
    }

    /// The virtualized scene instances.
    pub fn virtual_instances(&self) -> &VirtualSet {
        &self.virtual_set
    }

    /// Sets logical `FrameGpu` feature `id` (S10) for every later `prepare`, after its own
    /// tiling bit. Id 24 (tiling, owned by `prepare`) and ids >= 64 (no address) are ignored.
    pub fn set_frame_feature(&mut self, id: u32, on: bool) {
        if id == TILING_FEATURE_ID || id >= 64 {
            return;
        }
        let bit = 1u64 << id;
        if on {
            self.frame_features |= bit;
        } else {
            self.frame_features &= !bit;
        }
    }

    /// The stored value of feature `id`; false for ignored ids.
    pub fn frame_feature(&self, id: u32) -> bool {
        id < 64 && self.frame_features & (1u64 << id) != 0
    }

    // ---- frame ----

    /// Opens scene invocation `index`; one index spans all of its viewports.
    pub fn begin_scene_frame(&mut self, index: u64) {
        self.scene_frame.begin(index);
    }

    /// Selects viewport 0..3 for the next `prepare`; `discontinuity` drops its history.
    pub fn prepare_scene_view(&mut self, view: usize, discontinuity: bool) {
        self.scene_frame.selected = Some((view % MAX_VIEWS, discontinuity));
    }

    /// Acknowledges a really submitted view (never a mere prepare); only these commit.
    pub fn submitted_scene_view(&mut self, view: usize) {
        self.scene_frame.submitted(view);
    }

    /// Closes the invocation; `complete` false invalidates camera history before the next
    /// begin and commits only the acknowledged instances. The pipeline's submission
    /// observation is captured here and the bounded overflow pool resolves it (with the
    /// GPU the map needs) at the next `prepare`.
    pub fn finish_scene_frame(&mut self, complete: bool) {
        if !self.scene_frame.open {
            return;
        }
        let index = self.scene_frame.frame_index;
        let observation = self
            .pipeline
            .try_borrow()
            .map(|p| p.scene_submission())
            .unwrap_or_default();
        self.scene_frame.finish(complete);
        if let Ok(mut guard) = self.shared.try_borrow_mut() {
            self.scene_frame.flush(&mut guard.store);
        }
        self.pending_resolve = Some((index, observation, complete));
    }

    /// Prepares a frame: syncs the uploaded mesh bounds into the scene, `scene.update_world`,
    /// `scene.build_render_list_slots(list, tags, slots)`, the persistent instance records
    /// and extras, the index-based camera selection (`IdPrep`: posed instances are tested
    /// through their skin bounds, the rest through the scene index), the `FrameGpu` from the
    /// scene camera, light and ambient, then the queue writes (frame, changed instance
    /// pages, id stream and, when changed, every material).
    ///
    /// Refusals (nothing written): `render.too_many_instances`, `render.too_many_materials`,
    /// `render.unknown_mesh`, `material.unknown_id`, `material.blend_unavailable` (a
    /// material's class has no handler), `render.invalid_camera`, `pipeline.busy`.
    pub fn prepare(
        &mut self,
        gpu: &GpuContext,
        scene: &mut Scene,
        list: &mut RenderList,
        tags: &mut Vec<u32>,
        slots: &mut Vec<u32>,
    ) -> Result<FrameCounts, RaaError> {
        if let Some(rgb) = self.pending_background {
            self.set_background(rgb);
        }
        if let Some((index, observation, submitted)) = self.pending_resolve.take() {
            ReadbackCoordinator::resolve(&self.readback, gpu, index, observation, submitted);
        }
        let (viewport, encode_srgb, tiling) = {
            let p = self.pipeline.try_borrow().map_err(|_| pipeline_busy())?;
            (
                p.targets().map_or((1, 1), |t| t.size()),
                !p.surface_is_srgb(),
                p.entry_enabled(LIGHT_TILES_NAME).unwrap_or(false),
            )
        };
        // Direct native callers keep the bounded single-view lifecycle: one invocation per
        // prepare, its only viewport is view 0 and it is submitted by returning Ok.
        let single_view = !self.scene_frame.open;
        if single_view {
            let index = self.scene_frame.frame_index;
            self.scene_frame.begin(index);
        }
        let camera = self.scene_frame.prepare(scene, viewport, JITTER_CONSUMER);
        let mut guard = self.shared.try_borrow_mut().map_err(|_| pipeline_busy())?;
        let s: &mut LitShared = &mut guard;
        // Commit the previous invocation before this one stages its records.
        self.scene_frame.flush(&mut s.store);

        // The scene index follows the uploaded mesh bounds; send only what changed.
        {
            let meshes = &s.meshes;
            for ids in self.submeshes.values() {
                for &mesh in ids {
                    let Some(info) = meshes.submesh(mesh).and_then(|m| meshes.info(m.lit)) else {
                        continue;
                    };
                    if self.sent_bounds.get(&mesh) != Some(&info.bounds) {
                        scene.set_mesh_bounds(mesh, info.bounds);
                        self.sent_bounds.insert(mesh, info.bounds);
                    }
                }
            }
        }
        scene.update_world();
        scene.build_render_list_slots(list, tags, slots);

        let max_materials = self.opts.max_materials as usize;
        if s.materials.len() > max_materials {
            return Err(refusal(
                "render.too_many_materials",
                format!(
                    "{} materials exceed max_materials {max_materials}",
                    s.materials.len()
                ),
            ));
        }
        if s.materials.is_dirty() || self.material_classes.len() != s.materials.len() {
            self.material_classes.clear();
            for id in 0..s.materials.len() as u32 {
                if let Some(m) = s.materials.get(MaterialId(id)) {
                    self.material_classes.push(MaterialClass::of(m));
                }
            }
        }
        // The palette bind group follows the skin palette buffer (re-created on growth).
        s.sync_palette_group(gpu);

        let meshes = &s.meshes;
        let handlers = &self.class_handlers;
        let known_mesh = |m: MeshId| meshes.submesh(m).is_some();
        let class_handled = |c: DrawClass| c == DrawClass::Opaque || handlers.contains_key(&c);

        // Fill the persistent records of every listed instance by its stable slot, then
        // select the camera section through the scene index.
        let end = slots.iter().copied().max().map_or(0, |slot| slot + 1);
        s.store.ensure(end);
        // Compare every frame: the store's doubling capacity, not the live count,
        // determines the range of stable slots, including holes.
        s.levels
            .resize(s.store.capacity() as usize, NO_CAMERA_LEVEL);
        let source = s.skin_source.clone();
        let skins = source.0.borrow();
        let has_palette = s.palette_ready();
        let skinned = |tag: u32| has_palette.then(|| skins.draw(tag)).flatten();
        s.store.sync(list, tags, slots, &self.extras, &|tag| {
            skinned(tag).map_or(0, |draw| draw.base)
        });
        // Virtualized members are mirrored by live slot after every record is written, so
        // the camera and cascade selections below see the same set.
        let capacity_words = (s.store.capacity() as usize).div_ceil(64).max(1);
        self.virtual_set
            .mirror(scene, capacity_words, &mut s.virtual_bits);
        // A posed (deformed) instance cannot reconstruct its earlier geometry, so only
        // rigid previous models participate in the previous-instance commit.
        let recorded = &mut self.scene_frame.slots[self.scene_frame.current_view];
        recorded.clear();
        recorded.extend(
            list.items
                .iter()
                .zip(tags.iter())
                .zip(slots.iter())
                .filter_map(|((_, &tag), &slot)| skinned(tag).is_none().then_some(slot)),
        );
        camera_levels(
            &CameraLodInput {
                list,
                tags,
                slots,
                index: scene.instance_index(),
                camera: &scene.camera,
                viewport_height: viewport.1,
                lods: &s.lods,
                bias: &s.lod_bias,
                options: &s.lod_options,
                skinned: &skinned,
            },
            &mut s.levels,
        );

        let planes = frustum_planes(scene.camera.view_proj());
        let words = (s.store.capacity() as usize).div_ceil(64).max(1);
        self.visible_bits.clear();
        self.visible_bits.resize(words, 0);
        self.visible_slots.clear();
        scene
            .instance_index()
            .query_frustum(&planes, &mut self.visible_slots);
        for &slot in &self.visible_slots {
            let slot = slot as usize;
            if let Some(word) = self.visible_bits.get_mut(slot / 64) {
                *word |= 1u64 << (slot % 64);
            }
        }
        let bits = &self.visible_bits;
        let virtual_bits = &s.virtual_bits;
        let vp = scene.camera.view_proj();
        let selected = |i: usize| -> bool {
            let Some(item) = list.items.get(i) else {
                return false;
            };
            if s.lods.is_level(item.mesh) {
                return false;
            }
            let slot = slots.get(i).copied().unwrap_or(u32::MAX) as usize;
            if slot_member(virtual_bits, slot) {
                return false;
            }
            let tag = tags.get(i).copied().unwrap_or(0);
            if skinned(tag).is_some() {
                let Some(info) = meshes.submesh(item.mesh).and_then(|m| meshes.info(m.lit)) else {
                    return true;
                };
                let pose = skins.joints(tag);
                let bounds = crate::skin_draw::instance_bounds(info.bounds, pose.is_some(), pose);
                return crate::skin_bounds::visible(bounds, item.world, vp);
            }
            bits.get(slot / 64)
                .is_some_and(|word| word & (1u64 << (slot % 64)) != 0)
        };
        let level_mesh = |slot: u32, mesh: MeshId| camera_mesh(&s.lods, &s.levels, slot, mesh);
        let input = SelectInput {
            list,
            tags,
            slots,
            selected: &selected,
            materials: &self.material_classes,
            view: scene.camera.view(),
            known_mesh: &known_mesh,
            class_handled: &class_handled,
            level_mesh: &level_mesh,
        };
        s.select.prepare_camera(&input, &skinned)?;

        let mut stats = InstanceFrameStats {
            instances: scene.instance_count() as u32,
            camera: s.select.camera_count(),
            ..InstanceFrameStats::default()
        };
        for id in s.select.id_bytes().chunks_exact(4) {
            let slot = u32::from_le_bytes(id.try_into().unwrap()) as usize;
            let level = s.levels[slot];
            stats.lods[if level == NO_CAMERA_LEVEL {
                0
            } else {
                level as usize
            }] += 1;
        }

        let mut counts = FrameCounts {
            instances: s.select.camera_count(),
            ..FrameCounts::default()
        };
        for class in [DrawClass::Opaque, DrawClass::Fur, DrawClass::Transparent] {
            for d in s.select.draws(class) {
                let (_, index_count) = s
                    .meshes
                    .submesh_range(s.select.mesh_binding(d.mesh).0)
                    .unwrap_or((0, 0));
                counts.draw_calls += 1;
                counts.triangles += u64::from(index_count / 3) * u64::from(d.instance_count);
            }
        }

        let [active_lights, tile_columns, tile_rows, tile_stride] =
            light_info(self.local_lights.active_count(), viewport, tiling);
        let mut params = FrameParams {
            viewport,
            encode_srgb,
            msaa: self.opts.msaa,
            frame_index: camera.frame_index,
            camera_jitter_ndc: camera.current_jitter,
            previous_jitter_ndc: camera.previous_jitter,
            previous_view_proj: camera.previous_view_proj,
            history_valid: camera.history_valid,
            frame_parity: (camera.frame_index & 1) as u32,
            active_lights,
            tile_columns,
            tile_rows,
            tile_stride,
            ..FrameParams::default()
        };
        params.set_feature(TILING_FEATURE_ID, tiling);
        for id in (0..64u32).filter(|&id| self.frame_features & (1u64 << id) != 0) {
            params.set_feature(id, true);
        }
        write_frame_gpu(&mut self.frame_bytes, scene, &params)?;
        // Publish the order-35 pass inputs before it prepares: the same frame snapshot,
        // the invocation/view identity and the current dense registry records (attachments
        // are applied by the Engine before prepare; `world_desc` carries their latest
        // world transform). Authored lights are never deleted by disabling the pass.
        {
            let mut inputs = self.tile_inputs.borrow_mut();
            inputs.frame.copy_from_slice(&self.frame_bytes);
            inputs.index = self.scene_frame.frame_index;
            inputs.view = self.scene_frame.current_view;
            publish_local_lights(&mut inputs, &mut self.local_lights);
        }
        if tiling {
            self.readback
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .begin(self.scene_frame.frame_index, (tile_columns, tile_rows));
        }

        // Snapshot the frame the shadow cascade selection will read.
        s.index.clone_from(scene.instance_index());
        s.items.clone_from(list);
        s.item_tags.clone_from(tags);
        s.item_slots.clone_from(slots);

        s.upload(gpu, &self.frame_bytes);
        stats.uploaded_bytes = s.store.last_uploaded_bytes();
        s.stats = stats;
        if single_view {
            let index = self.scene_frame.frame_index;
            self.scene_frame.submitted(0);
            self.scene_frame.finish(true);
            self.scene_frame.flush(&mut s.store);
            // The caller submits this frame after prepare; the next prepare resolves it.
            let observation = SubmissionState {
                submitted_mask: 1,
                ..SubmissionState::default()
            };
            self.pending_resolve = Some((index, observation, true));
        }
        Ok(counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_math::Mat4;
    use raa_scene::RenderItem;

    struct LodFrame {
        list: RenderList,
        tags: Vec<u32>,
        slots: Vec<u32>,
        index: InstanceIndex,
        camera: Camera,
        lods: LodRegistry,
        bias: HashMap<u32, i8>,
        levels: Vec<u8>,
    }

    impl LodFrame {
        fn new(distances: &[f32]) -> Self {
            let mut lods = LodRegistry::new();
            // Registration deliberately knows nothing about instance poses.
            register_mesh_lods(
                &mut lods,
                MeshId(0),
                &[
                    LodLevel {
                        mesh: MeshId(1),
                        error: 0.01,
                    },
                    LodLevel {
                        mesh: MeshId(2),
                        error: 0.04,
                    },
                    LodLevel {
                        mesh: MeshId(3),
                        error: 0.16,
                    },
                ],
                &|_| Some((1, 0)),
            )
            .unwrap();
            Self {
                list: RenderList {
                    items: distances
                        .iter()
                        .map(|&distance| RenderItem {
                            mesh: MeshId(0),
                            material: MaterialId(0),
                            world: Mat4::from_translation(Vec3::new(0.0, 0.0, -distance)),
                        })
                        .collect(),
                    batches: Vec::new(),
                },
                tags: (10..10 + distances.len() as u32).collect(),
                // Deliberately sparse and different from tags and list indices.
                slots: (0..distances.len() as u32).map(|i| i * 2 + 1).collect(),
                index: InstanceIndex::new(),
                camera: Camera {
                    eye: Vec3::ZERO,
                    ..Camera::default()
                },
                lods,
                bias: HashMap::new(),
                levels: vec![NO_CAMERA_LEVEL; distances.len() * 2 + 1],
            }
        }

        fn pass(&mut self, skinned: &dyn Fn(u32) -> Option<SkinDraw>) {
            camera_levels(
                &CameraLodInput {
                    list: &self.list,
                    tags: &self.tags,
                    slots: &self.slots,
                    index: &self.index,
                    camera: &self.camera,
                    viewport_height: 1080,
                    lods: &self.lods,
                    bias: &self.bias,
                    options: &LodParams::default(),
                    skinned,
                },
                &mut self.levels,
            );
        }

        fn selected_levels(&self) -> Vec<u8> {
            self.slots
                .iter()
                .map(|&slot| self.levels[slot as usize])
                .collect()
        }
    }

    #[test]
    fn camera_levels_select_by_distance_and_return_near_in_one_frame() {
        let mut f = LodFrame::new(&[5.0, 60.0, 2000.0]);
        f.pass(&|_| None);
        // Q3's projected-error rule allows level 2 at 60 m: 0.04 * 935.3 / 60 < 1.
        assert_eq!(f.selected_levels(), [0, 2, 3]);
        f.list.items[2].world = f.list.items[0].world;
        f.pass(&|_| None);
        assert_eq!(f.selected_levels(), [0, 2, 0]);
        assert_eq!(f.levels[0], NO_CAMERA_LEVEL);
    }

    #[test]
    fn camera_levels_apply_tag_bias_before_hysteresis() {
        let mut f = LodFrame::new(&[5.0, 5.0]);
        f.bias.insert(f.tags[0], 1);
        f.pass(&|_| None);
        assert_eq!(f.selected_levels(), [1, 0]);
    }

    #[test]
    fn camera_levels_use_world_sphere_distance_and_largest_column_scale() {
        let mut f = LodFrame::new(&[60.0, 60.0]);
        // Centre distance 60 would select 1; sphere-surface distance 5 selects 0.
        f.index.insert(
            f.slots[0],
            Some(Aabb {
                min: Vec3::new(0.0, 0.0, -115.0),
                max: Vec3::new(0.0, 0.0, -5.0),
            }),
        );
        f.list.items[1].world.cols[1][1] = -10.0;
        f.pass(&|_| None);
        assert_eq!(f.selected_levels(), [0, 0]);
    }

    #[test]
    fn mesh_lods_refuse_missing_or_mismatched_uploads_atomically() {
        let mut f = LodFrame::new(&[]);
        let original = f.lods.chain(MeshId(0)).unwrap().to_vec();
        let levels = [LodLevel {
            mesh: MeshId(4),
            error: 0.02,
        }];
        for layouts in [
            [None, Some((1, 0))],
            [Some((1, 0)), None],
            [Some((1, 0)), Some((2, 0))],
            [Some((2, 0)), Some((2, 1))],
        ] {
            let err = register_mesh_lods(&mut f.lods, MeshId(0), &levels, &|mesh| {
                layouts[usize::from(mesh != MeshId(0))]
            })
            .unwrap_err();
            assert_eq!(err.code(), "render.invalid_lod");
            assert_eq!(f.lods.chain(MeshId(0)).unwrap(), original);
        }
    }

    #[test]
    fn posed_camera_instances_keep_full_geometry_and_drop_hysteresis() {
        let mut f = LodFrame::new(&[20.0, 20.0]);
        let posed_tag = f.tags[0];
        let skinned = |tag| (tag == posed_tag).then_some(SkinDraw { base: 4, skin: 7 });
        f.pass(&|_| None);
        assert_eq!(f.selected_levels(), [1, 1]);
        f.pass(&skinned);
        assert_eq!(f.selected_levels(), [NO_CAMERA_LEVEL, 1]);

        let materials = [MaterialClass {
            class: DrawClass::Opaque,
            material_fn: MaterialFnId(7),
        }];
        let mut prep = crate::IdPrep::new(2, 1);
        let level_mesh = |slot, mesh| camera_mesh(&f.lods, &f.levels, slot, mesh);
        prep.prepare_camera(
            &SelectInput {
                list: &f.list,
                tags: &f.tags,
                slots: &f.slots,
                selected: &|_| true,
                materials: &materials,
                view: Mat4::IDENTITY,
                known_mesh: &|_| true,
                class_handled: &|_| true,
                level_mesh: &level_mesh,
            },
            &skinned,
        )
        .unwrap();
        let draws = prep.draws(DrawClass::Opaque);
        assert_eq!(draws.len(), 2);
        assert_eq!(prep.mesh_binding(draws[0].mesh), (MeshId(1), None));
        assert_eq!(
            prep.mesh_binding(draws[1].mesh),
            (MeshId(0), Some(posed_tag))
        );
        assert!(draws.iter().all(|draw| {
            draw.material == MaterialId(0) && draw.material_fn == materials[0].material_fn
        }));

        // Level 1 is allowed at 0.9 px, but coarsening from previous 0 needs <= 0.8.
        // Clearing the pose must use None rather than a fabricated previous level 0.
        let distance = 0.01 * 1080.0 / (2.0 * (f.camera.fov_y_rad * 0.5).tan()) / 0.9;
        f.list.items[0].world = Mat4::from_translation(Vec3::new(0.0, 0.0, -distance));
        f.pass(&|_| None);
        assert_eq!(f.selected_levels(), [1, 1]);
    }

    struct Fake(&'static str, u32);

    impl ForwardContributor for Fake {
        fn name(&self) -> &str {
            self.0
        }

        fn order(&self) -> u32 {
            self.1
        }

        fn record(&self, _: &mut wgpu::RenderPass<'_>, _: &LitHandle) -> Result<(), RaaError> {
            Ok(())
        }
    }

    #[test]
    fn options_default_to_the_contract_limits() {
        let o = LitOptions::default();
        assert_eq!((o.max_instances, o.max_materials, o.msaa), (16384, 256, 4));
        assert!(o.validate().is_ok());
    }

    #[test]
    fn options_refuse_msaa_other_than_1_or_4() {
        for msaa in [1, 4] {
            let o = LitOptions {
                msaa,
                ..LitOptions::default()
            };
            assert!(o.validate().is_ok(), "msaa {msaa}");
        }
        for msaa in [0, 2, 3, 8, 16] {
            let o = LitOptions {
                msaa,
                ..LitOptions::default()
            };
            let err = o.validate().unwrap_err();
            assert_eq!(err.code(), "render.invalid_msaa");
            assert!(err.message().contains(&msaa.to_string()));
        }
    }

    #[test]
    fn contributors_insert_by_order_then_registration() {
        let mut slots: Vec<ContributorSlot> = Vec::new();
        for (name, order) in [("a", 900), ("b", 500), ("c", 900), ("d", 600), ("e", 500)] {
            let at = insert_position(&slots, order);
            slots.insert(
                at,
                ContributorSlot {
                    order,
                    enabled: Rc::new(Cell::new(true)),
                    contributor: Box::new(Fake(name, order)),
                },
            );
        }
        let names: Vec<&str> = slots.iter().map(|s| s.contributor.name()).collect();
        assert_eq!(names, ["b", "e", "d", "a", "c"]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn clear_mesh_lods_returns_true_once_then_false() {
        let Some(gpu) = raa_core::testing::software_gpu() else {
            raa_core::testing::skip("clear_mesh_lods", "no adapter");
            return;
        };
        let pipeline = Rc::new(RefCell::new(Pipeline::new(
            &gpu,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        )));
        let mut lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).unwrap();
        let levels = [LodLevel {
            mesh: MeshId(1),
            error: 1.0,
        }];
        lit.shared
            .borrow_mut()
            .lods
            .set(MeshId(0), &levels)
            .unwrap();
        assert!(lit.has_mesh_lods(MeshId(0)));
        assert!(lit.clear_mesh_lods(MeshId(0)));
        assert!(!lit.has_mesh_lods(MeshId(0)));
        assert!(!lit.clear_mesh_lods(MeshId(0)));
    }

    const LIFE_SIZE: (u32, u32) = (640, 360);

    fn life_scene(eye: Vec3) -> Scene {
        let mut scene = Scene::new();
        scene.camera.eye = eye;
        scene.camera.set_viewport(LIFE_SIZE.0, LIFE_SIZE.1);
        scene
    }

    /// The sequence `LitRenderer::prepare` runs, without a GPU device.
    struct LifeFrame {
        store: InstanceStore,
        list: RenderList,
        tags: Vec<u32>,
        slots: Vec<u32>,
        lifecycle: SceneLifecycle,
    }

    impl LifeFrame {
        fn new(worlds: &[Mat4]) -> Self {
            let list = RenderList {
                items: worlds
                    .iter()
                    .enumerate()
                    .map(|(i, &world)| RenderItem {
                        mesh: MeshId(i as u32),
                        material: MaterialId(0),
                        world,
                    })
                    .collect(),
                ..RenderList::default()
            };
            LifeFrame {
                store: InstanceStore::new(worlds.len().max(1) as u32),
                tags: (0..worlds.len() as u32).collect(),
                slots: (0..worlds.len() as u32).collect(),
                list,
                lifecycle: SceneLifecycle::new(),
            }
        }

        fn prepare(&mut self, scene: &Scene, view: usize, discontinuity: bool) -> CameraHistory {
            self.lifecycle.selected = Some((view % MAX_VIEWS, discontinuity));
            let history = self.lifecycle.prepare(scene, LIFE_SIZE, JITTER_CONSUMER);
            self.store
                .sync(&self.list, &self.tags, &self.slots, &HashMap::new(), &|_| 0);
            let recorded = &mut self.lifecycle.slots[view % MAX_VIEWS];
            recorded.clear();
            recorded.extend(self.slots.iter().copied());
            history
        }

        fn finish(&mut self, complete: bool) {
            self.lifecycle.finish(complete);
            self.lifecycle.flush(&mut self.store);
        }
    }

    fn previous_valid(bytes: &[u8], slot: usize) -> u32 {
        u32::from_le_bytes(bytes[slot * 64 + 48..slot * 64 + 52].try_into().unwrap())
    }

    fn previous_tx(bytes: &[u8], slot: usize) -> f32 {
        f32::from_le_bytes(bytes[slot * 64 + 12..slot * 64 + 16].try_into().unwrap())
    }

    #[test]
    fn frame_lifecycle_two_cameras_share_frozen_instance_history() {
        let mut frame = LifeFrame::new(&[
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0)),
        ]);
        let (a, b) = (
            life_scene(Vec3::new(0.0, 0.0, 6.0)),
            life_scene(Vec3::new(3.0, 0.0, 6.0)),
        );
        frame.lifecycle.begin(0);
        frame.prepare(&a, 0, false);
        frame.lifecycle.submitted(0);
        frame.finish(true);
        let frozen = frame.store.previous_bytes().to_vec();
        assert_eq!(
            (previous_valid(&frozen, 0), previous_valid(&frozen, 1)),
            (1, 1)
        );

        // One invocation draws two cameras: both see the same frozen previous records.
        frame.lifecycle.begin(1);
        let view0 = frame.prepare(&b, 0, false);
        assert_eq!((view0.history_valid, view0.frame_index), (true, 1));
        assert_eq!(view0.previous_view_proj, a.camera.view_proj());
        assert_eq!(frame.store.previous_bytes(), frozen.as_slice());
        let view1 = frame.prepare(&a, 1, false);
        assert!(!view1.history_valid && view1.frame_index == 1);
        assert_eq!(frame.store.previous_bytes(), frozen.as_slice());
        frame.lifecycle.submitted(1);
        frame.finish(true);
        assert_eq!(frame.store.previous_bytes(), frozen.as_slice());
    }

    #[test]
    fn frame_lifecycle_previous_history_catches_up_without_another_transform_edit() {
        let mut frame = LifeFrame::new(&[Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))]);
        let scene = life_scene(Vec3::new(0.0, 0.0, 6.0));
        // Frame 1 moves the instance; frame 2 has no edit and finds the catch-up model.
        for (index, x, committed) in [(0, 1.0, 1.0), (1, 4.0, 1.0), (2, 4.0, 4.0)] {
            frame.list.items[0].world = Mat4::from_translation(Vec3::new(x, 0.0, 0.0));
            frame.lifecycle.begin(index);
            frame.prepare(&scene, 0, false);
            assert_eq!(previous_tx(frame.store.previous_bytes(), 0), committed);
            frame.lifecycle.submitted(0);
            frame.finish(true);
        }
        assert_eq!(previous_tx(frame.store.previous_bytes(), 0), 4.0);
    }
}
