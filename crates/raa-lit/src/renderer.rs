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
use raa_pipeline::{
    InterlaceSignals, IrradianceRate, LightingInterlace, Pipeline, ShadowRate, SubmissionState,
};
use raa_scene::{Camera, InstanceIndex, RenderList, Scene, frustum_planes};
use raa_shader::{
    Composer, LIGHT_GPU_SIZE, MAX_LOCAL_LIGHTS, MaterialFnId, MaterialFnRegistry, PassKind,
    SlotProvider, VariantKey,
};

use crate::area_shadow::{
    AreaShadowCamera, AreaShadowSettings, AreaShadowTier, HeroSelector, MAX_HERO_LIGHTS,
};
use crate::area_stochastic::{AreaStochasticFrame, StochasticSelector, StochasticShadowSettings};
use crate::billboard_bake::{
    BillboardBakeInput, BillboardBakeResult, billboard_camera, billboard_framing,
};
use crate::billboard_targets::BillboardTargets;
use crate::contributor::ForwardContributor;
use crate::extras::{INSTANCE_EXTRA_GPU_SIZE, InstanceExtra};
use crate::foliage_bake::{
    FoliageAsset, FoliageBakeSettings, FoliageBakeState, bake_foliage, check_material,
    publish_foliage, retire_foliage,
};
use crate::foliage_resources::FoliageSource;
use crate::frame::{FRAME_GPU_SIZE, FrameParams, write_frame_flags, write_frame_gpu};
use crate::frame_history::{CameraHistory, FrameHistory, MAX_VIEWS, frame_jitter};
use crate::handle::{LitHandle, LitShared};
use crate::instance_store::{InstanceStore, write_instance_record};
use crate::library::SurfaceLibrary;
use crate::light_stats::{LocalLightStats, ReadbackCoordinator};
use crate::local_lights::LocalLights;
use crate::lod::{LodLevel, LodParams, LodRegistry, MAX_LOD_LEVELS, camera_level};
use crate::material::{DrawClass, FLAG_DOUBLE_SIDED, SURFACE_MATERIAL_GPU_SIZE, SurfaceMaterial};
use crate::material_data::MaterialDataStats;
use crate::meshes::{LitMeshId, LitMeshInfo};
use crate::passes::install_passes;
use crate::prep::{INSTANCE_GPU_SIZE, MaterialClass};
use crate::select::SelectInput;
use crate::skin_views::SkinDraw;
use crate::textures::{ATLAS_FORMAT, AtlasRect, TexelData, TextureOptions};
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

/// Name of the TAA pass in the pipeline (S12): its enabled entry makes the frame a
/// temporal consumer, so the viewport gets the S10 jitter and accumulation (S43).
pub(crate) const TAA_PASS_NAME: &str = "taa";

/// True only when the pipeline's `taa` entry exists and is enabled; a missing or disabled
/// entry leaves every viewport with zero current and previous jitter and no accumulation.
fn jitter_consumer(taa_enabled: Option<bool>) -> bool {
    taa_enabled == Some(true)
}

/// The enabled pipeline entries whose reads keep a signal pass running on a G frame of forward
/// "layers-interlaced" (names of raa-shafts and raa-vg).
///
/// `transparent.oit`, `fur.shells` and `particles.opaque` (raa_slot_shadow, raa_slot_ao) do
/// not: on a G frame they read the previous R frame's cascade maps with that frame's
/// `ShadowGpu` (raa-shadow freezes the cascade matrices and splits while the cascades are not
/// rendered, so map and matrices agree) and the contact / AO textures that R frame left bound
/// (screen-space, one frame old).
#[derive(Clone, Copy, Debug, Default)]
struct SignalReaders {
    /// `sunshafts`: marches the cascade layers along view rays; left running (not reviewed for
    /// stale cascades).
    shafts: bool,
    /// `vg.shade`: raa_opaque_lighting of opaque virtualized geometry (cascades, contact, AO and
    /// DDGI resolve): an opaque surface must not take one-frame-old screen-space AO / contact,
    /// so every signal pass keeps running.
    vg_shade: bool,
}

impl SignalReaders {
    fn of(p: &Pipeline) -> SignalReaders {
        let on = |name: &str| p.entry_enabled(name).unwrap_or(false);
        SignalReaders {
            shafts: on("sunshafts"),
            vg_shade: on("vg.shade"),
        }
    }
}

/// The forward "layers-interlaced" state of real frame `frame` (the scene frame index). On a
/// G frame of the mode the signal passes are skipped unless a reader that needs the current
/// frame's output is on: VG shade keeps all four, sun shafts and the planar reflection capture
/// (raa_slot_shadow of the mirrored view) keep the cascades, the planar reflection keeps the
/// contact. The signal rates keep what a G frame samples: s at every frame
/// ([`ShadowRate::Every`]) keeps the contact (the cascades stay those of the R frame), E on G
/// frames ([`IrradianceRate::Phase60`] / `Full`) keeps the DDGI screen resolve. The cascade
/// hero layers (local lights) are always rendered, and the DDGI probe update always runs.
fn interlace_state(
    enabled: bool,
    frame: u64,
    signals: InterlaceSignals,
    readers: SignalReaders,
    planar_reflection: bool,
) -> LightingInterlace {
    let mut state = LightingInterlace::with_signals(enabled, frame, signals);
    if state.generated() {
        let vg = readers.vg_shade;
        let shadow_g = state.signals.shadow == ShadowRate::Every;
        let irradiance_g = state.signals.irradiance != IrradianceRate::Phase30;
        state.skip_cascades = !(vg || readers.shafts || planar_reflection);
        state.skip_contact = !(vg || planar_reflection || shadow_g);
        state.skip_ao = !vg;
        state.skip_gi_resolve = !(vg || irradiance_g);
    }
    state
}

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
    implicit: bool,
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
            implicit: false,
            current_view: 0,
            slots: std::array::from_fn(|_| Vec::new()),
            acknowledged: 0,
            commit_pending: false,
        }
    }

    fn begin(&mut self, index: u64) {
        if self.open {
            self.history.finish();
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
        let jitter = frame_jitter(self.frame_index, size, consumer, self.history.jitter_phases);
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
        if view >= MAX_VIEWS || self.acknowledged & (1 << view) != 0 {
            return;
        }
        self.history.submitted(view);
        self.acknowledged |= 1 << view;
    }

    fn finish(&mut self) {
        if !self.open {
            return;
        }
        self.history.finish();
        self.open = false;
        self.implicit = false;
        self.selected = None;
        self.commit_pending = true;
        if self.acknowledged != 0 {
            self.frame_index = self.frame_index.saturating_add(1);
        }
    }

    fn flush(&mut self, store: &mut InstanceStore) {
        if !self.commit_pending {
            return;
        }
        for view in 0..MAX_VIEWS {
            if self.acknowledged & (1 << view) != 0 {
                for &slot in &self.slots[view] {
                    store.previous_submitted(slot);
                }
            }
        }
        self.acknowledged = 0;
        store.finish_previous_frame();
        self.commit_pending = false;
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
    camera_jitter: Option<bool>,
    /// Extra material texture mip bias while the camera is jittered (temporal accumulation
    /// resolves the finer level's detail), added to the TAAU bias; see
    /// [`Renderer::set_jitter_texture_lod_bias`].
    jitter_lod_bias: f32,
    jitter_phases: Option<u32>,
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
    /// Forward "layers-interlaced" state of the latest prepare, and of the latest invocation
    /// with an acknowledged submission ([`LitRenderer::lighting_interlace`]).
    interlace_prepared: LightingInterlace,
    interlace_submitted: Option<LightingInterlace>,
    /// Signal rates of forward "layers-interlaced" ([`LitRenderer::set_interlace_signals`]).
    interlace_signals: InterlaceSignals,
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
    /// FrameGpu.reserved water lanes for every later `prepare` (zero = no water).
    water: [f32; 16],
    /// Area-light soft-shadow settings; `AreaShadowSettings::default()` until set.
    area_shadow_settings: AreaShadowSettings,
    /// One hero selector per viewport, created once so prepare allocates nothing.
    hero_selectors: [HeroSelector; MAX_VIEWS],
    /// One local-map selector per viewport (`shadowMode: 'map'` lights).
    local_map_selectors: [crate::LocalMapSelector; MAX_VIEWS],
    /// Heroes occupied by the most recently prepared view (0 before any prepare).
    active_area_heroes: u32,
    /// Mid-tier stochastic shadow settings; `StochasticShadowSettings::standard()` until set.
    area_stochastic_settings: StochasticShadowSettings,
    /// One stochastic selector per viewport, created once so prepare allocates nothing.
    stochastic_selectors: [StochasticSelector; MAX_VIEWS],
    /// Mid-tier slots occupied by the most recently prepared view (0 before any prepare).
    active_area_stochastic: u32,
    /// Latest requested foliage generation per material and the proxy meshes published by
    /// pending [`LitRenderer::prepare_foliage`] futures since the last frame.
    foliage_bakes: Rc<RefCell<FoliageBakeState>>,
}

// Keeps partial atlas allocations private to the bake until successful completion.
// Also cleans up if the caller drops the returned future before polling it.
struct BakePublication {
    shared: Rc<RefCell<LitShared>>,
    gpu: GpuContext,
    images: Option<[u32; 3]>,
    _targets: BillboardTargets,
}

impl BakePublication {
    fn complete(mut self) {
        self.images = None;
    }
}

impl Drop for BakePublication {
    fn drop(&mut self) {
        if let Some(images) = self.images.take() {
            let mut shared = self.shared.borrow_mut();
            for id in images {
                shared.remove_texture(&self.gpu, id);
            }
        }
    }
}

/// One submitted draw: its sub-mesh, material and root-frame instance record.
struct BakeSource {
    mesh: MeshId,
    material: u32,
    record: [u8; INSTANCE_GPU_SIZE],
    extra: InstanceExtra,
}

/// Depth state and format of the bake's own depth attachment.
const BAKE_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

fn bake_error(message: String) -> RaaError {
    RaaError::FrameGraph {
        code: "billboard.invalid",
        message,
    }
}

fn bake_shader_error(e: raa_shader::ShaderError) -> RaaError {
    RaaError::FrameGraph {
        code: e.code,
        message: e.message,
    }
}

/// A bake colour attachment cleared to zero, stored for the tile copy.
fn bake_color(view: &wgpu::TextureView) -> Option<wgpu::RenderPassColorAttachment<'_>> {
    Some(wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: wgpu::StoreOp::Store,
        },
    })
}

/// A vertex/storage buffer over `bytes` (COPY_DST for a single queue write).
fn bake_buffer(
    gpu: &GpuContext,
    label: &str,
    bytes: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len().max(4) as u64,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !bytes.is_empty() {
        gpu.queue().write_buffer(&buffer, 0, bytes);
    }
    buffer
}

/// The bake's draw pipeline for one (material function, double-sided) pair: the unlit
/// bake shader of `compose_billboard_bake`, the standard lit group layouts and vertex
/// buffers, three rgba8unorm colour targets and depth Less.
fn bake_pipeline(
    gpu: &GpuContext,
    module: &wgpu::ShaderModule,
    frame_layout: &wgpu::BindGroupLayout,
    lighting_layout: &wgpu::BindGroupLayout,
    double_sided: bool,
) -> wgpu::RenderPipeline {
    let device = gpu.device();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("raa.lit.billboard_bake.layout"),
        bind_group_layouts: &[Some(frame_layout), Some(lighting_layout)],
        immediate_size: 0,
    });
    let buffers = crate::handle::bake_vertex_buffer_layouts();
    let targets = [ATLAS_FORMAT; 3].map(|format| {
        Some(wgpu::ColorTargetState {
            format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("raa.lit.billboard_bake.pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: (!double_sided).then_some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: BAKE_DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &targets,
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// A vertex buffer over `bytes` for a source-draw submission (the M7 reference geometry
/// streams share the M6 bake's private GPU path).
pub(crate) fn source_vertex_buffer(gpu: &GpuContext, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    bake_buffer(
        gpu,
        &format!("raa.lit.source.{label}"),
        bytes,
        wgpu::BufferUsages::VERTEX,
    )
}

/// Colour formats of the M7 reference targets, in location order: world normal XYZ and
/// coverage W (Rgba16Float), standard-Z depth X (Rgba32Float) and linear unlit albedo RGB
/// and coverage A (Rgba8Unorm).
pub(crate) const REFERENCE_TARGET_FORMATS: [wgpu::TextureFormat; 3] = [
    wgpu::TextureFormat::Rgba16Float,
    wgpu::TextureFormat::Rgba32Float,
    wgpu::TextureFormat::Rgba8Unorm,
];

/// The reusable source-draw submission state of the M6 billboard bake and the M7 reference
/// renderer: a private frame uniform, the material, extras, instance and previous buffers,
/// the bake's compacted instance id stream and a group-0 bind group over the renderer's
/// atlas and material data. It never touches the prepared frame's camera or records.
pub(crate) struct SourceDrawBuffers {
    pub(crate) frame: wgpu::Buffer,
    pub(crate) ids: wgpu::Buffer,
    pub(crate) bind: wgpu::BindGroup,
}

impl SourceDrawBuffers {
    pub(crate) fn new(
        gpu: &GpuContext,
        handle: &LitHandle,
        material_bytes: &[u8],
        instance_bytes: &[u8],
        extra_bytes: &[u8],
        id_bytes: &[u8],
    ) -> Self {
        let frame = bake_buffer(
            gpu,
            "raa.lit.source.frame",
            &[0u8; FRAME_GPU_SIZE],
            wgpu::BufferUsages::UNIFORM,
        );
        let materials = bake_buffer(
            gpu,
            "raa.lit.source.materials",
            material_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let extras = bake_buffer(
            gpu,
            "raa.lit.source.extras",
            extra_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let instances = bake_buffer(
            gpu,
            "raa.lit.source.instances",
            instance_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let previous = bake_buffer(
            gpu,
            "raa.lit.source.previous",
            &vec![0u8; instance_bytes.len() / INSTANCE_GPU_SIZE * 64],
            wgpu::BufferUsages::STORAGE,
        );
        let ids = bake_buffer(
            gpu,
            "raa.lit.source.ids",
            id_bytes,
            wgpu::BufferUsages::VERTEX,
        );
        let bind =
            handle.bake_frame_bind_group(gpu, [&frame, &materials, &extras, &instances, &previous]);
        SourceDrawBuffers { frame, ids, bind }
    }
}

/// The M7 reference pipeline for one (material function, double-sided) pair: the reference
/// shader of `compose_foliage_reference`, group 0 and the standard lit vertex buffers, the
/// three reference colour targets and a Depth32Float attachment with Less.
pub(crate) fn reference_pipeline(
    gpu: &GpuContext,
    module: &wgpu::ShaderModule,
    frame_layout: &wgpu::BindGroupLayout,
    double_sided: bool,
) -> wgpu::RenderPipeline {
    let device = gpu.device();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("raa.lit.foliage_reference.layout"),
        bind_group_layouts: &[Some(frame_layout)],
        immediate_size: 0,
    });
    let buffers = crate::handle::bake_vertex_buffer_layouts();
    let targets = REFERENCE_TARGET_FORMATS.map(|format| {
        Some(wgpu::ColorTargetState {
            format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("raa.lit.foliage_reference.pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: (!double_sided).then_some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: BAKE_DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &targets,
        }),
        multiview_mask: None,
        cache: None,
    })
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
                p.lighting().opaque_layout(),
                p.lighting().transparent_layout(),
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
            shadow_slots: Vec::new(),
            gi_excluded: [0; raa_shader::MAX_LOCAL_LIGHTS / 64],
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
            camera_jitter: None,
            jitter_lod_bias: crate::frame::JITTER_TEXTURE_LOD_BIAS,
            jitter_phases: None,
            local_lights: LocalLights::new(),
            tile_inputs,
            readback,
            pending_resolve: None,
            pending_background: None,
            interlace_prepared: LightingInterlace::default(),
            interlace_submitted: None,
            interlace_signals: InterlaceSignals::default(),
            sent_bounds: HashMap::new(),
            submeshes: HashMap::new(),
            visible_slots: Vec::new(),
            visible_bits: Vec::new(),
            virtual_set: VirtualSet::new(),
            frame_features: 0,
            water: [0.0; 16],
            area_shadow_settings: AreaShadowSettings::default(),
            hero_selectors: std::array::from_fn(|_| HeroSelector::new()),
            local_map_selectors: std::array::from_fn(|_| crate::LocalMapSelector::new()),
            active_area_heroes: 0,
            area_stochastic_settings: StochasticShadowSettings::default(),
            stochastic_selectors: std::array::from_fn(|_| StochasticSelector::new()),
            active_area_stochastic: 0,
            foliage_bakes: Rc::new(RefCell::new(FoliageBakeState::default())),
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
        self.shared
            .borrow_mut()
            .set_mesh_colors(id, m.colors.is_some());
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
        let before = self.submesh_list(id);
        self.shared.borrow_mut().meshes.update(gpu, id, m)?;
        self.shared
            .borrow_mut()
            .set_mesh_colors(id, m.colors.is_some());
        self.refresh_submeshes(id);
        let after = self.submesh_list(id);
        let freed: Vec<MeshId> = before.into_iter().filter(|m| !after.contains(m)).collect();
        self.forget_lods(&freed);
        Ok(())
    }

    /// Frees mesh `id` and its sub-mesh ids (reused lowest first), with every detail chain
    /// that names one of them ([`LodRegistry::forget`]). Unknown ids are ignored.
    pub fn remove(&mut self, id: LitMeshId) {
        let freed = self.submesh_list(id);
        self.shared.borrow_mut().meshes.remove(id);
        self.shared.borrow_mut().remove_mesh_colors(id);
        self.submeshes.remove(&id);
        self.forget_lods(&freed);
    }

    /// The sub-mesh ids of `id` (empty when unknown).
    fn submesh_list(&self, id: LitMeshId) -> Vec<MeshId> {
        self.submesh_ids(id)
            .map(|ids| ids.to_vec())
            .unwrap_or_default()
    }

    /// Drops the detail chains that name a freed sub-mesh id; the selected levels are
    /// recomputed, as after [`clear_mesh_lods`](Self::clear_mesh_lods).
    fn forget_lods(&mut self, freed: &[MeshId]) {
        if freed.is_empty() {
            return;
        }
        let mut s = self.shared.borrow_mut();
        let mut dropped = false;
        for &mesh in freed {
            dropped |= s.lods.forget(mesh);
        }
        if dropped {
            s.levels.fill(NO_CAMERA_LEVEL);
            s.shadow_levels.fill(NO_CAMERA_LEVEL);
        }
    }

    // ---- foliage bake (m7-bake-service) ----

    /// Prepares one native foliage generation of `source` for the compiled LOD1
    /// `settings.material` and resolves to the published [`FoliageAsset`].
    ///
    /// Settings, the source (non-empty, generation equal to `settings.params.generation`)
    /// and the material lease (live, with the '97. M7 foliage material data' envelope read
    /// through [`LitRenderer::material_data`]'s store) are validated before any allocation.
    /// The future owns every submission and partial allocation and holds no renderer borrow
    /// across an await; a failure or a dropped future releases them. The asset is published
    /// atomically (proxy mesh upload, then `LitHandle::set_foliage`) only after GPU
    /// completion and only while this is the material's newest generation; source draws are
    /// untouched until then. The upstream entry points are
    /// `crates/raa-lit/src/foliage_resources.rs:301 FoliageResources::new`,
    /// `crates/raa-lit/src/foliage_reference.rs:409 render_foliage_references`,
    /// `crates/raa-lit/src/foliage_reference.rs:627 pack_foliage_layers`,
    /// `crates/raa-lit/src/foliage_hull.rs:248 build_foliage_hull`,
    /// `crates/raa-lit/src/foliage_stochastic.rs:74 build_stochastic_foliage`,
    /// `crates/raa-lit/src/foliage_optimize.rs:1045 optimize_foliage` and
    /// `crates/raa-lit/src/handle.rs:2151 LitHandle::set_foliage`.
    ///
    /// Handoff: 'Expose asynchronous foliage preparation to JavaScript' calls
    /// [`LitRenderer::prepare_foliage`] with the FoliageSource it resolved through
    /// `LitHandle::foliage_source` (handoff at `crates/raa-lit/src/handle.rs` on
    /// `LitHandle::foliage_source`) and [`LitRenderer::release_foliage`] before releasing
    /// the material lease; today they publish and retire one generation as described.
    pub fn prepare_foliage(
        &mut self,
        gpu: &GpuContext,
        source: FoliageSource,
        settings: FoliageBakeSettings,
    ) -> impl std::future::Future<Output = Result<FoliageAsset, RaaError>> + 'static + use<> {
        let checked = (|| {
            let route = settings.validate()?;
            crate::foliage_bake::validate_source(&source, &settings.params)?;
            check_material(&self.shared.borrow(), settings.material)?;
            Ok(route)
        })();
        if checked.is_ok() {
            self.foliage_bakes
                .borrow_mut()
                .begin(settings.material, settings.params.generation);
        }
        let gpu = gpu.clone();
        let shared = Rc::clone(&self.shared);
        let state = Rc::clone(&self.foliage_bakes);
        let lit = self.handle();
        async move {
            let route = checked?;
            let completed = bake_foliage(gpu.clone(), lit.clone(), source, settings, route).await?;
            publish_foliage(&gpu, &shared, &lit, &state, &settings, completed)
        }
    }

    /// Retires `asset`: detaches its material association, removes its proxy mesh, then
    /// drops its resource generation (submitted work keeps it alive until completion). The
    /// borrowed material and its data block are never removed.
    pub fn release_foliage(&mut self, asset: FoliageAsset) {
        let lit = self.handle();
        let shared = Rc::clone(&self.shared);
        let mesh = asset.mesh;
        retire_foliage(
            asset.resources,
            || {
                let _ = lit.set_foliage(asset.material, None);
            },
            || {
                let mut s = shared.borrow_mut();
                s.meshes.remove(mesh);
                s.remove_mesh_colors(mesh);
            },
        );
        self.submeshes.remove(&mesh);
    }

    /// Registers the sub-meshes of proxy meshes published since the last frame.
    fn refresh_published_foliage(&mut self) {
        let published = self.foliage_bakes.borrow_mut().take_published();
        for id in published {
            self.refresh_submeshes(id);
        }
    }

    // ---- billboard bake ----

    /// Bakes the submitted draw groups of `input.tags` into the eight-direction atlas
    /// images and returns them with the source-root framing.
    ///
    /// The source draws are the complete render list entries carrying one of the tags
    /// (no frustum culling and no on-screen visibility requirement). They are enumerated
    /// with [`Scene::build_all_render_list_slots`], so retained instances hidden in the
    /// active view (including those hidden with
    /// [`Scene::set_instance_visible(false)`](Scene::set_instance_visible)) still
    /// contribute; the bake never changes visibility. Retained LOD0 meshes stay valid off
    /// screen because the scene's full sub-mesh is drawn, not a selected LOD.
    /// Instance models become `root_world^-1 * world`, so positions and the shader's
    /// object normals share the source-root frame. Every tag must name at least one
    /// submitted draw and the resolution must be 64, 128, 256 or 512; otherwise the call
    /// refuses `billboard.invalid` before any GPU work.
    ///
    /// The bake uses private uniform, material, extras, instance and id buffers with the
    /// renderer's own frame layout and atlas bindings, so the active view's camera and
    /// prepared instance records are untouched. Per direction it creates an orthographic
    /// [`billboard_camera`], clears diffuse/normal/opacity to zero and depth to one,
    /// renders every source draw in LOD0 order and calls
    /// [`BillboardTargets::encode_tile`](crate::BillboardTargets) after that view. After all
    /// views it submits the work, publishes the three atlas images
    /// ([`BillboardTargets::submit_publish`](crate::BillboardTargets)) and only then releases the
    /// temporary targets. Submission is synchronous; the returned future owns all GPU
    /// handles and holds no renderer or scene borrow. The scene camera is untouched; a failed
    /// publication removes its partial atlas allocations. The returned atlas ids stay live
    /// until the caller removes them with [`LitRenderer::remove_texture`].
    ///
    /// Handoff: 'Expose asynchronous billboard baking in RAA' calls
    /// [`LitRenderer::bake_billboard`] with a [`BillboardBakeInput`] and keeps the ids of
    /// its [`BillboardBakeResult`] until it removes them with
    /// [`LitRenderer::remove_texture`]. Today the method renders the submitted groups in
    /// the eight orthographic directions, publishes the three atlas images and returns the
    /// common framing and tile layout; a failure leaves no atlas allocation behind.
    pub fn bake_billboard(
        &mut self,
        gpu: &GpuContext,
        scene: &mut Scene,
        input: BillboardBakeInput,
    ) -> impl std::future::Future<Output = Result<BillboardBakeResult, RaaError>> + 'static + use<>
    {
        let submitted = self.submit_billboard(gpu, scene, input);
        async move { submitted?.await }
    }

    fn submit_billboard(
        &mut self,
        gpu: &GpuContext,
        scene: &mut Scene,
        input: BillboardBakeInput,
    ) -> Result<
        impl std::future::Future<Output = Result<BillboardBakeResult, RaaError>> + 'static + use<>,
        RaaError,
    > {
        // Refuse bad input before touching the GPU.
        crate::billboard_layout(input.resolution)?;
        if input.tags.is_empty() {
            return Err(bake_error(
                "a billboard bake needs at least one submitted draw tag".into(),
            ));
        }
        let root_inverse = input.root_world.inverse().ok_or_else(|| {
            bake_error("the billboard source-root frame is not invertible".into())
        })?;
        let mut requested: Vec<u32> = Vec::with_capacity(input.tags.len());
        for &tag in &input.tags {
            if !requested.contains(&tag) {
                requested.push(tag);
            }
        }

        scene.update_world();
        let mut list = RenderList::with_capacity(scene.instance_count());
        let mut item_tags = Vec::with_capacity(scene.instance_count());
        let mut item_slots = Vec::with_capacity(scene.instance_count());
        scene.build_all_render_list_slots(&mut list, &mut item_tags, &mut item_slots);
        let handle = self.handle();

        let mut sources: Vec<BakeSource> = Vec::new();
        let mut bounds = Aabb::EMPTY;
        let mut material_count = 1u32;
        let mut seen = vec![false; requested.len()];
        for (i, item) in list.items.iter().enumerate() {
            let tag = item_tags[i];
            let Some(group) = requested.iter().position(|&t| t == tag) else {
                continue;
            };
            seen[group] = true;
            let model = root_inverse * item.world;
            let mut record = [0u8; INSTANCE_GPU_SIZE];
            write_instance_record(&mut record, &model, item.material.0, tag, 0);
            let extra = self.extras.get(&tag).copied().unwrap_or_default();
            let item_bounds = handle
                .mesh_bounds(item.mesh)
                .map(|b| b.transform(&model))
                .unwrap_or_else(|| {
                    let p = model.transform_point(Vec3::ZERO);
                    Aabb { min: p, max: p }
                });
            bounds = bounds.union(&item_bounds);
            material_count = material_count.max(item.material.0.saturating_add(1));
            sources.push(BakeSource {
                mesh: item.mesh,
                material: item.material.0,
                record,
                extra,
            });
        }
        for (i, &tag) in requested.iter().enumerate() {
            if !seen[i] {
                return Err(bake_error(format!(
                    "billboard tag {tag} has no submitted draw"
                )));
            }
        }
        if sources.is_empty() {
            return Err(bake_error("the billboard bake has no source draws".into()));
        }
        let framing = billboard_framing(&bounds)
            .ok_or_else(|| bake_error("the billboard source bounds are empty".into()))?;

        // Material records and their draw class, in the bake's own buffer.
        let mut material_bytes =
            Vec::with_capacity(material_count as usize * SURFACE_MATERIAL_GPU_SIZE);
        let mut material_classes = Vec::with_capacity(material_count as usize);
        {
            let materials = self.materials();
            for id in 0..material_count {
                let record = materials.get(MaterialId(id)).cloned().unwrap_or_default();
                material_classes.push((
                    MaterialClass::of(&record).material_fn,
                    record.flags & FLAG_DOUBLE_SIDED != 0,
                ));
                record.write_gpu(&mut material_bytes);
            }
        }

        let frame_layout = handle.frame_layout();
        let lighting_layout = handle.lighting_layout();
        let mut pipelines: HashMap<(u32, bool), wgpu::RenderPipeline> = HashMap::new();
        for &(fn_id, double_sided) in &material_classes {
            if pipelines.contains_key(&(fn_id.0, double_sided)) {
                continue;
            }
            let key = VariantKey {
                material_fn: fn_id,
                pass: PassKind::Prepass,
                msaa: 1,
                flags: if double_sided { FLAG_DOUBLE_SIDED } else { 0 },
            };
            let wgsl = {
                let composer = self.composer();
                let fns = self.fns();
                composer.compose_billboard_bake(&fns, &key)
            }
            .map_err(bake_shader_error)?;
            Composer::validate(&wgsl).map_err(bake_shader_error)?;
            let module = gpu
                .device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("raa.lit.billboard_bake"),
                    source: wgpu::ShaderSource::Wgsl(wgsl.into()),
                });
            let pipeline =
                bake_pipeline(gpu, &module, &frame_layout, &lighting_layout, double_sided);
            pipelines.insert((fn_id.0, double_sided), pipeline);
        }

        let mut targets = BillboardTargets::new(gpu, input.resolution)?;
        let mut instance_bytes = Vec::with_capacity(sources.len() * INSTANCE_GPU_SIZE);
        let mut extra_bytes = Vec::with_capacity(sources.len() * INSTANCE_EXTRA_GPU_SIZE);
        let mut id_bytes = Vec::with_capacity(sources.len() * 4);
        for (i, src) in sources.iter().enumerate() {
            instance_bytes.extend_from_slice(&src.record);
            let mut extra = Vec::with_capacity(INSTANCE_EXTRA_GPU_SIZE);
            src.extra.write_gpu(&mut extra);
            extra_bytes.extend_from_slice(&extra);
            id_bytes.extend_from_slice(&(i as u32).to_le_bytes());
        }
        let buffers = SourceDrawBuffers::new(
            gpu,
            &handle,
            &material_bytes,
            &instance_bytes,
            &extra_bytes,
            &id_bytes,
        );
        let lighting = {
            let pipeline = self.pipeline.try_borrow().map_err(|_| pipeline_busy())?;
            pipeline.lighting().bind_group().clone()
        };

        let render = (|| -> Result<(), RaaError> {
            let mut encoder =
                gpu.device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("raa.lit.billboard_bake"),
                    });
            for direction in 0..crate::billboard_bake::BILLBOARD_DIRECTIONS as u8 {
                let camera = billboard_camera(framing, direction)?;
                let mut frame_scene = Scene::new();
                frame_scene.camera = camera;
                frame_scene.light = scene.light;
                frame_scene.ambient = scene.ambient;
                let mut frame_bytes = [0u8; FRAME_GPU_SIZE];
                write_frame_gpu(
                    &mut frame_bytes,
                    &frame_scene,
                    &FrameParams {
                        viewport: (input.resolution, input.resolution),
                        msaa: 1,
                        ..FrameParams::default()
                    },
                )?;
                gpu.queue().write_buffer(&buffers.frame, 0, &frame_bytes);
                {
                    let a = targets.attachments();
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("raa.lit.billboard_bake.render"),
                        color_attachments: &[
                            bake_color(a.diffuse),
                            bake_color(a.normal),
                            bake_color(a.opacity),
                        ],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: a.depth,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_bind_group(0, &buffers.bind, &[]);
                    pass.set_bind_group(1, &lighting, &[]);
                    for (i, src) in sources.iter().enumerate() {
                        let (fn_id, double_sided) = material_classes[src.material as usize];
                        let pipeline = pipelines
                            .get(&(fn_id.0, double_sided))
                            .expect("every material class built its pipeline");
                        pass.set_pipeline(pipeline);
                        let count = handle.bake_bind_mesh(&mut pass, src.mesh, &buffers.ids)?;
                        pass.draw_indexed(0..count, 0, i as u32..i as u32 + 1);
                    }
                }
                targets.encode_tile(&mut encoder, direction)?;
            }
            gpu.queue().submit([encoder.finish()]);
            Ok(())
        })();
        if let Err(e) = render {
            targets.release();
            return Err(e);
        }

        // The atlas never leaves the renderer while the GPU is pending. Rebuild group 0
        // immediately if publication allocated it, so normal frames keep their textures.
        let (images, done) = self
            .shared
            .borrow_mut()
            .submit_billboard_publication(gpu, &mut targets)?;
        let publication = BakePublication {
            shared: Rc::clone(&self.shared),
            gpu: gpu.clone(),
            images: Some([images.diffuse, images.normal, images.opacity]),
            _targets: targets,
        };
        Ok(async move {
            done.await?;
            publication.complete();
            Ok(BillboardBakeResult {
                diffuse: images.diffuse,
                normal: images.normal,
                opacity: images.opacity,
                center: framing.center,
                size: framing.size,
                tiles: images.tiles,
                extent: images.extent,
            })
        })
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
        self.shared
            .borrow_mut()
            .mirror_instance_extra(tag, Some(&x));
        self.extras.insert(tag, x);
    }

    /// The extras of tag `tag` (S10 defaults when none were set).
    pub fn instance_extra(&self, tag: u32) -> InstanceExtra {
        self.extras.get(&tag).copied().unwrap_or_default()
    }

    /// Returns tag `tag` to the default extras.
    pub fn clear_instance_extra(&mut self, tag: u32) {
        self.shared.borrow_mut().mirror_instance_extra(tag, None);
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
        self.add_texture_with(gpu, width, height, data, TextureOptions::default())
    }

    /// [`LitRenderer::add_texture`] with explicit [`TextureOptions`]: same id rule and
    /// refusals (`texture.invalid`, `texture.atlas_full`, `pipeline.busy`; nothing changes).
    /// With `options.mips` it uploads levels 1..`texture_levels(w, h)` made by
    /// `downsample_rgba8(.., options.srgb)`; with `mips: false` only level 0, and the rect
    /// width is negated so the shader samples level 0 only.
    ///
    /// Handoff: "raa-js m22: addTexture mips/srgb options, docs and typings" calls
    /// `LitRenderer::add_texture_with(&mut self, gpu: &GpuContext, width: u32, height: u32,
    /// data: TexelData<'_>, options: TextureOptions) -> Result<u32, RaaError>` from
    /// `addTexture`, passing the options it read from JS (it builds `TextureOptions` per the
    /// handoff at textures.rs `TextureOptions`). The 5-level atlas texture is created by the
    /// first `TextureAtlas::add` (the placeholder before it has 1 level). No later call
    /// replaces or drops it; it drops with the TextureAtlas. The handle's `add_texture`
    /// rebuilds the group-0 bind group when `add` returns created = true. No pass resize or
    /// prepare hook touches the atlas, which is reached only through the frame bind group.
    pub fn add_texture_with(
        &mut self,
        gpu: &GpuContext,
        width: u32,
        height: u32,
        data: TexelData<'_>,
        options: TextureOptions,
    ) -> Result<u32, RaaError> {
        let mut s = self.shared.try_borrow_mut().map_err(|_| pipeline_busy())?;
        s.add_texture(gpu, width, height, data, options)
    }

    /// Removes texture `id`; its id then samples (0, 0, 0, 0). Idempotent: an unknown or
    /// removed id does nothing. Returns whether it was live.
    pub fn remove_texture(&mut self, gpu: &GpuContext, id: u32) -> Result<bool, RaaError> {
        let mut s = self.shared.try_borrow_mut().map_err(|_| pipeline_busy())?;
        Ok(s.remove_texture(gpu, id))
    }

    /// The same material atlas with an sRGB view for base-colour ray sampling.
    pub fn trace_texture_atlas(&self) -> wgpu::TextureView {
        self.shared.borrow().trace_texture_atlas()
    }

    /// Atlas region and content generation for a live material texture.
    pub fn texture_content(&self, id: u32) -> Option<(AtlasRect, u64)> {
        self.shared.borrow().texture_content(id)
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

    // ---- area-light soft shadows (M4) ----

    /// The accepted area-light soft-shadow settings, `AreaShadowSettings::default()` until a
    /// [`LitRenderer::set_area_shadows`] call is accepted.
    ///
    /// Handoff: card 'raa-js: setAreaLightShadows and areaLightShadows on the m28 light API'
    /// calls [`LitRenderer::area_shadows`], [`LitRenderer::set_area_shadows`] and
    /// [`LitRenderer::active_area_heroes`]. LIFECYCLE: `LitRenderer::new` creates the four
    /// per-view `HeroSelector`s and `LitShared` construction sets `area_heroes` to all `None`
    /// and `area_hero_capacity` to 0; `LitRenderer::prepare` rewrites the kind-3 lanes,
    /// `area_heroes` and `area_hero_capacity` on every prepare, after `publish_local_lights`
    /// and before the frame graph runs (`ShadowPass::prepare`, order 20, reads them in the same
    /// frame, followed by the order-35 `light.tiles` prepare, which uploads `inputs.lights`
    /// when it is dirty). Nothing drops them while the renderer lives.
    pub fn area_shadows(&self) -> AreaShadowSettings {
        self.area_shadow_settings
    }

    /// Validates and stores `settings`; a refusal changes nothing.
    ///
    /// Handoff: card 'raa-shadow: hero shadow-map layers rendered by shadow.csm' calls
    /// [`LitRenderer::set_area_shadows`] and [`LitRenderer::prepare`]; its hero targets come
    /// from the `Handoff` at [`LitHandle::area_hero_targets`] in
    /// `crates/raa-lit/src/handle.rs:1192`. LIFECYCLE: `LitRenderer::new` creates the four
    /// per-view `HeroSelector`s and `LitShared` construction sets `area_heroes` to all `None`
    /// and `area_hero_capacity` to 0; `LitRenderer::prepare` rewrites the kind-3 lanes,
    /// `area_heroes` and `area_hero_capacity` on every prepare, after `publish_local_lights`
    /// and before the frame graph runs (`ShadowPass::prepare`, order 20, reads them in the same
    /// frame, followed by the order-35 `light.tiles` prepare, which uploads `inputs.lights`
    /// when it is dirty). Nothing drops them while the renderer lives.
    pub fn set_area_shadows(&mut self, settings: AreaShadowSettings) -> Result<(), RaaError> {
        self.area_shadow_settings = settings.validate()?;
        Ok(())
    }

    /// Heroes occupied by the most recently prepared view (0 before any prepare).
    pub fn active_area_heroes(&self) -> u32 {
        self.active_area_heroes
    }

    // ---- area-light stochastic shadows (M6) ----

    /// Validates and stores `settings`; a refusal changes nothing.
    ///
    /// Handoff: card 'raa-js: stochastic keys of setAreaLightShadows/areaLightShadows, docs and typings, and the GI trace-scene wiring' calls [`LitRenderer::set_area_stochastic`],
    /// [`LitRenderer::area_stochastic`] and [`LitRenderer::active_area_stochastic`]. Today
    /// `set_area_stochastic` stores `settings.validate()?` and leaves the current settings in
    /// place on refusal, `area_stochastic` returns [`StochasticShadowSettings::standard`]
    /// until the first accepted call, and `active_area_stochastic` returns the occupied
    /// mid-tier slots of the most recently prepared view (0 before any prepare).
    pub fn set_area_stochastic(
        &mut self,
        settings: StochasticShadowSettings,
    ) -> Result<(), RaaError> {
        self.area_stochastic_settings = settings.validate()?;
        Ok(())
    }

    /// The accepted mid-tier stochastic shadow settings,
    /// [`StochasticShadowSettings::standard`] until a [`LitRenderer::set_area_stochastic`] call
    /// is accepted.
    pub fn area_stochastic(&self) -> StochasticShadowSettings {
        self.area_stochastic_settings
    }

    /// Mid-tier slots occupied by the most recently prepared view (0 before any prepare).
    pub fn active_area_stochastic(&self) -> u32 {
        self.active_area_stochastic
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

    /// Configures the lit forward pass's optional planar mirror receiver.
    pub fn set_planar_reflection(&mut self, reflection: Option<crate::PlanarReflection>) {
        let mut shared = self.shared.borrow_mut();
        shared.planar_reflection = reflection;
        if reflection.is_none() {
            shared.planar_view = None;
        }
    }

    /// Sets logical `FrameGpu` feature `id` (S10) for every later `prepare`, after its own
    /// tiling bit. Ignores reserved ids 24 (tiling) and 40 (planar capture) and ids >= 64.
    /// Sets the water body lanes written to `FrameGpu.reserved` by every later `prepare`
    /// (raa_shader::WaterBody::frame_lanes; all zero = no water).
    pub fn set_water_frame(&mut self, lanes: [f32; 16]) {
        self.water = lanes;
    }

    /// The water lanes of the next frames.
    pub fn water_frame(&self) -> [f32; 16] {
        self.water
    }

    pub fn set_frame_feature(&mut self, id: u32, on: bool) {
        if matches!(
            id,
            TILING_FEATURE_ID | crate::planar_reflection::CAPTURE_FEATURE
        ) || id >= 64
        {
            return;
        }
        let bit = 1u64 << id;
        if on {
            self.frame_features |= bit;
        } else {
            self.frame_features &= !bit;
        }
    }

    /// Selects forward lightingOutput "layers" / "layers-signals" / "layers-interlaced" for
    /// every later camera forward recording ([`ForwardLayers`](crate::ForwardLayers)). The
    /// modes need a forward pass of 1 sample: with MSAA the forward record refuses them.
    pub fn set_forward_layers(&mut self, mode: crate::ForwardLayers) {
        self.shared.borrow_mut().forward_layers = mode;
    }

    /// The forward layers mode set by [`set_forward_layers`](Self::set_forward_layers).
    pub fn forward_layers(&self) -> crate::ForwardLayers {
        self.shared.borrow().forward_layers
    }

    /// MSAA sample count of the forward pass (1 or 4).
    pub fn forward_msaa(&self) -> u32 {
        self.shared.borrow().msaa()
    }

    /// The forward "layers-interlaced" state of the latest scene invocation whose frame was
    /// submitted (`None` before one): mode, real-frame counter n, R/G class, phase and the
    /// signal passes it skipped. The state of a frame is fixed by its prepare.
    pub fn lighting_interlace(&self) -> Option<LightingInterlace> {
        self.interlace_submitted
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
        assert!(view < MAX_VIEWS, "viewport index must fit the history bank");
        self.scene_frame.selected = Some((view, discontinuity));
    }

    /// Acknowledges a really submitted view (never a mere prepare); only these commit.
    pub fn submitted_scene_view(&mut self, view: usize) {
        self.scene_frame.submitted(view);
    }

    /// Overrides the legacy TAA-driven jitter policy. `None` uses the pipeline's TAA setting.
    pub fn set_camera_jitter(&mut self, enabled: Option<bool>) {
        self.camera_jitter = enabled;
    }

    /// Sets the extra material texture mip bias of jittered frames (finite, clamped to
    /// [-2, 0]); unjittered frames never take it.
    pub fn set_jitter_texture_lod_bias(&mut self, bias: f32) {
        self.jitter_lod_bias = if bias.is_finite() { bias.clamp(-2.0, 0.0) } else { 0.0 };
    }

    /// Fixes the Halton jitter cycle length (1..=64); 0 restores the automatic length
    /// (the base period at 1:1, 8 by default and 15 with pass.lighting.reconstruct, times the
    /// display/render area ratio when upscaling).
    pub fn set_jitter_phases(&mut self, phases: u32) {
        self.jitter_phases = (phases > 0).then(|| phases.min(64));
    }

    /// Sets the signal rates of forward "layers-interlaced" from the next prepare on (default:
    /// s, AO and E at the phase pixel of R frames). Set by pass.lighting.reconstruct while it is
    /// installed.
    pub fn set_interlace_signals(&mut self, signals: InterlaceSignals) {
        self.interlace_signals = signals;
    }

    /// The signal rates set by [`set_interlace_signals`](Self::set_interlace_signals).
    pub fn interlace_signals(&self) -> InterlaceSignals {
        self.interlace_signals
    }

    /// Sets the 1:1 base length of the Halton jitter sequence (default 8, see `frame_history`);
    /// the automatic upscaled cycle scales it, an explicit `set_jitter_phases` overrides it.
    pub fn set_camera_jitter_period(&mut self, period: u32) {
        crate::frame_history::set_jitter_period(period);
    }

    /// The length of the Halton jitter sequence.
    pub fn camera_jitter_period(&self) -> u32 {
        crate::frame_history::jitter_period()
    }

    /// Drops the camera history for a viewport absent from the rendered layout.
    pub fn suspend_scene_view(&mut self, view: usize) {
        if view < MAX_VIEWS {
            self.scene_frame.history.suspend(view);
        }
    }

    /// Closes the invocation and commits its acknowledged instances. The pipeline's submission
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
        if self.scene_frame.acknowledged != 0 {
            self.interlace_submitted = Some(self.interlace_prepared);
        }
        self.scene_frame.finish();
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
    ///
    /// Direct callers acknowledge the submitted FrameGraph command buffer with
    /// `FrameGraph::notify_submitted` before preparing another scene frame.
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
        if self.scene_frame.implicit {
            let selected = self.scene_frame.selected.take();
            let observation = self
                .pipeline
                .try_borrow()
                .map_err(|_| pipeline_busy())?
                .scene_submission();
            if observation.submitted_mask & 1 != 0 {
                self.submitted_scene_view(0);
            }
            self.finish_scene_frame(observation.submitted_mask & 1 != 0);
            self.scene_frame.selected = selected;
        }
        self.refresh_published_foliage();
        if let Some((index, observation, submitted)) = self.pending_resolve.take() {
            ReadbackCoordinator::resolve(&self.readback, gpu, index, observation, submitted);
        }
        let (viewport, display_size, encode_srgb, tiling, taa_enabled, readers) = {
            let p = self.pipeline.try_borrow().map_err(|_| pipeline_busy())?;
            (
                p.targets().map_or((1, 1), |t| t.size()),
                p.targets().map_or((1, 1), |t| t.display_rect().size),
                !p.surface_is_srgb(),
                p.entry_enabled(LIGHT_TILES_NAME).unwrap_or(false),
                jitter_consumer(p.entry_enabled(TAA_PASS_NAME)),
                SignalReaders::of(&p),
            )
        };
        if !self.scene_frame.open {
            self.pipeline
                .try_borrow_mut()
                .map_err(|_| pipeline_busy())?
                .begin_scene_submission();
            let selected = self.scene_frame.selected;
            self.scene_frame.begin(self.scene_frame.frame_index);
            self.scene_frame.selected = selected;
            self.scene_frame.implicit = true;
        }
        let jitter = self.camera_jitter.unwrap_or(taa_enabled);
        let discontinuity = self.scene_frame.selected.is_some_and(|(_, cut)| cut);
        self.scene_frame.history.jitter_phases = self
            .jitter_phases
            .unwrap_or_else(|| crate::frame_history::jitter_phases(viewport, display_size));
        let camera = self.scene_frame.prepare(scene, viewport, jitter);
        if discontinuity || camera.jitter_mode_changed {
            self.pipeline
                .try_borrow_mut()
                .map_err(|_| pipeline_busy())?
                .invalidate_current_history();
        }
        let mut guard = self.shared.try_borrow_mut().map_err(|_| pipeline_busy())?;
        let s: &mut LitShared = &mut guard;
        s.temporal_consumer = taa_enabled || jitter;
        s.camera_jitter = camera.current_jitter;
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
        let select_all = s.select_all.get();
        if select_all {
            self.visible_bits.fill(u64::MAX);
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
                return select_all || crate::skin_bounds::visible(bounds, item.world, vp);
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
        // This selection starts from the whole scene list, independently of the
        // main camera's visible slots and LODs.
        let planar_reflection = s.planar_reflection;
        let planar_view = if let Some(reflection) = planar_reflection {
            let mut view = s
                .planar_view
                .take()
                .unwrap_or_else(|| crate::planar_reflection::PlanarView::new(gpu, self.opts));
            let capture = |i: usize| {
                tags[i] != reflection.instance
                    && !s.lods.is_level(list.items[i].mesh)
                    && self
                        .material_classes
                        .get(list.items[i].material.0 as usize)
                        .is_some_and(|material| material.class != DrawClass::Transparent)
            };
            let capture_input = SelectInput {
                selected: &capture,
                level_mesh: &|_, mesh| mesh,
                ..input
            };
            view.select_scene(&input, &capture_input, &skinned, reflection, scene.camera)?;
            Some(view)
        } else {
            None
        };

        let mut stats = InstanceFrameStats {
            instances: scene.instance_count() as u32,
            camera: s.select.camera_count(),
            ..InstanceFrameStats::default()
        };
        for id in s.select.id_bytes().as_chunks::<4>().0 {
            let slot = u32::from_le_bytes(*id) as usize;
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

        let interlace = interlace_state(
            s.forward_layers == crate::ForwardLayers::Interlaced && s.msaa() == 1,
            camera.frame_index,
            self.interlace_signals,
            readers,
            s.planar_reflection.is_some(),
        );
        self.interlace_prepared = interlace;
        self.pipeline
            .try_borrow_mut()
            .map_err(|_| pipeline_busy())?
            .lighting_mut()
            .set_interlace(interlace);

        let [active_lights, tile_columns, tile_rows, tile_stride] =
            light_info(
                self.local_lights.gpu_count(),
                viewport,
                tiling,
                self.local_lights.forward_budget(),
            );
        let mut params = FrameParams {
            viewport,
            encode_srgb,
            msaa: self.opts.msaa,
            frame_index: camera.frame_index,
            camera_jitter_ndc: camera.current_jitter,
            texture_lod_bias: crate::frame::taau_texture_lod_bias(jitter, viewport, display_size)
                + if jitter { self.jitter_lod_bias } else { 0.0 },
            previous_jitter_ndc: camera.previous_jitter,
            previous_view_proj: camera.previous_view_proj,
            history_valid: camera.history_valid,
            accumulation_enabled: taa_enabled,
            frame_parity: (camera.frame_index & 1) as u32,
            interlace: interlace.word(),
            active_lights,
            tile_columns,
            tile_rows,
            tile_stride,
            water: self.water,
            ..FrameParams::default()
        };
        use raa_shader::{RAA_DIRECT_ONLY_FEATURE, RAA_INDIRECT_ONLY_FEATURE};
        params.set_feature(TILING_FEATURE_ID, tiling);
        let mut lighting_view = None;
        for id in (0..64u32).filter(|&id| self.frame_features & (1u64 << id) != 0) {
            if matches!(id, RAA_DIRECT_ONLY_FEATURE | RAA_INDIRECT_ONLY_FEATURE) {
                // Both set: forward's "shadow" isolation view (see opaque_lighting.wgsl).
                lighting_view = Some(lighting_view.unwrap_or(0u64) | (1u64 << id));
            } else {
                params.set_feature(id, true);
            }
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
            // Area-light hero tiers: rewrite the kind-3 lanes, the hero targets and the
            // capacity of this view before the shadow pass reads them.
            let view = self.scene_frame.current_view.min(MAX_VIEWS - 1);
            let settings = self.area_shadow_settings;
            let camera = AreaShadowCamera {
                eye: scene.camera.eye,
                forward: (scene.camera.target - scene.camera.eye).normalize_or_zero(),
                fov_y_rad: scene.camera.fov_y_rad,
                viewport,
            };
            let mut heroes = [None; MAX_HERO_LIGHTS];
            if let Some(selector) = self.hero_selectors.get_mut(view) {
                let changed = selector.update(
                    &settings,
                    &camera,
                    &self.local_lights,
                    &mut inputs.lights,
                    &mut heroes,
                );
                inputs.lights_dirty |= changed;
                self.active_area_heroes = selector.active_heroes();
            }
            s.area_heroes = heroes;
            s.area_hero_capacity = match settings.tier {
                AreaShadowTier::Auto | AreaShadowTier::Hero => settings.hero_count,
                AreaShadowTier::Cone | AreaShadowTier::Fade => 0,
            };
            // Own maps of the `shadowMode: 'map'` point/spot lights, in the layers after the
            // area heroes; their lights leave the traced pass's slot list this frame.
            if let Some(selector) = self.local_map_selectors.get_mut(view) {
                let slots = self.local_lights.shadow_slots();
                let area = s.area_hero_capacity.min(crate::MAX_HERO_LIGHTS as u32);
                let mut targets = [None; crate::MAX_LOCAL_MAP_LAYERS];
                let (capacity, changed, traced) = selector.update(
                    &camera,
                    &self.local_lights,
                    &slots,
                    area,
                    &mut inputs.lights,
                    &mut targets,
                );
                inputs.lights_dirty |= changed;
                inputs.shadow_slots = traced;
                s.local_maps = targets;
                s.local_map_capacity = capacity;
            }
            // Mid-tier slots follow the hero lanes in the same light records.
            let mut stochastic = AreaStochasticFrame::empty();
            if let Some(selector) = self.stochastic_selectors.get_mut(view) {
                selector.update(
                    &settings,
                    &self.area_stochastic_settings,
                    &camera,
                    &self.local_lights,
                    &inputs.lights,
                    &mut stochastic,
                );
                self.active_area_stochastic = selector.active();
            }
            s.area_stochastic = stochastic;
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
        if let Some(mut view) = planar_view {
            view.upload(gpu, s, scene, params, &self.frame_bytes)?;
            s.planar_view = Some(view);
        }
        if let Some(feature) = lighting_view {
            let buffer = s.lighting_view_frame.get_or_insert_with(|| {
                gpu.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("raa.lit.forward.lighting-view.frame"),
                    size: FRAME_GPU_SIZE as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            });
            let mut preview = self.frame_bytes;
            for id in [RAA_DIRECT_ONLY_FEATURE, RAA_INDIRECT_ONLY_FEATURE] {
                if feature & (1u64 << id) != 0 {
                    params.set_feature(id, true);
                }
            }
            write_frame_flags(&mut preview, &params);
            gpu.queue().write_buffer(buffer, 0, &preview);
        } else {
            s.lighting_view_groups = None;
            s.lighting_view_frame = None;
        }
        stats.uploaded_bytes = s.store.last_uploaded_bytes();
        s.stats = stats;
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
            shadow_only: false,
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
    fn jitter_consumer_needs_an_enabled_taa_entry() {
        assert_eq!(TAA_PASS_NAME, "taa");
        assert!(!jitter_consumer(None), "no taa entry");
        assert!(!jitter_consumer(Some(false)), "a disabled taa entry");
        assert!(jitter_consumer(Some(true)), "an enabled taa entry");
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

    /// The sub-mesh ids of a removed mesh are reused by the next upload: the detail chain
    /// of the removed mesh, or a chain drawing it as a level, must not survive on them
    /// (Game view, 2026-11-20: `render.unknown_mesh: mesh 6 is not uploaded`).
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn removing_a_mesh_drops_the_detail_chains_that_name_it() {
        use raa_geom::GeometryProvider;
        let Some(gpu) = raa_core::testing::software_gpu() else {
            raa_core::testing::skip("removing_a_mesh_drops_lods", "no adapter");
            return;
        };
        let pipeline = Rc::new(RefCell::new(Pipeline::new(
            &gpu,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        )));
        let mut lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).unwrap();
        let cube = || {
            let mut mesh = raa_geom::MeshData::new();
            let params = raa_geom::Params::new().with("half", raa_geom::ParamValue::Scalar(0.5));
            raa_geom::builtin::DebugCube
                .generate(&params, raa_geom::Quality::LOD0, &mut mesh)
                .unwrap();
            AttributedMesh::from_mesh(mesh)
        };
        let first = |lit: &LitRenderer, id| lit.submesh_ids(id).unwrap()[0];
        let full = lit.upload(&gpu, &cube()).unwrap();
        let level = lit.upload(&gpu, &cube()).unwrap();
        let (full_sub, level_sub) = (first(&lit, full), first(&lit, level));
        let chain = |mesh| [LodLevel { mesh, error: 1.0 }];
        lit.set_mesh_lods(full_sub, &chain(level_sub)).unwrap();
        lit.remove(full);
        let next = lit.upload(&gpu, &cube()).unwrap();
        assert_eq!(first(&lit, next), full_sub, "the freed id is reused");
        assert!(!lit.has_mesh_lods(full_sub));
        // A removed level takes the chain that draws it with it.
        lit.set_mesh_lods(full_sub, &chain(level_sub)).unwrap();
        lit.remove(level);
        assert!(!lit.has_mesh_lods(full_sub));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn pending_billboard_bake_leaves_state_and_atlas_available() {
        use raa_core::testing::{block_on, skip, software_gpu};
        let Some(gpu) = software_gpu() else {
            skip(
                "pending_billboard_bake_leaves_state_and_atlas_available",
                "no adapter",
            );
            return;
        };
        let pipeline = Rc::new(RefCell::new(Pipeline::new(
            &gpu,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        )));
        let mut lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).unwrap();
        let mesh = AttributedMesh::from_mesh(crate::billboard_proxy([0.0; 3], [2.0; 2]).unwrap());
        let uploaded = lit.upload(&gpu, &mesh).unwrap();
        let mesh = lit.submesh_ids(uploaded).unwrap()[0];
        let mut scene = Scene::new();
        scene.camera.eye = Vec3::new(0.0, 0.0, 6.0);
        let node = scene
            .add_node(None, raa_scene::Transform::IDENTITY)
            .unwrap();
        let instance = scene.add_instance(node, mesh, MaterialId(0)).unwrap();
        scene.set_instance_tag(instance, 7).unwrap();
        let state = Rc::new(RefCell::new(Some((lit, scene))));
        fn owned<F: std::future::Future + 'static>(future: F) -> F {
            future
        }
        let submit = || {
            let mut state = state.borrow_mut();
            let (lit, scene) = state.as_mut().unwrap();
            owned(lit.bake_billboard(
                &gpu,
                scene,
                BillboardBakeInput {
                    tags: vec![7],
                    root_world: Mat4::IDENTITY,
                    resolution: 64,
                },
            ))
        };
        let pending = submit();
        let atlas_before = {
            let mut guard = state.try_borrow_mut().expect("bake holds no state borrow");
            let (lit, scene) = guard.as_mut().expect("state stays in its cell");
            let mut list = RenderList::with_capacity(1);
            let counts = lit
                .prepare(&gpu, scene, &mut list, &mut Vec::new(), &mut Vec::new())
                .unwrap();
            assert_eq!(counts.instances, 1, "the normal LOD0 view still prepares");
            assert_eq!(lit.shared.borrow().texture_packer().len(), 3);
            lit.shared.borrow().texture_packer().clone()
        };
        let result = block_on(pending).expect("bake completes after other state access");
        let mut guard = state.borrow_mut();
        let (lit, _) = guard.as_mut().unwrap();
        for id in [result.diffuse, result.normal, result.opacity] {
            assert_eq!(
                lit.texture_rect(id),
                atlas_before.rect(id),
                "atlas stays installed"
            );
            lit.remove_texture(&gpu, id).unwrap();
        }
        drop(guard);
        // Cancellation also releases submitted atlas allocations without taking state.
        drop(submit());
        assert_eq!(
            state
                .borrow()
                .as_ref()
                .unwrap()
                .0
                .shared
                .borrow()
                .texture_packer()
                .len(),
            0
        );
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
            let history = self.lifecycle.prepare(scene, LIFE_SIZE, false);
            self.store
                .sync(&self.list, &self.tags, &self.slots, &HashMap::new(), &|_| 0);
            let recorded = &mut self.lifecycle.slots[view % MAX_VIEWS];
            recorded.clear();
            recorded.extend(self.slots.iter().copied());
            history
        }

        fn finish(&mut self) {
            self.lifecycle.finish();
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
        frame.finish();
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
        frame.finish();
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
            frame.finish();
        }
        assert_eq!(previous_tx(frame.store.previous_bytes(), 0), 4.0);
    }
}
