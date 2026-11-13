//! The frame state shared by the virtualized-geometry passes and the `vg.cull`, `vg.raster`
//! and `vg.resolve` feature passes of the main view (S32, S12 orders 21-23). The main view
//! runs two occlusion phases: the first tests the previous frame's merged depth pyramid and
//! the second re-tests the retries against this frame's visibility keys merged with the
//! previous frame's forward depth while the camera is unchanged (forward occluders are drawn by
//! `prepass` only after `vg.raster`), then the hardware and software rasterisers fill the
//! visibility buffer.
//!
//! The state owns the lit handle, the settings, the main camera and viewport, the scene pool,
//! the GPU pipelines and the surface-sized resources. [`VgFrameState::ensure_resources`]
//! creates whatever is missing (and is safe to call every frame), and
//! [`VgFrameState::release`] drops every vg buffer and texture so the disabled state owns
//! nothing (P3-Q8, S32).
//!
//!
//! no handoff: raa-vg: vg.shade forward contributor and VgShadeGpu slots takes nothing from this card

// The feature share that installs these passes lands after this card; until then the items
// are reached only by this module's tests.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError};
use raa_lit::{LitHandle, MaterialFnId};
use raa_math::Mat4;
use raa_pipeline::{FeaturePass, PrepareCtx, RecordCtx, Targets};
use raa_scene::Camera;
use raa_shadow::{CascadeTargets, ShadowHandle};

use crate::culling::{
    CASCADE_BASE, CullPhase, CullPipelines, CullWork, MAIN_FIRST, MAIN_SECOND, PassView, ViewSlots,
};
use crate::hzb::Hzb;
use crate::raster::{RasterBindings, RasterTarget, Rasterizer, VisibilityBuffer};
use crate::resolve::{Resolve, VG_FRAME_GPU_SIZE, VgFrameParams, VgSurface, write_vg_frame};
use crate::scene::VgScene;
use crate::settings::VgSettings;
use crate::shadow::{CascadeWork, cascade_views, record_cascades};
use crate::stats::{VgCullTotals, VgStatsReadback};

/// The frame state shared between the virtualized-geometry passes and their handle.
///
/// Handoff: `raa-vg: vg.gbuffer merge into depth and G-buffer` takes
/// `pub(crate) type SharedVg = Rc<RefCell<VgFrameState>>` and
/// `pub(crate) fn ensure_resources(&mut self, gpu: &GpuContext) -> ()` exported by
/// crates/raa-vg/src/frame_passes.rs; it attaches by holding a `SharedVg` and installing its
/// order-32 `FeaturePass` beside `VgCullPass`, `VgRasterPass` and `VgResolvePass`. In this file
/// it reads `VgFrameState::scene`, `VgFrameState::lit` and `VgFrameState::material_fns`, plus the
/// resolve outputs in `VgResources::surface` (written by `VgResolvePass::record`) with
/// `VgResources::visibility` and `VgResources::main`; state/keys: the `VgResources` registry and
/// the lit material storage.
pub(crate) type SharedVg = Rc<RefCell<VgFrameState>>;

/// The builders and GPU objects of the three passes, created on the first active frame.
pub(crate) struct VgPipelines {
    pub(crate) cull: CullPipelines,
    pub(crate) raster: Rasterizer,
    pub(crate) resolve: Resolve,
    /// `merge_forward_occluders` of [`occluder_keys_module_wgsl`], run before the second phase.
    pub(crate) occluders: wgpu::ComputePipeline,
}

/// Bytes of the `OccluderParams` uniform of [`occluder_keys_module_wgsl`].
const OCCLUDER_PARAMS_SIZE: u64 = 16;

/// The body of the second-phase occluder merge: per pixel, this frame's visibility key, or the
/// nearer key of the previous frame's forward surface when `use_forward` is set. A previous
/// pixel counts as forward where no virtualized surface was resolved (`ids.x == 0`) or where the
/// merged depth is nearer than the resolved virtualized depth (`ids.w`), so a forward surface
/// won the depth test of `vg.gbuffer`; where the virtualized surface won, the forward depth
/// behind it is unknown and nothing is merged. Keys order like `raa_depth_key` (larger is
/// nearer, 0 is far or empty), so `max` keeps the nearer one, as `vis_write.wgsl` does.
const OCCLUDER_KEYS_WGSL: &str = r#"
struct OccluderParams {
    size: vec2u,
    use_forward: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: OccluderParams;
@group(0) @binding(1) var<storage, read> vis_depth: array<u32>;
@group(0) @binding(2) var<storage, read_write> occluder_keys: array<u32>;
@group(0) @binding(3) var previous_depth: texture_depth_2d;
@group(0) @binding(4) var previous_ids: texture_2d<u32>;

@compute @workgroup_size(8, 8)
fn merge_forward_occluders(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.size.x || id.y >= params.size.y {
        return;
    }
    let index = id.y * params.size.x + id.x;
    var key = vis_depth[index];
    if params.use_forward != 0u {
        let pixel = vec2i(id.xy);
        let merged = textureLoad(previous_depth, pixel, 0);
        let ids = textureLoad(previous_ids, pixel, 0);
        if ids.x == 0u || raa_depth_nearer(merged, bitcast<f32>(ids.w)) {
            key = max(key, raa_depth_key(merged));
        }
    }
    occluder_keys[index] = key;
}
"#;

/// The second-phase occluder merge module: `DEPTH_WGSL` and [`OCCLUDER_KEYS_WGSL`]. Its one
/// group (0) is private to the pass, like the occlusion pyramid modules.
pub(crate) fn occluder_keys_module_wgsl() -> String {
    [raa_shader::DEPTH_WGSL, OCCLUDER_KEYS_WGSL].concat()
}

/// The labelled compute pipeline of [`occluder_keys_module_wgsl`], with its layout derived
/// from the module.
fn occluder_pipeline(gpu: &GpuContext) -> wgpu::ComputePipeline {
    let module = gpu
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-vg occluder keys module"),
            source: wgpu::ShaderSource::Wgsl(occluder_keys_module_wgsl().into()),
        });
    gpu.device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raa-vg merge_forward_occluders"),
            layout: None,
            module: &module,
            entry_point: Some("merge_forward_occluders"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
}

/// The occluder keys the second phase's pyramid is seeded from, sized by the viewport.
pub(crate) struct OccluderKeys {
    /// One key per pixel, written by `merge_forward_occluders`.
    pub(crate) keys: wgpu::Buffer,
    /// `OccluderParams`: the viewport and the `use_forward` flag.
    pub(crate) params: wgpu::Buffer,
    /// The `use_forward` flag last written into `params`.
    pub(crate) use_forward: Option<bool>,
    /// The bind group and the `Targets.depth` view it reads, rebuilt when the targets change.
    pub(crate) group: Option<(wgpu::TextureView, wgpu::BindGroup)>,
}

impl OccluderKeys {
    fn new(gpu: &GpuContext, size: (u32, u32)) -> OccluderKeys {
        let keys = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-vg occluder keys"),
            size: u64::from(size.0) * u64::from(size.1) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let params = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-vg occluder params"),
            size: OCCLUDER_PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        OccluderKeys {
            keys,
            params,
            use_forward: None,
            group: None,
        }
    }

    fn allocated_bytes(&self) -> u64 {
        self.keys.size() + self.params.size()
    }
}

/// Everything sized by the viewport, recreated on a resize or a capacity change.
pub(crate) struct VgResources {
    pub(crate) size: (u32, u32),
    pub(crate) capacity: u32,
    pub(crate) slots: ViewSlots,
    pub(crate) main: CullWork,
    pub(crate) visibility: VisibilityBuffer,
    /// The previous frame's merged depth, the first phase's occlusion source.
    pub(crate) history: Hzb,
    /// This frame's visibility keys and forward occluders, the second phase's occlusion source.
    pub(crate) current: Hzb,
    /// The cull scene group bound by the first phase: `PassView` over `history`.
    pub(crate) cull_history: wgpu::BindGroup,
    /// The cull scene group bound by the second phase: `PassView` over `current`.
    pub(crate) cull_current: wgpu::BindGroup,
    pub(crate) surface: VgSurface,
    pub(crate) frame: wgpu::Buffer,
    /// The per-cascade cull work and stamped depth, at `shadow.csm`'s cascade resolution.
    pub(crate) cascades: Option<CascadeWork>,
    /// The `VgScene` generation the cascade bind groups were built for.
    pub(crate) cascade_bound: Option<u64>,
    pub(crate) groups: Option<VgGroups>,
    pub(crate) bound: (u64, u64),
    /// The asynchronous cull counters of this frame path, captured by `vg.resolve`.
    pub(crate) stats: VgStatsReadback,
    /// The keys `VgResources::current` is seeded from before the second phase.
    pub(crate) occluders: OccluderKeys,
}

/// The group-2 bind groups of the three passes, rebuilt when the scene or lit storage changes.
pub(crate) struct VgGroups {
    pub(crate) cull: wgpu::BindGroup,
    pub(crate) raster: RasterBindings,
    pub(crate) resolve: wgpu::BindGroup,
}

/// The live state of one virtualized-geometry frame path.
pub(crate) struct VgFrameState {
    pub(crate) lit: LitHandle,
    pub(crate) settings: VgSettings,
    pub(crate) active: bool,
    pub(crate) camera: Camera,
    pub(crate) viewport: (u32, u32),
    /// True while the previous frame's occlusion pyramid may be read.
    pub(crate) occlusion_history: bool,
    pub(crate) previous_view_proj: Option<Mat4>,
    pub(crate) scene: VgScene,
    /// The scene's material functions in use, ascending; the gbuffer and shade shares read it.
    #[allow(
        dead_code,
        reason = "filled by the feature share; the gbuffer share reads it"
    )]
    pub(crate) material_fns: Vec<MaterialFnId>,
    pub(crate) pipelines: Option<VgPipelines>,
    pub(crate) resources: Option<VgResources>,
    /// The `shadow.csm` handle the cascade views and layers come from; `None` when the feature
    /// was installed without one.
    pub(crate) shadow: Option<ShadowHandle>,
    /// The cascade views `VgCullPass::prepare` wrote into `CASCADE_BASE..` this frame.
    pub(crate) cascade_views: [Option<PassView>; 4],
    /// The stamps `CascadeWork::next_stamps` handed out for this frame's cascade views.
    pub(crate) cascade_stamps: [u32; 4],
    /// The cascades `shadow.csm` prepared this frame; `None` when shadows are off or absent.
    pub(crate) cascade_targets: Option<CascadeTargets>,
    pub(crate) recorded: Vec<&'static str>,
    /// The newest published cull totals. Kept outside `VgResources`, so they survive a release.
    pub(crate) published: Option<VgCullTotals>,
}

impl VgFrameState {
    /// An inactive state: no pipeline and no resource exist until the first
    /// [`ensure_resources`](Self::ensure_resources).
    pub(crate) fn new(lit: LitHandle) -> VgFrameState {
        VgFrameState {
            lit,
            settings: VgSettings::default(),
            active: false,
            camera: Camera::default(),
            viewport: (1, 1),
            occlusion_history: false,
            previous_view_proj: None,
            scene: VgScene::new(),
            material_fns: Vec::new(),
            pipelines: None,
            resources: None,
            shadow: None,
            cascade_views: [None; 4],
            cascade_stamps: [0; 4],
            cascade_targets: None,
            recorded: Vec::new(),
            published: None,
        }
    }

    /// The viewport with zero dimensions clamped to one, so no buffer or texture is empty.
    fn size(&self) -> (u32, u32) {
        (self.viewport.0.max(1), self.viewport.1.max(1))
    }

    /// True when every scene buffer the raster and resolve groups bind is uploaded. The
    /// placeholder stand-ins keep the groups valid but a draw or dispatch bound to one fails
    /// wgpu validation, so a pass skips until the scene is uploaded.
    fn scene_ready(&self) -> bool {
        self.scene.instances().is_some()
            && self.scene.nodes().is_some()
            && self.scene.clusters().is_some()
            && self.scene.vertices().is_some()
            && self.scene.triangles().is_some()
    }

    /// Creates the pipelines when absent, the surface-sized resources when absent or when the
    /// viewport or `maxVisibleClusters` changed (clearing the occlusion history), and the
    /// group-2 bind groups when the scene or the lit storage generation changed.
    ///
    /// Handoff: `raa-vg: VgFeature::install and VgHandle (settings, instances, active state,
    /// prepare_frame)` takes `VgFrameState::new`, `VgFrameState::ensure_resources`,
    /// `VgFrameState::release`, `VgFrameState::allocated_bytes` and the constructors
    /// `VgCullPass::new(state: SharedVg)`, `VgRasterPass::new(state: SharedVg)` and
    /// `VgResolvePass::new(state: SharedVg)` exported by crates/raa-vg/src/frame_passes.rs; it
    /// attaches by holding a `SharedVg`, writing `VgFrameState::active`, `camera`, `viewport`,
    /// `settings`, `previous_view_proj` and `scene`, and registering the three pass structs
    /// `VgCullPass`, `VgRasterPass` and `VgResolvePass`. `recorded` is cleared only on ACTIVE
    /// frames, by each pass's own `prepare`: `VgCullPass::prepare`, `VgRasterPass::prepare` and
    /// `VgResolvePass::prepare`, up to three times per active frame. Each `prepare` returns early
    /// on `!VgFrameState::active` before the clear, and raa-pipeline skips `prepare` for disabled
    /// entries, so on an inactive frame the names from the last active frame stay until
    /// `VgFrameState::release` clears them; the VgFeature card must check `VgFrameState::active`
    /// (or rely on `release`) before reading `recorded`. State/keys: the
    /// `VgPipelines`/`VgResources`/`VgGroups` registry, the `VgScene` pool and the lit storage
    /// generation.
    pub(crate) fn ensure_resources(&mut self, gpu: &GpuContext) {
        if self.pipelines.is_none() {
            self.pipelines = Some(VgPipelines {
                cull: CullPipelines::new(gpu),
                raster: Rasterizer::new(gpu),
                resolve: Resolve::new(gpu),
                occluders: occluder_pipeline(gpu),
            });
        }
        let size = self.size();
        let capacity = self.settings.max_visible_clusters;
        let recreate = self
            .resources
            .as_ref()
            .is_none_or(|resources| resources.size != size || resources.capacity != capacity);
        if recreate {
            self.occlusion_history = false;
            let Some(pipelines) = self.pipelines.as_ref() else {
                return;
            };
            let slots = ViewSlots::new(gpu);
            let history = Hzb::new(gpu, size);
            let current = Hzb::new(gpu, size);
            let cull_history =
                pipelines
                    .cull
                    .scene_bind_group(gpu, &slots, &self.scene, history.view());
            let cull_current =
                pipelines
                    .cull
                    .scene_bind_group(gpu, &slots, &self.scene, current.view());
            self.resources = Some(VgResources {
                size,
                capacity,
                slots,
                main: CullWork::new(
                    gpu,
                    &pipelines.cull,
                    "raa-vg main",
                    u64::from(capacity),
                    u64::from(capacity),
                ),
                visibility: VisibilityBuffer::new(gpu, size),
                history,
                current,
                cull_history,
                cull_current,
                surface: VgSurface::new(gpu, size),
                frame: gpu.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("raa-vg frame"),
                    size: VG_FRAME_GPU_SIZE as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                cascades: None,
                cascade_bound: None,
                groups: None,
                bound: (u64::MAX, u64::MAX),
                stats: VgStatsReadback::new(gpu),
                occluders: OccluderKeys::new(gpu, size),
            });
        }
        let bound = (self.scene.generation(), self.lit.storage_generation());
        let rebuild = self
            .resources
            .as_ref()
            .is_some_and(|resources| resources.groups.is_none() || resources.bound != bound);
        if !rebuild {
            return;
        }
        let materials = self.lit.material_storage();
        let Some(pipelines) = self.pipelines.as_ref() else {
            return;
        };
        let Some(resources) = self.resources.as_mut() else {
            return;
        };
        let cull_history = pipelines.cull.scene_bind_group(
            gpu,
            &resources.slots,
            &self.scene,
            resources.history.view(),
        );
        let cull_current = pipelines.cull.scene_bind_group(
            gpu,
            &resources.slots,
            &self.scene,
            resources.current.view(),
        );
        let raster = pipelines.raster.bindings(
            gpu,
            &resources.slots,
            MAIN_FIRST,
            &self.scene,
            &resources.main,
            &resources.visibility,
        );
        let resolve = pipelines.resolve.bindings(
            gpu,
            &resources.frame,
            &resources.visibility,
            &resources.main,
            &self.scene,
            &materials,
            &resources.surface,
        );
        resources.groups = Some(VgGroups {
            cull: cull_history.clone(),
            raster,
            resolve,
        });
        resources.cull_history = cull_history;
        resources.cull_current = cull_current;
        resources.bound = bound;
    }

    /// Drops every vg buffer and texture, including the pipelines and their placeholders, and
    /// clears the material functions: the next [`ensure_resources`](Self::ensure_resources)
    /// recreates them. The scene keeps only its CPU layout.
    pub(crate) fn release(&mut self) {
        self.resources = None;
        self.pipelines = None;
        self.scene.release();
        self.material_fns.clear();
        self.cascade_views = [None; 4];
        self.cascade_stamps = [0; 4];
        self.cascade_targets = None;
        self.recorded.clear();
        self.occlusion_history = false;
    }

    /// Every vg buffer and texture this state owns; 0 once [`release`](Self::release) ran.
    pub(crate) fn allocated_bytes(&self) -> u64 {
        let mut total = self.scene.allocated_bytes();
        if let Some(pipelines) = &self.pipelines {
            total += pipelines.cull.allocated_bytes();
            total += pipelines.raster.allocated_bytes();
            total += pipelines.resolve.allocated_bytes();
        }
        if let Some(resources) = &self.resources {
            total += resources.slots.allocated_bytes();
            total += resources.main.allocated_bytes();
            total += resources.visibility.allocated_bytes();
            total += resources.history.allocated_bytes();
            total += resources.current.allocated_bytes();
            total += resources.surface.allocated_bytes();
            total += resources.frame.size();
            total += resources.stats.allocated_bytes();
            total += resources.occluders.allocated_bytes();
            if let Some(cascades) = &resources.cascades {
                total += cascades.allocated_bytes();
            }
        }
        total
    }
}

/// `vg.cull` (order 21): writes the main view's first-phase `PassView` and records
/// [`CullPipelines::record`] with [`CullPhase::First`].
pub(crate) struct VgCullPass {
    state: SharedVg,
}

impl VgCullPass {
    pub(crate) const NAME: &'static str = "vg.cull";
    pub(crate) const ORDER: u32 = 21;

    pub(crate) fn new(state: SharedVg) -> VgCullPass {
        VgCullPass { state }
    }
}

impl FeaturePass for VgCullPass {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn order(&self) -> u32 {
        Self::ORDER
    }

    /// Nothing to drop: VgFrameState::ensure_resources recreates the surface-sized resources (and clears the occlusion history) when the viewport or capacity changes.
    fn resize(&mut self, _gpu: &GpuContext, _targets: &Targets) {}

    /// Writes `MAIN_FIRST` (phase 0, occlusion tested against `VgResources::history` when
    /// `settings.occlusion` and `VgFrameState::occlusion_history` hold, else disabled) and then
    /// `MAIN_SECOND`, the same view at phase 1 tested against `VgResources::current`.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let mut state = self.state.borrow_mut();
        if !state.active {
            return Ok(());
        }
        let state = &mut *state;
        state.ensure_resources(ctx.gpu);
        state.recorded.clear();
        let Some(resources) = state.resources.as_ref() else {
            return Ok(());
        };
        let view_proj = state.camera.view_proj();
        let previous = state.previous_view_proj.unwrap_or(view_proj);
        let projection = state.camera.projection();
        let size = state.size();
        let mut view = PassView::perspective(
            view_proj,
            previous,
            state.camera.eye,
            [size.0 as f32, size.1 as f32],
            [projection.cols[0][0], projection.cols[1][1]],
            state.camera.near,
            state.settings.error_pixels,
        );
        let occlusion = state.settings.occlusion && state.occlusion_history;
        view.occlusion_enabled = u32::from(occlusion);
        view.allow_software = u32::from(state.settings.software_raster);
        let history_size = resources.history.size();
        view.hzb_size = [history_size.0 as f32, history_size.1 as f32];
        resources.slots.write(ctx.gpu, MAIN_FIRST, &view);

        let mut second = view;
        second.phase = 1;
        second.occlusion_view = view_proj;
        let current_size = resources.current.size();
        second.hzb_size = [current_size.0 as f32, current_size.1 as f32];
        resources.slots.write(ctx.gpu, MAIN_SECOND, &second);

        // The cascade views of this frame, when `shadow.csm` (order 20, prepared before this
        // pass) handed out its targets. Without shadows or targets no cascade work happens.
        state.cascade_views = [None; 4];
        state.cascade_stamps = [0; 4];
        state.cascade_targets = None;
        if state.settings.shadows
            && let Some(targets) = state
                .shadow
                .as_ref()
                .and_then(|shadow| shadow.cascade_targets())
        {
            let views = cascade_views(&targets, &state.camera, size.1 as f32, &state.settings);
            let resolution = targets.resolution;
            let stamps = {
                let Some(pipelines) = state.pipelines.as_ref() else {
                    return Ok(());
                };
                let Some(resources) = state.resources.as_mut() else {
                    return Ok(());
                };
                if resources
                    .cascades
                    .as_ref()
                    .is_none_or(|cascades| cascades.resolution() != resolution)
                {
                    resources.cascades = Some(CascadeWork::new(
                        ctx.gpu,
                        &pipelines.cull,
                        &pipelines.raster,
                        resolution,
                        u64::from(resources.capacity),
                    ));
                    resources.cascade_bound = None;
                }
                let Some(cascades) = resources.cascades.as_mut() else {
                    return Ok(());
                };
                cascades.next_stamps()
            };
            let Some(resources) = state.resources.as_ref() else {
                return Ok(());
            };
            for (index, view) in views.iter().enumerate() {
                if let Some(mut view) = *view {
                    view.stamp = stamps[index];
                    resources
                        .slots
                        .write(ctx.gpu, CASCADE_BASE + index as u32, &view);
                }
            }
            let scene_generation = state.scene.generation();
            let Some(pipelines) = state.pipelines.as_ref() else {
                return Ok(());
            };
            let Some(resources) = state.resources.as_mut() else {
                return Ok(());
            };
            if resources.cascade_bound != Some(scene_generation) {
                let Some(cascades) = resources.cascades.as_mut() else {
                    return Ok(());
                };
                cascades.rebind(
                    ctx.gpu,
                    &pipelines.cull,
                    &pipelines.raster,
                    &resources.slots,
                    &state.scene,
                );
                resources.cascade_bound = Some(scene_generation);
            }
            state.cascade_targets = Some(targets);
            state.cascade_views = views;
            state.cascade_stamps = stamps;
        }
        Ok(())
    }

    /// Handoff: `raa-vg: cascade integration into vg.cull and vg.raster, feature id 36` takes
    /// `VgCullPass::prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError>`,
    /// `VgCullPass::record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError>`,
    /// `VgRasterPass::record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError>` and the
    /// `VgResources` registry exported by crates/raa-vg/src/frame_passes.rs; it attaches by adding
    /// the cascade `CullWork`s and bind groups to `VgResources` and recording them from those
    /// methods in this file. State/keys: the `CASCADE_BASE`..`PassView` slots and `CullPhase` from
    /// crates/raa-vg/src/culling.rs, `settings.shadows`, the cascade `CullWork`s and feature id 36.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let mut state = self.state.borrow_mut();
        if !state.active || !state.scene_ready() {
            return Ok(());
        }
        let state = &mut *state;
        let Some(resources) = state.resources.as_ref() else {
            return Ok(());
        };
        let Some(groups) = resources.groups.as_ref() else {
            return Ok(());
        };
        let Some(pipelines) = state.pipelines.as_ref() else {
            return Ok(());
        };
        pipelines.cull.record(
            ctx.encoder,
            &state.scene,
            &groups.cull,
            MAIN_FIRST,
            &resources.main,
            CullPhase::First,
        );
        if let Some(cascades) = resources.cascades.as_ref()
            && state.cascade_targets.is_some()
        {
            cascades.record_cull(
                ctx.encoder,
                &pipelines.cull,
                &state.scene,
                &groups.cull,
                &state.cascade_views,
            );
        }
        state.recorded.push(Self::NAME);
        Ok(())
    }
}

/// `vg.raster` (order 22): clears the visibility buffers and records the hardware raster of
/// the first-phase list.
pub(crate) struct VgRasterPass {
    state: SharedVg,
}

impl VgRasterPass {
    pub(crate) const NAME: &'static str = "vg.raster";
    pub(crate) const ORDER: u32 = 22;

    pub(crate) fn new(state: SharedVg) -> VgRasterPass {
        VgRasterPass { state }
    }
}

impl FeaturePass for VgRasterPass {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn order(&self) -> u32 {
        Self::ORDER
    }

    /// Nothing to drop: VgFrameState::ensure_resources recreates the surface-sized resources (and clears the occlusion history) when the viewport or capacity changes.
    fn resize(&mut self, _gpu: &GpuContext, _targets: &Targets) {}

    /// Prepares the occluder merge of the second phase: the previous frame's forward depth
    /// (`Targets.depth` before `prepass` clears it) is merged only while the camera is the one
    /// that frame drew with and the targets have the viewport's size, because the depth is then
    /// already in this frame's screen space. A forward occluder that moved since keeps
    /// its previous depth for one frame; a moving camera merges nothing, so the second phase
    /// then tests the virtualized keys alone.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let mut state = self.state.borrow_mut();
        if !state.active {
            return Ok(());
        }
        state.recorded.clear();
        let state = &mut *state;
        let use_forward = state.previous_view_proj == Some(state.camera.view_proj());
        let Some(pipelines) = state.pipelines.as_ref() else {
            return Ok(());
        };
        let Some(resources) = state.resources.as_mut() else {
            return Ok(());
        };
        let use_forward = use_forward && ctx.targets.size() == resources.size;
        let depth = ctx.targets.depth();
        let occluders = &mut resources.occluders;
        if occluders.use_forward != Some(use_forward) {
            let mut bytes = [0u8; OCCLUDER_PARAMS_SIZE as usize];
            bytes[0..4].copy_from_slice(&resources.size.0.to_le_bytes());
            bytes[4..8].copy_from_slice(&resources.size.1.to_le_bytes());
            bytes[8..12].copy_from_slice(&u32::from(use_forward).to_le_bytes());
            ctx.gpu.queue().write_buffer(&occluders.params, 0, &bytes);
            occluders.use_forward = Some(use_forward);
        }
        if occluders
            .group
            .as_ref()
            .is_none_or(|(view, _)| view != depth)
        {
            let group = ctx
                .gpu
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("raa-vg occluder keys"),
                    layout: &pipelines.occluders.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: occluders.params.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: resources.visibility.depth().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: occluders.keys.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(depth),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(resources.surface.ids()),
                        },
                    ],
                });
            occluders.group = Some((depth.clone(), group));
        }
        Ok(())
    }

    /// Handoff: `VgRasterPass::record` is the share of card 'raa-vg: software rasterisation of
    /// small clusters in the frame' in this file: it calls
    /// `Rasterizer::record(&self, encoder: &mut wgpu::CommandEncoder, bindings: &RasterBindings,
    /// work: &CullWork, target: RasterTarget<'_>) -> ()` and `raster_sw_module_wgsl() -> String`
    /// (crates/raa-vg/src/raster.rs), whose earlier-landed Handoff (raster.rs:60, on
    /// `raster_sw_module_wgsl`) this share read and follows: `Rasterizer::record` in raster.rs
    /// records the software dispatch at `CullWork::software_args_offset()` after the hardware
    /// draw. In this file `VgRasterPass::record` calls it with `RasterTarget::Visibility`, and
    /// `VgCullPass::prepare` sets `view.allow_software` from `settings.software_raster`; a pass
    /// skips while a scene buffer is not uploaded. With `settings.occlusion` and
    /// `VgFrameState::occlusion_history` it then seeds `VgResources::current` from
    /// `VgResources::occluders` (the first raster's visibility depth, merged by
    /// `merge_forward_occluders` with the previous frame's forward depth while the camera is
    /// unchanged) and records the second cull phase through
    /// `VgResources::cull_current` before rasterising the recovered clusters again. State/keys:
    /// `settings.software_raster`, the main `CullWork` indirect arguments and the `VgResources`
    /// registry (visibility, main, history, current, groups.raster).
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let mut state = self.state.borrow_mut();
        if !state.active || !state.scene_ready() {
            return Ok(());
        }
        let state = &mut *state;
        let Some(resources) = state.resources.as_ref() else {
            return Ok(());
        };
        let Some(groups) = resources.groups.as_ref() else {
            return Ok(());
        };
        let Some(pipelines) = state.pipelines.as_ref() else {
            return Ok(());
        };
        resources.visibility.clear(ctx.encoder);
        pipelines.raster.record(
            ctx.encoder,
            &groups.raster,
            &resources.main,
            RasterTarget::Visibility(&resources.visibility),
        );
        if state.settings.occlusion && state.occlusion_history {
            // This frame's pyramid: the visibility keys, plus the forward occluders that
            // `prepass` (order 30) has not drawn yet this frame, taken from the previous
            // frame's depth while the camera is unchanged. Without them the second phase
            // recovered every cluster the first rejected behind forward geometry.
            if let Some((_, group)) = resources.occluders.group.as_ref() {
                let mut pass = ctx
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("raa-vg occluder keys"),
                        timestamp_writes: None,
                    });
                pass.set_pipeline(&pipelines.occluders);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(
                    resources.size.0.div_ceil(8),
                    resources.size.1.div_ceil(8),
                    1,
                );
                drop(pass);
                resources
                    .current
                    .record_from_keys(ctx.encoder, &resources.occluders.keys);
            } else {
                resources
                    .current
                    .record_from_keys(ctx.encoder, resources.visibility.depth());
            }
            pipelines.cull.record(
                ctx.encoder,
                &state.scene,
                &resources.cull_current,
                MAIN_SECOND,
                &resources.main,
                CullPhase::Second,
            );
            pipelines.raster.record(
                ctx.encoder,
                &groups.raster,
                &resources.main,
                RasterTarget::Visibility(&resources.visibility),
            );
        }
        if state.cascade_targets.is_some() {
            let Some(pipelines) = state.pipelines.as_ref() else {
                return Ok(());
            };
            let Some(resources) = state.resources.as_mut() else {
                return Ok(());
            };
            let Some(cascades) = resources.cascades.as_mut() else {
                return Ok(());
            };
            let Some(targets) = state.cascade_targets.as_ref() else {
                return Ok(());
            };
            record_cascades(
                ctx.encoder,
                &pipelines.raster,
                cascades,
                targets,
                &state.cascade_views,
                state.cascade_stamps,
            );
        }
        state.recorded.push(Self::NAME);
        Ok(())
    }
}

/// `vg.resolve` (order 23): writes the 320-byte `VgFrameGpu` record and records the resolve
/// dispatch, captures the cull counters of this frame, then makes this frame's view the previous
/// one.
pub(crate) struct VgResolvePass {
    state: SharedVg,
}

impl VgResolvePass {
    pub(crate) const NAME: &'static str = "vg.resolve";
    pub(crate) const ORDER: u32 = 23;

    pub(crate) fn new(state: SharedVg) -> VgResolvePass {
        VgResolvePass { state }
    }
}

impl FeaturePass for VgResolvePass {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn order(&self) -> u32 {
        Self::ORDER
    }

    /// Nothing to drop: VgFrameState::ensure_resources recreates the surface-sized resources (and clears the occlusion history) when the viewport or capacity changes.
    fn resize(&mut self, _gpu: &GpuContext, _targets: &Targets) {}

    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let mut state = self.state.borrow_mut();
        if !state.active {
            return Ok(());
        }
        state.recorded.clear();
        let Some(resources) = state.resources.as_ref() else {
            return Ok(());
        };
        let view_proj = state.camera.view_proj();
        let previous = state.previous_view_proj.unwrap_or(view_proj);
        let size = state.size();
        let width = size.0 as f32;
        let height = size.1 as f32;
        let params = VgFrameParams {
            view_proj,
            unjittered_view_proj: view_proj,
            previous_view_proj: previous,
            inv_view_proj: view_proj.inverse().unwrap_or(Mat4::IDENTITY),
            viewport: [width, height, 1.0 / width, 1.0 / height],
            camera_jitter: [0.0; 4],
            camera_pos: [
                state.camera.eye.x,
                state.camera.eye.y,
                state.camera.eye.z,
                0.0,
            ],
            debug_view: state.settings.debug_view.code(),
            history_valid: state.previous_view_proj.is_some(),
        };
        let mut bytes = [0u8; VG_FRAME_GPU_SIZE];
        write_vg_frame(&mut bytes, &params);
        ctx.gpu.queue().write_buffer(&resources.frame, 0, &bytes);
        Ok(())
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let mut state = self.state.borrow_mut();
        if !state.active || !state.scene_ready() {
            return Ok(());
        }
        let state = &mut *state;
        let size = state.size();
        let view_proj = state.camera.view_proj();
        {
            let Some(resources) = state.resources.as_mut() else {
                return Ok(());
            };
            let Some(groups) = resources.groups.as_ref() else {
                return Ok(());
            };
            let Some(pipelines) = state.pipelines.as_ref() else {
                return Ok(());
            };
            pipelines.resolve.record(ctx.encoder, &groups.resolve, size);
            // The cascade counters only when this frame recorded the cascades; their work
            // otherwise holds an earlier frame's counters.
            let cascades = if state.cascade_targets.is_some() {
                resources.cascades.as_ref().map(CascadeWork::works)
            } else {
                None
            };
            resources
                .stats
                .capture(ctx.encoder, &resources.main, cascades);
        }
        state.recorded.push(Self::NAME);
        state.previous_view_proj = Some(view_proj);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use raa_core::testing::{block_on, skip, software_gpu};
    use raa_lit::{LitOptions, LitRenderer};
    use raa_pipeline::{LightingResources, Pipeline};

    use super::*;

    /// The second-phase occluder merge composes into a valid module whose depth reasoning goes
    /// through the standard-Z helpers only.
    #[test]
    fn occluder_keys_module_validates() {
        let wgsl = occluder_keys_module_wgsl();
        assert_eq!(raa_shader::Composer::validate(&wgsl), Ok(()));
        assert!(OCCLUDER_KEYS_WGSL.contains("raa_depth_nearer(merged, bitcast<f32>(ids.w))"));
        assert!(OCCLUDER_KEYS_WGSL.contains("raa_depth_key(merged)"));
        for comparison in [" < ", " > ", "<=", ">=", "min("] {
            let body = OCCLUDER_KEYS_WGSL
                .lines()
                .filter(|line| !line.contains("params.size"))
                .collect::<String>();
            assert!(
                !body.contains(comparison),
                "hand-written depth compare {comparison}"
            );
        }
    }

    #[test]
    fn pass_names_and_orders() {
        assert_eq!(VgCullPass::NAME, "vg.cull");
        assert_eq!(VgCullPass::ORDER, 21);
        assert_eq!(VgRasterPass::NAME, "vg.raster");
        assert_eq!(VgRasterPass::ORDER, 22);
        assert_eq!(VgResolvePass::NAME, "vg.resolve");
        assert_eq!(VgResolvePass::ORDER, 23);
    }

    /// An inactive state records nothing and owns no resource; `ensure_resources` creates the
    /// 64x64 resources and `release` returns the allocation to zero. No instance is uploaded
    /// and nothing records into the encoder, so the depth buffers are never read.
    #[test]
    fn inactive_state_records_nothing_and_releases_every_resource() {
        let test = "inactive_state_records_nothing_and_releases_every_resource";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let pipeline = Rc::new(RefCell::new(Pipeline::new(
            &gpu,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        )));
        let renderer = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit");
        let shared: SharedVg = Rc::new(RefCell::new(VgFrameState::new(renderer.handle())));
        shared.borrow_mut().viewport = (64, 64);
        let mut cull = VgCullPass::new(Rc::clone(&shared));
        let mut raster = VgRasterPass::new(Rc::clone(&shared));
        let mut resolve = VgResolvePass::new(Rc::clone(&shared));
        let targets = Targets::new(&gpu, (64, 64));
        let lighting = LightingResources::new(&gpu);
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for pass in [&mut cull as &mut dyn FeaturePass, &mut raster, &mut resolve] {
            let mut ctx = RecordCtx {
                gpu: &gpu,
                encoder: &mut encoder,
                size: (64, 64),
                frame: 0,
                time_s: 0.0,
                targets: &targets,
                lighting: &lighting,
                backbuffer: targets.material(),
                backbuffer_format: wgpu::TextureFormat::Rgba8Unorm,
                surface_is_srgb: false,
            };
            pass.record(&mut ctx).expect("record");
        }
        assert!(shared.borrow().recorded.is_empty());
        shared.borrow_mut().ensure_resources(&gpu);
        assert!(shared.borrow().allocated_bytes() > 0);
        shared.borrow_mut().release();
        assert_eq!(shared.borrow().allocated_bytes(), 0);
        let _ = encoder.finish();
    }

    /// An active frame with resources but an empty scene records nothing: every pass skips
    /// while a scene buffer is missing, so the placeholder stand-ins never reach a draw or a
    /// dispatch and the validation scope stays clean.
    #[test]
    fn active_state_without_a_scene_records_nothing_without_validation_error() {
        let test = "active_state_without_a_scene_records_nothing_without_validation_error";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let pipeline = Rc::new(RefCell::new(Pipeline::new(
            &gpu,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        )));
        let renderer = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit");
        let shared: SharedVg = Rc::new(RefCell::new(VgFrameState::new(renderer.handle())));
        {
            let mut state = shared.borrow_mut();
            state.active = true;
            state.settings.software_raster = true;
            state.viewport = (64, 64);
            state.ensure_resources(&gpu);
        }
        assert!(shared.borrow().allocated_bytes() > 0);
        let mut cull = VgCullPass::new(Rc::clone(&shared));
        let mut raster = VgRasterPass::new(Rc::clone(&shared));
        let mut resolve = VgResolvePass::new(Rc::clone(&shared));
        let targets = Targets::new(&gpu, (64, 64));
        let mut lighting = LightingResources::new(&gpu);
        let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for pass in [&mut cull as &mut dyn FeaturePass, &mut raster, &mut resolve] {
            let mut prepare = PrepareCtx {
                gpu: &gpu,
                size: (64, 64),
                frame: 0,
                time_s: 0.0,
                dt_s: 0.0,
                targets: &targets,
                lighting: &mut lighting,
            };
            pass.prepare(&mut prepare).expect("prepare");
            let mut record = RecordCtx {
                gpu: &gpu,
                encoder: &mut encoder,
                size: (64, 64),
                frame: 0,
                time_s: 0.0,
                targets: &targets,
                lighting: &lighting,
                backbuffer: targets.material(),
                backbuffer_format: wgpu::TextureFormat::Rgba8Unorm,
                surface_is_srgb: false,
            };
            pass.record(&mut record).expect("record");
        }
        gpu.queue().submit(Some(encoder.finish()));
        let _ = gpu.device().poll(wgpu::PollType::wait_indefinitely());
        let error = block_on(errors.pop());
        assert!(error.is_none(), "empty-scene frame: {error:?}");
        assert!(shared.borrow().recorded.is_empty());
    }
}
