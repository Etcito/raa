//! The frame state shared by the virtualized-geometry passes and the `vg.cull`, `vg.raster`
//! and `vg.resolve` feature passes of the main view (S32, S12 orders 21-23). This share is the
//! vertical slice's first phase: the main view, the first cull phase, no occlusion test, and
//! the hardware rasteriser plus the software rasteriser of the small clusters.
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

use crate::culling::{CullPhase, CullPipelines, CullWork, MAIN_FIRST, PassView, ViewSlots};
use crate::raster::{RasterBindings, RasterTarget, Rasterizer, VisibilityBuffer};
use crate::resolve::{Resolve, VG_FRAME_GPU_SIZE, VgFrameParams, VgSurface, write_vg_frame};
use crate::scene::VgScene;
use crate::settings::VgSettings;

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
}

/// Everything sized by the viewport, recreated on a resize or a capacity change.
pub(crate) struct VgResources {
    pub(crate) size: (u32, u32),
    pub(crate) capacity: u32,
    pub(crate) slots: ViewSlots,
    pub(crate) main: CullWork,
    pub(crate) visibility: VisibilityBuffer,
    pub(crate) surface: VgSurface,
    pub(crate) frame: wgpu::Buffer,
    pub(crate) groups: Option<VgGroups>,
    pub(crate) bound: (u64, u64),
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
    #[allow(dead_code, reason = "the second occlusion-phase share reads it")]
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
    pub(crate) recorded: Vec<&'static str>,
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
            recorded: Vec::new(),
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
            self.resources = Some(VgResources {
                size,
                capacity,
                slots: ViewSlots::new(gpu),
                main: CullWork::new(
                    gpu,
                    &pipelines.cull,
                    "raa-vg main",
                    u64::from(capacity),
                    u64::from(capacity),
                ),
                visibility: VisibilityBuffer::new(gpu, size),
                surface: VgSurface::new(gpu, size),
                frame: gpu.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("raa-vg frame"),
                    size: VG_FRAME_GPU_SIZE as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                groups: None,
                bound: (u64::MAX, u64::MAX),
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
        let cull = pipelines.cull.scene_bind_group(
            gpu,
            &resources.slots,
            &self.scene,
            &pipelines.cull.placeholder_pyramid,
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
            cull,
            raster,
            resolve,
        });
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
            total += resources.surface.allocated_bytes();
            total += resources.frame.size();
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

    /// Handoff: `raa-vg: second occlusion phase` takes
    /// `VgCullPass::prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError>` and
    /// `VgCullPass::record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError>`, the main
    /// `CullWork` (`VgResources::main`) and the `previous_view_proj`/`occlusion_history` fields from
    /// crates/raa-vg/src/frame_passes.rs; `PassView`, `ViewSlots` and `MAIN_SECOND` are imported
    /// from crates/raa-vg/src/culling.rs, not exported here. It attaches by extending those two
    /// methods in this file, where `CullPhase::First` is the only phase today, and by setting
    /// `view.occlusion_enabled` from `settings.occlusion`; state/keys: `previous_view_proj`,
    /// `occlusion_history`, `MAIN_SECOND` and the second-phase `CullPhase`.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let mut state = self.state.borrow_mut();
        if !state.active {
            return Ok(());
        }
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
        view.occlusion_enabled = 0;
        view.allow_software = u32::from(state.settings.software_raster);
        resources.slots.write(ctx.gpu, MAIN_FIRST, &view);
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

    fn prepare(&mut self, _ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let mut state = self.state.borrow_mut();
        if state.active {
            state.recorded.clear();
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
    /// skips while a scene buffer is not uploaded. State/keys: `settings.software_raster`, the
    /// main `CullWork` indirect arguments and the `VgResources` registry (visibility, main,
    /// groups.raster).
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let mut state = self.state.borrow_mut();
        if !state.active || !state.scene_ready() {
            return Ok(());
        }
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
        state.recorded.push(Self::NAME);
        Ok(())
    }
}

/// `vg.resolve` (order 23): writes the 320-byte `VgFrameGpu` record and records the resolve
/// dispatch, then makes this frame's view the previous one.
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
        let size = state.size();
        let view_proj = state.camera.view_proj();
        {
            let Some(resources) = state.resources.as_ref() else {
                return Ok(());
            };
            let Some(groups) = resources.groups.as_ref() else {
                return Ok(());
            };
            let Some(pipelines) = state.pipelines.as_ref() else {
                return Ok(());
            };
            pipelines.resolve.record(ctx.encoder, &groups.resolve, size);
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
