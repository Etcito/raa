//! The GI feature handle (S36, S37): the activity rule [`inactive_reason`], the crate-private
//! state [`GiCore`] and the cloneable [`GiHandle`] that shares it with the Engine.
//!
//! [`GiFeature::install`] builds the feature: the two passes `gi.volume` and `gi.history` over
//! the shared [`GiCore`], the built-in forward driver and the parts. [`GiHandle::prepare_frame`]
//! runs once per Engine frame before the pipeline executes (contract 'gi-frame-lifecycle').

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError};
use raa_ibl::IblHandle;
use raa_lit::LitRenderer;
use raa_math::Mat4;
use raa_pipeline::{FeaturePass, LightingResources, Pipeline, PrepareCtx, RecordCtx};
use raa_scene::{Camera, DirectionalLight, Scene};
use raa_vg::VgHandle;
use raa_vg::scene::{ItemScratch, SyncItem};

use crate::clipmap::{RebuildRounds, RelightCursor};
use crate::debug::DebugRenderer;
use crate::forward_driver::ForwardDriver;
use crate::layouts::GiFrameGpu;
use crate::membership::{InstanceGi, MembershipReport, plan_members};
use crate::parts::{GiPart, GiShared, PartFrame, PartsCtx, VoxelDriver};
use crate::settings::{
    DebugView, FINEST_VOXEL_METRES, GiQuality, GiQualityPatch, GiSettings, GiSettingsPatch,
};
use crate::stats::GiStats;
use crate::volume::{GiVolume, VolumeWork};

/// The pass names and orders of S12 (`gi.volume` 25, `gi.history` 64).
const VOLUME_PASS: &str = "gi.volume";
const VOLUME_ORDER: u32 = 25;
const HISTORY_PASS: &str = "gi.history";
const HISTORY_ORDER: u32 = 64;

/// The cap of `GiStats::frames_since_rebuild` (S36).
const MAX_FRAMES_SINCE_REBUILD: u32 = 1_000_000;

fn refusal(code: &'static str, message: String) -> RaaError {
    RaaError::FrameGraph { code, message }
}

fn pipeline_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the pipeline is borrowed (called while a frame executes)".to_string(),
    )
}

fn duplicate(name: &str) -> RaaError {
    refusal(
        "pipeline.duplicate_pass",
        format!("an entry named `{name}` is already in the pipeline"),
    )
}

/// The S36 rank table of `inactive_reason`: the first match wins.
pub fn inactive_reason(
    enabled: bool,
    volume_pass_enabled: bool,
    submitted_views: u8,
    members: u32,
) -> Option<&'static str> {
    if !enabled {
        Some("disabled")
    } else if !volume_pass_enabled {
        Some("passDisabled")
    } else if submitted_views != 1 {
        Some("multiView")
    } else if members == 0 {
        Some("noInstance")
    } else {
        None
    }
}

/// The statistics before any frame: inactive for `disabled`, at the given quality.
fn initial_stats(quality: GiQuality) -> GiStats {
    GiStats {
        active: false,
        inactive_reason: Some("disabled"),
        quality,
        finest_voxel_metres: FINEST_VOXEL_METRES,
        ..GiStats::default()
    }
}

/// Handoff: `GiCore` is the state that card "raa-gi: feature.rs share 2, passes, install and
/// prepare_frame" takes over. Today `GiCore::new` builds it with the defaults of contract
/// 'gi-core-state' and has no caller, and `GiHandle` reads and writes its settings, quality,
/// requests, view, sun, statistics and recorded passes; no frame runs yet. That card constructs
/// `GiCore` through `GiCore::new` (the install context it reuses is `parts.rs:50 PartsCtx::new`),
/// adds `GiFeature::install` and `GiHandle::prepare_frame`, and adds the two passes over the
/// fields `volume`, `rounds`, `cursor`, `work`, `prepared`, `published` and `pending`.
pub(crate) struct GiCore {
    settings: GiSettings,
    quality: GiQuality,
    allocated_quality: Option<GiQuality>,
    requests: BTreeMap<u32, bool>,
    camera: Camera,
    previous_view_proj: Option<Mat4>,
    viewport: (u32, u32),
    submitted_views: u8,
    sun_direction: [f32; 3],
    sun_color: [f32; 3],
    local_light_count: u32,
    history_valid: bool,
    /// Frames prepared so far; 0 before the first frame.
    frame_index: u64,
    shared: GiShared,
    volume: Option<GiVolume>,
    rounds: RebuildRounds,
    cursor: RelightCursor,
    work: Option<VolumeWork>,
    drivers: Vec<Rc<dyn VoxelDriver>>,
    parts: Vec<Rc<dyn GiPart>>,
    debug: DebugRenderer,
    scratch: ItemScratch,
    report: MembershipReport,
    signature: Option<u64>,
    frames_since_rebuild: u32,
    prepared: GiStats,
    published: Option<GiStats>,
    pending: bool,
    recorded: Vec<&'static str>,
    pipeline: Rc<RefCell<Pipeline>>,
}

impl GiCore {
    /// The state at construction: default settings and quality, no volume and no frame, one
    /// submitted view, the sun straight up, and `prepared` holding the initial statistics.
    pub(crate) fn new(
        _gpu: &GpuContext,
        pipeline: Rc<RefCell<Pipeline>>,
        shared: GiShared,
        drivers: Vec<Rc<dyn VoxelDriver>>,
        parts: Vec<Rc<dyn GiPart>>,
        debug: DebugRenderer,
    ) -> GiCore {
        GiCore {
            settings: GiSettings::default(),
            quality: GiQuality::default(),
            allocated_quality: None,
            requests: BTreeMap::new(),
            camera: Camera::default(),
            previous_view_proj: None,
            viewport: (0, 0),
            submitted_views: 1,
            sun_direction: [0.0, 1.0, 0.0],
            sun_color: [0.0; 3],
            local_light_count: 0,
            history_valid: false,
            frame_index: 0,
            shared,
            volume: None,
            rounds: RebuildRounds::new(5),
            cursor: RelightCursor::default(),
            work: None,
            drivers,
            parts,
            debug,
            scratch: ItemScratch::default(),
            report: MembershipReport::default(),
            signature: None,
            frames_since_rebuild: 0,
            prepared: initial_stats(GiQuality::default()),
            published: None,
            pending: false,
            recorded: Vec::new(),
            pipeline,
        }
    }
}

/// The cloneable handle of the GI feature: settings, quality, instance requests, the view and
/// sun, and the published statistics. Clones share one [`GiCore`].
#[derive(Clone)]
pub struct GiHandle {
    core: Rc<RefCell<GiCore>>,
}

impl GiHandle {
    /// Applies `patch` to the settings and publishes them to the shared frame state. A refusal
    /// changes nothing.
    pub fn set(&self, patch: GiSettingsPatch) -> Result<(), RaaError> {
        let mut core = self.core.borrow_mut();
        let next = patch.apply(&core.settings)?;
        core.settings = next;
        core.shared.set_settings(next);
        Ok(())
    }

    /// The current settings.
    pub fn get(&self) -> GiSettings {
        self.core.borrow().settings
    }

    /// Applies `patch` to the quality and publishes it. A change takes effect at the next active
    /// frame, which reallocates the volume. A refusal changes nothing.
    pub fn set_quality(&self, patch: GiQualityPatch) -> Result<(), RaaError> {
        let mut core = self.core.borrow_mut();
        let next = patch.apply(&core.quality)?;
        core.quality = next;
        core.shared.set_quality(next);
        Ok(())
    }

    /// The current quality.
    pub fn quality(&self) -> GiQuality {
        self.core.borrow().quality
    }

    /// Stores the request of `tag`; `None` removes it, so the group takes the default membership.
    pub fn set_instance(&self, tag: u32, request: Option<bool>) {
        let mut core = self.core.borrow_mut();
        match request {
            Some(value) => {
                core.requests.insert(tag, value);
            }
            None => {
                core.requests.remove(&tag);
            }
        }
    }

    /// Removes the request and the latest membership entry of `tag`.
    pub fn forget_instance(&self, tag: u32) {
        let mut core = self.core.borrow_mut();
        core.requests.remove(&tag);
        core.report.per_tag.remove(&tag);
    }

    /// The request of `tag` and its membership in the latest prepared frame. A tag with no entry,
    /// including every tag before a frame, is not voxelized and has no driver, code or message.
    pub fn instance(&self, tag: u32) -> InstanceGi {
        let core = self.core.borrow();
        let latest = core.report.per_tag.get(&tag).cloned().unwrap_or_default();
        InstanceGi {
            requested: core.requests.get(&tag).copied(),
            voxelized: latest.voxelized,
            driver: latest.driver,
            code: latest.code,
            message: latest.message,
        }
    }

    /// Stores the camera with its aspect set from `viewport`, the viewport and the number of
    /// submitted views.
    pub fn set_view(&self, camera: &Camera, viewport: (u32, u32), submitted_views: u8) {
        let mut core = self.core.borrow_mut();
        core.camera = *camera;
        core.camera.set_viewport(viewport.0, viewport.1);
        core.viewport = viewport;
        core.submitted_views = submitted_views;
    }

    /// Stores the sun: the direction toward it is `-light.direction` normalised, and the colour
    /// is the linear colour times the intensity.
    pub fn set_sun(&self, light: &DirectionalLight) {
        let mut core = self.core.borrow_mut();
        let toward = (-light.direction).normalize_or_zero();
        core.sun_direction = [toward.x, toward.y, toward.z];
        core.sun_color = light.color.map(|channel| channel * light.intensity);
    }

    /// Stores the number of active local lights.
    pub fn set_local_light_count(&self, count: u32) {
        self.core.borrow_mut().local_light_count = count;
    }

    /// Marks the frame history invalid and drops the history of every part.
    pub fn invalidate_history(&self) {
        let mut core = self.core.borrow_mut();
        core.history_valid = false;
        for part in &core.parts {
            part.invalidate_history();
        }
    }

    /// Records the outcome of submission `invocation`: nothing is pending any more, and a
    /// complete submission publishes the prepared statistics with that frame index.
    pub fn observe_submission(&self, invocation: u64, complete: bool) {
        let mut core = self.core.borrow_mut();
        core.pending = false;
        if complete {
            let mut published = core.prepared.clone();
            published.stats_frame_index = Some(invocation);
            core.published = Some(published);
        }
    }

    /// The published statistics, or the prepared ones before the first complete submission, with
    /// `pending` from the core. Before any frame, the initial statistics at the current quality.
    pub fn stats(&self) -> GiStats {
        let core = self.core.borrow();
        let mut stats = match &core.published {
            Some(published) => published.clone(),
            None if core.frame_index == 0 => initial_stats(core.quality),
            None => GiStats {
                stats_frame_index: None,
                ..core.prepared.clone()
            },
        };
        stats.pending = core.pending;
        stats
    }

    /// The GPU bytes the feature owns: `allocated_bytes` of [`GiHandle::stats`].
    pub fn allocated_bytes(&self) -> u64 {
        self.stats().allocated_bytes
    }

    /// The pass names recorded by the latest frame, in record order.
    pub fn recorded_passes(&self) -> Vec<&'static str> {
        self.core.borrow().recorded.clone()
    }

    /// Prepares one frame (contract 'gi-frame-lifecycle'): plans the members of `scene`, decides
    /// the activity, allocates or releases the GPU state, sets the lit features 29, 30 and 37,
    /// writes the frame record and volume of an active frame, runs the parts, and returns the
    /// statistics of this frame. Nothing is published until its submission completes.
    pub fn prepare_frame(
        &self,
        gpu: &GpuContext,
        scene: &mut Scene,
        lit: &mut LitRenderer,
    ) -> GiStats {
        let mut guard = self.core.borrow_mut();
        let core = &mut *guard;
        scene.update_world();
        let previous_members = core.report.members.len();
        let items = core.scratch.fill(scene);
        let drivers = &core.drivers;
        let claimed = |item: &SyncItem| drivers.iter().skip(1).any(|d| d.claims(item, &*lit));
        let source = lit.handle();
        plan_members(items, &source, &core.requests, &claimed, &mut core.report);
        let members = core.report.forward_count + core.report.virtual_count;

        let volume_pass_enabled = core.pipeline.try_borrow().is_ok_and(|p| {
            p.passes()
                .iter()
                .any(|info| info.name == VOLUME_PASS && info.enabled)
        });
        let reason = inactive_reason(
            core.settings.enabled,
            volume_pass_enabled,
            core.submitted_views,
            members,
        );
        let active = reason.is_none();
        let quality = core.quality;

        let mut allocated = false;
        if active {
            if core.volume.is_none() || core.allocated_quality != Some(quality) {
                release_resources(core);
                let frame = gpu.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("raa-gi frame"),
                    size: GiFrameGpu::SIZE as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let layout = core.shared.compute_group1_layout();
                let volume = GiVolume::new(gpu, quality, &frame, &layout);
                core.shared.set_atlas(Some(volume.atlas_view()));
                core.shared.set_frame(Some(frame));
                core.volume = Some(volume);
                core.allocated_quality = Some(quality);
                core.rounds = RebuildRounds::new(quality.levels);
                allocated = true;
            }
        } else {
            release_resources(core);
            core.shared.clear();
            core.work = None;
            core.allocated_quality = None;
        }

        let settings = core.settings;
        lit.set_frame_feature(29, active);
        lit.set_frame_feature(30, active && settings.local_light_shadows);
        lit.set_frame_feature(37, active && settings.reflections);

        let changed = core.signature != Some(core.report.signature)
            || previous_members != core.report.members.len();
        if active && !allocated && changed {
            core.rounds.mark_stale();
        }
        core.signature = Some(core.report.signature);

        if active && let Some(frame) = core.shared.frame_buffer() {
            let record = frame_record(core);
            gpu.queue().write_buffer(&frame, 0, &record.to_bytes());
        }
        if active && let Some(volume) = core.volume.as_mut() {
            let work = volume.prepare(
                gpu,
                core.camera.eye,
                &mut core.rounds,
                &mut core.cursor,
                &core.settings,
            );
            for driver in &core.drivers {
                driver.prepare_frame(gpu, &core.report.members, &work.voxelize);
            }
            core.work = Some(work);
        }

        let frame = PartFrame {
            active,
            settings,
            quality,
            viewport: core.viewport,
            members: &core.report.members,
            history_valid: core.history_valid,
        };
        for part in &core.parts {
            part.prepare_frame(gpu, &frame);
        }

        let pending_slice = core.volume.as_ref().is_some_and(GiVolume::pending_slice);
        let pending_rebuild = pending_slice || core.rounds.pending();
        core.frames_since_rebuild = if pending_rebuild {
            0
        } else {
            core.frames_since_rebuild
                .saturating_add(1)
                .min(MAX_FRAMES_SINCE_REBUILD)
        };
        let allocated_bytes = if active {
            let volume = core.volume.as_ref().map_or(0, GiVolume::allocated_bytes);
            let drivers: u64 = core.drivers.iter().map(|d| d.allocated_bytes()).sum();
            let parts: u64 = core.parts.iter().map(|p| p.allocated_bytes()).sum();
            volume + GiFrameGpu::SIZE as u64 + drivers + parts
        } else {
            0
        };
        let prepared = GiStats {
            active,
            inactive_reason: reason,
            quality,
            finest_voxel_metres: FINEST_VOXEL_METRES,
            voxelized_forward_instances: if active { core.report.forward_count } else { 0 },
            voxelized_virtual_instances: if active { core.report.virtual_count } else { 0 },
            pending_rebuild,
            frames_since_rebuild: core.frames_since_rebuild,
            allocated_bytes,
            stats_frame_index: None,
            pending: true,
        };
        core.prepared = prepared.clone();
        core.pending = true;
        core.recorded.clear();
        core.frame_index += 1;
        prepared
    }
}

/// Drops the volume and releases every driver and part (contract 'gi-frame-lifecycle').
fn release_resources(core: &mut GiCore) {
    core.volume = None;
    for driver in &core.drivers {
        driver.release();
    }
    for part in &core.parts {
        part.release();
    }
}

/// `v` as a reciprocal, or 0 for a non-positive size.
fn reciprocal(v: f32) -> f32 {
    if v > 0.0 { v.recip() } else { 0.0 }
}

/// The `GiFrameGpu` record of an active frame (S38). The previous view-projection is the one of
/// the last active frame, or the current one; the current one becomes the next previous.
fn frame_record(core: &mut GiCore) -> GiFrameGpu {
    let camera = core.camera;
    let (view, proj, view_proj) = (camera.view(), camera.projection(), camera.view_proj());
    let previous_view_proj = core.previous_view_proj.unwrap_or(view_proj);
    core.previous_view_proj = Some(view_proj);
    let (w, h) = (core.viewport.0 as f32, core.viewport.1 as f32);
    let [sx, sy, sz] = core.sun_direction;
    let [cr, cg, cb] = core.sun_color;
    let settings = core.settings;
    GiFrameGpu {
        view_proj,
        inv_view_proj: view_proj.inverse().unwrap_or(Mat4::IDENTITY),
        inv_proj: proj.inverse().unwrap_or(Mat4::IDENTITY),
        inv_view: view.inverse().unwrap_or(Mat4::IDENTITY),
        previous_view_proj,
        camera_position: [camera.eye.x, camera.eye.y, camera.eye.z, 0.0],
        sun_direction: [sx, sy, sz, 0.0],
        sun_color: [cr, cg, cb, 0.0],
        viewport: [w, h, reciprocal(w), reciprocal(h)],
        counts: [
            core.local_light_count,
            core.frame_index as u32,
            settings.debug_view.code(),
            core.quality.levels,
        ],
        settings: [
            settings.intensity,
            settings.environment_diffuse,
            camera.near,
            camera.far,
        ],
        flags: [
            u32::from(core.history_valid),
            u32::from(settings.screen_traces),
            u32::from(settings.local_light_shadows),
            u32::from(settings.reflections),
        ],
    }
}

/// The `gi.volume` pass (order 25): uploads the volume's atlas and parameters in `prepare`, and
/// records the frame's rebuild and relight from the stored [`VolumeWork`] (once per frame).
pub(crate) struct GiVolumePass {
    core: Rc<RefCell<GiCore>>,
}

impl FeaturePass for GiVolumePass {
    fn name(&self) -> &str {
        VOLUME_PASS
    }

    fn order(&self) -> u32 {
        VOLUME_ORDER
    }

    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        if !enabled {
            lighting.clear_gi();
        }
    }

    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let core = self.core.borrow();
        match core.volume.as_ref() {
            Some(volume) => ctx
                .lighting
                .set_gi(None, volume.atlas_view(), &volume.params_bytes()),
            None => {
                ctx.lighting.clear_gi();
                Ok(())
            }
        }
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let mut guard = self.core.borrow_mut();
        let core = &mut *guard;
        let Some(work) = core.work.take() else {
            return Ok(());
        };
        let Some(group1) = core.shared.compute_group1(ctx.lighting) else {
            return Ok(());
        };
        let Some(volume) = core.volume.as_ref() else {
            return Ok(());
        };
        volume.record(ctx.encoder, &work, &core.drivers, &group1);
        core.recorded.push(VOLUME_PASS);
        Ok(())
    }
}

/// The `gi.history` pass (order 64): with a debug view selected, records the debug view of the
/// active frame. It records nothing while GI is inactive or the view is `none`.
pub(crate) struct GiHistoryPass {
    core: Rc<RefCell<GiCore>>,
}

impl FeaturePass for GiHistoryPass {
    fn name(&self) -> &str {
        HISTORY_PASS
    }

    fn order(&self) -> u32 {
        HISTORY_ORDER
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let mut guard = self.core.borrow_mut();
        let core = &mut *guard;
        if core.settings.debug_view == DebugView::None {
            return Ok(());
        }
        let (Some(frame), Some(volume)) = (core.shared.frame_buffer(), core.volume.as_ref()) else {
            return Ok(());
        };
        let Some(group1) = core.shared.compute_group1(ctx.lighting) else {
            return Ok(());
        };
        core.debug.record(ctx, &frame, volume.voxels(), &group1);
        core.recorded.push(HISTORY_PASS);
        Ok(())
    }
}

/// The GI feature installer: [`GiFeature::install`] builds it.
pub struct GiFeature;

impl GiFeature {
    /// Handoff: `GiFeature::install` is the entry point the card "raa-gi: feature tests (GI off
    /// records nothing)" calls. Today it refuses atomically with `pipeline.busy` (the pipeline is
    /// borrowed) or `pipeline.duplicate_pass` (`gi.volume` or `gi.history` already exists) before
    /// any GPU object is built. It then creates the built-in `ForwardDriver` (driver A) as the
    /// first driver, runs `install_parts` through a `PartsCtx`, creates `DebugRenderer::new`,
    /// builds `GiCore::new` and adds the passes `gi.volume` (25) and `gi.history` (64) through
    /// `Pipeline::add`, both left enabled, so GI stays inactive until `GiHandle::set` enables it.
    /// It returns the `GiHandle` over the shared core. That card calls `GiHandle::prepare_frame`,
    /// `GiHandle::set`, `GiHandle::set_quality`, `GiHandle::set_view`,
    /// `GiHandle::observe_submission`, `GiHandle::recorded_passes` and `GiHandle::allocated_bytes`
    /// on it; the state they share is the `GiCore` frame state of contract 'gi-core-state' and
    /// its recorded pass list.
    pub fn install(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        lit: &mut LitRenderer,
        vg: Option<VgHandle>,
        ibl: Option<IblHandle>,
    ) -> Result<GiHandle, RaaError> {
        {
            let p = pipeline.try_borrow().map_err(|_| pipeline_busy())?;
            let entries = p.passes();
            for name in [VOLUME_PASS, HISTORY_PASS] {
                if entries.iter().any(|info| info.name == name) {
                    return Err(duplicate(name));
                }
            }
        }
        let shared = GiShared::new(gpu);
        let builtin = Rc::new(ForwardDriver::new(gpu, lit.handle()));
        let mut ctx = PartsCtx::new(gpu, pipeline, lit, vg, ibl, shared.clone());
        ctx.add_driver(builtin);
        crate::parts::install_parts(&mut ctx)?;
        let (parts, drivers) = ctx.take();
        let debug = DebugRenderer::new(gpu, &shared.compute_group1_layout());
        let core = GiCore::new(gpu, Rc::clone(pipeline), shared, drivers, parts, debug);
        let core = Rc::new(RefCell::new(core));
        {
            let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
            p.add(Box::new(GiVolumePass {
                core: Rc::clone(&core),
            }))?;
            p.add(Box::new(GiHistoryPass {
                core: Rc::clone(&core),
            }))?;
        }
        Ok(GiHandle { core })
    }
}
