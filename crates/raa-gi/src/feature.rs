//! The GI feature handle (S36, S37): the activity rule [`inactive_reason`], the crate-private
//! state [`GiCore`] and the cloneable [`GiHandle`] that shares it with the Engine.
//!
//! [`GiFeature::install`] builds the feature: the two passes `gi.volume` and `gi.history` over
//! the shared [`GiCore`], the built-in forward driver and the parts. [`GiHandle::prepare_frame`]
//! runs once per Engine frame before the pipeline executes (contract 'gi-frame-lifecycle').

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use raa_core::{GpuContext, MeshId, RaaError};
use raa_ibl::IblHandle;
use raa_lit::LitRenderer;
use raa_math::Mat4;
use raa_pipeline::{
    FeaturePass, ImageDesc, ImageDim, LightingResources, Pipeline, PrepareCtx, RecordCtx,
};
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
use crate::volume::{GiVolume, LightFrame, VolumeWork};

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
    /// raa-ibl, read each frame for the sky radiance scale ([`sky_scale`]).
    ibl: Option<IblHandle>,
    local_light_count: u32,
    history_valid: bool,
    /// The viewport of the previous prepared frame when it was active; `None` after an inactive
    /// frame, a reallocation or a camera cut.
    last_active_viewport: Option<(u32, u32)>,
    /// The copy of the previous lit HDR (HDR_FORMAT, surface-sized) and its view, present while
    /// a part wants the history (S38 `gi.history`).
    history: Option<(wgpu::Texture, wgpu::TextureView)>,
    /// The size of the history copy recorded by the latest frame; the next prepare takes it.
    history_copied: Option<(u32, u32)>,
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
    colors_revision: u64,
    frames_since_rebuild: u32,
    /// The world-space local-light records of the latest frame (upload layout), hashed into
    /// `light_signature`.
    light_scratch: Box<[u8; raa_shader::MAX_LOCAL_LIGHTS * raa_shader::LIGHT_GPU_SIZE]>,
    /// The FNV-1a of the sun and the local lights of the latest frame.
    light_signature: Option<u64>,
    /// Bumped when `light_signature` changes: every voxel restarts its radiance average at its
    /// next update (voxel_light.wgsl), so a moved, dimmed or removed light leaves no stale bounce.
    light_epoch: u32,
    /// Frames left of full relight after a lighting change (`HOT_RELIGHT_FRAMES`).
    hot_relight: u32,
    /// The emitted-power proxy of the latest frame ([`lighting_power`]).
    light_power: Option<f32>,
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
            ibl: None,
            local_light_count: 0,
            history_valid: false,
            last_active_viewport: None,
            history: None,
            history_copied: None,
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
            colors_revision: 0,
            light_scratch: Box::new([0; raa_shader::MAX_LOCAL_LIGHTS * raa_shader::LIGHT_GPU_SIZE]),
            light_signature: None,
            light_epoch: 0,
            hot_relight: 0,
            light_power: None,
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

    /// Registers (`Some`) or removes (`None`) the driver A vertex colours of the lit mesh that
    /// sub-mesh `mesh` belongs to: linear rgba, four floats per vertex of the whole uploaded
    /// vertex array. The voxelizer multiplies the albedo of each triangle (base colour x
    /// instance tint) by the mean of its three vertex colours; a mesh without a stream, or one
    /// whose vertex count differs, keeps white. The next prepared frame rebuilds the voxels. A
    /// length that is not a multiple of 4 or a non-finite value is `gi.invalid` and changes
    /// nothing.
    pub fn set_vertex_colors(&self, mesh: MeshId, colors: Option<&[f32]>) -> Result<(), RaaError> {
        self.core
            .borrow()
            .shared
            .vertex_colors()
            .set_mesh(mesh, colors)
    }

    /// Registers (`Some`) or removes (`None`) the driver B (virtualized geometry) vertex colours
    /// of section `ordinal` of cluster-store key `key`: linear rgba, four floats per vertex of
    /// `ClusterMesh::vertices`, in that order. The albedo of a clustered triangle is the base
    /// colour times the mean of its three vertex colours (driver B has no instance tint). Refusals
    /// as for [`set_vertex_colors`](Self::set_vertex_colors).
    pub fn set_cluster_vertex_colors(
        &self,
        key: u32,
        ordinal: usize,
        colors: Option<&[f32]>,
    ) -> Result<(), RaaError> {
        self.core
            .borrow()
            .shared
            .vertex_colors()
            .set_cluster(key, ordinal, colors)
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

    /// Attaches the engine's SDF trace scene, or detaches it with `None`, through
    /// [`GiShared::set_trace`].
    ///
    /// Handoff: the card "raa-js: stochastic keys of setAreaLightShadows/areaLightShadows, docs
    /// and typings, and the GI trace-scene wiring" calls `GiHandle::set_trace_scene` once at
    /// install with the engine's trace handle. Today it stores the handle on the shared frame
    /// state, where `GiShared::trace` reads it until it is replaced.
    pub fn set_trace_scene(&self, trace: Option<raa_sdf_trace::TraceHandle>) {
        self.core.borrow().shared.set_trace(trace);
    }

    /// Marks the frame history invalid and drops the history of every part.
    pub fn invalidate_history(&self) {
        let mut core = self.core.borrow_mut();
        core.history_valid = false;
        core.history_copied = None;
        core.last_active_viewport = None;
        for part in &core.parts {
            part.invalidate_history();
        }
    }

    /// Records the outcome of submission `invocation`: nothing is pending any more, and a
    /// complete submission publishes the prepared statistics with that frame index.
    pub fn observe_submission(&self, invocation: u64, complete: bool) {
        let mut core = self.core.borrow_mut();
        core.pending = false;
        for part in &core.parts {
            part.observe_submission();
        }
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
        let screen = core.parts.iter().find_map(|part| part.screen_stats());
        stats.history_length_mean = screen.map(|s| s.history_length_mean);
        stats.rejected_percent = screen.map(|s| s.rejected_percent);
        stats.clamped_percent = screen.map(|s| s.clamped_percent);
        stats
    }

    /// The GPU bytes the feature owns: `allocated_bytes` of [`GiHandle::stats`].
    pub fn allocated_bytes(&self) -> u64 {
        self.stats().allocated_bytes
    }

    /// The core pass names (gi.volume, gi.history) recorded by the latest frame, in record order;
    /// part passes such as gi.screen are not listed.
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
        // The volume depends on the resolution and the levels only: a probe-ray change (a GI
        // preset) reallocates the screen part, not the voxels.
        let volume_key = |q: GiQuality| (q.resolution, q.levels);
        if active {
            if core.volume.is_none()
                || core.allocated_quality.map(volume_key) != Some(volume_key(quality))
            {
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

        let copied = core.history_copied.take();
        core.history_valid = active && !allocated && copied == Some(core.viewport);
        let camera_valid = active && !allocated && core.last_active_viewport == Some(core.viewport);
        core.last_active_viewport = active.then_some(core.viewport);

        let settings = core.settings;
        lit.set_frame_feature(29, active);
        lit.set_frame_feature(30, active && settings.local_light_shadows);
        lit.set_frame_feature(37, active && settings.reflections);

        let changed = core.signature != Some(core.report.signature)
            || previous_members != core.report.members.len();
        let colors_revision = core.shared.vertex_colors().revision();
        let colors_changed = core.colors_revision != colors_revision;
        core.colors_revision = colors_revision;
        if active && !allocated && (changed || colors_changed) {
            core.rounds.mark_stale();
        }
        core.signature = Some(core.report.signature);

        if active && let Some(frame) = core.shared.frame_buffer() {
            let record = frame_record(core);
            gpu.queue().write_buffer(&frame, 0, &record.to_bytes());
        }
        let sky = sky_scale(core);
        let light_signature = lighting_signature(
            lit,
            core.sun_direction,
            core.sun_color,
            sky,
            &mut core.light_scratch,
        );
        if core
            .light_signature
            .is_some_and(|previous| previous != light_signature)
        {
            core.light_epoch = core.light_epoch.wrapping_add(1);
            core.hot_relight = HOT_RELIGHT_FRAMES;
        }
        core.light_signature = Some(light_signature);
        // A changed scene (an object moved, appeared or left) changes the light paths too: the
        // volume replaces its radiance while it follows.
        if changed {
            core.hot_relight = HOT_RELIGHT_FRAMES;
        }
        let power = lighting_power(lit, core.sun_color) + sky;
        let clear_radiance = core
            .light_power
            .is_some_and(|previous| power < DARKER * previous);
        core.light_power = Some(power);
        let full_relight = core.hot_relight > 0;
        core.hot_relight = core.hot_relight.saturating_sub(1);
        if active && let Some(volume) = core.volume.as_mut() {
            let work = volume.prepare(
                gpu,
                core.camera.eye,
                &mut core.rounds,
                &mut core.cursor,
                &core.settings,
                LightFrame {
                    epoch: core.light_epoch,
                    full_relight,
                    clear_radiance,
                },
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
            camera_valid,
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
            let history = core.history.as_ref().map_or(0, |(texture, _)| {
                u64::from(texture.width()) * u64::from(texture.height()) * 8
            });
            volume + GiFrameGpu::SIZE as u64 + drivers + parts + history
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
            history_length_mean: None,
            rejected_percent: None,
            clamped_percent: None,
        };
        core.prepared = prepared.clone();
        core.pending = true;
        core.recorded.clear();
        core.frame_index += 1;
        prepared
    }
}

/// Frames of full voxel relight (every voxel of the relit level, not a checkerboard half, each
/// update replacing the radiance instead of averaging it) after a lighting change: four passes of
/// the five-level relight schedule, so every level is relit whole at least four times and the
/// bounce of the old lighting has died out before the volume averages again.
const HOT_RELIGHT_FRAMES: u32 = 32;

/// The share of the previous frame's emitted power below which the volume is cleared: a light
/// removed, switched off or dimmed by more than 2 %.
const DARKER: f32 = 0.98;

/// A proxy of the emitted power of the sun and the active local lights: each light's colour
/// luminance times its intensity, an area light's also times its rectangle area. It only has to
/// fall when a light is removed, switched off or dimmed.
pub(crate) fn lighting_power(lit: &LitRenderer, sun_color: [f32; 3]) -> f32 {
    let luminance = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let lights: f32 = lit
        .local_lights()
        .world_lights()
        .map(|(_, _, light)| {
            let area = if light.kind == raa_lit::LocalLightKind::Area {
                (light.size[0] * light.size[1]).max(0.0)
            } else {
                1.0
            };
            luminance(light.color) * light.intensity.max(0.0) * area
        })
        .sum();
    luminance(sun_color) + lights
}

/// The radiance scale of the GI rays that reach the sky: the environment's authored diffuse
/// intensity (raa-ibl `setEnvironmentLighting`), so the sky lights a surface with the same
/// strength whether GI or the IBL ambient term carries it; 1 without raa-ibl.
fn sky_scale(core: &GiCore) -> f32 {
    core.ibl.as_ref().map_or(1.0, IblHandle::diffuse)
}

/// An order-stable FNV-1a 64 of the sun (direction, colour), the sky radiance scale and the
/// world-space records of the active local lights, as uploaded: a change detector for the voxel
/// lighting epoch.
fn lighting_signature(
    lit: &LitRenderer,
    sun_direction: [f32; 3],
    sun_color: [f32; 3],
    sky: f32,
    scratch: &mut [u8; raa_shader::MAX_LOCAL_LIGHTS * raa_shader::LIGHT_GPU_SIZE],
) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let lights = lit.local_lights();
    lights.write_gpu(scratch);
    let used = (lights.active_count() as usize).min(raa_shader::MAX_LOCAL_LIGHTS)
        * raa_shader::LIGHT_GPU_SIZE;
    let sun = sun_direction
        .into_iter()
        .chain(sun_color)
        .chain([sky])
        .flat_map(f32::to_le_bytes);
    sun.chain(scratch[..used].iter().copied())
        .fold(FNV_OFFSET, |h, byte| {
            (h ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
        })
}

/// Drops the volume and releases every driver and part (contract 'gi-frame-lifecycle').
fn release_resources(core: &mut GiCore) {
    core.volume = None;
    core.history = None;
    core.history_copied = None;
    core.shared.set_history(None);
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
        sun_color: [cr, cg, cb, sky_scale(core)],
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
            Some(volume) => {
                // Part order rank (generated file-name order): area_shadow = 0, screen = 1,
                // slots = 2, ssgi = 3. The LAST part with a screen output is bound at binding 14,
                // so the SSGI composite replaces the probe output while gi.ssgi is active.
                let screen = core
                    .parts
                    .iter()
                    .rev()
                    .find_map(|part| part.screen_output());
                let atlas = volume.atlas_view();
                let quality = core.allocated_quality.unwrap_or(core.quality);
                let resolution = quality.resolution;
                let desc = ImageDesc::new(
                    ImageDim::D3,
                    wgpu::TextureFormat::Rgba16Float,
                    resolution,
                    resolution,
                    resolution * quality.levels,
                    1,
                );
                ctx.resources
                    .publish(VOLUME_PASS, "volumeAtlas", &atlas, desc);
                ctx.lighting.set_gi(screen, atlas, &volume.params_bytes())
            }
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

/// Copies the lit HDR of this frame (`hdr.src` after forward) into the history texture, which is
/// recreated on a size change, and publishes its view for the next frame's screen traces.
fn copy_history(core: &mut GiCore, ctx: &mut RecordCtx<'_>) {
    let (width, height) = ctx.size;
    let current = core
        .history
        .as_ref()
        .is_some_and(|(texture, _)| (texture.width(), texture.height()) == (width, height));
    if !current {
        let texture = ctx.gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("raa-gi.history"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: raa_pipeline::HDR_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        core.history = Some((texture, view));
    }
    let Some((history, view)) = core.history.as_ref() else {
        return;
    };
    let extent = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    ctx.encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: ctx.targets.hdr().src_texture(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: history,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        extent,
    );
    core.shared.set_history(Some(view.clone()));
    core.history_copied = Some((width, height));
}

/// The `gi.history` pass (order 64). While GI is active, it copies the lit HDR into the history
/// when a part wants it, and otherwise drops the history. With a debug view selected, it then
/// records the debug view of the active frame. It records nothing while GI is inactive or the
/// view is `none`; `HISTORY_PASS` is listed at most once per frame.
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
        let result = self.record_history(ctx);
        let size = ctx.size;
        let hdr = ImageDesc::new(ImageDim::D2, raa_pipeline::HDR_FORMAT, size.0, size.1, 1, 1);
        ctx.resources
            .publish(HISTORY_PASS, "color", ctx.targets.hdr().src(), hdr);
        if let Some((texture, view)) = self.core.borrow().history.as_ref() {
            let (w, h) = (texture.width(), texture.height());
            let desc = ImageDesc::new(ImageDim::D2, raa_pipeline::HDR_FORMAT, w, h, 1, 1);
            ctx.resources.publish(HISTORY_PASS, "history", view, desc);
        }
        result
    }
}

impl GiHistoryPass {
    /// The copy and debug record of the pass; `record` publishes the taps after it.
    fn record_history(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let mut guard = self.core.borrow_mut();
        let core = &mut *guard;
        if core.shared.frame_buffer().is_some() {
            if core.parts.iter().any(|part| part.wants_history()) {
                copy_history(core, ctx);
                core.recorded.push(HISTORY_PASS);
            } else {
                core.history = None;
                core.shared.set_history(None);
            }
        }
        if core.settings.debug_view == DebugView::None {
            return Ok(());
        }
        let (Some(frame), Some(volume)) = (core.shared.frame_buffer(), core.volume.as_ref()) else {
            return Ok(());
        };
        let Some(group1) = core.shared.compute_group1(ctx.lighting) else {
            return Ok(());
        };
        let screen_debug = core.shared.screen_debug_view();
        let screen_output = core.shared.probe_output_view();
        core.debug.record(
            ctx,
            &frame,
            volume.voxels(),
            &group1,
            screen_debug.as_ref(),
            screen_output.as_ref(),
        );
        if !core.recorded.contains(&HISTORY_PASS) {
            core.recorded.push(HISTORY_PASS);
        }
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
        let builtin = Rc::new(ForwardDriver::new(
            gpu,
            lit.handle(),
            shared.vertex_colors(),
        ));
        let sky_ibl = ibl.clone();
        let mut ctx = PartsCtx::new(gpu, pipeline, lit, vg, ibl, shared.clone());
        ctx.add_driver(builtin);
        crate::parts::install_parts(&mut ctx)?;
        let (parts, drivers) = ctx.take();
        let debug = DebugRenderer::new(gpu, &shared.compute_group1_layout());
        let mut core = GiCore::new(gpu, Rc::clone(pipeline), shared, drivers, parts, debug);
        core.ibl = sky_ibl;
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

#[cfg(test)]
mod tests {
    //! The S36 rank table as a pure table, and the lifecycle of contract 'gi-frame-lifecycle' on
    //! a software GPU (SKIP without an adapter): GI off records nothing, a second install
    //! refuses, enabling at quality 64/3 records `gi.volume` once, and disabling the pass drops
    //! the volume.
    //!
    //! The rig follows `crates/raa-vg/tests/feature.rs`: one instance of a 1 m cube at the
    //! origin, the camera at (0, 1, 4) looking at the origin with near 0.1 and far 16. At 64/3
    //! level 0 spans x and y in [-6.4, 6.4] m and z in [-3.2, 9.6] m around the snapped origin,
    //! so the cube and the camera lie inside the grid. The graph's prepass clears depth to 1.0,
    //! and no mesh pass writes depth before `gi.volume`. `gi.volume` reads no depth or HDR target:
    //! its frame record (sun, counts, camera) is written by `prepare_frame` itself, and the sun
    //! is left at its default of no light.

    use super::*;
    use raa_core::testing::skip;
    use raa_core::{FrameGraph, MaterialId, OffscreenTarget};
    use raa_lit::{AttributedMesh, LitOptions};
    use raa_math::{Quat, Vec3};
    use raa_scene::{RenderList, Transform};

    const SIZE: (u32, u32) = (64, 64);
    const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    /// The `GiVolume` allocation at 64/3 (GI_VOLUME_BYTES, S38 and S37).
    const BYTES_64_3: u64 = 27_262_976;

    #[test]
    fn inactive_reason_follows_the_rank_table() {
        assert_eq!(inactive_reason(false, true, 1, 3), Some("disabled"));
        assert_eq!(inactive_reason(true, false, 1, 3), Some("passDisabled"));
        assert_eq!(inactive_reason(true, true, 2, 3), Some("multiView"));
        assert_eq!(inactive_reason(true, true, 1, 0), Some("noInstance"));
        assert_eq!(inactive_reason(true, true, 1, 3), None);
        assert_eq!(
            inactive_reason(false, false, 2, 0),
            Some("disabled"),
            "the first match wins"
        );
    }

    /// A 1 m cube centred at the origin: four vertices per face (hard edges, outward normals)
    /// and 12 counter-clockwise triangles.
    fn cube() -> AttributedMesh {
        let mut mesh = AttributedMesh::from_mesh(Default::default());
        for axis in 0..3 {
            for sign in [-0.5_f32, 0.5] {
                let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                let mut normal = [0.0_f32; 3];
                normal[axis] = sign.signum();
                let first = mesh.base.positions.len() as u32;
                for (a, b) in [(-0.5_f32, -0.5_f32), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)] {
                    let mut p = [0.0_f32; 3];
                    p[axis] = sign;
                    p[u] = a;
                    p[v] = b;
                    mesh.base.positions.push(p);
                    mesh.base.normals.push(normal);
                }
                let quad = if sign > 0.0 {
                    [0u32, 1, 2, 0, 2, 3]
                } else {
                    [0u32, 2, 1, 0, 3, 2]
                };
                mesh.base.indices.extend(quad.map(|i| first + i));
            }
        }
        mesh.submeshes[0].index_count = mesh.base.indices.len() as u32;
        mesh
    }

    /// A pipeline with the feature installed, one cube instance and the graph of the pipeline.
    struct Rig {
        handle: GiHandle,
        pipeline: Rc<RefCell<Pipeline>>,
        lit: LitRenderer,
        scene: Scene,
        camera: Camera,
        list: RenderList,
        tags: Vec<u32>,
        slots: Vec<u32>,
        graph: FrameGraph,
        backbuffer: OffscreenTarget,
    }

    impl Rig {
        fn new(gpu: &GpuContext) -> Rig {
            let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, SURFACE, false)));
            let opts = LitOptions {
                msaa: 4,
                ..LitOptions::default()
            };
            let mut lit = LitRenderer::new(gpu, &pipeline, opts).expect("lit renderer");
            let handle = GiFeature::install(gpu, &pipeline, &mut lit, None, None)
                .expect("the feature installs");
            let id = lit.upload(gpu, &cube()).expect("upload");
            let submeshes = lit.submesh_ids(id).expect("sub-meshes").to_vec();
            let mut scene = Scene::with_capacity(1, 1);
            let node = scene
                .add_node(
                    None,
                    Transform::from_trs(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE),
                )
                .expect("node");
            scene
                .add_instance(node, submeshes[0], MaterialId(0))
                .expect("instance");
            let camera = Camera {
                eye: Vec3::new(0.0, 1.0, 4.0),
                target: Vec3::ZERO,
                near: 0.1,
                far: 16.0,
                ..Camera::default()
            };
            scene.camera = camera;
            scene.camera.set_viewport(SIZE.0, SIZE.1);
            let mut graph = FrameGraph::new(SURFACE);
            Pipeline::install(&pipeline, &mut graph);
            graph.compile().expect("compile");
            Rig {
                handle,
                pipeline,
                lit,
                scene,
                camera,
                list: RenderList::with_capacity(1),
                tags: Vec::with_capacity(1),
                slots: Vec::with_capacity(1),
                graph,
                backbuffer: OffscreenTarget::new(gpu, SIZE.0, SIZE.1, SURFACE),
            }
        }

        /// One frame in the contract's order: the view, `prepare_frame`, the lit prepare, then
        /// the graph executed into the offscreen target and waited for. Returns the statistics
        /// of `prepare_frame`.
        fn frame(&mut self, gpu: &GpuContext, views: u8) -> GiStats {
            self.handle.set_view(&self.camera, SIZE, views);
            let stats = self
                .handle
                .prepare_frame(gpu, &mut self.scene, &mut self.lit);
            self.lit
                .prepare(
                    gpu,
                    &mut self.scene,
                    &mut self.list,
                    &mut self.tags,
                    &mut self.slots,
                )
                .expect("prepare");
            let commands = self
                .graph
                .execute(gpu, self.backbuffer.view(), SIZE)
                .expect("execute");
            gpu.queue().submit([commands]);
            gpu.device()
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("device poll");
            stats
        }

        /// Enables GI at quality 64/3 (the default settings keep it off).
        fn enable_at_64_3(&self) {
            self.handle
                .set(GiSettingsPatch {
                    enabled: Some(true),
                    ..GiSettingsPatch::default()
                })
                .expect("enabled");
            self.handle
                .set_quality(GiQualityPatch {
                    resolution: Some(64.0),
                    levels: Some(3.0),
                    ..GiQualityPatch::default()
                })
                .expect("64/3 is a listed quality");
        }
    }

    #[test]
    fn gi_off_records_nothing() {
        let test = "gi_off_records_nothing";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let mut rig = Rig::new(&gpu);
        let stats = rig.frame(&gpu, 1);
        assert!(!stats.active, "{stats:?}");
        assert_eq!(stats.inactive_reason, Some("disabled"));
        assert_eq!(stats.allocated_bytes, 0);
        assert_eq!(rig.handle.allocated_bytes(), 0);
        assert!(rig.handle.recorded_passes().is_empty());
    }

    #[test]
    fn second_install_refuses_and_adds_nothing() {
        let test = "second_install_refuses_and_adds_nothing";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let mut rig = Rig::new(&gpu);
        let before = rig.pipeline.borrow().passes().len();
        let error = GiFeature::install(&gpu, &rig.pipeline, &mut rig.lit, None, None)
            .err()
            .expect("a second install refuses");
        assert_eq!(error.code(), "pipeline.duplicate_pass");
        assert_eq!(rig.pipeline.borrow().passes().len(), before);
    }

    #[test]
    fn enabled_records_gi_volume_once_per_frame() {
        let test = "enabled_records_gi_volume_once_per_frame";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let mut rig = Rig::new(&gpu);
        rig.enable_at_64_3();
        let stats = rig.frame(&gpu, 1);
        assert!(stats.active, "{stats:?}");
        assert_eq!(stats.inactive_reason, None);
        assert_eq!(stats.voxelized_forward_instances, 1);
        assert!(
            stats.allocated_bytes >= BYTES_64_3,
            "allocated {} < {BYTES_64_3}",
            stats.allocated_bytes
        );
        assert_eq!(rig.handle.recorded_passes(), ["gi.volume", "gi.history"]);
        rig.handle.observe_submission(1, true);
        assert_eq!(rig.handle.stats().stats_frame_index, Some(1));
    }

    /// The history copy is the forward output of the frame: the rig's prepass clears depth to
    /// 1.0 and no mesh pass writes the HDR before forward, so the copied HDR is what forward
    /// wrote into the 64x64 offscreen frame. The camera and cube stay inside level 0.
    #[test]
    fn history_copy_is_valid_after_two_active_frames() {
        let test = "history_copy_is_valid_after_two_active_frames";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let mut rig = Rig::new(&gpu);
        rig.enable_at_64_3();
        rig.frame(&gpu, 1);
        assert!(
            !rig.handle.core.borrow().history_valid,
            "the first frame allocates, so no copy is valid yet"
        );
        rig.frame(&gpu, 1);
        let core = rig.handle.core.borrow();
        assert!(core.history_valid);
        assert!(core.shared.history_view().is_some());
    }

    #[test]
    fn invalidate_history_clears_validity_on_the_next_frame() {
        let test = "invalidate_history_clears_validity_on_the_next_frame";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let mut rig = Rig::new(&gpu);
        rig.enable_at_64_3();
        rig.frame(&gpu, 1);
        rig.frame(&gpu, 1);
        assert!(rig.handle.core.borrow().history_valid);
        rig.handle.invalidate_history();
        rig.frame(&gpu, 1);
        assert!(!rig.handle.core.borrow().history_valid);
    }

    /// Runs `gi.volume`'s `prepare` and `gi.history`'s `record` against an 8x8 `Targets` and a
    /// fresh `ResourceTable` before GI allocated its resources (no tap but the chain colour), then
    /// after an enabled 64/3 frame (every tap with its descriptor).
    #[test]
    fn taps_publish_volume_and_history_ports() {
        let test = "taps_publish_volume_and_history_ports";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let mut rig = Rig::new(&gpu);
        let core = Rc::clone(&rig.handle.core);
        let targets = raa_pipeline::Targets::new(&gpu, (8, 8));
        let mut lighting = LightingResources::new(&gpu);
        let backbuffer = OffscreenTarget::new(&gpu, 8, 8, SURFACE);
        let mut run = |table: &raa_pipeline::ResourceTable| {
            let mut prepare = PrepareCtx {
                gpu: &gpu,
                size: (8, 8),
                frame: 0,
                time_s: 0.0,
                dt_s: 0.0,
                targets: &targets,
                lighting: &mut lighting,
                resources: table,
            };
            let mut volume = GiVolumePass {
                core: Rc::clone(&core),
            };
            volume.prepare(&mut prepare).expect("prepare");
            let mut encoder = gpu
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let mut record = RecordCtx {
                gpu: &gpu,
                encoder: &mut encoder,
                size: (8, 8),
                frame: 0,
                time_s: 0.0,
                targets: &targets,
                lighting: &lighting,
                resources: table,
                backbuffer: backbuffer.view(),
                backbuffer_format: SURFACE,
                surface_is_srgb: false,
            };
            let mut history = GiHistoryPass {
                core: Rc::clone(&core),
            };
            history.record(&mut record).expect("record");
        };
        let hdr = ImageDesc::new(ImageDim::D2, raa_pipeline::HDR_FORMAT, 8, 8, 1, 1);
        let before = raa_pipeline::ResourceTable::new();
        run(&before);
        let color = ("gi.history".to_string(), "color".to_string(), hdr);
        assert_eq!(before.entries(), std::slice::from_ref(&color));
        rig.enable_at_64_3();
        rig.frame(&gpu, 1);
        let table = raa_pipeline::ResourceTable::new();
        run(&table);
        let d3 = ImageDesc::new(
            ImageDim::D3,
            wgpu::TextureFormat::Rgba16Float,
            64,
            64,
            64 * 3,
            1,
        );
        let history = ("gi.history".to_string(), "history".to_string(), hdr);
        let volume = ("gi.volume".to_string(), "volumeAtlas".to_string(), d3);
        assert_eq!(table.entries(), [volume, color, history]);
    }

    #[test]
    fn screen_traces_off_records_only_gi_volume() {
        let test = "screen_traces_off_records_only_gi_volume";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let mut rig = Rig::new(&gpu);
        rig.enable_at_64_3();
        rig.handle
            .set(GiSettingsPatch {
                screen_traces: Some(false),
                ..GiSettingsPatch::default()
            })
            .expect("screenTraces is a boolean");
        assert!(rig.frame(&gpu, 1).active);
        assert_eq!(rig.handle.recorded_passes(), ["gi.volume"]);
        assert!(rig.handle.core.borrow().shared.history_view().is_none());
    }

    #[test]
    fn disabling_gi_volume_drops_the_volume() {
        let test = "disabling_gi_volume_drops_the_volume";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let mut rig = Rig::new(&gpu);
        rig.enable_at_64_3();
        assert!(rig.frame(&gpu, 1).active);
        rig.pipeline
            .borrow_mut()
            .set_enabled("gi.volume", false)
            .expect("gi.volume is installed");
        let stats = rig.frame(&gpu, 1);
        assert!(!stats.active, "{stats:?}");
        assert_eq!(stats.inactive_reason, Some("passDisabled"));
        assert_eq!(stats.allocated_bytes, 0);
        assert!(rig.handle.recorded_passes().is_empty());
    }
}
