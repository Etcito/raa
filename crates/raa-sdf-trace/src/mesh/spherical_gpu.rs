//! Spherical diffuse capture and lookup on the shared world-space mesh BVH.

use std::cell::RefCell;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::rc::Rc;
use std::sync::mpsc::{Receiver, sync_channel};

use raa_core::{GpuContext, RaaError};
use raa_pipeline::{FeaturePass, LightingResources, Pipeline, PrepareCtx, RecordCtx};
use wgpu::{Buffer, BufferUsages as Usage};

use super::MeshTraceHandle;
use super::pass::{buffer, buffer_entry, image_entry};
use super::spherical::{
    ALL_FACES, CellKey, FaceBudget, FaceJob, FaceToken, RadialShell, SKY_LIGHT_ID, SUN_LIGHT_ID,
    SchedulerStats, SphericalLight, SphericalScheduler, SphericalSchedulerConfig,
};
use super::{MirrorPriority, ReflectedSnapshot};

pub const SPHERICAL_GI_NAME: &str = "gi.spherical";
const ORDER: u32 = 58;
const TEXELS_PER_FACE: usize = 64;
const FACE_COUNT: usize = 6;
const FACE_LIGHTING_BYTES: usize = 64;
const TEXEL_BYTES: usize = 48;
const CELL_BYTES: usize = TEXELS_PER_FACE * FACE_COUNT * TEXEL_BYTES;
const WORK_BYTES: usize = 48;
const LOOKUP_HEADER: usize = 4 + 64 + 2;
/// Header rows of the sun and sky group scales, written every frame by `spherical_scales`.
const SUN_SCALE_ROW: usize = 68;
const SKY_SCALE_ROW: usize = 69;
// The scale rows sit after the 64 shell rows, inside the header.
const _: () = assert!(SUN_SCALE_ROW >= 4 + 64 && SKY_SCALE_ROW < LOOKUP_HEADER);
const LOOKUP_ENTRY: usize = 3;
const CHANNELS: usize = 3;
/// Legacy projected rows, then 24 parallax patches of four rows, 96 depth-moment
/// bins two per row, and one probe information row (see mesh_spherical_gi.wgsl).
const LEGACY_SLOT_ROWS: usize = CHANNELS * (1 + FACE_COUNT);
const PATCH_ROWS: usize = 4 * 4 * FACE_COUNT;
const MOMENT_ROWS: usize = 16 * FACE_COUNT / 2;
/// Info row, then one light row (sun visibility, two local light indices and visibilities).
const SLOT_ROWS: usize = LEGACY_SLOT_ROWS + PATCH_ROWS + MOMENT_ROWS + 2;
const MAX_FACES_PER_BUDGET: u32 = 4096;
pub const MAX_CAPTURE_MS: f32 = 50.0;
pub const MAX_CAPTURE_SAMPLES: u32 = 64;
/// Sky transfer buffer: 16 header rows (sky SH, norm, reach light list), then two rows per
/// (slot, patch) (mesh_spherical_gi.wgsl `spherical_transfer`).
const TRANSFER_HEADER_ROWS: usize = 16;
const TRANSFER_ROWS_PER_SLOT: usize = 2 * 4 * FACE_COUNT;
const TRANSFER_REACH_ROW: usize = 10;
/// Lights one reach test takes (transfer row 10); more fall back to the faces' hit bounds.
const REACH_LIGHTS: usize = 4;
/// Half angle of the cone around the sun left out of the sky with the transfer (its disc is
/// the sun light).
const SUN_CONE_DEGREES: f32 = 2.5;
/// Sun rotation (degrees, accumulated per slot) that re-shades the sun group.
const SUN_STEP_DEGREES: f32 = 0.5;
/// Rotation since a slot's last shade at which the re-shade replaces instead of blending.
const SUN_SNAP_DEGREES: f32 = 3.0;
const SUN_MIN_BLEND: f32 = 0.25;
/// Relative change of the sky SH shape that re-shades the sky group (the residual and the
/// bounce rays); with the transfer the SH part follows at once, so it may be coarser.
const SKY_STEP_TRANSFER: f32 = 0.05;
/// Frames the sky projection and relight run after a change (the environment filter and its
/// published SH land a frame or more after their source).
const RELIGHT_FRAMES: u32 = 4;

/// What a change of the sun direction does to captured probes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SphericalSunRefresh {
    /// Re-shade the sun group in place: one shadow ray per hit texel from the stored hit
    /// (and the sky group's bounce rays), in view order, blended by the angle.
    #[default]
    Reshade,
    /// Recapture every face (primary rays and all groups), the original rule.
    Recapture,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum SphericalGiDebug {
    #[default]
    Off,
    Boundaries,
    Occupancy,
    Age,
    LightHash,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphericalGiSettings {
    pub enabled: bool,
    pub scheduler: SphericalSchedulerConfig,
    /// Fixed face budget, used when the GPU-time budget is off or timestamps are unavailable.
    pub budget: FaceBudget,
    /// Steady-state capture target in GPU milliseconds per frame; 0 keeps the fixed budget.
    pub capture_ms: f32,
    /// Capture target while visible probes are missing or undiscovered (after open or
    /// a camera move); 0 uses `capture_ms`.
    pub burst_ms: f32,
    /// Cosine bounce rays per hit texel toward the sky (0 = none).
    pub sky_samples: u32,
    /// Local lights sampled per hit texel, stratified with weight count / samples;
    /// 0 evaluates every light (one shadow ray each). The sun is always one shadow ray.
    pub light_samples: u32,
    /// Receivers farther than this from the grid anchor (metres) read each probe's
    /// six-face irradiance instead of its 24 parallax patches, blended over the next
    /// half of that distance; 0 always evaluates the patches.
    pub parallax_range: f32,
    /// Sphere-only direct lighting (phase 2): forward skips its local lights and, beyond
    /// `direct_sun_range` metres from the camera, the cascaded sun shadow. The resolve adds
    /// each probe pair's two strongest local lights, evaluated at the receiver with the
    /// probes' visibility, and the probes' sun visibility replaces the far shadow.
    pub sphere_direct: bool,
    pub direct_sun_range: f32,
    /// Experimental lookup (user sketch): a receiver uses the probes on the side its normal
    /// faces, weighted by a compact distance kernel, instead of the trilinear cell corners.
    pub front_probes: bool,
    /// Diagnostic: resolve texel traced by the lookup, x + 1 | (y + 1) << 16 (0 = off);
    /// its rows are read back into `SphericalGiState::trace`.
    pub trace_texel: u32,
    pub intensity: f32,
    pub debug: SphericalGiDebug,
    /// Bit 0 selects sky, bit 1 sun and bit 2 other.
    pub channels: u32,
    /// Multiply the spherical irradiance by the screen-space AO (min(material AO, GTAO)) as the
    /// ambient term it replaces is (header row 2.w bit 16), like DDGI's `screen_ao`.
    pub screen_ao: bool,
    pub sun_refresh: SphericalSunRefresh,
    /// Faces re-shaded per frame after a sun or sky change.
    pub sun_faces: u32,
    /// Sky misses as 9 SH visibility coefficients plus a normalised residual per patch: a sky
    /// colour or shape change relights the patches (no ray).
    pub sky_transfer: bool,
    /// Per-face light invalidation; added or moved lights take a GPU reach test over the
    /// stored hits instead of the faces' hit bounds.
    pub light_reach: bool,
    /// Luma under which a light's unshadowed contribution does not count as reaching a hit.
    pub light_threshold: f32,
}

impl Default for SphericalGiSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            scheduler: SphericalSchedulerConfig::default(),
            budget: FaceBudget { near: 24, far: 12 },
            capture_ms: 0.0,
            burst_ms: 0.0,
            sky_samples: 1,
            light_samples: 0,
            parallax_range: 0.0,
            sphere_direct: false,
            direct_sun_range: 12.0,
            front_probes: false,
            trace_texel: 0,
            intensity: 1.0,
            debug: SphericalGiDebug::Off,
            channels: 7,
            screen_ao: true,
            sun_refresh: SphericalSunRefresh::Reshade,
            sun_faces: 1536,
            sky_transfer: true,
            light_reach: true,
            light_threshold: 1e-3,
        }
    }
}

impl SphericalGiSettings {
    /// What the probe projection depends on: the channel mask and the sphere-direct light
    /// rows. A change reprojects every published probe from its captured faces.
    pub fn projection_key(&self) -> u32 {
        self.channels | (u32::from(self.sphere_direct) << 8) | (u32::from(self.sky_transfer) << 9)
    }

    pub fn validate(self) -> Result<Self, RaaError> {
        if !self.intensity.is_finite()
            || !(0.0..=16.0).contains(&self.intensity)
            || self.channels > 7
            || self.budget.near > MAX_FACES_PER_BUDGET
            || self.budget.far > MAX_FACES_PER_BUDGET
            || !(0.0..=MAX_CAPTURE_MS).contains(&self.capture_ms)
            || !(0.0..=MAX_CAPTURE_MS).contains(&self.burst_ms)
            || self.sky_samples > MAX_CAPTURE_SAMPLES
            || self.light_samples > MAX_CAPTURE_SAMPLES
            || !self.parallax_range.is_finite()
            || !(0.0..=1.0e4).contains(&self.parallax_range)
            || !self.direct_sun_range.is_finite()
            || !(0.0..=1.0e4).contains(&self.direct_sun_range)
            || !(6..=6 * MAX_FACES_PER_BUDGET).contains(&self.sun_faces)
            || !self.light_threshold.is_finite()
            || !(0.0..=1.0e3).contains(&self.light_threshold)
        {
            return Err(RaaError::FrameGraph {
                code: "gi.spherical.invalid_settings",
                message: format!(
                    "expected intensity in [0, 16], channel mask in [0, 7], face budgets in [0, {MAX_FACES_PER_BUDGET}], capture times in [0, {MAX_CAPTURE_MS}] ms, sample counts in [0, {MAX_CAPTURE_SAMPLES}], parallaxRange, directSunRange in [0, 10000] m, sunFaces in [6, 24576] and lightThreshold in [0, 1000]"
                ),
            });
        }
        SphericalScheduler::new(self.scheduler)?;
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SphericalGiStats {
    pub scheduler: SchedulerStats,
    pub gpu_bytes: u64,
    pub captured_faces: u64,
    pub capture_frames: u64,
    pub light_revision: u64,
    pub origin: [f32; 3],
    pub last_shell: RadialShell,
    /// Faces issued by the latest prepare, whether it ran the burst target, and the
    /// latest measured capture GPU time (0 before the first timestamp sample).
    pub face_budget: u32,
    pub burst: bool,
    pub capture_gpu_ms: f32,
    /// Bytes per pool slot (capture faces, slot rows, work entries), for pool sizing.
    pub slot_bytes: u64,
    /// Cumulative faces re-shaded (sun / sky groups in place, no primary ray), frames that
    /// re-shaded, and resident probes still waiting for a re-shade after the latest prepare.
    pub reshaded_faces: u64,
    pub reshade_frames: u64,
    pub reshade_pending: u32,
    /// Cumulative frames that relit the sky transfer (no ray).
    pub relights: u64,
    /// Cumulative reach tests, faces they flagged and faces of lights beyond a test's four.
    pub reach_tests: u64,
    pub reach_faces: u64,
    pub reach_fallback_faces: u64,
}

/// Readback from one fully submitted, current cubemap.
#[derive(Clone, Debug)]
pub struct SphericalGiProbe {
    pub slot: u32,
    pub frame: u64,
    pub face_hits: [u32; FACE_COUNT],
    pub face_valid_geometry: [u32; FACE_COUNT],
    pub face_valid_misses: [u32; FACE_COUNT],
    pub channel_sums: [[f32; 3]; CHANNELS],
    pub light_hash: u64,
    pub contributing_light_ids: Vec<u64>,
}

#[derive(Clone, Copy)]
struct View {
    index: usize,
    origin: [f32; 3],
    forward: [f32; 3],
    half_fov_cos: f32,
    light: [f32; 10],
}

pub type SphericalGiHandle = Rc<RefCell<SphericalGiState>>;

#[derive(Default)]
pub struct SphericalGiState {
    pub settings: SphericalGiSettings,
    view: Option<View>,
    light_ids: Rc<[u64]>,
    next_light_ids: Vec<u64>,
    local_lights: Vec<SphericalLight>,
    scheduler: Option<SphericalScheduler>,
    grid_revision: u64,
    pass_enabled: bool,
    stats: Option<SphericalGiStats>,
    probe_requested: bool,
    probe: Option<Probe>,
    probe_result: Option<(FaceIdentity, SphericalGiProbe)>,
    trace_pending: Option<(Buffer, Receiver<Option<Vec<u8>>>)>,
    trace_result: Option<Vec<u32>>,
}

impl SphericalGiState {
    pub fn configure(&mut self, settings: SphericalGiSettings) -> Result<(), RaaError> {
        let settings = settings.validate()?;
        if settings.enabled
            && (self.scheduler.is_none() || self.settings.scheduler != settings.scheduler)
        {
            self.reset_grid(settings)?;
            self.stats = None;
            self.probe = None;
            self.probe_result = None;
        }
        self.settings = settings;
        Ok(())
    }

    fn reset_grid(&mut self, settings: SphericalGiSettings) -> Result<(), RaaError> {
        self.scheduler = Some(SphericalScheduler::new(settings.scheduler)?);
        self.grid_revision = self.grid_revision.wrapping_add(1);
        Ok(())
    }

    pub fn active(&self) -> bool {
        self.pass_enabled && self.settings.enabled
    }

    pub fn stats(&self) -> Option<SphericalGiStats> {
        self.stats
    }

    /// The last diagnostic trace rows read back (settings.trace_texel), as u32 words.
    pub fn trace(&mut self) -> Option<Vec<u32>> {
        if let Some((_, result)) = &self.trace_pending {
            match result.try_recv() {
                Ok(bytes) => {
                    self.trace_result = bytes.map(|bytes| {
                        bytes
                            .chunks_exact(4)
                            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                            .collect()
                    });
                    self.trace_pending = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.trace_pending = None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        self.trace_result.clone()
    }

    pub fn probe(&mut self) -> Option<SphericalGiProbe> {
        if !self.active() {
            return None;
        }
        self.poll_probe();
        if let Some((_, result)) = &self.probe_result {
            return Some(result.clone());
        }
        self.probe_requested = true;
        None
    }

    /// The bridge supplies the effective view after scene and light updates.
    pub fn set_view(
        &mut self,
        index: usize,
        origin: [f32; 3],
        forward: [f32; 3],
        half_fov_cos: f32,
        light: [f32; 10],
    ) {
        self.view = Some(View {
            index,
            origin,
            forward,
            half_fov_cos,
            light,
        });
    }

    /// Updates the dense-index mapping in the same order as the frame's packed lights.
    pub fn set_light_ids(&mut self, ids: impl IntoIterator<Item = u64>) {
        self.next_light_ids.clear();
        self.next_light_ids.extend(ids);
        if self.light_ids.as_ref() != self.next_light_ids {
            self.light_ids = Rc::from(self.next_light_ids.as_slice());
        }
    }

    fn disable(&mut self) {
        self.scheduler = None;
        self.probe = None;
        self.probe_result = None;
        self.probe_requested = false;
        self.local_lights.clear();
        self.light_ids = Rc::default();
        self.next_light_ids.clear();
        if let Some(stats) = &mut self.stats {
            stats.scheduler = SchedulerStats::default();
            stats.gpu_bytes = 0;
            stats.captured_faces = 0;
            stats.capture_frames = 0;
            stats.light_revision = 0;
        }
    }

    fn poll_probe(&mut self) {
        let current = |id: FaceIdentity| {
            self.scheduler
                .as_ref()
                .is_some_and(|scheduler| id.current(scheduler))
        };
        if self
            .probe_result
            .as_ref()
            .is_some_and(|(id, _)| !current(*id))
        {
            self.probe_result = None;
        }
        if let Some(probe) = &self.probe {
            match probe.result.try_recv() {
                Ok(Some(result)) if current(probe.id) => {
                    self.probe_result = Some((probe.id, result));
                    self.probe = None;
                    self.probe_requested = false;
                }
                Ok(_) | Err(std::sync::mpsc::TryRecvError::Disconnected) => self.probe = None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
    }
}

pub struct SphericalGiFeature;

impl SphericalGiFeature {
    pub fn install(
        pipeline: &mut Pipeline,
        mesh: MeshTraceHandle,
    ) -> Result<SphericalGiHandle, RaaError> {
        let state = SphericalGiHandle::default();
        pipeline.add(Box::new(SphericalPass {
            state: state.clone(),
            mesh,
            gpu: None,
            staged: Staged::default(),
            sky_shape: SkyShape::default(),
            track: LightTrack::default(),
        }))?;
        pipeline.set_enabled(SPHERICAL_GI_NAME, false)?;
        Ok(state)
    }
}

#[derive(Default)]
struct Staged {
    jobs: Vec<FaceJob>,
    light_ids: Rc<[u64]>,
    feedback: Option<LightFeedback>,
    grid_revision: u64,
    frame: u64,
    light_revision: u64,
    primary: bool,
    recorded: bool,
    /// Slots re-shaded this frame (sun and/or sky group in place).
    reshade: Vec<ReshadeJob>,
    /// Sun direction and sky shape epoch of this frame (what captures and re-shades hold).
    sun: [f32; 3],
    sky_epoch: u64,
    /// Sky transfer active this frame (environment present and the setting on).
    transfer: bool,
    relight: bool,
    reach: Option<ReachRequest>,
}

/// One slot to re-shade: groups bit 0 sun, bit 1 sky; the blend weight of the new value.
#[derive(Clone, Copy, Debug)]
struct ReshadeJob {
    key: CellKey,
    slot: u32,
    epoch: u64,
    probe: [f32; 3],
    groups: u32,
    alpha: f32,
}

/// A reach test: dense indices of up to four lights and the ready faces when it was issued.
struct ReachRequest {
    indices: Vec<u32>,
    tokens: Vec<FaceToken>,
}

/// What each pool slot's sun and sky groups currently hold.
#[derive(Clone, Copy, Debug, Default)]
struct SlotSun {
    valid: bool,
    epoch: u64,
    dir: [f32; 3],
    /// False after a blended (alpha < 1) re-shade or a capture spread over a sun move.
    settled: bool,
    sky_epoch: u64,
}

/// Frame-to-frame lighting the probes follow without recapture.
#[derive(Default)]
struct LightTrack {
    relight: Option<u64>,
    relight_frames: u32,
    previous_sun: Option<[f32; 3]>,
    /// Sky shape at the coarser re-shade tolerance; its epoch is the sky group's version.
    sky: SkyShape,
}

struct SphericalPass {
    state: SphericalGiHandle,
    mesh: MeshTraceHandle,
    gpu: Option<SphericalGpu>,
    staged: Staged,
    sky_shape: SkyShape,
    track: LightTrack,
}

impl FeaturePass for SphericalPass {
    fn name(&self) -> &str {
        SPHERICAL_GI_NAME
    }

    fn order(&self) -> u32 {
        ORDER
    }

    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        self.mesh.borrow_mut().set_spherical_active(enabled);
        let mut state = self.state.borrow_mut();
        state.pass_enabled = enabled;
        if !enabled {
            lighting.set_spherical_gi(None);
            state.disable();
            self.gpu = None;
            self.staged = Staged::default();
        }
    }

    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let mut state = self.state.borrow_mut();
        let settings = state.settings;
        let view = state.view;
        if state.active() && state.scheduler.is_none() {
            state.reset_grid(settings)?;
        }
        let grid_revision = state.grid_revision;
        let previous = std::mem::take(&mut self.staged);
        let (light_ids, local_lights, scheduler) = {
            let state = &mut *state;
            (
                state.light_ids.clone(),
                &mut state.local_lights,
                &mut state.scheduler,
            )
        };
        let Some(scheduler) = scheduler else {
            ctx.lighting.set_spherical_gi(None);
            return Ok(());
        };
        // A frame that failed after prepare must not retire those faces from the queue.
        if previous.grid_revision == grid_revision {
            for job in previous.jobs {
                scheduler.cancel(job);
            }
        }
        let Some(view) = view.filter(|view| view.index == 0 && settings.enabled) else {
            ctx.lighting.set_spherical_gi(None);
            return Ok(());
        };
        let mut mesh = self.mesh.borrow_mut();
        let reshade_mode = settings.sun_refresh == SphericalSunRefresh::Reshade;
        let environment = ctx.lighting.environment_diffuse_sh().copied();
        let transfer = settings.sky_transfer && environment.is_some();
        let light_revision = light_revision(
            &mut self.sky_shape,
            environment.as_ref(),
            view.light,
            RevisionKey {
                sun: !reshade_mode,
                sky: !reshade_mode,
                transfer: settings.sky_transfer,
            },
        );
        let sun = [view.light[0], view.light[1], view.light[2]];
        let track = &mut self.track;
        if let Some(sh) = &environment {
            let tolerance = if transfer {
                SKY_STEP_TRANSFER
            } else {
                SkyShape::TOLERANCE
            };
            track.sky.update_with(sh, tolerance);
        }
        let sky_epoch = track.sky.epoch;
        let moving = track.previous_sun.is_some_and(|previous| previous != sun);
        track.previous_sun = Some(sun);
        if transfer {
            let mut hash = DefaultHasher::new();
            view.light[..6]
                .iter()
                .map(|x| x.to_bits())
                .for_each(|x| x.hash(&mut hash));
            if let Some(sh) = &environment {
                sh.iter()
                    .flatten()
                    .map(|x| x.to_bits())
                    .for_each(|x| x.hash(&mut hash));
            }
            let hash = hash.finish();
            if track.relight.replace(hash) != Some(hash) {
                track.relight_frames = RELIGHT_FRAMES;
            }
        } else {
            track.relight = None;
            track.relight_frames = 0;
        }
        let capacity = settings.scheduler.pool_slots();
        if self.gpu.as_ref().is_none_or(|gpu| gpu.capacity != capacity) {
            self.gpu = Some(SphericalGpu::new(ctx.gpu, capacity, grid_revision));
            track.relight_frames = RELIGHT_FRAMES;
        }
        let relight = transfer && track.relight_frames > 0;
        track.relight_frames = track.relight_frames.saturating_sub(1);
        let gpu = self.gpu.as_mut().expect("spherical GI pool");
        if gpu.grid_revision != grid_revision {
            gpu.grid_revision = grid_revision;
            gpu.ready_at.fill(None);
            gpu.slot_sun.fill(SlotSun::default());
            gpu.waiting_lights.clear();
            gpu.light_feedback = None;
            gpu.reach_pending.clear();
        }
        gpu.poll_light_feedback(scheduler);
        gpu.poll_reach(scheduler);
        scheduler.set_light_reach(settings.light_reach);
        local_lights.clear();
        local_lights.extend(
            light_ids
                .iter()
                .zip(
                    ctx.lighting
                        .local_lights_bytes()
                        .chunks_exact(raa_shader::LIGHT_GPU_SIZE),
                )
                .filter_map(|(&id, record)| SphericalLight::from_packed(id, record)),
        );
        scheduler.set_view_priority(view.forward, view.half_fov_cos);
        let mirror_priority = mesh.active_mirror_priority() == MirrorPriority::On;
        if mirror_priority {
            mesh.report.poll_pending();
            if let Some(feedback) = mesh.report.reflected_feedback.take().filter(|feedback| {
                feedback.snapshot.grid_revision == grid_revision
                    && feedback.snapshot.configuration == mesh.mirror_priority_configuration
            }) {
                scheduler.set_reflected_receivers(
                    feedback.snapshot.origin_epoch,
                    feedback.snapshot.geometry_revision,
                    &feedback.receivers,
                );
            }
        } else {
            scheduler.clear_reflected_receivers();
        }
        scheduler.update_with_lights(&mesh, view.origin, light_revision, local_lights);
        mesh.reflected_snapshot = mirror_priority.then(|| ReflectedSnapshot {
            grid_revision,
            origin_epoch: scheduler.stats().origin_epoch,
            geometry_revision: mesh.geometry.revision,
            configuration: mesh.mirror_priority_configuration,
        });
        gpu.waiting_lights
            .retain(|sample| captured_face_current(scheduler, sample.job));
        // Added or moved lights: a GPU reach test over the faces ready now (up to four lights;
        // the rest fall back to the faces' own hit bounds).
        let mut reach = None;
        let reach_ids = scheduler.take_reach_requests();
        if !reach_ids.is_empty() {
            let mut indices = Vec::new();
            let mut rest = Vec::new();
            for id in reach_ids {
                match light_ids.iter().position(|&dense| dense == id) {
                    Some(index) if indices.len() < REACH_LIGHTS && index < 256 => {
                        if !indices.contains(&(index as u32)) {
                            indices.push(index as u32);
                        }
                    }
                    _ => rest.push(id),
                }
            }
            if !rest.is_empty() {
                gpu.reach_fallback_faces += u64::from(
                    scheduler.invalidate_light_bounds(&rest, mesh.geometry.world_bounds()),
                );
            }
            if !indices.is_empty() {
                reach = Some(ReachRequest {
                    indices,
                    tokens: scheduler.face_tokens(),
                });
            }
        }
        for slot in scheduler.take_released_slots() {
            gpu.ready_at[slot as usize] = None;
            gpu.slot_sun[slot as usize] = SlotSun::default();
        }
        // An adopted cached probe keeps its projection and fades in again.
        for slot in scheduler.take_adopted_slots() {
            gpu.ready_at[slot as usize] = Some((ctx.frame as u32).max(1));
        }
        gpu.set_screen(ctx.gpu, ctx.size);
        gpu.update_lookup(ctx.gpu, scheduler, settings, ctx.frame as u32);
        ctx.lighting
            .set_spherical_gi(Some((gpu.lookup.clone(), gpu.faces.clone())));
        let burst = scheduler.stats().visible_missing();
        let target = if burst && settings.burst_ms > 0.0 {
            settings.burst_ms
        } else {
            settings.capture_ms
        };
        let jobs = match gpu.clock.as_mut().filter(|_| target > 0.0) {
            Some(clock) => {
                clock.poll();
                let fixed = settings.budget.near + settings.budget.far;
                let pending = scheduler.stats().pending_faces;
                scheduler.take_jobs_shared(clock.faces(target, fixed, pending))
            }
            None => scheduler.take_jobs(settings.budget),
        };
        gpu.face_budget = jobs.len() as u32;
        gpu.burst = burst;
        let reshade = if reshade_mode {
            gpu.plan_reshade(scheduler, settings, sun, sky_epoch, moving)
        } else {
            gpu.reshade_pending = 0;
            Vec::new()
        };
        self.staged = Staged {
            jobs,
            light_ids,
            feedback: None,
            grid_revision,
            frame: ctx.frame,
            light_revision,
            primary: true,
            recorded: false,
            reshade,
            sun,
            sky_epoch,
            transfer,
            relight,
            reach,
        };
        Ok(())
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if !self.staged.primary {
            return Ok(());
        }
        let gpu = self.gpu.as_mut().expect("prepared spherical GI pool");
        let mut state = self.state.borrow_mut();
        state.poll_probe();
        let Some(scheduler) = &state.scheduler else {
            return Ok(());
        };
        let projection = state.settings.projection_key();
        if !self.staged.jobs.is_empty()
            || gpu.projected_channels != projection
            || !self.staged.reshade.is_empty()
            || self.staged.relight
            || self.staged.reach.is_some()
        {
            let mut mesh = self.mesh.borrow_mut();
            let Some(frame) = mesh.frame.clone() else {
                return Ok(());
            };
            let texture_atlas = mesh
                .texture_atlas
                .clone()
                .expect("prepared mesh trace atlas");
            let geometry = mesh.geometry_buffers(ctx.gpu);
            gpu.record(
                ctx,
                scheduler,
                &self.staged.jobs,
                &frame,
                &geometry,
                &texture_atlas,
                mesh.geometry.count(),
                mesh.sky_diffuse,
                projection,
                [state.settings.sky_samples, state.settings.light_samples],
                FrameExtras {
                    reshade: &self.staged.reshade,
                    transfer: self.staged.transfer,
                    relight: self.staged.relight,
                    reach: self.staged.reach.take(),
                    light_threshold: state.settings.light_threshold,
                    inside_capture: state.settings.scheduler.inside_capture,
                },
            );
        }
        if let Some(frame) = self.mesh.borrow().frame.clone() {
            let sky_diffuse = self.mesh.borrow().sky_diffuse;
            gpu.record_scales(ctx, &frame, sky_diffuse);
            gpu.record_resolve(ctx, &frame);
        }
        self.staged.recorded = true;
        self.staged.feedback = gpu.record_light_feedback(ctx);
        let probe_input = if state.probe_requested && state.probe.is_none() {
            scheduler
                .cells()
                .filter(|cell| cell.is_ready())
                .find_map(|cell| {
                    scheduler
                        .contributing_light_ids(cell.key)
                        .map(|ids| (*cell, ids.to_vec()))
                })
        } else {
            None
        };
        if let Some((cell, light_ids)) = probe_input {
            let id = FaceIdentity {
                key: cell.key,
                slot: cell.slot,
                epoch: cell.epoch,
            };
            let probe = gpu.record_probe(ctx, id, cell.probe, cell.light_hash, light_ids);
            state.probe = Some(probe);
        }
        if state.settings.trace_texel != 0 {
            state.trace();
            if state.trace_pending.is_none() {
                state.trace_pending = gpu.record_trace(ctx);
            }
        }
        Ok(())
    }

    fn submitted(&mut self) {
        let staged = std::mem::take(&mut self.staged);
        if !staged.recorded {
            if let Some(scheduler) = self.state.borrow_mut().scheduler.as_mut() {
                for job in staged.jobs {
                    scheduler.cancel(job);
                }
            }
            return;
        }
        let mut state = self.state.borrow_mut();
        let gpu = self.gpu.as_mut().expect("recorded spherical GI pool");
        let Some(scheduler) = &mut state.scheduler else {
            return;
        };
        if let Some(feedback) = staged.feedback {
            gpu.waiting_lights.clear();
            gpu.light_feedback = Some(feedback);
        }
        let mut committed = 0;
        for job in &staged.reshade {
            let st = &mut gpu.slot_sun[job.slot as usize];
            st.valid = true;
            st.epoch = job.epoch;
            if job.groups & 1 != 0 {
                st.dir = staged.sun;
                st.settled = job.alpha >= 1.0;
            }
            if job.groups & 2 != 0 && job.alpha >= 1.0 {
                st.sky_epoch = staged.sky_epoch;
            }
        }
        gpu.reshaded_faces += (FACE_COUNT * staged.reshade.len()) as u64;
        gpu.reshade_frames += u64::from(!staged.reshade.is_empty());
        gpu.relights += u64::from(staged.relight);
        for job in staged.jobs {
            if scheduler.complete(job) {
                // What the slot's sun and sky groups hold: the oldest capture of this epoch.
                let st = &mut gpu.slot_sun[job.slot as usize];
                if !st.valid || st.epoch != job.epoch {
                    *st = SlotSun {
                        valid: true,
                        epoch: job.epoch,
                        dir: staged.sun,
                        settled: true,
                        sky_epoch: staged.sky_epoch,
                    };
                } else if st.dir != staged.sun || st.sky_epoch != staged.sky_epoch {
                    st.settled = false;
                }
                committed += 1;
                gpu.waiting_lights.push(FaceSample {
                    job,
                    ids: staged.light_ids.clone(),
                });
                if scheduler
                    .progress(job.key)
                    .is_some_and(|cell| cell.is_ready())
                {
                    // Frame zero is reserved for "unpublished" in the lookup row.
                    gpu.ready_at[job.slot as usize].get_or_insert((staged.frame as u32).max(1));
                }
            }
        }
        let grid = scheduler.grid();
        let origin = grid.origin();
        let last_shell = grid
            .radial_shell(grid.shell_count() - 1)
            .expect("validated radial grid");
        let scheduler = scheduler.stats();
        let previous = state.stats;
        state.stats = Some(SphericalGiStats {
            scheduler,
            gpu_bytes: gpu.bytes() + state.probe.as_ref().map_or(0, |probe| probe.buffer.size()),
            captured_faces: previous.map_or(0, |s| s.captured_faces) + committed,
            capture_frames: previous.map_or(0, |s| s.capture_frames) + u64::from(committed != 0),
            light_revision: staged.light_revision,
            origin,
            last_shell,
            face_budget: gpu.face_budget,
            burst: gpu.burst,
            capture_gpu_ms: gpu.clock.as_ref().map_or(0.0, |clock| clock.last_ms as f32),
            slot_bytes: gpu.bytes() / u64::from(gpu.capacity.max(1)),
            reshaded_faces: gpu.reshaded_faces,
            reshade_frames: gpu.reshade_frames,
            reshade_pending: gpu.reshade_pending,
            relights: gpu.relights,
            reach_tests: gpu.reach_tests,
            reach_faces: gpu.reach_faces,
            reach_fallback_faces: gpu.reach_fallback_faces,
        });
    }
}

/// Light groups of the probes: the sun group is captured for a white unit sun and the
/// sky group divided by the sky norm (GPU, `spherical_sky_norm`), both scaled back at
/// lookup by `spherical_scales` every frame, so intensity and tint changes of the sun,
/// the sky (sky diffuse, the physical sky's magnitude, which follows the sun intensity)
/// or the ambient recapture nothing. The revision only holds what changes a group's
/// SHAPE: sun direction, environment present, sky dark/lit, the scale-free SH shape
/// (coefficients / luma(DC), changed by more than 0.02) and the trace distance. The
/// other group (local lights, emission) is absolute and invalidated per light.
#[derive(Clone, Debug, Default)]
struct SkyShape {
    shape: Option<[[f32; 3]; 9]>,
    epoch: u64,
}

impl SkyShape {
    const TOLERANCE: f32 = 0.02;

    fn update(&mut self, sh: &[[f32; 4]; 9]) -> bool {
        self.update_with(sh, Self::TOLERANCE)
    }

    fn update_with(&mut self, sh: &[[f32; 4]; 9], tolerance: f32) -> bool {
        let [r, g, b, _] = sh[0];
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        if !(luma.is_finite() && luma > 0.0) {
            return false;
        }
        let shape = sh.map(|[r, g, b, _]| [r / luma, g / luma, b / luma]);
        let moved = self.shape.is_none_or(|old| {
            old.iter()
                .flatten()
                .zip(shape.iter().flatten())
                .any(|(a, b)| (a - b).abs() > tolerance)
        });
        if moved {
            self.shape = Some(shape);
            self.epoch += 1;
        }
        true
    }
}

/// What the revision (full recapture) holds besides environment presence, sky dark/lit and the
/// trace distance: the sun direction and the sky shape unless they are re-shaded in place, and
/// the transfer setting (it changes what a missed texel stores).
#[derive(Clone, Copy, Debug)]
struct RevisionKey {
    sun: bool,
    sky: bool,
    transfer: bool,
}

impl Default for RevisionKey {
    fn default() -> Self {
        Self {
            sun: true,
            sky: true,
            transfer: false,
        }
    }
}

fn light_revision(
    sky: &mut SkyShape,
    environment_sh: Option<&[[f32; 4]; 9]>,
    view: [f32; 10],
    key: RevisionKey,
) -> u64 {
    let mut hash = DefaultHasher::new();
    view[9].to_bits().hash(&mut hash);
    if key.sun {
        [view[0], view[1], view[2]]
            .map(f32::to_bits)
            .hash(&mut hash);
    }
    key.transfer.hash(&mut hash);
    match environment_sh {
        Some(sh) => {
            let lit = sky.update(sh);
            (true, lit, (lit && key.sky).then_some(sky.epoch)).hash(&mut hash);
        }
        None => false.hash(&mut hash),
    }
    // Fits exactly through the public JS number counters; the scheduler still gets 53 bits.
    hash.finish() & ((1 << 53) - 1)
}

fn hash_key(mut key: u32) -> usize {
    key = (key ^ (key >> 16)).wrapping_mul(0x7feb352d);
    key = (key ^ (key >> 15)).wrapping_mul(0x846ca68b);
    (key ^ (key >> 16)) as usize
}

fn encode_key(key: CellKey) -> u32 {
    1 + (u32::from(key.shell) << 15)
        + ((key.face as u32) << 12)
        + (u32::from(key.y) << 6)
        + u32::from(key.x)
}

fn write_words(dst: &mut Vec<u8>, words: [u32; 4]) {
    for word in words {
        dst.extend_from_slice(&word.to_le_bytes());
    }
}

/// Captures start at the free-space probe; the zero normal adds no surface offset.
/// One work entry: probe position, `w` = the cell spacing for a projection entry (the depth
/// moments clamp distances to the DDGI maximum depth, 1.5 * sqrt(3) * spacing), 0 otherwise.
fn write_work(dst: &mut Vec<u8>, slot: u32, probe: [f32; 3], index: u32, spacing: f32) {
    let [x, y, z] = probe;
    write_words(dst, [x, y, z, spacing].map(f32::to_bits));
    write_words(dst, [0.0f32; 4].map(f32::to_bits));
    write_words(dst, [slot, index, 0, 0]);
}

struct SphericalGpu {
    capacity: u32,
    grid_revision: u64,
    capture: wgpu::ComputePipeline,
    project: wgpu::ComputePipeline,
    bins: wgpu::ComputePipeline,
    lights: wgpu::ComputePipeline,
    scales: wgpu::ComputePipeline,
    sky_project: wgpu::ComputePipeline,
    relight: wgpu::ComputePipeline,
    reshade: wgpu::ComputePipeline,
    reach: wgpu::ComputePipeline,
    /// Sky SH header and per-patch sky transfer; reach test output (one word per pool face).
    transfer: Buffer,
    reach_words: Buffer,
    reach_pending: Vec<ReachPending>,
    slot_sun: Vec<SlotSun>,
    reshaded_faces: u64,
    reshade_frames: u64,
    reshade_pending: u32,
    relights: u64,
    reach_tests: u64,
    reach_faces: u64,
    reach_fallback_faces: u64,
    /// Half-resolution screen resolve of the lookup (rows after the slots, forward reads them).
    resolve: wgpu::ComputePipeline,
    /// Resolve size this frame, when it fits the storage binding.
    screen: Option<(u32, u32)>,
    /// Resolve rows the lookup buffer holds after its slots (grows, never shrinks).
    resolve_rows: usize,
    /// Channel mask of the published projections; a change reprojects every probe.
    projected_channels: u32,
    options: Buffer,
    work: Buffer,
    faces: Buffer,
    face_lighting: Buffer,
    lookup: Buffer,
    waiting_lights: Vec<FaceSample>,
    light_feedback: Option<LightFeedback>,
    table_entries: usize,
    lookup_rows: Vec<[u32; 4]>,
    uploaded_rows: Vec<[u32; 4]>,
    lookup_bytes: Vec<u8>,
    work_bytes: Vec<u8>,
    /// Frame of the occupant's first publication, kept while the probe is rekeyed
    /// or relit so the shading fade-in happens once per probe.
    ready_at: Vec<Option<u32>>,
    face_masks: Vec<u8>,
    /// GPU-time budget: timestamps around the capture work, when the device has them.
    clock: Option<CaptureClock>,
    face_budget: u32,
    burst: bool,
}

impl SphericalGpu {
    fn new(gpu: &GpuContext, capacity: u32, grid_revision: u64) -> Self {
        let device = gpu.device();
        let source = spherical_module();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(SPHERICAL_GI_NAME),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gi.spherical.resolve"),
            source: wgpu::ShaderSource::Wgsl(spherical_resolve_module().into()),
        });
        let resolve = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gi.spherical.resolve"),
            layout: None,
            module: &resolve_shader,
            entry_point: Some("spherical_resolve"),
            compilation_options: Default::default(),
            cache: None,
        });
        let slots = capacity as usize;
        let table_entries = (2 * slots).next_power_of_two();
        let lookup_rows = LOOKUP_HEADER + LOOKUP_ENTRY * table_entries;
        let sized = |label, bytes: usize, usage| buffer(device, label, bytes as u64, usage);
        Self {
            capacity,
            grid_revision,
            capture: pipeline("spherical_capture"),
            project: pipeline("spherical_project"),
            bins: pipeline("spherical_bins"),
            lights: pipeline("spherical_lights"),
            scales: pipeline("spherical_scales"),
            sky_project: pipeline("spherical_sky_project"),
            relight: pipeline("spherical_relight"),
            reshade: pipeline("spherical_reshade"),
            reach: pipeline("spherical_reach_test"),
            transfer: sized(
                "gi.spherical.transfer",
                16 * (TRANSFER_HEADER_ROWS + TRANSFER_ROWS_PER_SLOT * slots),
                Usage::STORAGE | Usage::COPY_DST,
            ),
            reach_words: sized(
                "gi.spherical.reach",
                4 * FACE_COUNT * slots,
                Usage::STORAGE | Usage::COPY_SRC,
            ),
            reach_pending: Vec::new(),
            slot_sun: vec![SlotSun::default(); slots],
            reshaded_faces: 0,
            reshade_frames: 0,
            reshade_pending: 0,
            relights: 0,
            reach_tests: 0,
            reach_faces: 0,
            reach_fallback_faces: 0,
            resolve,
            screen: None,
            resolve_rows: 0,
            projected_channels: u32::MAX,
            options: sized("gi.spherical.options", 64, Usage::UNIFORM | Usage::COPY_DST),
            work: sized(
                "gi.spherical.work",
                WORK_BYTES * (2 * FACE_COUNT + 1) * slots,
                Usage::STORAGE | Usage::COPY_DST,
            ),
            faces: sized(
                "gi.spherical.faces",
                CELL_BYTES * slots,
                Usage::STORAGE | Usage::COPY_SRC,
            ),
            face_lighting: sized(
                "gi.spherical.face-lighting",
                FACE_LIGHTING_BYTES * FACE_COUNT * slots,
                Usage::STORAGE | Usage::COPY_SRC,
            ),
            waiting_lights: Vec::new(),
            light_feedback: None,
            lookup: sized(
                "gi.spherical.lookup",
                16 * (lookup_rows + SLOT_ROWS * slots),
                Usage::STORAGE | Usage::COPY_DST | Usage::COPY_SRC,
            ),
            table_entries,
            lookup_rows: vec![[0; 4]; lookup_rows],
            uploaded_rows: Vec::new(),
            lookup_bytes: Vec::new(),
            work_bytes: Vec::new(),
            ready_at: vec![None; slots],
            face_masks: vec![0; slots],
            clock: gpu
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
                .then(|| CaptureClock::new(gpu)),
            face_budget: 0,
            burst: false,
        }
    }

    fn resolve_first_row(&self) -> usize {
        self.lookup_rows.len() + SLOT_ROWS * self.capacity as usize
    }

    /// Size the screen resolve for a frame of `size` pixels. The rows follow the slots in
    /// the one lookup buffer forward binds; a larger frame grows that buffer and copies the
    /// lookup and slots over (no recapture). Off when the rows would exceed the binding limit.
    fn set_screen(&mut self, gpu: &GpuContext, size: (u32, u32)) {
        let width = size.0.div_ceil(RESOLVE_SCALE).min(0xffff);
        let height = size.1.div_ceil(RESOLVE_SCALE).min(0xffff);
        let rows = width as usize * height as usize;
        let first = self.resolve_first_row();
        let bytes = 16 * (first + rows) as u64;
        if rows == 0 || bytes > u64::from(gpu.device().limits().max_storage_buffer_binding_size) {
            self.screen = None;
            return;
        }
        if rows > self.resolve_rows {
            // Headroom so a window that grows a little does not reallocate every frame.
            let grown = rows + rows / 4;
            let total = (16 * (first + grown) as u64).min(u64::from(
                gpu.device().limits().max_storage_buffer_binding_size,
            ));
            let lookup = buffer(
                gpu.device(),
                "gi.spherical.lookup",
                total,
                Usage::STORAGE | Usage::COPY_DST | Usage::COPY_SRC,
            );
            let mut encoder =
                gpu.device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("gi.spherical.lookup-grow"),
                    });
            encoder.copy_buffer_to_buffer(&self.lookup, 0, &lookup, 0, 16 * first as u64);
            gpu.queue().submit([encoder.finish()]);
            self.lookup = lookup;
            self.resolve_rows = (total / 16) as usize - first;
        }
        self.screen = Some((width, height));
    }

    /// Header rows 68/69 (sun and sky group scales) for this frame's lookup. Only the
    /// options words it reads are written (environment flag, sky diffuse), with the same
    /// values a capture of this frame writes.
    fn record_scales(&self, ctx: &mut RecordCtx<'_>, frame: &Buffer, sky_diffuse: f32) {
        let env = ctx.lighting.environment_diffuse_sh().is_some();
        let queue = ctx.gpu.queue();
        queue.write_buffer(&self.options, 8, &u32::from(env).to_le_bytes());
        queue.write_buffer(&self.options, 32, &sky_diffuse.to_bits().to_le_bytes());
        let group = ctx
            .gpu
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gi.spherical.scales"),
                layout: &self.scales.get_bind_group_layout(0),
                entries: &[
                    buffer_entry(0, frame),
                    buffer_entry(1, &self.options),
                    image_entry(14, ctx.lighting.env_specular_view()),
                    wgpu::BindGroupEntry {
                        binding: 15,
                        resource: wgpu::BindingResource::Sampler(ctx.lighting.linear_clamp()),
                    },
                    buffer_entry(23, &self.lookup),
                ],
            });
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gi.spherical.scales"),
                ..Default::default()
            });
        pass.set_pipeline(&self.scales);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    fn record_trace(&self, ctx: &mut RecordCtx<'_>) -> Option<(Buffer, Receiver<Option<Vec<u8>>>)> {
        let (width, height) = self.screen?;
        let first = self.resolve_first_row() + (width * height) as usize;
        if 16 * (first + TRACE_ROWS) as u64 > self.lookup.size() {
            return None;
        }
        let readback = buffer(
            ctx.gpu.device(),
            "gi.spherical.trace",
            16 * TRACE_ROWS as u64,
            Usage::MAP_READ | Usage::COPY_DST,
        );
        ctx.encoder.copy_buffer_to_buffer(
            &self.lookup,
            16 * first as u64,
            &readback,
            0,
            16 * TRACE_ROWS as u64,
        );
        let (send, result) = sync_channel(1);
        let mapped = readback.clone();
        ctx.encoder
            .map_buffer_on_submit(&readback, wgpu::MapMode::Read, .., move |loaded| {
                let bytes = loaded.ok().and_then(|()| {
                    let data = mapped.get_mapped_range(..).ok().map(|bytes| bytes.to_vec());
                    mapped.unmap();
                    data
                });
                let _ = send.send(bytes);
            });
        Some((readback, result))
    }

    fn record_resolve(&self, ctx: &mut RecordCtx<'_>, frame: &Buffer) {
        let Some((width, height)) = self.screen else {
            return;
        };
        let depth = ctx.targets.depth_input(ctx.resources, SPHERICAL_GI_NAME);
        let normal = ctx.targets.normal_input(ctx.resources, SPHERICAL_GI_NAME);
        let group = ctx
            .gpu
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gi.spherical.resolve"),
                layout: &self.resolve.get_bind_group_layout(0),
                entries: &[
                    buffer_entry(0, frame),
                    image_entry(6, &depth),
                    image_entry(7, &normal),
                    buffer_entry(13, ctx.lighting.local_lights_buffer()),
                    buffer_entry(crate::wgsl::MESH_SPHERICAL_BINDING, &self.lookup),
                ],
            });
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gi.spherical.resolve"),
                ..Default::default()
            });
        pass.set_pipeline(&self.resolve);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }

    fn bytes(&self) -> u64 {
        self.options.size()
            + self.transfer.size()
            + self.reach_words.size()
            + self.work.size()
            + self.faces.size()
            + self.face_lighting.size()
            + self.lookup.size()
            + self
                .light_feedback
                .as_ref()
                .map_or(0, |pending| pending.readback.size())
    }

    /// Slots whose sun group no longer matches the sun (rotated by SUN_STEP_DEGREES since their
    /// last shade, or blended) or whose sky group is older than the sky shape: the most visible
    /// and stalest first, `sun_faces / 6` slots per frame. While the sun keeps moving a re-shade
    /// blends by the angle (alpha = angle / SUN_SNAP_DEGREES, at least SUN_MIN_BLEND), so probes
    /// re-shaded on different frames do not pop; once it stops, alpha is 1.
    fn plan_reshade(
        &mut self,
        scheduler: &SphericalScheduler,
        settings: SphericalGiSettings,
        sun: [f32; 3],
        sky_epoch: u64,
        moving: bool,
    ) -> Vec<ReshadeJob> {
        let mut candidates = Vec::new();
        for cell in scheduler.cells() {
            if !cell.is_ready() || !cell.published || self.ready_at[cell.slot as usize].is_none() {
                continue;
            }
            let st = self.slot_sun[cell.slot as usize];
            let (angle, need_sun, need_sky) = if st.valid {
                let angle = angle_degrees(st.dir, sun);
                (
                    angle,
                    angle >= SUN_STEP_DEGREES || !st.settled,
                    st.sky_epoch != sky_epoch,
                )
            } else {
                (180.0, true, true)
            };
            if !(need_sun || need_sky) {
                continue;
            }
            let groups =
                u32::from(need_sun) | (u32::from(need_sky || need_sun && !st.settled) << 1);
            candidates.push((
                scheduler.view_rank(cell.key),
                angle,
                cell.slot,
                cell.epoch,
                groups,
                cell.probe,
                cell.key,
            ));
        }
        self.reshade_pending = candidates.len() as u32;
        candidates.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
        candidates.truncate((settings.sun_faces as usize / FACE_COUNT).max(1));
        candidates
            .into_iter()
            .map(|(_, angle, slot, epoch, groups, probe, key)| ReshadeJob {
                key,
                slot,
                epoch,
                probe,
                groups,
                alpha: if moving && angle < SUN_SNAP_DEGREES {
                    (angle / SUN_SNAP_DEGREES).max(SUN_MIN_BLEND)
                } else {
                    1.0
                },
            })
            .collect()
    }

    /// Reach readbacks that arrived: recapture the faces they flagged.
    fn poll_reach(&mut self, scheduler: &mut SphericalScheduler) {
        let mut i = 0;
        while i < self.reach_pending.len() {
            let words = match self.reach_pending[i].result.try_recv() {
                Ok(words) => words,
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    i += 1;
                    continue;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => None,
            };
            let pending = self.reach_pending.remove(i);
            let Some(words) = words else { continue };
            let mut flagged = Vec::new();
            for token in &pending.tokens {
                let mut bits = 0u8;
                for face in 0..FACE_COUNT {
                    let at = 4 * (FACE_COUNT * token.slot as usize + face);
                    let Some(word) = words.get(at..at + 4) else {
                        continue;
                    };
                    if u32::from_le_bytes(word.try_into().expect("reach word")) != 0 {
                        bits |= 1 << face;
                    }
                }
                if bits & token.ready != 0 {
                    flagged.push((*token, bits));
                }
            }
            self.reach_faces += u64::from(scheduler.invalidate_faces(&flagged));
        }
    }

    fn update_lookup(
        &mut self,
        gpu: &GpuContext,
        scheduler: &SphericalScheduler,
        settings: SphericalGiSettings,
        frame: u32,
    ) {
        let resolve_first_row = self.resolve_first_row() as u32;
        let rows = &mut self.lookup_rows;
        rows.fill([0; 4]);
        let grid = scheduler.grid();
        let config = grid.config();
        let mask = self.table_entries - 1;
        let [x, y, z] = grid.origin();
        rows[0] = [x, y, z, settings.intensity].map(f32::to_bits);
        rows[1] = [
            grid.angular_resolution().into(),
            grid.shell_count().into(),
            mask as u32,
            settings.debug as u32,
        ];
        // The frame drives both the age diagnostic and the probe fade-in.
        rows[2] = [
            config.clip_start.to_bits(),
            config.z_max.to_bits(),
            frame,
            settings.channels | (u32::from(settings.screen_ao) << 16),
        ];
        rows[3][1] = settings.parallax_range.to_bits();
        if let Some((width, height)) = self.screen {
            rows[3][2] = resolve_first_row;
            rows[3][3] = width | (height << 16);
        }
        for index in 0..grid.shell_count() {
            let shell = grid.radial_shell(index).expect("validated radial shell");
            rows[4 + usize::from(index)] =
                [shell.inner, shell.center, shell.outer, 0.0].map(f32::to_bits);
        }
        // Row 4 w: sphere-only direct near-sun radius (0 = off, else at least 1e-6 m).
        if settings.sphere_direct {
            rows[4][3] = settings.direct_sun_range.max(1e-6).to_bits();
        }
        // Row 5 w: normal-side probe selection (frontProbes).
        rows[5][3] = u32::from(settings.front_probes);
        // Row 6 w: traced resolve texel (diagnostic).
        rows[6][3] = settings.trace_texel;
        for cell in scheduler.cells() {
            let key = encode_key(cell.key);
            let mut hash = hash_key(key) & mask;
            let row = loop {
                let row = LOOKUP_HEADER + LOOKUP_ENTRY * hash;
                if rows[row][0] == 0 {
                    break row;
                }
                hash = (hash + 1) & mask;
            };
            if cell.key.is_center() {
                rows[3][0] = row as u32;
            }
            let ready_frame = self.ready_at[cell.slot as usize].unwrap_or(0);
            let (radial, angular) = grid.cell_extent(cell.key);
            let spacing = radial.max(angular) as f32;
            let hash = cell.light_hash;
            rows[row] = [key, cell.slot, ready_frame, (hash ^ (hash >> 32)) as u32];
            let [x, y, z] = cell.probe;
            rows[row + 1] = [x, y, z, spacing].map(f32::to_bits);
            let published = cell.published && ready_frame != 0;
            rows[row + 2] = [0, 0, 0, u32::from(published)];
        }
        let from = rows
            .iter()
            .zip(&self.uploaded_rows)
            .take_while(|(a, b)| a == b)
            .count();
        if from == rows.len() {
            return;
        }
        let to = if self.uploaded_rows.is_empty() {
            rows.len()
        } else {
            rows.iter()
                .zip(&self.uploaded_rows)
                .rposition(|(a, b)| a != b)
                .expect("changed lookup")
                + 1
        };
        self.lookup_bytes.clear();
        for row in &rows[from..to] {
            write_words(&mut self.lookup_bytes, *row);
        }
        gpu.queue()
            .write_buffer(&self.lookup, (16 * from) as u64, &self.lookup_bytes);
        self.uploaded_rows.clone_from(rows);
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        ctx: &mut RecordCtx<'_>,
        scheduler: &SphericalScheduler,
        jobs: &[FaceJob],
        frame: &Buffer,
        geometry: &[Buffer; 3],
        texture_atlas: &wgpu::TextureView,
        triangle_count: u32,
        sky_diffuse: f32,
        projection: u32,
        samples: [u32; 2],
        extras: FrameExtras<'_>,
    ) {
        let device = ctx.gpu.device();
        let channels = projection & 7;
        let direct = projection & 0x100 != 0;
        self.work_bytes.clear();
        for &job in jobs {
            write_work(
                &mut self.work_bytes,
                job.slot,
                job.probe,
                job.face as u32,
                0.0,
            );
        }
        let slot_index = |rows: usize, slot: u32| (rows + SLOT_ROWS * slot as usize) as u32;
        let spacing = |key| {
            let (radial, angular) = scheduler.grid().cell_extent(key);
            radial.max(angular) as f32
        };
        self.face_masks.fill(0);
        let mut projected = 0;
        let mut projected_slots = Vec::new();
        for &job in jobs {
            let mask = &mut self.face_masks[job.slot as usize];
            let ready = scheduler
                .progress(job.key)
                .expect("scheduled cell")
                .ready_faces;
            let bit = 1 << job.face as u8;
            if (ready | *mask) != ALL_FACES && (ready | *mask | bit) == ALL_FACES {
                let index = slot_index(self.lookup_rows.len(), job.slot);
                write_work(
                    &mut self.work_bytes,
                    job.slot,
                    job.probe,
                    index,
                    spacing(job.key),
                );
                projected_slots.push(job.slot);
                projected += 1;
            }
            *mask |= bit;
        }
        // A channel-mask change reprojects every published probe from its faces.
        if projection != self.projected_channels {
            for cell in scheduler.cells() {
                if cell.published && cell.is_ready() && !projected_slots.contains(&cell.slot) {
                    let index = slot_index(self.lookup_rows.len(), cell.slot);
                    write_work(
                        &mut self.work_bytes,
                        cell.slot,
                        cell.probe,
                        index,
                        spacing(cell.key),
                    );
                    projected_slots.push(cell.slot);
                    projected += 1;
                }
            }
            self.projected_channels = projection;
        }
        // Re-shaded slots are reprojected after the re-shade (their work entries come last).
        for job in extras.reshade {
            if projected_slots.contains(&job.slot) {
                continue;
            }
            let index = slot_index(self.lookup_rows.len(), job.slot);
            write_work(
                &mut self.work_bytes,
                job.slot,
                job.probe,
                index,
                spacing(job.key),
            );
            projected_slots.push(job.slot);
            projected += 1;
        }
        let reshade_first = (self.work_bytes.len() / WORK_BYTES) as u32;
        for job in extras.reshade {
            for face in 0..FACE_COUNT as u32 {
                write_words(&mut self.work_bytes, [0; 4]);
                write_words(&mut self.work_bytes, [0; 4]);
                write_words(
                    &mut self.work_bytes,
                    [job.slot, face, job.alpha.to_bits(), job.groups],
                );
            }
        }
        let reshade_jobs = FACE_COUNT as u32 * extras.reshade.len() as u32;
        if jobs.is_empty()
            && projected == 0
            && reshade_jobs == 0
            && !extras.relight
            && extras.reach.is_none()
        {
            return;
        }
        let env = ctx.lighting.environment_diffuse_sh().is_some();
        let mut options = Vec::with_capacity(64);
        let slot_base = self.lookup_rows.len() as u32;
        write_words(
            &mut options,
            [triangle_count, jobs.len() as u32, u32::from(env), slot_base],
        );
        write_words(
            &mut options,
            [
                channels,
                u32::from(extras.transfer) | u32::from(extras.inside_capture) << 1,
                reshade_first,
                reshade_jobs,
            ],
        );
        let end = scheduler
            .grid()
            .radial_shell(scheduler.grid().shell_count() - 1)
            .expect("last shell")
            .outer;
        write_words(
            &mut options,
            [
                sky_diffuse.to_bits(),
                end.to_bits(),
                SUN_CONE_DEGREES.to_radians().cos().to_bits(),
                extras.light_threshold.to_bits(),
            ],
        );
        // The shared local-light integrator uses its one-tap area estimator for this capture;
        // y, z: sky and local-light samples per hit texel.
        write_words(&mut options, [1, samples[0], samples[1], 0]);
        let queue = ctx.gpu.queue();
        queue.write_buffer(&self.options, 0, &options);
        queue.write_buffer(&self.work, 0, &self.work_bytes);
        // Time the whole capture (capture, project, bins) of one frame when no sample is pending.
        let clock = self
            .clock
            .as_ref()
            .filter(|clock| clock.pending.is_none() && !jobs.is_empty());
        let last_pass = if projected == 0 { 0 } else { 2 };
        // WebGPU rejects timestampWrites that name neither index, so a middle pass gets none.
        let stamps = |pass: usize| {
            clock
                .filter(|_| pass == 0 || pass == last_pass)
                .map(|clock| wgpu::ComputePassTimestampWrites {
                    query_set: &clock.queries,
                    beginning_of_pass_write_index: (pass == 0).then_some(0),
                    end_of_pass_write_index: (pass == last_pass).then_some(1),
                })
        };
        // The sky SH and norm of this frame, before anything reads the transfer header.
        if extras.transfer
            && (!jobs.is_empty() || projected > 0 || reshade_jobs > 0 || extras.relight)
        {
            self.dispatch(
                ctx,
                &self.sky_project,
                "gi.spherical.sky-project",
                &[
                    buffer_entry(0, frame),
                    buffer_entry(1, &self.options),
                    image_entry(14, ctx.lighting.env_specular_view()),
                    wgpu::BindGroupEntry {
                        binding: 15,
                        resource: wgpu::BindingResource::Sampler(ctx.lighting.linear_clamp()),
                    },
                    buffer_entry(34, &self.transfer),
                ],
                (1, 1, 1),
            );
        }
        if reshade_jobs > 0 {
            self.dispatch(
                ctx,
                &self.reshade,
                "gi.spherical.reshade",
                &[
                    buffer_entry(0, frame),
                    buffer_entry(1, &self.options),
                    buffer_entry(2, &geometry[0]),
                    buffer_entry(3, &geometry[1]),
                    image_entry(14, ctx.lighting.env_specular_view()),
                    wgpu::BindGroupEntry {
                        binding: 15,
                        resource: wgpu::BindingResource::Sampler(ctx.lighting.linear_clamp()),
                    },
                    buffer_entry(21, &self.work),
                    buffer_entry(22, &self.faces),
                    buffer_entry(34, &self.transfer),
                ],
                (1, 1, reshade_jobs),
            );
        }
        if !jobs.is_empty() {
            let capture_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gi.spherical.capture"),
                layout: &self.capture.get_bind_group_layout(0),
                entries: &[
                    buffer_entry(0, frame),
                    buffer_entry(1, &self.options),
                    buffer_entry(2, &geometry[0]),
                    buffer_entry(3, &geometry[1]),
                    buffer_entry(4, &geometry[2]),
                    buffer_entry(13, ctx.lighting.local_lights_buffer()),
                    image_entry(14, ctx.lighting.env_specular_view()),
                    wgpu::BindGroupEntry {
                        binding: 15,
                        resource: wgpu::BindingResource::Sampler(ctx.lighting.linear_clamp()),
                    },
                    buffer_entry(21, &self.work),
                    buffer_entry(22, &self.faces),
                    buffer_entry(24, &self.face_lighting),
                    buffer_entry(34, &self.transfer),
                    image_entry(crate::wgsl::MESH_TRACE_ATLAS_BINDING, texture_atlas),
                ],
            });
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gi.spherical.capture"),
                    timestamp_writes: stamps(0),
                });
            pass.set_pipeline(&self.capture);
            pass.set_bind_group(0, &capture_group, &[]);
            pass.dispatch_workgroups(1, 1, jobs.len() as u32);
        }
        if projected > 0 {
            self.record_projection(ctx, projected, &stamps);
            if direct {
                self.record_lights(ctx, projected, frame, geometry);
            }
        }
        if extras.relight {
            let patches = (TRANSFER_ROWS_PER_SLOT / 2) as u32 * self.capacity;
            self.dispatch(
                ctx,
                &self.relight,
                "gi.spherical.relight",
                &[
                    buffer_entry(1, &self.options),
                    buffer_entry(23, &self.lookup),
                    buffer_entry(34, &self.transfer),
                ],
                (patches.div_ceil(64), 1, 1),
            );
        }
        if clock.is_some() {
            let faces = jobs.len() as u32;
            self.clock
                .as_mut()
                .expect("capture clock")
                .resolve(ctx.encoder, faces);
        }
        if let Some(reach) = extras.reach {
            self.record_reach(ctx, reach, frame, geometry);
        }
    }

    fn record_projection<'q>(
        &self,
        ctx: &mut RecordCtx<'_>,
        projected: u32,
        stamps: &dyn Fn(usize) -> Option<wgpu::ComputePassTimestampWrites<'q>>,
    ) {
        let device = ctx.gpu.device();
        for (index, (pipeline, label, groups)) in [
            (&self.project, "gi.spherical.project", 1 + FACE_COUNT as u32),
            (&self.bins, "gi.spherical.bins", FACE_COUNT as u32),
        ]
        .into_iter()
        .enumerate()
        {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    buffer_entry(1, &self.options),
                    buffer_entry(21, &self.work),
                    buffer_entry(22, &self.faces),
                    buffer_entry(23, &self.lookup),
                    buffer_entry(34, &self.transfer),
                ],
            });
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: stamps(index + 1),
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, projected * groups);
        }
    }

    fn dispatch(
        &self,
        ctx: &mut RecordCtx<'_>,
        pipeline: &wgpu::ComputePipeline,
        label: &str,
        entries: &[wgpu::BindGroupEntry<'_>],
        groups: (u32, u32, u32),
    ) {
        let group = ctx
            .gpu
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &pipeline.get_bind_group_layout(0),
                entries,
            });
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                ..Default::default()
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
    }

    /// Reach test of up to four lights over every pool face's stored hits, read back
    /// asynchronously (`poll_reach`).
    fn record_reach(
        &mut self,
        ctx: &mut RecordCtx<'_>,
        reach: ReachRequest,
        frame: &Buffer,
        geometry: &[Buffer; 3],
    ) {
        let mut words = [0u32; 8];
        for (j, &index) in reach.indices.iter().take(REACH_LIGHTS).enumerate() {
            words[j] = index;
        }
        words[4] = reach.indices.len().min(REACH_LIGHTS) as u32;
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        ctx.gpu
            .queue()
            .write_buffer(&self.transfer, (16 * TRANSFER_REACH_ROW) as u64, &bytes);
        let faces = FACE_COUNT as u32 * self.capacity;
        self.dispatch(
            ctx,
            &self.reach,
            "gi.spherical.reach",
            &[
                buffer_entry(0, frame),
                buffer_entry(1, &self.options),
                buffer_entry(2, &geometry[0]),
                buffer_entry(3, &geometry[1]),
                buffer_entry(13, ctx.lighting.local_lights_buffer()),
                buffer_entry(22, &self.faces),
                buffer_entry(34, &self.transfer),
                buffer_entry(35, &self.reach_words),
            ],
            (faces, 1, 1),
        );
        let size = self.reach_words.size();
        let readback = buffer(
            ctx.gpu.device(),
            "gi.spherical.reach-readback",
            size,
            Usage::MAP_READ | Usage::COPY_DST,
        );
        ctx.encoder
            .copy_buffer_to_buffer(&self.reach_words, 0, &readback, 0, size);
        let (send, result) = sync_channel(1);
        let mapped = readback.clone();
        ctx.encoder
            .map_buffer_on_submit(&readback, wgpu::MapMode::Read, .., move |loaded| {
                let bytes = loaded.ok().and_then(|()| {
                    let data = mapped.get_mapped_range(..).ok().map(|bytes| bytes.to_vec());
                    mapped.unmap();
                    data
                });
                let _ = send.send(bytes);
            });
        self.reach_tests += 1;
        self.reach_pending.push(ReachPending {
            tokens: reach.tokens,
            _readback: readback,
            result,
        });
    }

    /// Sphere-only direct: per projected probe, the sun visibility and the two strongest
    /// local lights at the probe with their visibility (16 shadow rays each).
    fn record_lights(
        &self,
        ctx: &mut RecordCtx<'_>,
        projected: u32,
        frame: &Buffer,
        geometry: &[Buffer; 3],
    ) {
        let group = ctx
            .gpu
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gi.spherical.lights"),
                layout: &self.lights.get_bind_group_layout(0),
                entries: &[
                    buffer_entry(0, frame),
                    buffer_entry(1, &self.options),
                    buffer_entry(2, &geometry[0]),
                    buffer_entry(3, &geometry[1]),
                    buffer_entry(4, &geometry[2]),
                    buffer_entry(13, ctx.lighting.local_lights_buffer()),
                    buffer_entry(21, &self.work),
                    buffer_entry(23, &self.lookup),
                ],
            });
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gi.spherical.lights"),
                ..Default::default()
            });
        pass.set_pipeline(&self.lights);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, projected);
    }

    fn record_light_feedback(&self, ctx: &mut RecordCtx<'_>) -> Option<LightFeedback> {
        if self.light_feedback.is_some() || self.waiting_lights.is_empty() {
            return None;
        }
        let jobs = self.waiting_lights.clone();
        let size = (FACE_LIGHTING_BYTES * jobs.len()) as u64;
        let readback = buffer(
            ctx.gpu.device(),
            "gi.spherical.light-feedback",
            size,
            Usage::MAP_READ | Usage::COPY_DST,
        );
        for (i, sample) in jobs.iter().enumerate() {
            let source = u64::from(sample.job.slot) * FACE_COUNT as u64 + sample.job.face as u64;
            ctx.encoder.copy_buffer_to_buffer(
                &self.face_lighting,
                source * FACE_LIGHTING_BYTES as u64,
                &readback,
                (i * FACE_LIGHTING_BYTES) as u64,
                FACE_LIGHTING_BYTES as u64,
            );
        }
        let (send, result) = sync_channel(1);
        let mapped = readback.clone();
        ctx.encoder
            .map_buffer_on_submit(&readback, wgpu::MapMode::Read, .., move |loaded| {
                let bytes = loaded.ok().and_then(|()| {
                    let data = mapped.get_mapped_range(..).ok().map(|bytes| bytes.to_vec());
                    mapped.unmap();
                    data
                });
                let _ = send.send(bytes);
            });
        Some(LightFeedback {
            jobs,
            readback,
            result,
        })
    }

    fn poll_light_feedback(&mut self, scheduler: &mut SphericalScheduler) {
        let Some(feedback) = &self.light_feedback else {
            return;
        };
        let bytes = match feedback.result.try_recv() {
            Ok(bytes) => bytes,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => None,
        };
        let feedback = self.light_feedback.take().expect("pending light feedback");
        if let Some(bytes) = bytes {
            let mut ids = Vec::new();
            for (sample, metadata) in feedback
                .jobs
                .iter()
                .zip(bytes.chunks_exact(FACE_LIGHTING_BYTES))
            {
                let hit_bounds = decode_face_lights(metadata, &sample.ids, &mut ids);
                scheduler.record_face_lighting(sample.job, hit_bounds, &ids);
            }
        } else {
            // The next prepare removes stale jobs from this retry queue.
            self.waiting_lights.extend(feedback.jobs);
        }
    }

    fn record_probe(
        &self,
        ctx: &mut RecordCtx<'_>,
        id: FaceIdentity,
        origin: [f32; 3],
        hash: u64,
        ids: Vec<u64>,
    ) -> Probe {
        let readback = buffer(
            ctx.gpu.device(),
            "gi.spherical.probe",
            CELL_BYTES as u64,
            Usage::MAP_READ | Usage::COPY_DST,
        );
        ctx.encoder.copy_buffer_to_buffer(
            &self.faces,
            u64::from(id.slot) * CELL_BYTES as u64,
            &readback,
            0,
            CELL_BYTES as u64,
        );
        let (send, result) = sync_channel(1);
        let frame = ctx.frame;
        let mapped = readback.clone();
        ctx.encoder
            .map_buffer_on_submit(&readback, wgpu::MapMode::Read, .., move |loaded| {
                let report = loaded.ok().and_then(|()| {
                    let data = mapped
                        .get_mapped_range(..)
                        .ok()
                        .map(|bytes| inspect_probe(&bytes, id.slot, frame, origin, hash, ids));
                    mapped.unmap();
                    data
                });
                let _ = send.send(report);
            });
        Probe {
            id,
            buffer: readback,
            result,
        }
    }
}

/// GPU-time face budget. Two timestamps bracket one frame's capture work (capture,
/// project, bins); the samples fit an affine cost `a + b * faces` and a credit of
/// target milliseconds per frame pays for whole batches (see [`CostModel`]).
struct CaptureClock {
    queries: wgpu::QuerySet,
    resolve: Buffer,
    readback: Buffer,
    ms_per_tick: f64,
    pending: Option<(u32, Receiver<Option<f64>>)>,
    model: CostModel,
    last_ms: f64,
}

const MIN_CLOCK_FACES: u32 = 6;

/// Capture time is latency bound: one face of 64 rays through a large BVH with a
/// shadow ray per light costs about as much as a hundred faces, so the fixed term
/// `a` can exceed a small per-frame target. The credit then accumulates over a few
/// frames and pays for one large batch, keeping the average at the target instead
/// of trickling a handful of faces per frame.
#[derive(Clone, Copy, Debug, Default)]
struct CostModel {
    s1: f64,
    sn: f64,
    sm: f64,
    snn: f64,
    snm: f64,
    max_faces: u32,
    credit: f64,
}

impl CostModel {
    const DECAY: f64 = 0.8;

    fn observe(&mut self, faces: u32, ms: f64) {
        let n = f64::from(faces);
        for sum in [
            &mut self.s1,
            &mut self.sn,
            &mut self.sm,
            &mut self.snn,
            &mut self.snm,
        ] {
            *sum *= Self::DECAY;
        }
        self.s1 += 1.0;
        self.sn += n;
        self.sm += ms;
        self.snn += n * n;
        self.snm += n * ms;
        self.max_faces = self.max_faces.max(faces);
    }

    /// Fixed and per-face milliseconds; without spread in batch sizes, all per face.
    fn cost(&self) -> Option<(f64, f64)> {
        if self.s1 <= 0.0 || self.sn <= 0.0 {
            return None;
        }
        let spread = self.s1 * self.snn - self.sn * self.sn;
        let (a, b) = if spread > 1e-3 * self.s1 * self.snn {
            let b = (self.s1 * self.snm - self.sn * self.sm) / spread;
            (((self.sm - b * self.sn) / self.s1), b)
        } else {
            (0.0, self.sm / self.sn)
        };
        let b = b.max(1e-5);
        Some((a.clamp(0.0, self.sm / self.s1), b))
    }

    /// Faces to issue this frame: whatever the accumulated credit pays for, at most
    /// four times the largest measured batch, none while the credit is short.
    fn faces(&mut self, target_ms: f64, fixed: u32, pending: u32) -> u32 {
        let Some((a, b)) = self.cost() else {
            return fixed.min(pending);
        };
        // The ceiling always fits one minimal batch: with a fixed cost above four
        // targets (a slow first measurement) the credit could otherwise never pay for
        // a batch, no capture would run and the model would never be re-measured.
        // The ceiling pays for one fixed-size batch (and a margin, so that rounding
        // in the division below cannot land one face short of the minimum).
        let batch = a + b * f64::from(fixed.max(MIN_CLOCK_FACES)) + 1e-6;
        self.credit = (self.credit + target_ms).min((4.0 * target_ms).max(batch));
        if pending == 0 {
            return 0;
        }
        let affordable = ((self.credit - a) / b).floor().max(0.0) as u32;
        let cap = 4 * self.max_faces.max(fixed).max(MIN_CLOCK_FACES);
        let faces = affordable.min(cap).min(MAX_FACES_PER_BUDGET).min(pending);
        if faces < MIN_CLOCK_FACES.min(pending) {
            return 0;
        }
        self.credit -= a + b * f64::from(faces);
        faces
    }
}

impl CaptureClock {
    fn new(gpu: &GpuContext) -> Self {
        let device = gpu.device();
        let size = 2 * u64::from(wgpu::QUERY_SIZE);
        Self {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("gi.spherical.clock"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }),
            resolve: buffer(
                device,
                "gi.spherical.clock",
                size,
                Usage::QUERY_RESOLVE | Usage::COPY_SRC,
            ),
            readback: buffer(
                device,
                "gi.spherical.clock-readback",
                size,
                Usage::MAP_READ | Usage::COPY_DST,
            ),
            ms_per_tick: f64::from(gpu.queue().get_timestamp_period()) / 1e6,
            pending: None,
            model: CostModel::default(),
            last_ms: 0.0,
        }
    }

    fn faces(&mut self, target_ms: f32, fixed: u32, pending: u32) -> u32 {
        self.model.faces(f64::from(target_ms), fixed, pending)
    }

    fn poll(&mut self) {
        let Some((faces, result)) = &self.pending else {
            return;
        };
        let ms = match result.try_recv() {
            Ok(ms) => ms,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => None,
        };
        let faces = *faces;
        self.pending = None;
        if let Some(ms) = ms.filter(|ms| ms.is_finite() && *ms > 0.0) {
            self.last_ms = ms;
            self.model.observe(faces, ms);
        }
    }

    fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder, faces: u32) {
        let bytes = 2 * u64::from(wgpu::QUERY_SIZE);
        encoder.resolve_query_set(&self.queries, 0..2, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, bytes);
        let (send, result) = sync_channel(1);
        let readback = self.readback.clone();
        let ms_per_tick = self.ms_per_tick;
        encoder.map_buffer_on_submit(&self.readback, wgpu::MapMode::Read, .., move |loaded| {
            let ms = loaded.ok().and_then(|()| {
                let ms = readback.get_mapped_range(..).ok().map(|data| {
                    let tick = |i: usize| {
                        u64::from_le_bytes(data[8 * i..8 * i + 8].try_into().expect("timestamp"))
                    };
                    tick(1).saturating_sub(tick(0)) as f64 * ms_per_tick
                });
                readback.unmap();
                ms
            });
            let _ = send.send(ms);
        });
        self.pending = Some((faces, result));
    }
}

/// Per-frame work beyond the capture jobs (see `SphericalGpu::record`).
struct FrameExtras<'a> {
    reshade: &'a [ReshadeJob],
    transfer: bool,
    relight: bool,
    reach: Option<ReachRequest>,
    light_threshold: f32,
    inside_capture: bool,
}

struct ReachPending {
    tokens: Vec<FaceToken>,
    _readback: Buffer,
    result: Receiver<Option<Vec<u8>>>,
}

fn angle_degrees(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dot: f32 = (0..3).map(|i| a[i] * b[i]).sum();
    let la = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let lb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if la <= 0.0 || lb <= 0.0 {
        return 0.0;
    }
    (dot / (la * lb)).clamp(-1.0, 1.0).acos().to_degrees()
}

#[derive(Clone)]
struct FaceSample {
    job: FaceJob,
    ids: Rc<[u64]>,
}

struct LightFeedback {
    jobs: Vec<FaceSample>,
    readback: Buffer,
    result: Receiver<Option<Vec<u8>>>,
}

fn captured_face_current(scheduler: &SphericalScheduler, job: FaceJob) -> bool {
    scheduler.progress(job.key).is_some_and(|cell| {
        cell.slot == job.slot
            && cell.epoch == job.epoch
            && cell.ready_faces & (1 << job.face as u8) != 0
    })
}

fn decode_face_lights(
    bytes: &[u8],
    handles: &[u64],
    ids: &mut Vec<u64>,
) -> Option<([f32; 3], [f32; 3])> {
    let word = |i: usize| {
        u32::from_le_bytes(
            bytes[4 * i..4 * i + 4]
                .try_into()
                .expect("face lighting word"),
        )
    };
    ids.clear();
    for (dense, &handle) in handles.iter().enumerate() {
        if word(dense / 32) & (1 << (dense % 32)) != 0 {
            ids.push(handle);
        }
    }
    let flags = word(15);
    if flags & 1 != 0 {
        ids.push(SKY_LIGHT_ID);
    }
    if flags & 2 != 0 {
        ids.push(SUN_LIGHT_ID);
    }
    let ordered_float = |i| {
        let bits = word(i);
        f32::from_bits(if bits & 0x8000_0000 == 0 {
            !bits
        } else {
            bits ^ 0x8000_0000
        })
    };
    (word(11) != 0).then(|| {
        (
            std::array::from_fn(|i| ordered_float(8 + i)),
            std::array::from_fn(|i| ordered_float(12 + i)),
        )
    })
}

/// The screen resolve runs at 1/RESOLVE_SCALE of the frame on each axis
/// (RAA_SPHERICAL_RESOLVE_SCALE in spherical_gi.wgsl).
const RESOLVE_SCALE: u32 = 2;

/// The resolve kernel: the mesh-trace frame uniform, the G-buffer depth and normal, and the
/// lookup buffer read-write (it reads probes and writes its own rows), with the shared lookup.
fn spherical_resolve_module() -> String {
    let mesh = include_str!("../wgsl/mesh_trace.wgsl");
    let block = |name: &str| {
        let start = mesh.find(name).expect("mesh trace struct");
        let end = start + mesh[start..].find("\n}\n").expect("struct end") + 3;
        &mesh[start..end]
    };
    format!(
        "{}{}\n@group(0) @binding(0) var<uniform> frame: Frame;\n\
         @group(0) @binding(13) var<uniform> lights: array<Light, 256>;\n\
         @group(0) @binding(6) var depth: texture_depth_2d;\n\
         @group(0) @binding(7) var normals: texture_2d<f32>;\n\
         @group(0) @binding({}) var<storage, read_write> raa_spherical_data: array<vec4<u32>>;\n\
         {}{}{}",
        block("struct Frame {"),
        block("struct Light {"),
        crate::wgsl::MESH_SPHERICAL_BINDING,
        raa_shader::SPHERICAL_CUBE_WGSL,
        raa_shader::SPHERICAL_GI_WGSL.replace(
            "fn raa_spherical_trace(row: u32, value: vec4<u32>) {}",
            "fn raa_spherical_trace(row: u32, value: vec4<u32>) {\n\
             \x20   let size = raa_spherical_data[3].w;\n\
             \x20   let first = raa_spherical_data[3].z + (size & 0xffffu) * (size >> 16u);\n\
             \x20   if first + row < arrayLength(&raa_spherical_data) { raa_spherical_data[first + row] = value; }\n\
             }",
        ),
        include_str!("../wgsl/mesh_spherical_resolve.wgsl"),
    )
}

/// Rows after the screen resolve rows written by the diagnostic trace (traceTexel).
const TRACE_ROWS: usize = 8 + 4 * 24 + 2 * 24;

fn spherical_module() -> String {
    format!(
        "{}\n{}",
        crate::wgsl::mesh_trace_module(),
        include_str!("../wgsl/mesh_spherical_gi.wgsl"),
    )
}

#[derive(Clone, Copy)]
struct FaceIdentity {
    key: CellKey,
    slot: u32,
    epoch: u64,
}

impl FaceIdentity {
    fn current(self, scheduler: &SphericalScheduler) -> bool {
        scheduler.progress(self.key).is_some_and(|cell| {
            cell.slot == self.slot && cell.epoch == self.epoch && cell.is_ready()
        })
    }
}

struct Probe {
    id: FaceIdentity,
    buffer: Buffer,
    result: Receiver<Option<SphericalGiProbe>>,
}

fn inspect_probe(
    bytes: &[u8],
    slot: u32,
    frame: u64,
    origin: [f32; 3],
    light_hash: u64,
    contributing_light_ids: Vec<u64>,
) -> SphericalGiProbe {
    let mut report = SphericalGiProbe {
        slot,
        frame,
        face_hits: [0; FACE_COUNT],
        face_valid_geometry: [0; FACE_COUNT],
        face_valid_misses: [0; FACE_COUNT],
        channel_sums: [[0.0; 3]; CHANNELS],
        light_hash,
        contributing_light_ids,
    };
    for (i, texel) in bytes.chunks_exact(TEXEL_BYTES).enumerate() {
        let f32_at =
            |offset| f32::from_le_bytes(texel[offset..offset + 4].try_into().expect("capture f32"));
        let f16_at = |offset| {
            half::f16::from_le_bytes(texel[offset..offset + 2].try_into().expect("capture f16"))
                .to_f32()
        };
        let face = i / TEXELS_PER_FACE;
        let px = i % TEXELS_PER_FACE;
        let normal = [f16_at(40), f16_at(42), f16_at(44)];
        let position = [f32_at(0), f32_at(4), f32_at(8)];
        if (1.0..2.0).contains(&f32_at(12)) {
            report.face_hits[face] += 1;
            let [u, v] = [(px % 8) as f32, (px / 8) as f32].map(|x| (x + 0.5) * 0.25 - 1.0);
            let expected = match face {
                0 => [1.0, v, -u],
                1 => [-1.0, v, u],
                2 => [u, 1.0, -v],
                3 => [u, -1.0, v],
                4 => [u, v, 1.0],
                _ => [-u, v, -1.0],
            };
            let offset = raa_math::Vec3::from_array(position) - raa_math::Vec3::from_array(origin);
            let axis = raa_math::Vec3::from_array(expected).normalize_or_zero();
            let forward = offset.normalize_or_zero().dot(axis);
            let length = raa_math::Vec3::from_array(normal).length();
            if offset.length() >= 0.0009 && forward > 0.999 && (0.98..=1.02).contains(&length) {
                report.face_valid_geometry[face] += 1;
            }
        } else if f32_at(12) == 0.0 && position == [0.0; 3] && normal == [0.0; 3] {
            report.face_valid_misses[face] += 1;
        }
        for (group, rgb) in report.channel_sums.iter_mut().enumerate() {
            for (lane, total) in rgb.iter_mut().enumerate() {
                *total += f16_at(16 + 8 * group + 2 * lane);
            }
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_feedback_decodes_dense_indices_and_signed_world_bounds() {
        let ordered = |f: f32| {
            let b = f.to_bits();
            if f.is_sign_negative() {
                !b
            } else {
                b ^ 0x8000_0000
            }
        };
        let mut words = [0u32; 16];
        words[0] = 1 << 31;
        words[1] = 1;
        words[8..11].copy_from_slice(&[-3.0, -0.5, 10.0].map(ordered));
        words[11] = 5;
        words[12..15].copy_from_slice(&[2.0, 0.0, 80.0].map(ordered));
        words[15] = 3;
        let bytes = words
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let handles = (501..=540).collect::<Vec<_>>();
        let mut ids = Vec::new();
        assert_eq!(
            decode_face_lights(&bytes, &handles, &mut ids),
            Some(([-3.0, -0.5, 10.0], [2.0, 0.0, 80.0]))
        );
        assert_eq!(ids, [532, 533, SKY_LIGHT_ID, SUN_LIGHT_ID]);
    }

    #[test]
    fn latency_bound_capture_pays_large_batches_from_accumulated_credit() {
        let mut model = CostModel::default();
        // Before any sample the fixed budget applies.
        assert_eq!(model.faces(0.75, 144, 1000), 144);
        // Measured: 1.7 ms fixed latency plus 2 us per face.
        for faces in [30, 62, 120, 240] {
            model.observe(faces, 1.7 + 0.002 * f64::from(faces));
        }
        let (a, b) = model.cost().unwrap();
        assert!((a - 1.7).abs() < 1e-6 && (b - 0.002).abs() < 1e-6);
        // A 4 ms burst pays (4 - 1.7) / 0.002 faces, capped at four times 240.
        assert_eq!(model.faces(4.0, 144, 100_000), 960);
        // A 0.75 ms steady target waits for three frames of credit, then one batch.
        let mut steady = CostModel {
            credit: 0.0,
            ..model
        };
        let batches: Vec<_> = (0..6).map(|_| steady.faces(0.75, 144, 100_000)).collect();
        assert_eq!(&batches[..2], &[0, 0]);
        assert!(batches[2] > 200);
        assert_eq!(steady.faces(0.75, 144, 0), 0);
        let total_ms: f64 = batches
            .iter()
            .filter(|&&n| n > 0)
            .map(|&n| a + b * f64::from(n))
            .sum();
        assert!(total_ms <= 6.0 * 0.75 + 1e-9);
    }

    #[test]
    fn resolve_module_validates_and_matches_the_forward_scale() {
        raa_shader::Composer::validate(&spherical_resolve_module()).expect("resolve WGSL");
        assert!(raa_shader::SPHERICAL_GI_WGSL.contains(&format!(
            "RAA_SPHERICAL_RESOLVE_SCALE: u32 = {RESOLVE_SCALE}u;"
        )));
    }

    #[test]
    fn capture_clock_never_starves_when_the_fixed_cost_exceeds_the_credit_cap() {
        // A slow first sample (6 ms for one batch of 96 faces) against a 0.75 ms target.
        let mut model = CostModel::default();
        model.observe(96, 6.2);
        model.observe(48, 6.0);
        let (a, _) = model.cost().expect("cost");
        assert!(a > 4.0 * 0.75);
        let issued: u32 = (0..32).map(|_| model.faces(0.75, 96, 7000)).sum();
        assert!(issued >= MIN_CLOCK_FACES, "issued {issued}");
        // A latency-bound fixed cost just above the cap with a tiny per-face cost: the
        // ceiling lands exactly on a minimal batch, where rounding must not starve it.
        for (a, b) in [(3.5, 0.002), (3.0001, 1e-5), (6.0, 0.1)] {
            let mut model = CostModel::default();
            for n in [24u32, 144, 400] {
                model.observe(n, a + b * f64::from(n));
            }
            let issued: u32 = (0..64).map(|_| model.faces(0.75, 144, 7000)).sum();
            assert!(issued >= MIN_CLOCK_FACES, "a {a} b {b}: issued {issued}");
        }
    }

    #[test]
    fn sun_and_sky_intensity_or_tint_changes_keep_the_light_revision() {
        let mut sh = [[0.0f32; 4]; 9];
        sh[0] = [0.4, 0.5, 0.7, 0.0];
        sh[1] = [0.1, 0.05, -0.02, 0.0];
        let view = [0.3, -0.9, 0.2, 3.0, 2.8, 2.5, 0.1, 0.1, 0.12, 40.0];
        let mut sky = SkyShape::default();
        let base = light_revision(&mut sky, Some(&sh), view, RevisionKey::default());
        // Sun intensity and tint, ambient, and a uniformly scaled (brighter) sky.
        let mut brighter = view;
        brighter[3..6].copy_from_slice(&[9.0, 7.0, 5.0]);
        brighter[6..9].copy_from_slice(&[0.3, 0.2, 0.1]);
        let scaled = sh.map(|c| c.map(|v| 2.5 * v));
        assert_eq!(
            light_revision(&mut sky, Some(&scaled), brighter, RevisionKey::default()),
            base
        );
        // A sky shape drift below the tolerance keeps it too.
        let mut drift = sh;
        drift[1][0] = 0.105;
        assert_eq!(
            light_revision(&mut sky, Some(&drift), view, RevisionKey::default()),
            base
        );
        // Shape changes: sun direction, sky shape, sky dark, environment gone.
        let mut turned = view;
        turned[0] = 0.31;
        assert_ne!(
            light_revision(&mut sky, Some(&sh), turned, RevisionKey::default()),
            base
        );
        let mut reshaped = sh;
        reshaped[1][0] = 0.2;
        let reshaped_revision =
            light_revision(&mut sky, Some(&reshaped), view, RevisionKey::default());
        assert_ne!(reshaped_revision, base);
        assert_eq!(
            light_revision(&mut sky, Some(&reshaped), brighter, RevisionKey::default()),
            reshaped_revision
        );
        assert_ne!(
            light_revision(&mut sky, Some(&[[0.0; 4]; 9]), view, RevisionKey::default()),
            reshaped_revision
        );
        let ambient = light_revision(&mut sky, None, view, RevisionKey::default());
        assert_eq!(
            light_revision(&mut sky, None, brighter, RevisionKey::default()),
            ambient
        );
    }

    #[test]
    fn slot_rows_match_the_shared_lookup_layout() {
        assert_eq!(SLOT_ROWS, 167);
        assert_eq!(LOOKUP_HEADER, 70);
        let module = spherical_module();
        assert!(module.contains(&format!("SPHERICAL_SUN_SCALE_ROW: u32 = {SUN_SCALE_ROW}u;")));
        assert!(module.contains(&format!("SPHERICAL_SKY_SCALE_ROW: u32 = {SKY_SCALE_ROW}u;")));
        assert!(raa_shader::SPHERICAL_GI_WGSL.contains("RAA_SPHERICAL_TABLE: u32 = 70u;"));
        assert!(raa_shader::SPHERICAL_GI_WGSL.contains("RAA_SPHERICAL_SUN_SCALE_ROW: u32 = 68u;"));
        assert!(raa_shader::SPHERICAL_GI_WGSL.contains("RAA_SPHERICAL_SKY_SCALE_ROW: u32 = 69u;"));
        assert!(raa_shader::SPHERICAL_GI_WGSL.contains("RAA_SPHERICAL_SLOT_ROWS: u32 = 167u;"));
        assert!(raa_shader::SPHERICAL_GI_WGSL.contains("RAA_SPHERICAL_LIGHT_ROW: u32 = 166u;"));
        assert!(module.contains("SPHERICAL_INFO_ROW: u32 = 165u;"));
        assert!(module.contains("SPHERICAL_LIGHT_ROW: u32 = 166u;"));
    }

    #[test]
    fn spherical_compute_module_validates() {
        raa_shader::Composer::validate(&super::spherical_module())
            .expect("spherical capture and project WGSL");
    }
}
