//! Dynamic diffuse global illumination (Majercik et al. 2019, with the probe relocation and
//! classification of Majercik et al. 2021) on the shared world-space mesh BVH.
//!
//! A camera-centred, world-aligned probe grid (kept inside the scene bounds) is addressed
//! toroidally, so a scroll recycles only the planes that leave the volume. Each frame a scheduled
//! subset of probes (see `schedule`: visible first, under a GPU-time budget with a burst while
//! visible probes are unconverged) traces rotated spherical-Fibonacci rays, shades their hits
//! (sun, local lights, emission and last frame's volume), relocates and classifies itself and
//! blends octahedral irradiance (8x8) and distance moments (16x16). A screen resolve then
//! interpolates the volume once per prepass pixel. Forward and VG shading read the same storage
//! buffer through the spherical GI binding while `RAA_DDGI_GI_FEATURE` is set.
//!
//! Lighting changes (see `mesh_ddgi.wgsl`): a probe stores its irradiance as linear groups (unit
//! sun, sky SH transfer + residual, other lights), composed with the frame's sun colour, sky SH
//! and sky scale. So a change of sun colour or intensity, of the sky (colour, intensity, the sky
//! that follows the sun) costs one `relight` pass over the probes and no ray; a change of sun
//! DIRECTION invalidates the sun group only (shadows moved), frustum first; an edit of a local
//! light invalidates the other group of the probes whose light mask holds that light (plus the
//! probes its new bound reaches). `DdgiInvalidation` selects that tracking, the naive
//! "everything on any change" or none (periodic refresh only) for comparison.

mod extras;
mod lights;
mod schedule;

use std::hash::{Hash, Hasher};

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc::{Receiver, TryRecvError, sync_channel};

use raa_core::{GpuContext, RaaError};
use raa_pipeline::{FeaturePass, LightingResources, Pipeline, PrepareCtx, RecordCtx};
use wgpu::{Buffer, BufferUsages as Usage};

use self::extras::{Extras, TABLE_ROWS_PER_SLOT};
use self::lights::{LightMask, MaskReadback};
pub use self::schedule::{
    DdgiView, PERIOD_BUCKETS as DDGI_PERIOD_BUCKETS, PERIOD_CLASSES as DDGI_PERIOD_CLASSES,
    Visibility as DdgiVisibility,
};
use self::schedule::{GROUP_ALL, GROUP_OTHER, GROUP_SUN, Schedule};
use super::MeshTraceHandle;
use super::pass::{buffer, buffer_entry, image_entry};

pub const DDGI_GI_NAME: &str = "gi.ddgi";
/// Frame feature bit read by `opaque_lighting.wgsl` (raa-shader `RAA_DDGI_GI_FEATURE`).
pub const DDGI_FRAME_FEATURE: u32 = raa_shader::RAA_DDGI_GI_FEATURE;
const ORDER: u32 = 57;
const MAGIC: u32 = 0x4947_4444;
/// Rows 0-15 are written by the CPU; rows 16-24 hold the sky SH written by `ddgi_sky_project`.
const HEADER_ROWS: u64 = 32;
const CPU_HEADER_ROWS: u64 = 16;
const SKY_ROW: u64 = 16;
/// Record, offset and the 256-bit light mask.
const META_ROWS: u64 = 4;
const IRRADIANCE_ROWS: u64 = 50;
const DEPTH_ROWS: u64 = 81;
/// 64 irradiance texels x 5 rows (36 f16: unit sun, sky residual, other, 9 rgb transfer).
const GROUP_ROWS: u64 = 320;
/// Ray scratch of an update: 4 rows per ray and job (the bounced sky transfer of the hit).
const SCRATCH_ROWS: u64 = 262_144;
const SCRATCH_ROWS_PER_RAY: u64 = 4;
/// Frames the sky projection and the relight run after a change (the environment's own filter
/// lands a frame or more after its source changed).
const RELIGHT_FRAMES: u32 = 4;
/// Sun rotation (degrees) that invalidates the sun group, and the age it leaves: the larger the
/// step, the younger (the faster the blend).
const SUN_STEP_DEGREES: f32 = 0.5;
/// Relative change of the sky SH that refreshes the sky residual (with the sun group).
const SKY_STEP: f32 = 0.05;
const SKY_AGE: u8 = 12;
/// Age left to an invalidated group: its next updates blend as a running mean from there.
const RELIGHT_AGE: u8 = 2;
/// Age left to the other group of a probe whose light mask grew after a light edit.
const LATE_AGE: u8 = 8;
/// Frames after a light edit during which a grown light mask invalidates its probe.
const LATE_FRAMES: u32 = 240;
/// Half angle of the cone around the sun left out of the sky (its disc is the sun light).
const SUN_CONE_DEGREES: f32 = 2.5;
const MAX_PROBES: u32 = 32_768;
const MAX_AXIS: u32 = 128;
const PARAMS_BYTES: u64 = 144;
/// Most surface probes (the pool after the cascades).
const MAX_SURFACE_PROBES: u32 = 16_384;
const OPTIONS_BYTES: u64 = 64;
const FLAG_RELOCATE: u32 = 1;
const FLAG_CLASSIFY: u32 = 2;
const FLAG_BOUNCE: u32 = 4;
const FLAG_SKY_TRANSFER: u32 = 8;
/// Header flags (row 4.w) read by the lookup: screen AO on the DDGI term, legacy far fade.
const FLAG_SCREEN_AO: u32 = 1 << 16;
const FLAG_FAR_AMBIENT: u32 = 1 << 17;

/// What a lighting change makes the probes trace again.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum DdgiInvalidation {
    /// Nothing: only the periodic refresh (the behaviour before lighting groups).
    None = 0,
    /// Every group of every probe on any change (a full recapture).
    All = 1,
    /// The sun group on a sun rotation, the other group of the probes a changed light reaches.
    #[default]
    Tracked = 2,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum DdgiDebug {
    #[default]
    Off = 0,
    /// The interpolated irradiance / pi.
    Irradiance = 1,
    /// Red where no usable probe covers the surface, green where it is fully covered.
    Coverage = 2,
    /// Green where forward used the screen resolve, red where it fell back to the full lookup.
    Resolve = 3,
    /// Weight per cascade (red: cascade 0, green: coarser), blue where inactive probes filled a
    /// hole; black where no probe was usable.
    Probes = 4,
    /// Red where a surface probe would be requested, green the share the surface probes take,
    /// blue the quality of the cascades' estimate.
    Quality = 5,
}

/// What a surface gets where the finer cascades do not cover it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum DdgiFar {
    /// Beyond the last cascade the lookup clamps to that cascade's border probes, and the
    /// estimate of whatever usable probes cover a point is renormalized: forward never falls back
    /// to the flat ambient term while any probe is usable.
    #[default]
    Clamp = 0,
    /// The behaviour before cascades: confidence fades over the outer half cell of the last
    /// cascade and with the probe coverage, the rest is the flat ambient term (comparison only).
    Ambient = 1,
}

/// How a probe update blends into the stored irradiance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum DdgiBlend {
    /// Fixed weight per update (`hysteresis`): the lag in time grows with the refresh period of
    /// the probe (the volume before time-based blending).
    #[default]
    Update = 0,
    /// Weight from the probe's actual time since its previous update and one time constant
    /// (`blendSlowMs`): every probe lags by the same time whatever its refresh period.
    Time = 1,
    /// Time-based, between two time constants: the slow one while an update agrees with the
    /// texel's own update-to-update noise, toward the fast one as the luma change exceeds
    /// `changeSigma` times that noise.
    Adaptive = 2,
}

/// How a probe that sits in or against geometry moves inside its cell.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum DdgiRelocation {
    /// Out through the nearest back face; away from a front face closer than the minimum
    /// distance by the missing distance (the rule this volume started with).
    #[default]
    Nearest = 0,
    /// The RTXGI SDK rule: out through the nearest back face; from a too-close front face
    /// toward the FARTHEST front face (at most one unit) unless that leads back to the closest
    /// one; back toward the grid position when nothing is close.
    Rtxgi = 1,
}

/// Most cascades (cascade k has spacing `spacing * 2^k`).
pub const DDGI_MAX_CASCADES: u32 = 3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DdgiSettings {
    pub enabled: bool,
    pub counts: [u32; 3],
    /// Probe spacing in metres.
    pub spacing: f32,
    /// Nested volumes, 1 to 3: cascade 0 is `counts` at `spacing`; cascade k >= 1 is
    /// `cascade_counts` at `spacing * 2^k`, each camera-centred, scrolled and kept inside the
    /// scene bounds independently. A surface takes the finest cascade that covers it, blended
    /// into the next one over a band of 1.5 cells where a cascade ends inside the scene.
    pub cascades: u32,
    pub cascade_counts: [u32; 3],
    pub far: DdgiFar,
    /// Multiply the DDGI irradiance by the screen-space AO (GTAO) as the ambient term it
    /// replaces is, not by the material AO alone: probes 1.5 m apart carry no small-scale
    /// occlusion.
    pub screen_ao: bool,
    pub rays_per_probe: u32,
    pub hysteresis: f32,
    pub blend: DdgiBlend,
    /// Time constants of the time-based blends (ms): fast (adaptive, on a detected change) and
    /// slow (time blend; adaptive while the probe is stationary).
    pub blend_fast_ms: f32,
    pub blend_slow_ms: f32,
    /// Adaptive blend: a luma change counts as real past this many standard deviations of the
    /// texel's update-to-update change.
    pub change_sigma: f32,
    /// Fraction of each probe's rays drawn from its previous radiance (per octahedral cell); the
    /// rest uniform, every ray weighted by its mixture density (0 = uniform Fibonacci rays).
    pub guide: f32,
    /// Many lights: a probe's rays evaluate only the lights of its stored light mask (the lights
    /// noticeable at its hits since the mask's last reset), every light on its first 8 updates
    /// after a reset and on one update in 8.
    pub light_cull: bool,
    /// Most probes updated per frame outside a burst, as a fraction of the volume.
    pub update_fraction: f32,
    /// GPU time of the probe update per frame (ms), converted with the measured cost per probe.
    pub budget_ms: f32,
    /// GPU time per frame while a visible probe is invalid or still converging (ms).
    pub burst_ms: f32,
    pub intensity: f32,
    /// Surface biases as fractions of the probe spacing.
    pub normal_bias: f32,
    pub view_bias: f32,
    pub relocation: bool,
    pub relocation_rule: DdgiRelocation,
    /// Largest relocation offset per axis, as a fraction of the probe spacing (0.5 = the whole
    /// cell of the probe).
    pub relocation_limit: f32,
    pub classification: bool,
    /// Surface probes: size of the pool of extra probes placed in front of badly served
    /// surfaces (0 = off), see `extras`.
    pub surface_probes: u32,
    /// Lattice pitch of the surface probes (m): at most one per cell.
    pub surface_spacing: f32,
    /// Distance from the surface at which a surface probe is requested (m), shortened to half
    /// the free distance along the normal.
    pub surface_offset: f32,
    /// A ray hit is tested for a surface probe when the lookup quality there (trilinear mass of
    /// the probes in front of the surface, visible by their distance moments and within reach)
    /// is below this; the request is made when rays then find no probe of its cage in front of
    /// the surface with a free segment to it.
    pub surface_need: f32,
    /// Distance (m) beyond which a cascade probe no longer counts as serving a surface in the
    /// quality above; 0 = any distance.
    pub surface_reach: f32,
    /// Surface probes are placed within this distance of the camera (m); when the pool is full
    /// the nearest cells win.
    pub surface_range: f32,
    /// Shade ray hits with the previous volume (infinite bounces over time).
    pub bounce: bool,
    /// Sky light as an SH transfer composed with the frame's sky (off: every escaped ray's sky
    /// radiance is stored as captured and only a recapture follows a sky change).
    pub sky_transfer: bool,
    pub invalidation: DdgiInvalidation,
    /// Smallest unshadowed contribution (linear luma) that enters a probe's light mask.
    pub light_threshold: f32,
    pub debug: DdgiDebug,
}

impl Default for DdgiSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            // 36 x 18 x 24 m: an Intel Sponza sized interior with little scrolling.
            counts: [24, 12, 16],
            spacing: 1.5,
            // 48 x 24 x 36 m at 3 m: Intel Sponza's whole bounds (36 x 20 x 24 m) whatever the
            // camera, for 1/3 of cascade 0's probes.
            cascades: 2,
            cascade_counts: [16, 8, 12],
            far: DdgiFar::Clamp,
            screen_ao: true,
            rays_per_probe: 64,
            hysteresis: 0.97,
            blend: DdgiBlend::Update,
            blend_fast_ms: 150.0,
            blend_slow_ms: 2000.0,
            change_sigma: 3.0,
            guide: 0.0,
            light_cull: false,
            update_fraction: 0.25,
            budget_ms: 0.6,
            burst_ms: 2.0,
            intensity: 1.0,
            normal_bias: 0.1,
            view_bias: 0.3,
            relocation: true,
            relocation_rule: DdgiRelocation::Nearest,
            relocation_limit: 0.45,
            classification: true,
            surface_probes: 0,
            surface_spacing: 0.75,
            surface_offset: 0.5,
            surface_need: 0.5,
            surface_reach: 0.0,
            surface_range: 40.0,
            bounce: true,
            sky_transfer: true,
            invalidation: DdgiInvalidation::Tracked,
            light_threshold: 1e-3,
            debug: DdgiDebug::Off,
        }
    }
}

impl DdgiSettings {
    /// Probes of every cascade.
    pub fn probe_count(&self) -> u32 {
        self.cascade_layouts().iter().map(|c| c.probes).sum()
    }

    /// The cascades in slot order.
    pub(crate) fn cascade_layouts(&self) -> Vec<CascadeLayout> {
        let mut offset = 0;
        (0..self.cascades.clamp(1, DDGI_MAX_CASCADES))
            .map(|k| {
                let counts = if k == 0 {
                    self.counts
                } else {
                    self.cascade_counts
                };
                let probes = counts.iter().product::<u32>();
                let layout = CascadeLayout {
                    counts,
                    spacing: self.spacing * (1u32 << k) as f32,
                    offset,
                    probes,
                };
                offset += probes;
                layout
            })
            .collect()
    }

    pub fn validate(self) -> Result<Self, RaaError> {
        let counts_ok = self.counts.iter().all(|&n| (2..=MAX_AXIS).contains(&n))
            && self
                .cascade_counts
                .iter()
                .all(|&n| (2..=MAX_AXIS).contains(&n))
            && (1..=DDGI_MAX_CASCADES).contains(&self.cascades)
            && self.probe_count() + self.surface_probes <= MAX_PROBES
            && self.surface_probes <= MAX_SURFACE_PROBES
            && (0.1..=0.5).contains(&self.relocation_limit)
            && (0.1..=10.0).contains(&self.surface_spacing)
            && (0.02..=10.0).contains(&self.surface_offset)
            && (0.0..=1.0).contains(&self.surface_need)
            && (0.0..=100.0).contains(&self.surface_reach)
            && (1.0..=1000.0).contains(&self.surface_range);
        let ok = counts_ok
            && self.spacing.is_finite()
            && (0.05..=100.0).contains(&self.spacing)
            && (8..=256).contains(&self.rays_per_probe)
            && (1.0..=60_000.0).contains(&self.blend_fast_ms)
            && (1.0..=60_000.0).contains(&self.blend_slow_ms)
            && (0.5..=20.0).contains(&self.change_sigma)
            && (0.0..=0.95).contains(&self.guide)
            && (0.0..=0.995).contains(&self.hysteresis)
            && self.update_fraction > 0.0
            && self.update_fraction <= 1.0
            && (0.05..=20.0).contains(&self.budget_ms)
            && (0.05..=20.0).contains(&self.burst_ms)
            && (0.0..=16.0).contains(&self.intensity)
            && (0.0..=1.0).contains(&self.normal_bias)
            && (0.0..=1.0).contains(&self.view_bias)
            && (0.0..=1.0).contains(&self.light_threshold);
        if !ok {
            return Err(RaaError::FrameGraph {
                code: "gi.ddgi.invalid_settings",
                message: format!(
                    "expected 1 to {DDGI_MAX_CASCADES} cascades, probe counts in [2, {MAX_AXIS}] with at most {MAX_PROBES} probes in all, spacing in [0.05, 100] m, \
                     8-256 rays, blend time constants in [1, 60000] ms, change sigma in [0.5, 20], guide in [0, 0.95], hysteresis in [0, 0.995], update fraction in (0, 1], budgets in [0.05, 20] ms, intensity in [0, 16] and biases in [0, 1]"
                ),
            });
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DdgiStats {
    pub probes: u32,
    pub scheduled: u32,
    /// Whether this frame ran the burst budget (a visible probe invalid or young).
    pub burst: bool,
    /// Visible probes after this frame's updates: total, valid, converged.
    pub visibility: DdgiVisibility,
    /// Updates a probe needs to count as converged.
    pub converge_age: u32,
    /// Measured GPU cost of one probe update (ms), the budget's conversion factor.
    pub ms_per_probe: f64,
    /// Last measured GPU time of the update dispatch (ms), when timestamps are available.
    pub update_ms: Option<f64>,
    /// Steady state: frames between update batches (1 while bursting).
    pub interval: u32,
    /// Whether the screen resolve ran (it does unless its rows exceed the storage binding limit;
    /// forward then gets no DDGI).
    pub screen_resolve: bool,
    pub update_frames: u64,
    pub recycled: u64,
    pub base: [i32; 3],
    /// Grid origin of each cascade (unused entries zero) and the cascade count.
    pub cascade_bases: [[i32; 3]; 3],
    pub cascades: u32,
    pub gpu_bytes: u64,
    /// World bounds of the traced geometry the volume is kept inside.
    pub scene_bounds: Option<([f32; 3], [f32; 3])>,
    /// Group refreshes traced since the volume started: geometry (sky transfer), sun, other.
    pub refreshed: [u64; 3],
    /// Probe groups invalidated by lighting changes since the volume started (same order).
    pub invalidated: [u64; 3],
    /// Frames that ran the relight pass (no ray) since the volume started.
    pub relights: u64,
    /// Local-light edits seen, and the probes the last one invalidated.
    pub light_edits: u64,
    pub last_light_probes: u32,
    /// Whether the probes' light masks were read back at least once.
    pub light_masks: bool,
    /// Surface probes: pool size, assigned, and totals since the volume started (assignments,
    /// evictions, valid requests read back).
    pub surface_pool: u32,
    pub surface_live: u32,
    pub surface_assigned: u64,
    pub surface_evicted: u64,
    pub surface_requests: u64,
    /// Frames between two updates of the same probe, per class and power-of-two bucket.
    pub periods: [[u64; DDGI_PERIOD_BUCKETS]; DDGI_PERIOD_CLASSES],
    /// Rows of the probe buffer (16 bytes each): probe records, screen resolve, surface table.
    pub rows: [u32; 3],
}

pub type DdgiHandle = Rc<RefCell<DdgiState>>;

#[derive(Default)]
pub struct DdgiState {
    pub settings: DdgiSettings,
    view: Option<(usize, DdgiView)>,
    pass_enabled: bool,
    revision: u64,
    stats: Option<DdgiStats>,
    buffer: Option<Buffer>,
}

impl DdgiState {
    pub fn configure(&mut self, settings: DdgiSettings) -> Result<(), RaaError> {
        let settings = settings.validate()?;
        if settings.cascade_layouts() != self.settings.cascade_layouts()
            || settings.surface_probes != self.settings.surface_probes
            || settings.surface_spacing != self.settings.surface_spacing
        {
            self.revision = self.revision.wrapping_add(1);
        }
        self.settings = settings;
        Ok(())
    }

    pub fn active(&self) -> bool {
        self.pass_enabled && self.settings.enabled
    }

    pub fn stats(&self) -> Option<DdgiStats> {
        self.stats
    }

    /// The probe buffer (diagnostic readbacks).
    pub fn probe_buffer(&self) -> Option<Buffer> {
        self.buffer.clone()
    }

    /// The bridge supplies the effective camera of each view before the frame.
    pub fn set_view(&mut self, index: usize, view: DdgiView) {
        self.view = Some((index, view));
    }
}

pub struct DdgiFeature;

impl DdgiFeature {
    pub fn install(pipeline: &mut Pipeline, mesh: MeshTraceHandle) -> Result<DdgiHandle, RaaError> {
        let state = DdgiHandle::default();
        pipeline.add(Box::new(DdgiPass {
            state: state.clone(),
            mesh,
            gpu: None,
            schedule: Schedule::default(),
            extras: None,
            staged: None,
            published: false,
            rng: 0x9e37_79b9_7f4a_7c15,
            timer: None,
            ms_per_probe: INITIAL_MS_PER_PROBE,
            update_ms: None,
            batch_ms: 0.0,
            idle_frames: 0,
            lighting: LightingTrack::default(),
            clock: 0.0,
            geometry_revision: None,
        }))?;
        pipeline.set_enabled(DDGI_GI_NAME, false)?;
        Ok(state)
    }
}

/// The grid origin (in probe units) of the volume around `eye`: camera-centred, then kept inside
/// the scene bounds (padded by one probe) so no plane is spent outside the geometry. An axis the
/// volume spans entirely is centred on the scene and never scrolls.
fn volume_base(
    eye: [f32; 3],
    counts: [u32; 3],
    spacing: f32,
    bounds: Option<([f32; 3], [f32; 3])>,
) -> [i32; 3] {
    std::array::from_fn(|i| {
        let n = counts[i] as i32;
        let centred = (eye[i] / spacing).floor() as i32 - n / 2;
        let Some((min, max)) = bounds.filter(|(min, max)| min[i] <= max[i]) else {
            return centred;
        };
        let lo = (min[i] / spacing).floor() as i32 - 1;
        let hi = (max[i] / spacing).ceil() as i32 + 1;
        if hi - lo + 1 <= n {
            lo - (n - (hi - lo + 1)) / 2
        } else {
            centred.clamp(lo, hi - n + 1)
        }
    })
}

/// One cascade of the volume: its grid, spacing and slot range in the shared probe buffer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CascadeLayout {
    pub counts: [u32; 3],
    pub spacing: f32,
    /// First slot of the cascade.
    pub offset: u32,
    pub probes: u32,
}

/// Sides of a volume at `base` that reach the padded scene bounds (bit 2i: the low side of axis
/// i, bit 2i + 1: its high side): no geometry lies beyond, so the lookup neither fades nor blends
/// into the next cascade there. Without bounds every side is open.
fn closed_sides(
    base: [i32; 3],
    counts: [u32; 3],
    spacing: f32,
    bounds: Option<([f32; 3], [f32; 3])>,
) -> u32 {
    let mut mask = 0;
    for i in 0..3 {
        let Some((min, max)) = bounds.filter(|(min, max)| min[i] <= max[i]) else {
            continue;
        };
        let lo = (min[i] / spacing).floor() as i32 - 1;
        let hi = (max[i] / spacing).ceil() as i32 + 1;
        if base[i] <= lo {
            mask |= 1 << (2 * i);
        }
        if base[i] + counts[i] as i32 - 1 >= hi {
            mask |= 2 << (2 * i);
        }
    }
    mask
}

struct Staged {
    bounds: Option<([f32; 3], [f32; 3])>,
    jobs: Vec<u32>,
    /// Group refreshes of the jobs, invalidations of the frame, light edits.
    refreshed: [u32; 3],
    invalidated: [u32; 3],
    light_edits: u32,
    edit_probes: Option<u32>,
    relight: bool,
    read_masks: bool,
    recycled: u32,
    burst: bool,
    visibility: DdgiVisibility,
    converge_age: u32,
    interval: u32,
    /// Screen resolve size, when it runs this frame.
    screen: Option<(u32, u32)>,
    /// Probes per update workgroup.
    pack: u32,
}

/// Starting cost model before the first timestamp readback (RTX 3090, Sponza, 64 rays).
const INITIAL_MS_PER_PROBE: f64 = 0.0033;
/// Smallest update batch. Probe rays are latency bound (a few dependent BVH traversals per
/// thread): a dispatch costs about the same wall time until it fills the GPU, so the update runs
/// as GPU-filling batches and a steady-state budget below a batch's cost is met by updating only
/// every k-th frame. Per-probe costs are learned from batches of at least this size.
const MIN_BATCH: u32 = 256;
/// The screen resolve runs at 1/RESOLVE_SCALE of the frame on each axis.
const RESOLVE_SCALE: u32 = 2;

/// GPU timestamps around the update dispatch; one readback in flight at a time.
struct UpdateTimer {
    queries: wgpu::QuerySet,
    resolve: Buffer,
    readback: Buffer,
    ms_per_tick: f64,
    pending: Option<(u32, Receiver<Option<f64>>)>,
    recorded: Option<u32>,
}

impl UpdateTimer {
    fn new(gpu: &GpuContext) -> Option<Self> {
        if !gpu.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let device = gpu.device();
        let size = 2 * u64::from(wgpu::QUERY_SIZE);
        Some(Self {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("gi.ddgi.timer"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }),
            resolve: buffer(
                device,
                "gi.ddgi.timer",
                size,
                Usage::QUERY_RESOLVE | Usage::COPY_SRC,
            ),
            readback: buffer(
                device,
                "gi.ddgi.timer",
                size,
                Usage::MAP_READ | Usage::COPY_DST,
            ),
            ms_per_tick: f64::from(gpu.queue().get_timestamp_period()) / 1e6,
            pending: None,
            recorded: None,
        })
    }

    /// A finished readback: (probes, ms).
    fn poll(&mut self) -> Option<(u32, f64)> {
        let (jobs, receiver) = self.pending.as_ref()?;
        match receiver.try_recv() {
            Ok(ms) => {
                let jobs = *jobs;
                self.pending = None;
                ms.map(|ms| (jobs, ms))
            }
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                self.pending = None;
                None
            }
        }
    }

    fn writes(&self) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.pending
            .is_none()
            .then_some(wgpu::ComputePassTimestampWrites {
                query_set: &self.queries,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            })
    }

    fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder, jobs: u32) {
        let bytes = 2 * u64::from(wgpu::QUERY_SIZE);
        encoder.resolve_query_set(&self.queries, 0..2, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, bytes);
        let (send, receive) = sync_channel(1);
        let readback = self.readback.clone();
        let ms_per_tick = self.ms_per_tick;
        encoder.map_buffer_on_submit(
            &self.readback,
            wgpu::MapMode::Read,
            0..bytes,
            move |result| {
                let ms = result.ok().and_then(|()| {
                    let ms = readback
                        .slice(0..bytes)
                        .get_mapped_range()
                        .ok()
                        .and_then(|data| {
                            let begin = u64::from_le_bytes(data[..8].try_into().ok()?);
                            let end = u64::from_le_bytes(data[8..16].try_into().ok()?);
                            end.checked_sub(begin)
                                .filter(|_| {
                                    begin != 0 && ms_per_tick.is_finite() && ms_per_tick > 0.0
                                })
                                .map(|ticks| ticks as f64 * ms_per_tick)
                        });
                    readback.unmap();
                    ms
                });
                let _ = send.send(ms);
            },
        );
        self.pending = Some((jobs, receive));
        self.recorded = Some(jobs);
    }
}

struct DdgiPass {
    state: DdgiHandle,
    mesh: MeshTraceHandle,
    gpu: Option<DdgiGpu>,
    schedule: Schedule,
    extras: Option<Extras>,
    staged: Option<Staged>,
    published: bool,
    rng: u64,
    timer: Option<UpdateTimer>,
    ms_per_probe: f64,
    update_ms: Option<f64>,
    /// Smoothed GPU time of one update batch (ms), 0 before the first readback.
    batch_ms: f64,
    idle_frames: u32,
    lighting: LightingTrack,
    /// Seconds of rendered time (frame deltas, each capped at 50 ms so a stall or capture pause
    /// does not count as time the probes had to converge).
    clock: f64,
    /// Revision of the traced geometry the probes were last invalidated against.
    geometry_revision: Option<u64>,
}

/// What the lighting looked like when the groups were last invalidated or relit.
#[derive(Default)]
struct LightingTrack {
    /// Hash of everything the composed tiles depend on besides the groups.
    relight: Option<u64>,
    relight_frames: u32,
    /// Sun direction and sky SH at the last invalidation of the sun group.
    sun: Option<[f32; 3]>,
    sky: Option<[[f32; 4]; 9]>,
    /// The packed light records of the previous frame.
    lights: Vec<u8>,
    masks: Option<Vec<LightMask>>,
    readback: Option<MaskReadback>,
    frames_since_edit: u32,
    frame: u32,
}

fn angle_degrees(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dot: f32 = (0..3).map(|i| a[i] * b[i]).sum();
    let (la, lb) = (
        a.iter().map(|x| x * x).sum::<f32>().sqrt(),
        b.iter().map(|x| x * x).sum::<f32>().sqrt(),
    );
    if la <= 0.0 || lb <= 0.0 {
        return 0.0;
    }
    (dot / (la * lb)).clamp(-1.0, 1.0).acos().to_degrees()
}

/// Age left to the sun group after the sun turned by `degrees`.
fn sun_age(degrees: f32) -> u8 {
    (6.0 / degrees.max(1e-3))
        .round()
        .clamp(f32::from(RELIGHT_AGE), 12.0) as u8
}

/// Change of the SHAPE of a sky SH: the largest coefficient difference after each sky is divided
/// by its band-0 magnitude, so a sky that only gets brighter or darker (the physical sky follows
/// the sun intensity) is no change. A sky that turns black or lights up is a full change.
fn sky_change(a: &[[f32; 4]; 9], b: &[[f32; 4]; 9]) -> f32 {
    let norm = |sh: &[[f32; 4]; 9]| sh[0][..3].iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let (na, nb) = (norm(a), norm(b));
    if na <= 1e-7 || nb <= 1e-7 {
        return if na <= 1e-7 && nb <= 1e-7 { 0.0 } else { 1.0 };
    }
    let mut worst = 0.0f32;
    for k in 0..9 {
        for c in 0..3 {
            worst = worst.max((a[k][c] / na - b[k][c] / nb).abs());
        }
    }
    worst
}

impl DdgiPass {
    /// Geometry edits: the probes within two cells of a changed bound restart (every group and
    /// the depth moments, their classification and relocation redone on that first update and
    /// their light mask rescanned); those within six cells keep their groups as a prior of
    /// `RELIGHT_AGE` updates (occlusion and bounces changed). Missed revisions (more than one
    /// since the last frame, only the last delta is kept) relight the whole volume. Returns the
    /// probes touched.
    fn invalidate_geometry(&mut self, dirty: &[([f32; 3], [f32; 3])], delta: bool) -> u32 {
        if !delta {
            return self.schedule.invalidate_all(GROUP_ALL, RELIGHT_AGE);
        }
        let near = |p: [f32; 3], margin: f32| {
            dirty
                .iter()
                .any(|(lo, hi)| (0..3).all(|i| p[i] >= lo[i] - margin && p[i] <= hi[i] + margin))
        };
        let mut restart = Vec::new();
        let mut relight = Vec::new();
        self.schedule.for_each_probe(|slot, p, spacing| {
            if near(p, 2.0 * spacing) {
                restart.push(slot);
            } else if near(p, 6.0 * spacing) {
                relight.push(slot);
            }
        });
        let mut n = 0;
        for slot in restart {
            self.schedule.restart_slot(slot);
            n += 1;
        }
        for slot in relight {
            n += u32::from(
                self.schedule
                    .invalidate_slot(slot, GROUP_ALL, RELIGHT_AGE, true),
            );
        }
        n
    }

    /// Applies the frame's lighting changes to the schedule; returns the invalidated probe groups
    /// (geometry, sun, other), the light edits and the probes the last edit invalidated.
    fn track_lighting(
        &mut self,
        settings: DdgiSettings,
        camera: &DdgiView,
        sky_diffuse: f32,
        lighting: &LightingResources,
    ) -> ([u32; 3], u32, Option<u32>) {
        let track = &mut self.lighting;
        track.frame = track.frame.wrapping_add(1);
        track.frames_since_edit = track.frames_since_edit.saturating_add(1);
        let sky = lighting.environment_diffuse_sh().copied();
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        camera.sun_direction.map(f32::to_bits).hash(&mut hash);
        camera.sun_radiance.map(f32::to_bits).hash(&mut hash);
        sky_diffuse.to_bits().hash(&mut hash);
        settings.sky_transfer.hash(&mut hash);
        if let Some(sky) = &sky {
            for coefficient in sky {
                coefficient.map(f32::to_bits).hash(&mut hash);
            }
        }
        let hash = hash.finish();
        let relit = track.relight.replace(hash) != Some(hash);
        if relit {
            track.relight_frames = RELIGHT_FRAMES;
        }

        let turned = track
            .sun
            .map_or(0.0, |sun| angle_degrees(sun, camera.sun_direction));
        let sun_event = turned >= SUN_STEP_DEGREES;
        if sun_event || track.sun.is_none() {
            track.sun = Some(camera.sun_direction);
        }
        let sky_event = match (&track.sky, &sky) {
            (Some(a), Some(b)) => sky_change(a, b) >= SKY_STEP,
            (None, None) => false,
            _ => true,
        };
        if sky_event || sun_event {
            track.sky = sky;
        }

        let now = lighting.local_lights_bytes();
        let first = track.lights.is_empty();
        let changes = if first {
            Vec::new()
        } else {
            lights::diff(&track.lights, now, settings.light_threshold)
        };
        if first || !changes.is_empty() {
            track.lights.clear();
            track.lights.extend_from_slice(now);
        }
        // A grown light mask shortly after an edit: the edited light reaches that probe too.
        let mut late = Vec::new();
        if let Some(masks) = track.readback.as_mut().and_then(MaskReadback::poll) {
            if let Some(old) = track.masks.as_ref().filter(|old| old.len() == masks.len())
                && track.frames_since_edit < LATE_FRAMES
            {
                late.extend((0..masks.len() as u32).filter(|&slot| {
                    (0..8).any(|w| masks[slot as usize][w] & !old[slot as usize][w] != 0)
                }));
            }
            track.masks = Some(masks);
        }

        let mut invalidated = [0u32; 3];
        let mut edit_probes = None;
        let edits = changes.len() as u32;
        match settings.invalidation {
            DdgiInvalidation::None => {}
            DdgiInvalidation::All => {
                if sun_event || sky_event || relit || edits > 0 {
                    let n = self.schedule.invalidate_all(GROUP_ALL, RELIGHT_AGE);
                    invalidated = [n; 3];
                    if edits > 0 {
                        edit_probes = Some(n);
                    }
                }
            }
            DdgiInvalidation::Tracked => {
                if sun_event {
                    invalidated[1] = self.schedule.invalidate_all(GROUP_SUN, sun_age(turned));
                } else if sky_event {
                    invalidated[1] = self.schedule.invalidate_all(GROUP_SUN, SKY_AGE);
                }
                if edits > 0 {
                    // A probe sees lit surfaces beyond the light's own bound: two cells of margin,
                    // the rest comes from the masks (and from grown masks afterwards).
                    let masks = track.masks.as_ref();
                    let mut slots = Vec::new();
                    self.schedule.for_each_probe(|slot, p, spacing| {
                        let margin = 2.0 * spacing;
                        let mut moved = false;
                        let mut hit = false;
                        for change in &changes {
                            let known = masks.and_then(|masks| masks.get(slot as usize));
                            let by_mask =
                                known.is_some_and(|mask| lights::mask_has(mask, change.index));
                            let by_bound = (change.moved || change.brighter || known.is_none())
                                && change.new.reaches(p, margin)
                                || known.is_none() && change.old.reaches(p, margin);
                            if by_mask || by_bound {
                                hit = true;
                                moved |= change.moved;
                            }
                        }
                        if hit {
                            slots.push((slot, moved));
                        }
                    });
                    let mut n = 0;
                    for (slot, moved) in slots {
                        n += u32::from(self.schedule.invalidate_slot(
                            slot,
                            GROUP_OTHER,
                            RELIGHT_AGE,
                            moved,
                        ));
                    }
                    invalidated[2] += n;
                    edit_probes = Some(n);
                }
                for slot in late {
                    invalidated[2] += u32::from(self.schedule.invalidate_slot(
                        slot,
                        GROUP_OTHER,
                        LATE_AGE,
                        false,
                    ));
                }
            }
        }
        if edits > 0 {
            self.lighting.frames_since_edit = 0;
        }
        (invalidated, edits, edit_probes)
    }

    fn unpublish(&mut self, lighting: &mut LightingResources) {
        if std::mem::take(&mut self.published) {
            lighting.set_spherical_gi(None);
        }
    }

    fn next_random(&mut self) -> f32 {
        // xorshift64*
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let bits = self.rng.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40;
        bits as f32 / (1u64 << 24) as f32
    }

    /// A uniformly distributed rotation (Shoemake), as three columns.
    fn random_rotation(&mut self) -> [[f32; 3]; 3] {
        let (u1, u2, u3) = (self.next_random(), self.next_random(), self.next_random());
        let tau = std::f32::consts::TAU;
        let (a, b) = ((1.0 - u1).sqrt(), u1.sqrt());
        let [x, y, z, w] = [
            a * (tau * u2).sin(),
            a * (tau * u2).cos(),
            b * (tau * u3).sin(),
            b * (tau * u3).cos(),
        ];
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y + w * z),
                2.0 * (x * z - w * y),
            ],
            [
                2.0 * (x * y - w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z + w * x),
            ],
            [
                2.0 * (x * z + w * y),
                2.0 * (y * z - w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    }
}

impl FeaturePass for DdgiPass {
    fn name(&self) -> &str {
        DDGI_GI_NAME
    }

    fn order(&self) -> u32 {
        ORDER
    }

    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        self.mesh.borrow_mut().set_ddgi_active(enabled);
        self.state.borrow_mut().pass_enabled = enabled;
        if !enabled {
            self.unpublish(lighting);
            self.gpu = None;
            self.staged = None;
            self.schedule = Schedule::default();
            self.extras = None;
            self.timer = None;
            self.lighting = LightingTrack::default();
            let mut state = self.state.borrow_mut();
            state.stats = None;
            state.buffer = None;
        }
    }

    fn invalidate_history(&mut self) {
        // Probes live in world space; a camera cut keeps them.
    }

    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        self.staged = None;
        let (settings, view, revision) = {
            let state = self.state.borrow();
            if !state.active() {
                drop(state);
                self.unpublish(ctx.lighting);
                return Ok(());
            }
            (state.settings, state.view, state.revision)
        };
        let Some((_, camera)) = view.filter(|(index, _)| *index == 0) else {
            self.unpublish(ctx.lighting);
            return Ok(());
        };
        let grid = settings.probe_count();
        let extra_count = settings.surface_probes;
        let probes = grid + extra_count;
        let cascades = settings.cascade_layouts();
        // One row per resolve texel inside the forward's one storage binding.
        let resolve_size = (
            ctx.size.0.div_ceil(RESOLVE_SCALE),
            ctx.size.1.div_ceil(RESOLVE_SCALE),
        );
        let rows = layout(probes).rows
            + u64::from(resolve_size.0) * u64::from(resolve_size.1)
            + SCRATCH_ROWS
            + TABLE_ROWS_PER_SLOT * u64::from(extra_count);
        let fits = 16 * rows <= u64::from(ctx.gpu.limits().max_storage_buffer_binding_size);
        let screen = fits.then_some(resolve_size);
        if self.gpu.as_ref().is_none_or(|gpu| {
            gpu.cascades != cascades || gpu.revision != revision || gpu.screen != screen
        }) {
            // A resize restarts the volume: the screen rows share the probe buffer.
            self.gpu = Some(DdgiGpu::new(
                ctx.gpu,
                cascades.clone(),
                revision,
                screen,
                extra_count,
            ));
            self.schedule
                .reset(&cascades, extra_count, settings.surface_spacing);
            self.extras = (extra_count > 0)
                .then(|| Extras::new(ctx.gpu, extra_count, settings.surface_spacing));
            self.lighting = LightingTrack {
                readback: Some(MaskReadback::new(ctx.gpu, probes)),
                ..LightingTrack::default()
            };
            self.state.borrow_mut().stats = None;
        }
        if self.timer.is_none() {
            self.timer = UpdateTimer::new(ctx.gpu);
        }
        if let Some((jobs, ms)) = self.timer.as_mut().and_then(UpdateTimer::poll) {
            self.update_ms = Some(ms);
            self.batch_ms = if self.batch_ms > 0.0 {
                self.batch_ms + 0.25 * (ms - self.batch_ms)
            } else {
                ms
            };
            if jobs >= MIN_BATCH && ms > 0.0 {
                let sample = (ms / f64::from(jobs)).clamp(1e-5, 0.1);
                self.ms_per_probe += 0.25 * (sample - self.ms_per_probe);
            }
        }
        let mesh = self.mesh.borrow();
        let bounds = mesh.geometry.world_bounds().map(|b| (b.min, b.max));
        let geometry_revision = mesh.geometry.revision;
        let geometry_dirty: Vec<([f32; 3], [f32; 3])> = if self
            .geometry_revision
            .is_some_and(|old| old.wrapping_add(1) == geometry_revision)
        {
            mesh.geometry
                .dirty_bounds
                .iter()
                .map(|b| (b.min, b.max))
                .collect()
        } else {
            Vec::new()
        };
        let sky_diffuse = mesh.sky_diffuse;
        let triangles = mesh.geometry.count();
        let traceable = triangles > 0 && mesh.frame.is_some() && mesh.texture_atlas.is_some();
        drop(mesh);
        let bases: Vec<[i32; 3]> = cascades
            .iter()
            .map(|c| volume_base(camera.eye, c.counts, c.spacing, bounds))
            .collect();
        let closed: Vec<u32> = cascades
            .iter()
            .zip(&bases)
            .map(|(c, &base)| closed_sides(base, c.counts, c.spacing, bounds))
            .collect();
        let recycled = self.schedule.scroll(&bases, &closed);
        // New surface probes from the requests read back.
        if let (Some(extras), Some(gpu)) = (self.extras.as_mut(), self.gpu.as_ref()) {
            for change in extras.poll(camera.eye, settings.surface_range) {
                let bytes: Vec<u8> = change.row.iter().flat_map(|w| w.to_le_bytes()).collect();
                ctx.gpu.queue().write_buffer(
                    &gpu.probes,
                    16 * (gpu.table_row() + u64::from(change.index)),
                    &bytes,
                );
                self.schedule
                    .set_extra(grid + change.index, change.position);
            }
        }
        let (mut invalidated, light_edits, edit_probes) =
            self.track_lighting(settings, &camera, sky_diffuse, ctx.lighting);
        if let Some(old) = self.geometry_revision.replace(geometry_revision)
            && old != geometry_revision
        {
            invalidated[0] +=
                self.invalidate_geometry(&geometry_dirty, old.wrapping_add(1) == geometry_revision);
        }
        let floor = MIN_BATCH.min(probes);
        // The ray scratch bounds a dispatch.
        let job_cap =
            (SCRATCH_ROWS / (SCRATCH_ROWS_PER_RAY * u64::from(settings.rays_per_probe))) as u32;
        let probes_for = |ms: f32| {
            ((f64::from(ms) / self.ms_per_probe) as u32)
                .clamp(floor, probes)
                .min(job_cap)
        };
        let cap = ((probes as f32 * settings.update_fraction).ceil() as u32).clamp(1, probes);
        let burst = probes_for(settings.burst_ms);
        let batch = burst.min(cap.max(floor));
        // Steady state: one batch every `interval` frames keeps the average within budgetMs.
        let interval = if self.batch_ms > 0.0 {
            (self.batch_ms / f64::from(settings.budget_ms))
                .ceil()
                .clamp(1.0, 16.0) as u32
        } else {
            1
        };
        self.idle_frames += 1;
        let budget = if self.idle_frames >= interval {
            batch
        } else {
            0
        };
        let converge_age = converge_age_of(&settings);
        self.clock += if ctx.dt_s > 0.0 {
            f64::from(ctx.dt_s).min(0.05)
        } else {
            1.0 / 60.0
        };
        self.schedule.set_clock(self.clock);
        let mut jobs = Vec::with_capacity(burst as usize);
        let (visibility, bursting, refreshed) =
            self.schedule
                .plan(Some(&camera), converge_age, budget, burst, &mut jobs);
        if !jobs.is_empty() {
            self.idle_frames = 0;
        }

        let rotation = self.random_rotation();
        let gpu = self.gpu.as_mut().expect("DDGI pool");
        let mut flags = settings.debug as u32 * 256;
        if settings.screen_ao {
            flags |= FLAG_SCREEN_AO;
        }
        if settings.far == DdgiFar::Ambient {
            flags |= FLAG_FAR_AMBIENT;
        }
        if settings.classification {
            flags |= FLAG_CLASSIFY;
        }
        if settings.relocation {
            flags |= FLAG_RELOCATE;
        }
        gpu.write_header(ctx.gpu, &bases, &closed, flags, settings);
        let resolve = gpu.screen;
        ctx.lighting
            .set_spherical_gi(Some((gpu.probes.clone(), gpu.probes.clone())));
        self.published = true;
        self.state.borrow_mut().buffer = Some(gpu.probes.clone());
        if !traceable {
            return Ok(());
        }
        let extent = cascades
            .iter()
            .flat_map(|c| c.counts.map(|n| n as f32 * c.spacing))
            .fold(0.0, f32::max);
        let mut update_flags = flags & (FLAG_CLASSIFY | FLAG_RELOCATE);
        if settings.bounce {
            update_flags |= FLAG_BOUNCE;
        }
        if settings.sky_transfer {
            update_flags |= FLAG_SKY_TRANSFER;
        }
        let env = ctx.lighting.environment_diffuse_sh().is_some();
        let mut params = Vec::with_capacity(PARAMS_BYTES as usize);
        for column in rotation {
            push_f32(&mut params, [column[0], column[1], column[2], 0.0]);
        }
        push_u32(
            &mut params,
            [
                settings.rays_per_probe,
                (jobs.len() / 2) as u32,
                ctx.frame as u32,
                update_flags,
            ],
        );
        push_f32(
            &mut params,
            [
                settings.hysteresis,
                (4.0 * extent).max(100.0),
                0.25,
                0.1 * settings.spacing,
            ],
        );
        push_f32(
            &mut params,
            [
                sky_diffuse,
                0.25,
                SUN_CONE_DEGREES.to_radians().cos(),
                settings.light_threshold,
            ],
        );
        push_f32(
            &mut params,
            [
                settings.relocation_limit,
                settings.relocation_rule as u32 as f32,
                0.0,
                0.0,
            ],
        );
        push_f32(
            &mut params,
            [
                settings.blend_fast_ms / 1000.0,
                settings.blend_slow_ms / 1000.0,
                settings.change_sigma * settings.change_sigma,
                settings.blend as u32 as f32,
            ],
        );
        push_f32(
            &mut params,
            [
                probes_per_group(settings.rays_per_probe) as f32,
                settings.guide,
                f32::from(converge_age),
                f32::from(u8::from(settings.light_cull)),
            ],
        );
        let mut options = Vec::with_capacity(OPTIONS_BYTES as usize);
        push_u32(
            &mut options,
            [triangles, (jobs.len() / 2) as u32, u32::from(env), 0],
        );
        push_u32(&mut options, [0; 4]);
        push_u32(
            &mut options,
            [sky_diffuse.to_bits(), (4.0 * extent).to_bits(), 0, 0],
        );
        // The shared local-light integrator uses its one-tap area estimator for probe rays.
        push_u32(&mut options, [1, 0, 0, 0]);
        let queue = ctx.gpu.queue();
        gpu.ensure_jobs(ctx.gpu, jobs.len());
        let relight = self.lighting.relight_frames > 0;
        self.lighting.relight_frames = self.lighting.relight_frames.saturating_sub(1);
        let read_masks = self.lighting.frame % 4 == 0
            && self
                .lighting
                .readback
                .as_ref()
                .is_some_and(MaskReadback::idle);
        queue.write_buffer(&gpu.params, 0, &params);
        queue.write_buffer(&gpu.options, 0, &options);
        let job_bytes: Vec<u8> = jobs.iter().flat_map(|slot| slot.to_le_bytes()).collect();
        queue.write_buffer(&gpu.jobs, 0, &job_bytes);
        self.staged = Some(Staged {
            bounds,
            jobs,
            refreshed,
            invalidated,
            light_edits,
            edit_probes,
            relight,
            read_masks,
            recycled,
            burst: bursting,
            visibility,
            converge_age: u32::from(converge_age),
            interval: if bursting { 1 } else { interval },
            screen: resolve,
            pack: probes_per_group(settings.rays_per_probe),
        });
        Ok(())
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let Some(staged) = &self.staged else {
            return Ok(());
        };
        let gpu = self.gpu.as_mut().expect("prepared DDGI pool");
        let mut mesh = self.mesh.borrow_mut();
        let (Some(frame), Some(atlas)) = (mesh.frame.clone(), mesh.texture_atlas.clone()) else {
            return Ok(());
        };
        let geometry = mesh.geometry_buffers(ctx.gpu);
        drop(mesh);
        let device = ctx.gpu.device();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi.ddgi.update"),
            layout: &gpu.update.get_bind_group_layout(0),
            entries: &[
                buffer_entry(0, &frame),
                buffer_entry(1, &gpu.options),
                buffer_entry(2, &geometry[0]),
                buffer_entry(3, &geometry[1]),
                buffer_entry(4, &geometry[2]),
                buffer_entry(13, ctx.lighting.local_lights_buffer()),
                image_entry(14, ctx.lighting.env_specular_view()),
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: wgpu::BindingResource::Sampler(ctx.lighting.linear_clamp()),
                },
                image_entry(crate::wgsl::MESH_TRACE_ATLAS_BINDING, &atlas),
                buffer_entry(27, &gpu.probes),
                buffer_entry(28, &gpu.params),
                buffer_entry(29, &gpu.jobs),
            ],
        });
        if staged.relight {
            // The sky SH of this frame, then every composed tile from the stored groups.
            let sky = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gi.ddgi.sky"),
                layout: &gpu.sky.get_bind_group_layout(0),
                entries: &[
                    buffer_entry(0, &frame),
                    buffer_entry(1, &gpu.options),
                    image_entry(14, ctx.lighting.env_specular_view()),
                    wgpu::BindGroupEntry {
                        binding: 15,
                        resource: wgpu::BindingResource::Sampler(ctx.lighting.linear_clamp()),
                    },
                    buffer_entry(27, &gpu.probes),
                    buffer_entry(28, &gpu.params),
                ],
            });
            let relight = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gi.ddgi.relight"),
                layout: &gpu.relight.get_bind_group_layout(0),
                entries: &[
                    buffer_entry(0, &frame),
                    buffer_entry(27, &gpu.probes),
                    buffer_entry(28, &gpu.params),
                ],
            });
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gi.ddgi.relight"),
                    ..Default::default()
                });
            pass.set_pipeline(&gpu.sky);
            pass.set_bind_group(0, &sky, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&gpu.relight);
            pass.set_bind_group(0, &relight, &[]);
            pass.dispatch_workgroups(gpu.total(), 1, 1);
        }
        let jobs = (staged.jobs.len() / 2) as u32;
        if jobs > 0 {
            let timer = self.timer.as_mut();
            let writes = timer.as_ref().and_then(|timer| timer.writes());
            let timed = writes.is_some();
            {
                let mut pass = ctx
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("gi.ddgi.update"),
                        timestamp_writes: writes,
                    });
                pass.set_pipeline(&gpu.update);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(jobs.div_ceil(staged.pack), 1, 1);
            }
            if timed && let Some(timer) = timer {
                timer.resolve(ctx.encoder, jobs);
            }
        }
        if let Some((width, height)) = staged.screen {
            let depth = ctx.targets.depth_input(ctx.resources, DDGI_GI_NAME);
            let normal = ctx.targets.normal_input(ctx.resources, DDGI_GI_NAME);
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gi.ddgi.resolve"),
                layout: &gpu.resolve.get_bind_group_layout(0),
                entries: &[
                    buffer_entry(0, &frame),
                    image_entry(6, &depth),
                    image_entry(7, &normal),
                    buffer_entry(27, &gpu.probes),
                ],
            });
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gi.ddgi.resolve"),
                    ..Default::default()
                });
            pass.set_pipeline(&gpu.resolve);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        if staged.read_masks
            && let Some(readback) = self.lighting.readback.as_mut()
        {
            let rows = layout(gpu.total());
            readback.copy(ctx.encoder, &gpu.probes, 16 * rows.meta);
        }
        if self.lighting.frame % 4 == 2
            && let Some(extras) = self.extras.as_mut().filter(|e| e.idle())
        {
            let offset = 16 * (gpu.table_row() + u64::from(gpu.extras));
            extras.copy(ctx.encoder, &gpu.probes, offset);
        }
        Ok(())
    }

    fn submitted(&mut self) {
        let Some(staged) = self.staged.take() else {
            return;
        };
        let mut state = self.state.borrow_mut();
        let probes = state.settings.probe_count();
        let previous = state.stats.unwrap_or_default();
        let extras = self.extras.as_ref();
        let rows = self.gpu.as_ref().map_or([0; 3], |gpu| {
            let rows = layout(gpu.total());
            [rows.meta as u32, rows.rows as u32, gpu.table_row() as u32]
        });
        let gpu_bytes = self.gpu.as_ref().map_or(0, DdgiGpu::bytes);
        let add = |total: [u64; 3], frame: [u32; 3]| {
            std::array::from_fn(|i| total[i] + u64::from(frame[i]))
        };
        state.stats = Some(DdgiStats {
            refreshed: add(previous.refreshed, staged.refreshed),
            invalidated: add(previous.invalidated, staged.invalidated),
            relights: previous.relights + u64::from(staged.relight),
            light_edits: previous.light_edits + u64::from(staged.light_edits),
            last_light_probes: staged.edit_probes.unwrap_or(previous.last_light_probes),
            light_masks: self.lighting.masks.is_some(),
            probes,
            scheduled: (staged.jobs.len() / 2) as u32,
            burst: staged.burst,
            visibility: staged.visibility,
            converge_age: staged.converge_age,
            ms_per_probe: self.ms_per_probe,
            update_ms: self.update_ms,
            interval: staged.interval,
            screen_resolve: staged.screen.is_some(),
            scene_bounds: staged.bounds,
            update_frames: previous.update_frames + 1,
            recycled: previous.recycled + u64::from(staged.recycled),
            base: self
                .schedule
                .bases
                .as_ref()
                .and_then(|b| b.first().copied())
                .unwrap_or_default(),
            cascade_bases: self.schedule.bases.as_ref().map_or([[0; 3]; 3], |b| {
                std::array::from_fn(|k| b.get(k).copied().unwrap_or_default())
            }),
            cascades: self.schedule.bases.as_ref().map_or(0, |b| b.len() as u32),
            gpu_bytes,
            surface_pool: extras.map_or(0, |e| e.count),
            surface_live: extras.map_or(0, Extras::live),
            surface_assigned: extras.map_or(0, |e| e.assigned_total),
            surface_evicted: extras.map_or(0, |e| e.evicted_total),
            surface_requests: extras.map_or(0, |e| e.requests_total),
            periods: self.schedule.periods,
            rows,
        });
    }
}

/// Probes one 64-thread update workgroup takes: one thread per ray (at most 8 probes).
fn probes_per_group(rays: u32) -> u32 {
    (64 / rays.max(1)).clamp(1, 8)
}

/// Updates after which a probe counts as converged under the configured blend. A time-based
/// blend converges within a few refreshes (a probe refreshed every ~1 s takes its slow time
/// constant in updates), but never fewer than 8, so the burst also carries the bounces that
/// arrive one generation per update.
fn converge_age_of(settings: &DdgiSettings) -> u16 {
    match settings.blend {
        DdgiBlend::Update => converge_age(settings.hysteresis),
        DdgiBlend::Time | DdgiBlend::Adaptive => {
            let h = (-1000.0 / settings.blend_slow_ms.max(1.0)).exp();
            converge_age(h).max(8)
        }
    }
}

/// Updates after which a probe counts as converged: the progressive phase of the blend, where the
/// running mean's weight 1/(age+1) reaches the configured hysteresis (capped to 4..64).
fn converge_age(hysteresis: f32) -> u16 {
    (1.0 / (1.0 - hysteresis.min(0.985)))
        .round()
        .clamp(4.0, 64.0) as u16
}

fn push_u32(dst: &mut Vec<u8>, words: [u32; 4]) {
    for word in words {
        dst.extend_from_slice(&word.to_le_bytes());
    }
}

fn push_f32(dst: &mut Vec<u8>, words: [f32; 4]) {
    push_u32(dst, words.map(f32::to_bits));
}

/// Row layout of the one probe buffer, in 16-byte rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DdgiLayout {
    pub meta: u64,
    pub irradiance: u64,
    pub depth: u64,
    pub groups: u64,
    pub rows: u64,
}

pub(crate) fn layout(probes: u32) -> DdgiLayout {
    let probes = u64::from(probes);
    let meta = HEADER_ROWS;
    let irradiance = meta + META_ROWS * probes;
    let depth = irradiance + IRRADIANCE_ROWS * probes;
    let groups = depth + DEPTH_ROWS * probes;
    DdgiLayout {
        meta,
        irradiance,
        depth,
        groups,
        rows: groups + GROUP_ROWS * probes,
    }
}

struct DdgiGpu {
    cascades: Vec<CascadeLayout>,
    revision: u64,
    screen: Option<(u32, u32)>,
    /// Surface-probe pool size (slots after the cascades).
    extras: u32,
    update: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    relight: wgpu::ComputePipeline,
    sky: wgpu::ComputePipeline,
    probes: Buffer,
    params: Buffer,
    options: Buffer,
    jobs: Buffer,
    header: Vec<u8>,
}

impl DdgiGpu {
    fn new(
        gpu: &GpuContext,
        cascades: Vec<CascadeLayout>,
        revision: u64,
        screen: Option<(u32, u32)>,
        extras: u32,
    ) -> Self {
        let device = gpu.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(DDGI_GI_NAME),
            source: wgpu::ShaderSource::Wgsl(ddgi_module().into()),
        });
        let pipeline = |label, entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let update = pipeline("gi.ddgi.update", "ddgi_update");
        let resolve = pipeline("gi.ddgi.resolve", "ddgi_resolve");
        let relight = pipeline("gi.ddgi.relight", "ddgi_relight");
        let sky = pipeline("gi.ddgi.sky", "ddgi_sky_project");
        let pixels = screen.map_or(0, |(w, h)| u64::from(w) * u64::from(h));
        let probes = cascades.iter().map(|c| c.probes).sum::<u32>() + extras;
        let rows =
            layout(probes).rows + pixels + SCRATCH_ROWS + TABLE_ROWS_PER_SLOT * u64::from(extras);
        Self {
            cascades,
            revision,
            screen,
            extras,
            update,
            resolve,
            relight,
            sky,
            probes: buffer(
                device,
                "gi.ddgi.probes",
                16 * rows,
                Usage::STORAGE | Usage::COPY_DST | Usage::COPY_SRC,
            ),
            params: buffer(
                device,
                "gi.ddgi.params",
                PARAMS_BYTES,
                Usage::UNIFORM | Usage::COPY_DST,
            ),
            options: buffer(
                device,
                "gi.ddgi.options",
                OPTIONS_BYTES,
                Usage::UNIFORM | Usage::COPY_DST,
            ),
            jobs: buffer(
                device,
                "gi.ddgi.jobs",
                4 * 1024,
                Usage::STORAGE | Usage::COPY_DST,
            ),
            header: Vec::new(),
        }
    }

    fn ensure_jobs(&mut self, gpu: &GpuContext, count: usize) {
        // `count` words: two per job.
        let bytes = (4 * count as u64).max(16);
        if self.jobs.size() < bytes {
            self.jobs = buffer(
                gpu.device(),
                "gi.ddgi.jobs",
                bytes.next_power_of_two(),
                Usage::STORAGE | Usage::COPY_DST,
            );
        }
    }

    /// Probes of the cascades.
    fn probes(&self) -> u32 {
        self.cascades.iter().map(|c| c.probes).sum()
    }

    /// Every slot: the cascades, then the surface-probe pool.
    fn total(&self) -> u32 {
        self.probes() + self.extras
    }

    /// First row of the surface table: one assignment row per pool slot (cell, live | start
    /// offset << 1), then two request rows per slot.
    fn table_row(&self) -> u64 {
        let pixels = self.screen.map_or(0, |(w, h)| u64::from(w) * u64::from(h));
        layout(self.total()).rows + pixels + SCRATCH_ROWS
    }

    fn bytes(&self) -> u64 {
        self.probes.size() + self.params.size() + self.options.size() + self.jobs.size()
    }

    fn write_header(
        &mut self,
        gpu: &GpuContext,
        bases: &[[i32; 3]],
        closed: &[u32],
        flags: u32,
        settings: DdgiSettings,
    ) {
        let probes = self.probes();
        let rows = layout(self.total());
        let table_row = self.table_row() as u32;
        let spacing = settings.spacing;
        let base = bases[0];
        let mut header = Vec::with_capacity(16 * CPU_HEADER_ROWS as usize);
        push_u32(&mut header, [0; 4]);
        push_u32(&mut header, [0; 4]);
        push_u32(&mut header, [0; 4]);
        // Magic, version, the reach of the lookup quality (m, 0 = any distance).
        push_u32(&mut header, [MAGIC, 1, settings.surface_reach.to_bits(), 0]);
        push_u32(
            &mut header,
            [base[0] as u32, base[1] as u32, base[2] as u32, flags],
        );
        let [x, y, z] = self.cascades[0].counts;
        push_u32(&mut header, [x, y, z, probes]);
        push_f32(
            &mut header,
            [
                spacing,
                settings.intensity,
                settings.normal_bias * spacing,
                settings.view_bias * spacing,
            ],
        );
        push_u32(
            &mut header,
            [
                rows.meta as u32,
                rows.irradiance as u32,
                rows.depth as u32,
                settings.surface_need.to_bits(),
            ],
        );
        let pixels = self.screen.map_or(0, |(w, h)| u64::from(w) * u64::from(h));
        // Distance moments saturate at one and a half probe diagonals; cascade count, closed
        // sides of cascade 0.
        push_u32(
            &mut header,
            [
                (1.5 * 3f32.sqrt() * spacing).to_bits(),
                self.cascades.len() as u32,
                closed[0],
                settings.surface_range.to_bits(),
            ],
        );
        // Screen resolve rows: first row, width, height, pixels per texel side (0 = off).
        let (width, height) = self.screen.unwrap_or((0, 0));
        push_u32(
            &mut header,
            [
                rows.rows as u32,
                width,
                height,
                if self.screen.is_some() {
                    RESOLVE_SCALE
                } else {
                    0
                },
            ],
        );
        // Row 10: texel groups, sky SH rows, ray scratch (after the screen resolve rows).
        push_u32(
            &mut header,
            [
                rows.groups as u32,
                SKY_ROW as u32,
                (rows.rows + pixels) as u32,
                settings.surface_offset.to_bits(),
            ],
        );
        // Rows 11 + 2(k - 1), 12 + 2(k - 1): cascade k >= 1, base and first slot | closed sides
        // << 16, then counts and spacing.
        for (k, c) in self.cascades.iter().enumerate().skip(1) {
            let b = bases[k];
            push_u32(
                &mut header,
                [
                    b[0] as u32,
                    b[1] as u32,
                    b[2] as u32,
                    c.offset | (closed[k] << 16),
                ],
            );
            push_u32(
                &mut header,
                [c.counts[0], c.counts[1], c.counts[2], c.spacing.to_bits()],
            );
        }
        // Row 15: the surface-probe pool (table row, size, first slot, lattice pitch).
        header.resize(16 * 15, 0);
        push_u32(
            &mut header,
            [
                table_row,
                self.extras,
                probes,
                settings.surface_spacing.to_bits(),
            ],
        );
        if header != self.header {
            gpu.queue().write_buffer(&self.probes, 0, &header);
            self.header = header;
        }
    }
}

pub(crate) fn ddgi_module() -> String {
    format!(
        "{}\n{}\n{}",
        crate::wgsl::mesh_trace_module(),
        include_str!("../wgsl/mesh_ddgi.wgsl"),
        raa_shader::DDGI_GI_WGSL,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ddgi_update_module_validates() {
        raa_shader::Composer::validate(&ddgi_module()).expect("DDGI update WGSL");
    }

    #[test]
    fn layout_keeps_tiles_row_aligned() {
        let rows = layout(16 * 8 * 16);
        assert_eq!(rows.meta, 32);
        assert_eq!(rows.irradiance, 32 + 4 * 2048);
        assert_eq!(rows.depth, rows.irradiance + 50 * 2048);
        assert_eq!(rows.groups, rows.depth + 81 * 2048);
        assert_eq!(rows.rows, rows.groups + 320 * 2048);
        // The default volume with its 960x540 resolve and the ray scratch: 51 MB.
        let total = layout(24 * 12 * 16).rows + 960 * 540 + SCRATCH_ROWS;
        assert!(16 * total < 64 << 20, "{total}");
    }

    #[test]
    fn defaults_validate_and_reject_bad_counts() {
        DdgiSettings::default()
            .validate()
            .expect("default DDGI settings");
        let mut bad = DdgiSettings::default();
        bad.counts = [1, 8, 8];
        assert!(bad.validate().is_err());
        bad.counts = [128, 128, 128];
        assert!(bad.validate().is_err());
    }

    #[test]
    fn volume_stays_inside_the_scene_bounds() {
        let bounds = Some(([-15.0, -0.2, -6.5], [14.0, 15.0, 6.5]));
        // z spans 13 m + padding = 13 probe cells at 1.3 m: fits in 24, centred, independent of the eye
        let a = volume_base([-8.8, 1.6, -0.9], [24, 12, 24], 1.3, bounds);
        let b = volume_base([5.0, 7.0, 3.0], [24, 12, 24], 1.3, bounds);
        assert_eq!(a[2], b[2]);
        assert!(a[2] <= (-6.5f32 / 1.3).floor() as i32 - 1);
        assert!(a[2] + 23 >= (6.5f32 / 1.3).ceil() as i32 + 1);
        // x (29 m) does not fit 24 * 1.3: the camera-centred origin is clamped to the padded bounds
        assert_eq!(a[0], (-15.0f32 / 1.3).floor() as i32 - 1);
        assert_eq!(b[0] + 23, (14.0f32 / 1.3).ceil() as i32 + 1);
        // no geometry: camera-centred
        assert_eq!(volume_base([0.0; 3], [4, 4, 4], 1.0, None), [-2, -2, -2]);
    }

    #[test]
    fn cascades_share_one_slot_space_and_close_at_the_scene_bounds() {
        let settings = DdgiSettings::default();
        let cascades = settings.cascade_layouts();
        assert_eq!(cascades.len(), 2);
        assert_eq!(cascades[1].offset, 24 * 12 * 16);
        assert_eq!(cascades[1].spacing, 3.0);
        assert_eq!(settings.probe_count(), 24 * 12 * 16 + 16 * 8 * 12);
        assert!(
            settings.probe_count() < 1 << 16,
            "a job carries a 16-bit slot"
        );
        // Intel Sponza's measured bounds: the coarse cascade spans x and z whole (both sides
        // closed); cascade 0 at the V1 camera stops 3.5 m short of the far end of the nave (+x
        // open) while its -x side is clamped to the bounds (closed).
        let bounds = Some(([-16.25, -1.13, -9.40], [20.0, 18.76, 14.36]));
        let eye = [-8.8, 1.6, -0.86];
        let b0 = volume_base(eye, cascades[0].counts, 1.5, bounds);
        let m0 = closed_sides(b0, cascades[0].counts, 1.5, bounds);
        assert_eq!(m0 & 3, 1, "cascade 0: -x closed, +x open ({b0:?})");
        let b1 = volume_base(eye, cascades[1].counts, 3.0, bounds);
        let m1 = closed_sides(b1, cascades[1].counts, 3.0, bounds);
        assert_eq!(m1 & 0b110011, 0b110011, "cascade 1 spans x and z ({b1:?})");
        assert_eq!(closed_sides([0; 3], [4; 3], 1.0, None), 0);
        let mut bad = DdgiSettings::default();
        bad.cascades = 4;
        assert!(bad.validate().is_err());
        bad.cascades = 1;
        assert_eq!(bad.validate().unwrap().probe_count(), 24 * 12 * 16);
    }

    #[test]
    fn lighting_steps_map_to_ages() {
        assert!(angle_degrees([0.0, -1.0, 0.0], [0.0, -1.0, 0.0]) < 1e-3);
        let turned = angle_degrees([0.0, -1.0, 0.0], [1.0, -1.0, 0.0]);
        assert!((turned - 45.0).abs() < 1e-3);
        // A large rotation restarts the running mean, a small one keeps most of the history.
        assert_eq!(sun_age(45.0), RELIGHT_AGE);
        assert_eq!(sun_age(1.0), 6);
        assert_eq!(sun_age(SUN_STEP_DEGREES), 12);
        let mut a = [[0.0f32; 4]; 9];
        a[0] = [1.0, 2.0, 4.0, 0.0];
        let mut b = a;
        assert_eq!(sky_change(&a, &b), 0.0);
        b[3][1] = 0.4;
        assert!((sky_change(&a, &b) - 0.1).abs() < 1e-6);
        // Brighter with the same shape (a sun intensity edit under the physical sky): no change.
        let brighter = b.map(|c| c.map(|x| 3.0 * x));
        assert!(sky_change(&b, &brighter) < 1e-6);
        assert_eq!(sky_change(&a, &[[0.0; 4]; 9]), 1.0);
    }

    #[test]
    fn random_rotations_are_orthonormal() {
        let mut pass_rng = DdgiPass {
            state: DdgiHandle::default(),
            mesh: MeshTraceHandle::default(),
            gpu: None,
            schedule: Schedule::default(),
            extras: None,
            staged: None,
            published: false,
            rng: 12345,
            timer: None,
            ms_per_probe: INITIAL_MS_PER_PROBE,
            update_ms: None,
            batch_ms: 0.0,
            idle_frames: 0,
            lighting: LightingTrack::default(),
            clock: 0.0,
            geometry_revision: None,
        };
        for _ in 0..16 {
            let m = pass_rng.random_rotation();
            for i in 0..3 {
                for j in 0..3 {
                    let dot: f32 = (0..3).map(|k| m[i][k] * m[j][k]).sum();
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((dot - want).abs() < 1e-4, "{m:?}");
                }
            }
        }
    }
}
