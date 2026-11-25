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

mod schedule;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc::{Receiver, TryRecvError, sync_channel};

use raa_core::{GpuContext, RaaError};
use raa_pipeline::{FeaturePass, LightingResources, Pipeline, PrepareCtx, RecordCtx};
use wgpu::{Buffer, BufferUsages as Usage};

use self::schedule::Schedule;
pub use self::schedule::{DdgiView, Visibility as DdgiVisibility};
use super::MeshTraceHandle;
use super::pass::{buffer, buffer_entry, image_entry};

pub const DDGI_GI_NAME: &str = "gi.ddgi";
/// Frame feature bit read by `opaque_lighting.wgsl` (raa-shader `RAA_DDGI_GI_FEATURE`).
pub const DDGI_FRAME_FEATURE: u32 = raa_shader::RAA_DDGI_GI_FEATURE;
const ORDER: u32 = 57;
const MAGIC: u32 = 0x4947_4444;
const HEADER_ROWS: u64 = 16;
const META_ROWS: u64 = 2;
const IRRADIANCE_ROWS: u64 = 50;
const DEPTH_ROWS: u64 = 81;
const MAX_PROBES: u32 = 32_768;
const MAX_AXIS: u32 = 128;
const PARAMS_BYTES: u64 = 96;
const OPTIONS_BYTES: u64 = 64;
const FLAG_RELOCATE: u32 = 1;
const FLAG_CLASSIFY: u32 = 2;
const FLAG_BOUNCE: u32 = 4;

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
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DdgiSettings {
    pub enabled: bool,
    pub counts: [u32; 3],
    /// Probe spacing in metres.
    pub spacing: f32,
    pub rays_per_probe: u32,
    pub hysteresis: f32,
    /// Most probes updated per frame outside a burst, as a fraction of the volume.
    pub update_fraction: f32,
    /// GPU time of the probe update per frame (ms), converted with the measured cost per probe.
    pub budget_ms: f32,
    /// GPU time per frame while a visible probe is invalid or still converging (ms).
    pub burst_ms: f32,
    /// Interpolate the volume once per prepass pixel instead of in the forward shader.
    pub screen_resolve: bool,
    pub intensity: f32,
    /// Surface biases as fractions of the probe spacing.
    pub normal_bias: f32,
    pub view_bias: f32,
    pub relocation: bool,
    pub classification: bool,
    /// Shade ray hits with the previous volume (infinite bounces over time).
    pub bounce: bool,
    pub debug: DdgiDebug,
}

impl Default for DdgiSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            // 36 x 18 x 24 m: an Intel Sponza sized interior with little scrolling.
            counts: [24, 12, 16],
            spacing: 1.5,
            rays_per_probe: 64,
            hysteresis: 0.97,
            update_fraction: 0.25,
            budget_ms: 0.6,
            burst_ms: 2.0,
            screen_resolve: true,
            intensity: 1.0,
            normal_bias: 0.1,
            view_bias: 0.3,
            relocation: true,
            classification: true,
            bounce: true,
            debug: DdgiDebug::Off,
        }
    }
}

impl DdgiSettings {
    pub fn probe_count(&self) -> u32 {
        self.counts.iter().product()
    }

    pub fn validate(self) -> Result<Self, RaaError> {
        let counts_ok = self.counts.iter().all(|&n| (2..=MAX_AXIS).contains(&n))
            && self.probe_count() <= MAX_PROBES;
        let ok = counts_ok
            && self.spacing.is_finite()
            && (0.05..=100.0).contains(&self.spacing)
            && (16..=256).contains(&self.rays_per_probe)
            && (0.0..=0.995).contains(&self.hysteresis)
            && self.update_fraction > 0.0
            && self.update_fraction <= 1.0
            && (0.05..=20.0).contains(&self.budget_ms)
            && (0.05..=20.0).contains(&self.burst_ms)
            && (0.0..=16.0).contains(&self.intensity)
            && (0.0..=1.0).contains(&self.normal_bias)
            && (0.0..=1.0).contains(&self.view_bias);
        if !ok {
            return Err(RaaError::FrameGraph {
                code: "gi.ddgi.invalid_settings",
                message: format!(
                    "expected probe counts in [2, {MAX_AXIS}] with at most {MAX_PROBES} probes, spacing in [0.05, 100] m, \
                     16-256 rays, hysteresis in [0, 0.995], update fraction in (0, 1], budgets in [0.05, 20] ms, intensity in [0, 16] and biases in [0, 1]"
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
    pub screen_resolve: bool,
    pub update_frames: u64,
    pub recycled: u64,
    pub base: [i32; 3],
    pub gpu_bytes: u64,
    /// World bounds of the traced geometry the volume is kept inside.
    pub scene_bounds: Option<([f32; 3], [f32; 3])>,
}

pub type DdgiHandle = Rc<RefCell<DdgiState>>;

#[derive(Default)]
pub struct DdgiState {
    pub settings: DdgiSettings,
    view: Option<(usize, DdgiView)>,
    pass_enabled: bool,
    revision: u64,
    stats: Option<DdgiStats>,
}

impl DdgiState {
    pub fn configure(&mut self, settings: DdgiSettings) -> Result<(), RaaError> {
        let settings = settings.validate()?;
        if settings.counts != self.settings.counts || settings.spacing != self.settings.spacing {
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
            staged: None,
            published: false,
            rng: 0x9e37_79b9_7f4a_7c15,
            timer: None,
            ms_per_probe: INITIAL_MS_PER_PROBE,
            update_ms: None,
            batch_ms: 0.0,
            idle_frames: 0,
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

struct Staged {
    bounds: Option<([f32; 3], [f32; 3])>,
    jobs: Vec<u32>,
    recycled: u32,
    burst: bool,
    visibility: DdgiVisibility,
    converge_age: u32,
    interval: u32,
    /// Screen resolve size, when it runs this frame.
    screen: Option<(u32, u32)>,
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
    staged: Option<Staged>,
    published: bool,
    rng: u64,
    timer: Option<UpdateTimer>,
    ms_per_probe: f64,
    update_ms: Option<f64>,
    /// Smoothed GPU time of one update batch (ms), 0 before the first readback.
    batch_ms: f64,
    idle_frames: u32,
}

impl DdgiPass {
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
            self.timer = None;
            self.state.borrow_mut().stats = None;
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
        let probes = settings.probe_count();
        // One row per resolve texel inside the forward's one storage binding.
        let resolve_size = (
            ctx.size.0.div_ceil(RESOLVE_SCALE),
            ctx.size.1.div_ceil(RESOLVE_SCALE),
        );
        let rows = layout(probes).rows + u64::from(resolve_size.0) * u64::from(resolve_size.1);
        let fits = 16 * rows <= u64::from(ctx.gpu.limits().max_storage_buffer_binding_size);
        let screen = (settings.screen_resolve && fits).then_some(resolve_size);
        if self.gpu.as_ref().is_none_or(|gpu| {
            gpu.counts != settings.counts || gpu.revision != revision || gpu.screen != screen
        }) {
            // A resize restarts the volume: the screen rows share the probe buffer.
            self.gpu = Some(DdgiGpu::new(ctx.gpu, settings.counts, revision, screen));
            self.schedule.reset(settings.counts);
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
        let base = volume_base(camera.eye, settings.counts, settings.spacing, bounds);
        let recycled = self.schedule.scroll(base);
        let floor = MIN_BATCH.min(probes);
        let probes_for =
            |ms: f32| ((f64::from(ms) / self.ms_per_probe) as u32).clamp(floor, probes);
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
        let converge_age = converge_age(settings.hysteresis);
        let mut jobs = Vec::with_capacity(burst as usize);
        let (visibility, bursting) = self.schedule.plan(
            Some(&camera),
            settings.spacing,
            converge_age,
            budget,
            burst,
            &mut jobs,
        );
        if !jobs.is_empty() {
            self.idle_frames = 0;
        }

        let triangles = mesh.geometry.count();
        let traceable = triangles > 0 && mesh.frame.is_some() && mesh.texture_atlas.is_some();
        let sky_diffuse = mesh.sky_diffuse;
        drop(mesh);

        let rotation = self.random_rotation();
        let gpu = self.gpu.as_mut().expect("DDGI pool");
        let mut flags = settings.debug as u32 * 256;
        if settings.classification {
            flags |= FLAG_CLASSIFY;
        }
        if settings.relocation {
            flags |= FLAG_RELOCATE;
        }
        gpu.write_header(ctx.gpu, base, flags, settings);
        let resolve = gpu.screen;
        ctx.lighting
            .set_spherical_gi(Some((gpu.probes.clone(), gpu.probes.clone())));
        self.published = true;
        if !traceable {
            return Ok(());
        }
        let extent = settings
            .counts
            .iter()
            .map(|&n| n as f32 * settings.spacing)
            .fold(0.0, f32::max);
        let mut update_flags = flags & (FLAG_CLASSIFY | FLAG_RELOCATE);
        if settings.bounce {
            update_flags |= FLAG_BOUNCE;
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
                jobs.len() as u32,
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
        push_f32(&mut params, [sky_diffuse, 0.25, 0.0, 0.0]);
        let mut options = Vec::with_capacity(OPTIONS_BYTES as usize);
        push_u32(
            &mut options,
            [triangles, jobs.len() as u32, u32::from(env), 0],
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
        queue.write_buffer(&gpu.params, 0, &params);
        queue.write_buffer(&gpu.options, 0, &options);
        let job_bytes: Vec<u8> = jobs.iter().flat_map(|slot| slot.to_le_bytes()).collect();
        queue.write_buffer(&gpu.jobs, 0, &job_bytes);
        self.staged = Some(Staged {
            bounds,
            jobs,
            recycled,
            burst: bursting,
            visibility,
            converge_age: u32::from(converge_age),
            interval: if bursting { 1 } else { interval },
            screen: resolve,
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
        let jobs = staged.jobs.len() as u32;
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
                pass.dispatch_workgroups(jobs, 1, 1);
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
        Ok(())
    }

    fn submitted(&mut self) {
        let Some(staged) = self.staged.take() else {
            return;
        };
        let mut state = self.state.borrow_mut();
        let probes = state.settings.probe_count();
        let previous = state.stats.unwrap_or_default();
        let gpu_bytes = self.gpu.as_ref().map_or(0, DdgiGpu::bytes);
        state.stats = Some(DdgiStats {
            probes,
            scheduled: staged.jobs.len() as u32,
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
            base: self.schedule.base.unwrap_or_default(),
            gpu_bytes,
        });
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
    pub rows: u64,
}

pub(crate) fn layout(probes: u32) -> DdgiLayout {
    let probes = u64::from(probes);
    let meta = HEADER_ROWS;
    let irradiance = meta + META_ROWS * probes;
    let depth = irradiance + IRRADIANCE_ROWS * probes;
    DdgiLayout {
        meta,
        irradiance,
        depth,
        rows: depth + DEPTH_ROWS * probes,
    }
}

struct DdgiGpu {
    counts: [u32; 3],
    revision: u64,
    screen: Option<(u32, u32)>,
    update: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    probes: Buffer,
    params: Buffer,
    options: Buffer,
    jobs: Buffer,
    header: Vec<u8>,
}

impl DdgiGpu {
    fn new(gpu: &GpuContext, counts: [u32; 3], revision: u64, screen: Option<(u32, u32)>) -> Self {
        let device = gpu.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(DDGI_GI_NAME),
            source: wgpu::ShaderSource::Wgsl(ddgi_module().into()),
        });
        let update = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gi.ddgi.update"),
            layout: None,
            module: &shader,
            entry_point: Some("ddgi_update"),
            compilation_options: Default::default(),
            cache: None,
        });
        let resolve = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gi.ddgi.resolve"),
            layout: None,
            module: &shader,
            entry_point: Some("ddgi_resolve"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pixels = screen.map_or(0, |(w, h)| u64::from(w) * u64::from(h));
        let rows = layout(counts.iter().product()).rows + pixels;
        Self {
            counts,
            revision,
            screen,
            update,
            resolve,
            probes: buffer(
                device,
                "gi.ddgi.probes",
                16 * rows,
                Usage::STORAGE | Usage::COPY_DST,
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

    fn bytes(&self) -> u64 {
        self.probes.size() + self.params.size() + self.options.size() + self.jobs.size()
    }

    fn write_header(
        &mut self,
        gpu: &GpuContext,
        base: [i32; 3],
        flags: u32,
        settings: DdgiSettings,
    ) {
        let probes = settings.probe_count();
        let rows = layout(probes);
        let spacing = settings.spacing;
        let mut header = Vec::with_capacity(16 * HEADER_ROWS as usize);
        push_u32(&mut header, [0; 4]);
        push_u32(&mut header, [0; 4]);
        push_u32(&mut header, [0; 4]);
        push_u32(&mut header, [MAGIC, 1, 0, 0]);
        push_u32(
            &mut header,
            [base[0] as u32, base[1] as u32, base[2] as u32, flags],
        );
        let [x, y, z] = settings.counts;
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
                0,
            ],
        );
        // Distance moments saturate at one and a half probe diagonals.
        push_f32(&mut header, [1.5 * 3f32.sqrt() * spacing, 0.0, 0.0, 0.0]);
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
        header.resize(16 * HEADER_ROWS as usize, 0);
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
        assert_eq!(rows.meta, 16);
        assert_eq!(rows.irradiance, 16 + 2 * 2048);
        assert_eq!(rows.depth, rows.irradiance + 50 * 2048);
        assert_eq!(rows.rows, rows.depth + 81 * 2048);
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
    fn random_rotations_are_orthonormal() {
        let mut pass_rng = DdgiPass {
            state: DdgiHandle::default(),
            mesh: MeshTraceHandle::default(),
            gpu: None,
            schedule: Schedule::default(),
            staged: None,
            published: false,
            rng: 12345,
            timer: None,
            ms_per_probe: INITIAL_MS_PER_PROBE,
            update_ms: None,
            batch_ms: 0.0,
            idle_frames: 0,
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
