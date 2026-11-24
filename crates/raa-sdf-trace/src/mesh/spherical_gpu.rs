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
    ALL_FACES, CellKey, FaceBudget, FaceJob, RadialShell, SKY_LIGHT_ID, SUN_LIGHT_ID,
    SchedulerStats, SphericalLight, SphericalScheduler, SphericalSchedulerConfig,
};

pub const SPHERICAL_GI_NAME: &str = "gi.spherical";
const ORDER: u32 = 58;
const TEXELS_PER_FACE: usize = 64;
const FACE_COUNT: usize = 6;
const FACE_LIGHTING_BYTES: usize = 64;
const TEXEL_BYTES: usize = 48;
const CELL_BYTES: usize = TEXELS_PER_FACE * FACE_COUNT * TEXEL_BYTES;
const WORK_BYTES: usize = 48;
const LOOKUP_HEADER: usize = 4 + 64;
const LOOKUP_ENTRY: usize = 3;
const CHANNELS: usize = 3;
const MAX_FACES_PER_BUDGET: u32 = 4096;

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
    pub budget: FaceBudget,
    pub intensity: f32,
    pub debug: SphericalGiDebug,
    /// Bit 0 selects sky, bit 1 sun and bit 2 other.
    pub channels: u32,
}

impl Default for SphericalGiSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            scheduler: SphericalSchedulerConfig::default(),
            budget: FaceBudget { near: 24, far: 12 },
            intensity: 1.0,
            debug: SphericalGiDebug::Off,
            channels: 7,
        }
    }
}

impl SphericalGiSettings {
    pub fn validate(self) -> Result<Self, RaaError> {
        if !self.intensity.is_finite()
            || !(0.0..=16.0).contains(&self.intensity)
            || self.channels > 7
            || self.budget.near > MAX_FACES_PER_BUDGET
            || self.budget.far > MAX_FACES_PER_BUDGET
        {
            return Err(RaaError::FrameGraph {
                code: "gi.spherical.invalid_settings",
                message: format!(
                    "expected intensity in [0, 16], channel mask in [0, 7] and face budgets in [0, {MAX_FACES_PER_BUDGET}]"
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
}

struct SphericalPass {
    state: SphericalGiHandle,
    mesh: MeshTraceHandle,
    gpu: Option<SphericalGpu>,
    staged: Staged,
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
        let mesh = self.mesh.borrow();
        let light_revision = light_revision(ctx.lighting, view.light, mesh.sky_diffuse);
        let capacity = settings.scheduler.max_cells;
        if self.gpu.as_ref().is_none_or(|gpu| gpu.capacity != capacity) {
            self.gpu = Some(SphericalGpu::new(ctx.gpu, capacity, grid_revision));
        }
        let gpu = self.gpu.as_mut().expect("spherical GI pool");
        if gpu.grid_revision != grid_revision {
            gpu.grid_revision = grid_revision;
            gpu.ready_at.fill(None);
            gpu.waiting_lights.clear();
            gpu.light_feedback = None;
        }
        gpu.poll_light_feedback(scheduler);
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
        scheduler.update_with_lights(&mesh, view.origin, light_revision, local_lights);
        gpu.waiting_lights
            .retain(|sample| captured_face_current(scheduler, sample.job));
        for slot in scheduler.take_released_slots() {
            gpu.ready_at[slot as usize] = None;
        }
        gpu.update_lookup(ctx.gpu, scheduler, settings, ctx.frame as u32);
        ctx.lighting.set_spherical_gi(Some(gpu.lookup.clone()));
        self.staged = Staged {
            jobs: scheduler.take_jobs(settings.budget),
            light_ids,
            feedback: None,
            grid_revision,
            frame: ctx.frame,
            light_revision,
            primary: true,
            recorded: false,
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
        if !self.staged.jobs.is_empty() {
            let mut mesh = self.mesh.borrow_mut();
            let Some(frame) = mesh.frame.clone() else {
                return Ok(());
            };
            let geometry = mesh.geometry_buffers(ctx.gpu);
            gpu.record(
                ctx,
                scheduler,
                &self.staged.jobs,
                &frame,
                &geometry,
                mesh.geometry.count(),
                mesh.sky_diffuse,
            );
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
            let origin = std::array::from_fn(|i| cell.surface_position[i] + 0.002 * cell.normal[i]);
            let probe = gpu.record_probe(ctx, id, origin, cell.light_hash, light_ids);
            state.probe = Some(probe);
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
        for job in staged.jobs {
            if scheduler.complete(job) {
                committed += 1;
                gpu.waiting_lights.push(FaceSample {
                    job,
                    ids: staged.light_ids.clone(),
                });
                if scheduler
                    .progress(job.key)
                    .is_some_and(|cell| cell.is_ready())
                {
                    gpu.ready_at[job.slot as usize] = Some((job.epoch, staged.frame as u32));
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
        });
    }
}

fn light_revision(lighting: &LightingResources, view: [f32; 10], sky_diffuse: f32) -> u64 {
    let mut hash = DefaultHasher::new();
    view.map(f32::to_bits).hash(&mut hash);
    if let Some(sh) = lighting.environment_diffuse_sh() {
        sky_diffuse.to_bits().hash(&mut hash);
        for coefficient in sh {
            coefficient.map(f32::to_bits).hash(&mut hash);
        }
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

fn write_work(dst: &mut Vec<u8>, job: FaceJob, index: u32) {
    let [x, y, z] = job.surface_position;
    write_words(dst, [x, y, z, 0.0].map(f32::to_bits));
    let [x, y, z] = job.normal;
    write_words(dst, [x, y, z, 0.0].map(f32::to_bits));
    write_words(dst, [job.slot, index, 0, 0]);
}

struct SphericalGpu {
    capacity: u32,
    grid_revision: u64,
    capture: wgpu::ComputePipeline,
    project: wgpu::ComputePipeline,
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
    ready_at: Vec<Option<(u64, u32)>>,
    face_masks: Vec<u8>,
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
        let slots = capacity as usize;
        let table_entries = (2 * slots).next_power_of_two();
        let lookup_rows = LOOKUP_HEADER + LOOKUP_ENTRY * table_entries;
        let sized = |label, bytes: usize, usage| buffer(device, label, bytes as u64, usage);
        Self {
            capacity,
            grid_revision,
            capture: pipeline("spherical_capture"),
            project: pipeline("spherical_project"),
            options: sized("gi.spherical.options", 64, Usage::UNIFORM | Usage::COPY_DST),
            work: sized(
                "gi.spherical.work",
                WORK_BYTES * (FACE_COUNT + 1) * slots,
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
                16 * (lookup_rows + CHANNELS * slots),
                Usage::STORAGE | Usage::COPY_DST,
            ),
            table_entries,
            lookup_rows: vec![[0; 4]; lookup_rows],
            uploaded_rows: Vec::new(),
            lookup_bytes: Vec::new(),
            work_bytes: Vec::new(),
            ready_at: vec![None; slots],
            face_masks: vec![0; slots],
        }
    }

    fn bytes(&self) -> u64 {
        self.options.size()
            + self.work.size()
            + self.faces.size()
            + self.face_lighting.size()
            + self.lookup.size()
            + self
                .light_feedback
                .as_ref()
                .map_or(0, |pending| pending.readback.size())
    }

    fn update_lookup(
        &mut self,
        gpu: &GpuContext,
        scheduler: &SphericalScheduler,
        settings: SphericalGiSettings,
        frame: u32,
    ) {
        let rows = &mut self.lookup_rows;
        rows.fill([0; 4]);
        let grid = scheduler.grid();
        let config = settings.scheduler.grid;
        let mask = self.table_entries - 1;
        let [x, y, z] = grid.origin();
        rows[0] = [x, y, z, settings.intensity].map(f32::to_bits);
        rows[1] = [
            grid.angular_resolution().into(),
            grid.shell_count().into(),
            mask as u32,
            settings.debug as u32,
        ];
        let age_frame = if settings.debug == SphericalGiDebug::Age {
            frame
        } else {
            0
        };
        rows[2] = [
            config.clip_start.to_bits(),
            config.z_max.to_bits(),
            age_frame,
            settings.channels,
        ];
        for index in 0..grid.shell_count() {
            let shell = grid.radial_shell(index).expect("validated radial shell");
            rows[4 + usize::from(index)] =
                [shell.inner, shell.center, shell.outer, 0.0].map(f32::to_bits);
        }
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
            let ready_frame = self.ready_at[cell.slot as usize]
                .filter(|&(epoch, _)| epoch == cell.epoch)
                .map_or(0, |(_, frame)| frame);
            let shell = grid
                .radial_shell(cell.key.shell)
                .expect("cell radial shell");
            let radius = (shell.outer - shell.inner)
                .max(2.0 * shell.center / f32::from(config.angular_resolution));
            let hash = cell.light_hash;
            rows[row] = [key, cell.slot, ready_frame, (hash ^ (hash >> 32)) as u32];
            let [x, y, z] = cell.surface_position;
            rows[row + 1] = [x, y, z, radius].map(f32::to_bits);
            let [x, y, z] = cell.normal.map(f32::to_bits);
            rows[row + 2] = [x, y, z, u32::from(cell.is_ready())];
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

    fn record(
        &mut self,
        ctx: &mut RecordCtx<'_>,
        scheduler: &SphericalScheduler,
        jobs: &[FaceJob],
        frame: &Buffer,
        geometry: &[Buffer; 3],
        triangle_count: u32,
        sky_diffuse: f32,
    ) {
        let device = ctx.gpu.device();
        self.work_bytes.clear();
        for &job in jobs {
            write_work(&mut self.work_bytes, job, job.face as u32);
        }
        self.face_masks.fill(0);
        let mut projected = 0;
        for &job in jobs {
            let mask = &mut self.face_masks[job.slot as usize];
            let ready = scheduler
                .progress(job.key)
                .expect("scheduled cell")
                .ready_faces;
            let bit = 1 << job.face as u8;
            if (ready | *mask) != ALL_FACES && (ready | *mask | bit) == ALL_FACES {
                let index = self.lookup_rows.len() + CHANNELS * job.slot as usize;
                write_work(&mut self.work_bytes, job, index as u32);
                projected += 1;
            }
            *mask |= bit;
        }
        let env = ctx.lighting.environment_diffuse_sh().is_some();
        let mut options = Vec::with_capacity(64);
        write_words(
            &mut options,
            [triangle_count, jobs.len() as u32, u32::from(env), 0],
        );
        write_words(&mut options, [0; 4]);
        let end = scheduler
            .grid()
            .radial_shell(scheduler.grid().shell_count() - 1)
            .expect("last shell")
            .outer;
        write_words(&mut options, [sky_diffuse.to_bits(), end.to_bits(), 0, 0]);
        // The shared local-light integrator uses its one-tap area estimator for this capture.
        write_words(&mut options, [1, 0, 0, 0]);
        let queue = ctx.gpu.queue();
        queue.write_buffer(&self.options, 0, &options);
        queue.write_buffer(&self.work, 0, &self.work_bytes);
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
            ],
        });
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gi.spherical.capture"),
                ..Default::default()
            });
        pass.set_pipeline(&self.capture);
        pass.set_bind_group(0, &capture_group, &[]);
        pass.dispatch_workgroups(1, 1, jobs.len() as u32);
        drop(pass);
        if projected == 0 {
            return;
        }
        let project_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi.spherical.project"),
            layout: &self.project.get_bind_group_layout(0),
            entries: &[
                buffer_entry(1, &self.options),
                buffer_entry(21, &self.work),
                buffer_entry(22, &self.faces),
                buffer_entry(23, &self.lookup),
            ],
        });
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gi.spherical.project"),
                ..Default::default()
            });
        pass.set_pipeline(&self.project);
        pass.set_bind_group(0, &project_group, &[]);
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

fn spherical_module() -> String {
    format!(
        "{}\n{}\n{}",
        crate::wgsl::mesh_trace_module(),
        raa_shader::SPHERICAL_CUBE_WGSL,
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
        if f32_at(12) == 1.0 {
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
    fn spherical_compute_module_validates() {
        raa_shader::Composer::validate(&super::spherical_module())
            .expect("spherical capture and project WGSL");
    }
}
