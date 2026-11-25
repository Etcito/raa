//! Dynamic diffuse global illumination (Majercik et al. 2019, with the probe relocation and
//! classification of Majercik et al. 2021) on the shared world-space mesh BVH.
//!
//! A camera-centred, world-aligned probe grid is addressed toroidally, so a scroll recycles only
//! the planes that leave the volume. Each frame a scheduled subset of probes traces rotated
//! spherical-Fibonacci rays, shades their hits (sun, local lights, emission and last frame's
//! volume), relocates and classifies itself and blends octahedral irradiance (8x8) and distance
//! moments (16x16). Forward and VG shading read the same storage buffer through the spherical GI
//! binding while `RAA_DDGI_GI_FEATURE` is set.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError};
use raa_pipeline::{FeaturePass, LightingResources, Pipeline, PrepareCtx, RecordCtx};
use wgpu::{Buffer, BufferUsages as Usage};

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
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DdgiSettings {
    pub enabled: bool,
    pub counts: [u32; 3],
    /// Probe spacing in metres.
    pub spacing: f32,
    pub rays_per_probe: u32,
    pub hysteresis: f32,
    /// Fraction of the probes scheduled each frame (recycled probes first).
    pub update_fraction: f32,
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
            counts: [16, 8, 16],
            spacing: 1.5,
            rays_per_probe: 64,
            hysteresis: 0.97,
            update_fraction: 0.25,
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
            && (0.0..=16.0).contains(&self.intensity)
            && (0.0..=1.0).contains(&self.normal_bias)
            && (0.0..=1.0).contains(&self.view_bias);
        if !ok {
            return Err(RaaError::FrameGraph {
                code: "gi.ddgi.invalid_settings",
                message: format!(
                    "expected probe counts in [2, {MAX_AXIS}] with at most {MAX_PROBES} probes, spacing in [0.05, 100] m, \
                     16-256 rays, hysteresis in [0, 0.995], update fraction in (0, 1], intensity in [0, 16] and biases in [0, 1]"
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
    pub update_frames: u64,
    pub recycled: u64,
    pub base: [i32; 3],
    pub gpu_bytes: u64,
}

pub type DdgiHandle = Rc<RefCell<DdgiState>>;

#[derive(Default)]
pub struct DdgiState {
    pub settings: DdgiSettings,
    view: Option<(usize, [f32; 3])>,
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

    /// The bridge supplies the effective camera position of each view before the frame.
    pub fn set_view(&mut self, index: usize, eye: [f32; 3]) {
        self.view = Some((index, eye));
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
        }))?;
        pipeline.set_enabled(DDGI_GI_NAME, false)?;
        Ok(state)
    }
}

/// Which probe slots to update: recycled slots first, then a round robin over the volume.
#[derive(Default)]
struct Schedule {
    base: Option<[i32; 3]>,
    counts: [u32; 3],
    cursor: u32,
    recycled: VecDeque<u32>,
    pending: Vec<bool>,
}

impl Schedule {
    fn reset(&mut self, counts: [u32; 3]) {
        let total = counts.iter().product::<u32>() as usize;
        *self = Self {
            counts,
            pending: vec![false; total],
            ..Self::default()
        };
    }

    fn slot(&self, g: [i32; 3]) -> u32 {
        let n = self.counts.map(|n| n as i32);
        let s: [u32; 3] = std::array::from_fn(|i| g[i].rem_euclid(n[i]) as u32);
        s[0] + self.counts[0] * (s[1] + self.counts[1] * s[2])
    }

    /// Moves the volume to `base`, queuing every slot whose world coordinate changed.
    fn scroll(&mut self, base: [i32; 3]) -> u32 {
        let Some(old) = self.base.replace(base) else {
            return 0;
        };
        if old == base {
            return 0;
        }
        let n = self.counts.map(|n| n as i32);
        let mut queued = 0;
        for z in 0..n[2] {
            for y in 0..n[1] {
                for x in 0..n[0] {
                    let g = [base[0] + x, base[1] + y, base[2] + z];
                    let kept = (0..3).all(|i| g[i] >= old[i] && g[i] < old[i] + n[i]);
                    if !kept {
                        let slot = self.slot(g);
                        if !std::mem::replace(&mut self.pending[slot as usize], true) {
                            self.recycled.push_back(slot);
                            queued += 1;
                        }
                    }
                }
            }
        }
        queued
    }

    fn take(&mut self, budget: u32, out: &mut Vec<u32>) {
        out.clear();
        let total = self.pending.len() as u32;
        while out.len() < budget as usize {
            let Some(slot) = self.recycled.pop_front() else {
                break;
            };
            self.pending[slot as usize] = false;
            out.push(slot);
        }
        let mut guard = 0;
        while out.len() < budget as usize && guard < total {
            let slot = self.cursor;
            self.cursor = (self.cursor + 1) % total;
            guard += 1;
            if !self.pending[slot as usize] {
                out.push(slot);
            }
        }
    }
}

struct Staged {
    jobs: Vec<u32>,
    recycled: u32,
}

struct DdgiPass {
    state: DdgiHandle,
    mesh: MeshTraceHandle,
    gpu: Option<DdgiGpu>,
    schedule: Schedule,
    staged: Option<Staged>,
    published: bool,
    rng: u64,
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
        let Some((_, eye)) = view.filter(|(index, _)| *index == 0) else {
            self.unpublish(ctx.lighting);
            return Ok(());
        };
        let probes = settings.probe_count();
        if self
            .gpu
            .as_ref()
            .is_none_or(|gpu| gpu.counts != settings.counts || gpu.revision != revision)
        {
            self.gpu = Some(DdgiGpu::new(ctx.gpu, settings.counts, revision));
            self.schedule.reset(settings.counts);
        }
        let base: [i32; 3] = std::array::from_fn(|i| {
            (eye[i] / settings.spacing).floor() as i32 - (settings.counts[i] / 2) as i32
        });
        let recycled = self.schedule.scroll(base);
        let budget = ((probes as f32 * settings.update_fraction).ceil() as u32).clamp(1, probes);
        let mut jobs = Vec::with_capacity(budget as usize);
        self.schedule.take(budget, &mut jobs);

        let mesh = self.mesh.borrow();
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
        ctx.lighting
            .set_spherical_gi(Some((gpu.probes.clone(), gpu.probes.clone())));
        self.published = true;
        if !traceable || jobs.is_empty() {
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
        self.staged = Some(Staged { jobs, recycled });
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
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gi.ddgi.update"),
                ..Default::default()
            });
        pass.set_pipeline(&gpu.update);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(staged.jobs.len() as u32, 1, 1);
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
            update_frames: previous.update_frames + 1,
            recycled: previous.recycled + u64::from(staged.recycled),
            base: self.schedule.base.unwrap_or_default(),
            gpu_bytes,
        });
    }
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
    update: wgpu::ComputePipeline,
    probes: Buffer,
    params: Buffer,
    options: Buffer,
    jobs: Buffer,
    header: Vec<u8>,
}

impl DdgiGpu {
    fn new(gpu: &GpuContext, counts: [u32; 3], revision: u64) -> Self {
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
        let rows = layout(counts.iter().product()).rows;
        Self {
            counts,
            revision,
            update,
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
    fn scroll_recycles_only_the_planes_that_left() {
        let mut schedule = Schedule::default();
        schedule.reset([4, 2, 4]);
        assert_eq!(schedule.scroll([0, 0, 0]), 0);
        // One cell along +x recycles one 2x4 plane.
        assert_eq!(schedule.scroll([1, 0, 0]), 8);
        let mut jobs = Vec::new();
        schedule.take(8, &mut jobs);
        let expected: Vec<u32> = (0..2)
            .flat_map(|y| (0..4).map(move |z| (y, z)))
            .map(|(y, z)| schedule.slot([4, y, z]))
            .collect();
        let mut sorted = jobs.clone();
        sorted.sort_unstable();
        let mut want = expected.clone();
        want.sort_unstable();
        assert_eq!(sorted, want);
        // The round robin then continues over every slot.
        schedule.take(32, &mut jobs);
        assert_eq!(jobs.len(), 32);
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
