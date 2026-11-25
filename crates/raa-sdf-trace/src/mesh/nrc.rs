//! Neural radiance caching (Müller, Rousselle, Novák, Keller 2021) on the shared world-space
//! mesh BVH.
//!
//! A small fully fused MLP (64 wide, FP32, no bias) learns R(x, n) = indirect diffuse irradiance
//! / pi online: each frame short path suffixes are traced from random visible points, terminated
//! into the cache itself (self-training) and used for one or more Adam steps, all in compute
//! shaders. The frame then queries the cache at reduced resolution (directly, or after one
//! cosine bounce), upsamples and accumulates the result and publishes it through the forward
//! GI screen slot (group 1 binding 14) while `RAA_NRC_GI_FEATURE` is set.
//!
//! Weights, their moving average and the Adam moments serialize to a checkpoint and load back
//! (warm start from a pre-trained cache instead of the random initialization).

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc::{Receiver, sync_channel};

use raa_core::{GpuContext, RaaError};
use raa_pipeline::{
    FeaturePass, HDR_FORMAT, ImageDesc, ImageDim, LightingResources, Pipeline, PrepareCtx,
    RecordCtx,
};
use wgpu::{Buffer, BufferUsages as Usage};

use super::MeshTraceHandle;
use super::pass::{buffer, buffer_entry, image_entry};

pub const NRC_GI_NAME: &str = "gi.nrc";
/// Frame feature bit read by `opaque_lighting.wgsl` (raa-shader `RAA_NRC_GI_FEATURE`).
pub const NRC_FRAME_FEATURE: u32 = raa_shader::RAA_NRC_GI_FEATURE;
const ORDER: u32 = 56;
const GI_SCREEN_SLOT: u32 = 14;
const WIDTH: u32 = 64;
const MAX_HIDDEN: u32 = 5;
const MAX_BATCH: u32 = 16_384;
const MIN_SAMPLES: u32 = 1024;
const MAX_STEPS: u32 = 4;
const CHUNK: u32 = 256;
const MAX_FREQUENCIES: u32 = 12;
const CHECKPOINT_MAGIC: u32 = 0x3143_524e; // "NRC1"
const CHECKPOINT_VERSION: u32 = 1;
const CHECKPOINT_HEADER: usize = 32;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NrcSettings {
    pub enabled: bool,
    /// Hidden layers of 64 neurons (the paper uses 5).
    pub hidden_layers: u32,
    /// Triangle-wave octaves of the position encoding per axis. The paper uses 12; queried at
    /// the visible point after a few seconds of training, 12 shows coloured high-frequency
    /// speckle on dark Sponza walls, 6 does not.
    pub frequencies: u32,
    /// Training samples per frame, a power of two; above 16 384 several Adam steps run.
    pub train_samples: u32,
    pub learning_rate: f32,
    /// Weight of the previous average in the exponential moving average of the weights.
    pub ema: f32,
    /// Epsilon of the relative loss (the paper's 0.01). The loss weights come from the
    /// prediction, so a small epsilon under noisy one-sample targets biases the cache upward
    /// (measured on Sponza: 1e-4 gave 2.7 times the indirect light of 0.01, 0.1 or 1).
    pub loss_epsilon: f32,
    /// Online training on or off (off renders a frozen cache, e.g. a loaded checkpoint).
    pub train: bool,
    /// Alternate frames between tracing the training rays and the Adam step (single-step
    /// budgets only): halves the average training cost.
    pub interleave: bool,
    /// Resolution divisor of the cache queries: 1, 2, 4 or 8.
    pub divisor: u32,
    /// 0 queries the cache at the visible point; 1 traces one cosine bounce first.
    pub bounces: u32,
    /// Weight of the current frame in the temporal accumulation (1 disables it).
    pub temporal_blend: f32,
}

impl Default for NrcSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            hidden_layers: 3,
            frequencies: 6,
            train_samples: 16_384,
            learning_rate: 2e-3,
            ema: 0.99,
            loss_epsilon: 0.01,
            train: true,
            interleave: true,
            divisor: 4,
            bounces: 0,
            temporal_blend: 0.2,
        }
    }
}

impl NrcSettings {
    pub fn validate(self) -> Result<Self, RaaError> {
        let ok = (1..=MAX_HIDDEN).contains(&self.hidden_layers)
            && (1..=MAX_FREQUENCIES).contains(&self.frequencies)
            && self.train_samples.is_power_of_two()
            && (MIN_SAMPLES..=MAX_BATCH * MAX_STEPS).contains(&self.train_samples)
            && self.learning_rate.is_finite()
            && (0.0..=0.1).contains(&self.learning_rate)
            && (0.0..=0.9999).contains(&self.ema)
            && (1e-8..=1.0).contains(&self.loss_epsilon)
            && matches!(self.divisor, 1 | 2 | 4 | 8)
            && self.bounces <= 1
            && self.temporal_blend > 0.0
            && self.temporal_blend <= 1.0;
        if !ok {
            return Err(RaaError::FrameGraph {
                code: "gi.nrc.invalid_settings",
                message: format!(
                    "expected 1-{MAX_HIDDEN} hidden layers, 1-{MAX_FREQUENCIES} frequencies, a power-of-two sample count in \
                     [{MIN_SAMPLES}, {}], a learning rate in [0, 0.1], an EMA in [0, 0.9999], a loss epsilon in [1e-8, 1], \
                     a divisor of 1, 2, 4 or 8, 0 or 1 bounce and a temporal blend in (0, 1]",
                    MAX_BATCH * MAX_STEPS
                ),
            });
        }
        Ok(self)
    }

    /// Samples of one Adam step and the steps per frame.
    fn batch(&self) -> (u32, u32) {
        (
            self.train_samples.min(MAX_BATCH),
            (self.train_samples / MAX_BATCH).clamp(1, MAX_STEPS),
        )
    }
}

/// vec4 rows of the weight matrices of a network with `hidden` hidden layers.
fn weight_rows(hidden: u32) -> u32 {
    hidden * (WIDTH * WIDTH / 4) + 3 * WIDTH / 4
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NrcStats {
    pub hidden_layers: u32,
    pub weights: u32,
    /// Adam steps since the last reset or checkpoint load (the loaded count included).
    pub steps: u64,
    pub frames: u64,
    pub batch: u32,
    pub steps_per_frame: u32,
    pub query_size: (u32, u32),
    pub gpu_bytes: u64,
}

pub type NrcHandle = Rc<RefCell<NrcState>>;

#[derive(Default)]
pub struct NrcState {
    pub settings: NrcSettings,
    view: Option<usize>,
    pass_enabled: bool,
    reset: bool,
    load: Option<Checkpoint>,
    export: bool,
    exported: Option<Receiver<Option<Vec<u8>>>>,
    stats: Option<NrcStats>,
}

/// A decoded checkpoint: the four weight-shaped arrays and the Adam step count.
#[derive(Clone, Debug, PartialEq)]
pub struct Checkpoint {
    pub hidden_layers: u32,
    pub frequencies: u32,
    pub steps: u32,
    /// Trained weights, their moving average, first and second Adam moments.
    pub arrays: [Vec<f32>; 4],
}

impl Checkpoint {
    /// Xavier-uniform initial weights from a fixed seed, zero moments.
    pub fn initial(hidden: u32, frequencies: u32) -> Self {
        let count = (weight_rows(hidden) * 4) as usize;
        let mut state = 0x1234_5678_9abc_def1u64;
        let weights: Vec<f32> = (0..count)
            .map(|i| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                let unit =
                    (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 24) as f32;
                let fan_out = if i >= (hidden * WIDTH * WIDTH) as usize {
                    3
                } else {
                    WIDTH
                };
                (unit * 2.0 - 1.0) * (6.0 / (WIDTH + fan_out) as f32).sqrt()
            })
            .collect();
        Self {
            hidden_layers: hidden,
            frequencies,
            steps: 0,
            arrays: [weights.clone(), weights, vec![0.0; count], vec![0.0; count]],
        }
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, RaaError> {
        let error = |message: &str| RaaError::FrameGraph {
            code: "gi.nrc.checkpoint",
            message: message.into(),
        };
        if bytes.len() < CHECKPOINT_HEADER {
            return Err(error("The radiance cache checkpoint is truncated."));
        }
        let word = |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().expect("word"));
        if word(0) != CHECKPOINT_MAGIC || word(1) != CHECKPOINT_VERSION {
            return Err(error("Not a version 1 radiance cache checkpoint."));
        }
        let hidden = word(3);
        if word(2) != WIDTH
            || !(1..=MAX_HIDDEN).contains(&hidden)
            || !(1..=MAX_FREQUENCIES).contains(&word(4))
        {
            return Err(error(
                "The checkpoint network shape is not supported by this engine.",
            ));
        }
        let count = (weight_rows(hidden) * 4) as usize;
        if bytes.len() != CHECKPOINT_HEADER + 16 * count {
            return Err(error("The radiance cache checkpoint has the wrong size."));
        }
        let mut arrays: [Vec<f32>; 4] = Default::default();
        for (k, array) in arrays.iter_mut().enumerate() {
            let start = CHECKPOINT_HEADER + k * 4 * count;
            *array = bytes[start..start + 4 * count]
                .chunks_exact(4)
                .map(|lane| f32::from_le_bytes(lane.try_into().expect("f32")))
                .collect();
            if array.iter().any(|value| !value.is_finite()) {
                return Err(error(
                    "The radiance cache checkpoint holds non-finite values.",
                ));
            }
        }
        Ok(Self {
            hidden_layers: hidden,
            frequencies: word(4),
            steps: word(5),
            arrays,
        })
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(CHECKPOINT_HEADER + 16 * self.arrays[0].len());
        for word in [
            CHECKPOINT_MAGIC,
            CHECKPOINT_VERSION,
            WIDTH,
            self.hidden_layers,
            self.frequencies,
            self.steps,
            0,
            0,
        ] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        for array in &self.arrays {
            for value in array {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }
}

impl NrcState {
    pub fn configure(&mut self, settings: NrcSettings) -> Result<(), RaaError> {
        self.settings = settings.validate()?;
        Ok(())
    }

    pub fn active(&self) -> bool {
        self.pass_enabled && self.settings.enabled
    }

    pub fn stats(&self) -> Option<NrcStats> {
        self.stats
    }

    /// The bridge names the view being prepared; the cache serves view 0.
    pub fn set_view(&mut self, index: usize) {
        self.view = Some(index);
    }

    /// Discards the learned cache and restarts from the random initialization.
    pub fn reset(&mut self) {
        self.reset = true;
        self.load = None;
    }

    /// Warm start: replaces weights and optimizer state at the next frame. The checkpoint must
    /// have the node's hidden layer count.
    pub fn load_checkpoint(&mut self, bytes: &[u8]) -> Result<(), RaaError> {
        let checkpoint = Checkpoint::decode(bytes)?;
        if checkpoint.hidden_layers != self.settings.hidden_layers
            || checkpoint.frequencies != self.settings.frequencies
        {
            return Err(RaaError::FrameGraph {
                code: "gi.nrc.checkpoint",
                message: format!(
                    "The checkpoint has {} hidden layers and {} frequencies, the node {} and {}.",
                    checkpoint.hidden_layers,
                    checkpoint.frequencies,
                    self.settings.hidden_layers,
                    self.settings.frequencies
                ),
            });
        }
        self.load = Some(checkpoint);
        self.reset = false;
        Ok(())
    }

    /// Asks for a checkpoint of the cache as of the next rendered frame.
    pub fn request_checkpoint(&mut self) {
        self.export = true;
        self.exported = None;
    }

    /// The requested checkpoint once the GPU readback has completed.
    pub fn take_checkpoint(&mut self) -> Option<Vec<u8>> {
        let bytes = self.exported.as_ref()?.try_recv().ok()??;
        self.exported = None;
        Some(bytes)
    }
}

pub struct NrcFeature;

impl NrcFeature {
    pub fn install(pipeline: &mut Pipeline, mesh: MeshTraceHandle) -> Result<NrcHandle, RaaError> {
        let state = NrcHandle::default();
        pipeline.add(Box::new(NrcPass {
            state: state.clone(),
            mesh,
            gpu: None,
            images: None,
            staged: None,
            published: false,
            steps: 0,
            frames: 0,
        }))?;
        pipeline.set_enabled(NRC_GI_NAME, false)?;
        Ok(state)
    }
}

struct Staged {
    /// Training steps whose rays are traced this frame.
    trace_steps: u32,
    /// Adam steps applied this frame.
    train_steps: u32,
    /// Origin buffer the first training step reads; steps alternate.
    origin: usize,
    render: bool,
    write: usize,
    export: Option<u32>,
}

struct NrcPass {
    state: NrcHandle,
    mesh: MeshTraceHandle,
    gpu: Option<NrcGpu>,
    images: Option<NrcImages>,
    staged: Option<Staged>,
    published: bool,
    steps: u64,
    frames: u64,
}

impl NrcPass {
    fn unpublish(&mut self, lighting: &mut LightingResources) {
        if std::mem::take(&mut self.published) {
            lighting.clear_gi();
        }
    }
}

fn push_u32(dst: &mut Vec<u8>, words: [u32; 4]) {
    for word in words {
        dst.extend_from_slice(&word.to_le_bytes());
    }
}

fn push_f32(dst: &mut Vec<u8>, words: [f32; 4]) {
    for word in words {
        dst.extend_from_slice(&word.to_le_bytes());
    }
}

impl FeaturePass for NrcPass {
    fn name(&self) -> &str {
        NRC_GI_NAME
    }

    fn order(&self) -> u32 {
        ORDER
    }

    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        self.mesh.borrow_mut().set_nrc_active(enabled);
        self.state.borrow_mut().pass_enabled = enabled;
        if !enabled {
            self.unpublish(lighting);
            self.images = None;
            self.staged = None;
        }
    }

    fn invalidate_history(&mut self) {
        if let Some(images) = &mut self.images {
            images.history_frame = None;
        }
    }

    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        self.staged = None;
        let (settings, view, reset, load, export) = {
            let mut state = self.state.borrow_mut();
            if !state.active() {
                drop(state);
                self.unpublish(ctx.lighting);
                return Ok(());
            }
            (
                state.settings,
                state.view,
                std::mem::take(&mut state.reset),
                state.load.take(),
                state.export,
            )
        };
        if view != Some(0) {
            self.unpublish(ctx.lighting);
            return Ok(());
        }
        let (batch, steps_per_frame) = settings.batch();
        let rebuilt = self
            .gpu
            .as_ref()
            .is_none_or(|gpu| !gpu.shaped(&settings) || gpu.batch != batch);
        if rebuilt {
            // A new batch size keeps the learned weights; a new depth cannot.
            let kept = self.gpu.take().filter(|gpu| gpu.shaped(&settings));
            let mut gpu = NrcGpu::new(ctx.gpu, settings.hidden_layers, settings.frequencies, batch);
            match kept {
                Some(old) => gpu.adopt(&old),
                None => {
                    gpu.upload(
                        ctx.gpu,
                        &Checkpoint::initial(settings.hidden_layers, settings.frequencies),
                    );
                    self.steps = 0;
                }
            }
            self.gpu = Some(gpu);
        }
        let gpu = self.gpu.as_mut().expect("radiance cache buffers");
        if let Some(checkpoint) = load {
            if checkpoint.hidden_layers == gpu.hidden && checkpoint.frequencies == gpu.frequencies {
                gpu.upload(ctx.gpu, &checkpoint);
                self.steps = u64::from(checkpoint.steps);
            }
        } else if reset {
            gpu.upload(ctx.gpu, &Checkpoint::initial(gpu.hidden, gpu.frequencies));
            self.steps = 0;
        }

        let divisor = settings.divisor;
        let low = (ctx.size.0.div_ceil(divisor), ctx.size.1.div_ceil(divisor));
        if self
            .images
            .as_ref()
            .is_none_or(|images| images.size != ctx.size || images.low != low)
        {
            self.images = Some(NrcImages::new(ctx.gpu.device(), ctx.size, low));
        }
        let images = self.images.as_mut().expect("radiance cache images");
        let write = (ctx.frame & 1) as usize;
        ctx.lighting
            .set_slot_view(GI_SCREEN_SLOT, images.history[write].clone())?;
        self.published = true;

        let mesh = self.mesh.borrow();
        let triangles = mesh.geometry.count();
        let traceable = triangles > 0 && mesh.frame.is_some() && mesh.texture_atlas.is_some();
        let sky_diffuse = mesh.sky_diffuse;
        drop(mesh);
        if !traceable {
            return Ok(());
        }
        // Interleaved training traces on even frames and learns from those samples on odd ones.
        let interleaved = settings.interleave && steps_per_frame == 1;
        let (trace_steps, train_steps) = match (settings.train, interleaved) {
            (false, _) => (0, 0),
            (true, false) => (steps_per_frame, steps_per_frame),
            (true, true) => ((1 - (ctx.frame & 1)) as u32, (ctx.frame & 1) as u32),
        };
        let queue = ctx.gpu.queue();
        let env = ctx.lighting.environment_diffuse_sh().is_some();
        let limit = 1.0e4f32;
        let mut options = Vec::with_capacity(64);
        push_u32(&mut options, [triangles, 0, u32::from(env), 0]);
        push_u32(&mut options, [0; 4]);
        push_u32(&mut options, [sky_diffuse.to_bits(), limit.to_bits(), 0, 0]);
        // The shared local-light integrator uses its one-tap area estimator for cache rays.
        push_u32(&mut options, [1, 0, 0, 0]);
        queue.write_buffer(&gpu.options, 0, &options);
        let seed = (ctx.frame as u32).wrapping_mul(MAX_STEPS + 1);
        for step in 0..=trace_steps.max(train_steps).min(MAX_STEPS) {
            // Slot 0 drives the render pass, slots 1.. the training steps.
            let mut params = Vec::with_capacity(48);
            push_u32(
                &mut params,
                [batch, seed.wrapping_add(step), settings.bounces, 4],
            );
            push_f32(&mut params, [sky_diffuse, limit, 64.0, 0.0]);
            push_u32(&mut params, [low.0, low.1, divisor, 0]);
            queue.write_buffer(&gpu.params[step as usize], 0, &params);
            if step == 0 {
                continue;
            }
            let t = (self.steps + u64::from(step)) as f32;
            let mut train = Vec::with_capacity(32);
            push_f32(&mut train, [settings.learning_rate, BETA1, BETA2, 1e-8]);
            push_f32(
                &mut train,
                [
                    1.0 / (1.0 - BETA1.powf(t)),
                    1.0 / (1.0 - BETA2.powf(t)).max(1e-6),
                    settings.ema.min(1.0 - 1.0 / t),
                    settings.loss_epsilon,
                ],
            );
            queue.write_buffer(&gpu.train[step as usize - 1], 0, &train);
        }
        let history_valid = images.history_frame.is_some_and(|f| f + 1 == ctx.frame);
        let mut resolve = Vec::with_capacity(32);
        push_u32(&mut resolve, [divisor, u32::from(history_valid), 0, 0]);
        push_f32(&mut resolve, [settings.temporal_blend, 1.0, 0.0, 0.0]);
        queue.write_buffer(&gpu.resolve_params, 0, &resolve);
        self.staged = Some(Staged {
            trace_steps,
            train_steps,
            origin: (self.steps & 1) as usize,
            render: true,
            write,
            export: export.then_some((self.steps + u64::from(train_steps)) as u32),
        });
        Ok(())
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let Some(staged) = &self.staged else {
            return Ok(());
        };
        let gpu = self.gpu.as_ref().expect("prepared radiance cache");
        let images = self
            .images
            .as_mut()
            .expect("prepared radiance cache images");
        images.history_frame = Some(ctx.frame);
        let mut mesh = self.mesh.borrow_mut();
        let (Some(frame), Some(atlas)) = (mesh.frame.clone(), mesh.texture_atlas.clone()) else {
            return Ok(());
        };
        let geometry = mesh.geometry_buffers(ctx.gpu);
        drop(mesh);
        let device = ctx.gpu.device();
        let depth = ctx.targets.depth_input(ctx.resources, NRC_GI_NAME);
        let normal = ctx.targets.normal_input(ctx.resources, NRC_GI_NAME);
        let sampler = wgpu::BindGroupEntry {
            binding: 15,
            resource: wgpu::BindingResource::Sampler(ctx.lighting.linear_clamp()),
        };
        let scene = vec![
            buffer_entry(0, &frame),
            buffer_entry(1, &gpu.options),
            buffer_entry(2, &geometry[0]),
            buffer_entry(3, &geometry[1]),
            buffer_entry(4, &geometry[2]),
            buffer_entry(13, ctx.lighting.local_lights_buffer()),
            image_entry(14, ctx.lighting.env_specular_view()),
            sampler,
            image_entry(crate::wgsl::MESH_TRACE_ATLAS_BINDING, &atlas),
            buffer_entry(31, &gpu.average),
        ];
        let compute = |encoder: &mut wgpu::CommandEncoder,
                       label: &'static str,
                       pipeline: &wgpu::ComputePipeline,
                       group: &wgpu::BindGroup,
                       groups: (u32, u32)| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        };

        // 1. Online training: paths with a self-trained tail, then Adam steps.
        let layers = gpu.hidden + 1;
        for step in 0..staged.trace_steps.max(staged.train_steps) as usize {
            if step < staged.trace_steps as usize {
                let mut entries = scene.clone();
                entries.push(buffer_entry(30, &gpu.params[step + 1]));
                entries.push(buffer_entry(32, &gpu.samples));
                entries.push(image_entry(6, &depth));
                entries.push(image_entry(7, &normal));
                let origin = (staged.origin + step) & 1;
                entries.push(buffer_entry(34, &gpu.origins[origin]));
                entries.push(buffer_entry(35, &gpu.origins[1 - origin]));
                let paths = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gi.nrc.paths"),
                    layout: &gpu.paths.get_bind_group_layout(0),
                    entries: &entries,
                });
                compute(
                    ctx.encoder,
                    "gi.nrc.paths",
                    &gpu.paths,
                    &paths,
                    (gpu.batch.div_ceil(64), 1),
                );
            }
            if step >= staged.train_steps as usize {
                continue;
            }
            let learn = |pipeline: &wgpu::ComputePipeline, partial: bool| {
                let mut entries = vec![
                    buffer_entry(0, &gpu.train[step]),
                    buffer_entry(1, &gpu.trained),
                    buffer_entry(2, &gpu.samples),
                    buffer_entry(3, &gpu.acts),
                    buffer_entry(4, &gpu.deltas),
                ];
                if partial {
                    entries = vec![
                        buffer_entry(3, &gpu.acts),
                        buffer_entry(4, &gpu.deltas),
                        buffer_entry(5, &gpu.partial),
                    ];
                }
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gi.nrc.learn"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &entries,
                })
            };
            compute(
                ctx.encoder,
                "gi.nrc.forward-backward",
                &gpu.forward_backward,
                &learn(&gpu.forward_backward, false),
                (gpu.batch.div_ceil(64), 1),
            );
            compute(
                ctx.encoder,
                "gi.nrc.gradient",
                &gpu.gradient,
                &learn(&gpu.gradient, true),
                ((gpu.batch / CHUNK * layers * 16).div_ceil(64), 1),
            );
            let adam = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gi.nrc.adam"),
                layout: &gpu.adam.get_bind_group_layout(0),
                entries: &[
                    buffer_entry(0, &gpu.train[step]),
                    buffer_entry(1, &gpu.trained),
                    buffer_entry(2, &gpu.average),
                    buffer_entry(3, &gpu.moment1),
                    buffer_entry(4, &gpu.moment2),
                    buffer_entry(5, &gpu.partial),
                ],
            });
            compute(
                ctx.encoder,
                "gi.nrc.adam",
                &gpu.adam,
                &adam,
                (weight_rows(gpu.hidden).div_ceil(64), 1),
            );
        }

        // 2. The cache for this frame at reduced resolution, then upsampling and accumulation.
        if staged.render {
            let mut entries = scene.clone();
            entries.push(buffer_entry(30, &gpu.params[0]));
            entries.push(image_entry(6, &depth));
            entries.push(image_entry(7, &normal));
            entries.push(image_entry(33, &images.noisy));
            let render = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gi.nrc.render"),
                layout: &gpu.render.get_bind_group_layout(0),
                entries: &entries,
            });
            compute(
                ctx.encoder,
                "gi.nrc.render",
                &gpu.render,
                &render,
                (images.low.0.div_ceil(8), images.low.1.div_ceil(8)),
            );
            let resolve = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gi.nrc.resolve"),
                layout: &gpu.resolve.get_bind_group_layout(0),
                entries: &[
                    buffer_entry(0, &frame),
                    buffer_entry(1, &gpu.resolve_params),
                    image_entry(2, &depth),
                    image_entry(3, &normal),
                    image_entry(4, &images.noisy),
                    image_entry(5, &images.history[1 - staged.write]),
                    image_entry(6, &images.history[staged.write]),
                ],
            });
            compute(
                ctx.encoder,
                "gi.nrc.resolve",
                &gpu.resolve,
                &resolve,
                (ctx.size.0.div_ceil(8), ctx.size.1.div_ceil(8)),
            );
            // A denoiser node reads the raw signal; `gi` is what forward shading samples.
            ctx.resources.publish(
                NRC_GI_NAME,
                "noisy",
                &images.noisy,
                ImageDesc::new(ImageDim::D2, HDR_FORMAT, images.low.0, images.low.1, 1, 1),
            );
            ctx.resources.publish(
                NRC_GI_NAME,
                "gi",
                &images.history[staged.write],
                ImageDesc::new(ImageDim::D2, HDR_FORMAT, ctx.size.0, ctx.size.1, 1, 1),
            );
        }

        // 3. Checkpoint readback of the weights as trained by this frame.
        if let Some(steps) = staged.export {
            let bytes = u64::from(weight_rows(gpu.hidden)) * 16;
            let readback = buffer(
                device,
                "gi.nrc.checkpoint",
                4 * bytes,
                Usage::MAP_READ | Usage::COPY_DST,
            );
            for (k, source) in [&gpu.trained, &gpu.average, &gpu.moment1, &gpu.moment2]
                .into_iter()
                .enumerate()
            {
                ctx.encoder
                    .copy_buffer_to_buffer(source, 0, &readback, k as u64 * bytes, bytes);
            }
            let (send, receive) = sync_channel(1);
            let hidden = gpu.hidden;
            let frequencies = gpu.frequencies;
            let mapped = readback.clone();
            ctx.encoder.map_buffer_on_submit(
                &readback,
                wgpu::MapMode::Read,
                0..4 * bytes,
                move |result| {
                    let checkpoint = result.ok().and_then(|()| {
                        let data = mapped.slice(0..4 * bytes).get_mapped_range().ok()?;
                        let mut encoded = Checkpoint {
                            hidden_layers: hidden,
                            frequencies,
                            steps,
                            arrays: Default::default(),
                        }
                        .encode();
                        encoded.extend_from_slice(&data);
                        Some(encoded)
                    });
                    mapped.unmap();
                    let _ = send.send(checkpoint);
                },
            );
            let mut state = self.state.borrow_mut();
            state.export = false;
            state.exported = Some(receive);
        }
        Ok(())
    }

    fn submitted(&mut self) {
        let Some(staged) = self.staged.take() else {
            return;
        };
        self.steps += u64::from(staged.train_steps);
        self.frames += 1;
        let (Some(gpu), Some(images)) = (&self.gpu, &self.images) else {
            return;
        };
        let mut state = self.state.borrow_mut();
        let (batch, steps_per_frame) = state.settings.batch();
        state.stats = Some(NrcStats {
            hidden_layers: gpu.hidden,
            weights: weight_rows(gpu.hidden) * 4,
            steps: self.steps,
            frames: self.frames,
            batch,
            steps_per_frame,
            query_size: images.low,
            gpu_bytes: gpu.bytes() + images.bytes(),
        });
    }
}

struct NrcImages {
    size: (u32, u32),
    low: (u32, u32),
    noisy: wgpu::TextureView,
    history: [wgpu::TextureView; 2],
    /// Frame index whose resolve the history holds.
    history_frame: Option<u64>,
}

impl NrcImages {
    fn new(device: &wgpu::Device, size: (u32, u32), low: (u32, u32)) -> Self {
        let image = |label, (width, height): (u32, u32)| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: HDR_FORMAT,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        Self {
            size,
            low,
            noisy: image("gi.nrc.noisy", low),
            history: [image("gi.nrc.gi.a", size), image("gi.nrc.gi.b", size)],
            history_frame: None,
        }
    }

    fn bytes(&self) -> u64 {
        8 * (u64::from(self.low.0) * u64::from(self.low.1)
            + 2 * u64::from(self.size.0) * u64::from(self.size.1))
    }
}

struct NrcGpu {
    hidden: u32,
    frequencies: u32,
    batch: u32,
    paths: wgpu::ComputePipeline,
    render: wgpu::ComputePipeline,
    forward_backward: wgpu::ComputePipeline,
    gradient: wgpu::ComputePipeline,
    adam: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    trained: Buffer,
    average: Buffer,
    moment1: Buffer,
    moment2: Buffer,
    samples: Buffer,
    origins: [Buffer; 2],
    acts: Buffer,
    deltas: Buffer,
    partial: Buffer,
    options: Buffer,
    params: Vec<Buffer>,
    train: Vec<Buffer>,
    resolve_params: Buffer,
}

impl NrcGpu {
    fn shaped(&self, settings: &NrcSettings) -> bool {
        self.hidden == settings.hidden_layers && self.frequencies == settings.frequencies
    }

    fn new(gpu: &GpuContext, hidden: u32, frequencies: u32, batch: u32) -> Self {
        let device = gpu.device();
        let pipeline = |label: &'static str, source: String, entry: &'static str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let rows = u64::from(weight_rows(hidden));
        let layers = u64::from(hidden + 1);
        let storage = |label, bytes: u64| {
            buffer(
                device,
                label,
                bytes,
                Usage::STORAGE | Usage::COPY_DST | Usage::COPY_SRC,
            )
        };
        let uniform = |label, bytes| buffer(device, label, bytes, Usage::UNIFORM | Usage::COPY_DST);
        let batch_bytes = u64::from(batch) * layers * 16 * 16;
        Self {
            hidden,
            frequencies,
            batch,
            paths: pipeline(
                "gi.nrc.paths",
                nrc_mesh_module(hidden, frequencies),
                "nrc_paths",
            ),
            render: pipeline(
                "gi.nrc.render",
                nrc_mesh_module(hidden, frequencies),
                "nrc_render",
            ),
            forward_backward: pipeline(
                "gi.nrc.forward-backward",
                nrc_train_module(hidden, frequencies, batch),
                "nrc_forward_backward",
            ),
            gradient: pipeline(
                "gi.nrc.gradient",
                nrc_train_module(hidden, frequencies, batch),
                "nrc_gradient",
            ),
            adam: pipeline(
                "gi.nrc.adam",
                nrc_adam_module(hidden, frequencies, batch),
                "nrc_adam",
            ),
            resolve: pipeline("gi.nrc.resolve", NRC_RESOLVE_WGSL.into(), "nrc_resolve"),
            trained: storage("gi.nrc.trained", rows * 16),
            average: storage("gi.nrc.average", rows * 16),
            moment1: storage("gi.nrc.moment1", rows * 16),
            moment2: storage("gi.nrc.moment2", rows * 16),
            samples: storage("gi.nrc.samples", u64::from(batch) * 48),
            origins: [
                storage("gi.nrc.origins.a", u64::from(batch) * 16),
                storage("gi.nrc.origins.b", u64::from(batch) * 16),
            ],
            acts: storage("gi.nrc.acts", batch_bytes),
            deltas: storage("gi.nrc.deltas", batch_bytes),
            partial: storage("gi.nrc.partial", u64::from(batch / CHUNK) * rows * 16),
            options: uniform("gi.nrc.options", 64),
            params: (0..=MAX_STEPS)
                .map(|_| uniform("gi.nrc.params", 48))
                .collect(),
            train: (0..MAX_STEPS)
                .map(|_| uniform("gi.nrc.train", 32))
                .collect(),
            resolve_params: uniform("gi.nrc.resolve-params", 32),
        }
    }

    fn weight_buffers(&self) -> [&Buffer; 4] {
        [&self.trained, &self.average, &self.moment1, &self.moment2]
    }

    fn upload(&mut self, gpu: &GpuContext, checkpoint: &Checkpoint) {
        for (target, array) in self.weight_buffers().into_iter().zip(&checkpoint.arrays) {
            let bytes: Vec<u8> = array.iter().flat_map(|value| value.to_le_bytes()).collect();
            gpu.queue().write_buffer(target, 0, &bytes);
        }
    }

    /// Takes over the weight buffers of a cache with the same depth.
    fn adopt(&mut self, old: &NrcGpu) {
        self.trained = old.trained.clone();
        self.average = old.average.clone();
        self.moment1 = old.moment1.clone();
        self.moment2 = old.moment2.clone();
    }

    fn bytes(&self) -> u64 {
        [
            &self.trained,
            &self.average,
            &self.moment1,
            &self.moment2,
            &self.samples,
            &self.acts,
            &self.deltas,
            &self.partial,
        ]
        .iter()
        .map(|buffer| buffer.size())
        .sum()
    }
}

const NRC_MLP_WGSL: &str = include_str!("../wgsl/nrc_mlp.wgsl");
const NRC_RESOLVE_WGSL: &str = include_str!("../wgsl/nrc_resolve.wgsl");
const INCLUDE_MLP: &str = "//!include nrc_mlp";

fn constants(hidden: u32, frequencies: u32, batch: u32) -> String {
    format!(
        "const NRC_HIDDEN: u32 = {hidden}u;\nconst NRC_LAYERS: u32 = {}u;\n\
         const NRC_FREQS: u32 = {frequencies}u;\nconst NRC_W4: u32 = {}u;\n\
         const NRC_BATCH: u32 = {batch}u;\nconst NRC_CHUNK: u32 = {CHUNK}u;\n\
         const NRC_CHUNKS: u32 = {}u;\n",
        hidden + 1,
        weight_rows(hidden),
        batch / CHUNK
    )
}

pub(crate) fn nrc_mesh_module(hidden: u32, frequencies: u32) -> String {
    format!(
        "{}\n{}\n{}",
        crate::wgsl::mesh_trace_module(),
        constants(hidden, frequencies, MAX_BATCH),
        include_str!("../wgsl/mesh_nrc.wgsl").replace(INCLUDE_MLP, NRC_MLP_WGSL),
    )
}

pub(crate) fn nrc_train_module(hidden: u32, frequencies: u32, batch: u32) -> String {
    format!(
        "{}\n{}",
        constants(hidden, frequencies, batch),
        include_str!("../wgsl/nrc_train.wgsl").replace(INCLUDE_MLP, NRC_MLP_WGSL),
    )
}

pub(crate) fn nrc_adam_module(hidden: u32, frequencies: u32, batch: u32) -> String {
    format!(
        "{}\n{}",
        constants(hidden, frequencies, batch),
        include_str!("../wgsl/nrc_adam.wgsl")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(source: &str) {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{error:?}"));
    }

    #[test]
    fn nrc_modules_validate_for_every_depth() {
        for hidden in 1..=MAX_HIDDEN {
            validate(&nrc_mesh_module(hidden, 12));
            validate(&nrc_train_module(hidden, 8, 4096));
            validate(&nrc_adam_module(hidden, 12, MAX_BATCH));
        }
        validate(NRC_RESOLVE_WGSL);
    }

    #[test]
    fn checkpoint_round_trips_and_rejects_bad_input() {
        let mut checkpoint = Checkpoint::initial(3, 12);
        checkpoint.steps = 77;
        checkpoint.arrays[2][5] = 0.25;
        let bytes = checkpoint.encode();
        assert_eq!(bytes.len(), 32 + 16 * 4 * weight_rows(3) as usize);
        assert_eq!(Checkpoint::decode(&bytes).unwrap(), checkpoint);
        assert!(Checkpoint::decode(&bytes[..bytes.len() - 4]).is_err());
        let mut bad = bytes.clone();
        bad[0] ^= 1;
        assert!(Checkpoint::decode(&bad).is_err());
        let mut nan = bytes;
        nan[40..44].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(Checkpoint::decode(&nan).is_err());
    }

    #[test]
    fn initial_weights_are_bounded_and_deterministic() {
        let a = Checkpoint::initial(5, 12);
        assert_eq!(a, Checkpoint::initial(5, 12));
        assert_eq!(a.arrays[0].len(), (5 * 4096 + 192) as usize);
        let bound = (6.0f32 / 67.0).sqrt();
        assert!(a.arrays[0].iter().all(|w| w.abs() <= bound));
        let mean = a.arrays[0].iter().sum::<f32>() / a.arrays[0].len() as f32;
        assert!(mean.abs() < 0.01);
    }

    #[test]
    fn settings_validate_and_split_batches() {
        let d = NrcSettings::default();
        assert!(d.validate().is_ok());
        assert_eq!(d.batch(), (16_384, 1));
        let big = NrcSettings {
            train_samples: 65_536,
            ..d
        };
        assert_eq!(big.validate().unwrap().batch(), (16_384, 4));
        let small = NrcSettings {
            train_samples: 4096,
            ..d
        };
        assert_eq!(small.batch(), (4096, 1));
        for bad in [
            NrcSettings {
                hidden_layers: 0,
                ..d
            },
            NrcSettings {
                train_samples: 5000,
                ..d
            },
            NrcSettings { divisor: 3, ..d },
            NrcSettings { bounces: 2, ..d },
            NrcSettings {
                temporal_blend: 0.0,
                ..d
            },
        ] {
            assert!(bad.validate().is_err());
        }
    }
}
