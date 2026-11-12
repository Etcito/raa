//! The world-space cloud dome: a persistent 2D texture of the clouds indexed by WORLD view
//! direction, so a moving or rotating camera never re-marches them (setting `cloudDome`).
//!
//! Direction mapping: a hemi-octahedral map of the upper hemisphere (no pole singularity,
//! every texel an equal-ish solid angle), extended below the horizon by shifting the
//! direction: `u = normalize(d + (0, DOME_BELOW, 0))` is encoded with the hemi-octahedral
//! map, so the dome covers every direction with `d.y >= -DOME_BELOW` (about -2.9 degrees,
//! the lowest direction the clouds can be seen at: the panorama stops at -0.02). The
//! texel centre `(i + 0.5) / size` decodes to the direction it marches; the consumer
//! samples with a bilinear clamping sampler (`raa_physical_sky_dome_uv` in `shaders.rs` and
//! `raa_cloud_dome_direction` in `wgsl/dome.wgsl` are the WGSL twins of
//! [`dome_uv`]/[`dome_direction`]).
//!
//! The dome is per sky (shared by views): one march per invocation, from the first
//! marchable view that claims it, from the viewer height of that view and the horizontal
//! origin (0, 0) (parallax is ignored, so a camera translation never moves the clouds).
//! It accumulates by count in its own history: 1/n running mean while the sky stands
//! still, then it FREEZES (no dispatch) after [`FREEZE_FRAMES`] unchanged frames; an
//! animated sky blends at max(1/n, 1/16) and never freezes.

use raa_core::{GpuContext, RaaError};

use super::history::{BURST_FRAMES, BURST_STRIDE, FREEZE_FRAMES, Marked};
use super::resources::CloudGpu;
use crate::atmosphere::PhysicalSkyGpu;

/// How far below the horizon (as a direction y) the dome reaches.
pub(super) const DOME_BELOW: f32 = 0.05;
/// Largest count kept.
const MAX_COUNT: u32 = 1 << 16;

/// The world direction a dome uv in `[0, 1]^2` marches.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn dome_direction(uv: [f32; 2]) -> [f32; 3] {
    let e = [uv[0] * 2.0 - 1.0, uv[1] * 2.0 - 1.0];
    let p = [(e[0] + e[1]) * 0.5, (e[0] - e[1]) * 0.5];
    let y = (1.0 - p[0].abs() - p[1].abs()).max(0.0);
    let length = (p[0] * p[0] + y * y + p[1] * p[1]).sqrt().max(1.0e-12);
    let u = [p[0] / length, y / length, p[1] / length];
    let b = DOME_BELOW;
    let t = b * u[1] + (b * b * u[1] * u[1] - b * b + 1.0).max(0.0).sqrt();
    let d = [u[0] * t, u[1] * t - b, u[2] * t];
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1.0e-12);
    [d[0] / n, d[1] / n, d[2] / n]
}

/// The dome uv of a unit world direction; directions below the dome clamp to its lowest ring.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn dome_uv(d: [f32; 3]) -> [f32; 2] {
    let v = [d[0], (d[1] + DOME_BELOW).max(0.0), d[2]];
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let u = if length > 1.0e-6 {
        [v[0] / length, v[1] / length, v[2] / length]
    } else {
        [1.0, 0.0, 0.0]
    };
    let sum = (u[0].abs() + u[1] + u[2].abs()).max(1.0e-6);
    let p = [u[0] / sum, u[2] / sum];
    [(p[0] + p[1]) * 0.5 + 0.5, (p[0] - p[1]) * 0.5 + 0.5]
}

/// What one invocation does with the dome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DomePlan {
    /// The view that marches it.
    pub(super) view: u32,
    /// Frames accumulated after this invocation: the committed count + 1, or after a change
    /// the `burst` marches of its convergence burst.
    pub(super) count: u32,
    /// Marches recorded in this invocation: [`BURST_FRAMES`] after a change (dither on), else 1.
    pub(super) burst: u32,
    /// Signature of the sky this march is made for.
    pub(super) signature: u64,
    /// Nothing is dispatched: the sky stood still for [`FREEZE_FRAMES`] frames.
    pub(super) frozen: bool,
}

/// Device-free lifecycle of the dome: the committed march, the claim of the prepared
/// invocation and the note of the recorded one.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct DomeState {
    committed: Option<Marked>,
    prepared: Option<(u64, DomePlan)>,
    noted: Option<(u64, DomePlan)>,
    last_frozen: bool,
    /// Invocations that ran a convergence burst.
    burst_frames: u32,
}

impl DomeState {
    /// Forgets the committed march (a new texture, or the sky model/weather/enable changed).
    pub(super) fn reset(&mut self) {
        let burst_frames = self.burst_frames;
        *self = DomeState {
            burst_frames,
            ..DomeState::default()
        };
    }

    /// Claims the dome of `invocation` for `view` unless another view already did; the plan of
    /// the claiming view (the same plan again when it prepares twice).
    pub(super) fn claim(
        &mut self,
        invocation: u64,
        view: u32,
        signature: u64,
        animated: bool,
        dither: bool,
    ) -> Option<DomePlan> {
        if let Some((claimed, plan)) = self.prepared
            && claimed == invocation
        {
            return (plan.view == view).then_some(plan);
        }
        let (count, burst) = match self.committed {
            Some(m) if m.signature == signature => ((m.count + 1).min(MAX_COUNT), 1),
            _ if dither => (BURST_FRAMES, BURST_FRAMES),
            _ => (1, 1),
        };
        let plan = DomePlan {
            view,
            count,
            burst,
            signature,
            frozen: count >= FREEZE_FRAMES && !animated,
        };
        self.prepared = Some((invocation, plan));
        Some(plan)
    }

    /// The plan `view` records at `invocation`, once; `None` for another view or a repeat.
    pub(super) fn record(&mut self, invocation: u64, view: u32) -> Option<DomePlan> {
        let (claimed, plan) = self.prepared?;
        if claimed != invocation || plan.view != view {
            return None;
        }
        if self.noted.is_some_and(|(noted, _)| noted == invocation) {
            return None;
        }
        self.noted = Some((invocation, plan));
        self.last_frozen = plan.frozen;
        if plan.burst > 1 {
            self.burst_frames = self.burst_frames.saturating_add(1);
        }
        Some(plan)
    }

    /// Commits the noted march of an acknowledged submission of `view` at `invocation`.
    pub(super) fn submitted(&mut self, view: u32, invocation: u64) {
        if let Some((noted, plan)) = self.noted
            && noted == invocation
            && plan.view == view
        {
            self.committed = Some(Marked {
                count: plan.count,
                signature: plan.signature,
            });
            self.noted = None;
        }
    }

    /// Whether a march was committed: the texture may be sampled, a stale one until the march
    /// of a changed signature (made in the same frame) replaces it.
    pub(super) fn has_committed(&self) -> bool {
        self.committed.is_some()
    }

    /// `(committed count, the last recorded frame did not march, burst frames so far)`.
    pub(super) fn observation(&self) -> (u32, bool, u32) {
        (
            self.committed.map_or(0, |m| m.count),
            self.last_frozen,
            self.burst_frames,
        )
    }
}

/// The dome's uniform record of burst march `slot` (0 for a single march): the view's cloud
/// record with the camera lanes unused, the accumulation count in `forward.w` (1, 2, ... over
/// a burst, ending at `plan.count`) and the screen lane (scale 1, flags, jitter, samples).
pub(super) fn dome_record(
    frame: &super::super::CloudFrame,
    plan: &DomePlan,
    slot: u32,
) -> [u8; CloudGpu::SIZE] {
    let count = plan.count + 1 + slot - plan.burst;
    let mut record = CloudGpu::from_frame(frame, None, false);
    // Nothing of the camera: not its basis, the snapped shadow corner or the history matrix.
    record.right = [0.0; 4];
    record.up = [0.0; 4];
    record.shadow_origin = [0.0; 2];
    record.forward = [0.0, 0.0, -1.0, count as f32];
    record.screen = [
        1.0,
        super::resources::pack_screen_flags(
            false,
            frame.animated,
            frame.cloud_dither,
            frame.cloud_noise_size,
        ),
        // Still frames walk the golden-ratio sequence by their count, the first one the invocation.
        if count >= 2 {
            (count & 255) as f32
        } else {
            (frame.invocation & 255) as f32
        },
        frame.cloud_samples as f32,
    ];
    record.to_bytes()
}

/// The dome's sky record: the frame's, from the viewer height over the horizontal origin.
pub(super) fn dome_sky(frame: &super::super::CloudFrame) -> [u8; PhysicalSkyGpu::SIZE] {
    let mut sky = super::resources::cloud_sky_parameters(frame);
    sky.viewer[0] = 0.0;
    sky.viewer[2] = 0.0;
    sky.to_bytes()
}

/// The dome textures and uniforms: `current` is written by the march and handed out, then
/// copied into `history`, which the next march reads.
pub(super) struct DomeGpu {
    pub(super) size: u32,
    pub(super) current: wgpu::TextureView,
    pub(super) history: wgpu::TextureView,
    pub(super) params: wgpu::Buffer,
    pub(super) sky: wgpu::Buffer,
}

impl DomeGpu {
    /// Creates a `size` x `size` dome; its history holds alpha -1 (not marched) everywhere.
    pub(super) fn new(gpu: &GpuContext, size: u32) -> Result<DomeGpu, RaaError> {
        gpu.ensure_live()?;
        let extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        };
        let create = |label: &str, usage: wgpu::TextureUsages| {
            gpu.device().create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage,
                view_formats: &[],
            })
        };
        use wgpu::TextureUsages as U;
        let current = create(
            "raa-sky.cloud_dome",
            U::STORAGE_BINDING | U::TEXTURE_BINDING | U::COPY_SRC,
        );
        let history = create("raa-sky.cloud_dome_history", U::TEXTURE_BINDING | U::COPY_DST);
        // f16 -1.0 is 0xBC00: (0, 0, 0, -1) marks every texel as not marched.
        let unknown = [0u8, 0, 0, 0, 0, 0, 0x00, 0xBC].repeat((size * size) as usize);
        gpu.queue().write_texture(
            history.as_image_copy(),
            &unknown,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 8),
                rows_per_image: Some(size),
            },
            extent,
        );
        let uniform = |label: &str, bytes: usize| {
            gpu.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Ok(DomeGpu {
            size,
            current: current.create_view(&Default::default()),
            history: history.create_view(&Default::default()),
            params: uniform(
                "raa-sky.cloud_dome_params",
                (BURST_STRIDE * u64::from(BURST_FRAMES)) as usize,
            ),
            sky: uniform("raa-sky.cloud_dome_sky", PhysicalSkyGpu::SIZE),
        })
    }
}
