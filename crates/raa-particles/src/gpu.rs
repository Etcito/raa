//! GPU records of the particle passes: the 256 B `EmitterGpu` record, the 16 B `StepGpu`
//! uniform and the little-endian table writers.
//!
//! Every writer clears and resizes a `Vec<u8>` scratch (so it keeps its capacity after
//! warm-up) and fills it with `to_le_bytes`; no bytemuck, no implicit padding. The step
//! uniforms of one frame are written into slots at the device's
//! `min_uniform_buffer_offset_alignment`, using the stride rule of
//! `raa_fur::shell_slot_stride`.
//!
//! # `EmitterGpu` (256 B)
//!
//! | offset | field |
//! |---|---|
//! | 0   | `position.xyz`, `spread_rad` |
//! | 16  | `direction.xyz` (normalised), `emissive` |
//! | 32  | `lifetime_s.x`, `lifetime_s.y`, `speed.x`, `speed.y` |
//! | 48  | `size.x`, `size.y`, `gravity`, `drag` |
//! | 64  | `volume_half.xyz`, `ground_y` (0 without a ground plane) |
//! | 80  | `kind` (u32), `seed`, `has_ground` (0|1), `color_count` |
//! | 96  | `slot_base`, `max_particles`, 0, 0 |
//! | 112 | `opacity` (splash droplet body), 0, 0, 0 |
//! | 128 | 8 x `vec4` linear colours (rgb, w = 1; unused entries zero) |
//!
//! # `StepGpu` (16 B)
//!
//! | offset | field |
//! |---|---|
//! | 0  | `dt` (s) |
//! | 4  | `sim_time` (s) |
//! | 8  | `step_index` (u32) |
//! | 12 | `emitter_count` (u32) |

#![allow(dead_code)] // consumed by ParticleFeature (particles.sim, particles.opaque, particles.blend)

use crate::desc::{EmitterDesc, ParticleKind};

/// Most emitters the table holds.
pub(crate) const MAX_EMITTERS: usize = 64;
/// Most fixed simulation steps a single frame may run.
pub(crate) const MAX_STEPS_PER_FRAME: usize = 240;
/// Fixed simulation step in seconds.
pub(crate) const STEP_S: f32 = 1.0 / 60.0;
/// Simulated seconds before the first presented frame: the clock starts at
/// `max(0, time_s - PREWARM_S)`.
pub(crate) const PREWARM_S: f32 = 4.0;
/// Most particle slots across all emitters.
pub(crate) const TOTAL_CAPACITY_MAX: u32 = 262_144;

/// Size of one `EmitterGpu` record in bytes.
pub(crate) const EMITTER_GPU_SIZE: usize = 256;
/// Size of one `StepGpu` uniform in bytes.
pub(crate) const STEP_GPU_SIZE: usize = 16;
/// Size of one spawn command (first serial, count) in bytes.
pub(crate) const SPAWN_COMMAND_SIZE: usize = 8;

/// Byte offsets of the `EmitterGpu` fields (see the module table).
pub(crate) const EMITTER_OFF_POSITION: usize = 0;
pub(crate) const EMITTER_OFF_DIRECTION: usize = 16;
pub(crate) const EMITTER_OFF_LIFETIME: usize = 32;
pub(crate) const EMITTER_OFF_SIZE: usize = 48;
pub(crate) const EMITTER_OFF_VOLUME: usize = 64;
pub(crate) const EMITTER_OFF_HEADER: usize = 80;
pub(crate) const EMITTER_OFF_ALLOC: usize = 96;
pub(crate) const EMITTER_OFF_BODY: usize = 112;
pub(crate) const EMITTER_OFF_COLORS: usize = 128;

/// The simulation's u32 for a kind (`EmitterGpu.header.x`).
pub(crate) fn kind_index(kind: ParticleKind) -> u32 {
    match kind {
        ParticleKind::Splash => 0,
        ParticleKind::Snow => 1,
        ParticleKind::Confetti => 2,
        ParticleKind::Sparkles => 3,
    }
}

/// One emitter of the GPU table. Build it with [`EmitterGpu::from_desc`].
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub(crate) struct EmitterGpu {
    /// `position.xyz`, `spread_rad`.
    pub position_spread: [f32; 4],
    /// `direction.xyz` (normalised), `emissive`.
    pub direction_emissive: [f32; 4],
    /// `lifetime_s.x`, `lifetime_s.y`, `speed.x`, `speed.y`.
    pub lifetime_speed: [f32; 4],
    /// `size.x`, `size.y`, `gravity`, `drag`.
    pub size_gravity_drag: [f32; 4],
    /// `volume_half.xyz`, `ground_y`.
    pub volume_ground: [f32; 4],
    /// `kind`, `seed`, `has_ground`, `color_count`.
    pub header: [u32; 4],
    /// `slot_base`, `max_particles`, 0, 0.
    pub alloc: [u32; 4],
    /// `opacity`, 0, 0, 0.
    pub body: [f32; 4],
    /// 8 linear colours; unused entries are zero.
    pub colors: [[f32; 4]; 8],
}

impl EmitterGpu {
    /// Packs `desc` (its direction normalised, zero falling back to +Y) for the slot range
    /// starting at `slot_base`.
    pub(crate) fn from_desc(desc: &EmitterDesc, slot_base: u32) -> EmitterGpu {
        let n = normalize_or(desc.direction, [0.0, 1.0, 0.0]);
        let mut colors = [[0.0; 4]; 8];
        for (out, c) in colors.iter_mut().zip(desc.colors.iter()) {
            *out = [c[0], c[1], c[2], 1.0];
        }
        EmitterGpu {
            position_spread: [
                desc.position[0],
                desc.position[1],
                desc.position[2],
                desc.spread_rad,
            ],
            direction_emissive: [n[0], n[1], n[2], desc.emissive],
            lifetime_speed: [
                desc.lifetime_s[0],
                desc.lifetime_s[1],
                desc.speed[0],
                desc.speed[1],
            ],
            size_gravity_drag: [desc.size[0], desc.size[1], desc.gravity, desc.drag],
            volume_ground: [
                desc.volume_half[0],
                desc.volume_half[1],
                desc.volume_half[2],
                desc.ground_y.unwrap_or(0.0),
            ],
            header: [
                kind_index(desc.kind),
                desc.seed,
                u32::from(desc.ground_y.is_some()),
                desc.colors.len().min(8) as u32,
            ],
            alloc: [slot_base, desc.max_particles, 0, 0],
            body: [desc.opacity, 0.0, 0.0, 0.0],
            colors,
        }
    }

    /// The 256 little-endian bytes of the record.
    pub(crate) fn to_le_bytes(self) -> [u8; EMITTER_GPU_SIZE] {
        let mut out = [0u8; EMITTER_GPU_SIZE];
        write_f32x4(&mut out[EMITTER_OFF_POSITION..], self.position_spread);
        write_f32x4(&mut out[EMITTER_OFF_DIRECTION..], self.direction_emissive);
        write_f32x4(&mut out[EMITTER_OFF_LIFETIME..], self.lifetime_speed);
        write_f32x4(&mut out[EMITTER_OFF_SIZE..], self.size_gravity_drag);
        write_f32x4(&mut out[EMITTER_OFF_VOLUME..], self.volume_ground);
        write_u32x4(&mut out[EMITTER_OFF_HEADER..], self.header);
        write_u32x4(&mut out[EMITTER_OFF_ALLOC..], self.alloc);
        write_f32x4(&mut out[EMITTER_OFF_BODY..], self.body);
        for (i, c) in self.colors.iter().enumerate() {
            write_f32x4(&mut out[EMITTER_OFF_COLORS + i * 16..], *c);
        }
        out
    }
}

/// One fixed simulation step uniform, bound with a dynamic offset.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub(crate) struct StepGpu {
    /// Step duration in seconds.
    pub dt: f32,
    /// Simulation time at the end of the step, in seconds.
    pub sim_time: f32,
    /// Index of the step inside the frame, in `0..steps`.
    pub step_index: u32,
    /// Number of emitter records the simulation loops over.
    pub emitter_count: u32,
}

impl StepGpu {
    /// A step uniform.
    pub(crate) fn new(dt: f32, sim_time: f32, step_index: u32, emitter_count: u32) -> Self {
        StepGpu {
            dt,
            sim_time,
            step_index,
            emitter_count,
        }
    }

    /// The 16 little-endian bytes of the uniform.
    pub(crate) fn to_le_bytes(self) -> [u8; STEP_GPU_SIZE] {
        let mut out = [0u8; STEP_GPU_SIZE];
        out[0..4].copy_from_slice(&self.dt.to_le_bytes());
        out[4..8].copy_from_slice(&self.sim_time.to_le_bytes());
        out[8..12].copy_from_slice(&self.step_index.to_le_bytes());
        out[12..16].copy_from_slice(&self.emitter_count.to_le_bytes());
        out
    }
}

/// Distance in bytes between two step-uniform slots: [`STEP_GPU_SIZE`] rounded up to the
/// device's `min_uniform_buffer_offset_alignment` (0 is treated as 1).
pub(crate) fn step_slot_stride(min_uniform_offset_alignment: u32) -> u64 {
    let a = u64::from(min_uniform_offset_alignment.max(1));
    (STEP_GPU_SIZE as u64).div_ceil(a) * a
}

/// Writes the `MAX_EMITTERS` records of the table into `out`: record `i` is
/// `records[i]`, missing records (holes and ids past the end) are zero. `out` keeps its
/// capacity.
pub(crate) fn write_emitter_table(records: &[EmitterGpu], out: &mut Vec<u8>) {
    out.clear();
    out.resize(MAX_EMITTERS * EMITTER_GPU_SIZE, 0);
    for (i, record) in records.iter().take(MAX_EMITTERS).enumerate() {
        let off = i * EMITTER_GPU_SIZE;
        out[off..off + EMITTER_GPU_SIZE].copy_from_slice(&record.to_le_bytes());
    }
}

/// Writes the step uniforms of one frame into `out` at `stride` byte slots, zero-padding
/// each slot after its 16 bytes. At most [`MAX_STEPS_PER_FRAME`] steps are written.
pub(crate) fn write_step_table(steps: &[StepGpu], stride: u64, out: &mut Vec<u8>) {
    let stride = stride.max(STEP_GPU_SIZE as u64) as usize;
    let n = steps.len().min(MAX_STEPS_PER_FRAME);
    out.clear();
    out.resize(n * stride, 0);
    for (i, step) in steps.iter().take(n).enumerate() {
        let off = i * stride;
        out[off..off + STEP_GPU_SIZE].copy_from_slice(&step.to_le_bytes());
    }
}

/// Writes the `(first serial, count)` spawn commands of one frame, `[step * MAX_EMITTERS +
/// emitter]` first, into `out`. At most `MAX_STEPS_PER_FRAME * MAX_EMITTERS` pairs are
/// written.
pub(crate) fn write_spawn_table(spawns: &[[u32; 2]], out: &mut Vec<u8>) {
    let n = spawns.len().min(MAX_STEPS_PER_FRAME * MAX_EMITTERS);
    out.clear();
    out.resize(n * SPAWN_COMMAND_SIZE, 0);
    for (chunk, pair) in out.chunks_exact_mut(SPAWN_COMMAND_SIZE).zip(spawns) {
        chunk[0..4].copy_from_slice(&pair[0].to_le_bytes());
        chunk[4..8].copy_from_slice(&pair[1].to_le_bytes());
    }
}

/// Writes the per-slot emitter index table (`u32` per slot) into `out`. At most
/// [`TOTAL_CAPACITY_MAX`] entries are written.
pub(crate) fn write_slot_table(slots: &[u32], out: &mut Vec<u8>) {
    let n = slots.len().min(TOTAL_CAPACITY_MAX as usize);
    out.clear();
    out.resize(n * 4, 0);
    for (chunk, v) in out.chunks_exact_mut(4).zip(slots) {
        chunk.copy_from_slice(&v.to_le_bytes());
    }
}

fn write_f32x4(out: &mut [u8], v: [f32; 4]) {
    for (chunk, x) in out.chunks_exact_mut(4).zip(v) {
        chunk.copy_from_slice(&x.to_le_bytes());
    }
}

fn write_u32x4(out: &mut [u8], v: [u32; 4]) {
    for (chunk, x) in out.chunks_exact_mut(4).zip(v) {
        chunk.copy_from_slice(&x.to_le_bytes());
    }
}

/// Unit direction, falling back to `fallback` for a zero or non-finite vector.
fn normalize_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq <= 0.0 || !len_sq.is_finite() {
        return fallback;
    }
    let inv = 1.0 / len_sq.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_at(b: &[u8], off: usize) -> f32 {
        f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
    }

    fn u32_at(b: &[u8], off: usize) -> u32 {
        u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
    }

    #[test]
    fn emitter_gpu_offsets_and_size() {
        let mut d = EmitterDesc::for_kind(ParticleKind::Splash);
        d.position = [1.0, 2.0, 3.0];
        d.direction = [0.0, 2.0, 0.0];
        d.emissive = 2.5;
        d.opacity = 0.75;
        d.colors = vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
        d.ground_y = Some(-1.5);
        d.seed = 7;
        let rec = EmitterGpu::from_desc(&d, 12);
        let b = rec.to_le_bytes();
        assert_eq!(b.len(), 256);
        assert_eq!(EMITTER_GPU_SIZE, 256);

        for (i, v) in d.position.iter().enumerate() {
            assert_eq!(f32_at(&b, EMITTER_OFF_POSITION + i * 4), *v);
        }
        assert_eq!(f32_at(&b, EMITTER_OFF_POSITION + 12), d.spread_rad);
        assert_eq!(f32_at(&b, EMITTER_OFF_DIRECTION), 0.0);
        assert_eq!(f32_at(&b, EMITTER_OFF_DIRECTION + 4), 1.0);
        assert_eq!(f32_at(&b, EMITTER_OFF_DIRECTION + 12), 2.5);
        assert_eq!(f32_at(&b, EMITTER_OFF_LIFETIME), d.lifetime_s[0]);
        assert_eq!(f32_at(&b, EMITTER_OFF_LIFETIME + 8), d.speed[0]);
        assert_eq!(f32_at(&b, EMITTER_OFF_SIZE), d.size[0]);
        assert_eq!(f32_at(&b, EMITTER_OFF_SIZE + 8), d.gravity);
        assert_eq!(f32_at(&b, EMITTER_OFF_VOLUME + 8), d.volume_half[2]);
        assert_eq!(f32_at(&b, EMITTER_OFF_VOLUME + 12), -1.5);
        assert_eq!(u32_at(&b, EMITTER_OFF_HEADER), 0);
        assert_eq!(u32_at(&b, EMITTER_OFF_HEADER + 4), 7);
        assert_eq!(u32_at(&b, EMITTER_OFF_HEADER + 8), 1);
        assert_eq!(u32_at(&b, EMITTER_OFF_HEADER + 12), 2);
        assert_eq!(u32_at(&b, EMITTER_OFF_ALLOC), 12);
        assert_eq!(u32_at(&b, EMITTER_OFF_ALLOC + 4), d.max_particles);
        assert_eq!(u32_at(&b, EMITTER_OFF_ALLOC + 8), 0);
        assert_eq!(f32_at(&b, EMITTER_OFF_BODY), 0.75);
        assert_eq!(f32_at(&b, EMITTER_OFF_BODY + 4), 0.0);
        assert_eq!(f32_at(&b, EMITTER_OFF_COLORS), 0.1);
        assert_eq!(f32_at(&b, EMITTER_OFF_COLORS + 4), 0.2);
        assert_eq!(f32_at(&b, EMITTER_OFF_COLORS + 12), 1.0);
        assert_eq!(f32_at(&b, EMITTER_OFF_COLORS + 16), 0.4);
        assert_eq!(f32_at(&b, EMITTER_OFF_COLORS + 32), 0.0);
        assert_eq!(f32_at(&b, EMITTER_OFF_COLORS + 7 * 16), 0.0);
    }

    #[test]
    fn step_gpu_is_16_bytes_and_stride_rounds_up() {
        let step = StepGpu::new(STEP_S, 1.5, 3, 2);
        let b = step.to_le_bytes();
        assert_eq!(b.len(), 16);
        assert_eq!(STEP_GPU_SIZE, 16);
        assert_eq!(f32_at(&b, 0), STEP_S);
        assert_eq!(f32_at(&b, 4), 1.5);
        assert_eq!(u32_at(&b, 8), 3);
        assert_eq!(u32_at(&b, 12), 2);
        assert_eq!(step_slot_stride(256), 256);
        assert_eq!(step_slot_stride(64), 64);
        assert_eq!(step_slot_stride(16), 16);
        assert_eq!(step_slot_stride(4), 16);
        assert_eq!(step_slot_stride(0), 16);
    }

    #[test]
    fn table_writers_keep_their_capacity() {
        let mut out = Vec::new();
        write_emitter_table(&[], &mut out);
        assert_eq!(out.len(), MAX_EMITTERS * EMITTER_GPU_SIZE);
        let cap = out.capacity();
        write_emitter_table(&[EmitterGpu::default()], &mut out);
        assert_eq!(out.capacity(), cap);

        write_step_table(&[StepGpu::default(); 2], 256, &mut out);
        assert_eq!(out.len(), 512);
        let cap = out.capacity();
        write_step_table(&[], 256, &mut out);
        assert_eq!(out.capacity(), cap);

        write_spawn_table(&[[1, 2]; 3], &mut out);
        assert_eq!(out.len(), 24);
        assert_eq!(u32_at(&out, 0), 1);
        assert_eq!(u32_at(&out, 4), 2);
        let cap = out.capacity();
        write_spawn_table(&[], &mut out);
        assert_eq!(out.capacity(), cap);

        write_slot_table(&[3, SLOT_FREE_ALIAS], &mut out);
        assert_eq!(out.len(), 8);
        assert_eq!(u32_at(&out, 0), 3);
        assert_eq!(u32_at(&out, 4), u32::MAX);
        let cap = out.capacity();
        write_slot_table(&[], &mut out);
        assert_eq!(out.capacity(), cap);
    }

    /// The slot table's free marker (kept here to avoid a state dependency in this test).
    const SLOT_FREE_ALIAS: u32 = u32::MAX;
}
