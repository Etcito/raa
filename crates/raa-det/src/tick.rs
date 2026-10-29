//! The contracted tick: a `u32` counter at [`ANIM_TICK_HZ`].

/// Ticks per second on the deterministic path: 30. A tick is one AClip v2
/// sample; the reference implementation the raa-anim sampling ports
/// (`sketchbook/unified.ts`) runs AClip v2 at a fixed 30 fps, so clip ticks,
/// rig edits and sim steps all count the same unit.
pub const ANIM_TICK_HZ: u32 = 30;

/// A simulation step: `u32` ticks at [`ANIM_TICK_HZ`]. Ticks wrap like
/// `u32` (advance with [`wrapping_advance`], never saturate), so a long
/// session stays identical on every client.
pub type Tick = u32;

/// One tick in seconds, `f64` (exactly 1/30).
pub const TICK_SECONDS_F64: f64 = 1.0 / ANIM_TICK_HZ as f64;

/// One tick in seconds, `f32`: the nearest `f32` (bit pattern `0x3d088889`).
pub const TICK_SECONDS_F32: f32 = 1.0 / ANIM_TICK_HZ as f32;

/// Adds `by` with `u32` wrapping: a deterministic restart at 0, never
/// saturation, so distant clients agree through the counter's wrap.
pub fn wrapping_advance(tick: Tick, by: u32) -> Tick {
    tick.wrapping_add(by)
}

/// The tick's instant in seconds, `f64`: exact one-tick resolution up to
/// [`u32::MAX`] ticks (143165576.5 s).
pub fn tick_seconds_f64(tick: Tick) -> f64 {
    tick as f64 / ANIM_TICK_HZ as f64
}

/// The tick's instant in seconds, `f32`: computed in `f64`, converted once.
/// Deterministic on wasm32 and native alike, but the `f32` result is finer
/// than one tick only below roughly 2^21 s (~2.4 y); deterministic code
/// compares ticks, not these seconds.
pub fn tick_seconds_f32(tick: Tick) -> f32 {
    tick_seconds_f64(tick) as f32
}
