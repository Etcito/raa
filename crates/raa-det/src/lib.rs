//! Determinism brick for RAA: the contracted simulation tick and the
//! deterministic FNV-1a 32-bit hash over `f32` values. Allocation-free,
//! no unsafe code, no third-party crates.

mod hash;
mod tick;

pub use hash::{DET_HASH, FNV1A32_OFFSET, FNV1A32_PRIME, Fnv1a32, det_hash_f32s};
pub use tick::{
    ANIM_TICK_HZ, TICK_SECONDS_F32, TICK_SECONDS_F64, Tick, tick_seconds_f32, tick_seconds_f64,
    wrapping_advance,
};
