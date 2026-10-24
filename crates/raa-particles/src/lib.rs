//! GPU particles for the RAA lit renderer.
//!
//! [`ParticleKind`] names the four effect families (splash, snow, confetti,
//! sparkles) and [`EmitterDesc`] is the validated description of one emitter.
//! [`EmitterDesc::for_kind`] gives the per-kind defaults and
//! [`EmitterDesc::validate`] refuses every malformed field with
//! `particles.invalid`, naming it in the message.
//!
//! [`pcg_hash`] and [`spawn_sample`] are the seeded CPU twin of the
//! `particles.sim` WGSL: a sample is deterministic in `(seed, serial)` and the
//! module documents the exact recipe and the 48 B [`PARTICLE_STRIDE`] state
//! record.

mod desc;
mod gpu;
mod opaque;
mod shaders;
mod sim;
mod spawn;
mod state;

pub use desc::*;
pub use spawn::*;

#[doc(hidden)]
pub use opaque::opaque_module_wgsl;
#[doc(hidden)]
pub use shaders::sim_module_wgsl;
