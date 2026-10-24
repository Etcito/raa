//! GPU particles for the RAA lit renderer.
//!
//! [`ParticleKind`] names the four effect families (splash, snow, confetti,
//! sparkles) and [`EmitterDesc`] is the validated description of one emitter.
//! [`EmitterDesc::for_kind`] gives the per-kind defaults and
//! [`EmitterDesc::validate`] refuses every malformed field with
//! `particles.invalid`, naming it in the message.

mod desc;

pub use desc::*;
