//! Animated liquid surfaces of the lit renderer: water, slime and mud material functions
//! with Gerstner vertex stages, ripple rings, foam and refraction through the
//! `Transmission` slot.
//!
//! [`wave_height`] is the CPU twin of the Gerstner wave sum every WGSL file of this crate
//! follows, and [`WAVE_COUNT`] is the number of packed waves. The documentation of
//! [`wave_height`] is the binding parameter-packing and displacement/normal contract of the
//! crate.

mod builtins;
mod feature;
mod shaders;
mod waves;
pub use builtins::LIQUID_BUILTINS;
pub use feature::install;
pub use waves::*;
