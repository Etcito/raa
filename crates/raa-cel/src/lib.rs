//! Cel palette resolve (order 190), geometric ink and opt-in toon materials.
pub mod color;
mod desc;
mod feature;
pub mod palette;
mod provider;
mod shaders;
pub use desc::{CelSettings, Dither};
pub use feature::{CelFeature, CelHandle};
pub use provider::{FLAG_CEL, install, provider};
