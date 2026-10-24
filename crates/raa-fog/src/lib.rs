//! Height fog and aerial perspective of the lit renderer (S12 `fog`, order 68).
//!
//! [`FogDesc`] describes exponential height fog (density, falloff, base height, start
//! distance), its colour and sun inscatter, and aerial perspective. [`FogDesc::preset`]
//! gives the built-in moods (`clear`, `haze`, `snow`, `jungle`, `night`); the default is
//! `clear` (density 0, so fog is off). [`FogDesc::validate`] refuses with `post.invalid`,
//! naming the field, and [`FogDesc::write_env`] packs the S10 `EnvGpu` fog fields.
//!
//! [`integral`] holds the analytic height-fog optical depth, its Simpson reference and the
//! transmittance. [`FogFeature::install`] adds the `height` Fog slot provider (priority 10)
//! to the lit renderer and the `fog` FeaturePass (order 68), which writes the `EnvGpu` fog
//! fields and applies the fog to the HDR chain; [`FogHandle`] sets and reads the description.

mod desc;
mod feature;
mod shaders;

pub mod integral;

pub use desc::FogDesc;
pub use feature::{FogFeature, FogHandle};
