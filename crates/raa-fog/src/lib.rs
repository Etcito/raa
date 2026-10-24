//! Height fog and aerial perspective of the lit renderer (S12 `fog`, order 68).
//!
//! [`FogDesc`] describes exponential height fog (density, falloff, base height, start
//! distance), its colour and sun inscatter, and aerial perspective. [`FogDesc::preset`]
//! gives the built-in moods (`clear`, `haze`, `snow`, `jungle`, `night`); the default is
//! `clear` (density 0, so fog is off). [`FogDesc::validate`] refuses with `post.invalid`,
//! naming the field, and [`FogDesc::write_env`] packs the S10 `EnvGpu` fog fields.
//!
//! The `fog` FeaturePass (order 68) and the `height` Fog slot provider with
//! `FogFeature`/`FogHandle` land with the pass task; this crate currently exports the
//! description.

mod desc;

pub use desc::FogDesc;
