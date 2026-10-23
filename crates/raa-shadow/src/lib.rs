//! Cascaded soft sun shadows (S12 `shadow.csm`).
//!
//! CSM values: 4 cascades of 2048x2048 in a Depth32Float array, practical split lambda 0.75,
//! shadow distance 60 m, 10% blend band, PCSS with 16 blocker and 32 PCF samples, slope bias
//! 2.0, constant bias 2, normal offset 1.5 texels. This module holds the description and its
//! validation (`shadow.invalid`), the practical cascade splits and the texel-snapped cascade
//! view-projection matrices, the caster rendering into the cascade array and the
//! `ShadowFeature` / `ShadowHandle` that install and drive the `shadow.csm` pass.

mod cascades;
mod caster;
mod desc;
mod feature;
mod provider;

pub use cascades::{cascade_matrix, cascade_splits};
pub use desc::ShadowDesc;
pub use feature::{ShadowFeature, ShadowHandle};
