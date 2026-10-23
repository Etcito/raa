//! Cascaded soft sun shadows (S12 `shadow.csm`).
//!
//! CSM values: 4 cascades of 2048x2048 in a Depth32Float array, practical split lambda 0.75,
//! shadow distance 60 m, 10% blend band, PCSS with 16 blocker and 32 PCF samples, slope bias
//! 2.0, constant bias 2, normal offset 1.5 texels. This module currently holds the
//! description and its validation (`shadow.invalid`).

mod desc;

pub use desc::ShadowDesc;
