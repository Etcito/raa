//! Cascaded soft sun shadows (S12 `shadow.csm`).
//!
//! CSM values: 4 cascades of 2048x2048 in a Depth32Float array, practical split lambda 0.75,
//! shadow distance 60 m, 10% blend band, PCSS with 16 blocker and 32 PCF samples, slope bias
//! 2.0, constant bias 2, normal offset 1.5 texels. With `cover` (default) the last cascade
//! reaches the farthest corner of the scene bounds and a cascade is fitted to the scene's
//! light-space bounds when smaller than its slice, so no surface of the scene is past the
//! last cascade (where the sun would be unshadowed). This module holds the description and its
//! validation (`shadow.invalid`), the practical cascade splits and the texel-snapped cascade
//! view-projection matrices, the caster rendering into the cascade array and the
//! `ShadowFeature` / `ShadowHandle` that install and drive the `shadow.csm` pass.

mod cascade_access;
mod cascades;
mod caster;
mod desc;
mod feature;
mod provider;
mod vsm;
mod vsm_gpu;

pub use cascade_access::{
    CascadeTargets, CoveredFrame, cascade_frame, cascade_frame_covered, cascade_frame_covered_n, cover_distance,
};
pub use cascades::{CascadeFit, FittedCascade, cascade_fit, cascade_matrix, cascade_splits};
pub use desc::{ShadowDesc, ShadowMode, VSM_BUDGET_DEFAULT, VSM_PAGES_DEFAULT};
pub use feature::{ShadowFeature, ShadowHandle, ShadowStats};
pub use vsm::VsmStats;
