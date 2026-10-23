//! Ground-truth ambient occlusion of the lit renderer (S12 `ao.gtao`).
//!
//! [`AoDesc`] carries the GTAO settings and [`AoDesc::validate`] refuses invalid ones with
//! `ao.invalid`. The pass itself is the FeaturePass `ao.gtao` (order 40, full resolution):
//! Jimenez 2016 GTAO with 4 slice directions rotated by a 4x4 noise and 8 horizon steps per
//! side over the projected radius, a cosine-weighted visibility integral with a thickness
//! heuristic, reading `depth` and `normal` and writing `1 - intensity * (1 - vis^power)` into
//! an R32Float texture; then 2 separable 7-tap depth-aware bilateral passes (horizontal into
//! R32Float, vertical into the final Rgba16Float texture with the AO in `.r`, since R16Float
//! is not a core WebGPU storage format; the weight is `exp(-|dz| / (0.02 z))`). The `Ao` slot
//! provider `gtao` (priority 10) samples it with `textureLoad` of binding 7 `.r`, and the
//! multi-bounce term is applied in the ambient slot.
//!
//! The values are fixed by the architecture: radius 0.8 m, 4 slices x 8 steps per side, 2
//! separable 7-tap depth-aware denoise passes.
//!
//! [`GtaoFeature::install`] adds the provider and the pass; the returned [`GtaoHandle`]
//! sets the description and the frame's camera. Until a camera is set, or while the pass is
//! disabled, the lighting keeps the dummy AO (1).

mod desc;
mod feature;
mod shaders;

pub use desc::AoDesc;
pub use feature::{GtaoFeature, GtaoHandle};
