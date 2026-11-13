//! Read access to the cascades of the existing `shadow.csm` pass: the values the pass computes
//! ([`cascade_frame`]) and the per-frame cascade targets a [`ShadowHandle`](crate::ShadowHandle)
//! exposes ([`CascadeTargets`]).
//!
//! The pass, its cascade array, its provider and `ShadowGpu` are unchanged; this module only
//! shares what the pass already computes so another pass (the virtualized-geometry seam) can
//! draw into the same layers with the same view-projections.

use raa_math::{Mat4, Vec3};

/// Number of cascades (CSM).
const CASCADES: usize = 4;
/// Practical split lambda (CSM).
const SPLIT_LAMBDA: f32 = 0.75;

/// The cascades the `shadow.csm` pass prepared this frame: their count and resolution, the
/// four view-projections, the view-space splits, the four single-layer `Depth32Float` views of
/// the cascade array and the sun travel direction.
pub struct CascadeTargets {
    /// Number of cascades (always 4).
    pub count: u32,
    /// Width and height of each cascade layer in texels.
    pub resolution: u32,
    /// One view-projection per cascade, in cascade order.
    pub view_proj: [Mat4; CASCADES],
    /// View-space far distance of each cascade.
    pub splits: [f32; CASCADES],
    /// One D2 `Depth32Float` view per cascade layer (single-layer views of the shared array).
    pub layer_views: [wgpu::TextureView; CASCADES],
    /// World travel direction of the sun.
    pub sun_dir: Vec3,
}

/// The splits and cascade matrices `shadow.csm` computes for one frame: the practical splits
/// of `[near, far.min(distance_m)]` with lambda 0.75 and the four `cascade_matrix` results for
/// `(previous split or near, split i)`.
///
/// Returns `(view_projections, splits)`.
#[allow(
    clippy::too_many_arguments,
    reason = "the contract signature mirrors the pass's inputs"
)]
pub fn cascade_frame(
    view: &Mat4,
    fov_y_rad: f32,
    aspect: f32,
    near: f32,
    far: f32,
    distance_m: f32,
    sun_dir: Vec3,
    resolution: u32,
) -> ([Mat4; CASCADES], [f32; CASCADES]) {
    let far = far.min(distance_m);
    let splits = crate::cascade_splits(near, far, SPLIT_LAMBDA, CASCADES);
    let mut matrices = [Mat4::IDENTITY; CASCADES];
    let mut previous = near;
    for (i, matrix) in matrices.iter_mut().enumerate() {
        *matrix = crate::cascade_matrix(
            view,
            fov_y_rad,
            aspect,
            (previous, splits[i]),
            sun_dir,
            resolution,
        );
        previous = splits[i];
    }
    (matrices, splits)
}
