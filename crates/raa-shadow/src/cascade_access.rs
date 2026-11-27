//! Read access to the cascades of the existing `shadow.csm` pass: the values the pass computes
//! ([`cascade_frame`]) and the per-frame cascade targets a [`ShadowHandle`](crate::ShadowHandle)
//! exposes ([`CascadeTargets`]).
//!
//! The pass, its cascade array, its provider and `ShadowGpu` are unchanged; this module only
//! shares what the pass already computes so another pass (the virtualized-geometry seam) can
//! draw into the same layers with the same view-projections.

use raa_math::{Aabb, Mat4, Vec3};

use crate::cascades::{CascadeFit, cascade_fit, ladder_up};

/// Number of cascades (CSM).
const CASCADES: usize = 4;
/// Practical split lambda (CSM).
pub(crate) const SPLIT_LAMBDA: f32 = 0.75;

/// With `spread`, the split range is at most this multiple of the shadow distance (a huge
/// scene keeps its near cascades; the last one still reaches the cover distance).
const SPREAD_CAP: f32 = 4.0;

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

/// The cover distance is rounded up on a ladder of 2^(1/4) steps (about 19 %), so it changes
/// only at a few camera positions and the last cascade does not resize every frame.
const COVER_LADDER: f32 = 4.0;

/// Distance from the camera of `view` to the farthest corner of `scene`, rounded up to the
/// next 2^(k/4). `None` for a non-invertible view or non-finite bounds.
pub fn cover_distance(view: &Mat4, scene: &Aabb) -> Option<f32> {
    let eye = view.inverse()?.transform_point(Vec3::ZERO);
    if !(eye.is_finite() && scene.min.is_finite() && scene.max.is_finite()) {
        return None;
    }
    let mut far = 0.0f32;
    for k in 0..8 {
        let p = Vec3::new(
            if k & 1 == 0 { scene.min.x } else { scene.max.x },
            if k & 2 == 0 { scene.min.y } else { scene.max.y },
            if k & 4 == 0 { scene.min.z } else { scene.max.z },
        );
        far = far.max((p - eye).length());
    }
    (far > 0.0).then(|| ladder_up(far, COVER_LADDER))
}

/// The cascades of one frame when the shadow covers the scene.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CoveredFrame {
    pub matrices: [Mat4; CASCADES],
    pub splits: [f32; CASCADES],
    pub fits: [CascadeFit; CASCADES],
    pub texels_m: [f32; CASCADES],
    /// The cover distance of the scene (`None` without scene bounds).
    pub cover_m: Option<f32>,
}

/// [`cascade_frame`] that keeps the sun shadowed everywhere in the scene. The first three
/// cascades are unchanged (practical splits of `[near, far.min(distance_m)]`, so the near
/// shadow quality is the same); the last one reaches `cover_distance(view, scene)` (capped by
/// the camera far plane) when that is farther than the shadow distance, so no surface of the
/// scene the camera can see lies past the last cascade (where the lighting would leave the
/// sun unshadowed). Each cascade is then fitted to the smaller of its slice sphere and the
/// scene's light-space bounds ([`cascade_fit`]). Without `scene` it equals [`cascade_frame`].
#[allow(
    clippy::too_many_arguments,
    reason = "the contract signature mirrors the pass's inputs"
)]
pub fn cascade_frame_covered(
    view: &Mat4,
    fov_y_rad: f32,
    aspect: f32,
    near: f32,
    far: f32,
    distance_m: f32,
    sun_dir: Vec3,
    resolution: u32,
    scene: Option<&Aabb>,
) -> CoveredFrame {
    cascade_frame_covered_n(
        view,
        fov_y_rad,
        aspect,
        near,
        far,
        distance_m,
        sun_dir,
        resolution,
        scene,
        CASCADES,
        SPLIT_LAMBDA,
        false,
    )
}

/// [`cascade_frame_covered`] with `n` (1..=4) cascades: the practical splits for `n`, the
/// cover reach applied to cascade `n - 1`; entries past `n` repeat the last used split and
/// matrix (they are never rendered nor selected: the shader clamps to cascade `n - 1`).
/// `lambda` is the practical split weight (0 uniform, 1 logarithmic; [`cascade_frame_covered`]
/// uses 0.75). `spread` (with a scene) runs the practical splits over
/// `[near, min(far, cover distance, 4 x distance)]` instead of `[near, min(far, distance)]`.
#[allow(
    clippy::too_many_arguments,
    reason = "the contract signature mirrors the pass's inputs"
)]
pub fn cascade_frame_covered_n(
    view: &Mat4,
    fov_y_rad: f32,
    aspect: f32,
    near: f32,
    far: f32,
    distance_m: f32,
    sun_dir: Vec3,
    resolution: u32,
    scene: Option<&Aabb>,
    n: usize,
    lambda: f32,
    spread: bool,
) -> CoveredFrame {
    let n = n.clamp(1, CASCADES);
    let cover_m = scene.and_then(|b| cover_distance(view, b));
    // `spread`: the practical splits run over the part of the scene the camera can see,
    // min(far, cover distance, SPREAD_CAP x distance), instead of [near, min(far, distance)],
    // so no cascade is spent past the scene and the middle cascades reach the far side.
    let range = match cover_m {
        Some(cover) if spread => far
            .min(cover)
            .min(SPREAD_CAP * distance_m)
            .max(near + 1e-3),
        _ => far.min(distance_m),
    };
    let mut splits = crate::cascade_splits(near, range, lambda, n);
    if let Some(cover) = cover_m {
        let reach = cover.min(far);
        if reach > splits[n - 1] {
            for s in splits.iter_mut().skip(n - 1) {
                *s = reach;
            }
        }
    }
    let mut out = CoveredFrame {
        matrices: [Mat4::IDENTITY; CASCADES],
        splits,
        fits: [CascadeFit::Slice; CASCADES],
        texels_m: [0.0; CASCADES],
        cover_m,
    };
    let mut previous = near;
    for (i, &split) in splits.iter().enumerate() {
        if i >= n {
            out.matrices[i] = out.matrices[n - 1];
            out.fits[i] = out.fits[n - 1];
            out.texels_m[i] = out.texels_m[n - 1];
            continue;
        }
        let fitted = cascade_fit(
            view,
            fov_y_rad,
            aspect,
            (previous, split),
            sun_dir,
            resolution,
            scene,
        );
        out.matrices[i] = fitted.matrix;
        out.fits[i] = fitted.fit;
        out.texels_m[i] = fitted.texel_m;
        previous = split;
    }
    out
}
