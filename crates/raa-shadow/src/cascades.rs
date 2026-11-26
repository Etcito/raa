//! Cascade splits (practical scheme) and texel-snapped cascade view-projection matrices.

use raa_math::{Aabb, Mat4, Vec3, Vec4};

/// Maximum number of cascades (CSM).
const MAX_CASCADES: usize = 4;

/// The sphere radius is rounded up to a multiple of this step (m), so the cascade extent does
/// not change while the camera rotates.
const RADIUS_STEP: f32 = 1.0 / 16.0;

/// Sun travel direction used when `sun_dir` is zero or not finite (straight down).
const FALLBACK_SUN: Vec3 = Vec3::new(0.0, -1.0, 0.0);

/// View-space far distance of each cascade by the practical split scheme:
/// `split_i = lambda * near * (far / near)^(i / n) + (1 - lambda) * (near + (far - near) * i / n)`
/// for `i = 1..=n`.
///
/// `n` is clamped to 1..=4 and `lambda` to [0, 1]; entries past `n` equal `far` and the last
/// used entry is `far` exactly. When `near <= 0` the logarithmic term has no meaning and the
/// uniform term alone is used.
pub fn cascade_splits(near: f32, far: f32, lambda: f32, n: usize) -> [f32; 4] {
    let n = n.clamp(1, MAX_CASCADES);
    let lambda = if lambda.is_finite() {
        lambda.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut out = [far; MAX_CASCADES];
    for (i, split) in out.iter_mut().enumerate().take(n - 1) {
        let t = (i + 1) as f32 / n as f32;
        let uniform = near + (far - near) * t;
        let log = if near > 0.0 {
            near * (far / near).powf(t)
        } else {
            uniform
        };
        *split = lambda * log + (1.0 - lambda) * uniform;
    }
    out
}

/// View-projection of one cascade for the view-space slice `split = (near, far)` of the camera
/// with view matrix `view`, vertical field of view `fov_y_rad` and `aspect`.
///
/// The slice's 8 world-space corners are enclosed by a sphere (centroid, max corner distance,
/// radius rounded up to 1/16 m). The light view looks along `sun_dir` (travel direction) from
/// the world origin, the sphere centre is snapped to a whole shadow-map texel
/// (`2 r / resolution`) in light x/y, and an orthographic projection maps x/y `[-r, r]` around
/// the snapped centre to clip [-1, 1] and the depth range `[centre - r, centre + r]` to clip z
/// [0, 1]. Casters in front of the near plane are pancaked by the shadow pass.
///
/// A non-invertible `view` (or non-finite inputs) yields `Mat4::IDENTITY`.
pub fn cascade_matrix(
    view: &Mat4,
    fov_y_rad: f32,
    aspect: f32,
    split: (f32, f32),
    sun_dir: Vec3,
    resolution: u32,
) -> Mat4 {
    cascade_fit(view, fov_y_rad, aspect, split, sun_dir, resolution, None).matrix
}

/// How a cascade's orthographic box was fitted.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CascadeFit {
    /// Around the bounding sphere of the camera frustum slice (rotation-stable).
    Slice,
    /// Around the light-space bounds of the whole scene (camera-independent): chosen when it
    /// is smaller than the slice's sphere, so every receiver of the scene is inside the box.
    Scene,
}

/// One fitted cascade: its view-projection, the fit used and the world size of one texel.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FittedCascade {
    pub matrix: Mat4,
    pub fit: CascadeFit,
    /// World size of one shadow-map texel in metres (0 for the identity fallback).
    pub texel_m: f32,
}

/// The light view used by every cascade: looks along `sun_dir` (travel direction) from the
/// world origin (a fixed origin keeps the texel grid fixed in world space).
fn light_view_of(sun_dir: Vec3) -> Mat4 {
    let dir = match sun_dir.normalize_or_zero() {
        d if d == Vec3::ZERO => FALLBACK_SUN,
        d => d,
    };
    let up = if dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    Mat4::look_at_rh(Vec3::ZERO, dir, up)
}

/// Orthographic box of half size `half` around light-space x/y `(cx, cy)` (already snapped);
/// clip z = `depth * z_scale + z_bias` with the light-view depth `-z`.
fn ortho(light_view: Mat4, half: f32, cx: f32, cy: f32, z_scale: f32, z_bias: f32) -> Mat4 {
    let inv_r = 1.0 / half;
    let proj = Mat4::from_cols(
        Vec4::new(inv_r, 0.0, 0.0, 0.0),
        Vec4::new(0.0, inv_r, 0.0, 0.0),
        Vec4::new(0.0, 0.0, -z_scale, 0.0),
        Vec4::new(-cx * inv_r, -cy * inv_r, z_bias, 1.0),
    );
    proj * light_view
}

/// Scene boxes: the half size is rounded up on a ladder of 2^(1/8) steps (about 9 %), so
/// small changes of the scene bounds (a moving character at the edge) keep the texel grid.
const SCENE_HALF_LADDER: f32 = 8.0;
/// Margin added around the scene bounds in light space (normal offset and PCF taps), metres.
const SCENE_MARGIN: f32 = 0.5;

/// `x` rounded up to the next `2^(k / steps)`.
pub(crate) fn ladder_up(x: f32, steps: f32) -> f32 {
    if !(x.is_finite() && x > 0.0) {
        return x;
    }
    ((x.log2() * steps).ceil() / steps).exp2()
}

/// [`cascade_matrix`] that also considers the scene bounds: when the light-space square that
/// holds all of `scene` (plus a margin, half size on a 2^(1/8) ladder) is smaller than the
/// slice's sphere, the cascade is fitted to the scene instead (centre snapped to whole texels,
/// depth range the scene's light-depth range). Such a box is independent of the camera, so it
/// does not shimmer, and it holds every receiver of the scene.
pub fn cascade_fit(
    view: &Mat4,
    fov_y_rad: f32,
    aspect: f32,
    split: (f32, f32),
    sun_dir: Vec3,
    resolution: u32,
    scene: Option<&Aabb>,
) -> FittedCascade {
    let identity = FittedCascade {
        matrix: Mat4::IDENTITY,
        fit: CascadeFit::Slice,
        texel_m: 0.0,
    };
    let Some(inv_view) = view.inverse() else {
        return identity;
    };
    let res = resolution.max(1) as f32;

    // (1) Frustum slice corners in world space.
    let tan_half = (0.5 * fov_y_rad).tan();
    let mut corners = [Vec3::ZERO; 8];
    for (k, corner) in corners.iter_mut().enumerate() {
        let d = if k < 4 { split.0 } else { split.1 };
        let h = d * tan_half;
        let w = h * aspect;
        let sx = if k & 1 == 0 { -1.0 } else { 1.0 };
        let sy = if k & 2 == 0 { -1.0 } else { 1.0 };
        *corner = inv_view.transform_point(Vec3::new(sx * w, sy * h, -d));
    }

    // (2) Bounding sphere, radius rounded up to the step.
    let mut centre = Vec3::ZERO;
    for c in corners {
        centre += c;
    }
    centre = centre / 8.0;
    let mut radius = 0.0f32;
    for c in corners {
        radius = radius.max((c - centre).length());
    }
    let radius = ((radius / RADIUS_STEP).ceil() * RADIUS_STEP).max(RADIUS_STEP);
    if !centre.is_finite() || !radius.is_finite() {
        return identity;
    }

    // (3) Light view along the sun travel direction.
    let light_view = light_view_of(sun_dir);

    // (4) The scene box, when it is the smaller square.
    if let Some(b) = scene.filter(|b| b.min.is_finite() && b.max.is_finite()) {
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for k in 0..8 {
            let p = Vec3::new(
                if k & 1 == 0 { b.min.x } else { b.max.x },
                if k & 2 == 0 { b.min.y } else { b.max.y },
                if k & 4 == 0 { b.min.z } else { b.max.z },
            );
            let l = light_view.transform_point(p);
            lo = lo.min(l);
            hi = hi.max(l);
        }
        let extent = (hi.x - lo.x).max(hi.y - lo.y);
        let half = ladder_up(0.5 * extent + SCENE_MARGIN, SCENE_HALF_LADDER);
        if half.is_finite() && half > 0.0 && half < radius {
            let texel = 2.0 * half / res;
            let cx = (0.5 * (lo.x + hi.x) / texel).round() * texel;
            let cy = (0.5 * (lo.y + hi.y) / texel).round() * texel;
            // Light-view depth along the travel direction is -z.
            let (d0, d1) = (-hi.z - SCENE_MARGIN, -lo.z + SCENE_MARGIN);
            let z_scale = 1.0 / (d1 - d0).max(1e-3);
            return FittedCascade {
                matrix: ortho(light_view, half, cx, cy, z_scale, -d0 * z_scale),
                fit: CascadeFit::Scene,
                texel_m: texel,
            };
        }
    }

    // (5) Snap the sphere centre to whole texels in light x/y; depth [depth - r, depth + r].
    let c = light_view.transform_point(centre);
    let texel = 2.0 * radius / res;
    let cx = (c.x / texel).round() * texel;
    let cy = (c.y / texel).round() * texel;
    // Light-view depth along the travel direction (the view looks down -Z): [depth - r,
    // depth + r] to clip z [0, 1].
    let depth = -c.z;
    let inv_2r = 0.5 * (1.0 / radius);
    FittedCascade {
        matrix: ortho(
            light_view,
            radius,
            cx,
            cy,
            inv_2r,
            (radius - depth) * inv_2r,
        ),
        fit: CascadeFit::Slice,
        texel_m: texel,
    }
}
