//! Cascade splits (practical scheme) and texel-snapped cascade view-projection matrices.

use raa_math::{Mat4, Vec3, Vec4};

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
    let Some(inv_view) = view.inverse() else {
        return Mat4::IDENTITY;
    };

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
        return Mat4::IDENTITY;
    }

    // (3) Light view along the sun travel direction, from the world origin (a fixed origin keeps
    // the texel grid fixed in world space).
    let dir = match sun_dir.normalize_or_zero() {
        d if d == Vec3::ZERO => FALLBACK_SUN,
        d => d,
    };
    let up = if dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let light_view = Mat4::look_at_rh(Vec3::ZERO, dir, up);

    // (4) Snap the centre to whole texels in light x/y.
    let c = light_view.transform_point(centre);
    let texel = 2.0 * radius / resolution.max(1) as f32;
    let cx = (c.x / texel).round() * texel;
    let cy = (c.y / texel).round() * texel;
    // Light-view depth along the travel direction (the view looks down -Z).
    let depth = -c.z;

    // (5) Orthographic projection: x/y [-r, r] around the snapped centre, depth
    // [depth - r, depth + r] to clip z [0, 1].
    let inv_r = 1.0 / radius;
    let inv_2r = 0.5 * inv_r;
    let proj = Mat4::from_cols(
        Vec4::new(inv_r, 0.0, 0.0, 0.0),
        Vec4::new(0.0, inv_r, 0.0, 0.0),
        Vec4::new(0.0, 0.0, -inv_2r, 0.0),
        Vec4::new(-cx * inv_r, -cy * inv_r, (radius - depth) * inv_2r, 1.0),
    );
    proj * light_view
}
