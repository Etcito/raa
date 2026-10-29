//! Conservative S18 bounds. Unknown or non-affine bounds never reject a draw.
use raa_math::{Aabb, Mat4};

fn valid(b: Aabb) -> bool {
    [b.min.x, b.min.y, b.min.z, b.max.x, b.max.y, b.max.z]
        .iter().all(|v| v.is_finite())
        && b.min.x <= b.max.x && b.min.y <= b.max.y && b.min.z <= b.max.z
}

fn affine(m: Mat4) -> bool {
    m.cols.iter().flatten().all(|v| v.is_finite())
        && [m.cols[0][3], m.cols[1][3], m.cols[2][3], m.cols[3][3]] == [0., 0., 0., 1.]
}

/// Union in mesh space, before applying the instance transform.
pub fn posed_bounds(rest: Aabb, joints: &[f32]) -> Option<Aabb> {
    if !valid(rest) || joints.is_empty() || !joints.len().is_multiple_of(16) {
        return None;
    }
    let mut bounds = Aabb::EMPTY;
    for joint in joints.chunks_exact(16) {
        let mut m = Mat4::IDENTITY;
        for (out, value) in m.cols.iter_mut().flatten().zip(joint) { *out = *value; }
        if !affine(m) { return None; }
        let b = rest.transform(&m);
        if !valid(b) { return None; }
        bounds = bounds.union(&b);
    }
    Some(bounds)
}

/// WebGPU clip planes (-w <= x,y <= w, 0 <= z <= w), tested against a world AABB.
pub fn visible(bounds: Option<Aabb>, world: Mat4, view_proj: Mat4) -> bool {
    let Some(bounds) = bounds.filter(|b| valid(*b)) else { return true };
    if !affine(world) { return true; }
    let b = bounds.transform(&world);
    if !valid(b) { return true; }
    let row = |r: usize| std::array::from_fn::<_, 4, _>(|c| view_proj.cols[c][r]);
    let [x, y, z, w] = [row(0), row(1), row(2), row(3)];
    let add = |a: [f32; 4], b: [f32; 4], sign: f32|
        std::array::from_fn(|i| a[i] + sign * b[i]);
    let planes = [add(w, x, 1.), add(w, x, -1.), add(w, y, 1.), add(w, y, -1.), z, add(w, z, -1.)];
    if !planes.iter().flatten().all(|v| v.is_finite()) { return true; }
    !planes.iter().any(|p| {
        let pick = |v: f32, lo: f32, hi: f32| v * if v >= 0. { hi } else { lo };
        let d = pick(p[0], b.min.x, b.max.x) + pick(p[1], b.min.y, b.max.y)
            + pick(p[2], b.min.z, b.max.z) + p[3];
        d.is_finite() && d < -1e-5
    })
}
