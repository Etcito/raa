//! Load-time Delaunay triangulation and hull of validated 2D blend points (S19).
//! Candidate triangles are visited in ascending source-index order and every
//! predicate uses source-ordered f32 arithmetic, so equal inputs give equal topology.

use crate::error::{GraphError, codes};

type P = [f32; 2];

/// Counter-clockwise triangles (smallest source index first) in acceptance order,
/// and the counter-clockwise hull cycle starting at its smallest source index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Triangulation {
    pub triangles: Vec<[usize; 3]>,
    pub hull: Vec<[usize; 2]>,
}

/// Twice the signed area of `a b c`; positive when counter-clockwise.
fn orient(a: P, b: P, c: P) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Positive when `m` lies strictly inside the circumcircle of the CCW triangle `a b c`.
fn incircle(a: P, b: P, c: P, m: P) -> f32 {
    let (adx, ady) = (a[0] - m[0], a[1] - m[1]);
    let (bdx, bdy) = (b[0] - m[0], b[1] - m[1]);
    let (cdx, cdy) = (c[0] - m[0], c[1] - m[1]);
    (adx * adx + ady * ady) * (bdx * cdy - bdy * cdx)
        + (bdx * bdx + bdy * bdy) * (cdx * ady - cdy * adx)
        + (cdx * cdx + cdy * cdy) * (adx * bdy - ady * bdx)
}

/// True when the CCW triangle `t` contains `q` in its interior or on its boundary.
fn covers(t: [P; 3], q: P) -> bool {
    (0..3).all(|e| orient(t[e], t[(e + 1) % 3], q) >= 0.0)
}

/// True when an edge line of `a` leaves every vertex of `b` on its closed outer side.
fn separated(a: [P; 3], b: [P; 3]) -> bool {
    (0..3).any(|e| b.iter().all(|&v| orient(a[e], a[(e + 1) % 3], v) <= 0.0))
}

/// Triangulate distinct, finite points spanning an area; other sets are refused
/// with `graph.blend_space` at `path`, as reference validation refuses them.
/// Cocircular points keep the earliest non-overlapping triangles by source index.
pub fn triangulate(points: &[P], path: &str) -> Result<Triangulation, GraphError> {
    let fail = |message: &str| GraphError::new(codes::BLEND_SPACE, path, message);
    if points.len() < 3 {
        return Err(fail("2D blend spaces need three points"));
    }
    for (i, p) in points.iter().enumerate() {
        if !(p[0].is_finite() && p[1].is_finite()) || points[..i].contains(p) {
            return Err(fail("2D points must be finite and distinct"));
        }
    }
    let n = points.len();
    let at = |t: [usize; 3]| t.map(|i| points[i]);
    // Each non-degenerate triple that covers no other point, flagged when its circumcircle is empty.
    let mut candidates = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                let o = orient(points[i], points[j], points[k]);
                if !o.is_finite() {
                    return Err(fail("geometry overflows"));
                }
                if o == 0.0 {
                    continue;
                }
                let t = if o > 0.0 { [i, j, k] } else { [i, k, j] };
                let (v, mut empty, mut covered) = (at(t), true, false);
                for (m, &q) in points.iter().enumerate() {
                    if t.contains(&m) {
                        continue;
                    }
                    let d = incircle(v[0], v[1], v[2], q);
                    if !d.is_finite() {
                        return Err(fail("geometry overflows"));
                    }
                    empty &= d <= 0.0;
                    covered |= covers(v, q);
                }
                if !covered {
                    candidates.push((t, empty));
                }
            }
        }
    }
    // Empty-circle triangles first; the second pass only fills gaps left by f32 rounding.
    let mut triangles: Vec<[usize; 3]> = Vec::new();
    for delaunay in [true, false] {
        for &(t, empty) in &candidates {
            let v = at(t);
            let free = triangles
                .iter()
                .all(|&u| separated(v, at(u)) || separated(at(u), v));
            if empty == delaunay && free {
                triangles.push(t);
            }
        }
    }
    let edges: Vec<[usize; 2]> = triangles
        .iter()
        .flat_map(|t| [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]])
        .collect();
    let open: Vec<[usize; 2]> = edges
        .iter()
        .copied()
        .filter(|&[a, b]| !edges.contains(&[b, a]))
        .collect();
    let Some(&start) = open.iter().min() else {
        return Err(fail("2D points must span an area"));
    };
    let mut hull = vec![start];
    while hull.len() < open.len() {
        let end = hull[hull.len() - 1][1];
        match open.iter().find(|e| e[0] == end) {
            Some(&e) if !hull.contains(&e) => hull.push(e),
            _ => return Err(fail("triangulation failed")),
        }
    }
    // A closed hull and Euler's count prove that every point is a vertex of a full triangulation.
    if hull[hull.len() - 1][1] != start[0] || triangles.len() + hull.len() + 2 != 2 * n {
        return Err(fail("triangulation failed"));
    }
    Ok(Triangulation { triangles, hull })
}
