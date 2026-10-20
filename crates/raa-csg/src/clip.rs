//! Convex clipping: exact plane splitting shared by intersect and subtract, the coplanar rule,
//! and `intersect_convex`. All iteration orders are fixed (faces in index order, planes in
//! `convex_planes` order).

use std::cmp::Ordering;

use raa_geom::MeshData;

use crate::plane::coincident;
use crate::{CsgError, Plane, Poly, bbox_diagonal_union, convex_planes, csg_eps, weld};

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn to_f64(p: [f32; 3]) -> V3 {
    [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]
}

fn lex_cmp(a: V3, b: V3) -> Ordering {
    a[0].total_cmp(&b[0])
        .then(a[1].total_cmp(&b[1]))
        .then(a[2].total_cmp(&b[2]))
}

/// One Poly per triangle, positions and normals widened to f64. Triangles with an index out
/// of range are skipped (callers validate the mesh first).
pub(crate) fn polys_from_mesh(mesh: &MeshData) -> Vec<Poly> {
    let mut polys = Vec::with_capacity(mesh.indices.len() / 3);
    for t in mesh.indices.chunks_exact(3) {
        let mut pos = Vec::with_capacity(3);
        let mut nrm = Vec::with_capacity(3);
        for &i in t {
            let (Some(p), Some(n)) = (mesh.positions.get(i as usize), mesh.normals.get(i as usize))
            else {
                break;
            };
            pos.push(to_f64(*p));
            nrm.push(to_f64(*n));
        }
        if pos.len() == 3 {
            polys.push(Poly { pos, nrm });
        }
    }
    polys
}

/// Crossing of edge `a`-`b` with `plane`, computed from the endpoints in lexicographic order
/// so that both faces sharing the edge get bit-identical points. The normal is the lerp of the
/// endpoint normals, renormalised.
fn crossing(plane: &Plane, a: (V3, V3), b: (V3, V3)) -> (V3, V3) {
    let (a, b) = if lex_cmp(a.0, b.0) == Ordering::Greater {
        (b, a)
    } else {
        (a, b)
    };
    let sa = plane.signed(a.0);
    let sb = plane.signed(b.0);
    let t = sa / (sa - sb);
    let lerp = |x: V3, y: V3| -> V3 {
        [
            x[0] + (y[0] - x[0]) * t,
            x[1] + (y[1] - x[1]) * t,
            x[2] + (y[2] - x[2]) * t,
        ]
    };
    let p = lerp(a.0, b.0);
    let n = lerp(a.1, b.1);
    let len = dot(n, n).sqrt();
    let n = if len > 1e-12 && len.is_finite() {
        [n[0] / len, n[1] / len, n[2] / len]
    } else {
        a.1
    };
    (p, n)
}

/// Keeps the polygon only when it has at least 3 vertices pairwise farther apart than `eps`.
fn keep(poly: Poly, eps: f64) -> Option<Poly> {
    let eps2 = eps * eps;
    let mut distinct: Vec<V3> = Vec::with_capacity(3);
    for &p in &poly.pos {
        if distinct.iter().all(|&q| {
            let d = sub(p, q);
            dot(d, d) > eps2
        }) {
            distinct.push(p);
            if distinct.len() >= 3 {
                return Some(poly);
            }
        }
    }
    None
}

#[derive(Clone, Copy, PartialEq)]
enum Side {
    In,
    Out,
    On,
}

/// Splits a convex polygon by `plane` into its inside (`signed <= eps`) and outside parts.
/// Vertices within `eps` of the plane are ON and go to both sides. Pieces with fewer than 3
/// distinct vertices are dropped.
pub(crate) fn split(poly: &Poly, plane: &Plane, eps: f64) -> (Option<Poly>, Option<Poly>) {
    let n = poly.pos.len();
    let sides: Vec<Side> = poly
        .pos
        .iter()
        .map(|&p| {
            let s = plane.signed(p);
            if s > eps {
                Side::Out
            } else if s < -eps {
                Side::In
            } else {
                Side::On
            }
        })
        .collect();
    if !sides.contains(&Side::Out) {
        return (keep(poly.clone(), eps), None);
    }
    if !sides.contains(&Side::In) {
        return (None, keep(poly.clone(), eps));
    }
    let mut inside = Poly {
        pos: Vec::with_capacity(n + 1),
        nrm: Vec::with_capacity(n + 1),
    };
    let mut outside = inside.clone();
    for i in 0..n {
        let j = (i + 1) % n;
        let (pi, ni, si) = (poly.pos[i], poly.nrm[i], sides[i]);
        if si != Side::Out {
            inside.pos.push(pi);
            inside.nrm.push(ni);
        }
        if si != Side::In {
            outside.pos.push(pi);
            outside.nrm.push(ni);
        }
        let sj = sides[j];
        if (si == Side::In && sj == Side::Out) || (si == Side::Out && sj == Side::In) {
            let (p, nn) = crossing(plane, (pi, ni), (poly.pos[j], poly.nrm[j]));
            inside.pos.push(p);
            inside.nrm.push(nn);
            outside.pos.push(p);
            outside.nrm.push(nn);
        }
    }
    (keep(inside, eps), keep(outside, eps))
}

/// Indices of the polygons lying on `plane` (every vertex within `eps`).
fn polys_on(polys: &[Poly], plane: &Plane, eps: f64) -> Vec<usize> {
    (0..polys.len())
        .filter(|&i| polys[i].pos.iter().all(|&p| plane.signed(p).abs() <= eps))
        .collect()
}

/// 2D projection dropping the dominant axis of `n`.
fn project(p: V3, n: V3) -> [f64; 2] {
    let a = [n[0].abs(), n[1].abs(), n[2].abs()];
    if a[0] >= a[1] && a[0] >= a[2] {
        [p[1], p[2]]
    } else if a[1] >= a[2] {
        [p[2], p[0]]
    } else {
        [p[0], p[1]]
    }
}

fn range(pts: &[[f64; 2]], axis: [f64; 2]) -> (f64, f64) {
    pts.iter()
        .map(|p| p[0] * axis[0] + p[1] * axis[1])
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        })
}

/// Separating-axis test of two convex 2D polygons: they overlap unless an edge normal
/// separates them (touching within `eps` counts as separated).
fn overlap_2d(a: &[[f64; 2]], b: &[[f64; 2]], eps: f64) -> bool {
    for poly in [a, b] {
        for i in 0..poly.len() {
            let p = poly[i];
            let q = poly[(i + 1) % poly.len()];
            let e = [q[0] - p[0], q[1] - p[1]];
            let len = (e[0] * e[0] + e[1] * e[1]).sqrt();
            if len.is_nan() || len <= 0.0 {
                continue;
            }
            let axis = [-e[1] / len, e[0] / len];
            let (alo, ahi) = range(a, axis);
            let (blo, bhi) = range(b, axis);
            if ahi <= blo + eps || bhi <= alo + eps {
                return false;
            }
        }
    }
    true
}

/// The coplanar rule: any tool plane coincident with a host plane whose faces overlap the
/// host faces on that plane gives `Coplanar`, naming both planes.
pub(crate) fn check_coplanar(
    host_polys: &[Poly],
    host_planes: &[Plane],
    tool_polys: &[Poly],
    tool_planes: &[Plane],
    eps: f64,
) -> Result<(), CsgError> {
    for hp in host_planes {
        let mut host_on: Option<Vec<Vec<[f64; 2]>>> = None;
        for tp in tool_planes {
            if !coincident(hp, tp, eps) {
                continue;
            }
            let host_2d = host_on.get_or_insert_with(|| {
                polys_on(host_polys, hp, eps)
                    .into_iter()
                    .map(|i| {
                        host_polys[i]
                            .pos
                            .iter()
                            .map(|&p| project(p, hp.n))
                            .collect()
                    })
                    .collect()
            });
            for ti in polys_on(tool_polys, tp, eps) {
                let t2: Vec<[f64; 2]> = tool_polys[ti]
                    .pos
                    .iter()
                    .map(|&p| project(p, hp.n))
                    .collect();
                if host_2d.iter().any(|h| overlap_2d(h, &t2, eps)) {
                    return Err(CsgError::Coplanar {
                        detail: format!(
                            "tool plane n=({:.6}, {:.6}, {:.6}) d={:.6} coincides with host \
                             plane n=({:.6}, {:.6}, {:.6}) d={:.6} and their faces overlap \
                             (eps {eps:e})",
                            tp.n[0], tp.n[1], tp.n[2], tp.d, hp.n[0], hp.n[1], hp.n[2], hp.d
                        ),
                    });
                }
            }
        }
    }
    Ok(())
}

/// True when every vertex of `mesh` is inside (within `eps`) every plane.
pub(crate) fn inside_all(mesh: &MeshData, planes: &[Plane], eps: f64) -> bool {
    mesh.positions
        .iter()
        .all(|&p| planes.iter().all(|pl| pl.signed(to_f64(p)) <= eps))
}

/// Strictly separated bounding boxes (gap larger than `eps` on some axis).
pub(crate) fn bboxes_disjoint(a: &MeshData, b: &MeshData, eps: f64) -> bool {
    let (ba, bb) = (a.bounds(), b.bounds());
    let (amin, amax) = (ba.min.to_array(), ba.max.to_array());
    let (bmin, bmax) = (bb.min.to_array(), bb.max.to_array());
    (0..3).any(|k| {
        f64::from(amin[k]) > f64::from(bmax[k]) + eps
            || f64::from(bmin[k]) > f64::from(amax[k]) + eps
    })
}

pub(crate) fn copy_into(src: &MeshData, out: &mut MeshData) {
    out.clear();
    out.positions.extend_from_slice(&src.positions);
    out.normals.extend_from_slice(&src.normals);
    out.indices.extend_from_slice(&src.indices);
}

/// Clips `poly` to the inside of every plane, in order.
fn clip_inside(poly: &Poly, planes: &[Plane], eps: f64) -> Option<Poly> {
    let mut cur = poly.clone();
    for pl in planes {
        cur = split(&cur, pl, eps).0?;
    }
    Some(cur)
}

/// Intersection of two convex meshes: host faces clipped to the inside of every tool plane,
/// plus tool faces clipped to the inside of every host plane (original winding).
pub fn intersect_convex(
    host: &MeshData,
    tool: &MeshData,
    out: &mut MeshData,
) -> Result<(), CsgError> {
    host.validate()
        .map_err(|e| CsgError::Degenerate(format!("host mesh: {e}")))?;
    tool.validate()
        .map_err(|e| CsgError::Degenerate(format!("tool mesh: {e}")))?;
    let eps = csg_eps(bbox_diagonal_union(&[host, tool]));
    let host_planes = convex_planes(host, eps).map_err(|_| CsgError::HostNotConvex {
        kind: "mesh".into(),
    })?;
    let tool_planes = convex_planes(tool, eps).map_err(|_| CsgError::ToolNotConvex {
        kind: "mesh".into(),
    })?;
    if host.indices.is_empty() || tool.indices.is_empty() || bboxes_disjoint(host, tool, eps) {
        return Err(CsgError::EmptyResult);
    }
    let host_polys = polys_from_mesh(host);
    let tool_polys = polys_from_mesh(tool);
    check_coplanar(&host_polys, &host_planes, &tool_polys, &tool_planes, eps)?;
    if inside_all(host, &tool_planes, eps) {
        copy_into(host, out);
        return Ok(());
    }
    if inside_all(tool, &host_planes, eps) {
        copy_into(tool, out);
        return Ok(());
    }
    let mut pieces: Vec<Poly> = Vec::new();
    pieces.extend(
        host_polys
            .iter()
            .filter_map(|p| clip_inside(p, &tool_planes, eps)),
    );
    pieces.extend(
        tool_polys
            .iter()
            .filter_map(|p| clip_inside(p, &host_planes, eps)),
    );
    if pieces.is_empty() {
        return Err(CsgError::EmptyResult);
    }
    weld::finish(&pieces, eps, out)
}

/// True when the bounding boxes of `a` and `b` intersect (`Aabb::intersects` in f32, `a`
/// grown by `eps`).
fn bboxes_touch(a: &MeshData, b: &MeshData, eps: f64) -> bool {
    let mut ba = a.bounds();
    let g = raa_math::Vec3::splat(eps as f32);
    ba.min -= g;
    ba.max += g;
    ba.intersects(&b.bounds())
}

/// The fragments of `poly` outside the convex region bounded by `planes`: for each plane in
/// order the outside part is kept and the inside part continues; what remains after the last
/// plane is inside and dropped. Only the planes carrying an edge of the hole (`poly` clipped
/// to the inside of every plane) split: the others do not change the result within the face,
/// and their near-parallel chords would only add grazing vertices. A polygon without a hole
/// is kept whole.
fn outside_parts(poly: Poly, planes: &[Plane], eps: f64, out: &mut Vec<Poly>) {
    let Some(hole) = clip_inside(&poly, planes, eps) else {
        out.push(poly);
        return;
    };
    let n = hole.pos.len();
    let on = |pl: &Plane, p: V3| pl.signed(p).abs() <= eps;
    let mut remaining = poly;
    for pl in planes {
        let bounds_hole = (0..n).any(|i| on(pl, hole.pos[i]) && on(pl, hole.pos[(i + 1) % n]));
        if !bounds_hole {
            continue;
        }
        let (inside, outside) = split(&remaining, pl, eps);
        out.extend(outside);
        match inside {
            Some(p) => remaining = p,
            None => return,
        }
    }
}

/// Convex host minus convex tools with pairwise disjoint bounding boxes: host faces keep their
/// parts outside every tool, tool faces clipped to the inside of every host plane are emitted
/// with reversed winding and negated normals.
pub fn subtract_convex(
    host: &MeshData,
    host_planes: &[Plane],
    tools: &[&MeshData],
    out: &mut MeshData,
) -> Result<(), CsgError> {
    host.validate()
        .map_err(|e| CsgError::Degenerate(format!("host mesh: {e}")))?;
    for (i, t) in tools.iter().enumerate() {
        t.validate()
            .map_err(|e| CsgError::Degenerate(format!("tool#{i} mesh: {e}")))?;
    }
    let mut all: Vec<&MeshData> = vec![host];
    all.extend(tools.iter().copied());
    let eps = csg_eps(bbox_diagonal_union(&all));
    for (i, a) in tools.iter().enumerate() {
        for (j, b) in tools.iter().enumerate().skip(i + 1) {
            if bboxes_touch(a, b, eps) {
                return Err(CsgError::OverlappingTools {
                    a: format!("tool#{i}"),
                    b: format!("tool#{j}"),
                });
            }
        }
    }
    let mut tool_planes: Vec<Vec<Plane>> = Vec::with_capacity(tools.len());
    for (i, t) in tools.iter().enumerate() {
        tool_planes.push(convex_planes(t, eps).map_err(|_| CsgError::ToolNotConvex {
            kind: format!("tool#{i}"),
        })?);
    }
    let active: Vec<usize> = (0..tools.len())
        .filter(|&i| !tools[i].indices.is_empty() && !bboxes_disjoint(host, tools[i], eps))
        .collect();
    if active.is_empty() {
        copy_into(host, out);
        return Ok(());
    }
    let host_polys = polys_from_mesh(host);
    let tool_polys: Vec<Vec<Poly>> = tools.iter().map(|t| polys_from_mesh(t)).collect();
    for &i in &active {
        check_coplanar(
            &host_polys,
            host_planes,
            &tool_polys[i],
            &tool_planes[i],
            eps,
        )?;
        if inside_all(host, &tool_planes[i], eps) {
            return Err(CsgError::EmptyResult);
        }
    }
    let mut pieces: Vec<Poly> = Vec::new();
    for poly in &host_polys {
        let mut frags = vec![poly.clone()];
        for &i in &active {
            let mut next = Vec::with_capacity(frags.len());
            for f in frags {
                outside_parts(f, &tool_planes[i], eps, &mut next);
            }
            frags = next;
        }
        pieces.extend(frags);
    }
    for &i in &active {
        for poly in &tool_polys[i] {
            if let Some(mut p) = clip_inside(poly, host_planes, eps) {
                p.pos.reverse();
                p.nrm.reverse();
                for n in &mut p.nrm {
                    *n = [-n[0], -n[1], -n[2]];
                }
                pieces.push(p);
            }
        }
    }
    if pieces.is_empty() {
        return Err(CsgError::EmptyResult);
    }
    weld::finish(&pieces, eps, out)
}
