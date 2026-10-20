//! Finishing step of every boolean: fan triangulation, deterministic CSG_EPS weld, T-junction
//! repair and the watertight verdict. All iteration orders are fixed (index order, BTreeMap).

use std::collections::BTreeMap;

use raa_geom::{MeshData, analysis};

use crate::{CsgError, Poly};

/// Maximum number of T-junction repair passes.
const TJUNCTION_PASSES: usize = 3;
/// Normals are keyed after quantisation to this step.
const NORMAL_QUANTUM: f64 = 1e-6;

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dist2(a: V3, b: V3) -> f64 {
    let d = sub(a, b);
    dot(d, d)
}

/// Welded positions on a quantised grid of cell `eps`: a position snaps to the first-inserted
/// representative within `eps` (searching the 27 neighbouring cells), else becomes a new one.
struct Welder {
    eps: f64,
    grid: BTreeMap<[i64; 3], Vec<u32>>,
    positions: Vec<V3>,
    /// (welded position, quantised normal) to vertex index.
    vertex_map: BTreeMap<(u32, [i64; 3]), u32>,
    /// Per vertex: welded position index and normal.
    vertices: Vec<(u32, V3)>,
}

impl Welder {
    fn new(eps: f64) -> Self {
        Welder {
            eps,
            grid: BTreeMap::new(),
            positions: Vec::new(),
            vertex_map: BTreeMap::new(),
            vertices: Vec::new(),
        }
    }

    fn cell(&self, p: V3) -> [i64; 3] {
        // `as` saturates for out-of-range floats; inputs are checked finite beforehand.
        [
            (p[0] / self.eps).floor() as i64,
            (p[1] / self.eps).floor() as i64,
            (p[2] / self.eps).floor() as i64,
        ]
    }

    fn weld_position(&mut self, p: V3) -> u32 {
        let c = self.cell(p);
        let eps2 = self.eps * self.eps;
        let mut best: Option<u32> = None;
        for dx in -1i64..=1 {
            for dy in -1i64..=1 {
                for dz in -1i64..=1 {
                    let key = [
                        c[0].saturating_add(dx),
                        c[1].saturating_add(dy),
                        c[2].saturating_add(dz),
                    ];
                    let Some(list) = self.grid.get(&key) else {
                        continue;
                    };
                    for &i in list {
                        // Invariant: grid ids index `positions`.
                        if dist2(self.positions[i as usize], p) <= eps2
                            && best.is_none_or(|b| i < b)
                        {
                            best = Some(i);
                        }
                    }
                }
            }
        }
        if let Some(i) = best {
            return i;
        }
        let id = self.positions.len() as u32;
        self.positions.push(p);
        self.grid.entry(c).or_default().push(id);
        id
    }

    fn vertex(&mut self, pos: u32, n: V3) -> u32 {
        let q = n.map(|c| (c / NORMAL_QUANTUM).round() as i64);
        if let Some(&v) = self.vertex_map.get(&(pos, q)) {
            return v;
        }
        let v = self.vertices.len() as u32;
        self.vertices.push((pos, n));
        self.vertex_map.insert((pos, q), v);
        v
    }
}

fn normalised_or(n: V3, fallback: V3) -> V3 {
    let len = dot(n, n).sqrt();
    if len > 1e-12 && len.is_finite() {
        [n[0] / len, n[1] / len, n[2] / len]
    } else {
        fallback
    }
}

/// Welded position ids sorted by x (ties by id), for edge range queries.
fn sorted_by_x(positions: &[V3]) -> Vec<(f64, u32)> {
    let mut v: Vec<(f64, u32)> = positions
        .iter()
        .enumerate()
        .map(|(i, p)| (p[0], i as u32))
        .collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    v
}

/// Welded positions strictly inside the segment `a`-`b` (distance <= eps, parameter in
/// (0, 1)), excluding `skip`, sorted by parameter then id.
fn inner_points(
    w: &Welder,
    by_x: &[(f64, u32)],
    a: u32,
    b: u32,
    skip: u32,
    found: &mut Vec<(f64, u32)>,
) {
    found.clear();
    let pa = w.positions[a as usize];
    let pb = w.positions[b as usize];
    let d = sub(pb, pa);
    let len2 = dot(d, d);
    if len2 <= 0.0 {
        return;
    }
    let eps = w.eps;
    let lo = pa[0].min(pb[0]) - eps;
    let hi = pa[0].max(pb[0]) + eps;
    let start = by_x.partition_point(|e| e.0 < lo);
    for &(x, i) in &by_x[start..] {
        if x > hi {
            break;
        }
        if i == a || i == b || i == skip {
            continue;
        }
        let p = w.positions[i as usize];
        let t = dot(sub(p, pa), d) / len2;
        if !(t > 0.0 && t < 1.0) {
            continue;
        }
        let closest = [pa[0] + t * d[0], pa[1] + t * d[1], pa[2] + t * d[2]];
        if dist2(p, closest) <= eps * eps {
            found.push((t, i));
        }
    }
    found.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
}

/// One T-junction pass: each triangle is split along its first edge (in corner order) that
/// has welded positions strictly inside it. Returns whether anything was split.
fn repair_pass(w: &mut Welder, tris: &mut Vec<[u32; 3]>) -> bool {
    let by_x = sorted_by_x(&w.positions);
    let mut out: Vec<[u32; 3]> = Vec::with_capacity(tris.len());
    let mut found = Vec::new();
    let mut chain: Vec<u32> = Vec::new();
    let mut split_any = false;
    for &tri in tris.iter() {
        let mut split = false;
        for e in 0..3 {
            let va = tri[e];
            let vb = tri[(e + 1) % 3];
            let vo = tri[(e + 2) % 3];
            // Invariant: triangle ids index `vertices`.
            let (pa, na) = w.vertices[va as usize];
            let (pb, nb) = w.vertices[vb as usize];
            let (po, _) = w.vertices[vo as usize];
            inner_points(w, &by_x, pa, pb, po, &mut found);
            if found.is_empty() {
                continue;
            }
            chain.clear();
            chain.push(va);
            for &(t, p) in &found {
                let n = [
                    na[0] * (1.0 - t) + nb[0] * t,
                    na[1] * (1.0 - t) + nb[1] * t,
                    na[2] * (1.0 - t) + nb[2] * t,
                ];
                let v = w.vertex(p, normalised_or(n, na));
                chain.push(v);
            }
            chain.push(vb);
            for k in 0..chain.len() - 1 {
                out.push([chain[k], chain[k + 1], vo]);
            }
            split = true;
            break;
        }
        if split {
            split_any = true;
        } else {
            out.push(tri);
        }
    }
    *tris = out;
    split_any
}

/// Finishes a boolean: triangulates `polys`, welds, repairs T-junctions, drops degenerate
/// triangles, writes `out` (cleared first) and refuses anything not closed with positive
/// volume. Never relaxes `eps`.
#[allow(dead_code)] // called by clip.rs once the clipping task lands
pub(crate) fn finish(polys: &[Poly], eps: f64, out: &mut MeshData) -> Result<(), CsgError> {
    out.clear();
    if !(eps > 0.0 && eps.is_finite()) {
        return Err(CsgError::Degenerate(format!("invalid weld eps {eps}")));
    }
    let mut w = Welder::new(eps);
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let area_min = eps * eps;
    for (pi, poly) in polys.iter().enumerate() {
        if poly.pos.len() != poly.nrm.len() {
            return Err(CsgError::Degenerate(format!(
                "polygon {pi} has {} positions and {} normals",
                poly.pos.len(),
                poly.nrm.len()
            )));
        }
        let finite = poly
            .pos
            .iter()
            .chain(poly.nrm.iter())
            .flatten()
            .all(|c| c.is_finite());
        if !finite {
            return Err(CsgError::Degenerate(format!(
                "polygon {pi} has a non-finite value"
            )));
        }
        let Some((&p0, rest)) = poly.pos.split_first() else {
            continue;
        };
        for k in 1..rest.len() {
            let p1 = poly.pos[k];
            let p2 = poly.pos[k + 1];
            let c = cross(sub(p1, p0), sub(p2, p0));
            let area = 0.5 * dot(c, c).sqrt();
            if area <= area_min {
                continue;
            }
            let mut tri = [0u32; 3];
            for (slot, idx) in tri.iter_mut().zip([0, k, k + 1]) {
                let pos = w.weld_position(poly.pos[idx]);
                *slot = w.vertex(pos, poly.nrm[idx]);
            }
            tris.push(tri);
        }
    }

    for _ in 0..TJUNCTION_PASSES {
        if !repair_pass(&mut w, &mut tris) {
            break;
        }
    }

    let verts = &w.vertices;
    tris.retain(|t| {
        let p = t.map(|v| verts[v as usize].0);
        p[0] != p[1] && p[1] != p[2] && p[0] != p[2]
    });
    if tris.is_empty() {
        return Err(CsgError::EmptyResult);
    }

    // Compact: only vertices referenced by surviving triangles, in first-use order.
    let mut remap = vec![u32::MAX; verts.len()];
    for t in &tris {
        let mut tri_out = [0u32; 3];
        for (slot, &v) in tri_out.iter_mut().zip(t) {
            let r = &mut remap[v as usize];
            if *r == u32::MAX {
                *r = out.positions.len() as u32;
                let (pos, n) = verts[v as usize];
                let p = w.positions[pos as usize];
                out.positions.push([p[0] as f32, p[1] as f32, p[2] as f32]);
                out.normals.push([n[0] as f32, n[1] as f32, n[2] as f32]);
            }
            *slot = *r;
        }
        out.indices.extend_from_slice(&tri_out);
    }

    let report = analysis::analyze(out);
    if report.closed && report.volume > 0.0 {
        Ok(())
    } else {
        out.clear();
        Err(CsgError::NotWatertight {
            boundary_edges: report.boundary_edges,
            nonmanifold_edges: report.nonmanifold_edges,
            inconsistent_edges: report.inconsistent_edges,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(pos: [V3; 4], n: V3) -> Poly {
        Poly {
            pos: pos.to_vec(),
            nrm: vec![n; 4],
        }
    }

    /// Unit cube [0,1]^3 as CCW-outward quads; `split_top` splits +Z into two quads at x = 0.5.
    fn cube(split_top: bool, skip_bottom: bool) -> Vec<Poly> {
        let mut v = Vec::new();
        if split_top {
            v.push(quad(
                [
                    [0.0, 0.0, 1.0],
                    [0.5, 0.0, 1.0],
                    [0.5, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                [0.0, 0.0, 1.0],
            ));
            v.push(quad(
                [
                    [0.5, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.5, 1.0, 1.0],
                ],
                [0.0, 0.0, 1.0],
            ));
        } else {
            v.push(quad(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                [0.0, 0.0, 1.0],
            ));
        }
        if !skip_bottom {
            v.push(quad(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                [0.0, 0.0, -1.0],
            ));
        }
        v.push(quad(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
            ],
            [1.0, 0.0, 0.0],
        ));
        v.push(quad(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            [-1.0, 0.0, 0.0],
        ));
        v.push(quad(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            [0.0, 1.0, 0.0],
        ));
        v.push(quad(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            [0.0, -1.0, 0.0],
        ));
        v
    }

    #[test]
    fn weld_unit_cube_gives_12_closed_triangles_of_volume_1() {
        let mut out = MeshData::new();
        finish(&cube(false, false), 1e-6, &mut out).expect("cube is closed");
        assert_eq!(out.triangle_count(), 12);
        assert_eq!(out.vertex_count(), 24, "hard edges keep per-face normals");
        let r = analysis::analyze(&out);
        assert!(r.closed);
        assert!((r.volume - 1.0).abs() <= 1e-9, "volume {}", r.volume);
    }

    #[test]
    fn weld_repairs_t_junction_of_split_face() {
        let mut out = MeshData::new();
        finish(&cube(true, false), 1e-6, &mut out).expect("T-junction is repaired");
        let r = analysis::analyze(&out);
        assert!(r.closed);
        assert!(out.triangle_count() > 12);
        assert!((r.volume - 1.0).abs() <= 1e-9, "volume {}", r.volume);
    }

    #[test]
    fn weld_open_cube_is_not_watertight() {
        let mut out = MeshData::new();
        match finish(&cube(false, true), 1e-6, &mut out) {
            Err(CsgError::NotWatertight { boundary_edges, .. }) => {
                assert!(boundary_edges > 0)
            }
            other => panic!("expected NotWatertight, got {other:?}"),
        }
        assert_eq!(out.triangle_count(), 0, "no silent output");
    }

    #[test]
    fn weld_empty_input_is_empty_result() {
        let mut out = MeshData::new();
        assert_eq!(finish(&[], 1e-6, &mut out), Err(CsgError::EmptyResult));
    }

    #[test]
    fn weld_snaps_positions_within_eps_to_first_occurrence() {
        let mut polys = cube(true, false);
        // Nudge one duplicate corner by less than eps: the weld must snap it back.
        polys[3].pos[3] = [1.0 + 4e-7, 0.0, 1.0];
        let mut out = MeshData::new();
        finish(&polys, 1e-6, &mut out).expect("closed after weld");
        assert!(out.positions.iter().all(|p| p[0] <= 1.0));
    }

    #[test]
    fn weld_runs_are_byte_identical() {
        let mut a = MeshData::new();
        let mut b = MeshData::new();
        finish(&cube(true, false), 1e-6, &mut a).expect("closed");
        finish(&cube(true, false), 1e-6, &mut b).expect("closed");
        assert_eq!(analysis::mesh_hash(&a), analysis::mesh_hash(&b));
        assert_eq!(a, b);
    }
}
