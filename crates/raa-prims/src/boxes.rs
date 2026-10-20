//! Box and roundBox tessellation: ports of SDF Modeler's `box(sx, sy, sz)` and `roundBox(sx, sy, sz, radius, cells)`.
//!
//! Like the reference, positions and normals are computed in f64 and stored as f32; zero-area
//! triangles are dropped (`Builder.done`) and each triangle is turned to agree with the sum of
//! its vertex normals (`fixWinding`).

use crate::PrimKind;
use crate::reference::count_rules;
use raa_geom::{GeomError, MeshData};
use std::collections::BTreeMap;

type V3 = [f64; 3];

/// `(n, u, v)` per face, in the reference's order (`leaf-mesh.ts:121-125` and `:154-158`).
const FACES: [[V3; 3]; 6] = [
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
];

/// The reference's `Builder`: f64 vertices, triangle-list indices.
struct Builder {
    pos: Vec<V3>,
    nrm: Vec<V3>,
    idx: Vec<u32>,
}

impl Builder {
    fn new() -> Builder {
        Builder {
            pos: Vec::new(),
            nrm: Vec::new(),
            idx: Vec::new(),
        }
    }

    fn vertex(&mut self, p: V3, n: V3) -> u32 {
        self.pos.push(p);
        self.nrm.push(n);
        // Vertex counts stay far below u32::MAX (at most 6 * 257^2).
        (self.pos.len() - 1) as u32
    }

    /// Quad a b c d (counter-clockwise seen from outside).
    fn quad(&mut self, a: u32, b: u32, c: u32, d: u32) {
        self.idx.extend_from_slice(&[a, b, c, a, c, d]);
    }

    fn grid(&mut self, base: u32, nu: u32, nv: u32) {
        for j in 0..nv {
            for i in 0..nu {
                let a = base + j * (nu + 1) + i;
                let d = a + (nu + 1);
                self.quad(a, a + 1, d + 1, d);
            }
        }
    }

    /// `done()` then `fixWinding()`, written into `out`. With `share`, vertices whose f32
    /// position and normal are bit-identical become one vertex (first occurrence order): the
    /// rounded borders between roundBox faces are smooth, so they share their vertex.
    fn finish(self, share: bool, out: &mut MeshData) {
        let mut seen: BTreeMap<[u32; 6], u32> = BTreeMap::new();
        let mut remap = Vec::with_capacity(self.pos.len());
        for (p, n) in self.pos.iter().zip(&self.nrm) {
            let (p, n) = (p.map(|x| x as f32), n.map(|x| x as f32));
            let key = [p[0], p[1], p[2], n[0], n[1], n[2]].map(f32::to_bits);
            let next = out.positions.len() as u32;
            let id = if share {
                *seen.entry(key).or_insert(next)
            } else {
                next
            };
            if id == next {
                out.positions.push(p);
                out.normals.push(n);
            }
            remap.push(id);
        }
        for t in self.idx.chunks_exact(3) {
            // drop zero-area triangles (on the f64 positions, as the reference does)
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| self.pos[i as usize]);
            if length_squared(cross(sub(b, a), sub(c, a))) > 1e-20 {
                out.indices.extend(t.iter().map(|&i| remap[i as usize]));
            }
        }
        // fixWinding runs on the stored f32 values, evaluated in f64.
        let p = |i: u32| out.positions[i as usize].map(f64::from);
        let n = |i: u32| out.normals[i as usize].map(f64::from);
        let mut flips = Vec::new();
        for (k, t) in out.indices.chunks_exact(3).enumerate() {
            let fnrm = cross(sub(p(t[1]), p(t[0])), sub(p(t[2]), p(t[0])));
            let vn = add(add(n(t[0]), n(t[1])), n(t[2]));
            if dot(fnrm, vn) < 0.0 {
                flips.push(k);
            }
        }
        for k in flips {
            out.indices.swap(3 * k + 1, 3 * k + 2);
        }
    }
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
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

fn length_squared(a: V3) -> f64 {
    dot(a, a)
}

fn hypot3(v: V3) -> f64 {
    v[0].hypot(v[1]).hypot(v[2])
}

/// The reference's `unit` (`[0, 1, 0]` for the zero vector).
fn unit(v: V3) -> V3 {
    let l = hypot3(v);
    if l > 0.0 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// SDF Modeler's `size` (full extents) from S3 `halfExtents`, scale 2 (mapping v1).
fn half_of_size(half_extents: [f32; 3]) -> V3 {
    half_extents.map(|e| 2.0 * f64::from(e) / 2.0)
}

/// roundBox `cells` at clamped `segments`, from the shared count table.
fn round_box_cells(segments: u32) -> u32 {
    count_rules(PrimKind::RoundBox)
        .first()
        .map_or(1, |c| c.count(segments))
}

/// Triangle count `box_generate` emits for the same inputs (README formula).
pub(crate) fn box_triangle_count(half_extents: [f32; 3], segments: u32) -> u32 {
    let _ = (half_extents, segments);
    12
}

/// Writes the box mesh into `out`, which the caller has cleared; `segments` is clamped.
pub(crate) fn box_generate(
    half_extents: [f32; 3],
    segments: u32,
    out: &mut MeshData,
) -> Result<(), GeomError> {
    let _ = segments;
    let h = half_of_size(half_extents);
    let mut b = Builder::new();
    for [n, u, v] in FACES {
        let c = [n[0] * h[0], n[1] * h[1], n[2] * h[2]];
        let at = |a: f64, bb: f64| -> V3 {
            [
                c[0] + u[0] * a * h[0] + v[0] * bb * h[0],
                c[1] + u[1] * a * h[1] + v[1] * bb * h[1],
                c[2] + u[2] * a * h[2] + v[2] * bb * h[2],
            ]
        };
        let i0 = b.vertex(at(-1.0, -1.0), n);
        let i1 = b.vertex(at(1.0, -1.0), n);
        let i2 = b.vertex(at(1.0, 1.0), n);
        let i3 = b.vertex(at(-1.0, 1.0), n);
        b.quad(i0, i1, i2, i3);
    }
    b.finish(false, out);
    Ok(())
}

/// Triangle count `round_box_generate` emits for the same inputs (README formula).
pub(crate) fn round_box_triangle_count(half_extents: [f32; 3], radius: f32, segments: u32) -> u32 {
    let _ = (half_extents, radius);
    let cells = round_box_cells(segments);
    12 * cells * cells
}

/// Writes the roundBox mesh into `out`, which the caller has cleared; `segments` is clamped.
pub(crate) fn round_box_generate(
    half_extents: [f32; 3],
    radius: f32,
    segments: u32,
    out: &mut MeshData,
) -> Result<(), GeomError> {
    let cells = round_box_cells(segments);
    let h = half_of_size(half_extents);
    let r = f64::from(radius).min(h[0]).min(h[1]).min(h[2]).max(0.0);
    let inner = [h[0] - r, h[1] - r, h[2] - r];
    let mut b = Builder::new();
    for [n, u, v] in FACES {
        let base = b.pos.len() as u32;
        for j in 0..=cells {
            for i in 0..=cells {
                let a = -1.0 + (2.0 * f64::from(i)) / f64::from(cells);
                let bb = -1.0 + (2.0 * f64::from(j)) / f64::from(cells);
                let g: V3 = std::array::from_fn(|k| (n[k] + u[k] * a + v[k] * bb) * h[k]);
                // nearest point of the inner box; the normal points from it to the vertex
                let c: V3 = std::array::from_fn(|k| g[k].min(inner[k]).max(-inner[k]));
                let d = sub(g, c);
                let nn = if hypot3(d) > 1e-9 { unit(d) } else { n };
                b.vertex(std::array::from_fn(|k| c[k] + nn[k] * r), nn);
            }
        }
        b.grid(base, cells, cells);
    }
    b.finish(true, out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_geom::analysis;

    const SEGMENTS: [u32; 5] = [3, 8, 24, 64, 256];
    const SETS: [([f32; 3], f32); 2] = [([0.7, 0.6, 0.5], 0.15), ([1.0, 1.0, 0.4], 0.35)];
    const BOX_SETS: [[f32; 3]; 2] = [[0.7, 0.6, 0.5], [1.2, 0.3, 2.0]];

    #[test]
    fn box_triangle_count_matches_generated_count() {
        for he in BOX_SETS {
            for s in SEGMENTS {
                let mut out = MeshData::new();
                box_generate(he, s, &mut out).unwrap();
                assert_eq!(
                    box_triangle_count(he, s) as usize,
                    out.triangle_count(),
                    "{he:?} s {s}"
                );
                assert_eq!(out.vertex_count(), 24);
            }
        }
    }

    #[test]
    fn round_box_triangle_count_matches_generated_count() {
        for (he, r) in SETS {
            for s in SEGMENTS {
                let mut out = MeshData::new();
                round_box_generate(he, r, s, &mut out).unwrap();
                let n = round_box_triangle_count(he, r, s) as usize;
                assert_eq!(n, out.triangle_count(), "{he:?} r {r} s {s}");
            }
        }
    }

    #[test]
    fn both_kinds_are_closed_outward_and_pass_the_normal_rule() {
        for ((he, r), bhe) in SETS.into_iter().zip(BOX_SETS) {
            for s in [3, 8, 24, 64] {
                let mut out = MeshData::new();
                box_generate(bhe, s, &mut out).unwrap();
                analysis::passes(&analysis::analyze(&out)).unwrap();
                out.clear();
                round_box_generate(he, r, s, &mut out).unwrap();
                analysis::passes(&analysis::analyze(&out)).unwrap();
            }
        }
    }

    #[test]
    fn round_box_radius_zero_is_a_closed_plain_box() {
        for s in [3, 8, 24] {
            let mut out = MeshData::new();
            round_box_generate([0.7, 0.6, 0.5], 0.0, s, &mut out).unwrap();
            let b = out.bounds();
            assert_eq!(
                (b.min.x, b.max.x, b.min.y, b.max.y),
                (-0.7, 0.7, -0.6, 0.6),
                "s {s}"
            );
            assert_eq!(analysis::passes(&analysis::analyze(&out)), Ok(()), "s {s}");
        }
    }

    #[test]
    fn round_box_lod0_matches_reference_counts() {
        let mut out = MeshData::new();
        round_box_generate([0.7, 0.6, 0.5], 0.15, 24, &mut out).unwrap();
        assert_eq!(out.triangle_count(), 1200);
        assert_eq!(analysis::unique_positions_1e6(&out), 602);
    }
}
