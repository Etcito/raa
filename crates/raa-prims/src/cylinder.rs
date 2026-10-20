//! Cylinder tessellation: port of SDF Modeler's `frustum(height, r, r, nu)`.
//!
//! The shared [`frustum`] port (also used by the cone) follows `leaf-mesh.ts:196-217@50b45d5`:
//! two side rings of `nu + 1` vertices (seam duplicated) at y = -h and +h with the slant
//! normal `unit(cos, slope, sin)`, then one flat cap per non-zero radius: a centre vertex and
//! its own ring of `nu + 1` rim vertices with normal ±Y (hard edge, S2). The reference's
//! `done()` drops zero-area triangles and `fixWinding` flips inward ones; here the zero-area
//! side triangles of a pointed end (r = 0) are skipped structurally and every triangle is
//! emitted already outward, which gives the same index list for valid parameters.

use raa_geom::{GeomError, MeshData};

use crate::PrimKind;
use crate::reference::count_rules;

/// The azimuth count `nu` of `kind` (cylinder or cone) at `segments`, from the count table.
pub(crate) fn azimuth_count(kind: PrimKind, segments: u32) -> u32 {
    // Invariant: COUNT_TABLE lists `nu` for both frustum kinds; 3 is its documented minimum.
    count_rules(kind)
        .first()
        .map_or(3, |rule| rule.count(segments))
}

/// Triangle count of [`frustum`]: `nu` per non-pointed end for the side band, plus `nu` per cap.
pub(crate) fn frustum_triangle_count(rb: f32, rt: f32, nu: u32) -> u32 {
    let ends = u32::from(rb > 0.0) + u32::from(rt > 0.0);
    // Side: 2 triangles per quad, minus the zero-area one at each pointed end; caps: nu each.
    2 * ends * nu
}

/// Triangle count `generate` emits for the same inputs (README formula: `4·nu`).
pub(crate) fn triangle_count(radius: f32, half_height: f32, segments: u32) -> u32 {
    let _ = half_height;
    frustum_triangle_count(radius, radius, azimuth_count(PrimKind::Cylinder, segments))
}

/// Writes the cylinder mesh into `out`, which the caller has cleared; `segments` is clamped.
pub(crate) fn generate(
    radius: f32,
    half_height: f32,
    segments: u32,
    out: &mut MeshData,
) -> Result<(), GeomError> {
    let nu = azimuth_count(PrimKind::Cylinder, segments);
    frustum(half_height, radius, radius, nu, out);
    Ok(())
}

fn unit(v: [f64; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l > 0.0 {
        [(v[0] / l) as f32, (v[1] / l) as f32, (v[2] / l) as f32]
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn push(out: &mut MeshData, p: [f64; 3], n: [f32; 3]) -> u32 {
    out.positions.push([p[0] as f32, p[1] as f32, p[2] as f32]);
    out.normals.push(n);
    // Invariant: nu <= 1e4 (count of a clamped segments value), so indices fit in u32.
    (out.positions.len() - 1) as u32
}

/// Frustum along Y with base radius `rb` at y = -`half_height` and top radius `rt` at
/// +`half_height` (both >= 0, not both 0), `nu` >= 3 azimuth steps, flat caps.
pub(crate) fn frustum(half_height: f32, rb: f32, rt: f32, nu: u32, out: &mut MeshData) {
    let height = 2.0 * f64::from(half_height);
    let h = height / 2.0;
    let (rb, rt) = (f64::from(rb), f64::from(rt));
    let slope = (rb - rt) / height;
    let angle = |i: u32| (2.0 * std::f64::consts::PI * f64::from(i)) / f64::from(nu);
    // Side rings: bottom ring at index 0, top ring at nu + 1.
    for (y, r) in [(-h, rb), (h, rt)] {
        for i in 0..=nu {
            let (s, c) = angle(i).sin_cos();
            push(out, [r * c, y, r * s], unit([c, slope, s]));
        }
    }
    // grid(0, nu, 1, flip): quad (a, d, c, b) = triangles (a, d, c) and (a, c, b), outward.
    for i in 0..nu {
        let (a, b) = (i, i + 1);
        let (d, c) = (i + nu + 1, i + nu + 2);
        if rt > 0.0 {
            out.indices.extend_from_slice(&[a, d, c]);
        }
        if rb > 0.0 {
            out.indices.extend_from_slice(&[a, c, b]);
        }
    }
    // Caps: centre vertex, then a separate rim ring with the cap normal.
    for (y, r, ny) in [(-h, rb, -1.0f32), (h, rt, 1.0f32)] {
        if r <= 0.0 {
            continue;
        }
        let c0 = push(out, [0.0, y, 0.0], [0.0, ny, 0.0]);
        let ring0 = c0 + 1;
        for i in 0..=nu {
            let ph = angle(i);
            push(out, [r * ph.cos(), y, r * ph.sin()], [0.0, ny, 0.0]);
        }
        for i in 0..nu {
            if ny > 0.0 {
                out.indices
                    .extend_from_slice(&[c0, ring0 + i + 1, ring0 + i]);
            } else {
                out.indices
                    .extend_from_slice(&[c0, ring0 + i, ring0 + i + 1]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_geom::analysis;

    fn mesh(radius: f32, half_height: f32, segments: u32) -> MeshData {
        let mut m = MeshData::new();
        generate(radius, half_height, segments, &mut m).expect("cylinder generates");
        m
    }

    #[test]
    fn triangle_count_equals_generated_count() {
        for (r, hh) in [(0.6, 0.7), (1.5, 0.2)] {
            for s in [3, 8, 24, 64, 256] {
                let m = mesh(r, hh, s);
                assert_eq!(
                    triangle_count(r, hh, s) as usize,
                    m.triangle_count(),
                    "s {s}"
                );
            }
        }
    }

    #[test]
    fn lod0_layout_matches_reference_counts() {
        let m = mesh(0.6, 0.7, 24);
        assert_eq!(m.vertex_count(), 4 * 40 + 6);
        assert_eq!(m.triangle_count(), 160);
    }

    #[test]
    fn mesh_is_closed_outward_and_passes_normal_rule() {
        for s in [3, 8, 24, 64, 256] {
            let report = analysis::analyze(&mesh(1.5, 0.2, s));
            assert_eq!(analysis::passes(&report), Ok(()), "s {s}");
        }
    }

    #[test]
    fn every_triangle_faces_its_vertex_normals() {
        // What the reference's fixWinding checks: nothing would be flipped.
        let m = mesh(0.6, 0.7, 8);
        for t in m.indices.chunks(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| m.positions[i as usize]);
            let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let f = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let n = t.iter().fold([0.0f32; 3], |s, &i| {
                let v = m.normals[i as usize];
                [s[0] + v[0], s[1] + v[1], s[2] + v[2]]
            });
            assert!(f[0] * n[0] + f[1] * n[1] + f[2] * n[2] > 0.0);
        }
    }
}
