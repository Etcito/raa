//! Torus tessellation: port of SDF Modeler's `torus(major, minor, nu, nv)`.
//!
//! Reference (sdfm@50b45d5 leaf-mesh.ts:208-221): a `(nu + 1) x (nv + 1)` vertex grid,
//! `u` (azimuth `ph = 2π·i/nu` around Y) fastest, `v` (tube angle `a = 2π·j/nv`) outer;
//! both seams (`i = nu`, `j = nv`) are duplicated vertices. Position
//! `((R + r·cos a)·cos ph, r·sin a, (R + r·cos a)·sin ph)`, normal
//! `(cos a·cos ph, sin a, cos a·sin ph)` (from the nearest point on the major circle).
//! `grid(0, nu, nv)` pushes quads `a b c d` as `(a, b, c), (a, c, d)`; that order faces
//! inward on this grid, so `fixWinding` swaps the last two indices of every triangle. The
//! port emits the swapped order directly: `(a, c, b), (a, d, c)`. No triangle has zero
//! area for `0 < minor < major`, so `done()` drops none.

use raa_geom::{GeomError, MeshData};

use crate::PrimKind;
use crate::reference::count_rules;

/// `(nu, nv)` at `segments` from the count table (torus rules: nu 48/3, nv 20/3).
fn counts(segments: u32) -> (u32, u32) {
    let rules = count_rules(PrimKind::Torus);
    let at = |name: &str, fallback: u32| {
        rules
            .iter()
            .find(|c| c.name == name)
            .map_or(fallback, |c| c.count(segments))
    };
    (at("nu", 48), at("nv", 20))
}

/// Triangle count `generate` emits for the same inputs (README formula `2·nu·nv`).
pub(crate) fn triangle_count(major_radius: f32, minor_radius: f32, segments: u32) -> u32 {
    let _ = (major_radius, minor_radius);
    let (nu, nv) = counts(segments);
    2 * nu * nv
}

/// Writes the torus mesh into `out`, which the caller has cleared; `segments` is clamped.
pub(crate) fn generate(
    major_radius: f32,
    minor_radius: f32,
    segments: u32,
    out: &mut MeshData,
) -> Result<(), GeomError> {
    let (nu, nv) = counts(segments);
    let (major, minor) = (f64::from(major_radius), f64::from(minor_radius));
    let row = nu as usize + 1;
    let verts = row * (nv as usize + 1);
    out.positions.reserve(verts);
    out.normals.reserve(verts);
    out.indices.reserve(6 * nu as usize * nv as usize);
    let tau = 2.0 * std::f64::consts::PI;
    for j in 0..=nv {
        let a = tau * f64::from(j) / f64::from(nv);
        let (ca, sa) = (a.cos(), a.sin());
        for i in 0..=nu {
            let ph = tau * f64::from(i) / f64::from(nu);
            let (c, s) = (ph.cos(), ph.sin());
            let r = major + minor * ca;
            out.positions
                .push([(r * c) as f32, (minor * sa) as f32, (r * s) as f32]);
            out.normals
                .push([(ca * c) as f32, sa as f32, (ca * s) as f32]);
        }
    }
    let row = nu + 1;
    for j in 0..nv {
        for i in 0..nu {
            let a = j * row + i;
            let (b, c, d) = (a + 1, a + row + 1, a + row);
            out.indices.extend_from_slice(&[a, c, b, a, d, c]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PrimParams;
    use raa_geom::analysis;

    fn radii(set: char) -> (f32, f32) {
        match PrimParams::reference_set(PrimKind::Torus, set) {
            PrimParams::Torus {
                major_radius,
                minor_radius,
            } => (major_radius, minor_radius),
            other => panic!("unexpected params {other:?}"),
        }
    }

    #[test]
    fn triangle_count_equals_generated_count_for_both_sets() {
        for set in ['A', 'B'] {
            let (major, minor) = radii(set);
            for segments in [3, 8, 24, 64, 256] {
                let mut mesh = MeshData::new();
                generate(major, minor, segments, &mut mesh).unwrap();
                assert_eq!(
                    triangle_count(major, minor, segments) as usize,
                    mesh.triangle_count(),
                    "set {set} segments {segments}"
                );
            }
        }
    }

    #[test]
    fn lod0_uses_reference_defaults_and_passes_analysis() {
        let (major, minor) = radii('A');
        let mut mesh = MeshData::new();
        generate(major, minor, 24, &mut mesh).unwrap();
        assert_eq!(mesh.vertex_count(), 49 * 21);
        assert_eq!(mesh.triangle_count(), 1920);
        let report = analysis::analyze(&mesh);
        assert_eq!(analysis::passes(&report), Ok(()));
        assert!(report.volume > 0.0);
    }

    #[test]
    fn normals_are_unit_and_point_away_from_the_major_circle() {
        let (major, minor) = radii('B');
        let mut mesh = MeshData::new();
        generate(major, minor, 8, &mut mesh).unwrap();
        for (p, n) in mesh.positions.iter().zip(&mesh.normals) {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() <= 1e-6);
            let h = (p[0] * p[0] + p[2] * p[2]).sqrt();
            let q = [p[0] - major * p[0] / h, p[1], p[2] - major * p[2] / h];
            let d = q[0] * n[0] + q[1] * n[1] + q[2] * n[2];
            assert!((d - minor).abs() <= 1e-5, "offset {d} vs minor {minor}");
        }
    }

    #[test]
    fn output_is_deterministic() {
        let (major, minor) = radii('A');
        let (mut a, mut b) = (MeshData::new(), MeshData::new());
        generate(major, minor, 64, &mut a).unwrap();
        generate(major, minor, 64, &mut b).unwrap();
        assert_eq!(analysis::mesh_hash(&a), analysis::mesh_hash(&b));
    }
}
