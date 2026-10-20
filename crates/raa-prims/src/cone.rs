//! Cone tessellation: port of SDF Modeler's `frustum(height, rb, 0, nu)`.
//!
//! Base of radius `radius` at y = -halfHeight, apex at +halfHeight (S3). The apex is the
//! reference's top ring at r = 0: `nu + 1` coincident vertices, each with the slant normal of
//! its azimuth `unit(cos, radius / height, sin)`; its zero-area side triangles and the top cap
//! are absent, as in the reference. Shared maths in [`crate::cylinder::frustum`].

use raa_geom::{GeomError, MeshData};

use crate::PrimKind;
use crate::cylinder::{azimuth_count, frustum, frustum_triangle_count};

/// Triangle count `generate` emits for the same inputs (README formula: `2·nu`).
pub(crate) fn triangle_count(radius: f32, half_height: f32, segments: u32) -> u32 {
    let _ = half_height;
    frustum_triangle_count(radius, 0.0, azimuth_count(PrimKind::Cone, segments))
}

/// Writes the cone mesh into `out`, which the caller has cleared; `segments` is clamped.
pub(crate) fn generate(
    radius: f32,
    half_height: f32,
    segments: u32,
    out: &mut MeshData,
) -> Result<(), GeomError> {
    let nu = azimuth_count(PrimKind::Cone, segments);
    frustum(half_height, radius, 0.0, nu, out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_geom::analysis;

    fn mesh(radius: f32, half_height: f32, segments: u32) -> MeshData {
        let mut m = MeshData::new();
        generate(radius, half_height, segments, &mut m).expect("cone generates");
        m
    }

    #[test]
    fn triangle_count_equals_generated_count() {
        for (r, hh) in [(0.7, 0.8), (0.3, 1.5)] {
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
        let m = mesh(0.7, 0.8, 24);
        assert_eq!(m.vertex_count(), 3 * 40 + 4);
        assert_eq!(m.triangle_count(), 80);
    }

    #[test]
    fn base_at_minus_half_height_and_apex_at_plus() {
        let b = mesh(0.7, 0.8, 8).bounds();
        assert_eq!((b.min.y, b.max.y), (-0.8, 0.8));
        let m = mesh(0.7, 0.8, 8);
        for (p, _) in m
            .positions
            .iter()
            .zip(&m.normals)
            .filter(|(p, _)| p[1] > 0.0)
        {
            assert_eq!((p[0].abs(), p[2].abs()), (0.0, 0.0));
        }
    }

    #[test]
    fn mesh_is_closed_outward_and_passes_normal_rule() {
        for (r, hh) in [(0.7, 0.8), (0.3, 1.5)] {
            for s in [3, 8, 24, 64, 256] {
                let report = analysis::analyze(&mesh(r, hh, s));
                assert_eq!(analysis::passes(&report), Ok(()), "s {s}");
            }
        }
    }
}
