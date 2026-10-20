//! Cone tessellation: port of SDF Modeler's `frustum(height, rb, rt, nu)`.
//!
//! Base of radius `radius` at y = -halfHeight, top of radius `radius_top` at y = +halfHeight.
//! The S3 cone has `radius_top = 0` (apex at +halfHeight): the apex is the reference's top
//! ring at r = 0, `nu + 1` coincident vertices, each with the slant normal of its azimuth;
//! its zero-area side triangles and the top cap are absent, as in the reference. A radius of
//! 0 at either end is a pointed end. Shared maths in [`crate::cylinder::frustum`].

use raa_geom::{GeomError, MeshData};

use crate::PrimKind;
use crate::cylinder::{azimuth_count, frustum, frustum_triangle_count};

/// Triangle count `generate` emits for the same inputs (README formula: `2·nu` per
/// non-pointed end, plus `nu` per cap).
pub(crate) fn triangle_count(radius: f32, radius_top: f32, half_height: f32, segments: u32) -> u32 {
    let _ = half_height;
    frustum_triangle_count(radius, radius_top, azimuth_count(PrimKind::Cone, segments))
}

/// Writes the cone mesh into `out`, which the caller has cleared; `segments` is clamped.
pub(crate) fn generate(
    radius: f32,
    radius_top: f32,
    half_height: f32,
    segments: u32,
    out: &mut MeshData,
) -> Result<(), GeomError> {
    let nu = azimuth_count(PrimKind::Cone, segments);
    frustum(half_height, radius, radius_top, nu, out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_geom::analysis;

    fn mesh(radius: f32, half_height: f32, segments: u32) -> MeshData {
        let mut m = MeshData::new();
        generate(radius, 0.0, half_height, segments, &mut m).expect("cone generates");
        m
    }

    #[test]
    fn triangle_count_equals_generated_count() {
        for (r, hh) in [(0.7, 0.8), (0.3, 1.5)] {
            for s in [3, 8, 24, 64, 256] {
                let m = mesh(r, hh, s);
                assert_eq!(
                    triangle_count(r, 0.0, hh, s) as usize,
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

    /// Frustum cases: both ends round, bottom pointed, top pointed.
    fn frustum(rb: f32, rt: f32, hh: f32, segments: u32) -> MeshData {
        let mut m = MeshData::new();
        generate(rb, rt, hh, segments, &mut m).expect("frustum generates");
        m
    }

    #[test]
    fn frustum_is_closed_and_counts_match_for_every_end_combination() {
        for (rb, rt) in [(0.35, 0.2), (0.0, 0.3), (0.4, 0.0)] {
            for s in [3, 8, 24, 64] {
                let m = frustum(rb, rt, 0.6, s);
                assert_eq!(
                    triangle_count(rb, rt, 0.6, s) as usize,
                    m.triangle_count(),
                    "rb {rb} rt {rt} s {s}"
                );
                assert_eq!(
                    analysis::passes(&analysis::analyze(&m)),
                    Ok(()),
                    "rb {rb} rt {rt} s {s}"
                );
            }
        }
    }

    #[test]
    fn frustum_bottom_radius_sits_at_minus_half_height() {
        // Bottom ring at y = -hh has radius rb, top ring at y = +hh has radius rt.
        let m = frustum(0.35, 0.2, 0.6, 24);
        let bottom = m
            .positions
            .iter()
            .filter(|p| p[1] < 0.0)
            .map(|p| p[0].hypot(p[2]));
        let top = m
            .positions
            .iter()
            .filter(|p| p[1] > 0.0)
            .map(|p| p[0].hypot(p[2]));
        let max_bottom = bottom.fold(0.0_f32, f32::max);
        let max_top = top.fold(0.0_f32, f32::max);
        assert!((max_bottom - 0.35).abs() < 1e-6, "bottom {max_bottom}");
        assert!((max_top - 0.2).abs() < 1e-6, "top {max_top}");
    }
}
