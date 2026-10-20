//! Ellipsoid tessellation: port of SDF Modeler's `ellipsoid(rx, ry, rz, nu, nv)`.

use raa_geom::{GeomError, MeshData};

use crate::{PrimKind, sphere};

/// Triangle count `generate` emits for the same inputs (README formula).
pub(crate) fn triangle_count(_radii: [f32; 3], segments: u32) -> u32 {
    let (nu, nv) = sphere::counts(PrimKind::Ellipsoid, segments);
    2 * nu * (nv - 1)
}

/// Writes the ellipsoid mesh into `out`, which the caller has cleared; `segments` is clamped.
pub(crate) fn generate(
    radii: [f32; 3],
    segments: u32,
    out: &mut MeshData,
) -> Result<(), GeomError> {
    let (nu, nv) = sphere::counts(PrimKind::Ellipsoid, segments);
    sphere::latitude_grid(radii, nu, nv, out);
    let squared_radii = radii.map(|r| r * r);
    out.normals.extend(out.positions.iter().map(|p| {
        let x = p[0] / squared_radii[0];
        let y = p[1] / squared_radii[1];
        let z = p[2] / squared_radii[2];
        let length = ((x * x + y * y) + z * z).sqrt();
        [x / length, y / length, z / length]
    }));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PrimParams;

    #[test]
    fn shared_poles_pass_normal_rule_for_flattened_low_ellipsoid() {
        let mut mesh = MeshData::new();
        generate([0.3, 1.4, 0.8], 8, &mut mesh).unwrap();
        let (nu, nv) = sphere::counts(PrimKind::Ellipsoid, 8);
        assert_eq!(mesh.vertex_count(), ((nu + 1) * (nv - 1) + 2) as usize);
        assert_eq!(mesh.normals.first(), Some(&[0.0, 1.0, 0.0]));
        assert_eq!(mesh.normals.last(), Some(&[0.0, -1.0, 0.0]));
        raa_geom::analysis::passes(&raa_geom::analysis::analyze(&mesh)).unwrap();
    }

    #[test]
    fn triangle_count_matches_generated_for_both_sets() {
        for set in ['A', 'B'] {
            let PrimParams::Ellipsoid { radii } =
                PrimParams::reference_set(PrimKind::Ellipsoid, set)
            else {
                unreachable!()
            };
            for segments in [3, 8, 24, 64, 256] {
                let mut mesh = MeshData::new();
                generate(radii, segments, &mut mesh).unwrap();
                assert_eq!(
                    triangle_count(radii, segments) as usize,
                    mesh.triangle_count()
                );
            }
        }
    }
}
