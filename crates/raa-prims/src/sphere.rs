//! Sphere tessellation: port of SDF Modeler's `ellipsoid(r, r, r, nu, nv)`.

use raa_geom::{GeomError, MeshData};

use crate::{PrimKind, reference::count_rules};

/// Triangle count `generate` emits for the same inputs (README formula).
pub(crate) fn triangle_count(_radius: f32, segments: u32) -> u32 {
    let (nu, nv) = counts(PrimKind::Sphere, segments);
    2 * nu * (nv - 1)
}

/// Writes the sphere mesh into `out`, which the caller has cleared; `segments` is clamped.
pub(crate) fn generate(radius: f32, segments: u32, out: &mut MeshData) -> Result<(), GeomError> {
    let (nu, nv) = counts(PrimKind::Sphere, segments);
    latitude_grid([radius; 3], nu, nv, out);
    out.normals
        .extend(out.positions.iter().map(|p| p.map(|v| v / radius)));
    Ok(())
}

/// Sphere and ellipsoid each have exactly the `nu`, `nv` rules, in that order.
pub(crate) fn counts(kind: PrimKind, segments: u32) -> (u32, u32) {
    let rules = count_rules(kind);
    // Invariant: callers use Sphere/Ellipsoid, whose count table has these two entries.
    (rules[0].count(segments), rules[1].count(segments))
}

/// Reference ellipsoid positions and grid indices, after `done()` and `fixWinding()`.
/// Azimuth starts at +X and increases toward +Z; latitude runs from +Y to -Y.
/// Each pole is shared so NORMAL_RULE measures the entire fan (approved Q6).
pub(crate) fn latitude_grid(radii: [f32; 3], nu: u32, nv: u32, out: &mut MeshData) {
    out.positions.reserve(((nu + 1) * (nv - 1) + 2) as usize);
    out.indices.reserve((6 * nu * (nv - 1)) as usize);
    for j in 0..=nv {
        let theta = (std::f32::consts::PI * j as f32) / nv as f32;
        let (sin_theta, cos_theta) = theta.sin_cos();
        // Keep the reference row order and its i = 0 position at each pole.
        let last_i = if j == 0 || j == nv { 0 } else { nu };
        for i in 0..=last_i {
            // Closing the seam at zero avoids the f32 residual of sin(2*PI).
            let phi = (std::f32::consts::TAU * (i % nu) as f32) / nu as f32;
            let (sin_phi, cos_phi) = phi.sin_cos();
            let direction = if j == 0 {
                [0.0, 1.0, 0.0]
            } else if j == nv {
                [0.0, -1.0, 0.0]
            } else {
                [sin_theta * cos_phi, cos_theta, sin_theta * sin_phi]
            };
            // Keep the reference's direction-then-scale evaluation order in f32.
            out.positions.push([
                direction[0] * radii[0],
                direction[1] * radii[1],
                direction[2] * radii[2],
            ]);
        }
    }
    // Remap only the coincident pole copies; interior and seam vertices stay split.
    let index = |j, i| {
        if j == 0 {
            0
        } else if j == nv {
            1 + (nv - 1) * (nu + 1)
        } else {
            1 + (j - 1) * (nu + 1) + i
        }
    };
    for j in 0..nv {
        for i in 0..nu {
            let a = index(j, i);
            let b = index(j, i + 1);
            let d = index(j + 1, i);
            let c = index(j + 1, i + 1);
            // grid(..., true) pushes (a,d,c), (a,c,b). fixWinding swaps
            // their last two indices; done drops the collapsed pole triangles.
            if j + 1 < nv {
                out.indices.extend_from_slice(&[a, c, d]);
            }
            if j > 0 {
                out.indices.extend_from_slice(&[a, b, c]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PrimParams;

    #[test]
    fn triangle_count_matches_generated_for_both_sets() {
        for set in ['A', 'B'] {
            let PrimParams::Sphere { radius } = PrimParams::reference_set(PrimKind::Sphere, set)
            else {
                unreachable!()
            };
            for segments in [3, 8, 24, 64, 256] {
                let mut mesh = MeshData::new();
                generate(radius, segments, &mut mesh).unwrap();
                assert_eq!(
                    triangle_count(radius, segments) as usize,
                    mesh.triangle_count()
                );
            }
        }
    }
}
