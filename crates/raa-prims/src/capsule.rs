//! Capsule tessellation: port of SDF Modeler's `capsule(height, radius, nu, nCap)`.
//!
//! Same rows, positions and triangles as the reference (rows from the top pole down: the
//! upper hemisphere centred on `y = +halfHeight`, then the lower one on `y = -halfHeight`;
//! the zero-area pole triangles are dropped, and so is the band between the two equator
//! rows when `halfHeight = 0`, as the reference's `done()` does). The smooth surface shares
//! its vertices (S2): one vertex per pole, `nu` per ring with the azimuth seam closed in the
//! index topology, and a single equator ring when `halfHeight = 0`. That changes only the
//! vertex count, not uniquePositions, triangles, bbox, area or volume, and keeps NORMAL_RULE
//! at the coarsest counts (`nCap = 1`).

use std::f64::consts::{FRAC_PI_2, TAU};

use raa_geom::{GeomError, MeshData};

use crate::kind::PrimKind;
use crate::reference::count_rules;

/// `(nu, nCap)` at the clamped `segments`, from the crate's count table.
fn counts(segments: u32) -> (u32, u32) {
    let mut nu = 3;
    let mut n_cap = 1;
    for rule in count_rules(PrimKind::Capsule) {
        match rule.name {
            "nu" => nu = rule.count(segments),
            "nCap" => n_cap = rule.count(segments),
            _ => {}
        }
    }
    (nu, n_cap)
}

/// Whether the middle band between the two equator rows is emitted (it has zero area
/// when the segment is a point).
fn has_band(half_height: f32) -> bool {
    half_height > 0.0
}

/// Triangle count `generate` emits for the same inputs (README formula).
pub(crate) fn triangle_count(radius: f32, half_height: f32, segments: u32) -> u32 {
    let _ = radius;
    let (nu, n_cap) = counts(segments);
    let hemispheres = 4 * nu * n_cap - 2 * nu;
    if has_band(half_height) {
        hemispheres + 2 * nu
    } else {
        hemispheres
    }
}

/// Writes the capsule mesh into `out`, which the caller has cleared; `segments` is clamped.
pub(crate) fn generate(
    radius: f32,
    half_height: f32,
    segments: u32,
    out: &mut MeshData,
) -> Result<(), GeomError> {
    let (nu, n_cap) = counts(segments);
    let r = f64::from(radius);
    let half = f64::from(half_height).max(0.0);
    let band = has_band(half_height);
    // Rings between the poles: upper j = nCap-1 down to 0, lower j = 0 (only with a band,
    // otherwise the equator ring is shared) up to nCap-1.
    let lower_start = if band { 0 } else { 1 };
    let rings = n_cap + (n_cap - lower_start);
    let vertices = (2 + rings * nu) as usize;
    out.positions.reserve(vertices);
    out.normals.reserve(vertices);
    out.indices
        .reserve(3 * triangle_count(radius, half_height, segments) as usize);

    let top = (half + r) as f32;
    out.positions.push([0.0, top, 0.0]);
    out.normals.push([0.0, 1.0, 0.0]);
    let upper = (0..n_cap).rev().map(|j| (j, 1.0));
    let lower = (lower_start..n_cap).map(|j| (j, -1.0));
    for (j, sign) in upper.chain(lower) {
        let (sin_a, cos_a) = (FRAC_PI_2 * (f64::from(j) / f64::from(n_cap))).sin_cos();
        let ring = r * cos_a;
        let y = sign * (half + r * sin_a);
        let ny = sign * sin_a;
        for i in 0..nu {
            let (s, c) = (TAU * f64::from(i) / f64::from(nu)).sin_cos();
            out.positions
                .push([(ring * c) as f32, y as f32, (ring * s) as f32]);
            // Unit direction from the nearest point on the segment [-hh, +hh] to the vertex.
            out.normals
                .push([(cos_a * c) as f32, ny as f32, (cos_a * s) as f32]);
        }
    }
    let bottom = 1 + rings * nu;
    out.positions.push([0.0, -top, 0.0]);
    out.normals.push([0.0, -1.0, 0.0]);

    // Counter-clockwise seen from outside: rings go down (-Y), i goes towards +Z at i = 0.
    let at = |k: u32, i: u32| 1 + k * nu + i % nu;
    for i in 0..nu {
        out.indices.extend_from_slice(&[0, at(0, i + 1), at(0, i)]);
    }
    for k in 0..rings - 1 {
        for i in 0..nu {
            let (a, b) = (at(k, i), at(k, i + 1));
            let (c, d) = (at(k + 1, i + 1), at(k + 1, i));
            out.indices.extend_from_slice(&[a, c, d, a, b, c]);
        }
    }
    for i in 0..nu {
        out.indices
            .extend_from_slice(&[at(rings - 1, i), at(rings - 1, i + 1), bottom]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PrimParams;
    use raa_geom::analysis;

    const SEGMENTS: [u32; 5] = [3, 8, 24, 64, 256];

    fn shapes() -> Vec<(f32, f32)> {
        let mut v = Vec::new();
        for set in ['A', 'B'] {
            match PrimParams::reference_set(PrimKind::Capsule, set) {
                PrimParams::Capsule {
                    radius,
                    half_height,
                } => v.push((radius, half_height)),
                other => panic!("capsule reference set is {other:?}"),
            }
        }
        v.push((0.5, 0.0));
        v
    }

    #[test]
    fn triangle_count_equals_generated_count_for_both_sets_and_zero_half_height() {
        for (radius, half_height) in shapes() {
            for s in SEGMENTS {
                let mut mesh = MeshData::new();
                generate(radius, half_height, s, &mut mesh).expect("capsule generates");
                assert_eq!(
                    triangle_count(radius, half_height, s) as usize,
                    mesh.triangle_count(),
                    "r {radius} hh {half_height} segments {s}"
                );
            }
        }
    }

    #[test]
    fn zero_half_height_mesh_passes_analysis() {
        for s in SEGMENTS {
            let mut mesh = MeshData::new();
            generate(0.5, 0.0, s, &mut mesh).expect("capsule generates");
            mesh.validate().expect("valid mesh");
            let report = analysis::analyze(&mesh);
            assert_eq!(analysis::passes(&report), Ok(()), "segments {s}");
        }
    }

    #[test]
    fn reference_sets_pass_analysis_with_unit_normals() {
        for (radius, half_height) in shapes() {
            for s in SEGMENTS {
                let mut mesh = MeshData::new();
                generate(radius, half_height, s, &mut mesh).expect("capsule generates");
                let report = analysis::analyze(&mesh);
                assert_eq!(
                    analysis::passes(&report),
                    Ok(()),
                    "hh {half_height} segments {s}"
                );
                assert!(report.volume > 0.0);
                for n in &mesh.normals {
                    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                    assert!((len - 1.0).abs() <= 1e-6, "normal length {len}");
                }
            }
        }
    }

    #[test]
    fn generation_is_deterministic() {
        let mut a = MeshData::new();
        let mut b = MeshData::new();
        generate(0.4, 0.5, 24, &mut a).expect("capsule generates");
        generate(0.4, 0.5, 24, &mut b).expect("capsule generates");
        assert_eq!(analysis::mesh_hash(&a), analysis::mesh_hash(&b));
    }
}
