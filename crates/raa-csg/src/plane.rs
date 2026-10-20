//! Planes of convex meshes.

use raa_geom::MeshData;

use crate::CsgError;

/// Points `p` with `n.p = d`; outside when `n.p - d > eps`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    pub n: [f64; 3],
    pub d: f64,
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl Plane {
    pub(crate) fn signed(&self, p: [f64; 3]) -> f64 {
        dot(self.n, p) - self.d
    }
}

/// Parallel (either sign), normals within 1e-9, offsets within `eps`.
#[allow(dead_code)]
pub(crate) fn coincident(a: &Plane, b: &Plane, eps: f64) -> bool {
    let same = (0..3).all(|k| (a.n[k] - b.n[k]).abs() <= 1e-9) && (a.d - b.d).abs() <= eps;
    let opposite = (0..3).all(|k| (a.n[k] + b.n[k]).abs() <= 1e-9) && (a.d + b.d).abs() <= eps;
    same || opposite
}

/// Face planes (merged, first-occurrence order) of a convex mesh. A non-convex mesh returns
/// `Degenerate` with the worst distance; the caller maps it to host/tool not convex.
pub fn convex_planes(mesh: &MeshData, eps: f64) -> Result<Vec<Plane>, CsgError> {
    let pos = |i: u32| -> Option<[f64; 3]> {
        let p = mesh.positions.get(i as usize)?;
        Some([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])])
    };
    let mut planes: Vec<Plane> = Vec::new();
    for t in mesh.indices.chunks_exact(3) {
        let (Some(a), Some(b), Some(c)) = (pos(t[0]), pos(t[1]), pos(t[2])) else {
            return Err(CsgError::Degenerate("index out of range".into()));
        };
        let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let cr = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let len = dot(cr, cr).sqrt();
        if len.is_nan() || len <= 0.0 {
            continue;
        }
        let n = [cr[0] / len, cr[1] / len, cr[2] / len];
        let cand = Plane { n, d: dot(n, a) };
        let merged = planes
            .iter()
            .any(|p| (0..3).all(|k| (p.n[k] - n[k]).abs() <= 1e-9) && (p.d - cand.d).abs() <= eps);
        if !merged {
            planes.push(cand);
        }
    }
    let mut worst = 0.0f64;
    for p in &mesh.positions {
        let q = [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])];
        for pl in &planes {
            worst = worst.max(pl.signed(q));
        }
    }
    if worst > eps {
        return Err(CsgError::Degenerate(format!(
            "mesh is not convex: worst vertex distance {worst:e} beyond a face plane (eps {eps:e})"
        )));
    }
    Ok(planes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_geom::GeometryProvider;
    use raa_geom::Quality;
    use raa_geom::builtin::DebugCube;
    use raa_geom::{ParamValue, Params};

    fn cube() -> MeshData {
        let mut m = MeshData::new();
        let params = Params::new().with("half", ParamValue::Scalar(0.5));
        DebugCube
            .generate(&params, Quality::LOD0, &mut m)
            .expect("cube");
        m
    }

    #[test]
    fn debug_cube_has_six_planes_in_fixed_order() {
        let planes = convex_planes(&cube(), 1e-6).expect("convex");
        assert_eq!(planes.len(), 6);
        for p in &planes {
            assert!((p.d - 0.5).abs() < 1e-6);
        }
        assert_eq!(planes, convex_planes(&cube(), 1e-6).expect("convex"));
    }

    #[test]
    fn l_shape_is_not_convex() {
        // L-shaped prism: outline in XY extruded along Z, triangulated by hand.
        let outline = [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ];
        let mut m = MeshData::new();
        for z in [0.0f32, 1.0] {
            for o in outline {
                m.positions.push([o[0], o[1], z]);
                m.normals.push([0.0, 0.0, 1.0]);
            }
        }
        // top/bottom caps and side quads
        for (a, b, c) in [(0u32, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5)] {
            m.indices.extend([a, c, b]);
            m.indices.extend([a + 6, b + 6, c + 6]);
        }
        for i in 0..6u32 {
            let j = (i + 1) % 6;
            m.indices.extend([i, j, j + 6, i, j + 6, i + 6]);
        }
        assert!(matches!(
            convex_planes(&m, 1e-6),
            Err(CsgError::Degenerate(_))
        ));
    }
}
