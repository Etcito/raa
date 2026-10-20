//! Built-in providers that need no tessellation scheme.

use crate::{GeomError, GeometryProvider, MeshData, Params, Quality};

/// Axis-aligned cube centred at the origin with half edge `half` (> 0).
///
/// 24 vertices (4 per face, flat face normals) and 12 triangles, counter-clockwise
/// seen from outside. Convex; independent of [`Quality`].
#[derive(Clone, Copy, Debug, Default)]
pub struct DebugCube;

const KIND: &str = "debug_cube";
const PARAMS: [&str; 1] = ["half"];

/// Per face: (normal, u, v) with u x v = normal, so corners -u-v, +u-v, +u+v, -u+v are CCW.
const FACES: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
    ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
    ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
    ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
    ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
];

impl DebugCube {
    fn half(params: &Params) -> Result<f32, GeomError> {
        let half = params.scalar(KIND, "half")?;
        if !(half.is_finite() && half > 0.0) {
            return Err(GeomError::InvalidParam {
                kind: KIND.to_string(),
                name: "half".to_string(),
                reason: "half must be > 0".to_string(),
            });
        }
        Ok(half)
    }
}

impl GeometryProvider for DebugCube {
    fn kind(&self) -> &str {
        KIND
    }

    fn param_names(&self) -> &[&'static str] {
        &PARAMS
    }

    fn is_convex(&self) -> bool {
        true
    }

    fn triangle_count(&self, params: &Params, _quality: Quality) -> Result<u32, GeomError> {
        DebugCube::half(params)?;
        Ok(12)
    }

    fn generate(
        &self,
        params: &Params,
        _quality: Quality,
        out: &mut MeshData,
    ) -> Result<(), GeomError> {
        out.clear();
        let h = DebugCube::half(params)?;
        for (n, u, v) in FACES {
            let base = out.positions.len() as u32;
            for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let p: [f32; 3] = std::array::from_fn(|i| h * (n[i] + su * u[i] + sv * v[i]));
                out.positions.push(p);
                out.normals.push(n);
            }
            out.indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube(half: f32) -> MeshData {
        let mut m = MeshData::new();
        DebugCube
            .generate(
                &Params::new().with("half", crate::ParamValue::Scalar(half)),
                Quality::LOD0,
                &mut m,
            )
            .unwrap();
        m
    }

    #[test]
    fn cube_has_24_vertices_and_12_triangles_within_bounds() {
        let m = cube(0.5);
        assert_eq!(m.vertex_count(), 24);
        assert_eq!(m.triangle_count(), 12);
        m.validate().unwrap();
        for p in &m.positions {
            assert!(p.iter().all(|c| c.abs() == 0.5));
        }
    }

    #[test]
    fn every_triangle_winds_along_its_face_normal() {
        let m = cube(1.0);
        for t in m.indices.chunks_exact(3) {
            let [a, b, c] = [0, 1, 2].map(|i| m.positions[t[i] as usize]);
            let e1: [f32; 3] = std::array::from_fn(|i| b[i] - a[i]);
            let e2: [f32; 3] = std::array::from_fn(|i| c[i] - a[i]);
            let cr = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let n = m.normals[t[0] as usize];
            let d: f32 = (0..3).map(|i| cr[i] * n[i]).sum();
            assert!(d > 0.0);
            // Outward: the face centre side matches the normal.
            let s: f32 = (0..3).map(|i| a[i] * n[i]).sum();
            assert!(s > 0.0);
        }
    }

    #[test]
    fn non_finite_half_is_invalid() {
        for h in [f32::NAN, f32::INFINITY, 0.0] {
            let p = Params::new().with("half", crate::ParamValue::Scalar(h));
            let e = DebugCube.triangle_count(&p, Quality::LOW).unwrap_err();
            assert_eq!(e.code(), "geom.invalid_param");
        }
    }
}
