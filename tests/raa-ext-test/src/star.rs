//! Star prism geometry provider (kind `star`).

use raa_geom::{GeomError, GeometryProvider, MeshData, Params, Quality};

const KIND: &str = "star";
const NAMES: [&str; 4] = ["points", "outerRadius", "innerRadius", "halfHeight"];

/// Closed star prism along Y with hard edges; not convex.
pub struct StarProvider;

struct Star {
    points: u32,
    outer: f32,
    inner: f32,
    half: f32,
}

fn bad(name: &str, reason: &str) -> GeomError {
    GeomError::InvalidParam {
        kind: KIND.to_string(),
        name: name.to_string(),
        reason: reason.to_string(),
    }
}

fn read(params: &Params) -> Result<Star, GeomError> {
    let points = params.scalar(KIND, "points")?;
    let outer = params.scalar(KIND, "outerRadius")?;
    let inner = params.scalar(KIND, "innerRadius")?;
    let half = params.scalar(KIND, "halfHeight")?;
    for (n, v) in NAMES.iter().zip([points, outer, inner, half]) {
        if !v.is_finite() {
            return Err(bad(n, "must be finite"));
        }
    }
    if points.fract() != 0.0 || !(3.0..=64.0).contains(&points) {
        return Err(bad("points", "must be an integer in [3, 64]"));
    }
    if inner <= 0.0 {
        return Err(bad("innerRadius", "must be > 0"));
    }
    if inner >= outer {
        return Err(bad("innerRadius", "must be < outerRadius"));
    }
    if half <= 0.0 {
        return Err(bad("halfHeight", "must be > 0"));
    }
    Ok(Star {
        points: points as u32,
        outer,
        inner,
        half,
    })
}

impl GeometryProvider for StarProvider {
    fn kind(&self) -> &str {
        KIND
    }

    fn param_names(&self) -> &[&'static str] {
        &NAMES
    }

    fn is_convex(&self) -> bool {
        false
    }

    fn triangle_count(&self, params: &Params, _quality: Quality) -> Result<u32, GeomError> {
        Ok(8 * read(params)?.points)
    }

    fn generate(
        &self,
        params: &Params,
        _quality: Quality,
        out: &mut MeshData,
    ) -> Result<(), GeomError> {
        let s = read(params)?;
        out.clear();
        let n = 2 * s.points;
        let ring: Vec<[f32; 2]> = (0..n)
            .map(|k| {
                let r = if k % 2 == 0 { s.outer } else { s.inner };
                let a = k as f32 * std::f32::consts::PI / s.points as f32;
                [r * a.cos(), -r * a.sin()]
            })
            .collect();
        let h = s.half;
        // Caps: centre vertex then the ring; outline is counter-clockwise seen from +Y.
        for (y, ny) in [(h, 1.0f32), (-h, -1.0)] {
            let base = out.positions.len() as u32;
            out.positions.push([0.0, y, 0.0]);
            out.normals.push([0.0, ny, 0.0]);
            for p in &ring {
                out.positions.push([p[0], y, p[1]]);
                out.normals.push([0.0, ny, 0.0]);
            }
            for k in 0..n {
                let a = base + 1 + k;
                let b = base + 1 + (k + 1) % n;
                if ny > 0.0 {
                    out.indices.extend_from_slice(&[base, a, b]);
                } else {
                    out.indices.extend_from_slice(&[base, b, a]);
                }
            }
        }
        for k in 0..n as usize {
            let a = ring[k];
            let b = ring[(k + 1) % n as usize];
            let (dx, dz) = (b[0] - a[0], b[1] - a[1]);
            let len = (dx * dx + dz * dz).sqrt();
            let nrm = [-dz / len, 0.0, dx / len];
            let base = out.positions.len() as u32;
            for p in [
                [a[0], -h, a[1]],
                [b[0], -h, b[1]],
                [b[0], h, b[1]],
                [a[0], h, a[1]],
            ] {
                out.positions.push(p);
                out.normals.push(nrm);
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
    use raa_geom::ParamValue::Scalar;
    use raa_geom::analysis::{analyze, mesh_hash, passes};

    fn params(points: f32, outer: f32, inner: f32, half: f32) -> Params {
        Params::new()
            .with("points", Scalar(points))
            .with("outerRadius", Scalar(outer))
            .with("innerRadius", Scalar(inner))
            .with("halfHeight", Scalar(half))
    }

    fn gen_mesh(p: &Params, q: Quality) -> MeshData {
        let mut m = MeshData::new();
        StarProvider.generate(p, q, &mut m).unwrap();
        m
    }

    #[test]
    fn five_point_star_passes_analysis_at_three_qualities() {
        for q in [Quality::LOW, Quality::LOD0, Quality::HIGH] {
            let m = gen_mesh(&params(5.0, 1.0, 0.4, 0.2), q);
            passes(&analyze(&m)).unwrap();
        }
    }

    #[test]
    fn triangle_count_matches_generated() {
        for pts in [3.0f32, 5.0, 12.0] {
            let p = params(pts, 1.0, 0.5, 0.3);
            let m = gen_mesh(&p, Quality::LOD0);
            let c = StarProvider.triangle_count(&p, Quality::LOD0).unwrap();
            assert_eq!(c as usize, m.triangle_count());
            assert_eq!(c, 8 * pts as u32);
        }
    }

    #[test]
    fn mesh_hash_is_deterministic() {
        let p = params(7.0, 1.0, 0.5, 0.3);
        let a = gen_mesh(&p, Quality::LOW);
        let b = gen_mesh(&p, Quality::LOW);
        assert_eq!(mesh_hash(&a), mesh_hash(&b));
    }

    #[test]
    fn invalid_params_are_refused() {
        for p in [
            params(2.0, 1.0, 0.5, 0.3),
            params(4.5, 1.0, 0.5, 0.3),
            params(5.0, 1.0, 1.0, 0.3),
            params(5.0, 1.0, 0.5, 0.0),
        ] {
            let e = StarProvider
                .generate(&p, Quality::LOW, &mut MeshData::new())
                .unwrap_err();
            assert_eq!(e.code(), "geom.invalid_param");
        }
    }

    #[test]
    fn missing_param_is_reported() {
        let p = Params::new().with("points", Scalar(5.0));
        let e = StarProvider
            .generate(&p, Quality::LOW, &mut MeshData::new())
            .unwrap_err();
        assert_eq!(e.code(), "geom.missing_param");
    }
}
