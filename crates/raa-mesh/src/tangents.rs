//! Tangent generation (S10 stream 1, location 11) and the default orthonormal basis.

/// Below this |det| of a triangle's uv edge matrix the triangle has no usable uv
/// parametrisation and contributes nothing to its vertices' tangents.
const UV_DET_EPS: f64 = 1e-12;
/// Below this squared length an accumulated or orthogonalised tangent is degenerate.
const LEN2_EPS: f64 = 1e-20;

/// Per-vertex tangents from positions, normals and uvs of an indexed triangle list.
///
/// Each triangle contributes its dP/du (tangent) and dP/dv (bitangent) directions to its
/// three corners, weighted by the corner angle. Per vertex the accumulated tangent is
/// orthogonalised against the normal (Gram-Schmidt) and normalised; w is
/// `sign(dot(cross(n, t), bitangent))` (+1 when zero). Triangles with degenerate uvs or
/// zero area contribute nothing, and a vertex left without a usable tangent gets
/// [`default_tangent`] of its normal. Triangles with an out-of-range index are skipped;
/// a missing uv reads as (0, 0) and a missing normal as +Z.
///
/// `out` is cleared (capacity kept) and receives one tangent per position.
pub fn generate_tangents(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
    out: &mut Vec<[f32; 4]>,
) {
    let vertex_count = positions.len();
    // Accumulators in f64: [tangent, bitangent] per vertex.
    let mut acc = vec![[[0.0f64; 3]; 2]; vertex_count];

    for tri in indices.chunks_exact(3) {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let (Some(p0), Some(p1), Some(p2)) =
            (positions.get(i0), positions.get(i1), positions.get(i2))
        else {
            continue;
        };
        let uv = |i: usize| uvs.get(i).copied().unwrap_or([0.0; 2]).map(f64::from);
        let (p0, p1, p2) = (to_f64(p0), to_f64(p1), to_f64(p2));
        let (w0, w1, w2) = (uv(i0), uv(i1), uv(i2));

        let e1 = sub(p1, p0);
        let e2 = sub(p2, p0);
        let du1 = w1[0] - w0[0];
        let dv1 = w1[1] - w0[1];
        let du2 = w2[0] - w0[0];
        let dv2 = w2[1] - w0[1];
        let det = du1 * dv2 - du2 * dv1;
        if det.abs() < UV_DET_EPS {
            continue;
        }
        let r = 1.0 / det;
        let (Some(t), Some(b)) = (
            normalized(scale(sub(scale(e1, dv2), scale(e2, dv1)), r)),
            normalized(scale(sub(scale(e2, du1), scale(e1, du2)), r)),
        ) else {
            continue;
        };

        for (corner, prev, next) in [(i0, p2, p1), (i1, p0, p2), (i2, p1, p0)] {
            let here = to_f64_any(positions, corner);
            let angle = corner_angle(here, prev, next);
            if let Some([at, ab]) = acc.get_mut(corner)
                && angle > 0.0
            {
                for (a, d) in at.iter_mut().zip(t).chain(ab.iter_mut().zip(b)) {
                    *a += d * angle;
                }
            }
        }
    }

    out.clear();
    out.reserve(vertex_count);
    for (i, &[t, b]) in acc.iter().enumerate() {
        let n_f32 = normals.get(i).copied().unwrap_or([0.0, 0.0, 1.0]);
        let n = normalized(to_f64(&n_f32)).unwrap_or([0.0, 0.0, 1.0]);
        // Gram-Schmidt: remove the normal component, then normalise.
        let t = normalized(sub(t, scale(n, dot(n, t))));
        match t {
            Some(t) => {
                let w = if dot(cross(n, t), b) < 0.0 { -1.0 } else { 1.0 };
                out.push([t[0] as f32, t[1] as f32, t[2] as f32, w]);
            }
            None => out.push(default_tangent(n_f32)),
        }
    }
}

/// Branchless orthonormal basis of Duff et al. 2017 ("Building an Orthonormal Basis,
/// Revisited"): the returned xyz is the first basis vector b1, so `cross(n, b1)` is b2
/// and w = 1. `n` is normalised first; a zero or non-finite normal reads as +Z.
pub fn default_tangent(n: [f32; 3]) -> [f32; 4] {
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    let [x, y, z] = if len > 0.0 && len.is_finite() {
        n.map(|c| c / len)
    } else {
        [0.0, 0.0, 1.0]
    };
    let sign = 1.0f32.copysign(z);
    let a = -1.0 / (sign + z);
    let b = x * y * a;
    [1.0 + sign * x * x * a, sign * b, -sign * x, 1.0]
}

fn to_f64(v: &[f32; 3]) -> [f64; 3] {
    v.map(f64::from)
}

fn to_f64_any(positions: &[[f32; 3]], i: usize) -> [f64; 3] {
    positions.get(i).map(to_f64).unwrap_or([0.0; 3])
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    a.map(|c| c * s)
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalized(a: [f64; 3]) -> Option<[f64; 3]> {
    let len2 = dot(a, a);
    if len2 > LEN2_EPS && len2.is_finite() {
        Some(scale(a, 1.0 / len2.sqrt()))
    } else {
        None
    }
}

/// Angle at `here` between the edges to `prev` and `next`; 0 for a degenerate corner.
fn corner_angle(here: [f64; 3], prev: [f64; 3], next: [f64; 3]) -> f64 {
    match (normalized(sub(prev, here)), normalized(sub(next, here))) {
        (Some(a), Some(b)) => dot(a, b).clamp(-1.0, 1.0).acos(),
        _ => 0.0,
    }
}
