//! S18 CPU reference; all outputs stay in mesh space.

use std::collections::BTreeSet;

use crate::{MAX_INFLUENCES, SkinError, SkinMesh};

/// Private vertex results for one instance; rest attributes and topology are untouched.
#[derive(Clone, Debug, PartialEq)]
pub struct CpuSkin {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tangents: Vec<[f32; 4]>,
    /// Unique triangle edges, sorted by (min vertex index, max vertex index).
    pub edge_segments: Vec<[[f32; 3]; 2]>,
}

/// Skins validated rest geometry with S16 skin matrices, already containing inverse
/// bind transforms. `pose` holds 16 column-major floats per joint; `None` means rest.
/// No instance/world transform or perspective divide is applied. Directions use the
/// blended upper 3x3, normalized once after blending; collapsed directions become zero.
pub fn skin_cpu(mesh: &SkinMesh, pose: Option<&[f32]>) -> Result<CpuSkin, SkinError> {
    if let Some(pose) = pose {
        if pose.is_empty() || !pose.len().is_multiple_of(16) {
            return Err(SkinError::new(
                "skin.pose_length",
                "expected 16 floats per joint",
            ));
        }
        mesh.validate_joint_count(pose.len() / 16)?;
        if pose.iter().any(|x| !x.is_finite()) {
            return Err(SkinError::new(
                "skin.pose_length",
                "pose matrices must be finite",
            ));
        }
    }
    let mut rest = mesh.rest.clone();
    rest.ensure_tangents();
    let mut out = CpuSkin {
        positions: Vec::with_capacity(rest.vertex_count()),
        normals: Vec::with_capacity(rest.vertex_count()),
        tangents: Vec::with_capacity(rest.vertex_count()),
        edge_segments: Vec::new(),
    };
    for (i, &p) in rest.base.positions.iter().enumerate() {
        let n = rest.base.normals[i];
        let t = rest
            .tangents
            .as_ref()
            .map_or_else(|| raa_mesh::default_tangent(n), |tangents| tangents[i]);
        let (p, n, xyz) = if let Some(pose) = pose {
            let mut m = [0.0; 16];
            for k in 0..MAX_INFLUENCES {
                let slot = i * MAX_INFLUENCES + k;
                let start = usize::from(mesh.joints[slot]) * 16;
                for (dst, src) in m.iter_mut().zip(&pose[start..start + 16]) {
                    *dst += mesh.weights[slot] * src;
                }
            }
            (
                transform(&m, p, 1.0),
                normalize(transform(&m, n, 0.0)),
                normalize(transform(&m, [t[0], t[1], t[2]], 0.0)),
            )
        } else {
            (p, n, [t[0], t[1], t[2]])
        };
        out.positions.push(p);
        out.normals.push(n);
        out.tangents.push([xyz[0], xyz[1], xyz[2], t[3]]);
    }
    let mut edges = BTreeSet::new();
    for tri in rest.base.indices.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            edges.insert((a.min(b), a.max(b)));
        }
    }
    out.edge_segments = edges
        .into_iter()
        .map(|(a, b)| [out.positions[a as usize], out.positions[b as usize]])
        .collect();
    Ok(out)
}

fn transform(m: &[f32; 16], v: [f32; 3], w: f32) -> [f32; 3] {
    std::array::from_fn(|r| m[r] * v[0] + m[4 + r] * v[1] + m[8 + r] * v[2] + m[12 + r] * w)
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if length > 0.0 {
        v.map(|x| x / length)
    } else {
        [0.0; 3]
    }
}
