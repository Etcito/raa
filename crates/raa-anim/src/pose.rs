//! S16 source-ordered pose arithmetic; matrices use column vectors.
use crate::{AnimError, Rig};
use raa_det::fmath::{normalize, trs_compose};

/// Column-major matrix, directly suitable for S18 skin uploads.
pub type Mat4 = [f32; 16];

/// Runtime local transform, independent of the preserved document numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalPose {
    pub t: [f32; 3],
    pub r: [f32; 4],
    pub s: [f32; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Pose {
    pub local: Vec<LocalPose>,
    pub model: Vec<Mat4>,
    pub skin: Vec<Mat4>,
}

fn error(i: usize, message: impl Into<String>) -> AnimError {
    AnimError::new("rig.schema", format!("/bones/{i}/rest"), message)
}

fn matrix(local: LocalPose, i: usize) -> Result<Mat4, AnimError> {
    trs_compose(local.t, local.r, local.s).map_err(|e| error(i, e.to_string()))
}

fn normalized(mut local: LocalPose, i: usize) -> Result<LocalPose, AnimError> {
    local.r = normalize(local.r).map_err(|e| error(i, e.to_string()))?;
    Ok(local)
}

fn finite(m: Mat4, i: usize) -> Result<Mat4, AnimError> {
    if m.iter().all(|x| x.is_finite()) {
        Ok(m)
    } else {
        Err(error(
            i,
            "matrix is not representable as finite runtime values",
        ))
    }
}

fn multiply(a: Mat4, b: Mat4, i: usize) -> Result<Mat4, AnimError> {
    let mut out = [0.0; 16];
    for c in 0..4 {
        for r in 0..4 {
            out[c * 4 + r] = ((a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1])
                + a[8 + r] * b[c * 4 + 2])
                + a[12 + r] * b[c * 4 + 3];
        }
    }
    finite(out, i)
}

impl Rig {
    /// Convert source values once to runtime precision and normalize rotations.
    /// Conversion leaves canonical serialization unchanged.
    pub fn local_rest_pose(&self) -> Result<Vec<LocalPose>, AnimError> {
        self.bones
            .iter()
            .enumerate()
            .map(|(i, bone)| {
                let local = normalized(
                    LocalPose {
                        t: bone.rest.t.map(|x| x as f32),
                        r: bone.rest.r.map(|x| x as f32),
                        s: bone.rest.s.map(|x| x as f32),
                    },
                    i,
                )?;
                if local.s.iter().any(|s| *s <= 0.0) {
                    return Err(error(i, "rest scale underflows runtime precision"));
                }
                matrix(local, i)?;
                Ok(local)
            })
            .collect()
    }

    /// Analytic inverse binds: inverse(local) * inverse(parent model).
    /// Each local inverse is S^-1 * R(conjugate) * T(-t). This retains
    /// hierarchy-induced shear without decomposing or inverting a model matrix.
    pub fn inverse_bind_matrices(&self) -> Result<Vec<Mat4>, AnimError> {
        let rest = self.local_rest_pose()?;
        let mut inverse = Vec::with_capacity(rest.len());
        for (i, local) in rest.iter().enumerate() {
            let [x, y, z, w] = local.r;
            let mut m = trs_compose([0.0; 3], [-x, -y, -z, w], [1.0; 3])
                .map_err(|e| error(i, e.to_string()))?;
            for row in 0..3 {
                let reciprocal = 1.0 / local.s[row];
                for col in 0..3 {
                    m[col * 4 + row] *= reciprocal;
                }
                m[12 + row] =
                    -((m[row] * local.t[0] + m[4 + row] * local.t[1]) + m[8 + row] * local.t[2]);
            }
            let m = finite(m, i)?;
            inverse.push(match self.bones[i].parent {
                Some(parent) if parent < i => multiply(m, inverse[parent], i)?,
                None => m,
                _ => return Err(error(i, "parent must precede child")),
            });
        }
        Ok(inverse)
    }

    /// Compose parent-first models and model * inverse_bind skin matrices.
    /// Local transforms must be in S16 bone order; rotations are normalized.
    pub fn pose(&self, local: &[LocalPose]) -> Result<Pose, AnimError> {
        if local.len() != self.bones.len() {
            return Err(AnimError::new(
                "rig.schema",
                "/bones",
                "pose length mismatch",
            ));
        }
        let inverse = self.inverse_bind_matrices()?;
        let mut pose = Pose {
            local: Vec::with_capacity(local.len()),
            model: Vec::with_capacity(local.len()),
            skin: Vec::with_capacity(local.len()),
        };
        for (i, &local) in local.iter().enumerate() {
            let local = normalized(local, i)?;
            let m = matrix(local, i)?;
            let m = match self.bones[i].parent {
                Some(parent) => multiply(pose.model[parent], m, i)?,
                None => m,
            };
            pose.local.push(local);
            pose.model.push(m);
            pose.skin.push(multiply(m, inverse[i], i)?);
        }
        Ok(pose)
    }

    pub fn rest_pose(&self) -> Result<Pose, AnimError> {
        self.pose(&self.local_rest_pose()?)
    }
}
