//! Pose-space conversions shared by the solvers. Only local rotations of the bones a
//! solver touches change; translations, scales and every other bone are preserved,
//! and model and skin matrices are regenerated through `Rig::pose`.

use crate::{IK_BONE_UNKNOWN, IK_CHAIN, IkError};
use raa_anim::{LocalPose, Pose, Rig};
use raa_math::{Mat4, Quat, Vec3, Vec4};

/// raa-anim column-major `[f32; 16]` as a raa-math matrix.
pub fn to_math(m: &raa_anim::Mat4) -> Mat4 {
    let col = |c: usize| Vec4::new(m[c * 4], m[c * 4 + 1], m[c * 4 + 2], m[c * 4 + 3]);
    Mat4::from_cols(col(0), col(1), col(2), col(3))
}

/// Smallest rotation taking direction `a` to direction `b`; identity for zero input.
pub fn shortest_arc(a: Vec3, b: Vec3) -> Quat {
    let (a, b) = (a.normalize_or_zero(), b.normalize_or_zero());
    if a == Vec3::ZERO || b == Vec3::ZERO {
        return Quat::IDENTITY;
    }
    let d = a.dot(b);
    if d < -0.999_999 {
        let side = if a.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        return Quat::from_axis_angle(a.cross(side), core::f32::consts::PI);
    }
    let c = a.cross(b);
    Quat::from_xyzw(c.x, c.y, c.z, 1.0 + d).normalize()
}

/// A rig pose under IK correction.
#[derive(Debug, Clone)]
pub struct IkPose<'r> {
    rig: &'r Rig,
    pose: Pose,
}

impl<'r> IkPose<'r> {
    /// Start from S17 pose locals; model and skin are regenerated from them.
    pub fn new(rig: &'r Rig, local: &[LocalPose]) -> Result<Self, IkError> {
        Ok(Self {
            rig,
            pose: rig.pose(local)?,
        })
    }

    pub fn rig(&self) -> &'r Rig {
        self.rig
    }

    pub fn pose(&self) -> &Pose {
        &self.pose
    }

    pub fn into_pose(self) -> Pose {
        self.pose
    }

    fn check(&self, bone: usize) -> Result<(), IkError> {
        if bone < self.rig.bones.len() {
            Ok(())
        } else {
            Err(IkError::new(
                IK_BONE_UNKNOWN,
                format!("/{bone}"),
                "bone index out of range",
            ))
        }
    }

    pub fn model(&self, bone: usize) -> Result<Mat4, IkError> {
        self.check(bone)?;
        Ok(to_math(&self.pose.model[bone]))
    }

    /// Model matrix of the parent, or identity for the root.
    pub fn parent_model(&self, bone: usize) -> Result<Mat4, IkError> {
        self.check(bone)?;
        Ok(match self.rig.bones[bone].parent {
            Some(parent) => to_math(&self.pose.model[parent]),
            None => Mat4::IDENTITY,
        })
    }

    /// World position of the bone origin.
    pub fn world_position(&self, bone: usize) -> Result<Vec3, IkError> {
        Ok(self.model(bone)?.transform_point(Vec3::ZERO))
    }

    /// World direction expressed in the bone's parent space, where its local
    /// rotation acts. Exact under non-uniform parent scale.
    pub fn world_to_parent_vector(&self, bone: usize, v: Vec3) -> Result<Vec3, IkError> {
        let inverse = self.parent_model(bone)?.inverse().ok_or_else(|| {
            IkError::new(IK_CHAIN, format!("/{bone}"), "degenerate parent transform")
        })?;
        Ok(inverse.transform_vector(v))
    }

    /// Replace one local rotation and regenerate model and skin matrices.
    pub fn set_local_rotation(&mut self, bone: usize, r: Quat) -> Result<(), IkError> {
        self.check(bone)?;
        let mut local = self.pose.local.clone();
        local[bone].r = r.normalize().to_array();
        self.pose = self.rig.pose(&local)?;
        Ok(())
    }

    /// Rotate `bone` about its origin so the world direction `from` turns toward
    /// `to`, blended by `weight` in [0, 1]. Both are directions, not points.
    pub fn rotate_toward(
        &mut self,
        bone: usize,
        from: Vec3,
        to: Vec3,
        weight: f32,
    ) -> Result<(), IkError> {
        let arc = shortest_arc(
            self.world_to_parent_vector(bone, from)?,
            self.world_to_parent_vector(bone, to)?,
        );
        let delta = Quat::slerp(Quat::IDENTITY, arc, weight);
        let current = Quat::from_array(self.pose.local[bone].r);
        self.set_local_rotation(bone, delta * current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    const RIG: &str = r#"{"format":"raa.rig","version":1,"name":"t","bones":[
 {"name":"root","parent":null,"rest":{"t":[0,1,0],"r":[0,0,0.6,0.8],"s":[2,1,0.5]}},
 {"name":"a","parent":"root","rest":{"t":[0,1,0],"r":[0,0,0.6,0.8],"s":[1,1.5,1]}},
 {"name":"b","parent":"a","rest":{"t":[0,1,0],"r":[0,0,0.6,0.8],"s":[1,1,1]}},
 {"name":"c","parent":"b","rest":{"t":[0,1,0],"r":[0,0,0.6,0.8],"s":[1,1,1]}},
 {"name":"side","parent":"root","rest":{"t":[0,1,0],"r":[0,0,0.6,0.8],"s":[1,1,1]}}],
 "roles":{},"ikChains":[{"name":"leg","solver":"twoBone","bones":["a","b","c"],
 "pole":null,"iterations":4}],"limits":[],"attach":{}}"#;
    fn rig() -> Rig {
        Rig::load(RIG).unwrap()
    }
    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-4
    }
    fn code<T: core::fmt::Debug>(r: Result<T, IkError>) -> &'static str {
        r.unwrap_err().code
    }

    #[test]
    fn validates_bones_chains_and_targets() {
        let (rig, ccd, two) = (rig(), ChainSolver::Ccd, ChainSolver::TwoBone);
        assert_eq!(bone_index(&rig, "c"), Ok(3));
        assert_eq!(code(bone_index(&rig, "nope")), IK_BONE_UNKNOWN);
        assert_eq!(chain(&rig, "leg").unwrap().bones, [1, 2, 3]);
        assert_eq!(code(chain(&rig, "arm")), IK_CHAIN);
        assert_eq!(code(validate_chain(&rig, &[1, 2], two)), IK_CHAIN);
        assert_eq!(code(validate_chain(&rig, &[1], ccd)), IK_CHAIN);
        assert_eq!(code(validate_chain(&rig, &[0, 4, 3], two)), IK_CHAIN);
        assert_eq!(code(validate_chain(&rig, &[1, 9], ccd)), IK_BONE_UNKNOWN);
        assert!(validate_chain(&rig, &[0, 1, 2, 3], ChainSolver::Fabrik).is_ok());
        assert_eq!(
            code(finite_target([0.0, f32::NAN, 0.0])),
            IK_TARGET_NOT_FINITE
        );
        assert_eq!(finite_target([1.0, 2.0, 3.0]), Ok(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(code(finite_weight(f32::INFINITY)), IK_TARGET_NOT_FINITE);
        assert_eq!(finite_weight(2.0), Ok(1.0));
        let ik = IkPose::new(&rig, &rig.local_rest_pose().unwrap()).unwrap();
        assert_eq!(code(ik.world_position(7)), IK_BONE_UNKNOWN);
    }

    #[test]
    fn rotate_toward_aligns_under_scale_and_preserves_other_bones() {
        let rig = rig();
        let rest = rig.rest_pose().unwrap();
        let mut ik = IkPose::new(&rig, &rest.local).unwrap();
        assert_eq!(ik.pose(), &rest);
        let (a, c) = (ik.world_position(1).unwrap(), ik.world_position(3).unwrap());
        let to = Vec3::new(3.0, -1.0, 2.0) - a;
        ik.rotate_toward(1, c - a, to, 1.0).unwrap();
        let moved = ik.world_position(3).unwrap() - ik.world_position(1).unwrap();
        assert!(close(moved.normalize_or_zero(), to.normalize_or_zero()));
        assert!(close(ik.world_position(1).unwrap(), a));
        let pose = ik.into_pose();
        for i in [0, 2, 3, 4] {
            assert_eq!(pose.local[i], rest.local[i]);
        }
        let (l, r) = (pose.local[1], rest.local[1]);
        assert_eq!((l.t, l.s), (r.t, r.s));
        assert_eq!(pose, rig.pose(&pose.local).unwrap());
        assert_eq!(pose.model[4], rest.model[4]);
        assert!(close(
            shortest_arc(Vec3::Y, -Vec3::Y).rotate(Vec3::Y),
            -Vec3::Y
        ));
    }
}
