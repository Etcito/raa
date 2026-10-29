//! S20 helpers built on the chain solvers: weighted hand reach and bounded look-at.

use crate::{IK_TARGET_NOT_FINITE, IkError, IkPose, MAX_ITERATIONS};
use crate::{ccd, clamp_rotation, fabrik, finite_target, finite_weight, two_bone};
use crate::validate_chain;
use raa_anim::rig::{ChainSolver, IkChain};
use raa_math::{Quat, Vec3};

/// Hand-reach stopping distance, in world units.
pub const REACH_TOL: f32 = 1e-4;

fn clamp_bone(ik: &mut IkPose, bone: usize) -> Result<(), IkError> {
    let Some(limit) = ik.rig().limits.iter().find(|l| l.bone == bone) else {
        return Ok(());
    };
    let rest = ik.rig().bones[bone].rest.r.map(|v| v as f32);
    let (min, max) = (limit.min.map(|v| v as f32), limit.max.map(|v| v as f32));
    let r = clamp_rotation(rest, ik.pose().local[bone].r, min, max);
    ik.set_local_rotation(bone, Quat::from_array(r))
}

/// Move the chain tip toward `target` with the chain's own S16 solver, blended by
/// `weight` (clamped to [0, 1]): 0 leaves the pose unchanged, 1 is the full solve.
/// Iterative solvers use `chain.iterations` (at most IK_RULE) and blend each chain
/// bone's local rotation by slerp, then re-apply S16 limits. Bones outside the
/// chain are untouched.
pub fn hand_reach(
    ik: &mut IkPose,
    chain: &IkChain,
    target: [f32; 3],
    weight: f32,
) -> Result<(), IkError> {
    validate_chain(ik.rig(), &chain.bones, chain.solver)?;
    finite_target(target)?;
    let weight = finite_weight(weight)?;
    if chain.solver == ChainSolver::TwoBone {
        two_bone(ik, chain, target, chain.pole.map(|p| p.map(|v| v as f32)), weight)?;
        for &bone in &chain.bones {
            clamp_bone(ik, bone)?;
        }
        return Ok(());
    }
    if weight == 0.0 {
        return Ok(());
    }
    let mut solved = ik.clone();
    let iterations = chain.iterations.min(MAX_ITERATIONS);
    match chain.solver {
        ChainSolver::Ccd => ccd(&mut solved, chain, target, iterations, REACH_TOL).map(|_| ())?,
        _ => fabrik(&mut solved, chain, target, iterations, REACH_TOL).map(|_| ())?,
    }
    if weight == 1.0 {
        *ik = solved;
        return Ok(());
    }
    for &bone in &chain.bones {
        let from = Quat::from_array(ik.pose().local[bone].r);
        let to = Quat::from_array(solved.pose().local[bone].r);
        ik.set_local_rotation(bone, Quat::slerp(from, to, weight))?;
        clamp_bone(ik, bone)?;
    }
    Ok(())
}

/// Rotate `bone` about its origin so its local `axis` points toward the world
/// `target`, turning by at most `max_deg` degrees (negative acts as 0), then apply
/// its S16 limit. A target at the bone origin or a zero axis leaves the pose
/// unchanged; an exactly opposite target turns about a fixed perpendicular axis
/// (see `shortest_arc`), so the result is deterministic and finite.
pub fn look_at(
    ik: &mut IkPose,
    bone: usize,
    target: [f32; 3],
    axis: [f32; 3],
    max_deg: f32,
) -> Result<(), IkError> {
    let target = finite_target(target)?;
    let axis = Vec3::new(axis[0], axis[1], axis[2]);
    if !axis.is_finite() || !max_deg.is_finite() {
        return Err(IkError::new(IK_TARGET_NOT_FINITE, "", "axis or angle is not finite"));
    }
    let model = ik.model(bone)?;
    let from = model.transform_vector(axis);
    let to = target - model.transform_point(Vec3::ZERO);
    if from.length() < 1e-10 || to.length() < 1e-10 {
        return Ok(());
    }
    let cos = from.normalize_or_zero().dot(to.normalize_or_zero()).clamp(-1.0, 1.0);
    let angle = cos.acos().to_degrees();
    let cap = max_deg.max(0.0);
    let weight = if angle <= cap { 1.0 } else { cap / angle };
    if weight > 0.0 {
        // The slerp in rotate_toward scales the arc angle linearly with weight.
        ik.rotate_toward(bone, from, to, weight)?;
    }
    clamp_bone(ik, bone)
}

