//! S20 `two_bone`: analytic pole-controlled IK for an S16 `twoBone` chain.
//!
//! The chain root and middle bone rotate about their origins (law of cosines) so the
//! tip origin reaches the world target, clamped to the reachable shell. The bend
//! plane follows the pole, a direction in the chain root's parent space; without a
//! usable pole it keeps the current bend, then any stable perpendicular. The solved
//! local rotations are blended from the input by `weight` and clamped by S16 limits.
//! Segment lengths are preserved exactly when the chain parent and the two rotated
//! bones have uniform scale.

use crate::limits::clamp_rotation;
use crate::{IK_CHAIN, IK_TARGET_NOT_FINITE, IkError, IkPose};
use crate::{finite_target, finite_weight, validate_chain};
use raa_anim::rig::{ChainSolver, IkChain};
use raa_math::{Quat, Vec3};

const EPS: f32 = 1e-6;

/// Unit component of `v` perpendicular to unit `dir`, or None when degenerate.
fn perpendicular(v: Vec3, dir: Vec3) -> Option<Vec3> {
    let p = v - dir * v.dot(dir);
    (p.length() > 1e-4 * v.length().max(EPS)).then(|| p.normalize_or_zero())
}

/// Solve `chain` toward the world `target`. `pole` is a direction in the root bone's
/// parent space (S16 `ikChains[].pole`); None keeps the current bend plane. `weight`
/// is clamped to [0, 1]; 0 leaves the pose unchanged.
pub fn two_bone(
    ik: &mut IkPose,
    chain: &IkChain,
    target: [f32; 3],
    pole: Option<[f32; 3]>,
    weight: f32,
) -> Result<(), IkError> {
    validate_chain(ik.rig(), &chain.bones, ChainSolver::TwoBone)?;
    let target = finite_target(target)?;
    let weight = finite_weight(weight)?;
    let pole = pole.map(finite_target).transpose();
    let pole = pole.map_err(|_| IkError::new(IK_TARGET_NOT_FINITE, "", "pole is not finite"))?;
    if weight == 0.0 {
        return Ok(());
    }
    let [root, mid, tip] = [chain.bones[0], chain.bones[1], chain.bones[2]];
    let a = ik.world_position(root)?;
    let b = ik.world_position(mid)?;
    let c = ik.world_position(tip)?;
    let (l1, l2) = ((b - a).length(), (c - b).length());
    if !(l1 > EPS && l2 > EPS) {
        return Err(IkError::new(
            IK_CHAIN,
            chain.name.as_str(),
            "zero-length segment",
        ));
    }

    // Reach direction: toward the target, else along the current chain.
    let to_target = target - a;
    let dir = [to_target, c - a, b - a]
        .into_iter()
        .find(|v| v.length() > EPS)
        .unwrap_or(Vec3::Y)
        .normalize_or_zero();
    let lo = (l1 - l2).abs().max(EPS);
    let dist = to_target.length().clamp(lo, l1 + l2);

    // Bend side: pole (parent space -> world), current knee, stable fallback.
    let pole_world = match pole {
        Some(p) => Some(ik.parent_model(root)?.transform_vector(p)),
        None => None,
    };
    let fallback = if dir.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let side = [pole_world, Some(b - a), Some(fallback)]
        .into_iter()
        .flatten()
        .find_map(|v| perpendicular(v, dir))
        .unwrap_or(Vec3::Z);

    let cos_a = ((l1 * l1 + dist * dist - l2 * l2) / (2.0 * l1 * dist)).clamp(-1.0, 1.0);
    let sin_a = (1.0 - cos_a * cos_a).max(0.0).sqrt();
    let knee = a + dir * (l1 * cos_a) + side * (l1 * sin_a);
    let reach = a + dir * dist;

    let start = [root, mid].map(|bone| Quat::from_array(ik.pose().local[bone].r));
    ik.rotate_toward(root, b - a, knee - a, 1.0)?;
    let (b, c) = (ik.world_position(mid)?, ik.world_position(tip)?);
    ik.rotate_toward(mid, c - b, reach - b, 1.0)?;

    for (i, bone) in [root, mid].into_iter().enumerate() {
        let solved = Quat::from_array(ik.pose().local[bone].r);
        let mut r = Quat::slerp(start[i], solved, weight).to_array();
        let rig = ik.rig();
        if let Some(limit) = rig.limits.iter().find(|l| l.bone == bone) {
            let rest = rig.bones[bone].rest.r.map(|v| v as f32);
            let (min, max) = (limit.min.map(|v| v as f32), limit.max.map(|v| v as f32));
            r = clamp_rotation(rest, r, min, max);
        }
        ik.set_local_rotation(bone, Quat::from_array(r))?;
    }
    Ok(())
}
