//! S20 `ccd`: cyclic coordinate descent, a port of Sketchbook `IKSolver.ts` `solveCCD`.
//!
//! Retained from the reference: the effector is the tip bone origin (zero effector
//! offset); before every pass the effector distance is compared with `tol` and the
//! solve stops once it is below; each pass walks the chain from tip to root and turns
//! every bone about its origin so the effector direction aims at the target, with the
//! reference's default slerp damping of 0.5; a bone whose effector or target direction
//! is degenerate is skipped (the tip bone always is).
//!
//! Port differences: the S16 rest-relative limit clamps each bone's local rotation
//! (the reference clamped the delta with its own constraint model), and descendant
//! positions come from the regenerated pose rather than from the reference's
//! propagation of original offsets by the latest delta only.

use crate::limits::clamp_rotation;
use crate::{IK_CHAIN, IK_TARGET_NOT_FINITE, IkError, IkPose, MAX_ITERATIONS};
use crate::{finite_target, validate_chain};
use raa_anim::rig::{ChainSolver, IkChain};
use raa_math::Quat;

/// Reference `solveCCD` default damping: each step applies half the aiming rotation.
pub const CCD_DAMPING: f32 = 0.5;

/// Reference threshold under which a direction is treated as zero.
const DEGENERATE: f32 = 1e-10;

/// What a `ccd` solve did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CcdOutcome {
    /// True when a pre-pass check found the effector within `tol` (reference
    /// `converged`); a target reached during the last allowed pass is not checked.
    pub converged: bool,
    /// Passes run, at most the requested iterations.
    pub passes: usize,
    /// Final distance from the tip bone origin to the target.
    pub distance: f32,
}

/// Re-apply the S16 limit of `bone`, if any, to its local rotation.
fn clamp_bone(ik: &mut IkPose, bone: usize) -> Result<(), IkError> {
    let rig = ik.rig();
    let Some(limit) = rig.limits.iter().find(|l| l.bone == bone) else {
        return Ok(());
    };
    let rest = rig.bones[bone].rest.r.map(|v| v as f32);
    let (min, max) = (limit.min.map(|v| v as f32), limit.max.map(|v| v as f32));
    let r = clamp_rotation(rest, ik.pose().local[bone].r, min, max);
    ik.set_local_rotation(bone, Quat::from_array(r))
}

/// Solve the root-to-tip `chain` (2 to 8 bones) so its tip origin approaches the
/// world `target`. `iterations` is bounded by IK_RULE (`MAX_ITERATIONS`); 0 leaves
/// the pose unchanged. `tol` is a world distance, negative values acting as 0.
pub fn ccd(
    ik: &mut IkPose,
    chain: &IkChain,
    target: [f32; 3],
    iterations: usize,
    tol: f32,
) -> Result<CcdOutcome, IkError> {
    validate_chain(ik.rig(), &chain.bones, ChainSolver::Ccd)?;
    let target = finite_target(target)?;
    if iterations > MAX_ITERATIONS {
        return Err(IkError::new(
            IK_CHAIN,
            chain.name.as_str(),
            format!("iterations above {MAX_ITERATIONS}"),
        ));
    }
    if !tol.is_finite() {
        return Err(IkError::new(
            IK_TARGET_NOT_FINITE,
            "",
            "tolerance is not finite",
        ));
    }
    let tol = tol.max(0.0);
    let tip = chain.bones[chain.bones.len() - 1];
    let mut outcome = CcdOutcome {
        converged: false,
        passes: 0,
        distance: 0.0,
    };
    for _ in 0..iterations {
        if (ik.world_position(tip)? - target).length() < tol {
            outcome.converged = true;
            break;
        }
        for &bone in chain.bones.iter().rev() {
            let origin = ik.world_position(bone)?;
            let to_effector = ik.world_position(tip)? - origin;
            let to_target = target - origin;
            if to_effector.length() < DEGENERATE || to_target.length() < DEGENERATE {
                continue;
            }
            ik.rotate_toward(bone, to_effector, to_target, CCD_DAMPING)?;
            clamp_bone(ik, bone)?;
        }
        outcome.passes += 1;
    }
    outcome.distance = (ik.world_position(tip)? - target).length();
    Ok(outcome)
}
