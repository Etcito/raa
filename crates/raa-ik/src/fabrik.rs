//! Sketchbook `IKSolver.ts::solveFABRIK`, with the S20 tip-origin effector.
//! Position passes and stopping order follow the reference. Rotation recovery
//! uses the updated parent pose (avoiding compounded original-world deltas),
//! then S16 rest-relative limits, rather than Sketchbook's delta constraints.

use crate::{IK_CHAIN, IK_TARGET_NOT_FINITE, IkError, IkPose, MAX_ITERATIONS};
use crate::{clamp_rotation, finite_target, validate_chain};
use raa_anim::rig::{ChainSolver, IkChain};
use raa_math::{Quat, Vec3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FabrikOutcome {
    /// A position pass converged and the reconstructed, constrained pose also
    /// reaches the target. The reference's unreachable branch remains false.
    pub converged: bool,
    /// Forward/backward pairs executed; unreachable extension uses zero passes.
    pub passes: usize,
    /// Distance from the recomputed pose's tip origin to the target.
    pub distance: f32,
}

fn direction(v: Vec3) -> Vec3 {
    let length = v.length();
    if length < 1e-10 { Vec3::ZERO } else { v / length }
}

/// Solve a 2..=8 bone chain. Iterations above 16 and non-finite tolerances are
/// refused; negative tolerance acts as zero. As in Sketchbook, zero iterations
/// skips reachable-target passes but still extends an unreachable chain.
pub fn fabrik(
    ik: &mut IkPose,
    chain: &IkChain,
    target: [f32; 3],
    iterations: usize,
    tol: f32,
) -> Result<FabrikOutcome, IkError> {
    validate_chain(ik.rig(), &chain.bones, ChainSolver::Fabrik)?;
    let target = finite_target(target)?;
    if iterations > MAX_ITERATIONS {
        return Err(IkError::new(IK_CHAIN, &chain.name, "iterations above 16"));
    }
    if !tol.is_finite() {
        return Err(IkError::new(IK_TARGET_NOT_FINITE, "", "tolerance is not finite"));
    }
    let tol = tol.max(0.0);
    let mut positions: Vec<Vec3> = chain
        .bones
        .iter()
        .map(|&b| ik.world_position(b))
        .collect::<Result<_, _>>()?;
    let lengths: Vec<f32> = positions.windows(2).map(|p| (p[1] - p[0]).length()).collect();
    let root = positions[0];
    let last = positions.len() - 1;
    let mut outcome = FabrikOutcome {
        converged: false,
        passes: 0,
        distance: 0.0,
    };
    if (target - root).length() > lengths.iter().sum::<f32>() - 1e-6 {
        let dir = direction(target - root);
        for i in 0..last {
            positions[i + 1] = positions[i] + dir * lengths[i];
        }
    } else {
        for _ in 0..iterations {
            positions[last] = target;
            for i in (0..last).rev() {
                let dir = direction(positions[i] - positions[i + 1]);
                positions[i] = positions[i + 1] + dir * lengths[i];
            }
            positions[0] = root;
            for i in 0..last {
                let dir = direction(positions[i + 1] - positions[i]);
                positions[i + 1] = positions[i] + dir * lengths[i];
            }
            outcome.passes += 1;
            if (positions[last] - target).length() < tol {
                outcome.converged = true;
                break;
            }
        }
    }
    // Preserve translations/scales: recovered local rotations regenerate the
    // actual pose, keeping segment lengths even when a position pass degenerates.
    for i in 0..last {
        let bone = chain.bones[i];
        let from = ik.world_position(chain.bones[i + 1])? - ik.world_position(bone)?;
        let to = positions[i + 1] - positions[i];
        if from.length() < 1e-10 || to.length() < 1e-10 {
            continue;
        }
        ik.rotate_toward(bone, from, to, 1.0)?;
        if let Some(limit) = ik.rig().limits.iter().find(|l| l.bone == bone) {
            let rotation = clamp_rotation(
                ik.rig().bones[bone].rest.r.map(|v| v as f32),
                ik.pose().local[bone].r,
                limit.min.map(|v| v as f32),
                limit.max.map(|v| v as f32),
            );
            ik.set_local_rotation(bone, Quat::from_array(rotation))?;
        }
    }
    outcome.distance = (ik.world_position(chain.bones[last])? - target).length();
    outcome.converged &= outcome.distance < tol;
    Ok(outcome)
}
