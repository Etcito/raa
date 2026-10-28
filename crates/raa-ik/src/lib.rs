//! S20 visual IK: corrects an S17 pose toward world targets supplied by the caller.
//!
//! IK runs after simulation and after deterministic graph evaluation; the sim never
//! consumes its output, so this crate is not on the sim path and depends on neither
//! raa-det nor raa-anim-graph.

use raa_anim::rig::{ChainSolver, IkChain};
use raa_anim::{AnimError, Rig};
use raa_math::Vec3;

pub mod ccd;
pub use ccd::{CcdOutcome, ccd};
pub mod limits;
pub use limits::{clamp_pose, clamp_rotation};
pub mod pose;
pub use pose::IkPose;
pub mod two_bone;
pub use two_bone::two_bone;

/// Refusals use the shared raa-anim error type with an `ik.*` code.
pub type IkError = AnimError;

pub const IK_CHAIN: &str = "ik.chain";
pub const IK_TARGET_NOT_FINITE: &str = "ik.target_not_finite";
pub const IK_BONE_UNKNOWN: &str = "ik.bone_unknown";

/// IK_RULE: iteration bound shared with S16 `ikChains[].iterations`.
pub const MAX_ITERATIONS: usize = 16;

/// Index of the bone named `name`, or `ik.bone_unknown`.
pub fn bone_index(rig: &Rig, name: &str) -> Result<usize, IkError> {
    rig.bones
        .iter()
        .position(|bone| bone.name == name)
        .ok_or_else(|| IkError::new(IK_BONE_UNKNOWN, name, "no bone with this name"))
}

/// The S16 chain named `name`, re-validated for its own solver.
pub fn chain<'r>(rig: &'r Rig, name: &str) -> Result<&'r IkChain, IkError> {
    let chain = rig.ik_chains.iter().find(|chain| chain.name == name);
    let chain = chain.ok_or_else(|| IkError::new(IK_CHAIN, name, "no IK chain"))?;
    validate_chain(rig, &chain.bones, chain.solver)?;
    Ok(chain)
}

/// Check a root-to-tip chain: known bones, each the direct child of the previous
/// one, and the S16 bone count for `solver`.
pub fn validate_chain(rig: &Rig, bones: &[usize], solver: ChainSolver) -> Result<(), IkError> {
    let count_ok = match solver {
        ChainSolver::TwoBone => bones.len() == 3,
        ChainSolver::Ccd | ChainSolver::Fabrik => (2..=8).contains(&bones.len()),
    };
    if !count_ok {
        return Err(IkError::new(IK_CHAIN, "", "wrong bone count for solver"));
    }
    for (i, &bone) in bones.iter().enumerate() {
        let (path, parent) = (format!("/{i}"), rig.bones.get(bone).map(|b| b.parent));
        match parent {
            None => return Err(IkError::new(IK_BONE_UNKNOWN, path, "no such bone")),
            Some(p) if i > 0 && p != Some(bones[i - 1]) => {
                return Err(IkError::new(IK_CHAIN, path, "chain order"));
            }
            _ => {}
        }
    }
    Ok(())
}

fn not_finite(what: &str) -> IkError {
    IkError::new(IK_TARGET_NOT_FINITE, "", format!("{what} is not finite"))
}

/// A finite world target, or `ik.target_not_finite`.
pub fn finite_target(target: [f32; 3]) -> Result<Vec3, IkError> {
    let v = Vec3::new(target[0], target[1], target[2]);
    v.is_finite()
        .then_some(v)
        .ok_or_else(|| not_finite("target"))
}

/// A finite weight clamped to [0, 1], or `ik.target_not_finite`.
pub fn finite_weight(weight: f32) -> Result<f32, IkError> {
    let w = weight.is_finite().then(|| weight.clamp(0.0, 1.0));
    w.ok_or_else(|| not_finite("weight"))
}
