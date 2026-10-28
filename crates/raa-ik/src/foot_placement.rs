//! S20 `foot_placement`: plant S16 `twoBone` legs on caller-supplied ground hits.
//!
//! World up is +Y. Each foot (chain tip) keeps its input height `h` (clamped at 0)
//! above the rig root bone origin, so its target is `point + normal * h`. The legs'
//! shared parent (the pelvis) drops along -Y by the largest of each foot's step-down
//! and the drop its hip needs to reach the target, at most PELVIS_DROP_LIMIT of the
//! shortest leg; it never rises. Each leg is then solved from its recomputed
//! root with `two_bone` and the chain pole, and each foot keeps its animated world
//! orientation turned by the shortest arc from +Y to the hit normal. S16 limits
//! apply to every rotated bone. Only the pelvis translation and the leg chains'
//! local rotations change. The hits come from the caller's deterministic state:
//! nothing here queries the simulation, and the sim never reads the result.

use crate::pose::shortest_arc;
use crate::{IK_CHAIN, IK_TARGET_NOT_FINITE, IkError, IkPose};
use crate::{clamp_rotation, finite_target, two_bone, validate_chain};
use raa_anim::rig::{ChainSolver, IkChain};
use raa_math::{Quat, Vec3};

/// Largest pelvis drop, as a fraction of the shortest leg's l1 + l2.
pub const PELVIS_DROP_LIMIT: f32 = 0.5;

/// A ground contact for one leg: world point and world surface normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundHit {
    pub point: [f32; 3],
    pub normal: [f32; 3],
}

/// Unit normal of a hit, refusing non-finite, zero or downward-facing normals.
fn hit_normal(hit: &GroundHit) -> Result<Vec3, IkError> {
    let invalid = |m: &str| IkError::new(IK_TARGET_NOT_FINITE, "", m);
    let n = Vec3::from_array(hit.normal);
    let n = n.is_finite().then(|| n.normalize_or_zero());
    let n = n.ok_or_else(|| invalid("normal is not finite"))?;
    (n.y > 1e-6)
        .then_some(n)
        .ok_or_else(|| invalid("normal is zero or faces down"))
}

/// World rotation of `bone`: the product of local rotations from the rig root
/// (exact when the hierarchy scales uniformly).
fn world_rotation(ik: &IkPose, bone: usize) -> Quat {
    let (rig, local) = (ik.rig(), &ik.pose().local);
    let mut q = Quat::from_array(local[bone].r).normalize();
    let mut parent = rig.bones[bone].parent;
    while let Some(p) = parent {
        q = Quat::from_array(local[p].r).normalize() * q;
        parent = rig.bones[p].parent;
    }
    q
}

/// Place each leg of `legs` on the matching entry of `hits` (same length). Legs are
/// distinct `twoBone` chains whose roots share one parent bone, the pelvis. Returns
/// the applied pelvis drop in world units. On any error (including a non-finite
/// foot target, refused by `two_bone`) the pose is unchanged.
pub fn foot_placement(
    ik: &mut IkPose,
    legs: &[&IkChain],
    hits: &[GroundHit],
) -> Result<f32, IkError> {
    if legs.len() != hits.len() {
        return Err(IkError::new(IK_CHAIN, "", "one ground hit per leg"));
    }
    if legs.is_empty() {
        return Ok(0.0);
    }
    let rig = ik.rig();
    let mut pelvis = None;
    for (i, leg) in legs.iter().enumerate() {
        validate_chain(rig, &leg.bones, ChainSolver::TwoBone)?;
        let parent = rig.bones[leg.bones[0]].parent;
        let path = format!("/{i}");
        if parent.is_none() || pelvis.is_some_and(|p| Some(p) != parent) {
            return Err(IkError::new(IK_CHAIN, path, "legs do not share a pelvis"));
        }
        if legs[..i].iter().any(|other| other.bones[0] == leg.bones[0]) {
            return Err(IkError::new(IK_CHAIN, path, "duplicate leg"));
        }
        pelvis = parent;
    }
    let pelvis = pelvis.unwrap_or_default();
    let normals = hits.iter().map(hit_normal).collect::<Result<Vec<_>, _>>()?;
    let points = hits.iter().map(|h| finite_target(h.point));
    let points = points.collect::<Result<Vec<_>, _>>()?;

    // Foot targets and the shared drop, measured on the input pose.
    // S16 lists parents first, so bone 0 is the rig root: the ground reference.
    let ground = ik.world_position(0)?.y;
    let (mut targets, mut drop, mut shortest) = (Vec::new(), 0.0f32, f32::INFINITY);
    for (i, leg) in legs.iter().enumerate() {
        let [a, b, c] = [0, 1, 2].map(|k| ik.world_position(leg.bones[k]));
        let (a, b, c) = (a?, b?, c?);
        let target = points[i] + normals[i] * (c.y - ground).max(0.0);
        // Keep the animated leg height, and lower the hip enough to reach.
        let length = (b - a).length() + (c - b).length();
        let level = Vec3::new(target.x - a.x, 0.0, target.z - a.z).length();
        let reach = (length * length - level * level).max(0.0).sqrt();
        drop = drop.max(c.y - target.y).max(a.y - target.y - reach);
        shortest = shortest.min(length);
        targets.push(target);
    }
    let drop = drop.min(PELVIS_DROP_LIMIT * shortest).max(0.0);

    let mut work = ik.clone();
    if drop > 0.0 {
        let delta = work.world_to_parent_vector(pelvis, Vec3::new(0.0, -drop, 0.0))?;
        let mut local = work.pose().local.clone();
        let t = &mut local[pelvis].t;
        *t = [t[0] + delta.x, t[1] + delta.y, t[2] + delta.z];
        work = IkPose::new(rig, &local)?;
    }
    for (i, leg) in legs.iter().enumerate() {
        let foot = leg.bones[2];
        // Desired foot world rotation: the animated one turned onto the normal.
        let desired = shortest_arc(Vec3::Y, normals[i]) * world_rotation(ik, foot);
        let pole = leg.pole.map(|p| p.map(|v| v as f32));
        two_bone(&mut work, leg, targets[i].to_array(), pole, 1.0)?;
        let parent1 = world_rotation(&work, leg.bones[1]);
        let mut r = (parent1.conjugate() * desired).normalize().to_array();
        if let Some(limit) = rig.limits.iter().find(|l| l.bone == foot) {
            let rest = rig.bones[foot].rest.r.map(|v| v as f32);
            let (min, max) = (limit.min.map(|v| v as f32), limit.max.map(|v| v as f32));
            r = clamp_rotation(rest, r, min, max);
        }
        work.set_local_rotation(foot, Quat::from_array(r))?;
    }
    *ik = work;
    Ok(drop)
}
