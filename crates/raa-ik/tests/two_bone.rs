use raa_anim::Rig;
use raa_ik::limits::to_euler_xyz_deg;
use raa_ik::{IK_TARGET_NOT_FINITE, IkPose, chain, two_bone};
use raa_math::Vec3;

/// `root` carries the chain parent transform; the leg hangs along -Y in 0.5 segments.
fn rig(root: &str, limits: &str) -> Rig {
    Rig::load(&format!(
        r#"{{"format":"raa.rig","version":1,"name":"t","bones":[
 {{"name":"root","parent":null,"rest":{root}}},
 {{"name":"hip","parent":"root","rest":{{"t":[0,1,0],"r":[0,0,0,1],"s":[1,1,1]}}}},
 {{"name":"knee","parent":"hip","rest":{{"t":[0,-0.5,0],"r":[0,0,0,1],"s":[1,1,1]}}}},
 {{"name":"foot","parent":"knee","rest":{{"t":[0,-0.5,0],"r":[0,0,0,1],"s":[1,1,1]}}}}],
 "roles":{{}},"ikChains":[{{"name":"leg","solver":"twoBone","bones":["hip","knee","foot"],
 "pole":[0,0,1],"iterations":1}}],"limits":[{limits}],"attach":{{}}}}"#
    ))
    .unwrap()
}

const IDENTITY: &str = r#"{"t":[0,0,0],"r":[0,0,0,1],"s":[1,1,1]}"#;
// 90 degrees about +Y (local +Z -> world +X), uniform scale 2.
const TURNED: &str = r#"{"t":[2,0,1],"r":[0,0.7071067811865476,0,0.7071067811865476],"s":[2,2,2]}"#;
const POLE: Option<[f32; 3]> = Some([0.0, 0.0, 1.0]);

struct Solved {
    hip: Vec3,
    knee: Vec3,
    foot: Vec3,
}

fn solve(rig: &Rig, offset: Vec3, pole: Option<[f32; 3]>, weight: f32) -> (Solved, IkPose<'_>) {
    let mut ik = IkPose::new(rig, &rig.local_rest_pose().unwrap()).unwrap();
    let leg = chain(rig, "leg").unwrap();
    let target = ik.world_position(1).unwrap() + offset;
    two_bone(&mut ik, leg, target.to_array(), pole, weight).unwrap();
    let at = |i| ik.world_position(i).unwrap();
    let s = Solved {
        hip: at(1),
        knee: at(2),
        foot: at(3),
    };
    for p in [s.hip, s.knee, s.foot] {
        assert!(p.is_finite(), "non-finite joint {p:?}");
    }
    (s, ik)
}

fn close(a: Vec3, b: Vec3, eps: f32) {
    assert!((a - b).length() < eps, "{a:?} vs {b:?}");
}

fn lengths(s: &Solved, l: f32) {
    assert!(((s.knee - s.hip).length() - l).abs() < 1e-4);
    assert!(((s.foot - s.knee).length() - l).abs() < 1e-4);
}

#[test]
fn reachable_target_is_hit_with_lengths_preserved() {
    let rig = rig(IDENTITY, "");
    for offset in [Vec3::new(0.0, -0.6, 0.2), Vec3::new(0.3, -0.2, -0.4)] {
        let (s, _) = solve(&rig, offset, POLE, 1.0);
        close(s.foot, s.hip + offset, 1e-4);
        lengths(&s, 0.5);
    }
}

#[test]
fn pole_selects_the_bend_side() {
    let rig = rig(IDENTITY, "");
    let down = Vec3::new(0.0, -0.7, 0.0);
    for pole in [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ] {
        let (s, _) = solve(&rig, down, Some(pole), 1.0);
        let bend = (s.knee - s.hip).normalize_or_zero();
        let side = bend - Vec3::new(0.0, bend.y, 0.0);
        assert!(
            side.dot(Vec3::from_array(pole)) > 0.5,
            "{pole:?} -> {bend:?}"
        );
        close(s.foot, s.hip + down, 1e-4);
    }
}

#[test]
fn pole_and_target_respect_a_transformed_parent() {
    let rig = rig(TURNED, "");
    let offset = Vec3::new(0.3, -1.2, 0.5);
    let (s, _) = solve(&rig, offset, POLE, 1.0);
    close(s.foot, s.hip + offset, 1e-4);
    lengths(&s, 1.0);
    // Parent-space +Z is world +X under the turned root.
    let reach = offset.normalize_or_zero();
    let bend = s.knee - s.hip;
    let side = bend - reach * bend.dot(reach);
    assert!(side.normalize_or_zero().x > 0.8, "{side:?}");
}

#[test]
fn unreachable_targets_straighten_without_nans() {
    let rig = rig(IDENTITY, "");
    let far = Vec3::new(3.0, -8.0, 4.0);
    let (s, _) = solve(&rig, far, POLE, 1.0);
    close(s.foot, s.hip + far.normalize_or_zero(), 1e-4);
    lengths(&s, 0.5);
    // Target at the root: the chain folds and stays finite.
    let (s, _) = solve(&rig, Vec3::ZERO, POLE, 1.0);
    lengths(&s, 0.5);
}

#[test]
fn straight_chain_and_degenerate_poles_stay_finite() {
    let rig = rig(IDENTITY, "");
    let down = Vec3::new(0.0, -0.6, 0.0);
    for pole in [None, Some([0.0, 1.0, 0.0]), Some([0.0, 0.0, 0.0])] {
        let (s, _) = solve(&rig, down, pole, 1.0);
        close(s.foot, s.hip + down, 1e-4);
        lengths(&s, 0.5);
    }
    let (s, _) = solve(&rig, Vec3::new(0.0, -1.0, 0.0), None, 1.0);
    close(s.knee, s.hip + Vec3::new(0.0, -0.5, 0.0), 1e-3);
}

#[test]
fn zero_weight_is_identity_and_partial_weight_blends() {
    let rig = rig(IDENTITY, "");
    let offset = Vec3::new(0.2, -0.5, 0.3);
    let (_, ik) = solve(&rig, offset, POLE, 0.0);
    assert_eq!(ik.into_pose(), rig.rest_pose().unwrap());
    let (full, _) = solve(&rig, offset, POLE, 1.0);
    let (half, _) = solve(&rig, offset, POLE, 0.5);
    let rest_foot = half.hip + Vec3::new(0.0, -1.0, 0.0);
    assert!((half.foot - full.foot).length() > 1e-2);
    assert!((half.foot - rest_foot).length() > 1e-2);
    lengths(&half, 0.5);
}

#[test]
fn joint_limits_clamp_the_solution() {
    let limit = r#"{"bone":"knee","min":[-30,-30,-30],"max":[30,30,30]}"#;
    let rig = rig(IDENTITY, limit);
    let offset = Vec3::new(0.0, -0.4, 0.0);
    let (s, ik) = solve(&rig, offset, POLE, 1.0);
    for angle in to_euler_xyz_deg(ik.pose().local[2].r) {
        assert!(angle.abs() <= 30.0 + 1e-3, "{angle}");
    }
    assert!((s.foot - (s.hip + offset)).length() > 1e-2);
    lengths(&s, 0.5);
    // The unconstrained hip still bends toward the pole.
    assert!(s.knee.z > s.hip.z);
}

#[test]
fn non_finite_inputs_are_refused() {
    let rig = rig(IDENTITY, "");
    let mut ik = IkPose::new(&rig, &rig.local_rest_pose().unwrap()).unwrap();
    let leg = chain(&rig, "leg").unwrap();
    let nan = f32::NAN;
    let code = |r: Result<(), raa_ik::IkError>| r.unwrap_err().code;
    assert_eq!(
        code(two_bone(&mut ik, leg, [nan, 0.0, 0.0], POLE, 1.0)),
        IK_TARGET_NOT_FINITE
    );
    assert_eq!(
        code(two_bone(&mut ik, leg, [0.0; 3], Some([nan; 3]), 1.0)),
        IK_TARGET_NOT_FINITE
    );
    assert_eq!(
        code(two_bone(&mut ik, leg, [0.0; 3], POLE, nan)),
        IK_TARGET_NOT_FINITE
    );
}
