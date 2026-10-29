use raa_anim::Rig;
use raa_ik::limits::to_euler_xyz_deg;
use raa_ik::{GroundHit, IK_CHAIN, IK_TARGET_NOT_FINITE, IkPose, chain, foot_placement};
use raa_math::Vec3;

/// name parent t: root on the ground, pelvis at y 1 with a spine, straight legs of
/// 0.45 + 0.45 at x = +-0.2 (feet 0.1 above the root) and a prop outside the legs.
const BONES: &str = "root null 0,0,0;pelvis root 0,1,0;spine pelvis 0,0.3,0;\
thighL pelvis 0.2,0,0;shinL thighL 0,-0.45,0;footL shinL 0,-0.45,0;\
thighR pelvis -0.2,0,0;shinR thighR 0,-0.45,0;footR shinR 0,-0.45,0;prop root 1,0,0";
const LEG: &str = r#"{"name":"legS","solver":"twoBone","bones":["thighS","shinS","footS"],
 "pole":[0,0,1],"iterations":1}"#;
const UP: [f32; 3] = [0.0, 1.0, 0.0];
const PELVIS: usize = 1;
const FOOT_L: usize = 5;
const FOOT_R: usize = 8;

fn rig(limits: &str) -> Rig {
    let bones = BONES.split(';').map(|b| {
        let [name, parent, t] = [0, 1, 2].map(|i| b.split(' ').nth(i).unwrap());
        let parent = if parent == "null" { "null".into() } else { format!("\"{parent}\"") };
        format!(r#"{{"name":"{name}","parent":{parent},"rest":{{"t":[{t}],"r":[0,0,0,1],"s":[1,1,1]}}}}"#)
    });
    let chains = [LEG.replace('S', "L"), LEG.replace('S', "R")].join(",");
    Rig::load(&format!(
        r#"{{"format":"raa.rig","version":1,"name":"t","bones":[{}],"roles":{{}},
 "ikChains":[{chains}],"limits":[{limits}],"attach":{{}}}}"#,
        bones.collect::<Vec<_>>().join(",")
    ))
    .unwrap()
}

fn rest(rig: &Rig) -> IkPose<'_> {
    IkPose::new(rig, &rig.local_rest_pose().unwrap()).unwrap()
}

fn hit(point: [f32; 3], normal: [f32; 3]) -> GroundHit {
    GroundHit { point, normal }
}

fn at(ik: &IkPose, bone: usize) -> Vec3 {
    let p = ik.world_position(bone).unwrap();
    assert!(p.is_finite());
    p
}

fn up(ik: &IkPose, bone: usize) -> Vec3 {
    ik.model(bone)
        .unwrap()
        .transform_vector(Vec3::Y)
        .normalize_or_zero()
}

/// Place both feet: left contact `l`, right contact `r`, each (point, normal).
fn place(ik: &mut IkPose, l: ([f32; 3], [f32; 3]), r: ([f32; 3], [f32; 3])) -> f32 {
    let rig = ik.rig();
    let legs = [chain(rig, "legL").unwrap(), chain(rig, "legR").unwrap()];
    foot_placement(ik, &legs, &[hit(l.0, l.1), hit(r.0, r.1)]).unwrap()
}

fn close(a: Vec3, b: Vec3) -> bool {
    (a - b).length() < 1e-3
}

#[test]
fn unequal_step_heights_plant_both_feet() {
    let rig = rig("");
    let mut ik = rest(&rig);
    let drop = place(&mut ik, ([0.2, 0.15, 0.0], UP), ([-0.2, -0.1, 0.0], UP));
    assert!((drop - 0.1).abs() < 1e-5);
    assert!(close(at(&ik, FOOT_L), Vec3::new(0.2, 0.25, 0.0)));
    assert!(close(at(&ik, FOOT_R), Vec3::new(-0.2, 0.0, 0.0)));
    // The raised leg bends its knee toward the pole (+Z); flat feet stay level.
    assert!(at(&ik, 4).z > 0.1);
    assert!(close(up(&ik, FOOT_L), Vec3::Y));
}

#[test]
fn sloped_normals_tilt_feet_and_offset_targets() {
    let rig = rig("");
    let n = Vec3::new(0.0, 1.0, 1.0).normalize_or_zero();
    let m = Vec3::new(0.5, 1.0, 0.0).normalize_or_zero();
    let (l, r) = ([0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]);
    let mut ik = rest(&rig);
    place(&mut ik, (l, n.to_array()), (r, m.to_array()));
    for (foot, normal, p) in [(FOOT_L, n, l), (FOOT_R, m, r)] {
        let target = Vec3::from_array(p) + normal * 0.1;
        assert!(close(at(&ik, foot), target) && close(up(&ik, foot), normal));
    }
    // A foot limit bounds the alignment.
    let rig = self::rig(r#"{"bone":"footL","min":[-10,-10,-10],"max":[10,10,10]}"#);
    let mut ik = rest(&rig);
    place(&mut ik, (l, n.to_array()), (r, UP));
    let angles = to_euler_xyz_deg(ik.pose().local[FOOT_L].r);
    assert!(angles.iter().all(|a| a.abs() <= 10.0 + 1e-3), "{angles:?}");
    assert!((angles[0] - 10.0).abs() < 1e-2, "{angles:?}");
}

#[test]
fn pelvis_drops_for_the_lower_contact_only() {
    let rig = rig("");
    let input = rest(&rig);
    let mut ik = input.clone();
    let drop = place(&mut ik, ([0.2, -0.3, 0.0], UP), ([-0.2, -0.3, 0.0], UP));
    assert!((drop - 0.3).abs() < 1e-5);
    assert!(close(at(&ik, PELVIS), Vec3::new(0.0, 0.7, 0.0)));
    assert!(close(at(&ik, FOOT_L), Vec3::new(0.2, -0.2, 0.0)));
    let (out, before) = (&ik.pose().local, &input.pose().local);
    for bone in [0, 2, 9] {
        assert_eq!(out[bone], before[bone]);
    }
    assert_eq!((out[1].r, out[1].s), (before[1].r, before[1].s));
    assert_eq!(ik.pose().model[9], input.pose().model[9]);
    // Contacts above the feet never raise the pelvis.
    let mut ik = input.clone();
    let drop = place(&mut ik, ([0.2, 0.2, 0.0], UP), ([-0.2, 0.1, 0.0], UP));
    assert_eq!(drop, 0.0);
    assert_eq!(at(&ik, PELVIS), at(&input, PELVIS));
    assert!(close(at(&ik, FOOT_R), Vec3::new(-0.2, 0.2, 0.0)));
}

#[test]
fn unreachable_contacts_stretch_toward_them() {
    let rig = rig("");
    let mut ik = rest(&rig);
    let drop = place(&mut ik, ([3.0, -5.0, 1.0], UP), ([-0.2, -0.1, 0.0], UP));
    assert!((drop - 0.45).abs() < 1e-5, "drop is capped at half a leg");
    let (hip, foot) = (at(&ik, 3), at(&ik, FOOT_L));
    assert!(((foot - hip).length() - 0.9).abs() < 1e-3);
    let aim = (Vec3::new(3.0, -4.9, 1.0) - hip).normalize_or_zero();
    assert!(close((foot - hip).normalize_or_zero(), aim));
    assert!(close(at(&ik, FOOT_R), Vec3::new(-0.2, 0.0, 0.0)));
    assert!(ik.pose().skin.iter().flatten().all(|v| v.is_finite()));
}

#[test]
fn invalid_hits_and_legs_are_refused_without_change() {
    let rig = rig("");
    let input = rest(&rig);
    let (l, r) = (chain(&rig, "legL").unwrap(), chain(&rig, "legR").unwrap());
    let (ok, o) = (hit([0.2, 0.0, 0.0], UP), [0.0; 3]);
    let points = [[f32::NAN, 0.0, 0.0], [0.0, f32::INFINITY, 0.0]];
    let normals = [[0.0, f32::NAN, 0.0], o, [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]];
    let bad = points.map(|p| hit(p, UP)).into_iter();
    for h in bad.chain(normals.map(|n| hit(o, n))) {
        let mut ik = input.clone();
        let err = foot_placement(&mut ik, &[l, r], &[ok, h]).unwrap_err();
        assert_eq!(err.code, IK_TARGET_NOT_FINITE, "{h:?}");
        assert_eq!(ik.pose(), input.pose());
    }
    let mut spine = l.clone();
    spine.bones = vec![0, 1, 2];
    for legs in [vec![l], vec![l, l], vec![l, &spine]] {
        let mut ik = input.clone();
        let err = foot_placement(&mut ik, &legs, &[ok, ok]).unwrap_err();
        assert_eq!(err.code, IK_CHAIN);
        assert_eq!(ik.pose(), input.pose());
    }
    let mut ik = input.clone();
    assert_eq!(foot_placement(&mut ik, &[], &[]), Ok(0.0));
    assert_eq!(ik.pose(), input.pose());
}
