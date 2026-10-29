use raa_anim::Rig;
use raa_anim::rig::{ChainSolver, IkChain};
use raa_ik::limits::to_euler_xyz_deg;
use raa_ik::{IkPose, fabrik, hand_reach, look_at};
use raa_math::Vec3;

/// Arm a-b-c-d rises along +Y from (0,1,0) in 0.5 segments; "head" sits on root.
fn rig(limits: &str) -> Rig {
    Rig::load(&format!(
        r#"{{"format":"raa.rig","version":1,"name":"t","bones":[
 {{"name":"root","parent":null,"rest":{{"t":[0,0,0],"r":[0,0,0,1],"s":[1,1,1]}}}},
 {{"name":"a","parent":"root","rest":{{"t":[0,1,0],"r":[0,0,0,1],"s":[1,1,1]}}}},
 {{"name":"b","parent":"a","rest":{{"t":[0,0.5,0],"r":[0,0,0,1],"s":[1,1,1]}}}},
 {{"name":"c","parent":"b","rest":{{"t":[0,0.5,0],"r":[0,0,0,1],"s":[1,1,1]}}}},
 {{"name":"d","parent":"c","rest":{{"t":[0,0.5,0],"r":[0,0,0,1],"s":[1,1,1]}}}},
 {{"name":"head","parent":"root","rest":{{"t":[2,0,0],"r":[0,0,0,1],"s":[1,1,1]}}}}],
 "roles":{{}},"ikChains":[],"limits":[{limits}],"attach":{{}}}}"#
    ))
    .unwrap()
}

fn chain(solver: ChainSolver, bones: &[usize]) -> IkChain {
    IkChain { name: "arm".into(), solver, bones: bones.to_vec(), pole: None, iterations: 16 }
}

fn rest(rig: &Rig) -> IkPose<'_> {
    IkPose::new(rig, &rig.local_rest_pose().unwrap()).unwrap()
}

fn tip(ik: &IkPose) -> Vec3 {
    let p = ik.world_position(4).unwrap();
    assert!(p.is_finite());
    p
}

const T: [f32; 3] = [0.7, 1.4, 0.3];
const ARM: [usize; 4] = [1, 2, 3, 4];

#[test]
fn zero_weight_is_unchanged() {
    let rig = rig("");
    for solver in [ChainSolver::Fabrik, ChainSolver::Ccd] {
        let mut ik = rest(&rig);
        hand_reach(&mut ik, &chain(solver, &ARM), T, 0.0).unwrap();
        assert_eq!(ik.pose(), rest(&rig).pose());
    }
    let mut ik = rest(&rig);
    hand_reach(&mut ik, &chain(ChainSolver::TwoBone, &[1, 2, 3]), T, 0.0).unwrap();
    assert_eq!(ik.pose(), rest(&rig).pose());
}

#[test]
fn full_weight_is_the_full_solution() {
    let rig = rig("");
    let mut ik = rest(&rig);
    hand_reach(&mut ik, &chain(ChainSolver::Fabrik, &ARM), T, 1.0).unwrap();
    let mut direct = rest(&rig);
    fabrik(&mut direct, &chain(ChainSolver::Fabrik, &ARM), T, 16, 1e-4).unwrap();
    assert_eq!(ik.pose(), direct.pose());
    assert!((tip(&ik) - Vec3::new(T[0], T[1], T[2])).length() < 1e-3);
    assert_eq!(ik.pose().local[5], rest(&rig).pose().local[5]);
}

#[test]
fn intermediate_weight_blends() {
    let rig = rig("");
    let start = tip(&rest(&rig));
    let target = Vec3::new(T[0], T[1], T[2]);
    let mut ik = rest(&rig);
    hand_reach(&mut ik, &chain(ChainSolver::Ccd, &ARM), T, 0.5).unwrap();
    let mid = tip(&ik);
    assert!(mid != start);
    assert!((mid - target).length() > 1e-2);
    assert!((mid - target).length() < (start - target).length());
}

#[test]
fn reach_respects_limits() {
    let rig = rig(r#"{"bone":"a","min":[-5,-5,-5],"max":[5,5,5]}"#);
    let mut ik = rest(&rig);
    hand_reach(&mut ik, &chain(ChainSolver::Fabrik, &ARM), T, 0.7).unwrap();
    for e in to_euler_xyz_deg(ik.pose().local[1].r) {
        assert!(e.abs() <= 5.0 + 1e-3);
    }
}

fn head_axis(ik: &IkPose) -> Vec3 {
    let a = ik.model(5).unwrap().transform_vector(Vec3::Z);
    assert!(a.is_finite());
    a
}

#[test]
fn look_at_full_and_capped() {
    let rig = rig("");
    let mut ik = rest(&rig);
    look_at(&mut ik, 5, [3.0, 0.0, 0.0], [0.0, 0.0, 1.0], 120.0).unwrap();
    assert!((head_axis(&ik) - Vec3::X).length() < 1e-4);
    let mut ik = rest(&rig);
    look_at(&mut ik, 5, [3.0, 0.0, 0.0], [0.0, 0.0, 1.0], 30.0).unwrap();
    let deg = head_axis(&ik).dot(Vec3::Z).clamp(-1.0, 1.0).acos().to_degrees();
    assert!((deg - 30.0).abs() < 1e-2, "{deg}");
    assert_eq!(ik.pose().local[1], rest(&rig).pose().local[1]);
}

#[test]
fn look_at_respects_limits() {
    let rig = rig(r#"{"bone":"head","min":[-10,-10,-10],"max":[10,10,10]}"#);
    let mut ik = rest(&rig);
    look_at(&mut ik, 5, [3.0, 0.0, 0.0], [0.0, 0.0, 1.0], 90.0).unwrap();
    for e in to_euler_xyz_deg(ik.pose().local[5].r) {
        assert!(e.abs() <= 10.0 + 1e-3);
    }
}

#[test]
fn look_at_degeneracies_are_finite() {
    let rig = rig("");
    let mut ik = rest(&rig);
    look_at(&mut ik, 5, [2.0, 0.0, 0.0], [0.0, 0.0, 1.0], 90.0).unwrap();
    assert_eq!(ik.pose(), rest(&rig).pose());
    look_at(&mut ik, 5, [2.0, 0.0, 5.0], [0.0, 0.0, 0.0], 90.0).unwrap();
    assert_eq!(ik.pose(), rest(&rig).pose());
    // Opposite direction: capped turn is deterministic and finite.
    look_at(&mut ik, 5, [2.0, 0.0, -5.0], [0.0, 0.0, 1.0], 45.0).unwrap();
    let deg = head_axis(&ik).dot(Vec3::Z).clamp(-1.0, 1.0).acos().to_degrees();
    assert!((deg - 45.0).abs() < 1e-2, "{deg}");
    let mut full = rest(&rig);
    look_at(&mut full, 5, [2.0, 0.0, -5.0], [0.0, 0.0, 1.0], 180.0).unwrap();
    assert!((head_axis(&full) + Vec3::Z).length() < 1e-3);
    let mut reach = rest(&rig);
    hand_reach(&mut reach, &chain(ChainSolver::Fabrik, &ARM), [0.0, 1.0, 0.0], 1.0).unwrap();
    tip(&reach);
}
