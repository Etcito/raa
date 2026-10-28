use raa_anim::Rig;
use raa_anim::rig::{ChainSolver, IkChain};
use raa_ik::limits::to_euler_xyz_deg;
use raa_ik::{CcdOutcome, IK_BONE_UNKNOWN, IK_CHAIN, IK_TARGET_NOT_FINITE, IkPose, ccd};
use raa_math::Vec3;

/// Bones 1..=4 rise along +Y from (0,1,0) in 0.5 segments; chains name prefixes of them.
fn rig(limits: &str) -> Rig {
    Rig::load(&format!(
        r#"{{"format":"raa.rig","version":1,"name":"t","bones":[
 {{"name":"root","parent":null,"rest":{{"t":[0,0,0],"r":[0,0,0,1],"s":[1,1,1]}}}},
 {{"name":"a","parent":"root","rest":{{"t":[0,1,0],"r":[0,0,0,1],"s":[1,1,1]}}}},
 {{"name":"b","parent":"a","rest":{{"t":[0,0.5,0],"r":[0,0,0,1],"s":[1,1,1]}}}},
 {{"name":"c","parent":"b","rest":{{"t":[0,0.5,0],"r":[0,0,0,1],"s":[1,1,1]}}}},
 {{"name":"d","parent":"c","rest":{{"t":[0,0.5,0],"r":[0,0,0,1],"s":[1,1,1]}}}}],
 "roles":{{}},"ikChains":[],"limits":[{limits}],"attach":{{}}}}"#
    ))
    .unwrap()
}

fn chain(bones: &[usize]) -> IkChain {
    IkChain {
        name: "arm".into(),
        solver: ChainSolver::Ccd,
        bones: bones.to_vec(),
        pole: None,
        iterations: 16,
    }
}

fn solve<'r>(
    rig: &'r Rig,
    bones: &[usize],
    target: Vec3,
    it: usize,
    tol: f32,
) -> (IkPose<'r>, CcdOutcome) {
    let mut ik = IkPose::new(rig, &rig.local_rest_pose().unwrap()).unwrap();
    let out = ccd(&mut ik, &chain(bones), target.to_array(), it, tol).unwrap();
    (ik, out)
}

fn at(ik: &IkPose, bone: usize) -> Vec3 {
    let p = ik.world_position(bone).unwrap();
    assert!(p.is_finite(), "non-finite joint {p:?}");
    p
}

fn close(a: Vec3, b: Vec3, eps: f32) {
    assert!((a - b).length() < eps, "{a:?} vs {b:?}");
}

fn lengths(ik: &IkPose, bones: &[usize]) {
    assert!((at(ik, bones[0]) - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6);
    for w in bones.windows(2) {
        assert!(((at(ik, w[1]) - at(ik, w[0])).length() - 0.5).abs() < 1e-4);
    }
}

const T: Vec3 = Vec3::new(0.6, 1.3, 0.2);

/// Expected values from `/refs/sketchbook/IKSolver.ts` `solveCCD` (node, f64, zero
/// effector offset, damping 0.5, tolerance 0.001): local deltas per bone and the tip.
#[test]
fn matches_the_sketchbook_reference() {
    let rig = rig("");
    let cases: [(&[usize], usize, &[[f32; 4]], [f32; 3]); 3] = [
        (
            &[1, 2],
            1,
            &[[0.08799027, 0.0, -0.2639708, 0.96050879]],
            [0.25354628, 1.42257713, 0.08451543],
        ),
        (
            &[1, 2],
            2,
            &[[0.12979842, 0.0, -0.38939527, 0.91187921]],
            [0.35508145, 1.33152369, 0.11836048],
        ),
        (
            &[1, 2, 3],
            1,
            &[
                [0.05183328, 0.0, -0.15549985, 0.98647509],
                [0.14300892, 0.0, -0.42902676, 0.89189937],
            ],
            [0.60613697, 1.62230219, 0.20204566],
        ),
    ];
    for (bones, it, deltas, tip) in cases {
        let (ik, out) = solve(&rig, bones, T, it, 0.001);
        assert_eq!((out.converged, out.passes), (false, it));
        for (bone, q) in bones.iter().zip(deltas) {
            let r = ik.pose().local[*bone].r;
            for k in 0..4 {
                assert!((r[k] - q[k]).abs() < 1e-5, "bone {bone}: {r:?} vs {q:?}");
            }
        }
        // The tip bone is never rotated (its effector direction is zero).
        assert_eq!(
            ik.pose().local[bones[bones.len() - 1]].r,
            [0.0, 0.0, 0.0, 1.0]
        );
        close(at(&ik, bones[bones.len() - 1]), Vec3::from_array(tip), 1e-5);
        lengths(&ik, bones);
    }
}

const ARM: [usize; 4] = [1, 2, 3, 4];

#[test]
fn reachable_target_converges_and_stops_early() {
    let rig = rig("");
    let target = Vec3::new(0.7, 1.4, 0.3);
    let (ik, out) = solve(&rig, &ARM, target, 16, 1e-3);
    assert!(out.converged && out.passes < 16, "{out:?}");
    assert!(out.distance < 1e-3);
    close(at(&ik, 4), target, 1e-3);
    lengths(&ik, &ARM);
    // A target already at the tip stops before the first pass, pose unchanged.
    let (ik, out) = solve(&rig, &ARM, Vec3::new(0.0, 2.5, 0.0), 16, 1e-3);
    assert_eq!((out.converged, out.passes), (true, 0));
    assert_eq!(ik.into_pose(), rig.rest_pose().unwrap());
}

#[test]
fn unreachable_target_extends_toward_it() {
    let rig = rig("");
    let target = Vec3::new(4.0, 3.0, 2.0);
    let (ik, out) = solve(&rig, &ARM, target, 16, 1e-3);
    assert_eq!((out.converged, out.passes), (false, 16));
    let root = at(&ik, 1);
    let ideal = (target - root).length() - 1.5;
    assert!(
        out.distance >= ideal - 1e-4 && out.distance < ideal + 0.02,
        "{out:?}"
    );
    let dir = (target - root).normalize_or_zero();
    assert!((at(&ik, 4) - root).normalize_or_zero().dot(dir) > 0.999);
    lengths(&ik, &ARM);
}

#[test]
fn constrained_joints_stay_within_limits() {
    let limit = |b: &str| format!(r#"{{"bone":"{b}","min":[-10,-10,-10],"max":[10,10,10]}}"#);
    let rig = rig(&[limit("a"), limit("b"), limit("c")].join(","));
    let target = Vec3::new(0.7, 1.4, 0.3);
    let (ik, out) = solve(&rig, &ARM, target, 16, 1e-3);
    for bone in 1..=3 {
        for angle in to_euler_xyz_deg(ik.pose().local[bone].r) {
            assert!(angle.abs() <= 10.0 + 1e-3, "bone {bone}: {angle}");
        }
    }
    assert!(!out.converged && out.distance > 0.1, "{out:?}");
    lengths(&ik, &ARM);
    // Unconstrained, the same target is reached.
    assert!(solve(&self::rig(""), &ARM, target, 16, 1e-3).1.converged);
}

#[test]
fn iterations_are_exhausted_and_bounded() {
    let rig = rig("");
    let target = Vec3::new(0.7, 1.4, 0.3);
    let (_, one) = solve(&rig, &ARM, target, 1, 1e-3);
    let (_, four) = solve(&rig, &ARM, target, 4, 1e-3);
    assert_eq!((one.converged, one.passes), (false, 1));
    assert_eq!((four.converged, four.passes), (false, 4));
    assert!(four.distance < one.distance);
    let (ik, zero) = solve(&rig, &ARM, target, 0, 1e-3);
    assert_eq!((zero.converged, zero.passes), (false, 0));
    assert_eq!(ik.into_pose(), rig.rest_pose().unwrap());
    // A zero tolerance never converges and runs every pass.
    assert_eq!(solve(&rig, &ARM, target, 16, 0.0).1.passes, 16);
    let mut ik = IkPose::new(&rig, &rig.local_rest_pose().unwrap()).unwrap();
    let err = ccd(&mut ik, &chain(&ARM), target.to_array(), 17, 1e-3);
    assert_eq!(err.unwrap_err().code, IK_CHAIN);
}

#[test]
fn invalid_chains_and_inputs_are_refused() {
    let rig = rig("");
    let mut ik = IkPose::new(&rig, &rig.local_rest_pose().unwrap()).unwrap();
    let mut code = |bones: &[usize], t: [f32; 3], tol: f32| {
        ccd(&mut ik, &chain(bones), t, 8, tol).unwrap_err().code
    };
    let t = T.to_array();
    assert_eq!(code(&[1], t, 1e-3), IK_CHAIN);
    assert_eq!(code(&[1; 9], t, 1e-3), IK_CHAIN);
    assert_eq!(code(&[1, 3], t, 1e-3), IK_CHAIN);
    assert_eq!(code(&[2, 1], t, 1e-3), IK_CHAIN);
    assert_eq!(code(&[1, 99], t, 1e-3), IK_BONE_UNKNOWN);
    assert_eq!(code(&ARM, [f32::NAN, 0.0, 0.0], 1e-3), IK_TARGET_NOT_FINITE);
    assert_eq!(code(&ARM, t, f32::INFINITY), IK_TARGET_NOT_FINITE);
    assert_eq!(ik.into_pose(), rig.rest_pose().unwrap());
}
