use raa_anim::Rig;
use raa_anim::rig::{ChainSolver, IkChain};
use raa_ik::limits::to_euler_xyz_deg;
use raa_ik::{FabrikOutcome, IK_BONE_UNKNOWN, IK_CHAIN, IK_TARGET_NOT_FINITE, IkPose, fabrik};
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
        solver: ChainSolver::Fabrik,
        bones: bones.to_vec(),
        pole: None,
        iterations: 16,
    }
}

fn solve<'r>(rig: &'r Rig, bones: &[usize], target: Vec3, it: usize, tol: f32) -> (IkPose<'r>, FabrikOutcome) {
    let mut ik = IkPose::new(rig, &rig.local_rest_pose().unwrap()).unwrap();
    let out = fabrik(&mut ik, &chain(bones), target.to_array(), it, tol).unwrap();
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

const T: Vec3 = Vec3::new(0.7, 1.4, 0.3);
const ARM: [usize; 4] = [1, 2, 3, 4];

// Positions captured by executing /refs/sketchbook/IKSolver.ts in Node with
// zero effector offset. Compare geometry, since S20 recovers actual local poses.
#[test]
fn reference_pass_and_convergence() {
    let rig = rig("");
    for (iterations, expected) in [
        (
            1,
            [
                [-0.06489165, 1.49499054, -0.02781071],
                [0.34812697, 1.71427071, 0.14919727],
                [0.70333893, 1.39701788, 0.30143097],
            ],
        ),
        (
            16,
            [
                [-0.06656711, 1.49472713, -0.02852876],
                [0.34494386, 1.71733713, 0.14783308],
                [0.70005506, 1.39995079, 0.30002360],
            ],
        ),
    ] {
        let (ik, out) = solve(&rig, &ARM, T, iterations, 0.001);
        assert_eq!(out.converged, iterations == 16);
        assert!(out.passes <= iterations);
        if iterations == 16 {
            assert!(out.passes < 16);
        }
        for (bone, p) in (2..=4).zip(expected) {
            close(at(&ik, bone), Vec3::from_array(p), 1e-5);
        }
        lengths(&ik, &ARM);
        let rebuilt = rig.pose(&ik.pose().local).unwrap();
        for (a, b) in ik
            .pose()
            .model
            .iter()
            .chain(&ik.pose().skin)
            .flatten()
            .zip(rebuilt.model.iter().chain(&rebuilt.skin).flatten())
        {
            assert!((a - b).abs() < 1e-6);
        }
    }
    // Convergence is checked after the pass, even on the last allowed pass.
    assert!(solve(&rig, &ARM, T, 1, 0.01).1.converged);
}

#[test]
fn unreachable_extends_even_with_zero_iterations() {
    let rig = rig("");
    for it in [0, 16] {
        let target = Vec3::new(4.0, 3.0, 2.0);
        let (ik, out) = solve(&rig, &ARM, target, it, 0.001);
        assert_eq!((out.converged, out.passes), (false, 0));
        let dir = (target - at(&ik, 1)).normalize_or_zero();
        for bone in 2..=4 {
            close(at(&ik, bone), at(&ik, 1) + dir * (0.5 * (bone - 1) as f32), 1e-5);
        }
        lengths(&ik, &ARM);
    }
    // Reference regards the exact reach boundary as unreachable, even at the tip.
    assert!(!solve(&rig, &ARM, Vec3::new(0.0, 2.5, 0.0), 16, 0.001).1.converged);
}

#[test]
fn constrained_pose_is_recomputed_and_root_stays_fixed() {
    let limit = |b| format!(r#"{{"bone":"{b}","min":[-10,-10,-10],"max":[10,10,10]}}"#);
    let rig = rig(&[limit("a"), limit("b"), limit("c")].join(","));
    let (ik, out) = solve(&rig, &ARM, T, 16, 0.001);
    assert!(!out.converged && out.distance > 0.1);
    for bone in 1..=3 {
        for angle in to_euler_xyz_deg(ik.pose().local[bone].r) {
            assert!(angle.abs() <= 10.001);
        }
    }
    lengths(&ik, &ARM);
    let rest = rig.rest_pose().unwrap();
    for (a, b) in ik.pose().local.iter().zip(&rest.local) {
        assert_eq!((a.t, a.s), (b.t, b.s));
    }
    assert_eq!(ik.pose().local[0], rest.local[0]);
    assert_eq!(ik.pose().local[4], rest.local[4]);
    let rebuilt = rig.pose(&ik.pose().local).unwrap();
    for (a, b) in ik
        .pose()
        .model
        .iter()
        .chain(&ik.pose().skin)
        .flatten()
        .zip(rebuilt.model.iter().chain(&rebuilt.skin).flatten())
    {
        assert!((a - b).abs() < 1e-6);
    }
    assert!((out.distance - (at(&ik, 4) - T).length()).abs() < 1e-6);
}

#[test]
fn bounded_iterations_and_invalid_inputs() {
    let rig = rig("");
    let (ik, out) = solve(&rig, &ARM, T, 0, 0.001);
    assert_eq!((out.converged, out.passes), (false, 0));
    assert_eq!(ik.pose(), &rig.rest_pose().unwrap());
    for tol in [0.0, -1.0] {
        assert_eq!(solve(&rig, &ARM, T, 16, tol).1.passes, 16);
    }
    let mut ik = IkPose::new(&rig, &rig.local_rest_pose().unwrap()).unwrap();
    assert_eq!(
        fabrik(&mut ik, &chain(&ARM), T.to_array(), 17, 0.001).unwrap_err().code,
        IK_CHAIN
    );
    for (bones, expected) in [
        (&[1][..], IK_CHAIN),
        (&[1; 9][..], IK_CHAIN),
        (&[1, 3][..], IK_CHAIN),
        (&[1, 99][..], IK_BONE_UNKNOWN),
    ] {
        assert_eq!(
            fabrik(&mut ik, &chain(bones), T.to_array(), 16, 0.001)
                .unwrap_err()
                .code,
            expected
        );
    }
    for (target, tol) in [([f32::NAN, 0.0, 0.0], 0.001), (T.to_array(), f32::INFINITY)] {
        assert_eq!(
            fabrik(&mut ik, &chain(&ARM), target, 16, tol).unwrap_err().code,
            IK_TARGET_NOT_FINITE
        );
    }
    assert_eq!(ik.pose(), &rig.rest_pose().unwrap());
}
