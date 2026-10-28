use raa_anim::Rig;
use raa_ik::limits::{from_euler_xyz_deg, to_euler_xyz_deg};
use raa_ik::{clamp_pose, clamp_rotation};

const REST: [f32; 4] = [0.0, 0.0, 0.6, 0.8];
const MIN: [f32; 3] = [-40.0, -70.0, -30.0];
const MAX: [f32; 3] = [40.0, 70.0, 30.0];

fn qmul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

fn rel_angles(q: [f32; 4]) -> [f32; 3] {
    to_euler_xyz_deg(qmul([-REST[0], -REST[1], -REST[2], REST[3]], q))
}

fn close(a: [f32; 3], b: [f32; 3]) {
    for i in 0..3 {
        assert!((a[i] - b[i]).abs() < 1e-3, "{a:?} vs {b:?}");
    }
}

fn posed(deg: [f32; 3]) -> [f32; 4] {
    qmul(REST, from_euler_xyz_deg(deg))
}

#[test]
fn euler_round_trip() {
    close(
        to_euler_xyz_deg(from_euler_xyz_deg([10.0, -20.0, 30.0])),
        [10.0, -20.0, 30.0],
    );
}

#[test]
fn each_axis_clamps_relative_to_nonidentity_rest() {
    let cases = [
        ([60.0, 0.0, 0.0], [40.0, 0.0, 0.0]),
        ([-60.0, 0.0, 0.0], [-40.0, 0.0, 0.0]),
        ([0.0, 80.0, 0.0], [0.0, 70.0, 0.0]),
        ([0.0, -80.0, 0.0], [0.0, -70.0, 0.0]),
        ([0.0, 0.0, 45.0], [0.0, 0.0, 30.0]),
        ([0.0, 0.0, -45.0], [0.0, 0.0, -30.0]),
        ([50.0, 10.0, -50.0], [40.0, 10.0, -30.0]),
    ];
    for (input, want) in cases {
        let out = clamp_rotation(REST, posed(input), MIN, MAX);
        close(rel_angles(out), want);
        assert!(out.iter().all(|c| c.is_finite()));
    }
}

#[test]
fn rest_itself_and_inside_values_are_unchanged() {
    assert_eq!(clamp_rotation(REST, REST, MIN, MAX), REST);
    let inside = posed([39.0, -69.0, 29.0]);
    assert_eq!(clamp_rotation(REST, inside, MIN, MAX), inside);
}

#[test]
fn boundary_values_stay_on_boundary() {
    let q = posed([40.0, 0.0, -30.0]);
    close(
        rel_angles(clamp_rotation(REST, q, MIN, MAX)),
        [40.0, 0.0, -30.0],
    );
    let fixed = clamp_rotation(REST, q, [0.0; 3], [0.0; 3]);
    close(rel_angles(fixed), [0.0; 3]);
}

#[test]
fn sign_equivalent_inputs_give_equivalent_outputs() {
    let q = posed([60.0, 0.0, 0.0]);
    let neg = q.map(|c| -c);
    let (a, b) = (
        clamp_rotation(REST, q, MIN, MAX),
        clamp_rotation(REST, neg, MIN, MAX),
    );
    for i in 0..4 {
        assert!((a[i] + b[i]).abs() < 1e-6, "hemisphere kept: {a:?} {b:?}");
    }
    close(rel_angles(b), [40.0, 0.0, 0.0]);
}

#[test]
fn non_finite_and_degenerate_input_unchanged() {
    let bad = [f32::NAN, 0.0, 0.0, 1.0];
    assert!(clamp_rotation(REST, bad, MIN, MAX)[0].is_nan());
    assert_eq!(clamp_rotation(REST, [0.0; 4], MIN, MAX), [0.0; 4]);
}

const RIG: &str = r#"{"format":"raa.rig","version":1,"name":"lim",
"bones":[{"name":"root","parent":null,"rest":{"t":[0,0,0],"r":[0,0,0,1],"s":[1,1,1]}},
{"name":"head","parent":"root","rest":{"t":[0,1,0],"r":[0,0,0.6,0.8],"s":[1,1,1]}}],
"roles":{},"ikChains":[],"limits":[{"bone":"head","min":[-40,-70,-30],"max":[40,70,30]}],"attach":{}}"#;

#[test]
fn clamp_pose_touches_only_limited_bones() {
    let rig = Rig::load(RIG).expect("rig");
    let mut local = rig.local_rest_pose().expect("rest");
    let free = from_euler_xyz_deg([90.0, 0.0, 0.0]);
    local[0].r = free;
    local[1].r = posed([0.0, 0.0, 45.0]);
    clamp_pose(&rig, &mut local);
    assert_eq!(local[0].r, free);
    close(rel_angles(local[1].r), [0.0, 0.0, 30.0]);
}
