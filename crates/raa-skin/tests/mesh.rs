use raa_mesh::{AttributedMesh, from_js_arrays};
use raa_skin::{MAX_INFLUENCES, MAX_JOINTS, SkinError, SkinMesh, WEIGHT_SUM_EPS};

fn quad() -> AttributedMesh {
    let positions = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let normals = [0.0, 0.0, 1.0].repeat(4);
    let indices = [0, 1, 2, 0, 2, 3];
    from_js_arrays(&positions, &normals, None, None, None, &indices, &[]).expect("quad")
}

fn skin(joints: &[u16], weights: &[f32]) -> Result<SkinMesh, SkinError> {
    SkinMesh::new(quad(), joints, weights)
}

fn valid() -> (Vec<u16>, Vec<f32>) {
    ([0, 1, 2, 3].repeat(4), [1.0, 0.0, 0.0, 0.0].repeat(4))
}

fn refused(result: Result<SkinMesh, SkinError>, code: &str, mentions: &str) {
    let err = result.expect_err("skin must be refused");
    assert_eq!(err.code(), code, "unexpected refusal: {err}");
    assert!(
        err.message.contains(mentions),
        "{code} message {:?} does not mention {mentions:?}",
        err.message
    );
}

#[test]
fn joints_and_weights_come_both_or_neither() {
    let (j, w) = valid();
    refused(skin(&[], &w), "skin.length_mismatch", "both or neither");
    refused(skin(&j, &[]), "skin.length_mismatch", "both or neither");
    refused(skin(&[], &[]), "skin.length_mismatch", "joints");
}

#[test]
fn lengths_are_vertex_aligned_four_per_vertex() {
    let (j, w) = valid();
    refused(
        skin(&j[..j.len() - 1], &w),
        "skin.length_mismatch",
        "joints",
    );
    refused(
        skin(&j, &w[..w.len() - 1]),
        "skin.length_mismatch",
        "weights",
    );
}

#[test]
fn weights_are_finite_non_negative_and_sum_to_one() {
    let (j, mut w) = valid();
    w[0] = f32::NAN;
    refused(skin(&j, &w), "skin.weights_sum", "weights[0][0]");
    let (j, mut w) = valid();
    w[0] = -0.5;
    w[1] = 1.5;
    refused(skin(&j, &w), "skin.weights_sum", "weights[0][0]");
    let (j, mut w) = valid();
    w[0] = 0.5;
    refused(skin(&j, &w), "skin.weights_sum", "weights[0] sum");
    let (j, mut w) = valid();
    w[0] = 1.0 + WEIGHT_SUM_EPS / 2.0;
    assert!(
        skin(&j, &w).is_ok(),
        "sum inside WEIGHT_SUM_EPS is accepted"
    );
    w[0] = 1.0 + WEIGHT_SUM_EPS * 2.0;
    refused(skin(&j, &w), "skin.weights_sum", "weights[0]");
}

#[test]
fn joint_range_and_cap_are_enforced() {
    let (mut j, w) = valid();
    j[3] = MAX_JOINTS as u16;
    refused(skin(&j, &w), "skin.joint_range", "joints[0][3]");
    let (j, w) = valid();
    let mesh = skin(&j, &w).expect("valid skin");
    mesh.validate_joint_count(MAX_INFLUENCES)
        .expect("four joints");
    let err = mesh
        .validate_joint_count(3)
        .expect_err("joint 3 is missing");
    assert_eq!(err.code(), "skin.joint_range");
    assert!(err.message.contains("rig's 3 joints"), "{err}");
    let err = mesh
        .validate_joint_count(MAX_JOINTS + 1)
        .expect_err("over cap");
    assert_eq!(err.code(), "skin.cap_joints");
}

#[test]
fn rest_geometry_is_preserved() {
    let rest = quad();
    let (j, w) = valid();
    let mesh = SkinMesh::new(rest.clone(), &j, &w).expect("valid skin");
    assert_eq!(mesh.rest.base.positions, rest.base.positions);
    assert_eq!(mesh.rest.base.normals, rest.base.normals);
}

#[test]
fn upload_packing_is_little_endian_u16x4_and_f32x4() {
    let joints = [1u16, 2, 3, 4].repeat(4);
    let weights = [0.25f32, 0.5, 0.25, 0.0].repeat(4);
    let mesh = skin(&joints, &weights).expect("valid skin");
    let mut jb = Vec::new();
    mesh.write_joint_bytes(&mut jb);
    assert_eq!(jb.len(), 4 * 8);
    let want: Vec<u8> = (0..4)
        .flat_map(|_| {
            let (lo, hi) = (0x0002_0001u32, 0x0004_0003u32);
            lo.to_le_bytes().into_iter().chain(hi.to_le_bytes())
        })
        .collect();
    assert_eq!(jb, want);
    assert_eq!(&jb[..8], &[1, 0, 2, 0, 3, 0, 4, 0]);
    let mut wb = Vec::new();
    mesh.write_weight_bytes(&mut wb);
    assert_eq!(wb.len(), 4 * 16);
    let want: Vec<u8> = [0.25f32, 0.5, 0.25, 0.0]
        .repeat(4)
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    assert_eq!(wb, want);
}
