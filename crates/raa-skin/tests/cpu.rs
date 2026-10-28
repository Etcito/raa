use raa_mesh::from_js_arrays;
use raa_skin::{CpuSkin, MAX_JOINTS, SkinMesh, skin_cpu};

const ID: [f32; 16] = [
    1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
];
// Child rest pivot (1,0,0): T(1,0,0) * Rz(90 degrees) * inverse_bind.
const BEND: [f32; 16] = [
    0., 1., 0., 0., -1., 0., 0., 0., 0., 0., 1., 0., 1., -1., 0., 1.,
];

fn fixture(joints: &[u16], weights: &[f32]) -> SkinMesh {
    let rest = from_js_arrays(
        &[0., 0., 0., 2., 0., 0., 1., 1., 0.],
        &[1., 0., 0.].repeat(3),
        None,
        Some(&[0., 1., 0., 1., 0., 1., 0., -1., 0., 1., 0., 1.]),
        None,
        &[0, 1, 2],
        &[],
    )
    .unwrap();
    SkinMesh::new(rest, joints, weights).unwrap()
}

fn close<const N: usize>(actual: [f32; N], expected: [f32; N]) {
    for (a, e) in actual.into_iter().zip(expected) {
        assert!(
            (a - e).abs() < 1e-6,
            "actual {actual:?}, expected {expected:?}"
        );
    }
}

fn edges(out: &CpuSkin) {
    assert_eq!(out.edge_segments.len(), 3);
    for (segment, (a, b)) in out.edge_segments.iter().zip([(0, 1), (0, 2), (1, 2)]) {
        assert_eq!(*segment, [out.positions[a], out.positions[b]]);
    }
}

#[test]
fn identity_and_missing_pose_preserve_rest() {
    let mesh = fixture(&[0; 12], &[1., 0., 0., 0.].repeat(3));
    for pose in [None, Some(ID.as_slice())] {
        let out = skin_cpu(&mesh, pose).unwrap();
        assert_eq!(out.positions, mesh.rest.base.positions);
        assert_eq!(out.normals, mesh.rest.base.normals);
        assert_eq!(out.tangents, *mesh.rest.tangents.as_ref().unwrap());
        edges(&out);
    }
}

#[test]
fn two_joint_bend_uses_mesh_space_skin_matrices() {
    let mesh = fixture(
        &[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        &[1., 0., 0., 0.].repeat(3),
    );
    let out = skin_cpu(&mesh, Some(&[ID, BEND].concat())).unwrap();
    for (actual, expected) in out
        .positions
        .iter()
        .zip([[0., 0., 0.], [1., 1., 0.], [0., 0., 0.]])
    {
        close(*actual, expected);
    }
    close(out.normals[0], [1., 0., 0.]);
    close(out.normals[1], [0., 1., 0.]);
    close(out.normals[2], [0., 1., 0.]);
    close(out.tangents[1], [-1., 0., 0., -1.]);
    close(out.tangents[2], [-1., 0., 0., 1.]);
    edges(&out);
}

#[test]
fn mixed_joint_bend_normalizes_after_matrix_blending() {
    let mesh = fixture(&[0, 1, 0, 0].repeat(3), &[0.25, 0.75, 0., 0.].repeat(3));
    let out = skin_cpu(&mesh, Some(&[ID, BEND].concat())).unwrap();
    close(out.positions[0], [0.75, -0.75, 0.]);
    close(out.positions[1], [1.25, 0.75, 0.]);
    close(out.positions[2], [0.25, 0.25, 0.]);
    let d = 10.0_f32.sqrt();
    close(out.normals[2], [1. / d, 3. / d, 0.]);
    close(out.tangents[1], [-3. / d, 1. / d, 0., -1.]);
    edges(&out);
}

#[test]
fn four_influences_use_upper_3x3_and_keep_handedness() {
    let mesh = fixture(&[0, 1, 2, 3].repeat(3), &[0.1, 0.2, 0.3, 0.4].repeat(3));
    let mut scale_x = ID;
    scale_x[0] = 3.;
    scale_x[12] = 4.;
    let mut scale_y = ID;
    scale_y[5] = 3.;
    scale_y[13] = -2.;
    let mut rotation = BEND;
    rotation[12] = 0.;
    rotation[13] = 0.;
    let out = skin_cpu(&mesh, Some(&[ID, scale_x, rotation, scale_y].concat())).unwrap();
    close(out.positions[0], [0.8, -0.8, 0.]);
    close(out.positions[1], [3., -0.2, 0.]);
    close(out.positions[2], [1.6, 1., 0.]);
    for i in 0..3 {
        let d = 130.0_f32.sqrt();
        close(out.normals[i], [11. / d, 3. / d, 0.]);
        let d = 234.0_f32.sqrt();
        let w = if i == 1 { -1. } else { 1. };
        close(out.tangents[i], [-3. / d, 15. / d, 0., w]);
    }
    edges(&out);
}

#[test]
fn instance_poses_are_independent_and_do_not_mutate_rest() {
    let mesh = fixture(&[0; 12], &[1., 0., 0., 0.].repeat(3));
    let first = skin_cpu(&mesh, Some(&BEND)).unwrap();
    let mut translated = ID;
    translated[12..15].copy_from_slice(&[5., -2., 3.]);
    let second = skin_cpu(&mesh, Some(&translated)).unwrap();
    close(first.positions[1], [1., 1., 0.]);
    close(second.positions[1], [7., -2., 3.]);
    close(second.normals[1], [1., 0., 0.]);
    close(second.tangents[1], [0., 1., 0., -1.]);
    assert_eq!(first, skin_cpu(&mesh, Some(&BEND)).unwrap());
    close(mesh.rest.base.positions[1], [2., 0., 0.]);
    assert_eq!(
        skin_cpu(&mesh, None).unwrap().positions,
        mesh.rest.base.positions
    );
    edges(&first);
    edges(&second);
}

#[test]
fn default_tangents_and_shared_edges_follow_the_pose() {
    let mut mesh = fixture(&[0; 12], &[1., 0., 0., 0.].repeat(3));
    mesh.rest.tangents = None;
    mesh.rest.base.indices.extend_from_slice(&[2, 1, 0]);
    let mut tilted = ID;
    tilted[5] = 0.;
    tilted[6] = 1.;
    tilted[9] = -1.;
    tilted[10] = 0.;
    let out = skin_cpu(&mesh, Some(&tilted)).unwrap();
    close(out.positions[2], [1., 0., 1.]);
    close(out.normals[2], [1., 0., 0.]);
    close(out.tangents[2], [0., 1., 0., 1.]);
    assert!(mesh.rest.tangents.is_none());
    edges(&out);
}

#[test]
fn pose_lengths_ranges_and_finiteness_are_checked() {
    let mesh = fixture(&[0; 12], &[1., 0., 0., 0.].repeat(3));
    let mut nonfinite = ID;
    nonfinite[0] = f32::NAN;
    for pose in [&[][..], &ID[..15], &nonfinite] {
        assert_eq!(
            skin_cpu(&mesh, Some(pose)).unwrap_err().code(),
            "skin.pose_length"
        );
    }
    assert_eq!(
        skin_cpu(&mesh, Some(&ID.repeat(MAX_JOINTS + 1)))
            .unwrap_err()
            .code(),
        "skin.cap_joints"
    );
    let mesh = fixture(&[1; 12], &[1., 0., 0., 0.].repeat(3));
    assert_eq!(
        skin_cpu(&mesh, Some(&ID)).unwrap_err().code(),
        "skin.joint_range"
    );
}
