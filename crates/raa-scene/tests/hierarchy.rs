use raa_math::{Mat4, Quat, Vec3, approx_eq};
use raa_scene::{Scene, SceneError, Transform};

fn assert_mat4_near(a: &Mat4, b: &Mat4, eps: f32) {
    let (a, b) = (a.to_cols_array(), b.to_cols_array());
    for (x, y) in a.iter().zip(b.iter()) {
        assert!(approx_eq(*x, *y, eps), "{a:?} != {b:?}");
    }
}

fn translation(x: f32, y: f32, z: f32) -> Transform {
    Transform::from_trs(Vec3::new(x, y, z), Quat::IDENTITY, Vec3::ONE)
}

fn bits(m: &Mat4) -> [u32; 16] {
    m.to_cols_array().map(f32::to_bits)
}

#[test]
fn grandchild_world_is_the_product_of_the_chain() {
    let mut scene = Scene::new();
    let root_t = translation(1.0, 2.0, 3.0);
    let child_t = Transform::from_trs(
        Vec3::new(0.0, 1.0, 0.0),
        Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
        Vec3::splat(2.0),
    );
    let grandchild_t = translation(1.0, 0.0, 0.0);
    let root = scene.add_node(None, root_t).unwrap();
    let child = scene.add_node(Some(root), child_t).unwrap();
    let grandchild = scene.add_node(Some(child), grandchild_t).unwrap();
    scene.update_world();

    let expected = root_t.to_mat4() * child_t.to_mat4() * grandchild_t.to_mat4();
    assert_mat4_near(&scene.world(grandchild).unwrap(), &expected, 1e-5);
    assert_mat4_near(
        &scene.world(child).unwrap(),
        &(root_t.to_mat4() * child_t.to_mat4()),
        1e-5,
    );
    // Rotation of +90 degrees about Y maps +X to -Z; scale 2 doubles the offset.
    let origin = scene.world(grandchild).unwrap().transform_point(Vec3::ZERO);
    let want = Vec3::new(1.0, 3.0, 1.0);
    assert!(
        approx_eq(origin.x, want.x, 1e-5)
            && approx_eq(origin.y, want.y, 1e-5)
            && approx_eq(origin.z, want.z, 1e-5),
        "{origin:?} != {want:?}"
    );
}

#[test]
fn set_local_on_one_root_leaves_another_subtree_bit_identical() {
    let mut scene = Scene::new();
    let a = scene.add_node(None, translation(1.0, 0.0, 0.0)).unwrap();
    let a_child = scene
        .add_node(
            Some(a),
            Transform::from_trs(
                Vec3::new(0.3, 0.7, -0.2),
                Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0).normalize_or_zero(), 0.7),
                Vec3::new(1.5, 0.5, 2.0),
            ),
        )
        .unwrap();
    let b = scene.add_node(None, translation(-1.0, 0.0, 0.0)).unwrap();
    let b_child = scene.add_node(Some(b), translation(0.0, 1.0, 0.0)).unwrap();
    scene.update_world();
    let before = bits(&scene.world(a_child).unwrap());
    let b_before = scene.world(b_child).unwrap();

    scene.set_local(b, translation(5.0, 5.0, 5.0)).unwrap();
    scene.update_world();

    assert_eq!(bits(&scene.world(a_child).unwrap()), before);
    let b_after = scene.world(b_child).unwrap();
    assert_ne!(b_after, b_before);
    assert_mat4_near(
        &b_after,
        &(translation(5.0, 5.0, 5.0).to_mat4() * translation(0.0, 1.0, 0.0).to_mat4()),
        1e-6,
    );
}

#[test]
fn removing_the_middle_node_makes_the_grandchild_stale() {
    let mut scene = Scene::new();
    let root = scene.add_node(None, Transform::IDENTITY).unwrap();
    let child = scene.add_node(Some(root), Transform::IDENTITY).unwrap();
    let grandchild = scene.add_node(Some(child), Transform::IDENTITY).unwrap();
    scene.remove_node(child).unwrap();

    assert_eq!(scene.node_count(), 1);
    assert_eq!(scene.world(grandchild), Err(SceneError::StaleNode));
    assert_eq!(scene.local(grandchild), Err(SceneError::StaleNode));
    assert_eq!(scene.world(child), Err(SceneError::StaleNode));
    assert_eq!(
        scene.set_local(grandchild, Transform::IDENTITY),
        Err(SceneError::StaleNode)
    );
    assert!(scene.world(root).is_ok());
    scene.update_world();
    assert_eq!(scene.world(root).unwrap(), Mat4::IDENTITY);
}

#[test]
fn removed_slot_is_reused_with_a_new_generation() {
    let mut scene = Scene::new();
    let keep = scene.add_node(None, Transform::IDENTITY).unwrap();
    let old = scene.add_node(None, translation(1.0, 0.0, 0.0)).unwrap();
    scene.remove_node(old).unwrap();

    let new = scene.add_node(None, translation(2.0, 0.0, 0.0)).unwrap();
    assert_ne!(new, old);
    assert_ne!(new, keep);
    // Same slot (low half), different generation (high half).
    assert_eq!(new.to_bits() as u32, old.to_bits() as u32);
    assert_ne!(new.to_bits() >> 32, old.to_bits() >> 32);

    assert_eq!(scene.local(old), Err(SceneError::StaleNode));
    assert_eq!(scene.remove_node(old), Err(SceneError::StaleNode));
    assert_eq!(
        scene.add_node(Some(old), Transform::IDENTITY),
        Err(SceneError::StaleNode)
    );
    assert_eq!(scene.local(new).unwrap(), translation(2.0, 0.0, 0.0));
    assert_eq!(scene.node_count(), 2);
}
