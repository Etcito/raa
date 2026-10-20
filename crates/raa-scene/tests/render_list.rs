use raa_core::{MaterialId, MeshId};
use raa_math::{Quat, Vec3};
use raa_scene::{InstanceId, RenderList, Scene, Transform};

const PARENTS: usize = 100;
const CHILDREN: usize = 10;
const INSTANCES: usize = PARENTS * CHILDREN;

/// 100 parents x 10 children, one instance per child: mesh i % 4, material i % 8.
fn build_scene() -> (Scene, Vec<InstanceId>) {
    let mut scene = Scene::with_capacity(PARENTS + INSTANCES, INSTANCES);
    let mut ids = Vec::with_capacity(INSTANCES);
    for p in 0..PARENTS {
        let parent = scene
            .add_node(
                None,
                Transform::from_trs(
                    Vec3::new(p as f32, 0.0, 0.0),
                    Quat::from_rotation_y(p as f32 * 0.1),
                    Vec3::ONE,
                ),
            )
            .unwrap();
        for c in 0..CHILDREN {
            let i = p * CHILDREN + c;
            let child = scene
                .add_node(
                    Some(parent),
                    Transform::from_trs(Vec3::new(0.0, c as f32, 0.0), Quat::IDENTITY, Vec3::ONE),
                )
                .unwrap();
            let id = scene
                .add_instance(child, MeshId((i % 4) as u32), MaterialId((i % 8) as u32))
                .unwrap();
            ids.push(id);
        }
    }
    (scene, ids)
}

#[test]
fn thousand_instances_give_four_sorted_batches() {
    let (mut scene, _) = build_scene();
    assert_eq!(scene.node_count(), PARENTS + INSTANCES);
    assert_eq!(scene.instance_count(), INSTANCES);
    scene.update_world();
    let mut list = RenderList::with_capacity(INSTANCES);
    scene.build_render_list(&mut list);

    assert_eq!(list.items.len(), INSTANCES);
    assert_eq!(list.batches.len(), 4);
    let total: u32 = list.batches.iter().map(|b| b.count).sum();
    assert_eq!(total as usize, INSTANCES);
    assert!(
        list.items
            .windows(2)
            .all(|w| (w[0].mesh, w[0].material) <= (w[1].mesh, w[1].material))
    );
    let mut next = 0;
    for (k, batch) in list.batches.iter().enumerate() {
        assert_eq!(batch.mesh, MeshId(k as u32));
        assert_eq!(batch.first, next);
        assert_eq!(batch.count as usize, INSTANCES / 4);
        let run = &list.items[batch.first as usize..(batch.first + batch.count) as usize];
        assert!(run.iter().all(|item| item.mesh == batch.mesh));
        next += batch.count;
    }
}

#[test]
fn second_build_reuses_capacity_and_is_equal() {
    let (mut scene, _) = build_scene();
    scene.update_world();
    let mut list = RenderList::with_capacity(INSTANCES);
    scene.build_render_list(&mut list);
    let first = list.items.clone();
    let capacity = list.items.capacity();
    let batches = list.batches.clone();

    scene.update_world();
    scene.build_render_list(&mut list);
    assert_eq!(list.items.capacity(), capacity);
    assert_eq!(list.items, first);
    assert_eq!(list.batches, batches);
}

#[test]
fn hiding_ten_instances_leaves_990_items() {
    let (mut scene, ids) = build_scene();
    for id in ids.iter().step_by(100) {
        scene.set_instance_visible(*id, false).unwrap();
    }
    scene.update_world();
    let mut list = RenderList::with_capacity(INSTANCES);
    scene.build_render_list(&mut list);

    assert_eq!(list.items.len(), INSTANCES - 10);
    assert_eq!(scene.instance_count(), INSTANCES);
    let total: u32 = list.batches.iter().map(|b| b.count).sum();
    assert_eq!(total as usize, INSTANCES - 10);
}
