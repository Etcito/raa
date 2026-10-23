use raa_core::{MaterialId, MeshId};
use raa_math::{Mat4, Quat, Vec3};
use raa_scene::{InstanceId, RenderList, Scene, Transform};

const COUNT: usize = 24;

/// One node per instance, inserted in reverse key order: mesh (i % 3), material
/// (i % 4); the tag of instance i is 1000 + i.
fn build_scene() -> (Scene, Vec<InstanceId>) {
    let mut scene = Scene::with_capacity(COUNT, COUNT);
    let mut ids = Vec::with_capacity(COUNT);
    for i in (0..COUNT).rev() {
        let node = scene
            .add_node(
                None,
                Transform::from_trs(Vec3::new(i as f32, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
            )
            .unwrap();
        let id = scene
            .add_instance(node, MeshId((i % 3) as u32), MaterialId((i % 4) as u32))
            .unwrap();
        scene.set_instance_tag(id, 1000 + i as u32).unwrap();
        ids.push(id);
    }
    ids.reverse();
    scene.update_world();
    (scene, ids)
}

/// The instance index encoded by an item's world translation (x = i).
fn index_of(world: Mat4) -> u32 {
    world.transform_point(Vec3::ZERO).x as u32
}

#[test]
fn tags_follow_the_sorted_item_order() {
    let (scene, _) = build_scene();
    let mut plain = RenderList::with_capacity(COUNT);
    scene.build_render_list(&mut plain);
    let mut list = RenderList::with_capacity(COUNT);
    let mut tags = Vec::with_capacity(COUNT);
    scene.build_render_list_tagged(&mut list, &mut tags);

    assert_eq!(list.items, plain.items);
    assert_eq!(list.batches, plain.batches);
    assert_eq!(list.batches.len(), 3);
    assert_eq!(tags.len(), COUNT);
    assert!(
        list.items
            .windows(2)
            .all(|w| (w[0].mesh, w[0].material) <= (w[1].mesh, w[1].material))
    );
    for (item, &tag) in list.items.iter().zip(&tags) {
        assert_eq!(tag, 1000 + index_of(item.world));
    }
}

#[test]
fn set_and_get_round_trip() {
    let mut scene = Scene::new();
    let node = scene.add_node(None, Transform::IDENTITY).unwrap();
    let id = scene.add_instance(node, MeshId(0), MaterialId(0)).unwrap();
    assert_eq!(scene.instance_tag(id).unwrap(), 0);
    scene.set_instance_tag(id, 42).unwrap();
    assert_eq!(scene.instance_tag(id).unwrap(), 42);
    scene.set_instance_tag(id, u32::MAX).unwrap();
    assert_eq!(scene.instance_tag(id).unwrap(), u32::MAX);
}

#[test]
fn stale_id_after_removal_is_refused() {
    let mut scene = Scene::new();
    let node = scene.add_node(None, Transform::IDENTITY).unwrap();
    let id = scene.add_instance(node, MeshId(0), MaterialId(0)).unwrap();
    scene.set_instance_tag(id, 7).unwrap();
    scene.remove_instance(id).unwrap();

    let err = scene.set_instance_tag(id, 8).unwrap_err();
    assert_eq!(err.code(), "scene.stale_instance");
    assert_eq!(
        scene.instance_tag(id).unwrap_err().code(),
        "scene.stale_instance"
    );

    // The reused slot starts again with tag 0.
    let reused = scene.add_instance(node, MeshId(0), MaterialId(0)).unwrap();
    assert_eq!(scene.instance_tag(reused).unwrap(), 0);
    assert_eq!(
        scene.set_instance_tag(id, 9).unwrap_err().code(),
        "scene.stale_instance"
    );
    assert_eq!(scene.instance_tag(reused).unwrap(), 0);
}

#[test]
fn removed_instances_tags_disappear() {
    let (mut scene, ids) = build_scene();
    for i in (0..COUNT).step_by(5) {
        scene.remove_instance(ids[i]).unwrap();
    }
    let mut list = RenderList::with_capacity(COUNT);
    let mut tags = Vec::with_capacity(COUNT);
    scene.build_render_list_tagged(&mut list, &mut tags);

    let removed = COUNT.div_ceil(5);
    assert_eq!(list.items.len(), COUNT - removed);
    assert_eq!(tags.len(), list.items.len());
    for i in (0..COUNT).step_by(5) {
        assert!(!tags.contains(&(1000 + i as u32)));
    }
    for (item, &tag) in list.items.iter().zip(&tags) {
        assert_eq!(tag, 1000 + index_of(item.world));
    }
}

#[test]
fn capacity_is_kept_across_calls() {
    let (scene, _) = build_scene();
    let mut list = RenderList::with_capacity(COUNT);
    let mut tags = Vec::with_capacity(COUNT);
    scene.build_render_list_tagged(&mut list, &mut tags);
    let first = tags.clone();
    let tags_capacity = tags.capacity();
    let tags_ptr = tags.as_ptr();
    let items_capacity = list.items.capacity();

    scene.build_render_list_tagged(&mut list, &mut tags);
    assert_eq!(tags, first);
    assert_eq!(tags.capacity(), tags_capacity);
    assert_eq!(tags.as_ptr(), tags_ptr);
    assert_eq!(list.items.capacity(), items_capacity);
}
