//! The scene keeps its instance index in step with transforms, bounds and visibility.

use raa_core::{MaterialId, MeshId};
use raa_math::{Aabb, Quat, Vec3};
use raa_scene::{Camera, RenderList, Scene, Transform, frustum_planes};

fn at(x: f32, y: f32, z: f32) -> Transform {
    Transform::from_trs(Vec3::new(x, y, z), Quat::IDENTITY, Vec3::ONE)
}

fn cube(h: f32) -> Aabb {
    Aabb {
        min: Vec3::splat(-h),
        max: Vec3::splat(h),
    }
}

fn camera() -> Camera {
    Camera {
        eye: Vec3::new(0.0, 0.0, 7.0),
        target: Vec3::ZERO,
        ..Camera::default()
    }
}

fn slot_of(id: raa_scene::InstanceId) -> u32 {
    id.to_bits() as u32
}

fn fully_outside(b: &Aabb, planes: &[[f32; 4]]) -> bool {
    planes.iter().any(|p| {
        let pick = |v: f32, lo: f32, hi: f32| v * if v >= 0.0 { hi } else { lo };
        let d = pick(p[0], b.min.x, b.max.x)
            + pick(p[1], b.min.y, b.max.y)
            + pick(p[2], b.min.z, b.max.z)
            + p[3];
        d.is_finite() && d < 0.0
    })
}

fn frustum_slots(scene: &Scene) -> Vec<u32> {
    let mut out = Vec::new();
    let planes = frustum_planes(camera().view_proj());
    scene.instance_index().query_frustum(&planes, &mut out);
    out
}

#[test]
fn frustum_query_matches_brute_force_over_a_grid() {
    let local = cube(0.4);
    let mesh = MeshId(7);
    let mut scene = Scene::new();
    scene.set_mesh_bounds(mesh, local);
    let mut entries = Vec::new();
    for iz in 0..3 {
        for iy in 0..10 {
            for ix in 0..10 {
                let node = scene
                    .add_node(None, at(ix as f32 - 4.5, iy as f32 - 4.5, iz as f32 - 1.0))
                    .unwrap();
                let id = scene.add_instance(node, mesh, MaterialId(0)).unwrap();
                entries.push((id, node));
            }
        }
    }
    assert_eq!(entries.len(), 300);
    scene.update_world();

    let planes = frustum_planes(camera().view_proj());
    let mut got = Vec::new();
    scene.instance_index().query_frustum(&planes, &mut got);
    let mut want: Vec<u32> = entries
        .iter()
        .filter(|(_, node)| {
            let wb = local.transform(&scene.world(*node).unwrap());
            !fully_outside(&wb, &planes)
        })
        .map(|(id, _)| slot_of(*id))
        .collect();
    want.sort_unstable();
    assert_eq!(got, want);
    assert!(!got.is_empty() && got.len() < entries.len());
}

#[test]
fn moving_a_parent_moves_its_children_entries() {
    let local = cube(0.5);
    let mesh = MeshId(2);
    let mut scene = Scene::new();
    scene.set_mesh_bounds(mesh, local);
    let parent = scene.add_node(None, at(0.0, 0.0, 0.0)).unwrap();
    let child = scene.add_node(Some(parent), at(2.0, 0.0, 0.0)).unwrap();
    let id = scene.add_instance(child, mesh, MaterialId(0)).unwrap();
    scene.update_world();
    let old = local.transform(&scene.world(child).unwrap());
    assert_eq!(scene.instance_index().bounds(slot_of(id)), Some(Some(old)));

    scene.set_local(parent, at(0.0, 7.0, 0.0)).unwrap();
    scene.update_world();
    let moved = local.transform(&scene.world(child).unwrap());
    assert_ne!(moved, old);
    assert_eq!(
        scene.instance_index().bounds(slot_of(id)),
        Some(Some(moved))
    );
}

#[test]
fn hiding_an_instance_removes_it_and_showing_it_adds_it_back() {
    let local = cube(1.0);
    let mesh = MeshId(1);
    let mut scene = Scene::new();
    scene.set_mesh_bounds(mesh, local);
    let node = scene.add_node(None, Transform::IDENTITY).unwrap();
    let id = scene.add_instance(node, mesh, MaterialId(0)).unwrap();
    scene.update_world();
    assert!(frustum_slots(&scene).contains(&slot_of(id)));

    scene.set_instance_visible(id, false).unwrap();
    assert_eq!(scene.instance_index().bounds(slot_of(id)), None);
    assert!(!frustum_slots(&scene).contains(&slot_of(id)));

    scene.set_instance_visible(id, true).unwrap();
    assert!(scene.instance_index().bounds(slot_of(id)).is_some());
    assert!(frustum_slots(&scene).contains(&slot_of(id)));
}

#[test]
fn removing_a_node_drops_its_instances_from_the_index() {
    let mesh = MeshId(3);
    let mut scene = Scene::new();
    scene.set_mesh_bounds(mesh, cube(1.0));
    let root = scene.add_node(None, Transform::IDENTITY).unwrap();
    let child = scene.add_node(Some(root), at(0.0, 1.0, 0.0)).unwrap();
    let on_root = scene.add_instance(root, mesh, MaterialId(0)).unwrap();
    let on_child = scene.add_instance(child, mesh, MaterialId(0)).unwrap();
    scene.update_world();
    assert_eq!(scene.instance_index().len(), 2);

    scene.remove_node(root).unwrap();
    assert_eq!(scene.instance_index().len(), 0);
    assert_eq!(scene.instance_index().bounds(slot_of(on_root)), None);
    assert_eq!(scene.instance_index().bounds(slot_of(on_child)), None);
    assert!(frustum_slots(&scene).is_empty());
}

#[test]
fn ray_instances_are_nearest_first_with_their_handles_and_max_truncates() {
    let mesh = MeshId(4);
    let mut scene = Scene::new();
    scene.set_mesh_bounds(mesh, cube(0.2));
    let node = scene.add_node(None, Transform::IDENTITY).unwrap();
    let ids: Vec<_> = [1.0, 2.0, 3.0]
        .iter()
        .map(|&x| {
            let n = scene.add_node(Some(node), at(x, 0.0, 0.0)).unwrap();
            scene.add_instance(n, mesh, MaterialId(0)).unwrap()
        })
        .collect();
    scene.update_world();

    let mut hits = Vec::new();
    scene.ray_instances(Vec3::ZERO, Vec3::X, 10.0, &mut hits);
    assert_eq!(hits.iter().map(|(id, _)| *id).collect::<Vec<_>>(), ids);
    let ts: Vec<f32> = hits.iter().map(|(_, t)| *t).collect();
    assert!(ts[0] < ts[1] && ts[1] < ts[2]);

    scene.ray_instances(Vec3::ZERO, Vec3::X, 2.5, &mut hits);
    assert_eq!(hits.iter().map(|(id, _)| *id).collect::<Vec<_>>(), ids[..2]);
}

#[test]
fn an_instance_of_a_mesh_without_bounds_is_always_selected_and_never_ray_hit() {
    let mesh = MeshId(9);
    let mut scene = Scene::new();
    let node = scene.add_node(None, at(0.0, 0.0, -500.0)).unwrap();
    let id = scene.add_instance(node, mesh, MaterialId(0)).unwrap();
    scene.update_world();
    assert_eq!(scene.instance_index().bounds(slot_of(id)), Some(None));
    assert!(frustum_slots(&scene).contains(&slot_of(id)));

    let mut hits = Vec::new();
    scene.ray_instances(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), 1000.0, &mut hits);
    assert!(hits.is_empty());

    // Giving the mesh bounds on a later update moves the same slot into the BVH.
    scene.set_mesh_bounds(mesh, cube(1.0));
    scene.update_world();
    assert!(
        scene
            .instance_index()
            .bounds(slot_of(id))
            .flatten()
            .is_some()
    );
}

#[test]
fn slots_and_tags_match_the_tagged_list() {
    let mesh = MeshId(5);
    let mut scene = Scene::new();
    scene.set_mesh_bounds(mesh, cube(0.5));
    let node = scene.add_node(None, Transform::IDENTITY).unwrap();
    let mut ids = Vec::new();
    for (i, x) in [0.0, 2.0, 4.0, 6.0].iter().enumerate() {
        let n = scene.add_node(Some(node), at(*x, 0.0, 0.0)).unwrap();
        let id = scene.add_instance(n, mesh, MaterialId(0)).unwrap();
        scene.set_instance_tag(id, 10 + i as u32).unwrap();
        ids.push(id);
    }
    scene.set_instance_visible(ids[1], false).unwrap();
    scene.update_world();

    let mut list = RenderList::default();
    let mut tags = Vec::new();
    let mut slots = Vec::new();
    scene.build_render_list_slots(&mut list, &mut tags, &mut slots);
    let mut tagged = RenderList::default();
    let mut tagged_tags = Vec::new();
    scene.build_render_list_tagged(&mut tagged, &mut tagged_tags);
    assert_eq!(list, tagged);
    assert_eq!(tags, tagged_tags);
    assert_eq!(slots.len(), list.items.len());
    for (i, &slot) in slots.iter().enumerate() {
        let id = scene.instance_at_slot(slot).unwrap();
        assert_eq!(slot_of(id), slot);
        assert_eq!(scene.instance_tag(id).unwrap(), tags[i]);
    }
    assert_eq!(slots.len(), 3);
}
