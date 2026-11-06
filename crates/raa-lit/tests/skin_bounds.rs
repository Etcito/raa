use raa_core::{MaterialId, MeshId};
use raa_lit::skin_bounds::{posed_bounds, visible};
use raa_lit::skin_draw::instance_bounds;
use raa_lit::{DrawClass, IdPrep, MaterialClass, SelectInput, SkinDraw, SurfaceMaterial};
use raa_math::{Aabb, Mat4, Vec3};
use raa_scene::{RenderList, Scene, Transform};

fn rest() -> Aabb {
    Aabb::from_points(&[[-0.1, -0.1, 0.2], [0.1, 0.1, 0.4]])
}
fn pose(x: f32) -> Vec<f32> {
    Mat4::from_translation(Vec3::new(x, 0., 0.))
        .cols
        .into_iter()
        .flatten()
        .collect()
}

#[test]
fn union_then_instance_transform_and_unknown_fallback() {
    let joints = [pose(-4.), pose(2.)].concat();
    let b = posed_bounds(rest(), &joints).unwrap();
    assert_eq!(b.min.x, -4.1);
    assert_eq!(b.max.x, 2.1);
    assert_eq!(instance_bounds(rest(), false, Some(&joints)), Some(rest()));
    assert_eq!(instance_bounds(rest(), false, None), Some(rest()));
    assert_eq!(instance_bounds(rest(), true, None), None);
    let world = Mat4::from_translation(Vec3::new(4., 0., 0.));
    assert!(!visible(Some(rest()), world, Mat4::IDENTITY));
    assert!(visible(Some(b), world, Mat4::IDENTITY));
    assert!(visible(posed_bounds(rest(), &[]), world, Mat4::IDENTITY));
    assert!(visible(
        posed_bounds(rest(), &[f32::NAN; 16]),
        world,
        Mat4::IDENTITY
    ));
}

#[test]
fn independent_poses_camera_passes_rest_fallback_and_off_camera_casters() {
    for class in [DrawClass::Opaque, DrawClass::Fur, DrawClass::Transparent] {
        let mut scene = Scene::with_capacity(3, 3);
        for tag in 0..3 {
            let node = scene.add_node(None, Transform::IDENTITY).unwrap();
            let id = scene.add_instance(node, MeshId(0), MaterialId(0)).unwrap();
            scene.set_instance_tag(id, tag).unwrap();
        }
        scene.update_world();
        let mut list = RenderList::with_capacity(3);
        let mut tags = Vec::new();
        let mut slots = Vec::new();
        scene.build_render_list_slots(&mut list, &mut tags, &mut slots);
        for item in &mut list.items {
            item.world = Mat4::from_translation(Vec3::new(4., 0., 0.));
        }
        let materials = [MaterialClass {
            class,
            ..MaterialClass::of(&SurfaceMaterial::default())
        }];
        // Only the two posed instances split their geometry bindings.
        let known_mesh = |m: MeshId| m.0 == 0;
        let level = |_: u32, mesh: MeshId| mesh;
        let handled = |_: DrawClass| true;
        let mut prep = IdPrep::new(3, 1);
        for enabled in [true, false] {
            let skinned = |tag: u32| {
                (enabled && tag < 2).then_some(SkinDraw {
                    base: tag * 4,
                    skin: 1,
                })
            };
            let selected = |i: usize| {
                let tag = tags[i];
                let bounds = if enabled && tag < 2 {
                    posed_bounds(rest(), &pose(if tag == 0 { -4. } else { 4. }))
                } else {
                    Some(rest())
                };
                visible(bounds, list.items[i].world, Mat4::IDENTITY)
            };
            let input = SelectInput {
                list: &list,
                tags: &tags,
                slots: &slots,
                selected: &selected,
                materials: &materials,
                view: Mat4::IDENTITY,
                known_mesh: &known_mesh,
                class_handled: &handled,
                level_mesh: &level,
            };
            prep.prepare_camera(&input, &skinned).unwrap();
            assert_eq!(prep.draws(class).len(), usize::from(enabled));
            if class != DrawClass::Transparent {
                // Off-camera instances remain casters in the unclipped cascade section.
                let all = |_| true;
                let casters = SelectInput {
                    selected: &all,
                    ..input
                };
                prep.prepare_cascade(0, &casters, &skinned).unwrap();
                assert_eq!(
                    prep.cascade_draws(0, class)
                        .iter()
                        .map(|d| d.instance_count)
                        .sum::<u32>(),
                    3,
                    "off-camera casters remain in the cascade section"
                );
            }
            if enabled {
                let draw = prep.draws(class)[0];
                assert_eq!(prep.mesh_binding(draw.mesh), (MeshId(0), Some(0)));
            }
        }
        // Replacing a pose can move it out; removing it restores rest bounds.
        assert!(!visible(
            posed_bounds(rest(), &pose(4.)),
            list.items[0].world,
            Mat4::IDENTITY
        ));
        assert!(!visible(Some(rest()), list.items[0].world, Mat4::IDENTITY));
    }
}
