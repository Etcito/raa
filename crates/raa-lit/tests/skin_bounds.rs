use raa_core::{MaterialId, MeshId};
use raa_lit::{DrawClass, DrawPrep, MaterialClass, PrepInput, SurfaceMaterial};
use raa_lit::skin_bounds::{posed_bounds, visible};
use raa_lit::skin_draw::instance_bounds;
use raa_math::{Aabb, Mat4, Vec3};
use raa_scene::{RenderList, Scene, Transform};
use std::collections::HashMap;

fn rest() -> Aabb {
    Aabb::from_points(&[[-0.1, -0.1, 0.2], [0.1, 0.1, 0.4]])
}
fn pose(x: f32) -> Vec<f32> {
    Mat4::from_translation(Vec3::new(x, 0., 0.)).cols.into_iter().flatten().collect()
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
    assert!(visible(posed_bounds(rest(), &[f32::NAN; 16]), world, Mat4::IDENTITY));
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
        scene.build_render_list_tagged(&mut list, &mut tags);
        for item in &mut list.items {
            item.world = Mat4::from_translation(Vec3::new(4., 0., 0.));
        }
        let materials = [MaterialClass { class, ..MaterialClass::of(&SurfaceMaterial::default()) }];
        let extras = HashMap::new();
        let input = PrepInput {
            list: &list, tags: &tags, materials: &materials, extras: &extras,
            view: Mat4::IDENTITY, known_mesh: &|_| true, class_handled: &|_| true,
        };
        // Only the two posed instances split their geometry bindings.
        let mut prep = DrawPrep::new(3, 1);
        let input = PrepInput { known_mesh: &|m| m.0 == 0, ..input };
        for enabled in [true, false] {
            prep.prepare_skinned(&input, &|tag| enabled && tag < 2).unwrap();
            prep.cull_camera(&|_, tag, world| {
                let bounds = if enabled && tag < 2 {
                    posed_bounds(rest(), &pose(if tag == 0 { -4. } else { 4. }))
                } else { Some(rest()) };
                visible(bounds, world, Mat4::IDENTITY)
            });
            assert_eq!(prep.camera_draws(class).len(), usize::from(enabled));
            assert_eq!(prep.draws(class).iter().map(|d| d.instance_count).sum::<u32>(), 3,
                "off-camera casters remain in the unfiltered path");
            if enabled {
                let draw = prep.camera_draws(class)[0];
                assert_eq!(prep.mesh_binding(draw.mesh), (MeshId(0), Some(0)));
            }
        }
        // Replacing a pose can move it out; removing it restores rest bounds.
        assert!(!visible(posed_bounds(rest(), &pose(4.)), list.items[0].world, Mat4::IDENTITY));
        assert!(!visible(Some(rest()), list.items[0].world, Mat4::IDENTITY));
    }
}
