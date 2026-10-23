//! CPU draw-list preparation: class partition, batching, the transparent sort and its tie
//! order, extras following tags, refusals and capacity reuse.

use std::collections::HashMap;

use raa_core::{MaterialId, MeshId};
use raa_lit::{
    DrawClass, DrawPrep, FLAG_FUR, FLAG_TRANSMISSIVE, FLAG_TRANSPARENT, INSTANCE_GPU_SIZE,
    InstanceExtra, LitDraw, MaterialClass, PrepInput, SurfaceMaterial,
};
use raa_math::{Quat, Vec3};
use raa_scene::{RenderList, Scene, Transform};

/// Material ids: 0 opaque, 1 transparent, 2 fur, 3 transmissive.
fn materials() -> Vec<MaterialClass> {
    [0, FLAG_TRANSPARENT, FLAG_FUR, FLAG_TRANSMISSIVE]
        .into_iter()
        .map(|flags| {
            MaterialClass::of(&SurfaceMaterial {
                flags,
                ..SurfaceMaterial::default()
            })
        })
        .collect()
}

/// One instance per spec: (mesh, material, position, tag).
fn scene_of(specs: &[(u32, u32, [f32; 3], u32)]) -> Scene {
    let mut scene = Scene::with_capacity(specs.len(), specs.len());
    for &(mesh, material, p, tag) in specs {
        let node = scene
            .add_node(
                None,
                Transform::from_trs(Vec3::new(p[0], p[1], p[2]), Quat::IDENTITY, Vec3::ONE),
            )
            .unwrap();
        let id = scene
            .add_instance(node, MeshId(mesh), MaterialId(material))
            .unwrap();
        scene.set_instance_tag(id, tag).unwrap();
    }
    scene.update_world();
    scene
}

struct Frame {
    scene: Scene,
    list: RenderList,
    tags: Vec<u32>,
    materials: Vec<MaterialClass>,
    extras: HashMap<u32, InstanceExtra>,
}

impl Frame {
    fn new(specs: &[(u32, u32, [f32; 3], u32)]) -> Frame {
        let scene = scene_of(specs);
        let mut list = RenderList::with_capacity(specs.len());
        let mut tags = Vec::with_capacity(specs.len());
        scene.build_render_list_tagged(&mut list, &mut tags);
        Frame {
            scene,
            list,
            tags,
            materials: materials(),
            extras: HashMap::new(),
        }
    }

    /// Prepares with every mesh known and every class handled.
    fn prep(&self, prep: &mut DrawPrep) -> Result<(), raa_core::RaaError> {
        self.prep_with(prep, &|_| true, &|_| true)
    }

    fn prep_with(
        &self,
        prep: &mut DrawPrep,
        known_mesh: &dyn Fn(MeshId) -> bool,
        class_handled: &dyn Fn(DrawClass) -> bool,
    ) -> Result<(), raa_core::RaaError> {
        prep.prepare(&PrepInput {
            list: &self.list,
            tags: &self.tags,
            materials: &self.materials,
            extras: &self.extras,
            view: self.scene.camera.view(),
            known_mesh,
            class_handled,
        })
    }
}

fn f32_at(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

/// World translation of instance record `i` (model column 3).
fn translation(prep: &DrawPrep, i: u32) -> [f32; 3] {
    let o = i as usize * INSTANCE_GPU_SIZE + 48;
    let b = prep.instance_bytes();
    [f32_at(b, o), f32_at(b, o + 4), f32_at(b, o + 8)]
}

fn extra_bytes(x: InstanceExtra) -> Vec<u8> {
    let mut out = Vec::new();
    x.write_gpu(&mut out);
    out
}

#[test]
fn classes_are_partitioned_in_draw_order() {
    let f = Frame::new(&[
        (0, 0, [0.0, 0.0, 0.0], 1),
        (0, 1, [1.0, 0.0, 0.0], 2),
        (1, 2, [2.0, 0.0, 0.0], 3),
        (1, 3, [3.0, 0.0, 0.0], 4),
        (2, 0, [4.0, 0.0, 0.0], 5),
    ]);
    let mut prep = DrawPrep::new(16, 8);
    f.prep(&mut prep).unwrap();
    assert_eq!(prep.instance_count(), 5);
    assert_eq!(prep.instance_bytes().len(), 5 * INSTANCE_GPU_SIZE);
    assert_eq!(prep.extra_bytes().len(), 5 * 128);
    let mats = |c: DrawClass| -> Vec<u32> { prep.draws(c).iter().map(|d| d.material.0).collect() };
    assert_eq!(mats(DrawClass::Opaque), vec![0, 0]);
    assert_eq!(mats(DrawClass::Fur), vec![2]);
    let mut transparent = mats(DrawClass::Transparent);
    transparent.sort();
    assert_eq!(transparent, vec![1, 3]);
    for class in [DrawClass::Opaque, DrawClass::Fur, DrawClass::Transparent] {
        assert!(prep.draws(class).iter().all(|d| d.class == class));
    }
    // Opaque, then fur, then transparent in the instance buffer.
    let first = |c: DrawClass| prep.draws(c)[0].first_instance;
    assert_eq!(first(DrawClass::Opaque), 0);
    assert_eq!(first(DrawClass::Fur), 2);
    assert!(
        prep.draws(DrawClass::Transparent)
            .iter()
            .all(|d| d.first_instance >= 3)
    );
    // Every instance record carries its material index at @112.
    for d in prep.draws(DrawClass::Fur) {
        let o = d.first_instance as usize * INSTANCE_GPU_SIZE + 112;
        assert_eq!(u32_at(prep.instance_bytes(), o), 2);
    }
}

#[test]
fn opaque_and_fur_draws_are_batched_by_mesh_and_material() {
    let mut specs = Vec::new();
    for i in 0..5 {
        specs.push((0, 0, [i as f32, 0.0, 0.0], i));
    }
    for i in 0..3 {
        specs.push((1, 0, [i as f32, 1.0, 0.0], 10 + i));
    }
    for i in 0..2 {
        specs.push((0, 2, [i as f32, 2.0, 0.0], 20 + i));
    }
    let f = Frame::new(&specs);
    let mut prep = DrawPrep::new(16, 8);
    f.prep(&mut prep).unwrap();
    let shape = |c: DrawClass| -> Vec<(u32, u32, u32, u32)> {
        prep.draws(c)
            .iter()
            .map(|d| (d.mesh.0, d.material.0, d.first_instance, d.instance_count))
            .collect()
    };
    assert_eq!(shape(DrawClass::Opaque), vec![(0, 0, 0, 5), (1, 0, 5, 3)]);
    assert_eq!(shape(DrawClass::Fur), vec![(0, 2, 8, 2)]);
    assert!(prep.draws(DrawClass::Transparent).is_empty());
    assert!(
        prep.draws(DrawClass::Opaque)
            .iter()
            .all(|d| d.view_depth > 0.0)
    );
}

#[test]
fn transparent_draws_sort_back_to_front_with_ties_by_tag() {
    // Default camera: eye (0, 2, 6) looking at the origin; smaller z is farther.
    let f = Frame::new(&[
        (0, 1, [0.0, 0.0, -1.0], 30),
        (0, 1, [0.0, 0.0, -5.0], 31),
        (1, 3, [0.0, 0.0, -3.0], 4),
        (1, 1, [0.0, 0.0, -3.0], 9),
        (0, 1, [0.0, 0.0, 2.0], 7),
        (0, 0, [0.0, 0.0, -9.0], 1),
    ]);
    let mut prep = DrawPrep::new(16, 8);
    f.prep(&mut prep).unwrap();
    let draws: Vec<LitDraw> = prep.draws(DrawClass::Transparent).to_vec();
    assert_eq!(draws.len(), 5);
    let z: Vec<f32> = draws
        .iter()
        .map(|d| translation(&prep, d.first_instance)[2])
        .collect();
    assert_eq!(z, vec![-5.0, -3.0, -3.0, -1.0, 2.0]);
    // The tie at z = -3: list order is material 1 (tag 9) then material 3 (tag 4); the
    // tag decides, so material 3 comes first.
    assert_eq!(draws[1].material.0, 3);
    assert_eq!(draws[2].material.0, 1);
    for (k, d) in draws.iter().enumerate() {
        assert_eq!(d.instance_count, 1);
        assert_eq!(d.first_instance, 1 + k as u32);
    }
    assert!(draws.windows(2).all(|w| w[0].view_depth >= w[1].view_depth));
}

#[test]
fn extras_follow_the_tags() {
    let mut f = Frame::new(&[
        (0, 0, [0.0, 0.0, 0.0], 100),
        (0, 0, [1.0, 0.0, 0.0], 101),
        (0, 1, [0.0, 0.0, -4.0], 102),
        (0, 1, [0.0, 0.0, -1.0], 103),
    ]);
    let extra = |v: f32| InstanceExtra {
        tint: [v, 0.5, 0.25, 1.0],
        ..InstanceExtra::default()
    };
    f.extras.insert(100, extra(0.1));
    f.extras.insert(102, extra(0.2));
    f.extras.insert(103, extra(0.3));
    let mut prep = DrawPrep::new(16, 8);
    f.prep(&mut prep).unwrap();
    let record = |i: u32| prep.extra_bytes()[i as usize * 128..(i as usize + 1) * 128].to_vec();
    for i in 0..prep.instance_count() {
        let t = translation(&prep, i);
        let tag = match (t[0], t[2]) {
            (0.0, 0.0) => 100,
            (1.0, 0.0) => 101,
            (_, -4.0) => 102,
            _ => 103,
        };
        let expected = f.extras.get(&tag).copied().unwrap_or_default();
        assert_eq!(record(i), extra_bytes(expected), "instance {i} tag {tag}");
    }
    // Back to front: tag 102 (z -4) then tag 103 (z -1).
    let t = prep.draws(DrawClass::Transparent);
    assert_eq!(record(t[0].first_instance), extra_bytes(extra(0.2)));
    assert_eq!(record(t[1].first_instance), extra_bytes(extra(0.3)));
}

fn assert_refused(r: Result<(), raa_core::RaaError>, code: &str, prep: &DrawPrep) {
    let err = r.unwrap_err();
    assert_eq!(err.code(), code, "{err}");
    assert!(!err.message().is_empty(), "{code} has an empty message");
    assert_eq!(prep.instance_count(), 0);
    assert!(prep.extra_bytes().is_empty());
    for class in [DrawClass::Opaque, DrawClass::Fur, DrawClass::Transparent] {
        assert!(prep.draws(class).is_empty());
    }
}

#[test]
fn refusals_carry_their_code_and_a_message() {
    let f = Frame::new(&[
        (0, 0, [0.0, 0.0, 0.0], 1),
        (1, 1, [1.0, 0.0, 0.0], 2),
        (2, 2, [2.0, 0.0, 0.0], 3),
    ]);
    let mut prep = DrawPrep::new(2, 8);
    assert_refused(f.prep(&mut prep), "render.too_many_instances", &prep);

    let mut prep = DrawPrep::new(16, 3);
    assert_refused(f.prep(&mut prep), "render.too_many_materials", &prep);

    let mut prep = DrawPrep::new(16, 8);
    let r = f.prep_with(&mut prep, &|m| m != MeshId(1), &|_| true);
    assert_refused(r, "render.unknown_mesh", &prep);

    let r = f.prep_with(&mut prep, &|_| true, &|c| c != DrawClass::Transparent);
    assert_refused(r, "material.blend_unavailable", &prep);
    let r = f.prep_with(&mut prep, &|_| true, &|c| c != DrawClass::Fur);
    assert_refused(r, "material.blend_unavailable", &prep);

    let mut short = Frame::new(&[(0, 0, [0.0; 3], 1)]);
    short.materials.truncate(0);
    assert_refused(short.prep(&mut prep), "material.unknown_id", &prep);
}

#[test]
fn opaque_is_always_handled() {
    let f = Frame::new(&[(0, 0, [0.0; 3], 1), (1, 0, [1.0, 0.0, 0.0], 2)]);
    let mut prep = DrawPrep::new(16, 8);
    f.prep_with(&mut prep, &|_| true, &|_| false).unwrap();
    assert_eq!(prep.draws(DrawClass::Opaque).len(), 2);
}

#[test]
fn second_identical_prep_keeps_capacity() {
    let mut specs = Vec::new();
    for i in 0..8u32 {
        specs.push((i % 2, i % 4, [i as f32, 0.0, -(i as f32)], i));
    }
    let f = Frame::new(&specs);
    let mut prep = DrawPrep::new(8, 8);
    let before = prep.capacity();
    f.prep(&mut prep).unwrap();
    let first = prep.capacity();
    let bytes = prep.instance_bytes().to_vec();
    f.prep(&mut prep).unwrap();
    assert_eq!(prep.capacity(), first);
    assert_eq!(first, before);
    assert_eq!(prep.instance_bytes(), &bytes[..]);
    prep.clear();
    assert_eq!(prep.capacity(), first);
    assert_eq!(prep.instance_count(), 0);
}
