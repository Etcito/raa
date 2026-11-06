//! Compacted id-list preparation: draw order, batch splitting by level mesh, the
//! transparent sort, cascade sections and their frame reset, refusals, posed grouping and
//! capacity reuse.

use std::cell::RefCell;

use raa_core::{MaterialId, MeshId, RaaError};
use raa_lit::{
    DrawClass, FLAG_FUR, FLAG_TRANSMISSIVE, FLAG_TRANSPARENT, IdPrep, LitDraw, MaterialClass,
    SelectInput, SkinDraw, SurfaceMaterial,
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
    slots: Vec<u32>,
    materials: Vec<MaterialClass>,
}

impl Frame {
    fn new(specs: &[(u32, u32, [f32; 3], u32)]) -> Frame {
        let scene = scene_of(specs);
        let mut list = RenderList::with_capacity(specs.len());
        let mut tags = Vec::with_capacity(specs.len());
        scene.build_render_list_tagged(&mut list, &mut tags);
        // Tags are unique in these tests, so they double as stable slots for the id list.
        let slots = tags.clone();
        Frame {
            scene,
            list,
            tags,
            slots,
            materials: materials(),
        }
    }
}

fn input<'a>(
    f: &'a Frame,
    selected: &'a dyn Fn(usize) -> bool,
    level_mesh: &'a dyn Fn(u32, MeshId) -> MeshId,
    known_mesh: &'a dyn Fn(MeshId) -> bool,
    class_handled: &'a dyn Fn(DrawClass) -> bool,
) -> SelectInput<'a> {
    SelectInput {
        list: &f.list,
        tags: &f.tags,
        slots: &f.slots,
        selected,
        materials: &f.materials,
        view: f.scene.camera.view(),
        known_mesh,
        class_handled,
        level_mesh,
    }
}

fn ids(prep: &IdPrep) -> Vec<u32> {
    prep.id_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn draw_ids(prep: &IdPrep, draw: &LitDraw) -> Vec<u32> {
    let all = ids(prep);
    all[draw.first_instance as usize..(draw.first_instance + draw.instance_count) as usize].to_vec()
}

fn shape(draw: &LitDraw) -> (u32, u32, u32, u32) {
    (
        draw.mesh.0,
        draw.material.0,
        draw.first_instance,
        draw.instance_count,
    )
}

fn shapes(prep: &IdPrep, class: DrawClass) -> Vec<(u32, u32, u32, u32)> {
    prep.draws(class).iter().map(shape).collect()
}

#[test]
fn opaque_and_transparent_ids_follow_the_list_with_identity_levels() {
    let f = Frame::new(&[
        (0, 0, [0.0, 0.0, 0.0], 1),
        (0, 0, [1.0, 0.0, 0.0], 2),
        (1, 0, [2.0, 0.0, 0.0], 3),
        (1, 0, [3.0, 0.0, 0.0], 4),
        (0, 2, [4.0, 0.0, 0.0], 5),
        (0, 2, [5.0, 0.0, 0.0], 6),
        (1, 0, [6.0, 0.0, 0.0], 7),
        (0, 1, [0.0, 0.0, -3.0], 8),
        (0, 1, [0.0, 0.0, 3.0], 9),
    ]);
    let mut ip = IdPrep::new(16, 8);
    let selected = |_| true;
    let level = |_: u32, mesh: MeshId| mesh;
    let known = |_: MeshId| true;
    let handled = |_: DrawClass| true;
    ip.prepare_camera(&input(&f, &selected, &level, &known, &handled), &|_| None)
        .unwrap();

    assert_eq!(
        shapes(&ip, DrawClass::Opaque),
        vec![(0, 0, 0, 2), (1, 0, 2, 3)]
    );
    assert_eq!(shapes(&ip, DrawClass::Fur), vec![(0, 2, 5, 2)]);
    assert_eq!(
        shapes(&ip, DrawClass::Transparent),
        vec![(0, 1, 7, 1), (0, 1, 8, 1)]
    );
    // Each layer appends after the previous one: opaque, fur, then back-to-front
    // transparent, every id the stable slot of its item.
    assert_eq!(ids(&ip), vec![1, 2, 3, 4, 7, 5, 6, 8, 9]);
    assert_eq!(ip.camera_count(), 9);
    assert!(
        ip.draws(DrawClass::Opaque)
            .iter()
            .all(|d| d.view_depth > 0.0)
    );
}

#[test]
fn selection_keeps_one_draw_per_run() {
    let f = Frame::new(&[
        (0, 0, [0.0, 0.0, 0.0], 10),
        (0, 0, [1.0, 0.0, 0.0], 11),
        (0, 0, [2.0, 0.0, 0.0], 12),
        (0, 0, [3.0, 0.0, 0.0], 13),
        (0, 0, [4.0, 0.0, 0.0], 14),
    ]);
    // max_instances bounds the selected items, not the length of the full list.
    let mut ip = IdPrep::new(3, 8);
    let selected = |i| matches!(i, 0 | 2 | 4);
    let level = |_: u32, mesh: MeshId| mesh;
    let known = |_: MeshId| true;
    let handled = |_: DrawClass| true;
    ip.prepare_camera(&input(&f, &selected, &level, &known, &handled), &|_| None)
        .unwrap();
    let draws = ip.draws(DrawClass::Opaque);
    assert_eq!(draws.len(), 1);
    assert_eq!(draws[0].instance_count, 3);
    assert_eq!(
        draw_ids(&ip, &draws[0]),
        vec![f.slots[0], f.slots[2], f.slots[4]]
    );
}

#[test]
fn level_mesh_splits_a_run_ascending() {
    let f = Frame::new(&[
        (0, 0, [0.0, 0.0, 0.0], 0),
        (0, 0, [1.0, 0.0, 0.0], 1),
        (0, 0, [2.0, 0.0, 0.0], 2),
        (0, 0, [3.0, 0.0, 0.0], 3),
        (0, 0, [4.0, 0.0, 0.0], 4),
    ]);
    let mut ip = IdPrep::new(16, 8);
    let selected = |_| true;
    let level = |slot: u32, mesh: MeshId| {
        if slot % 2 == 1 { MeshId(90) } else { mesh }
    };
    let known = |_: MeshId| true;
    let handled = |_: DrawClass| true;
    ip.prepare_camera(&input(&f, &selected, &level, &known, &handled), &|_| None)
        .unwrap();
    let draws = ip.draws(DrawClass::Opaque);
    assert_eq!(draws.len(), 2);
    assert_eq!(draws[0].mesh, MeshId(0));
    assert_eq!(draws[1].mesh, MeshId(90));
    assert_eq!(draw_ids(&ip, &draws[0]), vec![0, 2, 4]);
    assert_eq!(draw_ids(&ip, &draws[1]), vec![1, 3]);
}

#[test]
fn transparent_draws_one_per_id_back_to_front() {
    let f = Frame::new(&[
        (0, 1, [0.0, 0.0, -1.0], 30),
        (0, 1, [0.0, 0.0, -5.0], 31),
        (1, 3, [0.0, 0.0, -3.0], 4),
        (1, 1, [0.0, 0.0, -3.0], 9),
        (0, 1, [0.0, 0.0, 2.0], 7),
        (0, 0, [0.0, 0.0, -9.0], 1),
    ]);
    let mut ip = IdPrep::new(16, 8);
    let selected = |_| true;
    let level = |_: u32, mesh: MeshId| mesh;
    let known = |_: MeshId| true;
    let handled = |_: DrawClass| true;
    ip.prepare_camera(&input(&f, &selected, &level, &known, &handled), &|_| None)
        .unwrap();
    let draws = ip.draws(DrawClass::Transparent);
    assert_eq!(draws.len(), 5);
    let transparent_ids: Vec<u32> = draws.iter().flat_map(|d| draw_ids(&ip, d)).collect();
    assert_eq!(transparent_ids, vec![31, 4, 9, 30, 7]);
    assert!(draws.windows(2).all(|w| w[0].view_depth >= w[1].view_depth));
    assert_eq!(draws[1].material.0, 3);
    assert_eq!(draws[2].material.0, 1);
    for (k, d) in draws.iter().enumerate() {
        assert_eq!(d.instance_count, 1);
        assert_eq!(d.first_instance, 1 + k as u32);
    }
}

#[test]
fn cascade_appends_after_camera_and_camera_clears_it() {
    let f = Frame::new(&[
        (0, 0, [0.0, 0.0, 0.0], 1),
        (0, 0, [1.0, 0.0, 0.0], 2),
        (1, 0, [2.0, 0.0, 0.0], 3),
        (1, 0, [3.0, 0.0, 0.0], 4),
    ]);
    let mut ip = IdPrep::new(16, 8);
    let selected = |_| true;
    let level = |_: u32, mesh: MeshId| mesh;
    let known = |_: MeshId| true;
    let handled = |_: DrawClass| true;
    let inp = input(&f, &selected, &level, &known, &handled);
    ip.prepare_camera(&inp, &|_| None).unwrap();
    let camera_count = ip.camera_count();
    let camera_draws = ip.draws(DrawClass::Opaque).to_vec();
    let camera_ids = ids(&ip);
    assert!(camera_count > 0);
    assert_eq!(camera_count, ip.id_count());

    ip.prepare_cascade(0, &inp, &|_| None).unwrap();
    assert!(ip.id_count() > camera_count);
    assert_eq!(&ids(&ip)[..camera_count as usize], &camera_ids[..]);
    let cascade = ip.cascade_draws(0, DrawClass::Opaque);
    assert!(!cascade.is_empty());
    assert!(
        cascade
            .iter()
            .all(|d| d.first_instance >= camera_count && d.class == DrawClass::Opaque)
    );
    assert!(ip.cascade_draws(0, DrawClass::Transparent).is_empty());
    assert_eq!(ip.draws(DrawClass::Opaque), &camera_draws[..]);

    ip.prepare_camera(&inp, &|_| None).unwrap();
    assert!(ip.cascade_draws(0, DrawClass::Opaque).is_empty());
    assert!(ip.cascade_draws(0, DrawClass::Fur).is_empty());
    assert_eq!(ip.id_count(), ip.camera_count());
    assert_eq!(
        ip.id_bytes(),
        &camera_ids
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()[..]
    );
}

fn assert_refused(r: Result<(), RaaError>, code: &str, prep: &IdPrep) {
    let err = r.unwrap_err();
    assert_eq!(err.code(), code, "{err}");
    assert!(!err.message().is_empty(), "{code} has an empty message");
    assert_eq!(prep.id_count(), 0);
    for class in [DrawClass::Opaque, DrawClass::Fur, DrawClass::Transparent] {
        assert!(prep.draws(class).is_empty());
    }
    assert!(prep.cascade_draws(0, DrawClass::Opaque).is_empty());
}

#[test]
fn refusals_carry_their_code_and_leave_sections_empty() {
    let f = Frame::new(&[
        (0, 0, [0.0, 0.0, 0.0], 1),
        (0, 0, [1.0, 0.0, 0.0], 2),
        (0, 0, [2.0, 0.0, 0.0], 3),
    ]);
    let selected = |_| true;
    let level = |_: u32, mesh: MeshId| mesh;
    let known = |_: MeshId| true;
    let handled = |_: DrawClass| true;

    let mut ip = IdPrep::new(2, 8);
    assert_refused(
        ip.prepare_camera(&input(&f, &selected, &level, &known, &handled), &|_| None),
        "render.too_many_instances",
        &ip,
    );

    let mut ip = IdPrep::new(16, 8);
    let short_slots = &f.slots[..2];
    let inp = SelectInput {
        list: &f.list,
        tags: &f.tags,
        slots: short_slots,
        selected: &selected,
        materials: &f.materials,
        view: f.scene.camera.view(),
        known_mesh: &known,
        class_handled: &handled,
        level_mesh: &level,
    };
    assert_refused(
        ip.prepare_camera(&inp, &|_| None),
        "render.invalid_tags",
        &ip,
    );

    let mut ip = IdPrep::new(16, 8);
    let unknown_level = |_: u32, _: MeshId| MeshId(99);
    let known_but_level = |m: MeshId| m != MeshId(99);
    assert_refused(
        ip.prepare_camera(
            &input(&f, &selected, &unknown_level, &known_but_level, &handled),
            &|_| None,
        ),
        "render.unknown_mesh",
        &ip,
    );
}

#[test]
fn posed_items_follow_rest_and_skip_level_mesh() {
    let f = Frame::new(&[
        (0, 0, [0.0, 0.0, 0.0], 1),
        (0, 0, [1.0, 0.0, 0.0], 2),
        (0, 0, [2.0, 0.0, 0.0], 3),
        (0, 0, [3.0, 0.0, 0.0], 4),
    ]);
    let calls = RefCell::new(Vec::new());
    let level = |slot: u32, mesh: MeshId| {
        calls.borrow_mut().push(slot);
        mesh
    };
    let skinned = |tag: u32| match tag {
        2 => Some(SkinDraw { base: 0, skin: 9 }),
        4 => Some(SkinDraw { base: 1, skin: 3 }),
        _ => None,
    };
    let selected = |_| true;
    let known = |_: MeshId| true;
    let handled = |_: DrawClass| true;
    let mut ip = IdPrep::new(16, 8);
    ip.prepare_camera(&input(&f, &selected, &level, &known, &handled), &skinned)
        .unwrap();

    let called = calls.borrow().clone();
    assert_eq!(called, vec![1, 3]);
    assert!(!called.contains(&2) && !called.contains(&4));

    let draws = ip.draws(DrawClass::Opaque);
    assert_eq!(draws.len(), 3);
    assert_eq!(draws[0].mesh, MeshId(0));
    assert_eq!(draw_ids(&ip, &draws[0]), vec![1, 3]);
    assert_eq!(ip.mesh_binding(draws[1].mesh), (MeshId(0), Some(4)));
    assert_eq!(ip.mesh_binding(draws[2].mesh), (MeshId(0), Some(2)));
    assert_eq!(draw_ids(&ip, &draws[1]), vec![4]);
    assert_eq!(draw_ids(&ip, &draws[2]), vec![2]);
}

#[test]
fn frames_keep_their_preallocated_capacity() {
    let mut specs = Vec::new();
    for i in 0..8u32 {
        specs.push((i % 2, i % 4, [i as f32, 0.0, -(i as f32)], i));
    }
    let f = Frame::new(&specs);
    let mut ip = IdPrep::new(8, 8);
    let before = ip.capacity();
    assert_eq!(before.ids, 40);
    assert!(before.draws >= 8);
    assert!(before.sort >= 8);

    let selected = |_| true;
    let level = |_: u32, mesh: MeshId| mesh;
    let known = |_: MeshId| true;
    let handled = |_: DrawClass| true;
    let inp = input(&f, &selected, &level, &known, &handled);
    ip.prepare_camera(&inp, &|_| None).unwrap();
    let first = ip.capacity();
    assert_eq!(first, before);
    ip.prepare_cascade(0, &inp, &|_| None).unwrap();
    ip.prepare_cascade(1, &inp, &|_| None).unwrap();
    assert_eq!(ip.capacity(), first);
    ip.prepare_camera(&inp, &|_| None).unwrap();
    assert_eq!(ip.capacity(), first);
}
