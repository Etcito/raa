//! FramePrep integration tests: S1 byte offsets, refusal, determinism, buffer reuse.

use raa_core::{MaterialId, MeshId, RaaError};
use raa_math::{Mat4, Quat, Vec3};
use raa_render::{DrawRange, FramePrep};
use raa_scene::{Batch, RenderItem, RenderList, Scene, Transform};

fn u32_at(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
}

fn f32s_at(bytes: &[u8], off: usize, n: usize) -> Vec<f32> {
    bytes[off..off + 4 * n]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() <= 1e-6, "got {got:?}, want {want:?}");
    }
}

/// 3 instances on 2 meshes. After sorting by (mesh, material, slot):
/// item 0 = translated node (mesh 0, material 5), item 1 = rotated node (mesh 0,
/// material 6), item 2 = scaled node (mesh 1, material 7).
fn three_instance_scene() -> (Scene, RenderList) {
    let mut scene = Scene::with_capacity(4, 4);
    scene.camera.eye = Vec3::new(1.5, 2.5, 7.0);
    let translated = scene
        .add_node(
            None,
            Transform::from_trs(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY, Vec3::ONE),
        )
        .unwrap();
    let scaled = scene
        .add_node(
            None,
            Transform::from_trs(Vec3::ZERO, Quat::IDENTITY, Vec3::new(2.0, 4.0, 0.5)),
        )
        .unwrap();
    let rotated = scene
        .add_node(
            None,
            Transform::from_trs(
                Vec3::new(-1.0, 0.0, 0.0),
                Quat::from_rotation_y(0.5),
                Vec3::ONE,
            ),
        )
        .unwrap();
    scene
        .add_instance(scaled, MeshId(1), MaterialId(7))
        .unwrap();
    scene
        .add_instance(translated, MeshId(0), MaterialId(5))
        .unwrap();
    scene
        .add_instance(rotated, MeshId(0), MaterialId(6))
        .unwrap();
    scene.update_world();
    let mut list = RenderList::with_capacity(8);
    scene.build_render_list(&mut list);
    (scene, list)
}

#[test]
fn s1_byte_layout_of_instances_uniform_and_draws() {
    let (scene, list) = three_instance_scene();
    let mut prep = FramePrep::new(16);
    let counts = prep.prepare_cpu(&scene, &list, true).unwrap();
    assert_eq!(counts.instances, 3);
    assert_eq!(counts.draw_calls, 2);
    assert_eq!(counts.triangles, 0);

    let inst = prep.instance_bytes();
    assert_eq!(inst.len(), 3 * 128);

    // Item 0: model column 3 holds the translation, w = 1.
    assert_eq!(f32s_at(inst, 48, 3), [1.0, 2.0, 3.0]);
    assert_eq!(f32s_at(inst, 60, 1), [1.0]);
    assert_eq!(u32_at(inst, 112), 5);
    assert!(inst[116..128].iter().all(|&b| b == 0));

    // Item 1 is the rotated node on mesh 0.
    assert_eq!(u32_at(inst, 128 + 112), 6);

    // Item 2: the scaled node's normal matrix is the inverse scale, columns with w = 0.
    let base = 2 * 128;
    assert_close(
        &f32s_at(inst, base + 64, 12),
        &[
            0.5, 0.0, 0.0, 0.0, //
            0.0, 0.25, 0.0, 0.0, //
            0.0, 0.0, 2.0, 0.0,
        ],
    );
    assert_eq!(u32_at(inst, base + 112), 7);

    let u = prep.frame_uniform_bytes();
    assert_eq!(u.len(), 144);
    let mut vp = [0u8; 64];
    scene.camera.view_proj().write_le_bytes(&mut vp);
    assert_eq!(&u[0..64], &vp);
    assert_eq!(f32s_at(u, 64, 4), [1.5, 2.5, 7.0, 1.0]);
    let d = scene.light.direction.normalize_or_zero();
    assert_close(&f32s_at(u, 80, 3), &[d.x, d.y, d.z]);
    assert_eq!(f32s_at(u, 92, 1), [0.0]);
    assert_eq!(u32_at(u, 128), 0);

    assert_eq!(
        list.batches,
        [
            Batch {
                mesh: MeshId(0),
                first: 0,
                count: 2
            },
            Batch {
                mesh: MeshId(1),
                first: 2,
                count: 1
            },
        ]
    );
    let draws = prep.draws();
    assert_eq!(draws.len(), 2);
    for (draw, batch) in draws.iter().zip(&list.batches) {
        assert_eq!(
            *draw,
            DrawRange {
                mesh: batch.mesh,
                first_instance: batch.first,
                instance_count: batch.count,
            }
        );
    }
}

#[test]
fn too_many_instances_is_refused_with_render_code() {
    let mut list = RenderList::with_capacity(11);
    for i in 0..11 {
        list.items.push(RenderItem {
            mesh: MeshId(0),
            material: MaterialId(i),
            world: Mat4::IDENTITY,
        });
    }
    list.batches.push(Batch {
        mesh: MeshId(0),
        first: 0,
        count: 11,
    });
    let mut prep = FramePrep::new(10);
    let err = prep.prepare_cpu(&Scene::new(), &list, true).unwrap_err();
    assert_eq!(err.code(), "core.resource");
    // Display prefixes the core code; the error's own message starts with the render code.
    let RaaError::Resource(msg) = &err else {
        panic!("expected RaaError::Resource, got {err:?}");
    };
    assert!(
        msg.starts_with("render.too_many_instances"),
        "message was {msg:?}"
    );
    assert!(prep.instance_bytes().is_empty());
    assert!(prep.draws().is_empty());
}

#[test]
fn identical_scenes_give_identical_bytes() {
    let (scene_a, list_a) = three_instance_scene();
    let (scene_b, list_b) = three_instance_scene();
    let mut prep_a = FramePrep::new(8);
    let mut prep_b = FramePrep::new(8);
    prep_a.prepare_cpu(&scene_a, &list_a, false).unwrap();
    prep_b.prepare_cpu(&scene_b, &list_b, false).unwrap();
    assert_eq!(prep_a.instance_bytes(), prep_b.instance_bytes());
    assert_eq!(prep_a.frame_uniform_bytes(), prep_b.frame_uniform_bytes());
    assert_eq!(prep_a.draws(), prep_b.draws());
}

#[test]
fn second_prepare_keeps_buffers() {
    let (scene, list) = three_instance_scene();
    let mut prep = FramePrep::new(8);
    prep.prepare_cpu(&scene, &list, true).unwrap();
    let inst_ptr = prep.instance_bytes().as_ptr();
    let draws_ptr = prep.draws().as_ptr();
    let first = prep.instance_bytes().to_vec();

    prep.prepare_cpu(&scene, &list, true).unwrap();
    // Same backing storage: nothing was reallocated or grown.
    assert_eq!(prep.instance_bytes().as_ptr(), inst_ptr);
    assert_eq!(prep.draws().as_ptr(), draws_ptr);
    assert_eq!(prep.instance_bytes(), &first[..]);
}
