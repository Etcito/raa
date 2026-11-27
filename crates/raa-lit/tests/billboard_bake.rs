//! Billboard bake (M6, contract m6-lit-bake-api): eight camera matrices, the common
//! source-root framing, hand-derived view-projection transforms and the four-vertex proxy,
//! plus one software-GPU smoke test that publishes and releases the three atlas images.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{block_on, skip, software_gpu};
use raa_core::{GpuContext, MaterialId, RaaError};
use raa_geom::builtin::DebugCube;
use raa_geom::{GeometryProvider, MeshData, ParamValue, Params, Quality};
use raa_lit::{
    AttributedMesh, BILLBOARD_DIRECTIONS, BillboardBakeInput, LitOptions, LitRenderer,
    billboard_camera, billboard_framing, billboard_layout, billboard_proxy,
};
use raa_math::{Aabb, Mat4, Quat, Vec3};
use raa_pipeline::Pipeline;
use raa_scene::{RenderList, Scene, Transform};

const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-5
}

fn close_vec(a: Vec3, b: Vec3) -> bool {
    close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z)
}

fn close_vec3(a: [f32; 3], b: [f32; 3]) -> bool {
    close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2])
}

#[test]
fn framing_is_common_and_square() {
    let bounds = Aabb {
        min: Vec3::new(-1.0, -2.0, -0.5),
        max: Vec3::new(3.0, 4.0, 0.5),
    };
    let framing = billboard_framing(&bounds).expect("framing");
    assert_eq!(framing.center, [1.0, 1.0, 0.0]);
    assert_eq!(framing.size, [6.0, 6.0]);
    assert!(billboard_framing(&Aabb::EMPTY).is_none());
    assert!(
        billboard_framing(&Aabb {
            min: Vec3::ZERO,
            max: Vec3::ZERO,
        })
        .is_none()
    );
}

#[test]
fn eight_cameras_share_the_common_frame() {
    let framing = billboard_framing(&Aabb {
        min: Vec3::new(-1.0, -2.0, -0.5),
        max: Vec3::new(3.0, 4.0, 0.5),
    })
    .expect("framing");
    let center = Vec3::from_array(framing.center);
    let frame = framing.size[0];
    let mut views = Vec::new();
    for direction in 0..BILLBOARD_DIRECTIONS as u8 {
        let camera = billboard_camera(framing, direction).expect("camera");
        let angle = f32::from(direction) * std::f32::consts::FRAC_PI_4;
        let expected = center + Vec3::new(angle.sin(), 0.0, angle.cos()) * frame;
        assert!(
            close_vec(camera.eye, expected),
            "{direction}: {:?}",
            camera.eye
        );
        assert!(close_vec(camera.target, center));
        assert_eq!(camera.up, Vec3::Y);
        assert_eq!(camera.ortho_height, Some(frame));
        assert_eq!(camera.aspect, 1.0);
        assert_eq!(camera.near, 0.0);
        assert_eq!(camera.far, 2.0 * frame);
        let projected = camera.view_proj().transform_point(center);
        assert!(
            close(projected.x, 0.0) && close(projected.y, 0.0) && close(projected.z, 0.5),
            "{direction}: {projected:?}"
        );
        views.push(camera.view_proj());
    }
    assert_eq!(views.len(), 8);
    for i in 0..views.len() {
        for j in i + 1..views.len() {
            assert_ne!(views[i], views[j], "views {i} and {j} differ");
        }
    }
    assert_eq!(
        billboard_camera(framing, 8).unwrap_err().code(),
        "billboard.invalid"
    );
}

#[test]
fn hand_derived_direction_transforms() {
    let framing = billboard_framing(&Aabb {
        min: Vec3::splat(-2.0),
        max: Vec3::splat(2.0),
    })
    .expect("framing");
    assert_eq!(framing.center, [0.0; 3]);
    assert_eq!(framing.size, [4.0, 4.0]);

    // Direction 0 looks from +Z: +X is screen right and +Y is screen up.
    let front = billboard_camera(framing, 0).unwrap();
    assert!(close_vec(front.eye, Vec3::new(0.0, 0.0, 4.0)));
    let p = front.view_proj().transform_point(Vec3::new(2.0, 2.0, 0.0));
    assert!(close_vec(p, Vec3::new(1.0, 1.0, 0.5)), "{p:?}");

    // Direction 2 looks from +X: +Z maps to screen left, +Y stays up.
    let side = billboard_camera(framing, 2).unwrap();
    assert!(close_vec(side.eye, Vec3::new(4.0, 0.0, 0.0)));
    let left = side.view_proj().transform_point(Vec3::new(0.0, 1.0, 2.0));
    let right = side.view_proj().transform_point(Vec3::new(0.0, 1.0, -2.0));
    assert!(close_vec(left, Vec3::new(-1.0, 0.5, 0.5)), "{left:?}");
    assert!(close_vec(right, Vec3::new(1.0, 0.5, 0.5)), "{right:?}");

    // Opposite views sit on opposite sides of the same center.
    let back = billboard_camera(framing, 4).unwrap();
    let left_side = billboard_camera(framing, 6).unwrap();
    assert!(close_vec(back.eye, Vec3::new(0.0, 0.0, -4.0)));
    assert!(close_vec(left_side.eye, Vec3::new(-4.0, 0.0, 0.0)));
}

#[test]
fn proxy_is_one_quad_in_rust() {
    let mesh = billboard_proxy([1.0, 2.0, 3.0], [4.0, 2.0]).expect("proxy");
    assert_eq!(mesh.vertex_count(), 4);
    assert_eq!(mesh.triangle_count(), 2);
    assert_eq!(
        mesh.positions,
        vec![
            [-1.0, 1.0, 3.0],
            [3.0, 1.0, 3.0],
            [3.0, 3.0, 3.0],
            [-1.0, 3.0, 3.0],
        ]
    );
    assert_eq!(mesh.normals, vec![[0.0, 0.0, 1.0]; 4]);
    assert_eq!(mesh.indices, vec![0, 1, 2, 0, 2, 3]);
    mesh.validate().expect("valid quad");
    assert_eq!(
        billboard_proxy([0.0, 0.0, 0.0], [0.0, 1.0])
            .unwrap_err()
            .code(),
        "billboard.invalid"
    );
    assert_eq!(
        billboard_proxy([f32::NAN, 0.0, 0.0], [1.0, 1.0])
            .unwrap_err()
            .code(),
        "billboard.invalid"
    );
}

fn cube(half: f32) -> AttributedMesh {
    let mut mesh = MeshData::new();
    let params = Params::new().with("half", ParamValue::Scalar(half));
    DebugCube
        .generate(&params, Quality::LOD0, &mut mesh)
        .expect("debug cube");
    AttributedMesh::from_mesh(mesh)
}

#[test]
fn bake_publishes_and_releases_the_three_images() {
    let test = "bake_publishes_and_releases_the_three_images";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let run = |gpu: &GpuContext| -> Result<(), RaaError> {
        let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, SURFACE, false)));
        let mut lit = LitRenderer::new(
            gpu,
            &pipeline,
            LitOptions {
                msaa: 1,
                ..LitOptions::default()
            },
        )?;
        let small = lit.upload(gpu, &cube(0.5))?;
        let small_mesh = lit.submesh_ids(small).expect("sub-meshes")[0];
        let large = lit.upload(gpu, &cube(1.0))?;
        let large_mesh = lit.submesh_ids(large).expect("sub-meshes")[0];
        let mut scene = Scene::with_capacity(1, 2);
        let visible_node = scene
            .add_node(
                None,
                Transform::from_trs(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE),
            )
            .expect("node");
        let hidden_node = scene
            .add_node(
                None,
                Transform::from_trs(Vec3::new(10.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
            )
            .expect("node");
        let visible = scene
            .add_instance(visible_node, small_mesh, MaterialId(0))
            .expect("visible instance");
        let hidden = scene
            .add_instance(hidden_node, large_mesh, MaterialId(0))
            .expect("hidden instance");
        scene.set_instance_tag(visible, 7).expect("tag");
        scene.set_instance_tag(hidden, 7).expect("tag");
        // The retained source is hidden but still submitted: the production selection
        // path must pick it up through build_all_render_list_slots.
        scene.set_instance_visible(hidden, false).expect("hide");
        scene.update_world();
        // The active view never sees the source: the bake must not need it on screen.
        scene.camera.eye = Vec3::new(100.0, 100.0, 100.0);
        let before = scene.camera;

        // Hand-derived bounds of both source draws in the root frame:
        // small cube half 0.5 at the origin and large cube half 1 at x 10.
        let result = block_on(lit.bake_billboard(
            gpu,
            &mut scene,
            BillboardBakeInput {
                tags: vec![7],
                root_world: Mat4::IDENTITY,
                resolution: 64,
            },
        ))?;
        assert_eq!(scene.camera, before, "the scene camera is restored");
        assert!(close_vec3(result.center, [5.25, 0.0, 0.0]));
        assert!(close(result.size[0], 11.5));
        assert_eq!(result.size[0], result.size[1]);
        // The visible-only view is unchanged: the hidden draw stays out of it.
        let mut visible_list = RenderList::with_capacity(2);
        scene.build_render_list(&mut visible_list);
        assert_eq!(visible_list.items.len(), 1);
        assert_eq!(visible_list.items[0].mesh, small_mesh);
        let layout = billboard_layout(64).unwrap();
        assert_eq!(result.tiles, layout.tiles);
        assert_eq!(result.extent, layout.extent);
        assert!(result.center.iter().all(|v| v.is_finite()));
        assert!(result.size[0] > 0.0 && result.size[1] > 0.0);
        assert_eq!(result.size[0], result.size[1]);
        for id in [result.diffuse, result.normal, result.opacity] {
            assert!(id > 0);
            assert!(lit.texture_rect(id).is_some(), "id {id} is published");
            assert!(lit.remove_texture(gpu, id)?);
            assert!(lit.texture_rect(id).is_none(), "id {id} is removed");
        }

        // Every submitted source hidden: both retained draws still bake, so the common
        // bounds stay the union derived above.
        scene.set_instance_visible(visible, false).expect("hide");
        scene.update_world();
        let all_hidden = block_on(lit.bake_billboard(
            gpu,
            &mut scene,
            BillboardBakeInput {
                tags: vec![7],
                root_world: Mat4::IDENTITY,
                resolution: 64,
            },
        ))?;
        assert_eq!(scene.camera, before, "the scene camera is restored");
        assert!(close_vec3(all_hidden.center, [5.25, 0.0, 0.0]));
        assert!(close(all_hidden.size[0], 11.5));
        assert!(close(all_hidden.size[1], 11.5));
        assert_eq!(all_hidden.tiles, layout.tiles);
        assert_eq!(all_hidden.extent, layout.extent);
        scene.build_render_list(&mut visible_list);
        assert!(
            visible_list.items.is_empty(),
            "the active view still hides both"
        );
        for id in [all_hidden.diffuse, all_hidden.normal, all_hidden.opacity] {
            assert!(lit.remove_texture(gpu, id)?);
        }

        // Empty, missing and invalid inputs refuse before any GPU work.
        assert_eq!(
            block_on(lit.bake_billboard(
                gpu,
                &mut scene,
                BillboardBakeInput {
                    tags: Vec::new(),
                    root_world: Mat4::IDENTITY,
                    resolution: 64,
                },
            ))
            .unwrap_err()
            .code(),
            "billboard.invalid"
        );
        assert_eq!(
            block_on(lit.bake_billboard(
                gpu,
                &mut scene,
                BillboardBakeInput {
                    tags: vec![99],
                    root_world: Mat4::IDENTITY,
                    resolution: 64,
                },
            ))
            .unwrap_err()
            .code(),
            "billboard.invalid"
        );
        assert_eq!(
            block_on(lit.bake_billboard(
                gpu,
                &mut scene,
                BillboardBakeInput {
                    tags: vec![7],
                    root_world: Mat4::IDENTITY,
                    resolution: 100,
                },
            ))
            .unwrap_err()
            .code(),
            "billboard.invalid"
        );
        Ok(())
    };
    if let Err(e) = run(&gpu) {
        panic!("bake failed: {e}");
    }
    let error = block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");
}
