//! Reference views and convex-hull bake layers (M7, contract m7-reference-views): the
//! deterministic camera basis and near/far values against hand calculations, the
//! world-to-root normal transform, the accepted view counts and the refusal of empty or
//! mismatched reference extents.

#![cfg(not(target_arch = "wasm32"))]

use raa_core::testing::{skip, software_gpu};
use raa_core::{RaaError, wgpu};
use raa_lit::foliage_reference::{
    reference_cameras, validate_reference_extents, validate_resolution, validate_view_count,
    world_to_root_normal,
};
use raa_lit::foliage_resources::FoliageSource;
use raa_lit::{LitOptions, LitRenderer};
use raa_math::{Mat4, Quat, Vec3};
use raa_pipeline::Pipeline;
use std::cell::RefCell;
use std::rc::Rc;

fn invalid_code() -> &'static str {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message: String::new(),
    }
    .code()
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-4 * a.abs().max(b.abs()).max(1.0)
}

fn apply_normal(m: &[[f32; 4]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z,
        m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z,
        m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z,
    )
}

#[test]
fn view_counts_outside_the_optimization_range_are_refused() {
    for refused in [0, 1, 7, 9, 32, 63, 129, 256] {
        let err = validate_view_count(refused).expect_err("count must be refused");
        assert_eq!(
            err.code(),
            invalid_code(),
            "views {refused}: {}",
            err.message()
        );
    }
    validate_view_count(8).expect("impostor preparation uses 8 views");
    for accepted in [64, 65, 96, 127, 128] {
        validate_view_count(accepted).expect("optimization counts 64..=128");
    }
    for refused in [0, 63, 65, 511] {
        let err = validate_resolution(refused).expect_err("resolution must be refused");
        assert_eq!(err.code(), invalid_code());
    }
    for accepted in [64, 128, 256, 512] {
        validate_resolution(accepted).expect("supported texture resolutions");
    }
}

#[test]
fn spherical_cameras_match_hand_computed_basis_and_range() {
    let min = [-1.0, -1.0, -1.0];
    let max = [1.0, 1.0, 1.0];
    let cameras = reference_cameras(min, max, Mat4::IDENTITY, 8).expect("cameras");
    assert_eq!(cameras.len(), 8);
    let radius = 0.5f32 * Vec3::new(2.0, 2.0, 2.0).length();
    let distance = 3.0 * radius;
    let near = distance - 2.0 * radius;
    let far = distance + 2.0 * radius;
    for camera in &cameras {
        assert!(close(camera.near, near), "{} != {near}", camera.near);
        assert!(close(camera.far, far), "{} != {far}", camera.far);
        let offset = camera.root.eye - camera.root.target;
        assert!(close(offset.length(), distance));
        let forward = (camera.root.target - camera.root.eye).normalize_or_zero();
        assert!(forward.dot(camera.root.up).abs() < 1e-5);
        assert!(camera.root.eye.is_finite());
        // Identity root frame keeps the world view-projection equal to the root one.
        assert_eq!(camera.world_to_clip, camera.root_to_clip);
        assert_eq!(camera.eye_world, camera.root.eye);
        assert_eq!(
            camera.root_to_clip.to_cols_array(),
            camera.root.view_proj().to_cols_array()
        );
    }
    // A non-identity source-root frame: the world camera and matrices carry the root
    // transform, so world positions project exactly as source-root positions do.
    let root_world = Mat4::from_trs(
        Vec3::new(2.0, 1.0, -3.0),
        Quat::from_rotation_y(0.7),
        Vec3::new(1.5, 2.0, 0.5),
    );
    let framed = reference_cameras(min, max, root_world, 8).expect("framed cameras");
    for camera in &framed {
        assert_eq!(
            camera.eye_world,
            root_world.transform_point(camera.root.eye)
        );
        for root_point in [Vec3::ZERO, Vec3::new(0.4, -0.2, 0.7)] {
            let world = root_world.transform_point(root_point);
            let mapped = camera.world_to_clip.transform_point(world);
            let expected = camera.root_to_clip.transform_point(root_point);
            assert!((mapped - expected).length() < 1e-4);
        }
    }
    // Deterministic: the same request generates byte-identical cameras.
    let again = reference_cameras(min, max, Mat4::IDENTITY, 8).expect("cameras");
    assert_eq!(cameras, again);
    let err = reference_cameras(min, max, Mat4::IDENTITY, 7).unwrap_err();
    assert_eq!(err.code(), invalid_code());
}

#[test]
fn world_to_root_normal_transform_matches_hand_calculation() {
    let root = Mat4::from_trs(
        Vec3::new(3.0, -2.0, 1.0),
        Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
        Vec3::new(2.0, 1.0, 0.5),
    );
    let root_to_world = root.normal_matrix();
    let world_to_root = world_to_root_normal(root).expect("normal matrix");
    for root_normal in [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.3, -0.4, 0.866),
    ] {
        let world = apply_normal(&root_to_world, root_normal);
        let back = apply_normal(&world_to_root, world);
        assert!(
            (back - root_normal).length() < 1e-4,
            "{back:?} != {root_normal:?}"
        );
    }
    // A pure translation has an identity normal transform.
    let translated = Mat4::from_translation(Vec3::new(5.0, 6.0, 7.0));
    let identity = world_to_root_normal(translated).expect("normal matrix");
    assert_eq!(identity[0][0], 1.0);
    assert_eq!(identity[1][1], 1.0);
    assert_eq!(identity[2][2], 1.0);
    let singular = Mat4::default();
    let err = world_to_root_normal(singular).unwrap_err();
    assert_eq!(err.code(), invalid_code());
}

#[test]
fn mismatched_reference_extents_are_refused() {
    let layer = [64, 64, 8];
    validate_reference_extents(layer, layer, layer, 8).expect("consistent layers");
    for (albedo, normal, depth, views) in [
        ([64, 64, 8], [64, 64, 4], [64, 64, 8], 8),
        ([64, 64, 8], [64, 128, 8], [64, 64, 8], 8),
        ([64, 64, 8], [64, 64, 8], [64, 64, 8], 7),
        ([0, 0, 0], [0, 0, 0], [0, 0, 0], 0),
    ] {
        let err = validate_reference_extents(albedo, normal, depth, views)
            .expect_err("extents must be refused");
        assert_eq!(err.code(), invalid_code(), "{}", err.message());
    }
}

#[test]
fn empty_source_is_refused_before_any_gpu_work() {
    let test = "empty_source_is_refused_before_any_gpu_work";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let pipeline = Rc::new(RefCell::new(Pipeline::new(
        &gpu,
        wgpu::TextureFormat::Rgba8Unorm,
        false,
    )));
    let lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit renderer");
    let source = FoliageSource {
        triangles: Vec::new(),
        min: [0.0; 3],
        max: [0.0; 3],
        root_world: Mat4::IDENTITY,
        generation: 0,
    };
    let result =
        raa_lit::foliage_reference::render_foliage_references(&gpu, &lit.handle(), &source, 8, 64);
    let err = match result {
        Ok(_) => panic!("an empty source must be refused"),
        Err(err) => err,
    };
    assert_eq!(err.code(), invalid_code());
}

/// A one-triangle source in source-root metres with a +Z shading normal.
fn triangle_source() -> FoliageSource {
    use raa_lit::foliage_resources::SourceTriangle;
    FoliageSource {
        triangles: vec![SourceTriangle {
            positions: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: [[0.0, 0.0, 1.0]; 3],
            uv: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            material: 0,
            source_tag: 0,
            foliage: true,
        }],
        min: [-0.5, -0.5, -0.5],
        max: [1.5, 1.5, 0.5],
        root_world: Mat4::IDENTITY,
        generation: 1,
    }
}

#[test]
fn references_and_packed_layers_submit_on_the_gpu() {
    let test = "references_and_packed_layers_submit_on_the_gpu";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let pipeline = Rc::new(RefCell::new(Pipeline::new(
        &gpu,
        wgpu::TextureFormat::Rgba8Unorm,
        false,
    )));
    let lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit renderer");
    let source = triangle_source();
    let references =
        raa_lit::foliage_reference::render_foliage_references(&gpu, &lit.handle(), &source, 8, 64)
            .expect("references");
    assert_eq!(references.views.len(), 8);
    for view in &references.views {
        assert!(view.0.iter().all(|v| v.is_finite()));
        assert!(view.1[0] > 0.0 && view.1[1] > view.1[0]);
    }
    let packed = raa_lit::foliage_reference::pack_foliage_layers(&gpu, &source, &references)
        .expect("packed layers");
    assert_eq!(packed.views, references.views);
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
}
