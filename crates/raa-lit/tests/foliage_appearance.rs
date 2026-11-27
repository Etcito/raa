//! Unlit source appearance for foliage baking (M7, contract `m7-appearance`): dispatch
//! bounds, empty work, owned sample packing and the hand-derived barycentric source UV.
//! The GPU pass itself is accepted by the final M7 inventory, not here.

use raa_lit::LitHandle;

#[allow(dead_code)]
#[path = "../src/foliage_appearance.rs"]
mod foliage_appearance;
#[allow(dead_code)]
#[allow(clippy::wrong_self_convention)]
#[path = "../src/foliage_resources.rs"]
mod foliage_resources;
#[allow(dead_code)]
#[path = "../src/foliage_sampling.rs"]
mod foliage_sampling;

use foliage_appearance::{
    SAMPLE_INPUT_STRIDE, SAMPLE_OUTPUT_STRIDE, checked_dispatch, compose_appearance, dispatch_size,
    pack_sample, pack_samples, sample_uv,
};
use foliage_resources::{FoliageSource, SourceTriangle};
use foliage_sampling::FoliageSample;

fn triangle() -> SourceTriangle {
    SourceTriangle {
        positions: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        normals: [[0.0, 0.0, 1.0]; 3],
        uv: [[0.2, 0.4], [0.6, 0.8], [1.0, 0.0]],
        material: 3,
        source_tag: 9,
        foliage: true,
    }
}

fn source() -> FoliageSource {
    FoliageSource {
        triangles: vec![triangle()],
        min: [0.0; 3],
        max: [1.0, 1.0, 0.0],
        root_world: raa_math::Mat4::IDENTITY,
        generation: 4,
    }
}

fn sample() -> FoliageSample {
    FoliageSample {
        source_triangle: 0,
        barycentric: [0.25, 0.25, 0.5],
        position: [0.5, 0.5, 0.0],
        normal: [0.0, 0.0, 1.0],
        corners: [[0.0; 3]; 3],
        represented_area: 0.5,
    }
}

fn f32_at(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

#[test]
fn dispatch_is_sixty_four_threads_and_bounded_by_the_limit() {
    assert_eq!(
        [0, 1, 63, 64, 65, 128, 129].map(dispatch_size),
        [0, 1, 1, 1, 2, 2, 3]
    );
    assert_eq!(checked_dispatch(0, 4), Some(0));
    assert_eq!(checked_dispatch(256, 4), Some(4));
    assert_eq!(checked_dispatch(257, 4), None);
    assert_eq!(checked_dispatch(64 * 65535, 65535), Some(65535));
    assert_eq!(checked_dispatch(64 * 65535 + 1, 65535), None);
}

#[test]
fn barycentric_uv_is_the_hand_computed_triangle_interpolation() {
    let uv = sample_uv(&triangle(), [0.5, 0.25, 0.25]);
    assert!((uv[0] - 0.5).abs() < 1e-6, "u = {}", uv[0]);
    assert!((uv[1] - 0.4).abs() < 1e-6, "v = {}", uv[1]);
    assert_eq!(sample_uv(&triangle(), [1.0, 0.0, 0.0]), [0.2, 0.4]);
    assert_eq!(sample_uv(&triangle(), [0.0, 1.0, 0.0]), [0.6, 0.8]);
    assert_eq!(sample_uv(&triangle(), [0.0, 0.0, 1.0]), [1.0, 0.0]);
}

#[test]
fn packed_sample_owns_its_lanes_after_the_source_is_dropped() {
    let packed = {
        let source = source();
        pack_sample(&source, &sample()).unwrap()
    };
    assert_eq!(packed.len(), SAMPLE_INPUT_STRIDE);
    for (offset, expected) in [(0, [0.5, 0.5, 0.0]), (16, [0.0, 0.0, 1.0])] {
        for (i, value) in expected.iter().enumerate() {
            assert_eq!(f32_at(&packed, offset + 4 * i), *value);
        }
        assert_eq!(f32_at(&packed, offset + 12), 0.0);
    }
    let uv = sample_uv(&triangle(), sample().barycentric);
    assert_eq!(f32_at(&packed, 32), uv[0]);
    assert_eq!(f32_at(&packed, 36), uv[1]);
    assert_eq!(f32_at(&packed, 40), 0.0);
    assert_eq!(f32_at(&packed, 44), 0.0);
    for (i, value) in sample().barycentric.iter().enumerate() {
        assert_eq!(f32_at(&packed, 48 + 4 * i), *value);
    }
    assert_eq!(f32_at(&packed, 60), 0.0);
    assert_eq!(u32_at(&packed, 64), 0);
    assert_eq!(u32_at(&packed, 68), 3);
    assert_eq!(u32_at(&packed, 72), 0);
    assert_eq!(u32_at(&packed, 76), 0);
}

#[test]
fn empty_work_packs_to_an_empty_owned_buffer() {
    assert!(pack_samples(&source(), &[]).unwrap().is_empty());
    assert_eq!(dispatch_size(0), 0);
    assert_eq!(SAMPLE_OUTPUT_STRIDE, 16);
}

#[test]
fn invalid_samples_are_refused_before_submission() {
    let source = source();
    let mut missing = sample();
    missing.source_triangle = 7;
    assert_eq!(
        pack_sample(&source, &missing).unwrap_err().code(),
        "foliage.invalid"
    );
    let mut nonfinite = sample();
    nonfinite.position[0] = f32::NAN;
    assert_eq!(
        pack_sample(&source, &nonfinite).unwrap_err().code(),
        "foliage.invalid"
    );
    assert!(pack_samples(&source, &[missing]).is_err());
}

#[test]
fn composed_module_validates_the_source_material_function() {
    let registry = raa_shader::MaterialFnRegistry::new();
    let standard = registry
        .get(raa_shader::MaterialFnId::STANDARD)
        .expect("raa.standard is always registered")
        .wgsl
        .clone();
    let module = compose_appearance(&standard).unwrap();
    assert!(module.contains("fn material_main"));
    assert!(module.contains("fn sample_albedo"));
    assert!(!module.contains("raa_previous_instances"));
    let broken = "fn material_main(in: MaterialInput) -> MaterialOutput { return nope; }";
    assert_eq!(
        compose_appearance(broken).unwrap_err().code(),
        "foliage.bake_failed"
    );
}
