use raa_material::{Material, MaterialLibrary, WGSL_SHADING};

#[test]
fn wgsl_shading_declares_structs_and_functions() {
    for item in [
        "struct MaterialGpu",
        "struct FrameUniform",
        "fn raa_shade",
        "fn raa_encode_srgb",
    ] {
        assert!(WGSL_SHADING.contains(item), "WGSL_SHADING lacks `{item}`");
    }
}

#[test]
fn from_srgb_converts_base_color_to_linear() {
    let m = Material::from_srgb([1.0, 0.5, 0.0], 0.5, 0.0, [0.0; 3]);
    assert!((m.base_color[0] - 1.0).abs() < 1e-6);
    assert!((m.base_color[1] - 0.21404).abs() < 1e-5);
    assert_eq!(m.base_color[2], 0.0);
}

#[test]
fn library_of_three_materials_writes_144_bytes() {
    let mut lib = MaterialLibrary::new();
    lib.add(Material::default()).unwrap();
    lib.add(Material::flat([1.0, 0.0, 0.0])).unwrap();
    lib.add(Material::pbr([0.2, 0.4, 0.8], 0.3, 1.0)).unwrap();
    let mut out = Vec::new();
    lib.write_gpu_all(&mut out);
    assert_eq!(out.len(), 144);
    assert!(!lib.is_dirty());
}

#[test]
fn every_validate_refusal_has_a_message() {
    let bad = [
        Material::pbr([f32::NAN, 0.0, 0.0], 0.5, 0.0),
        Material::pbr([-0.1, 0.0, 0.0], 0.5, 0.0),
        Material {
            emissive: [0.0, f32::INFINITY, 0.0],
            ..Material::default()
        },
        Material {
            emissive: [0.0, 0.0, -1.0],
            ..Material::default()
        },
        Material::pbr([0.5; 3], f32::NAN, 0.0),
        Material::pbr([0.5; 3], 1.5, 0.0),
        Material::pbr([0.5; 3], 0.5, f32::NEG_INFINITY),
        Material::pbr([0.5; 3], 0.5, -0.5),
    ];
    for m in bad {
        let err = m.validate().unwrap_err();
        assert_eq!(err.code(), "material.invalid");
        assert!(!err.message.is_empty(), "empty message for {m:?}");
    }
}
