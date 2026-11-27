use raa_lit::{
    DrawClass, FLAG_DIFFUSE_TRANSMISSION, FLAG_DOUBLE_SIDED, FLAG_FUR, FLAG_NO_SHADOW, FLAG_TRANSMISSIVE, FLAG_TRANSPARENT,
    FLAG_UNLIT, FLAG_VERTEX_FN, MaterialFnId, SURFACE_MATERIAL_GPU_SIZE, SurfaceMaterial,
    SurfaceModel,
};
use raa_material::Material;

fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn vec4_at(b: &[u8], o: usize) -> [f32; 4] {
    [
        f32_at(b, o),
        f32_at(b, o + 4),
        f32_at(b, o + 8),
        f32_at(b, o + 12),
    ]
}

fn distinct() -> SurfaceMaterial {
    let mut params = [[0.0; 4]; 8];
    for (i, p) in params.iter_mut().enumerate() {
        for (j, v) in p.iter_mut().enumerate() {
            *v = 100.0 + (i * 4 + j) as f32;
        }
    }
    SurfaceMaterial {
        model: SurfaceModel::Function(MaterialFnId(7)),
        base_color: [0.11, 0.12, 0.13],
        opacity: 0.14,
        emissive: [0.21, 0.22, 0.23],
        roughness: 0.31,
        metallic: 0.32,
        clearcoat: 0.33,
        clearcoat_roughness: 0.34,
        sheen_color: [0.41, 0.42, 0.43],
        sheen_roughness: 0.44,
        transmission: 0.51,
        ior: 1.52,
        thickness: 0.53,
        attenuation_color: [0.61, 0.62, 0.63],
        attenuation_distance: 0.64,
        diffuse_transmission: 0.55,
        diffuse_transmission_color: [1.0, 0.5, 0.0],
        flags: FLAG_DOUBLE_SIDED | FLAG_NO_SHADOW,
        params,
        fur: [0.71, 0.72, 0.73, 32.0],
        data: 4097,
        // Ray-only metadata, not part of the raster GPU record.
        trace_base_color: None,
    }
}

#[test]
fn write_gpu_lays_out_every_256_byte_offset() {
    let mut out = vec![0xAA; 4];
    distinct().write_gpu(&mut out);
    assert_eq!(out.len(), 4 + SURFACE_MATERIAL_GPU_SIZE);
    assert_eq!(&out[..4], &[0xAA; 4], "write_gpu appends");
    let b = &out[4..];
    assert_eq!(vec4_at(b, 0), [0.11, 0.12, 0.13, 0.14]);
    // emissive.w: the diffuse transmission colour, 255 + 256 x 128 (0.5 x 255 rounds up) + 0.
    assert_eq!(vec4_at(b, 16), [0.21, 0.22, 0.23, 33023.0]);
    assert_eq!(vec4_at(b, 32), [0.31, 0.32, 0.33, 0.34]);
    assert_eq!(vec4_at(b, 48), [0.41, 0.42, 0.43, 0.44]);
    assert_eq!(vec4_at(b, 64), [0.51, 1.52, 0.53, 0.55]);
    assert_eq!(vec4_at(b, 80), [0.61, 0.62, 0.63, 0.64]);
    assert_eq!(
        [
            u32_at(b, 96),
            u32_at(b, 100),
            u32_at(b, 104),
            u32_at(b, 108)
        ],
        [
            2,
            7,
            FLAG_DOUBLE_SIDED | FLAG_NO_SHADOW | FLAG_DIFFUSE_TRANSMISSION,
            4097
        ]
    );
    for i in 0..8 {
        let base = 100.0 + (i * 4) as f32;
        assert_eq!(
            vec4_at(b, 112 + 16 * i),
            [base, base + 1.0, base + 2.0, base + 3.0],
            "p[{i}]"
        );
    }
    assert_eq!(vec4_at(b, 240), [0.71, 0.72, 0.73, 32.0]);
}

#[test]
fn write_gpu_clamps_roughness_and_writes_infinity_as_1e30() {
    for (r, want) in [(0.0, 0.03), (0.01, 0.03), (1.0, 1.0)] {
        let m = SurfaceMaterial {
            roughness: r,
            ..SurfaceMaterial::default()
        };
        let mut out = Vec::new();
        m.write_gpu(&mut out);
        assert_eq!(f32_at(&out, 32), want);
        assert_eq!(f32_at(&out, 92), 1e30);
    }
}

#[test]
fn header_model_and_fn_id_per_model() {
    for (model, want) in [
        (SurfaceModel::Flat, [0, 0]),
        (SurfaceModel::Standard, [1, 0]),
        (SurfaceModel::Function(MaterialFnId(3)), [2, 3]),
    ] {
        let mut out = Vec::new();
        SurfaceMaterial {
            model,
            ..SurfaceMaterial::default()
        }
        .write_gpu(&mut out);
        assert_eq!([u32_at(&out, 96), u32_at(&out, 100)], want, "{model:?}");
    }
}

#[test]
fn from_v0_flat_maps_to_model_flat() {
    let mut v0 = Material::flat([0.2, 0.3, 0.4]);
    v0.emissive = [1.0, 2.0, 3.0];
    let m = SurfaceMaterial::from_v0(&v0);
    let want = SurfaceMaterial {
        model: SurfaceModel::Flat,
        base_color: [0.2, 0.3, 0.4],
        emissive: [1.0, 2.0, 3.0],
        roughness: 1.0,
        metallic: 0.0,
        ..SurfaceMaterial::default()
    };
    assert_eq!(m, want);
    assert!(m.validate().is_ok());
}

#[test]
fn from_v0_pbr_maps_to_standard_with_other_fields_default() {
    let v0 = Material {
        emissive: [0.5, 0.0, 0.25],
        ..Material::pbr([0.7, 0.1, 0.2], 0.35, 0.8)
    };
    let m = SurfaceMaterial::from_v0(&v0);
    assert_eq!(m.model, SurfaceModel::Standard);
    assert_eq!(m.base_color, [0.7, 0.1, 0.2]);
    assert_eq!(m.roughness, 0.35);
    assert_eq!(m.metallic, 0.8);
    assert_eq!(m.emissive, [0.5, 0.0, 0.25]);
    let d = SurfaceMaterial::default();
    assert_eq!(m.opacity, 1.0);
    assert_eq!(m.ior, 1.5);
    assert_eq!(m.attenuation_distance, f32::INFINITY);
    assert_eq!(
        SurfaceMaterial {
            model: d.model,
            base_color: d.base_color,
            roughness: d.roughness,
            metallic: d.metallic,
            emissive: d.emissive,
            ..m
        },
        d
    );
}

fn assert_refused(m: SurfaceMaterial, field: &str) {
    let err = m.validate().unwrap_err();
    assert_eq!(err.code(), "material.invalid", "{field}");
    assert!(
        err.message().contains(field),
        "{field}: message `{}`",
        err.message()
    );
}

#[test]
fn validate_refuses_each_field_by_name() {
    let d = SurfaceMaterial::default;
    assert!(d().validate().is_ok());
    assert!(distinct().validate().is_ok());
    for (field, set) in [
        (
            "base_color",
            (|m, c| m.base_color = c) as fn(&mut SurfaceMaterial, [f32; 3]),
        ),
        ("emissive", |m, c| m.emissive = c),
        ("sheen_color", |m, c| m.sheen_color = c),
        ("attenuation_color", |m, c| m.attenuation_color = c),
    ] {
        for bad in [
            [0.0, -0.1, 0.0],
            [f32::NAN, 0.0, 0.0],
            [0.0, 0.0, f32::INFINITY],
        ] {
            let mut m = d();
            set(&mut m, bad);
            assert_refused(m, field);
        }
    }
    for (field, set) in [
        (
            "opacity",
            (|m, v| m.opacity = v) as fn(&mut SurfaceMaterial, f32),
        ),
        ("roughness", |m, v| m.roughness = v),
        ("metallic", |m, v| m.metallic = v),
        ("clearcoat_roughness", |m, v| m.clearcoat_roughness = v),
        ("clearcoat", |m, v| m.clearcoat = v),
        ("sheen_roughness", |m, v| m.sheen_roughness = v),
        ("transmission", |m, v| m.transmission = v),
        ("diffuse_transmission", |m, v| m.diffuse_transmission = v),
    ] {
        for bad in [-0.01, 1.01, f32::NAN, f32::INFINITY] {
            let mut m = d();
            set(&mut m, bad);
            assert_refused(m, field);
        }
    }
    for bad in [0.99, 3.01, f32::NAN, f32::INFINITY] {
        assert_refused(SurfaceMaterial { ior: bad, ..d() }, "ior");
    }
    for ok in [1.0, 3.0] {
        assert!(SurfaceMaterial { ior: ok, ..d() }.validate().is_ok());
    }
    for bad in [-0.01, f32::NAN, f32::INFINITY] {
        assert_refused(
            SurfaceMaterial {
                thickness: bad,
                ..d()
            },
            "thickness",
        );
    }
    for bad in [0.0, -1.0, f32::NAN, f32::NEG_INFINITY] {
        assert_refused(
            SurfaceMaterial {
                attenuation_distance: bad,
                ..d()
            },
            "attenuation_distance",
        );
    }
    for bad in [-0.01, 1.01, f32::NAN] {
        let mut m = d();
        m.diffuse_transmission_color[2] = bad;
        assert_refused(m, "diffuse_transmission_color[2]");
    }
    let mut m = d();
    m.params[5][2] = f32::NAN;
    assert_refused(m, "params[5][2]");
    let mut m = d();
    m.fur[1] = f32::INFINITY;
    assert_refused(m, "fur[1]");
}

#[test]
fn class_maps_flags() {
    let with = |flags| SurfaceMaterial {
        flags,
        ..SurfaceMaterial::default()
    };
    assert_eq!(with(0).class(), DrawClass::Opaque);
    assert_eq!(
        with(FLAG_DOUBLE_SIDED | FLAG_NO_SHADOW | FLAG_VERTEX_FN | FLAG_UNLIT).class(),
        DrawClass::Opaque
    );
    assert_eq!(with(FLAG_TRANSPARENT).class(), DrawClass::Transparent);
    assert_eq!(with(FLAG_TRANSMISSIVE).class(), DrawClass::Transparent);
    assert_eq!(with(FLAG_FUR).class(), DrawClass::Fur);
    assert_eq!(
        with(FLAG_FUR | FLAG_TRANSPARENT).class(),
        DrawClass::Transparent
    );
}

#[test]
fn diffuse_transmission_defaults_off_and_sets_its_flag_only_when_positive() {
    let d = SurfaceMaterial::default();
    assert_eq!((d.diffuse_transmission, d.diffuse_transmission_color), (0.0, [1.0; 3]));
    let mut out = Vec::new();
    d.write_gpu(&mut out);
    // Off: transmission.w 0, no flag; the colour word of white is 0xFFFFFF.
    assert_eq!(f32_at(&out, 76), 0.0);
    assert_eq!(f32_at(&out, 28), 16_777_215.0);
    assert_eq!(u32_at(&out, 104) & FLAG_DIFFUSE_TRANSMISSION, 0);
    // A caller-set bit without a weight is dropped; a weight sets it whatever the caller flags.
    let stale = SurfaceMaterial {
        flags: FLAG_DIFFUSE_TRANSMISSION | FLAG_DOUBLE_SIDED,
        ..SurfaceMaterial::default()
    };
    assert_eq!(stale.gpu_flags(), FLAG_DOUBLE_SIDED);
    let on = SurfaceMaterial {
        diffuse_transmission: 0.8,
        diffuse_transmission_color: [1.0, 0.6, 0.2],
        ..SurfaceMaterial::default()
    };
    assert_eq!(on.gpu_flags(), FLAG_DIFFUSE_TRANSMISSION);
    assert!(on.validate().is_ok());
    // Still the opaque class: drawn by the prepass and forward, never by OIT.
    assert_eq!(on.class(), DrawClass::Opaque);
    let mut out = Vec::new();
    on.write_gpu(&mut out);
    assert_eq!(f32_at(&out, 76), 0.8);
    assert_eq!(u32_at(&out, 104), FLAG_DIFFUSE_TRANSMISSION);
    // 0.6 x 255 = 153, 0.2 x 255 = 51: the WGSL decode u32(w) & 255, >> 8, >> 16.
    let word = f32_at(&out, 28) as u32;
    assert_eq!(f32_at(&out, 28), word as f32, "exact integer float");
    assert_eq!([word & 255, (word >> 8) & 255, (word >> 16) & 255], [255, 153, 51]);
}
