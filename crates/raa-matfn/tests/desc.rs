//! SurfaceDesc mapping and every refusal of `map_desc` (CPU only, Q12).

use raa_core::RaaError;
use raa_lit::{
    FLAG_DOUBLE_SIDED, FLAG_FUR, FLAG_NO_SHADOW, FLAG_TRANSPARENT, SurfaceMaterial, SurfaceModel,
};
use raa_matfn::{
    BUILTINS, Blend, DrawClass, MaterialFnRegistry, StandardOverrides, SurfaceDesc, map_desc,
    register_builtins,
};
use raa_shader::{Composer, SlotId, SlotProvider};

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Trivial providers for every slot, enough for the user fn's Forward variant to validate.
fn test_composer() -> Composer {
    let slots = [
        (
            SlotId::Deform,
            "fn raa_slot_deform(p: vec3<f32>, n: vec3<f32>, x: InstanceExtraGpu, t: f32) -> VertexMaterialOutput { return VertexMaterialOutput(p, n); }",
        ),
        (
            SlotId::Shadow,
            "fn raa_slot_shadow(world_pos: vec3<f32>, n: vec3<f32>, frag: vec4<f32>) -> f32 { return 1.0; }",
        ),
        (
            SlotId::Ao,
            "fn raa_slot_ao(frag: vec4<f32>) -> f32 { return 1.0; }",
        ),
        (
            SlotId::Ambient,
            "fn raa_slot_ambient(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> { return s.base_color * 0.1; }",
        ),
        (
            SlotId::Direct,
            "fn raa_slot_direct(s: MaterialOutput, c: ShadeCtx, l: vec3<f32>, radiance: vec3<f32>) -> vec3<f32> { return s.base_color * radiance; }",
        ),
        (
            SlotId::Transmission,
            "fn raa_slot_transmission(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> { return vec3<f32>(0.0); }",
        ),
        (
            SlotId::Fog,
            "fn raa_slot_fog(color: vec3<f32>, world_pos: vec3<f32>, c: ShadeCtx) -> vec3<f32> { return color; }",
        ),
    ];
    let mut composer = Composer::new();
    for (slot, wgsl) in slots {
        composer
            .add_slot(SlotProvider {
                slot,
                priority: 0,
                provider: "test",
                wgsl: wgsl.into(),
            })
            .expect("test slot");
    }
    composer
}

const USER_FN: &str = "fn material_main(in: MaterialInput) -> MaterialOutput {\n    var o = raa_default_output(in);\n    o.base_color = in.record.p[0].rgb;\n    return o;\n}\n";

fn registry() -> MaterialFnRegistry {
    let mut reg = MaterialFnRegistry::new();
    register_builtins(&mut reg).expect("builtins");
    reg.register_user("user.tint", USER_FN, &test_composer())
        .expect("user fn");
    reg
}

fn map(desc: &SurfaceDesc) -> Result<SurfaceMaterial, RaaError> {
    map_desc(desc, &registry(), |_| true)
}

fn inflatable() -> SurfaceDesc {
    SurfaceDesc {
        function: "raa.inflatable".to_string(),
        ..SurfaceDesc::default()
    }
}

fn param(name: &str, v: &[f32]) -> (String, Vec<f32>) {
    (name.to_string(), v.to_vec())
}

/// Asserts the refusal code and that the message is non-empty and names `field`.
fn assert_refused(r: Result<SurfaceMaterial, RaaError>, code: &str, field: &str) {
    let err = r.expect_err("expected a refusal");
    assert_eq!(err.code(), code, "{err}");
    assert!(!err.message().is_empty());
    assert!(
        err.message().contains(field),
        "message `{}` does not name `{field}`",
        err.message()
    );
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-6
}

#[test]
fn standard_desc_maps_to_the_standard_model() {
    let m = map(&SurfaceDesc::default()).expect("standard");
    assert_eq!(m, SurfaceMaterial::default());
    assert_eq!(m.model, SurfaceModel::Standard);
}

#[test]
fn inflatable_defaults_map_to_linear_params() {
    let reg = registry();
    let desc = SurfaceDesc {
        overrides: StandardOverrides {
            clearcoat: Some(1.0),
            ..StandardOverrides::default()
        },
        ..inflatable()
    };
    let m = map_desc(&desc, &reg, |_| true).expect("inflatable");
    let id = reg.id_of("raa.inflatable").expect("registered");
    assert_eq!(m.model, SurfaceModel::Function(id));
    assert_eq!(m.clearcoat, 1.0);
    assert_eq!(m.flags, 0);
    assert_eq!(m.class(), DrawClass::Opaque);
    let row = BUILTINS
        .iter()
        .find(|b| b.name == "raa.inflatable")
        .unwrap();
    for spec in row.params {
        let n = spec.kind.components();
        for i in 0..n {
            let got = m.params[spec.slot.0][spec.slot.1 + i];
            let want = if spec.kind == raa_matfn::ParamKind::Color {
                srgb_to_linear(spec.default[i])
            } else {
                spec.default[i]
            };
            assert!(close(got, want), "{} [{i}]: {got} vs {want}", spec.name);
        }
    }
    // colour default 0.95 sRGB is stored linear, not as sent
    assert!(m.params[0][0] < 0.95 && close(m.params[0][0], srgb_to_linear(0.95)));
    assert_eq!(m.params[2], [0.6, 0.1, 8.0, 0.0]);
}

#[test]
fn inflatable_param_overrides_its_default() {
    let desc = SurfaceDesc {
        params: vec![
            param("seamCount", &[12.0]),
            param("color", &[0.5, 0.0, 1.0]),
        ],
        ..inflatable()
    };
    let m = map(&desc).expect("inflatable");
    assert_eq!(m.params[2], [0.6, 0.1, 12.0, 0.0]);
    assert!(close(m.params[0][0], srgb_to_linear(0.5)));
    assert_eq!(m.params[0][1], 0.0);
    assert!(close(m.params[0][2], 1.0));
}

#[test]
fn user_fn_takes_positional_p0_to_p7() {
    let desc = SurfaceDesc {
        function: "user.tint".to_string(),
        params: vec![
            param("p0", &[0.25, 0.5, 0.75]),
            param("p7", &[1.0, 2.0, 3.0, 4.0]),
            param("p3", &[9.0]),
        ],
        ..SurfaceDesc::default()
    };
    let m = map(&desc).expect("user fn");
    assert!(matches!(m.model, SurfaceModel::Function(_)));
    assert_eq!(m.params[0], [0.25, 0.5, 0.75, 0.0]);
    assert_eq!(m.params[3], [9.0, 0.0, 0.0, 0.0]);
    assert_eq!(m.params[7], [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn fur_fields_set_flag_fur_and_class_fur() {
    let desc = SurfaceDesc {
        overrides: StandardOverrides {
            fur_length: Some(0.02),
            fur_droop: Some(0.3),
            ..StandardOverrides::default()
        },
        ..SurfaceDesc::default()
    };
    let m = map(&desc).expect("fur");
    assert_ne!(m.flags & FLAG_FUR, 0);
    assert_eq!(m.class(), DrawClass::Fur);
    assert_eq!(m.fur, [0.02, 2000.0, 0.3, 32.0]);
    let shells = SurfaceDesc {
        overrides: StandardOverrides {
            fur_shells: Some(64.0),
            ..StandardOverrides::default()
        },
        ..SurfaceDesc::default()
    };
    assert_eq!(map(&shells).expect("shells").fur[3], 64.0);
}

#[test]
fn overrides_convert_colours_and_set_flags() {
    let desc = SurfaceDesc {
        double_sided: true,
        cast_shadow: false,
        overrides: StandardOverrides {
            color: Some([0.5, 0.5, 0.5]),
            sheen_color: Some([1.0, 0.5, 0.0]),
            attenuation_color: Some([0.2, 0.4, 0.6]),
            emissive: Some([4.0, 2.0, 0.0]),
            roughness: Some(0.3),
            ..StandardOverrides::default()
        },
        ..SurfaceDesc::default()
    };
    let m = map(&desc).expect("overrides");
    assert!(close(m.base_color[0], srgb_to_linear(0.5)));
    assert!(close(m.sheen_color[1], srgb_to_linear(0.5)));
    assert!(close(m.attenuation_color[2], srgb_to_linear(0.6)));
    assert_eq!(m.emissive, [4.0, 2.0, 0.0]);
    assert_eq!(m.roughness, 0.3);
    assert_eq!(m.flags, FLAG_DOUBLE_SIDED | FLAG_NO_SHADOW);
}

#[test]
fn transparent_blend_maps_when_handled() {
    let desc = SurfaceDesc {
        blend: Blend::Transparent,
        overrides: StandardOverrides {
            opacity: Some(0.5),
            ..StandardOverrides::default()
        },
        ..SurfaceDesc::default()
    };
    let m = map(&desc).expect("handled");
    assert_ne!(m.flags & FLAG_TRANSPARENT, 0);
    assert_eq!(m.class(), DrawClass::Transparent);
}

#[test]
fn unknown_fn_is_refused() {
    let desc = SurfaceDesc {
        function: "raa.nope".to_string(),
        ..SurfaceDesc::default()
    };
    assert_refused(map(&desc), "material.unknown_fn", "raa.nope");
}

#[test]
fn unknown_param_is_refused() {
    let desc = SurfaceDesc {
        params: vec![param("glitter", &[1.0])],
        ..inflatable()
    };
    assert_refused(map(&desc), "material.unknown_param", "glitter");
    let standard = SurfaceDesc {
        params: vec![param("color", &[1.0, 1.0, 1.0])],
        ..SurfaceDesc::default()
    };
    assert_refused(map(&standard), "material.unknown_param", "color");
    for name in ["p8", "q0", "p", "p01"] {
        let user = SurfaceDesc {
            function: "user.tint".to_string(),
            params: vec![param(name, &[1.0])],
            ..SurfaceDesc::default()
        };
        assert_refused(map(&user), "material.unknown_param", name);
    }
}

#[test]
fn wrong_param_arity_is_refused() {
    let desc = SurfaceDesc {
        params: vec![param("stripeColor", &[1.0, 0.0])],
        ..inflatable()
    };
    assert_refused(map(&desc), "material.param_type", "stripeColor");
    let scalar = SurfaceDesc {
        params: vec![param("seamDepth", &[0.1, 0.2])],
        ..inflatable()
    };
    assert_refused(map(&scalar), "material.param_type", "seamDepth");
    for values in [&[][..], &[1.0, 2.0, 3.0, 4.0, 5.0][..]] {
        let user = SurfaceDesc {
            function: "user.tint".to_string(),
            params: vec![param("p2", values)],
            ..SurfaceDesc::default()
        };
        assert_refused(map(&user), "material.param_type", "p2");
    }
}

#[test]
fn non_finite_param_is_invalid() {
    let desc = SurfaceDesc {
        params: vec![param("stripeWidth", &[f32::NAN])],
        ..inflatable()
    };
    assert_refused(map(&desc), "material.invalid", "stripeWidth");
    let user = SurfaceDesc {
        function: "user.tint".to_string(),
        params: vec![param("p1", &[0.0, f32::INFINITY])],
        ..SurfaceDesc::default()
    };
    assert_refused(map(&user), "material.invalid", "p1");
}

fn with(o: StandardOverrides) -> SurfaceDesc {
    SurfaceDesc {
        overrides: o,
        ..SurfaceDesc::default()
    }
}

#[test]
fn fur_shells_outside_1_to_64_or_fractional_are_invalid() {
    for shells in [0.0, 65.0, 12.5, f32::NAN] {
        let desc = with(StandardOverrides {
            fur_shells: Some(shells),
            ..StandardOverrides::default()
        });
        assert_refused(map(&desc), "material.invalid", "furShells");
    }
    let length = with(StandardOverrides {
        fur_length: Some(-0.01),
        ..StandardOverrides::default()
    });
    assert_refused(map(&length), "material.invalid", "furLength");
}

#[test]
fn validate_refusals_are_kept() {
    let rough = with(StandardOverrides {
        roughness: Some(1.5),
        ..StandardOverrides::default()
    });
    assert_refused(map(&rough), "material.invalid", "roughness");
    let ior = with(StandardOverrides {
        ior: Some(4.0),
        ..StandardOverrides::default()
    });
    assert_refused(map(&ior), "material.invalid", "ior");
    let color = with(StandardOverrides {
        color: Some([f32::NAN, 0.0, 0.0]),
        ..StandardOverrides::default()
    });
    assert_refused(map(&color), "material.invalid", "color");
    let emissive = with(StandardOverrides {
        emissive: Some([-1.0, 0.0, 0.0]),
        ..StandardOverrides::default()
    });
    assert_refused(map(&emissive), "material.invalid", "emissive");
}

#[test]
fn transparent_blend_is_refused_when_unhandled() {
    let reg = registry();
    let desc = SurfaceDesc {
        blend: Blend::Transparent,
        ..SurfaceDesc::default()
    };
    let r = map_desc(&desc, &reg, |c| c == DrawClass::Opaque);
    assert_refused(r, "material.blend_unavailable", "Transparent");
    let transmissive = SurfaceDesc {
        blend: Blend::Transmissive,
        ..SurfaceDesc::default()
    };
    let r = map_desc(&transmissive, &reg, |c| c == DrawClass::Opaque);
    assert_refused(r, "material.blend_unavailable", "Transmissive");
    let fur = with(StandardOverrides {
        fur_length: Some(0.01),
        ..StandardOverrides::default()
    });
    let r = map_desc(&fur, &reg, |c| c == DrawClass::Opaque);
    assert_refused(r, "material.blend_unavailable", "Fur");
}
