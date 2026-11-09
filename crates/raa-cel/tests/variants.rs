//! Device-free shader ABI and full material/pass variant validation.
#[allow(dead_code)]
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;
#[path = "../src/shaders.rs"]
mod shaders;
use raa_shader::{Composer, MaterialFnRegistry, PassKind, SlotId, VariantKey};

#[test]
fn resolve_validates_and_matches_uniform_abi() {
    Composer::validate(shaders::CEL_WGSL).unwrap();
    let module = naga::front::wgsl::parse_str(shaders::CEL_WGSL).unwrap();
    let entries: Vec<_> = module
        .entry_points
        .iter()
        .map(|e| e.name.as_str())
        .collect();
    assert_eq!(entries, [shaders::VS_MAIN, shaders::FS_CEL]);
    let mut bindings: Vec<_> = module
        .global_variables
        .iter()
        .filter_map(|(_, v)| v.binding.as_ref().map(|b| (b.group, b.binding)))
        .collect();
    bindings.sort();
    assert_eq!(bindings, [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]);
    let ty = module
        .types
        .iter()
        .find(|(_, t)| t.name.as_deref() == Some("CelPost"))
        .unwrap()
        .1;
    let naga::TypeInner::Struct { members, span } = &ty.inner else {
        panic!("CelPost struct")
    };
    assert_eq!(u64::from(*span), shaders::PARAMS_SIZE);
    assert_eq!(
        members.iter().map(|m| m.offset).collect::<Vec<_>>(),
        [0, 16, 32, 48, 64, 80, 96]
    );
    assert_eq!(
        members
            .iter()
            .map(|m| m.name.as_deref().unwrap())
            .collect::<Vec<_>>(),
        [
            "settings", "palette", "toon", "ink", "post", "image", "grid"
        ]
    );
}

#[test]
fn every_material_and_pass_validates_at_msaa_one_and_four() {
    let mut c = Composer::new();
    for p in lit_slots::default_slot_providers()
        .into_iter()
        .chain(raa_brdf::providers())
        .chain([raa_cel::provider()])
    {
        c.add_slot(p).unwrap();
    }
    let winner = c.winner(SlotId::Direct).unwrap();
    assert_eq!((winner.provider, winner.priority), ("cel", 30));
    let mut reg = MaterialFnRegistry::new();
    raa_matfn::register_builtins(&mut reg).unwrap();
    for (name, flags) in [("raa.standard", 0)]
        .into_iter()
        .chain(raa_matfn::BUILTINS.iter().map(|b| (b.name, b.flags)))
    {
        for pass in PassKind::ALL {
            for msaa in [1, 4] {
                for extra in [
                    0,
                    raa_cel::FLAG_CEL,
                    raa_cel::FLAG_CEL | raa_lit::FLAG_DOUBLE_SIDED,
                ] {
                    let key = VariantKey {
                        material_fn: reg.id_of(name).unwrap(),
                        pass,
                        msaa,
                        flags: flags | extra,
                    };
                    let source = c.compose(&reg, &key).unwrap();
                    if name == "raa.standard" && pass == PassKind::Forward {
                        // One Direct definition and exactly the shared loop's call per light.
                        assert_eq!(source.matches("fn raa_slot_direct(").count(), 1);
                        assert_eq!(
                            source
                                .matches("raa_slot_direct(s, c, direction, radiance)")
                                .count(),
                            1
                        );
                    }
                    Composer::validate(&source).unwrap_or_else(|e| panic!("{name} {key:?}: {e}"));
                }
            }
        }
    }
}

#[test]
fn particle_modules_accept_the_cel_provider() {
    let mut composer = Composer::new();
    for p in lit_slots::default_slot_providers()
        .into_iter()
        .chain(raa_brdf::providers())
        .chain([raa_cel::provider()])
    {
        composer.add_slot(p).unwrap();
    }
    // Particles read MSAA from FrameGpu; their shader source is shared at 1 and 4.
    for module in [
        raa_particles::opaque_module_wgsl,
        raa_particles::blend_module_wgsl,
    ] {
        Composer::validate(&module(&composer).unwrap()).unwrap();
    }
}

#[test]
fn direct_is_priority_30_at_flag_bit_7() {
    let provider = raa_cel::provider();
    assert_eq!(
        (provider.slot, provider.priority, provider.provider),
        (SlotId::Direct, 30, "cel")
    );
    assert_eq!(raa_cel::FLAG_CEL, 1 << 7);
    assert_eq!(provider.wgsl.matches("fn raa_cel_fallback(").count(), 1);
    assert_eq!(provider.wgsl.matches("fn raa_slot_direct(").count(), 1);
    for token in ["textureSample(", "dpdx(", "dpdy(", "fwidth("] {
        assert!(!provider.wgsl.contains(token), "{token}");
    }
}
