//! Q11: the raa-lit default slot providers with the standard material function compose and
//! pass naga validation for every pass kind at MSAA 1 and 4.

// `default_slot_providers` is crate-private (used by `LitRenderer::new`); the test compiles
// the same source file.
#[path = "../src/slots.rs"]
mod slots;

use raa_shader::{Composer, MaterialFnId, MaterialFnRegistry, PassKind, SlotId, VariantKey};

fn default_composer() -> Composer {
    let mut composer = Composer::new();
    for p in slots::default_slot_providers() {
        if let Err(e) = composer.add_slot(p) {
            panic!("{e}");
        }
    }
    composer
}

/// Identifiers of the `fn` items defined by `wgsl`.
fn defined_fns(wgsl: &str) -> Vec<&str> {
    wgsl.lines()
        .filter_map(|l| l.strip_prefix("fn "))
        .filter_map(|l| l.split('(').next())
        .collect()
}

#[test]
fn one_default_provider_per_slot() {
    let providers = slots::default_slot_providers();
    assert_eq!(providers.len(), SlotId::ALL.len());
    for (p, slot) in providers.iter().zip(SlotId::ALL) {
        assert_eq!(p.slot, slot);
        assert_eq!(p.priority, 0);
        assert_eq!(p.provider, "raa.default");
        let fns = defined_fns(&p.wgsl);
        assert_eq!(
            fns.iter().filter(|f| **f == slot.fn_name()).count(),
            1,
            "{slot:?}: {fns:?}"
        );
        for f in fns {
            assert!(
                f == slot.fn_name() || f.starts_with("raa_default_"),
                "{slot:?}: helper `{f}` is not prefixed raa_default_"
            );
        }
    }
    let composer = default_composer();
    for slot in SlotId::ALL {
        let w = composer.winner(slot).expect("winner");
        assert_eq!((w.slot, w.provider), (slot, "raa.default"));
    }
}

#[test]
fn sections_cover_the_whole_file() {
    let joined: String = slots::default_slot_providers()
        .iter()
        .map(|p| p.wgsl.as_ref())
        .collect();
    assert!(slots::SLOTS_DEFAULT_WGSL.ends_with(&joined));
    for slot in SlotId::ALL {
        let marker = format!("// @slot {}\n", slots::section_name(slot));
        assert_eq!(
            slots::SLOTS_DEFAULT_WGSL.matches(&marker).count(),
            1,
            "{slot:?}"
        );
    }
}

#[test]
fn defaults_follow_s11() {
    let wgsl = |slot: SlotId| {
        slots::default_slot_providers()
            .into_iter()
            .find(|p| p.slot == slot)
            .expect("provider")
            .wgsl
    };
    assert!(wgsl(SlotId::Deform).contains("return VertexMaterialOutput(p, n);"));
    assert!(wgsl(SlotId::Shadow).contains("return 1.0;"));
    assert!(wgsl(SlotId::Ao).contains("return 1.0;"));
    assert!(
        wgsl(SlotId::Ambient)
            .contains("raa_frame.ambient.rgb * s.base_color * (1.0 - 0.5 * s.metallic)")
    );
    let direct = wgsl(SlotId::Direct);
    assert!(direct.contains("mix(vec3<f32>(0.04), s.base_color, s.metallic)"));
    assert!(direct.contains("pow(1.0 - v_dot_h, 5.0)"));
    assert!(direct.contains("(roughness + 1.0) * (roughness + 1.0) / 8.0"));
    assert!(direct.contains("(1.0 - s.metallic)"));
    assert!(wgsl(SlotId::Transmission).contains("return vec3<f32>(0.0);"));
    assert!(wgsl(SlotId::Fog).contains("return color;"));
}

#[test]
fn standard_fn_every_pass_and_msaa_validates() {
    let reg = MaterialFnRegistry::new();
    let composer = default_composer();
    let mut count = 0;
    for pass in PassKind::ALL {
        for msaa in [1, 4] {
            let key = VariantKey {
                material_fn: MaterialFnId::STANDARD,
                pass,
                msaa,
                flags: 0,
            };
            let wgsl = composer
                .compose(&reg, &key)
                .unwrap_or_else(|e| panic!("{key:?}: {e}"));
            assert!(wgsl.contains("provider \"raa.default\""), "{key:?}");
            if let Err(e) = Composer::validate(&wgsl) {
                panic!("{key:?}: {e}");
            }
            count += 1;
        }
    }
    assert_eq!(count, PassKind::ALL.len() * 2);
}

#[test]
fn flagged_variants_validate() {
    let reg = MaterialFnRegistry::new();
    let composer = default_composer();
    // DOUBLE_SIDED, UNLIT, and DOUBLE_SIDED | VERTEX_FN | UNLIT.
    for pass in PassKind::ALL {
        for flags in [8, 64, 8 | 32 | 64] {
            let key = VariantKey {
                material_fn: MaterialFnId::STANDARD,
                pass,
                msaa: 4,
                flags,
            };
            let wgsl = composer
                .compose(&reg, &key)
                .unwrap_or_else(|e| panic!("{key:?}: {e}"));
            if let Err(e) = Composer::validate(&wgsl) {
                panic!("{key:?}: {e}");
            }
        }
    }
}
