//! Q11: the `ibl` Ambient provider over raa-lit's default providers composes and passes naga
//! validation with the standard material function for every pass kind, MSAA and flag set.

// Both providers are crate-private; the test compiles their source files (raa-lit is only
// read).
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;
#[path = "../src/provider.rs"]
mod provider;

use raa_shader::{Composer, MaterialFnId, MaterialFnRegistry, PassKind, SlotId, VariantKey};

fn ibl_composer() -> Composer {
    let mut composer = Composer::new();
    for p in lit_slots::default_slot_providers()
        .into_iter()
        .chain([provider::ambient_provider()])
    {
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

fn compose_and_validate(composer: &Composer, key: VariantKey) {
    let reg = MaterialFnRegistry::new();
    let wgsl = composer
        .compose(&reg, &key)
        .unwrap_or_else(|e| panic!("{key:?}: {e}"));
    assert!(wgsl.contains("fn raa_ibl_irradiance("), "{key:?}");
    if let Err(e) = Composer::validate(&wgsl) {
        panic!("{key:?}: {e}");
    }
}

#[test]
fn ibl_wins_ambient() {
    let composer = ibl_composer();
    let w = composer.winner(SlotId::Ambient).expect("winner");
    assert_eq!((w.provider, w.priority), ("ibl", 10));
    for slot in SlotId::ALL.into_iter().filter(|s| *s != SlotId::Ambient) {
        assert_eq!(
            composer.winner(slot).expect("winner").provider,
            "raa.default"
        );
    }
}

#[test]
fn ibl_lib_defines_the_contract_functions() {
    let fns = defined_fns(raa_ibl::IBL_LIB_WGSL);
    for name in [
        "raa_ibl_irradiance",
        "raa_ibl_specular",
        "raa_ibl_dfg",
        "raa_specular_occlusion",
    ] {
        assert_eq!(
            fns.iter().filter(|f| **f == name).count(),
            1,
            "{name}: {fns:?}"
        );
    }
    for f in fns {
        assert!(
            f.starts_with("raa_ibl_") || f == "raa_specular_occlusion",
            "{f}"
        );
    }
}

#[test]
fn provider_defines_the_slot_and_prefixed_helpers() {
    let p = provider::ambient_provider();
    assert_eq!(p.slot, SlotId::Ambient);
    assert!(p.wgsl.starts_with(raa_ibl::IBL_LIB_WGSL));
    let fns = defined_fns(&p.wgsl);
    assert_eq!(fns.iter().filter(|f| **f == "raa_slot_ambient").count(), 1);
    for f in fns {
        assert!(
            f == "raa_slot_ambient" || f == "raa_specular_occlusion" || f.starts_with("raa_ibl_"),
            "helper `{f}` is not prefixed raa_ibl_"
        );
    }
    // Without an environment the S1 PbrLite ambient term is returned.
    assert!(p.wgsl.contains("raa_env_sh.info.w < 0.5"));
    assert!(
        p.wgsl
            .contains("raa_frame.ambient.rgb * s.base_color * (1.0 - 0.5 * s.metallic)")
    );
}

#[test]
fn standard_fn_every_pass_and_msaa_validates() {
    let composer = ibl_composer();
    let mut count = 0;
    for pass in PassKind::ALL {
        for msaa in [1, 4] {
            let key = VariantKey {
                material_fn: MaterialFnId::STANDARD,
                pass,
                msaa,
                flags: 0,
            };
            compose_and_validate(&composer, key);
            count += 1;
        }
    }
    assert_eq!(count, PassKind::ALL.len() * 2);
}

#[test]
fn flagged_variants_validate() {
    let composer = ibl_composer();
    // DOUBLE_SIDED, UNLIT, and DOUBLE_SIDED | VERTEX_FN | UNLIT.
    for pass in PassKind::ALL {
        for msaa in [1, 4] {
            for flags in [8, 64, 8 | 32 | 64] {
                let key = VariantKey {
                    material_fn: MaterialFnId::STANDARD,
                    pass,
                    msaa,
                    flags,
                };
                compose_and_validate(&composer, key);
            }
        }
    }
}
