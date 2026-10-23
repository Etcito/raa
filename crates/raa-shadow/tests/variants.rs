//! Q11: the `csm` Shadow provider over raa-lit's default providers composes and passes
//! naga validation with the standard material function for every pass kind, MSAA and flag
//! set.

// Both providers are crate-private; the test compiles their source files (raa-lit is only
// read).
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;
#[path = "../src/provider.rs"]
mod provider;

use raa_shader::{Composer, MaterialFnId, MaterialFnRegistry, PassKind, SlotId, VariantKey};

fn csm_composer() -> Composer {
    let mut composer = Composer::new();
    for p in lit_slots::default_slot_providers()
        .into_iter()
        .chain([provider::csm_provider()])
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
    assert!(wgsl.contains("fn raa_slot_shadow("), "{key:?}");
    assert!(wgsl.contains("raa_csm_"), "{key:?}");
    if let Err(e) = Composer::validate(&wgsl) {
        panic!("{key:?}: {e}");
    }
}

#[test]
fn csm_wins_shadow() {
    let composer = csm_composer();
    let w = composer.winner(SlotId::Shadow).expect("winner");
    assert_eq!((w.provider, w.priority), ("csm", 10));
    for slot in SlotId::ALL.into_iter().filter(|s| *s != SlotId::Shadow) {
        assert_eq!(
            composer.winner(slot).expect("winner").provider,
            "raa.default"
        );
    }
}

#[test]
fn provider_defines_the_slot_and_prefixed_helpers() {
    let p = provider::csm_provider();
    assert_eq!(p.slot, SlotId::Shadow);
    assert_eq!(p.priority, 10);
    assert_eq!(p.provider, "csm");
    let fns = defined_fns(&p.wgsl);
    assert_eq!(
        fns.iter().filter(|f| **f == "raa_slot_shadow").count(),
        1,
        "{fns:?}"
    );
    for f in fns {
        assert!(
            f == "raa_slot_shadow" || f.starts_with("raa_csm_"),
            "helper `{f}` is not prefixed raa_csm_"
        );
    }
    // The slot reads the contact-shadow texture.
    assert!(p.wgsl.contains("raa_contact_tex"));
}

#[test]
fn standard_fn_every_pass_and_msaa_validates() {
    let composer = csm_composer();
    let mut count = 0;
    for pass in PassKind::ALL {
        for msaa in [1, 4] {
            for flags in [0, 8, 64, 8 | 32 | 64] {
                let key = VariantKey {
                    material_fn: MaterialFnId::STANDARD,
                    pass,
                    msaa,
                    flags,
                };
                compose_and_validate(&composer, key);
                count += 1;
            }
        }
    }
    assert_eq!(count, PassKind::ALL.len() * 2 * 4);
}
