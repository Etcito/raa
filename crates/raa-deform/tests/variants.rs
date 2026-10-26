//! The `squash-jiggle` `Deform` provider (S11): it wins `Deform` at priority 10 while every
//! other slot keeps the raa-lit default; its source defines the slot function plus
//! `raa_squash_jiggle_` helpers only and binds no group; the standard material function
//! composes and validates with it for every pass kind, MSAA and flag set; and a second
//! install is refused.

// The default slot source module is crate-private; the test compiles its file.
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;

use raa_shader::{Composer, MaterialFnId, MaterialFnRegistry, PassKind, SlotId, VariantKey};

/// The raa-lit default providers plus the squash-jiggle `Deform` provider.
fn squash_jiggle_composer() -> Composer {
    let mut composer = Composer::new();
    for p in lit_slots::default_slot_providers()
        .into_iter()
        .chain([raa_deform::provider()])
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
        .map(str::trim)
        .collect()
}

#[test]
fn provider_is_squash_jiggle_deform_at_priority_10() {
    let p = raa_deform::provider();
    assert_eq!(p.slot, SlotId::Deform);
    assert_eq!((p.provider, p.priority), ("squash-jiggle", 10));
}

#[test]
fn squash_jiggle_wins_deform_and_other_slots_stay_default() {
    let composer = squash_jiggle_composer();
    let w = composer.winner(SlotId::Deform).expect("winner");
    assert_eq!((w.provider, w.priority), ("squash-jiggle", 10));
    for slot in SlotId::ALL.into_iter().filter(|s| *s != SlotId::Deform) {
        assert_eq!(
            composer.winner(slot).expect("winner").provider,
            "raa.default",
            "{slot:?}"
        );
    }
}

#[test]
fn provider_source_defines_only_the_slot_and_prefixed_helpers() {
    let p = raa_deform::provider();
    assert!(
        p.wgsl.contains(
            "fn raa_slot_deform(p: vec3<f32>, n: vec3<f32>, x: InstanceExtraGpu, t: f32) -> VertexMaterialOutput"
        ),
        "S11 signature"
    );
    assert!(!p.wgsl.contains("@group"), "@group in provider source");
    assert!(!p.wgsl.contains("@binding"), "@binding in provider source");
    let fns = defined_fns(&p.wgsl);
    assert_eq!(
        fns.iter().filter(|n| **n == "raa_slot_deform").count(),
        1,
        "{fns:?}"
    );
    for name in fns.iter().filter(|n| **n != "raa_slot_deform") {
        assert!(
            name.starts_with("raa_squash_jiggle_"),
            "unprefixed helper: {name}"
        );
    }
}

fn compose_and_validate(composer: &Composer, key: VariantKey) {
    let reg = MaterialFnRegistry::new();
    let wgsl = composer
        .compose(&reg, &key)
        .unwrap_or_else(|e| panic!("{key:?}: {e}"));
    assert!(wgsl.contains("fn raa_slot_deform("), "{key:?}");
    if let Err(e) = Composer::validate(&wgsl) {
        panic!("{key:?}: {e}");
    }
}

#[test]
fn standard_fn_every_pass_msaa_and_flags_validates() {
    let composer = squash_jiggle_composer();
    let mut count = 0;
    for pass in PassKind::ALL {
        for msaa in [1, 4] {
            for flags in [0, 8, 32, 64, 8 | 32 | 64] {
                compose_and_validate(
                    &composer,
                    VariantKey {
                        material_fn: MaterialFnId::STANDARD,
                        pass,
                        msaa,
                        flags,
                    },
                );
                count += 1;
            }
        }
    }
    assert_eq!(count, PassKind::ALL.len() * 2 * 5);
}

#[test]
fn adding_the_provider_twice_is_refused() {
    let mut composer = Composer::new();
    composer.add_slot(raa_deform::provider()).expect("first");
    let err = composer
        .add_slot(raa_deform::provider())
        .expect_err("second");
    assert_eq!(err.code, "shader.slot_conflict");
}
