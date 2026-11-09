//! S11: every built-in material function composes with the default slot providers, the
//! neutral LocalVisibility/Indirect providers and the shared local-lighting helper for
//! every pass kind at MSAA 1/4. The built-ins that take screen derivatives keep them in
//! their material function, outside the nonuniform light loop.

// The default providers are crate-private; the test compiles the same source file.
#[allow(dead_code)]
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;

use raa_matfn::{BUILTINS, register_builtins};
use raa_shader::{Composer, MaterialFnRegistry, PassKind, SlotId, VariantKey};

fn composer() -> Composer {
    let mut composer = Composer::new();
    for p in lit_slots::default_slot_providers() {
        if let Err(e) = composer.add_slot(p) {
            panic!("{e}");
        }
    }
    composer
}

fn registry() -> MaterialFnRegistry {
    let mut reg = MaterialFnRegistry::new();
    if let Err(e) = register_builtins(&mut reg) {
        panic!("{e}");
    }
    reg
}

#[test]
fn every_builtin_composes_with_neutral_extension_slots() {
    let composer = composer();
    for slot in [SlotId::Direct, SlotId::LocalVisibility, SlotId::Indirect] {
        let winner = composer.winner(slot).expect("neutral");
        assert_eq!(
            (winner.provider, winner.priority),
            ("raa.default", 0),
            "{slot:?}"
        );
    }
    // Derivative use is in these material functions (material_main or its helpers), which
    // stay outside raa_local_contribution and therefore keep uniform evaluation.
    for name in ["raa.inflatable", "raa.candy"] {
        let b = BUILTINS.iter().find(|b| b.name == name).expect(name);
        assert!(b.wgsl.contains("fwidth("), "{name}");
    }

    let reg = registry();
    for b in BUILTINS {
        let material_fn = reg.id_of(b.name).expect("registered");
        for pass in PassKind::ALL {
            for msaa in [1, 4] {
                let key = VariantKey {
                    material_fn,
                    pass,
                    msaa,
                    flags: b.flags,
                };
                let wgsl = composer
                    .compose(&reg, &key)
                    .unwrap_or_else(|e| panic!("{key:?}: {e}"));
                assert!(wgsl.contains("fn raa_local_lighting("), "{key:?}");
                assert_eq!(wgsl.matches("fn raa_slot_direct(").count(), 1, "{key:?}");
                Composer::validate(&wgsl).unwrap_or_else(|e| panic!("{key:?}: {e}"));
            }
        }
    }
}
