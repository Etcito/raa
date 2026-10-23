//! Q11: every built-in material function composes and passes naga validation for every pass
//! kind, MSAA 1/4 and its own flags, over raa-lit's default slot providers plus the installed
//! providers available to this crate (raa-ibl's `ibl` Ambient provider); every parameter
//! table is well formed.

// The providers are crate-private; the test compiles their source files (both crates are
// only read).
#[allow(dead_code)]
#[path = "../../raa-ibl/src/provider.rs"]
mod ibl_provider;
#[allow(dead_code)]
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;

use raa_matfn::{BUILTINS, ParamKind, register_builtins};
use raa_shader::{Composer, MaterialFnRegistry, PassKind, SlotId, VariantKey};

fn composer() -> Composer {
    let mut composer = Composer::new();
    for p in lit_slots::default_slot_providers()
        .into_iter()
        .chain([ibl_provider::ambient_provider()])
    {
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
fn installed_providers_win_their_slots() {
    let composer = composer();
    for slot in SlotId::ALL {
        let w = composer.winner(slot).expect("winner");
        let expected = if slot == SlotId::Ambient {
            "ibl"
        } else {
            "raa.default"
        };
        assert_eq!(w.provider, expected, "{slot:?}");
    }
}

#[test]
fn every_builtin_validates_for_every_pass_kind() {
    let composer = composer();
    let reg = registry();
    let mut count = 0;
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
                    .unwrap_or_else(|e| panic!("{} {key:?}: {e}", b.name));
                if let Err(e) = Composer::validate(&wgsl) {
                    panic!("{} {key:?}: {e}", b.name);
                }
                count += 1;
            }
        }
        println!("built-in {} validates", b.name);
    }
    println!("{} built-ins, {count} variants", BUILTINS.len());
    assert_eq!(BUILTINS.len(), 7);
    assert_eq!(count, BUILTINS.len() * PassKind::ALL.len() * 2);
}

#[test]
fn param_tables_are_well_formed() {
    for b in BUILTINS {
        for (i, p) in b.params.iter().enumerate() {
            let width = match p.kind {
                ParamKind::Float => 1,
                ParamKind::Vec3 | ParamKind::Color => 3,
                other => other.components(),
            };
            assert_eq!(width, p.kind.components(), "{} {}", b.name, p.name);
            assert!(p.slot.0 < 8, "{} {}: slot index", b.name, p.name);
            assert!(p.slot.1 + width <= 4, "{} {}: component", b.name, p.name);
            for q in &b.params[i + 1..] {
                assert_ne!(p.name, q.name, "{}: duplicate name", b.name);
                assert_ne!(
                    p.slot, q.slot,
                    "{} {} {}: same slot",
                    b.name, p.name, q.name
                );
                let overlap = p.slot.0 == q.slot.0
                    && p.slot.1 < q.slot.1 + q.kind.components()
                    && q.slot.1 < p.slot.1 + width;
                assert!(!overlap, "{} {} {}: overlapping", b.name, p.name, q.name);
            }
        }
    }
}
