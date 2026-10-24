//! Q11: every transmissive built-in material function composes and passes naga validation
//! for the Transparent pass at MSAA 1 and the Forward pass at MSAA 1 and 4, over raa-lit's
//! default slot providers, raa-brdf's `layered` providers and the `screen` Transmission
//! provider; the standalone `opaque.copy` shader validates.

// The default providers are crate-private; the test compiles their source file (the crate
// is only read).
#[allow(dead_code)]
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;

use std::borrow::Cow;

use raa_lit::{FLAG_TRANSMISSIVE, FLAG_TRANSPARENT};
use raa_shader::{Composer, MaterialFnRegistry, PassKind, SlotId, SlotProvider, VariantKey};
use raa_transmission::{SCREEN_PROVIDER, TRANSMISSIVE_BUILTINS};

/// The `screen` provider as `TransmissionFeature::install` adds it (the constructor is
/// crate-private): the same WGSL source at priority 10.
fn screen_provider() -> SlotProvider {
    SlotProvider {
        slot: SlotId::Transmission,
        priority: 10,
        provider: SCREEN_PROVIDER,
        wgsl: Cow::Borrowed(include_str!("../src/wgsl/transmission.wgsl")),
    }
}

/// Default + layered + screen: the slot set of an engine with the M4 and M5 features.
fn composer() -> Composer {
    let mut composer = Composer::new();
    for p in lit_slots::default_slot_providers()
        .into_iter()
        .chain(raa_brdf::providers())
        .chain([screen_provider()])
    {
        if let Err(e) = composer.add_slot(p) {
            panic!("{e}");
        }
    }
    composer
}

fn registry() -> MaterialFnRegistry {
    let mut reg = MaterialFnRegistry::new();
    for b in TRANSMISSIVE_BUILTINS {
        if let Err(e) = reg.register_builtin(b.name, b.wgsl) {
            panic!("{}: {e}", b.name);
        }
    }
    reg
}

#[test]
fn installed_providers_win_their_slots() {
    let composer = composer();
    for slot in SlotId::ALL {
        let w = composer.winner(slot).expect("winner");
        let expected = match slot {
            SlotId::Direct | SlotId::Ambient => "layered",
            SlotId::Transmission => SCREEN_PROVIDER,
            _ => "raa.default",
        };
        assert_eq!(w.provider, expected, "{slot:?}");
    }
}

#[test]
fn every_transmissive_builtin_validates_forward_and_transparent() {
    let composer = composer();
    let reg = registry();
    let mut count = 0;
    for b in TRANSMISSIVE_BUILTINS {
        let material_fn = reg.id_of(b.name).expect("registered");
        let transparent = b.flags | FLAG_TRANSPARENT | FLAG_TRANSMISSIVE;
        for (pass, msaa, flags) in [
            (PassKind::Transparent, 1, transparent),
            (PassKind::Forward, 1, b.flags),
            (PassKind::Forward, 4, b.flags),
        ] {
            let key = VariantKey {
                material_fn,
                pass,
                msaa,
                flags,
            };
            let wgsl = composer
                .compose(&reg, &key)
                .unwrap_or_else(|e| panic!("{} {key:?}: {e}", b.name));
            if pass == PassKind::Transparent {
                assert!(wgsl.contains("fn raa_screen_refract("), "{key:?}");
            }
            if let Err(e) = Composer::validate(&wgsl) {
                panic!("{} {key:?}: {e}", b.name);
            }
            count += 1;
        }
    }
    assert_eq!(TRANSMISSIVE_BUILTINS.len(), 4);
    assert_eq!(count, TRANSMISSIVE_BUILTINS.len() * 3);
}

#[test]
fn copy_shader_validates() {
    if let Err(e) = Composer::validate(include_str!("../src/wgsl/copy.wgsl")) {
        panic!("{e}");
    }
}
