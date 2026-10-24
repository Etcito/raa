//! Q11: every liquid built-in material function composes and passes naga validation for
//! every pass kind at MSAA 1 and 4, over raa-lit's default slot providers, raa-brdf's
//! `layered` providers and raa-transmission's `screen` Transmission provider.

// The default providers are crate-private; the test compiles their source file (the crate
// is only read).
#[allow(dead_code)]
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;

use std::borrow::Cow;

use raa_liquid::LIQUID_BUILTINS;
use raa_lit::{FLAG_TRANSMISSIVE, FLAG_TRANSPARENT};
use raa_shader::{Composer, MaterialFnRegistry, PassKind, SlotId, SlotProvider, VariantKey};
use raa_transmission::SCREEN_PROVIDER;

/// The `screen` provider as `TransmissionFeature::install` adds it (the constructor is
/// crate-private): the same WGSL source at priority 10.
fn screen_provider() -> SlotProvider {
    SlotProvider {
        slot: SlotId::Transmission,
        priority: 10,
        provider: SCREEN_PROVIDER,
        wgsl: Cow::Borrowed(include_str!(
            "../../raa-transmission/src/wgsl/transmission.wgsl"
        )),
    }
}

/// Default + layered + screen: the slot set of an engine with the liquid feature.
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
    for b in LIQUID_BUILTINS {
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
fn every_liquid_builtin_validates_in_every_pass_and_msaa() {
    let composer = composer();
    let reg = registry();
    let passes = [
        PassKind::Prepass,
        PassKind::Forward,
        PassKind::Shadow,
        PassKind::Transparent,
        PassKind::FurShell,
    ];
    let mut count = 0;
    for b in LIQUID_BUILTINS {
        let material_fn = reg.id_of(b.name).expect("registered");
        for pass in passes {
            let flags = if pass == PassKind::Transparent {
                b.flags | FLAG_TRANSPARENT | FLAG_TRANSMISSIVE
            } else {
                b.flags
            };
            for msaa in [1, 4] {
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
    }
    // 3 built-ins x 5 pass kinds x 2 MSAA settings.
    assert_eq!(LIQUID_BUILTINS.len(), 3);
    assert_eq!(count, 30);
}
