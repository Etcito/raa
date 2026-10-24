//! Q11: the three particle modules — the standalone `particles.sim` compute module and the
//! `particles.opaque` and `particles.blend` draw modules — compose and pass naga validation
//! over the slot set of an engine with the M4..M5 features: raa-lit's default providers,
//! raa-brdf's `layered` providers and raa-transmission's `screen` Transmission provider.

// The default providers are crate-private; the test compiles their source file (the crate
// is only read).
#[allow(dead_code)]
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;

use std::borrow::Cow;

use raa_particles::{blend_module_wgsl, opaque_module_wgsl, sim_module_wgsl};
use raa_shader::{Composer, SlotId, SlotProvider};
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
fn sim_module_validates_with_the_engine_slot_set() {
    let _composer = composer();
    let wgsl = sim_module_wgsl();
    assert!(wgsl.contains("fn cs_sim("));
    assert!(wgsl.contains("@workgroup_size(64)"));
    if let Err(e) = Composer::validate(wgsl) {
        panic!("particles.sim: {e}");
    }
}

#[test]
fn opaque_module_validates_with_the_engine_slot_set() {
    let composer = composer();
    let wgsl = opaque_module_wgsl(&composer).unwrap_or_else(|e| panic!("{e}"));
    for slot in SlotId::ALL {
        assert!(
            wgsl.contains(&format!("fn {}(", slot.fn_name())),
            "{slot:?}"
        );
    }
    for entry in ["fn vs_main(", "fn fs_main("] {
        assert!(wgsl.contains(entry), "{entry}");
    }
    if let Err(e) = Composer::validate(&wgsl) {
        panic!("particles.opaque: {e}");
    }
}

#[test]
fn blend_module_validates_with_the_engine_slot_set() {
    let composer = composer();
    let wgsl = blend_module_wgsl(&composer).unwrap_or_else(|e| panic!("{e}"));
    for slot in SlotId::ALL {
        assert!(
            wgsl.contains(&format!("fn {}(", slot.fn_name())),
            "{slot:?}"
        );
    }
    for entry in [
        "fn vs_splash(",
        "fn fs_splash(",
        "fn vs_sparkles(",
        "fn fs_sparkles(",
    ] {
        assert!(wgsl.contains(entry), "{entry}");
    }
    if let Err(e) = Composer::validate(&wgsl) {
        panic!("particles.blend: {e}");
    }
}

#[test]
fn draw_modules_refuse_a_missing_slot() {
    let mut without_fog = Composer::new();
    for p in lit_slots::default_slot_providers()
        .into_iter()
        .chain(raa_brdf::providers())
        .chain([screen_provider()])
    {
        if p.slot != SlotId::Fog {
            without_fog.add_slot(p).unwrap_or_else(|e| panic!("{e}"));
        }
    }
    for module in [opaque_module_wgsl, blend_module_wgsl] {
        let err = module(&Composer::new()).expect_err("no providers");
        assert_eq!(err.code(), "shader.slot_missing");
        let err = module(&without_fog).expect_err("missing Fog");
        assert_eq!(err.code(), "shader.slot_missing");
        assert!(err.message().contains("Fog"), "{}", err.message());
    }
}
