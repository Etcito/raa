//! S11: the neutral LocalVisibility/Indirect slots and the shared local-lighting helper are
//! extensible from outside RAA: an external provider pair overrides both neutrals, composes
//! with raa-lit's defaults for every pass kind, posed and unposed, and the lit passes keep
//! the four-group and fragment-storage budgets with the light resources bound.

// The default providers are crate-private; the test compiles the same source file.
#[allow(dead_code)]
#[path = "../../../crates/raa-lit/src/slots.rs"]
mod lit_slots;

use std::borrow::Cow;

use raa_lit::skin_draw;
use raa_shader::{
    Composer, MaterialFnId, MaterialFnRegistry, PassKind, SlotId, SlotProvider, VariantKey,
};

const VISIBILITY: &str = "fn raa_slot_local_visibility(light_index: u32, world_position: vec3<f32>, surface_normal: vec3<f32>, direction_to_light: vec3<f32>, light_distance: f32) -> f32 {
    return clamp(0.5 + 0.5 * dot(surface_normal, direction_to_light), 0.0, 1.0);
}
";
const INDIRECT: &str = "fn raa_slot_indirect(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    return s.base_color * 0.125;
}
";

fn composer() -> Composer {
    let mut composer = Composer::new();
    for p in lit_slots::default_slot_providers() {
        if let Err(e) = composer.add_slot(p) {
            panic!("{e}");
        }
    }
    composer
}

fn external(slot: SlotId, wgsl: &'static str) -> SlotProvider {
    SlotProvider {
        slot,
        priority: 7,
        provider: "ext",
        wgsl: Cow::Borrowed(wgsl),
    }
}

#[test]
fn external_local_visibility_and_indirect_extend_every_composed_variant() {
    let mut composer = composer();
    assert!(!raa_shader::PRELUDE_WGSL.contains("raa_slot_"));
    for (slot, body) in [
        (SlotId::LocalVisibility, "return 1.0;"),
        (SlotId::Indirect, "return vec3<f32>(0.0);"),
    ] {
        let w = composer.winner(slot).expect("neutral");
        assert_eq!((w.provider, w.priority), ("raa.default", 0), "{slot:?}");
        assert!(w.wgsl.contains(body), "{slot:?}");
    }
    let neutral = composer.slots_hash();
    for p in [
        external(SlotId::LocalVisibility, VISIBILITY),
        external(SlotId::Indirect, INDIRECT),
    ] {
        composer.add_slot(p).unwrap();
    }
    assert_ne!(composer.slots_hash(), neutral);
    for slot in [SlotId::LocalVisibility, SlotId::Indirect] {
        assert_eq!(composer.winner(slot).unwrap().provider, "ext", "{slot:?}");
    }

    let reg = MaterialFnRegistry::new();
    for pass in PassKind::ALL {
        for msaa in [1, 4] {
            let key = VariantKey {
                material_fn: MaterialFnId::STANDARD,
                pass,
                msaa,
                flags: 0,
            };
            let wgsl = composer.compose(&reg, &key).expect("compose");
            assert!(wgsl.contains("@vertex\nfn vs_main("), "{pass:?}");
            assert_eq!(
                wgsl.contains("@fragment\nfn fs_main("),
                pass != PassKind::Shadow,
                "{pass:?}"
            );
            assert_eq!(Composer::validate(&wgsl), Ok(()));
            if pass == PassKind::Forward && msaa == 1 {
                assert!(wgsl.contains("0.5 * dot(surface_normal, direction_to_light)"));
                assert!(wgsl.contains("s.base_color * 0.125"));
                assert!(wgsl.contains("fn raa_local_lighting("));
            }
            let posed = skin_draw::posed_module(&wgsl).expect("posed entry points");
            assert_eq!(Composer::validate(&posed), Ok(()));
            if matches!(
                pass,
                PassKind::Forward | PassKind::Transparent | PassKind::FurShell
            ) {
                check_budgets(&wgsl);
                check_budgets(&posed);
            }
        }
    }
}

// Count declared fragment storage conservatively, excluding only vertex-only resources:
// four bind groups maximum, at most eight fragment storage buffers with the lights enabled,
// and neither group-1 light binding a storage buffer.
fn check_budgets(source: &str) {
    use raa_core::wgpu::naga::{AddressSpace, front::wgsl};
    assert!(source.contains("@group(1) @binding(12) var<uniform> raa_local_lights"));
    assert!(source.contains("@group(1) @binding(13) var raa_light_tiles: texture_2d_array<u32>;"));
    let module = wgsl::parse_str(source).expect("parse budgets");
    let mut fragment_storage = 0;
    for (_, global) in module.global_variables.iter() {
        let Some(binding) = &global.binding else {
            continue;
        };
        assert!(binding.group < 4, "four bind groups maximum");
        let vertex_only = (binding.group, binding.binding) == (0, 8) || binding.group == 3;
        if matches!(global.space, AddressSpace::Storage { .. }) && !vertex_only {
            fragment_storage += 1;
        }
    }
    assert!(
        fragment_storage <= 8,
        "fragment storage budget: {fragment_storage}"
    );
}
