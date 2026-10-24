//! Q11: the OIT clear and resolve shaders and the Transparent pass variant of the standard
//! material function pass naga validation.

// `default_slot_providers` is crate-private to raa-lit (used by `LitRenderer::new`); the test
// compiles the same source file, as raa-lit's own variants test does.
#[allow(dead_code)]
#[path = "../../raa-lit/src/slots.rs"]
mod slots;

use raa_oit::{CLEAR_ENTRY, CLEAR_WGSL, RESOLVE_FS_ENTRY, RESOLVE_VS_ENTRY, RESOLVE_WGSL};
use raa_shader::{Composer, MaterialFnId, MaterialFnRegistry, PassKind, VariantKey};

fn default_composer() -> Composer {
    let mut composer = Composer::new();
    for p in slots::default_slot_providers() {
        if let Err(e) = composer.add_slot(p) {
            panic!("{e}");
        }
    }
    composer
}

#[test]
fn clear_shader_validates() {
    if let Err(e) = Composer::validate(CLEAR_WGSL) {
        panic!("clear.wgsl: {e}");
    }
    assert!(CLEAR_WGSL.contains("@workgroup_size(64)"));
    assert!(CLEAR_WGSL.contains(&format!("fn {CLEAR_ENTRY}(")));
}

#[test]
fn resolve_shader_validates() {
    if let Err(e) = Composer::validate(RESOLVE_WGSL) {
        panic!("resolve.wgsl: {e}");
    }
    assert!(RESOLVE_WGSL.contains(&format!("fn {RESOLVE_VS_ENTRY}(")));
    assert!(RESOLVE_WGSL.contains(&format!("fn {RESOLVE_FS_ENTRY}(")));
}

#[test]
fn transparent_standard_variant_validates() {
    let reg = MaterialFnRegistry::new();
    let composer = default_composer();
    // Plain, DOUBLE_SIDED and UNLIT variants.
    for flags in [0, 8, 64] {
        let key = VariantKey {
            material_fn: MaterialFnId::STANDARD,
            pass: PassKind::Transparent,
            msaa: 1,
            flags,
        };
        let wgsl = composer
            .compose(&reg, &key)
            .unwrap_or_else(|e| panic!("{key:?}: {e}"));
        assert!(wgsl.contains("raa_oit_heads"), "{key:?}");
        if let Err(e) = Composer::validate(&wgsl) {
            panic!("{key:?}: {e}");
        }
    }
}
