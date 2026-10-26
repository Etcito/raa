//! Q11: the `height` Fog slot provider, over raa-lit's default providers, composes and passes
//! naga validation with the standard material function for every pass kind at MSAA 1 and 4;
//! the composed fog pass validates; the provider defines only `raa_slot_fog` and
//! `raa_height_*` helpers.

// The default providers are crate-private; the test compiles their source file (the crate
// is only read).
#[allow(dead_code)]
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;

// shaders.rs is compiled here directly (it uses no `crate::` paths).
#[allow(dead_code)]
#[path = "../src/shaders.rs"]
mod shaders;

use raa_shader::{Composer, MaterialFnId, MaterialFnRegistry, PassKind, SlotId, VariantKey};

/// Defaults + the `height` provider: the slot set of an engine with the fog feature.
fn composer() -> Composer {
    let mut composer = Composer::new();
    for p in lit_slots::default_slot_providers()
        .into_iter()
        .chain([shaders::slot_provider()])
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
        .filter_map(|l| l.trim_start().strip_prefix("fn "))
        .filter_map(|l| l.split('(').next())
        .map(str::trim)
        .collect()
}

#[test]
fn provider_is_height_priority_10_and_wins_fog() {
    let p = shaders::slot_provider();
    assert_eq!(p.slot, SlotId::Fog);
    assert_eq!(p.priority, 10);
    assert_eq!(p.provider, "height");
    assert_eq!(p.wgsl, shaders::FOG_SLOT_WGSL);
    let composer = composer();
    for slot in SlotId::ALL {
        let w = composer.winner(slot).expect("winner");
        let expected = if slot == SlotId::Fog {
            "height"
        } else {
            "raa.default"
        };
        assert_eq!(w.provider, expected, "{slot:?}");
    }
}

#[test]
fn provider_defines_only_slot_fn_and_prefixed_helpers() {
    let fns = defined_fns(shaders::FOG_SLOT_WGSL);
    assert_eq!(
        fns.iter().filter(|f| **f == "raa_slot_fog").count(),
        1,
        "{fns:?}"
    );
    assert!(fns.len() > 1, "{fns:?}");
    for f in fns {
        assert!(
            f == "raa_slot_fog" || f.starts_with("raa_height_"),
            "helper `{f}` is not prefixed raa_height_"
        );
    }
    for line in shaders::FOG_SLOT_WGSL.lines() {
        if let Some(rest) = line.trim_start().strip_prefix("const ") {
            assert!(rest.starts_with("raa_height_"), "{line}");
        }
    }
}

#[test]
fn provider_uses_only_prelude_names() {
    let src = shaders::FOG_SLOT_WGSL;
    for forbidden in ["@group", "@binding", "@vertex", "@fragment", "var<"] {
        assert!(!src.contains(forbidden), "{forbidden}");
    }
    for name in [
        "raa_frame",
        "raa_env",
        "raa_env_specular",
        "raa_linear_clamp",
        "raa_env_sh",
    ] {
        assert!(src.contains(name), "{name}");
    }
}

#[test]
fn standard_fn_every_pass_and_msaa_validates() {
    let reg = MaterialFnRegistry::new();
    let composer = composer();
    let mut count = 0;
    for pass in PassKind::ALL {
        for msaa in [1, 4] {
            let key = VariantKey {
                material_fn: MaterialFnId::STANDARD,
                pass,
                msaa,
                flags: 0,
            };
            let wgsl = composer
                .compose(&reg, &key)
                .unwrap_or_else(|e| panic!("{key:?}: {e}"));
            assert!(wgsl.contains("fn raa_height_apply("), "{key:?}");
            if let Err(e) = Composer::validate(&wgsl) {
                panic!("{key:?}: {e}");
            }
            count += 1;
        }
    }
    assert_eq!(count, PassKind::ALL.len() * 2);
}

#[test]
fn fog_pass_validates() {
    let src = shaders::fog_pass_source();
    assert!(src.starts_with(raa_shader::PRELUDE_WGSL));
    assert!(src.contains(shaders::FOG_SLOT_WGSL));
    assert!(src.ends_with(shaders::FOG_PASS_WGSL));
    for line in [
        "@group(2) @binding(0) var fog_depth: texture_depth_2d;",
        "@group(2) @binding(1) var fog_hdr_src: texture_2d<f32>;",
        "@group(2) @binding(2) var fog_samp: sampler;",
    ] {
        assert!(shaders::FOG_PASS_WGSL.contains(line), "{line}");
    }
    if let Err(e) = Composer::validate(&src) {
        panic!("{e}");
    }
}
