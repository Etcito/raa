//! Q11: the `ao.gtao` and denoise compute modules pass naga validation, bind only group 0,
//! define their entry points and do not reference the S10 frame uniform; `GtaoParams` has
//! the contract field order; the `gtao` provider wins the `Ao` slot at priority 10; and the
//! standard material function composes and validates with the raa-lit defaults plus the gtao
//! provider for every pass kind, MSAA and flag set.

// The shader and slot source modules are crate-private; the test compiles their files.
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;
#[path = "../src/shaders.rs"]
mod shaders;

use raa_shader::{Composer, MaterialFnId, MaterialFnRegistry, PassKind, SlotId, VariantKey};

/// The raa-lit default providers plus the gtao `Ao` provider.
fn gtao_composer() -> Composer {
    let mut composer = Composer::new();
    for p in lit_slots::default_slot_providers()
        .into_iter()
        .chain([shaders::ao_provider()])
    {
        if let Err(e) = composer.add_slot(p) {
            panic!("{e}");
        }
    }
    composer
}

/// Lines of `wgsl` without `//` comments.
fn code_lines(wgsl: &str) -> impl Iterator<Item = &str> {
    wgsl.lines()
        .map(|l| l.split("//").next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
}

/// Identifiers of the `fn` items defined by `wgsl`.
fn defined_fns(wgsl: &str) -> Vec<&str> {
    wgsl.lines()
        .filter_map(|l| l.strip_prefix("fn "))
        .filter_map(|l| l.split('(').next())
        .collect()
}

#[test]
fn compute_modules_validate_and_are_standalone() {
    for (name, wgsl, entries, bindings) in [
        ("gtao", shaders::GTAO_WGSL, &["gtao"][..], 4),
        (
            "denoise",
            shaders::DENOISE_WGSL,
            &["denoise_h", "denoise_v"][..],
            5,
        ),
    ] {
        if let Err(e) = Composer::validate(wgsl) {
            panic!("{name}: {e}");
        }
        assert!(!wgsl.contains("raa_frame"), "{name}");
        let mut groups = 0;
        for l in code_lines(wgsl) {
            for (i, _) in l.match_indices("@group(") {
                assert!(l[i..].starts_with("@group(0)"), "{name}: {l}");
                groups += 1;
            }
        }
        assert_eq!(groups, bindings, "{name}");
        let fns = defined_fns(wgsl);
        for entry in entries {
            assert!(fns.contains(entry), "{name}: entry {entry}: {fns:?}");
        }
        assert_eq!(wgsl.matches("@compute").count(), entries.len(), "{name}");
    }
}

#[test]
fn gtao_params_field_order_matches_contract() {
    let body: Vec<(String, String)> = code_lines(shaders::GTAO_WGSL)
        .skip_while(|l| !l.starts_with("struct GtaoParams"))
        .skip(1)
        .take_while(|l| !l.starts_with('}'))
        .filter_map(|l| l.split_once(':'))
        .map(|(n, t)| {
            (
                n.trim().to_string(),
                t.trim().trim_end_matches(',').to_string(),
            )
        })
        .collect();
    let expected = [
        ("proj", "mat4x4<f32>", 64u64),
        ("inv_proj", "mat4x4<f32>", 64),
        ("viewport", "vec4<f32>", 16),
        ("params", "vec4<f32>", 16),
    ];
    let got: Vec<(&str, &str)> = body.iter().map(|(n, t)| (n.as_str(), t.as_str())).collect();
    let want: Vec<(&str, &str)> = expected.iter().map(|(n, t, _)| (*n, *t)).collect();
    assert_eq!(got, want);
    let size: u64 = expected.iter().map(|(_, _, s)| s).sum();
    assert_eq!(size, shaders::PARAMS_SIZE);
}

#[test]
fn gtao_wins_ao() {
    let composer = gtao_composer();
    let w = composer.winner(SlotId::Ao).expect("winner");
    assert_eq!((w.provider, w.priority), ("gtao", 10));
    for slot in SlotId::ALL.into_iter().filter(|s| *s != SlotId::Ao) {
        assert_eq!(
            composer.winner(slot).expect("winner").provider,
            "raa.default"
        );
    }
}

fn compose_and_validate(composer: &Composer, key: VariantKey) {
    let reg = MaterialFnRegistry::new();
    let wgsl = composer
        .compose(&reg, &key)
        .unwrap_or_else(|e| panic!("{key:?}: {e}"));
    assert!(wgsl.contains("fn raa_slot_ao("), "{key:?}");
    if let Err(e) = Composer::validate(&wgsl) {
        panic!("{key:?}: {e}");
    }
}

#[test]
fn standard_fn_every_pass_msaa_and_flags_validates() {
    let composer = gtao_composer();
    let mut count = 0;
    for pass in PassKind::ALL {
        for msaa in [1, 4] {
            for flags in [0, 8, 64, 8 | 32 | 64] {
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
    assert_eq!(count, PassKind::ALL.len() * 2 * 4);
}
