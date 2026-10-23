//! Q11: the `ibl` Ambient provider over raa-lit's default providers composes and passes naga
//! validation with the standard material function for every pass kind, MSAA and flag set; the
//! filter compute modules (downsample, GGX prefilter, SH9 reduction) validate on their own and
//! the SH reduction uses the constants of the CPU reference src/sh.rs.

// Both providers are crate-private; the test compiles their source files (raa-lit is only
// read).
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;
#[path = "../src/provider.rs"]
mod provider;

use raa_shader::{Composer, MaterialFnId, MaterialFnRegistry, PassKind, SlotId, VariantKey};

fn ibl_composer() -> Composer {
    let mut composer = Composer::new();
    for p in lit_slots::default_slot_providers()
        .into_iter()
        .chain([provider::ambient_provider()])
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
        .collect()
}

fn compose_and_validate(composer: &Composer, key: VariantKey) {
    let reg = MaterialFnRegistry::new();
    let wgsl = composer
        .compose(&reg, &key)
        .unwrap_or_else(|e| panic!("{key:?}: {e}"));
    assert!(wgsl.contains("fn raa_ibl_irradiance("), "{key:?}");
    if let Err(e) = Composer::validate(&wgsl) {
        panic!("{key:?}: {e}");
    }
}

#[test]
fn ibl_wins_ambient() {
    let composer = ibl_composer();
    let w = composer.winner(SlotId::Ambient).expect("winner");
    assert_eq!((w.provider, w.priority), ("ibl", 10));
    for slot in SlotId::ALL.into_iter().filter(|s| *s != SlotId::Ambient) {
        assert_eq!(
            composer.winner(slot).expect("winner").provider,
            "raa.default"
        );
    }
}

#[test]
fn ibl_lib_defines_the_contract_functions() {
    let fns = defined_fns(raa_ibl::IBL_LIB_WGSL);
    for name in [
        "raa_ibl_irradiance",
        "raa_ibl_specular",
        "raa_ibl_dfg",
        "raa_specular_occlusion",
    ] {
        assert_eq!(
            fns.iter().filter(|f| **f == name).count(),
            1,
            "{name}: {fns:?}"
        );
    }
    for f in fns {
        assert!(
            f.starts_with("raa_ibl_") || f == "raa_specular_occlusion",
            "{f}"
        );
    }
}

#[test]
fn provider_defines_the_slot_and_prefixed_helpers() {
    let p = provider::ambient_provider();
    assert_eq!(p.slot, SlotId::Ambient);
    assert!(p.wgsl.starts_with(raa_ibl::IBL_LIB_WGSL));
    let fns = defined_fns(&p.wgsl);
    assert_eq!(fns.iter().filter(|f| **f == "raa_slot_ambient").count(), 1);
    for f in fns {
        assert!(
            f == "raa_slot_ambient" || f == "raa_specular_occlusion" || f.starts_with("raa_ibl_"),
            "helper `{f}` is not prefixed raa_ibl_"
        );
    }
    // Without an environment the S1 PbrLite ambient term is returned.
    assert!(p.wgsl.contains("raa_env_sh.info.w < 0.5"));
    assert!(
        p.wgsl
            .contains("raa_frame.ambient.rgb * s.base_color * (1.0 - 0.5 * s.metallic)")
    );
}

#[test]
fn standard_fn_every_pass_and_msaa_validates() {
    let composer = ibl_composer();
    let mut count = 0;
    for pass in PassKind::ALL {
        for msaa in [1, 4] {
            let key = VariantKey {
                material_fn: MaterialFnId::STANDARD,
                pass,
                msaa,
                flags: 0,
            };
            compose_and_validate(&composer, key);
            count += 1;
        }
    }
    assert_eq!(count, PassKind::ALL.len() * 2);
}

#[test]
fn flagged_variants_validate() {
    let composer = ibl_composer();
    // DOUBLE_SIDED, UNLIT, and DOUBLE_SIDED | VERTEX_FN | UNLIT.
    for pass in PassKind::ALL {
        for msaa in [1, 4] {
            for flags in [8, 64, 8 | 32 | 64] {
                let key = VariantKey {
                    material_fn: MaterialFnId::STANDARD,
                    pass,
                    msaa,
                    flags,
                };
                compose_and_validate(&composer, key);
            }
        }
    }
}

const DOWNSAMPLE_WGSL: &str = include_str!("../src/wgsl/downsample.wgsl");
const PREFILTER_WGSL: &str = include_str!("../src/wgsl/prefilter.wgsl");
const SH_PROJECT_WGSL: &str = include_str!("../src/wgsl/sh_project.wgsl");
const SH_RS: &str = include_str!("../src/sh.rs");

#[test]
fn filter_shaders_validate() {
    for (name, wgsl, entries) in [
        ("downsample", DOWNSAMPLE_WGSL, &["downsample"][..]),
        ("prefilter", PREFILTER_WGSL, &["prefilter"][..]),
        (
            "sh_project",
            SH_PROJECT_WGSL,
            &["sh_partial", "sh_final"][..],
        ),
    ] {
        if let Err(e) = Composer::validate(wgsl) {
            panic!("{name}: {e}");
        }
        // Standalone compute modules: no PRELUDE, own bindings in group 0 only.
        assert!(!wgsl.contains("raa_frame"), "{name}");
        assert_eq!(
            wgsl.matches("@group(").count(),
            wgsl.matches("@group(0)").count(),
            "{name}"
        );
        let fns = defined_fns(wgsl);
        for entry in entries {
            assert!(fns.contains(entry), "{name}: entry point {entry}: {fns:?}");
        }
        assert_eq!(wgsl.matches("@compute").count(), entries.len(), "{name}");
    }
}

/// Decimal literals with exactly six fractional digits, in source order.
fn six_digit_literals(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() || (i > 0 && bytes[i - 1].is_ascii_digit()) {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'.' {
            let frac = i + 1;
            i = frac;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if i - frac == 6 {
                out.push(&text[start..i]);
            }
        }
    }
    out
}

/// Comma-separated items (whitespace-trimmed, empty dropped) between `open` after `marker`
/// and the next `close`.
fn list_after<'a>(text: &'a str, marker: &str, open: &str, close: &str) -> Vec<&'a str> {
    let rest = &text[text.find(marker).unwrap_or_else(|| panic!("{marker}"))..];
    let rest = &rest[rest.find(open).unwrap_or_else(|| panic!("{open}")) + open.len()..];
    let body = &rest[..rest.find(close).unwrap_or_else(|| panic!("{close}"))];
    body.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

#[test]
fn sh_constants_match_cpu() {
    // The five basis constants, used in the same order by both.
    let cpu = six_digit_literals(SH_RS);
    let gpu = six_digit_literals(SH_PROJECT_WGSL);
    for c in ["0.282095", "0.488603", "1.092548", "0.315392", "0.546274"] {
        assert!(cpu.contains(&c), "src/sh.rs lacks {c}");
        assert!(gpu.contains(&c), "sh_project.wgsl lacks {c}");
    }
    assert_eq!(cpu.len(), 9, "{cpu:?}");
    assert_eq!(cpu, gpu);
    // The basis polynomials are textually the same.
    let cpu_basis = list_after(SH_RS, "fn sh_basis", "= d;\n    [", "\n    ]");
    let gpu_basis = list_after(SH_PROJECT_WGSL, "fn sh_basis", "array<f32, 9>(", "\n    )");
    assert_eq!(cpu_basis.len(), 9, "{cpu_basis:?}");
    assert_eq!(cpu_basis, gpu_basis);
    // The A_l / pi factors.
    let cpu_cosine = list_after(SH_RS, "const COSINE", "= [", "];");
    let gpu_cosine = list_after(SH_PROJECT_WGSL, "const SH_COSINE", "array<f32, 9>(", ");");
    assert_eq!(
        cpu_cosine,
        [
            "1.0",
            "2.0 / 3.0",
            "2.0 / 3.0",
            "2.0 / 3.0",
            "0.25",
            "0.25",
            "0.25",
            "0.25",
            "0.25"
        ]
    );
    assert_eq!(cpu_cosine, gpu_cosine);
}
