//! Q11: the `layered` Direct and Ambient providers, over raa-lit's default slot providers and
//! raa-ibl's `ibl` Ambient provider, compose and pass naga validation with `raa.standard` and
//! every raa-matfn built-in, for every pass kind, MSAA 1/4 and the flag sets 0, the row's own
//! flags and DOUBLE_SIDED.

// The lower-priority providers are crate-private; the test compiles their source files (both
// crates are only read).
#[allow(dead_code)]
#[path = "../../raa-ibl/src/provider.rs"]
mod ibl_provider;
#[allow(dead_code)]
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;

use raa_lit::FLAG_DOUBLE_SIDED;
use raa_matfn::{BUILTINS, register_builtins};
use raa_shader::{Composer, MaterialFnId, MaterialFnRegistry, PassKind, SlotId, VariantKey};

fn composer() -> Composer {
    let mut composer = Composer::new();
    for p in lit_slots::default_slot_providers()
        .into_iter()
        .chain([ibl_provider::ambient_provider()])
        .chain(raa_brdf::providers())
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

fn forward_key() -> VariantKey {
    VariantKey {
        material_fn: MaterialFnId::STANDARD,
        pass: PassKind::Forward,
        msaa: 1,
        flags: 0,
    }
}

fn compose_with_direct(direct: String) -> String {
    let mut composer = composer();
    composer
        .add_slot(raa_shader::SlotProvider {
            slot: SlotId::Direct,
            priority: 30,
            provider: "probe",
            wgsl: direct.into(),
        })
        .expect("probe slot");
    composer
        .compose(&registry(), &forward_key())
        .expect("compose")
}

#[test]
fn layered_wins_direct_and_ambient() {
    let composer = composer();
    for slot in SlotId::ALL {
        let w = composer.winner(slot).expect("winner");
        if matches!(slot, SlotId::Direct | SlotId::Ambient) {
            assert_eq!((w.provider, w.priority), ("layered", 20), "{slot:?}");
        } else {
            assert_eq!(w.provider, "raa.default", "{slot:?}");
        }
    }
}

#[test]
fn every_builtin_validates_with_layered_providers() {
    let composer = composer();
    let reg = registry();
    let rows = [("raa.standard", 0)]
        .into_iter()
        .chain(BUILTINS.iter().map(|b| (b.name, b.flags)));
    let mut fns = 0;
    let mut count = 0;
    for (name, row_flags) in rows {
        let material_fn = reg.id_of(name).expect("registered");
        if name == "raa.standard" {
            assert_eq!(material_fn, MaterialFnId::STANDARD);
        }
        let mut flag_sets = vec![0, row_flags, FLAG_DOUBLE_SIDED];
        flag_sets.sort_unstable();
        flag_sets.dedup();
        for pass in PassKind::ALL {
            for msaa in [1, 4] {
                for &flags in &flag_sets {
                    let key = VariantKey {
                        material_fn,
                        pass,
                        msaa,
                        flags,
                    };
                    let wgsl = composer
                        .compose(&reg, &key)
                        .unwrap_or_else(|e| panic!("{name} {key:?}: {e}"));
                    assert!(wgsl.contains("fn raa_layered_"), "{name} {key:?}");
                    if let Err(e) = Composer::validate(&wgsl) {
                        panic!("{name} {key:?}: {e}");
                    }
                    count += 1;
                }
            }
        }
        fns += 1;
        println!("{name} validates with the layered providers");
    }
    println!("{fns} material functions, {count} variants");
    assert_eq!(fns, BUILTINS.len() + 1);
    assert!(count >= fns * PassKind::ALL.len() * 2 * 2);
}

#[test]
fn direct_sampling_follows_the_nonuniform_loop_rules() {
    for token in [
        "textureSample(",
        "textureSampleBias(",
        "dpdx(",
        "dpdy(",
        "fwidth(",
    ] {
        assert!(!raa_brdf::BRDF_WGSL.contains(token), "{token}");
    }
    let wgsl = composer()
        .compose(&registry(), &forward_key())
        .expect("compose");
    assert_eq!(wgsl.matches("fn raa_slot_direct(").count(), 1);
    assert_eq!(
        wgsl.matches("raa_slot_direct(s, c, direction, radiance)")
            .count(),
        1
    );
    assert_eq!(Composer::validate(&wgsl), Ok(()));

    let wgsl = compose_with_direct(
        "fn raa_slot_direct(s: MaterialOutput, c: ShadeCtx, l: vec3<f32>, radiance: vec3<f32>) -> vec3<f32> {
    let at = c.position.xy;
    let a = textureSampleLevel(raa_texture_atlas, raa_texture_sampler, at, 0.0).rgb;
    let g = vec2<f32>(0.0, 0.0);
    let b = textureSampleGrad(raa_texture_atlas, raa_texture_sampler, at, g, g).rgb;
    return (a + b) * radiance * max(dot(s.normal, l), 0.0);
}
"
        .into(),
    );
    assert_eq!(Composer::validate(&wgsl), Ok(()));
    for expression in [
        "dpdx(c.position.x)",
        "textureSample(raa_texture_atlas, raa_texture_sampler, c.position.xy).x",
    ] {
        let wgsl = compose_with_direct(format!(
            "fn raa_slot_direct(s: MaterialOutput, c: ShadeCtx, l: vec3<f32>, radiance: vec3<f32>) -> vec3<f32> {{ return vec3<f32>({expression}); }}"
        ));
        let error = Composer::validate(&wgsl).expect_err("implicit derivative must fail");
        assert_eq!(error.code(), "shader.validate", "{error}");
        assert!(error.message.contains("uniform"), "{error}");
    }
}
