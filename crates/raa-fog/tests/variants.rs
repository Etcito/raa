//! Q11: the `height` Fog slot provider, over raa-lit's default providers, composes and passes
//! naga validation with the standard material function for every pass kind at MSAA 1 and 4;
//! the composed fog pass validates; the provider defines only `raa_slot_fog` and
//! `raa_height_*` helpers. P2: the sun inscatter crosses the S11 directional boundary once,
//! the ambient fog radiance reads only an accepted published environment (flat ambient when
//! cleared), the pass keeps its private group 2 and stays the single aerial path.

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
        "raa_directional_radiance",
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

/// Body of the WGSL function `name` in `src` (from its `fn` line to the closing `}` line).
fn fn_body<'a>(src: &'a str, name: &str) -> &'a str {
    let start = src
        .find(&format!("fn {name}("))
        .unwrap_or_else(|| panic!("fn {name}"));
    let end = src[start..].find("\n}\n").expect("closing brace") + start;
    &src[start..end]
}

/// Non-definition lines of `src` calling `name(`.
fn calls<'a>(src: &'a str, name: &str) -> Vec<&'a str> {
    let call = format!("{name}(");
    src.lines()
        .map(str::trim)
        .filter(|l| l.contains(&call) && !l.starts_with("fn ") && !l.starts_with("//"))
        .collect()
}

#[test]
fn sun_inscatter_crosses_the_directional_boundary_once() {
    let src = shaders::FOG_SLOT_WGSL;
    assert!(
        !src.contains("fn raa_directional_radiance("),
        "prelude owns the helper"
    );
    let boundary = calls(src, "raa_directional_radiance");
    assert_eq!(
        boundary,
        ["let sun_radiance = raa_directional_radiance(raw_sun);"]
    );
    let radiance = fn_body(src, "raa_height_radiance");
    assert!(radiance.contains(boundary[0]));
    // The raw sun radiance is read once, and only feeds that boundary call.
    let raw: Vec<&str> = src
        .lines()
        .map(str::trim)
        .filter(|l| l.contains("raa_frame.sun_color"))
        .collect();
    assert_eq!(raw, ["let raw_sun = raa_frame.sun_color.rgb;"]);
    assert_eq!(
        src.matches("raw_sun").count(),
        2,
        "raw sun used only at the boundary"
    );
    assert!(radiance.contains("raa_height_hg(cos_sun, 0.6) * sun_radiance"));
    let pass = shaders::fog_pass_source();
    assert_eq!(pass.matches("fn raa_directional_radiance(").count(), 1);
    assert_eq!(pass.matches("struct WeatherLightingGpu").count(), 1);
    assert_eq!(pass.matches("@group(1) @binding(16)").count(), 1);
    assert_eq!(calls(&pass, "raa_directional_radiance").len(), 1);
}

#[test]
fn ambient_fog_reads_only_the_published_environment() {
    let env = fn_body(shaders::FOG_SLOT_WGSL, "raa_height_env");
    let gate = env
        .find("if (raa_env_sh.info.w < 0.5) {")
        .expect("publication gate");
    let fallback = env
        .find("return raa_frame.ambient.rgb;")
        .expect("flat fallback");
    let sample = env
        .find("textureSampleLevel(raa_env_specular")
        .expect("env sample");
    assert!(gate < fallback && fallback < sample, "{env}");
    assert!(
        env.contains("raa_env_sh.info.z"),
        "max mip of the published cube"
    );
    // The aerial perspective and the height radiance both go through that gate.
    assert!(fn_body(shaders::FOG_SLOT_WGSL, "raa_height_radiance").contains("raa_height_env(dir)"));
    assert!(fn_body(shaders::FOG_SLOT_WGSL, "raa_height_apply").contains("raa_height_env("));
    assert_eq!(
        shaders::FOG_SLOT_WGSL.matches("raa_env_specular").count(),
        1
    );
}

#[test]
fn background_horizon_weighting_and_single_aerial_path_are_kept() {
    let slot = shaders::FOG_SLOT_WGSL;
    let apply = fn_body(slot, "raa_height_apply");
    assert!(slot.contains("const raa_height_sky_distance_m: f32 = 20000.0;"));
    assert!(apply.contains("1.0 - smoothstep(0.0, 0.25, abs(dir.y))"));
    assert!(
        apply.contains("if (sky) {\n        return out;\n    }"),
        "sky skips aerial"
    );
    let pass = shaders::FOG_PASS_WGSL;
    // Sky, physical/legacy background and the flat colour all sit at clear depth 1.
    assert!(pass.contains("let sky = depth >= 1.0;"));
    assert!(pass.contains("raa_height_apply(src.rgb, world, camera, sky)"));
    assert_eq!(pass.matches("@vertex").count(), 1);
    assert_eq!(pass.matches("@fragment").count(), 1);
    assert_eq!(
        pass.matches("@group(").count(),
        3,
        "only the private group 2"
    );
    assert_eq!(pass.matches("@group(2)").count(), 3);
    for src in [slot, pass] {
        assert!(
            !src.contains("aerial_perspective"),
            "no second aerial implementation"
        );
    }
}

#[test]
fn forward_transparent_and_fur_compositions_share_one_boundary() {
    let reg = MaterialFnRegistry::new();
    let composer = composer();
    for pass in [PassKind::Forward, PassKind::Transparent, PassKind::FurShell] {
        let key = VariantKey {
            material_fn: MaterialFnId::STANDARD,
            pass,
            msaa: 4,
            flags: 0,
        };
        let wgsl = composer.compose(&reg, &key).unwrap();
        assert_eq!(wgsl.matches(shaders::FOG_SLOT_WGSL).count(), 1, "{pass:?}");
        assert_eq!(
            wgsl.matches("fn raa_directional_radiance(").count(),
            1,
            "{pass:?}"
        );
        assert_eq!(
            wgsl.matches("struct WeatherLightingGpu").count(),
            1,
            "{pass:?}"
        );
    }
}
