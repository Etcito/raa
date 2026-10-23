//! The `layered` slot providers (S11), priority 20 (above `ibl` at 10 and the raa-lit
//! defaults at 0): `Direct` is `wgsl/direct.wgsl` ([`BRDF_WGSL`], the WGSL mirror of
//! [`crate::eval_direct`]); `Ambient` is [`raa_ibl::IBL_LIB_WGSL`] followed by
//! `wgsl/ambient.wgsl`, as raa-ibl builds its own ambient provider.

use std::borrow::Cow;

use raa_core::RaaError;
use raa_lit::LitRenderer;
use raa_shader::{SlotId, SlotProvider};

/// Provider name of both layered slots.
const LAYERED_PROVIDER: &str = "layered";

/// Priority of both layered slots.
const LAYERED_PRIORITY: i32 = 20;

/// The `layered` `Direct` provider source: `raa_slot_direct` and its `raa_layered_` helpers,
/// the layered BRDF of the CPU reference [`crate::eval_direct`] term by term (same fits and
/// constants). The clearcoat uses the shading normal, as the S11 slot receives no other.
pub const BRDF_WGSL: &str = include_str!("wgsl/direct.wgsl");

/// `raa_slot_ambient` of provider `layered` and its `raa_layered_ambient_` helpers; the
/// provider source puts [`raa_ibl::IBL_LIB_WGSL`] before it.
const AMBIENT_WGSL: &str = include_str!("wgsl/ambient.wgsl");

/// The two `layered` providers, priority 20: `Direct` (base, sheen and clearcoat of
/// [`BRDF_WGSL`]) then `Ambient` (raa-ibl's base terms under the same sheen and clearcoat
/// attenuation, plus the sheen indirect and the clearcoat's prefiltered reflection).
pub fn providers() -> [SlotProvider; 2] {
    let mut ambient = String::with_capacity(raa_ibl::IBL_LIB_WGSL.len() + 1 + AMBIENT_WGSL.len());
    ambient.push_str(raa_ibl::IBL_LIB_WGSL);
    ambient.push('\n');
    ambient.push_str(AMBIENT_WGSL);
    [
        SlotProvider {
            slot: SlotId::Direct,
            priority: LAYERED_PRIORITY,
            provider: LAYERED_PROVIDER,
            wgsl: Cow::Borrowed(BRDF_WGSL),
        },
        SlotProvider {
            slot: SlotId::Ambient,
            priority: LAYERED_PRIORITY,
            provider: LAYERED_PROVIDER,
            wgsl: Cow::Owned(ambient),
        },
    ]
}

/// Adds both [`providers`] to `lit` with `add_slot` (which clears its pipeline cache).
///
/// Refusals (nothing installed): the `add_slot` refusal with its code
/// (`shader.slot_conflict` when installed twice or when another provider holds priority 20
/// on either slot). Both providers are first added to a copy of the composer, so a refused
/// `Ambient` never leaves the `Direct` provider installed alone.
pub fn install(lit: &mut LitRenderer) -> Result<(), RaaError> {
    let [direct, ambient] = providers();
    let mut probe = lit.composer().clone();
    for p in [&direct, &ambient] {
        probe
            .add_slot(p.clone())
            .map_err(|e| RaaError::FrameGraph {
                code: e.code,
                message: e.message,
            })?;
    }
    lit.add_slot(direct)?;
    lit.add_slot(ambient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_shader::{Composer, MaterialFnId, MaterialFnRegistry, PassKind, VariantKey};

    /// The S11 default of every slot at priority 0 (raa-lit's `raa.default` bodies, Direct
    /// reduced to Lambert): raa-lit's own providers are crate-private and only tests/ may
    /// include their source by path.
    const DEFAULTS: [(SlotId, &str); 7] = [
        (
            SlotId::Deform,
            "fn raa_slot_deform(p: vec3<f32>, n: vec3<f32>, x: InstanceExtraGpu, t: f32) -> VertexMaterialOutput {\n    return VertexMaterialOutput(p, n);\n}\n",
        ),
        (
            SlotId::Shadow,
            "fn raa_slot_shadow(world_pos: vec3<f32>, n: vec3<f32>, frag: vec4<f32>) -> f32 {\n    return 1.0;\n}\n",
        ),
        (
            SlotId::Ao,
            "fn raa_slot_ao(frag: vec4<f32>) -> f32 {\n    return 1.0;\n}\n",
        ),
        (
            SlotId::Ambient,
            "fn raa_slot_ambient(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {\n    return raa_frame.ambient.rgb * s.base_color * (1.0 - 0.5 * s.metallic);\n}\n",
        ),
        (
            SlotId::Direct,
            "fn raa_slot_direct(s: MaterialOutput, c: ShadeCtx, l: vec3<f32>, radiance: vec3<f32>) -> vec3<f32> {\n    return s.base_color * (1.0 - s.metallic) / 3.14159265 * radiance * max(dot(s.normal, l), 0.0);\n}\n",
        ),
        (
            SlotId::Transmission,
            "fn raa_slot_transmission(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {\n    return vec3<f32>(0.0);\n}\n",
        ),
        (
            SlotId::Fog,
            "fn raa_slot_fog(color: vec3<f32>, world_pos: vec3<f32>, c: ShadeCtx) -> vec3<f32> {\n    return color;\n}\n",
        ),
    ];

    const CPU_RS: &str = include_str!("cpu.rs");

    fn layered_composer() -> Composer {
        let mut composer = Composer::new();
        let defaults = DEFAULTS.map(|(slot, wgsl)| SlotProvider {
            slot,
            priority: 0,
            provider: "test.default",
            wgsl: Cow::Borrowed(wgsl),
        });
        for p in defaults.into_iter().chain(providers()) {
            if let Err(e) = composer.add_slot(p) {
                panic!("{e}");
            }
        }
        composer
    }

    /// Names of the module-scope declarations (`fn`, `const`, `struct`, `var`, `override`,
    /// `alias` at the start of a line) of `wgsl`.
    fn declared(wgsl: &str) -> Vec<&str> {
        let mut names = Vec::new();
        for line in wgsl.lines() {
            let Some((keyword, rest)) = line.split_once([' ', '<']) else {
                continue;
            };
            if !matches!(
                keyword,
                "fn" | "const" | "struct" | "var" | "override" | "alias"
            ) {
                continue;
            }
            // `var<private> name`: skip the address space.
            let rest = if line[keyword.len()..].starts_with('<') {
                rest.split_once('>').map_or(rest, |(_, r)| r)
            } else {
                rest
            };
            let name = rest
                .trim_start()
                .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .next()
                .unwrap_or("");
            names.push(name);
        }
        names
    }

    /// The numeric literals of the declaration `const <name>:` in
    /// `src`, from its `=` to the next `;` (Rust or WGSL).
    fn const_numbers(src: &str, name: &str) -> Vec<f32> {
        let decl = format!("const {name}:");
        let start = src.find(&decl).unwrap_or_else(|| panic!("no {decl}"));
        let rest = &src[start..];
        let rest = &rest[rest.find('=').expect("=") + 1..];
        let body = &rest[..rest.find(';').expect(";")];
        body.split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '-' | '+')))
            .filter(|t| t.contains(|c: char| c.is_ascii_digit()))
            .map(|t| {
                t.parse::<f32>()
                    .unwrap_or_else(|e| panic!("{name}: {t}: {e}"))
            })
            .collect()
    }

    #[test]
    fn providers_are_layered_direct_and_ambient_at_priority_20() {
        let [direct, ambient] = providers();
        assert_eq!(
            (direct.slot, direct.priority, direct.provider),
            (SlotId::Direct, 20, "layered")
        );
        assert_eq!(
            (ambient.slot, ambient.priority, ambient.provider),
            (SlotId::Ambient, 20, "layered")
        );
        assert_eq!(direct.wgsl, BRDF_WGSL);
        assert!(ambient.wgsl.starts_with(raa_ibl::IBL_LIB_WGSL));
        assert!(ambient.wgsl.ends_with(AMBIENT_WGSL));
    }

    #[test]
    fn provider_sources_define_their_slot_and_only_layered_or_ibl_helpers() {
        let [direct, ambient] = providers();
        for p in [&direct, &ambient] {
            assert!(
                !p.wgsl.lines().any(|l| l.starts_with('@')),
                "{:?}: attribute at module scope",
                p.slot
            );
        }

        let direct_names = declared(&direct.wgsl);
        assert_eq!(
            direct_names
                .iter()
                .filter(|n| n.starts_with("raa_slot_"))
                .count(),
            1,
            "{direct_names:?}"
        );
        assert!(direct_names.contains(&"raa_slot_direct"));
        for name in direct_names.iter().filter(|n| **n != "raa_slot_direct") {
            let lower = name.to_ascii_lowercase();
            assert!(
                lower.starts_with("raa_layered_") && !lower.starts_with("raa_layered_ambient_"),
                "direct: {name}"
            );
        }

        // IBL_LIB_WGSL holds raa-ibl's contract functions: raa_ibl_* and
        // raa_specular_occlusion.
        let ibl_names = declared(raa_ibl::IBL_LIB_WGSL);
        assert!(!ibl_names.is_empty());
        for name in &ibl_names {
            assert!(
                name.starts_with("raa_ibl_") || *name == "raa_specular_occlusion",
                "ibl lib: {name}"
            );
        }
        let ambient_names = declared(&ambient.wgsl);
        assert_eq!(
            ambient_names
                .iter()
                .filter(|n| n.starts_with("raa_slot_"))
                .count(),
            1,
            "{ambient_names:?}"
        );
        assert!(ambient_names.contains(&"raa_slot_ambient"));
        for name in ambient_names.iter().filter(|n| **n != "raa_slot_ambient") {
            let lower = name.to_ascii_lowercase();
            assert!(
                lower.starts_with("raa_layered_ambient_") || ibl_names.contains(name),
                "ambient: {name}"
            );
        }
        // Both sources win in one module: no name is defined twice.
        for name in &direct_names {
            assert!(!ambient_names.contains(name), "defined twice: {name}");
        }
    }

    #[test]
    fn standard_forward_msaa4_composes_and_validates_with_the_layered_providers() {
        let composer = layered_composer();
        for slot in [SlotId::Direct, SlotId::Ambient] {
            let w = composer.winner(slot).expect("winner");
            assert_eq!((w.provider, w.priority), ("layered", 20), "{slot:?}");
        }
        let key = VariantKey {
            material_fn: MaterialFnId::STANDARD,
            pass: PassKind::Forward,
            msaa: 4,
            flags: 0,
        };
        let wgsl = composer
            .compose(&MaterialFnRegistry::new(), &key)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(wgsl.contains("fn raa_layered_direct("));
        assert!(wgsl.contains("fn raa_ibl_irradiance("));
        if let Err(e) = Composer::validate(&wgsl) {
            panic!("{e}");
        }
    }

    #[test]
    fn wgsl_constants_and_fit_tables_are_the_cpu_ones() {
        let ambient = AMBIENT_WGSL;
        for (cpu, wgsl, src) in [
            ("GGX_E_FIT", "RAA_LAYERED_GGX_E_FIT", BRDF_WGSL),
            ("GGX_B_FIT", "RAA_LAYERED_GGX_B_FIT", BRDF_WGSL),
            ("SHEEN_FIT", "RAA_LAYERED_SHEEN_FIT", BRDF_WGSL),
            ("SHEEN_SV_FIT", "RAA_LAYERED_SHEEN_SV_FIT", BRDF_WGSL),
            ("SHEEN_FIT", "RAA_LAYERED_AMBIENT_SHEEN_FIT", ambient),
            ("SHEEN_SV_FIT", "RAA_LAYERED_AMBIENT_SHEEN_SV_FIT", ambient),
        ] {
            let expected = const_numbers(CPU_RS, cpu);
            assert!(expected.len() >= 3, "{cpu}: {expected:?}");
            assert_eq!(const_numbers(src, wgsl), expected, "{wgsl}");
        }
        for (wgsl, src, value) in [
            ("RAA_LAYERED_MIN_ROUGHNESS", BRDF_WGSL, crate::MIN_ROUGHNESS),
            (
                "RAA_LAYERED_MIN_SHEEN_ROUGHNESS",
                BRDF_WGSL,
                crate::MIN_SHEEN_ROUGHNESS,
            ),
            ("RAA_LAYERED_MIN_N_DOT_V", BRDF_WGSL, crate::MIN_N_DOT_V),
            ("RAA_LAYERED_CLEARCOAT_F0", BRDF_WGSL, crate::CLEARCOAT_F0),
            (
                "RAA_LAYERED_AMBIENT_MIN_SHEEN_ROUGHNESS",
                ambient,
                crate::MIN_SHEEN_ROUGHNESS,
            ),
            (
                "RAA_LAYERED_AMBIENT_MIN_N_DOT_V",
                ambient,
                crate::MIN_N_DOT_V,
            ),
            (
                "RAA_LAYERED_AMBIENT_CLEARCOAT_F0",
                ambient,
                crate::CLEARCOAT_F0,
            ),
        ] {
            assert_eq!(const_numbers(src, wgsl), [value], "{wgsl}");
        }
    }
}
