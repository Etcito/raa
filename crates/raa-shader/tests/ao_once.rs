//! Pins that ambient occlusion is applied exactly once (product-owner directive 2026-11-14,
//! module raa-ao-once). A lit pass hands the combined occlusion `min(material ao, screen ao)` to
//! the slots through `ShadeCtx.ao` and never multiplies a slot's result by AO; the Ambient
//! provider alone occludes the ambient term, once, in every branch. Direct light and the shared
//! local lighting are never occluded by AO. Text pins with no GPU.

const FORWARD: &str = raa_shader::PASS_FORWARD_WGSL;
const TRANSPARENT: &str = raa_shader::PASS_TRANSPARENT_WGSL;
const FURSHELL: &str = raa_shader::PASS_FURSHELL_WGSL;
const VG_SHADE: &str = raa_shader::PASS_VG_SHADE_WGSL;

const DEFAULT_AMBIENT: &str = include_str!("../../raa-lit/src/wgsl/slots_default.wgsl");
const IBL_AMBIENT: &str = include_str!("../../raa-ibl/src/wgsl/ambient.wgsl");
const BRDF_AMBIENT: &str = include_str!("../../raa-brdf/src/wgsl/ambient.wgsl");
const PARTICLES_OPAQUE: &str = include_str!("../../raa-particles/src/wgsl/particles_opaque.wgsl");
const PLUSH: &str = include_str!("../../raa-matfn/src/wgsl/plush.wgsl");
const SLOT_AO: &str = include_str!("../../raa-gi/src/wgsl/slots/slot_ao.wgsl");
const SLOT_INDIRECT: &str = include_str!("../../raa-gi/src/wgsl/slots/slot_indirect.wgsl");

/// The text from `start` through its first closing brace.
fn braced<'a>(source: &'a str, start: &str) -> &'a str {
    let begin = source
        .find(start)
        .unwrap_or_else(|| panic!("missing {start:?}"));
    let end = source[begin..]
        .find('}')
        .unwrap_or_else(|| panic!("unterminated {start:?}"))
        + begin;
    &source[begin..=end]
}

#[test]
fn passes_hand_the_combined_occlusion_and_apply_ambient_ao_once() {
    for (name, source) in [
        ("forward", FORWARD),
        ("transparent", TRANSPARENT),
        ("furshell", FURSHELL),
        ("vg.shade", VG_SHADE),
    ] {
        assert!(source.contains("c.ao = min(s.ao, ao);"), "{name}");
        assert_eq!(
            source.matches("raa_slot_ambient(s, c)").count(),
            1,
            "{name}: the ambient term is applied once"
        );
        assert!(!source.contains("raa_slot_ambient(s, c) * min("), "{name}");
        assert!(!source.contains("raa_slot_ambient(s, c) * ao"), "{name}");

        // Direct light is never occluded by AO.
        let direct = source.find("raa_slot_direct(").expect("direct term");
        let ambient = source.find("raa_slot_ambient(s, c)").expect("ambient term");
        assert!(
            !source[direct..ambient].contains("ao"),
            "{name}: direct term has no AO factor"
        );
        // The shared local lighting is never occluded by AO.
        let local = source.find("raa_local_lighting(s, c)").expect("local term");
        let end = source[local..].find(';').expect("local term end");
        assert!(
            !source[local..local + end].contains("ao"),
            "{name}: local term has no AO factor"
        );
    }
}

#[test]
fn fur_keeps_its_shell_self_shadowing_on_the_single_ambient_term() {
    assert!(FURSHELL.contains("raa_slot_ambient(s, c) * root"));
}

#[test]
fn ambient_providers_occlude_the_ambient_term_once() {
    let default_ambient = braced(DEFAULT_AMBIENT, "fn raa_slot_ambient(");
    assert!(default_ambient.contains("* c.ao"), "default provider");
    for (name, source) in [("ibl", IBL_AMBIENT), ("brdf", BRDF_AMBIENT)] {
        let no_environment = braced(source, "if raa_env_sh.info.w < 0.5 {");
        assert!(
            no_environment.contains("* c.ao"),
            "{name} no-environment branch"
        );
    }
}

#[test]
fn particles_plush_and_gi_slots_apply_ao_once() {
    assert!(!PARTICLES_OPAQUE.contains("raa_slot_ambient(s, c) * ao"));
    assert!(PARTICLES_OPAQUE.contains("+ raa_slot_ambient(s, c);"));

    assert!(PLUSH.contains("c.ao = min(o.ao, raa_slot_ao(in.frag_coord));"));
    assert!(PLUSH.contains("+ raa_slot_ambient(o, c);"));
    assert!(!PLUSH.contains("raa_slot_ambient(o, c) *"));

    assert!(SLOT_AO.contains("min(raa_gi_ao_gtao(frag), raa_gi_ao_screen(frag).a)"));
    assert!(!SLOT_AO.contains("* raa_gi_ao_screen(frag).a"));

    assert!(SLOT_INDIRECT.contains("s.ao"));
    assert!(!SLOT_INDIRECT.contains("c.ao"));
}
