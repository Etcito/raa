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
    let mut reg = MaterialFnRegistry::new();
    let composer = default_composer();
    let glass = reg
        .register_builtin(
            "raa.glass",
            include_str!("../../raa-transmission/src/wgsl/glass.wgsl"),
        )
        .expect("glass material");
    let custom = reg.register_user("test.transparent", "fn material_main(in: MaterialInput) -> MaterialOutput { return raa_default_output(in); }", &composer).expect("custom material");
    // Plain, DOUBLE_SIDED and UNLIT variants.
    for (material_fn, flags) in [MaterialFnId::STANDARD, glass, custom]
        .into_iter()
        .flat_map(|id| [0, 8, 64].map(|flags| (id, flags)))
    {
        let key = VariantKey {
            material_fn,
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
        #[cfg(not(target_arch = "wasm32"))]
        check_resources(&wgsl);
        let posed = raa_lit::skin_draw::posed_module(&wgsl).expect("posed transparent");
        Composer::validate(&posed).expect("posed validation");
        #[cfg(not(target_arch = "wasm32"))]
        check_resources(&posed);
    }
}

#[test]
fn shared_lighting_preserves_transmission_fog_and_oit() {
    let pass = raa_shader::PASS_TRANSPARENT_WGSL;
    for term in [
        "raa_local_lighting(s, c)",
        "raa_slot_indirect(s, c)",
        "raa_slot_transmission(s, c)",
    ] {
        assert!(pass.find(term).expect(term) < pass.find("raa_slot_fog(lit,").unwrap());
    }
    assert!(pass.contains("select(shaded, s.base_color + s.emissive, raa_pass_unlit(in.record))"));
    assert!(pass.contains("raa_pass_oit_append(in.frag_coord, color, clamp(s.opacity, 0.0, 1.0))"));
    assert!(pass.contains("atomicAdd(&raa_oit_counter, 1u)"));
    assert!(pass.contains("index >= raa_oit_params.capacity"));
}

// Reflect every entry point, including posed/custom variants. Count declared fragment
// storage conservatively (even unused bindings), excluding only vertex-only resources.
#[cfg(not(target_arch = "wasm32"))]
fn check_resources(source: &str) {
    use wgpu::naga::{AddressSpace, ShaderStage, TypeInner, front::wgsl, valid};
    let module = wgsl::parse_str(source).expect("parse resources");
    let oit = module
        .global_variables
        .iter()
        .filter(|(_, g)| {
            g.binding.as_ref().is_some_and(|b| b.group == 2)
                && matches!(g.space, AddressSpace::Storage { .. })
        })
        .count();
    assert_eq!(oit, 3, "OIT keeps its three storage allocations");
    let node = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("OitNode"))
        .unwrap()
        .1;
    assert!(matches!(node.inner, TypeInner::Struct { span: 16, .. }));
    let info = valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
        .validate(&module)
        .expect("validate resources");
    let mut fragment_storage = 0;
    for (handle, global) in module.global_variables.iter() {
        let Some(binding) = &global.binding else {
            continue;
        };
        assert!(binding.group < 4, "four bind groups maximum");
        let vertex_only = (binding.group, binding.binding) == (0, 8) || binding.group == 3;
        if binding.group == 3 {
            assert_eq!(binding.binding, 0);
            assert_eq!(global.name.as_deref(), Some("raa_skin_palette"));
        }
        if matches!(global.space, AddressSpace::Storage { .. }) && !vertex_only {
            fragment_storage += 1;
        }
        if binding.group == 1 && binding.binding == 12 {
            assert_eq!(global.space, AddressSpace::Uniform);
        }
        if binding.group == 1 && binding.binding == 13 {
            assert_eq!(global.space, AddressSpace::Handle);
            assert!(matches!(
                module.types[global.ty].inner,
                TypeInner::Image { .. }
            ));
        }
        for (index, entry) in module.entry_points.iter().enumerate() {
            if entry.stage == ShaderStage::Fragment && vertex_only {
                assert!(
                    info.get_entry_point(index)[handle].is_empty(),
                    "vertex-only resource"
                );
            }
            if entry.stage == ShaderStage::Fragment
                && binding.group == 1
                && [12, 13].contains(&binding.binding)
            {
                assert!(
                    !info.get_entry_point(index)[handle].is_empty(),
                    "lights reached by fragment"
                );
            }
        }
    }
    assert!(
        fragment_storage <= 8,
        "fragment storage budget: {fragment_storage}"
    );
}
