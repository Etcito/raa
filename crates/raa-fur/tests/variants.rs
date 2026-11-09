//! Q11: the FurShell variant of `raa.plush` composes and passes naga validation at MSAA 1/4
//! with the flags {FUR, FUR | DOUBLE_SIDED}, over raa-lit's default slot providers, raa-ibl's
//! `ibl` Ambient provider and raa-brdf's `layered` providers; the composed module declares
//! the group-2 `raa_fur_shell` uniform with the 16-byte layout the contributor writes.

// The default and ibl providers are crate-private; the test compiles their source files
// (both crates are only read).
#[allow(dead_code)]
#[path = "../../raa-ibl/src/provider.rs"]
mod ibl_provider;
#[allow(dead_code)]
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;

use raa_fur::{FUR_SHELLS_MAX, SHELL_UNIFORM_SIZE, write_shell_slots};
use raa_lit::{FLAG_DOUBLE_SIDED, FLAG_FUR};
use raa_matfn::register_builtins;
use raa_shader::{Composer, MaterialFnRegistry, PassKind, SlotId, SlotProvider, VariantKey};

const FLAG_SETS: [u32; 2] = [FLAG_FUR, FLAG_FUR | FLAG_DOUBLE_SIDED];

/// Default providers plus `extra`, in order.
fn composer(extra: impl IntoIterator<Item = SlotProvider>) -> Composer {
    let mut composer = Composer::new();
    for p in lit_slots::default_slot_providers().into_iter().chain(extra) {
        if let Err(e) = composer.add_slot(p) {
            panic!("{e}");
        }
    }
    composer
}

/// Default + ibl + layered: the slot set of an engine with every M4 feature installed.
fn full_composer() -> Composer {
    composer(
        [ibl_provider::ambient_provider()]
            .into_iter()
            .chain(raa_brdf::providers()),
    )
}

/// The slot sets the fur shells meet: default, + ibl, + ibl + layered.
fn slot_sets() -> Vec<(&'static str, Composer)> {
    vec![
        ("default", composer([])),
        ("default+ibl", composer([ibl_provider::ambient_provider()])),
        ("default+ibl+layered", full_composer()),
    ]
}

fn registry() -> MaterialFnRegistry {
    let mut reg = MaterialFnRegistry::new();
    if let Err(e) = register_builtins(&mut reg) {
        panic!("{e}");
    }
    reg
}

fn plush_furshell_keys(reg: &MaterialFnRegistry) -> Vec<VariantKey> {
    let material_fn = reg.id_of("raa.plush").expect("raa.plush registered");
    let mut keys = Vec::new();
    for msaa in [1, 4] {
        for flags in FLAG_SETS {
            keys.push(VariantKey {
                material_fn,
                pass: PassKind::FurShell,
                msaa,
                flags,
            });
        }
    }
    keys
}

fn compose(composer: &Composer, reg: &MaterialFnRegistry, key: &VariantKey) -> String {
    composer
        .compose(reg, key)
        .unwrap_or_else(|e| panic!("{key:?}: {e}"))
}

#[test]
fn layered_and_default_win_their_slots() {
    let composer = full_composer();
    for slot in SlotId::ALL {
        let w = composer.winner(slot).expect("winner");
        let expected = match slot {
            SlotId::Ambient | SlotId::Direct => ("layered", 20),
            _ => ("raa.default", 0),
        };
        assert_eq!((w.provider, w.priority), expected, "{slot:?}");
    }
}

#[test]
fn plush_furshell_variants_validate() {
    let mut reg = registry();
    let custom = reg.register_user("test.fur", "fn material_main(in: MaterialInput) -> MaterialOutput { return raa_default_output(in); }", &full_composer()).expect("custom material");
    let mut keys = plush_furshell_keys(&reg);
    keys.extend(plush_furshell_keys(&reg).into_iter().map(|key| VariantKey {
        material_fn: custom,
        ..key
    }));
    assert_eq!(keys.len(), 8);
    let mut count = 0;
    for (name, composer) in slot_sets() {
        for key in &keys {
            let wgsl = compose(&composer, &reg, key);
            if let Err(e) = Composer::validate(&wgsl) {
                panic!("{name} {key:?}: {e}");
            }
            #[cfg(not(target_arch = "wasm32"))]
            check_resources(&wgsl);
            let posed = raa_lit::skin_draw::posed_module(&wgsl).expect("posed fur");
            Composer::validate(&posed).expect("posed validation");
            #[cfg(not(target_arch = "wasm32"))]
            check_resources(&posed);
            count += 1;
        }
        println!("raa.plush FurShell validates with {name} slots");
    }
    assert_eq!(count, 24);
}

#[test]
fn shared_lighting_keeps_roots_fog_and_coverage() {
    let pass = raa_shader::PASS_FURSHELL_WGSL;
    for term in [
        "raa_local_lighting(s, c) * root",
        "raa_slot_indirect(s, c) * root",
    ] {
        assert!(pass.find(term).expect(term) < pass.find("raa_slot_fog(lit,").unwrap());
    }
    assert!(pass.contains("select(shaded, s.base_color + s.emissive, raa_pass_unlit(in.record))"));
    assert!(pass.contains("coverage * clamp(s.opacity, 0.0, 1.0)"));
    assert!(pass.contains("RAA_MSAA <= 1u && alpha < 0.5"));
    assert!(pass.contains("return vec4<f32>(color, alpha)"));
}

/// The fields of `struct <name> { ... }` in `wgsl` as (field, type) pairs.
fn struct_fields<'a>(wgsl: &'a str, name: &str) -> Vec<(&'a str, &'a str)> {
    let header = format!("struct {name} {{");
    assert_eq!(wgsl.matches(&header).count(), 1, "{header}");
    let start = wgsl.find(&header).expect("struct") + header.len();
    let end = start + wgsl[start..].find('}').expect("struct end");
    wgsl[start..end]
        .lines()
        .map(|l| l.split("//").next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .map(|l| {
            let (field, ty) = l.split_once(':').unwrap_or_else(|| panic!("field `{l}`"));
            (field.trim(), ty.trim().trim_end_matches(',').trim())
        })
        .collect()
}

#[test]
fn group2_uniform_matches_the_contributor_slots() {
    // The contributor's slot layout: index u32 little-endian at byte 0, zeros up to 16 B.
    let mut slots = Vec::new();
    write_shell_slots(SHELL_UNIFORM_SIZE, &mut slots);
    assert_eq!(SHELL_UNIFORM_SIZE, 16);
    assert_eq!(
        slots.len() as u64,
        SHELL_UNIFORM_SIZE * u64::from(FUR_SHELLS_MAX)
    );
    for (i, slot) in slots.chunks_exact(SHELL_UNIFORM_SIZE as usize).enumerate() {
        assert_eq!(&slot[..4], &(i as u32).to_le_bytes());
        assert!(slot[4..].iter().all(|b| *b == 0), "slot {i}");
    }

    let reg = registry();
    let composer = full_composer();
    for key in plush_furshell_keys(&reg) {
        let wgsl = compose(&composer, &reg, &key);
        // Group 2 holds exactly this one uniform, at binding 0.
        let decl = "@group(2) @binding(0) var<uniform> raa_fur_shell: RaaFurShellGpu;";
        assert_eq!(wgsl.matches(decl).count(), 1, "{key:?}");
        assert_eq!(wgsl.matches("@group(2)").count(), 1, "{key:?}");
        // 4 x u32 = 16 B with `index` first (offset 0), as the contributor writes it.
        let fields = struct_fields(&wgsl, "RaaFurShellGpu");
        assert_eq!(fields.len(), 4, "{fields:?}");
        assert_eq!(fields[0], ("index", "u32"), "{fields:?}");
        assert!(fields.iter().all(|(_, ty)| *ty == "u32"), "{fields:?}");
        assert_eq!(fields.len() as u64 * 4, SHELL_UNIFORM_SIZE);
        assert!(wgsl.contains("raa_fur_shell.index"), "{key:?}");
    }
}

// Reflect every entry point, including posed/custom variants. Count declared fragment
// storage conservatively (even unused bindings), excluding only vertex-only resources.
#[cfg(not(target_arch = "wasm32"))]
fn check_resources(source: &str) {
    use wgpu::naga::{AddressSpace, ShaderStage, TypeInner, front::wgsl, valid};
    let module = wgsl::parse_str(source).expect("parse resources");
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
