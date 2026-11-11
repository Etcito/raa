//! Q11/S10: the FurShell variant of `raa.plush` composes and passes naga validation at MSAA
//! 1/4 with the flags {FUR, FUR | DOUBLE_SIDED}, over raa-lit's default slot providers,
//! raa-ibl's `ibl` Ambient provider, raa-brdf's `layered` providers, raa-shadow's `csm`
//! cloud-shadow boundary and raa-fog's `height` provider; the composed module declares the
//! group-2 `raa_fur_shell` uniform with the 16-byte layout the contributor writes.
//!
//! naga reflection pins the S10 FurShell stage budgets (10 sampled textures / 3 samplers /
//! 7 uniform buffers / 5 fragment storage buffers, six or seven vertex storage buffers) for
//! the neutral (physical/cloud disabled) and the full physical/cloud composition, including
//! the vertex-only history and posed palette resources and the four-group maximum.

// The default, ibl, csm and fog providers are crate-private; the test compiles their source
// files (all crates are only read). The fixture interfaces are exercised directly below, so
// the included modules need no lint suppression.
#[path = "../../raa-fog/src/shaders.rs"]
mod fog_shaders;
#[allow(dead_code)]
#[path = "../../raa-ibl/src/provider.rs"]
mod ibl_provider;
#[allow(dead_code)]
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;
#[path = "../../raa-shadow/src/provider.rs"]
mod shadow_provider;

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

/// The weather composition: default + ibl + layered + the CSM cloud-shadow boundary +
/// the height fog. Physical and cloud enablement are runtime uniform states of the same
/// neutral S10 declarations, so this composition carries both enabled and disabled paths;
/// it is the slot set an engine with every weather feature installed.
fn weather_composer() -> Composer {
    composer(
        [
            ibl_provider::ambient_provider(),
            shadow_provider::csm_provider(),
            fog_shaders::slot_provider(),
        ]
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
fn weather_providers_win_their_slots() {
    let composer = weather_composer();
    for slot in SlotId::ALL {
        let w = composer.winner(slot).expect("winner");
        let expected = match slot {
            SlotId::Shadow => ("csm", 10),
            SlotId::Fog => ("height", 10),
            SlotId::Ambient | SlotId::Direct => ("layered", 20),
            _ => ("raa.default", 0),
        };
        assert_eq!((w.provider, w.priority), expected, "{slot:?}");
    }
}

/// The compiled fixture interfaces are the consumed S11 contract: provider names,
/// priorities and WGSL hooks the composed FurShell variants rely on.
#[test]
fn weather_provider_fixtures_match_the_consumed_contract() {
    assert_eq!(fog_shaders::HEIGHT_PROVIDER, "height");
    assert_eq!(fog_shaders::HEIGHT_PRIORITY, 10);
    assert!(fog_shaders::FOG_SLOT_WGSL.contains("fn raa_slot_fog("));
    assert!(fog_shaders::FOG_PASS_WGSL.contains("fn fs_main("));
    let fog_pass = fog_shaders::fog_pass_source();
    assert!(fog_pass.contains(fog_shaders::FOG_SLOT_WGSL));
    assert!(fog_pass.contains(fog_shaders::FOG_PASS_WGSL));
    assert_eq!(shadow_provider::CSM_PROVIDER, "csm");
    assert_eq!(shadow_provider::CSM_PRIORITY, 10);
    let slot = shadow_provider::csm_provider();
    assert_eq!(slot.wgsl, shadow_provider::CSM_WGSL);
    assert!(shadow_provider::CSM_WGSL.contains("fn raa_slot_shadow("));
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
fn weather_composition_reflects_stage_budgets() {
    let mut reg = registry();
    let custom = reg
        .register_user(
            "test.fur.weather",
            "fn material_main(in: MaterialInput) -> MaterialOutput { return raa_default_output(in); }",
            &weather_composer(),
        )
        .expect("custom material");
    let mut keys = plush_furshell_keys(&reg);
    keys.extend(plush_furshell_keys(&reg).into_iter().map(|key| VariantKey {
        material_fn: custom,
        ..key
    }));
    assert_eq!(keys.len(), 8);
    let mut count = 0;
    for (name, composer) in [
        ("neutral (physical/cloud disabled)", composer([])),
        ("physical/cloud (csm + fog)", weather_composer()),
    ] {
        let csm = composer.winner(SlotId::Shadow).expect("winner").provider == "csm";
        for key in &keys {
            let wgsl = compose(&composer, &reg, key);
            assert_weather_boundary(&wgsl, name, csm);
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
        println!("raa.plush FurShell keeps the S10 budgets with {name} slots");
    }
    assert_eq!(count, 16);
}

/// The S10 weather declarations are neutral for both states: physical disabled returns the
/// raw radiance and enabled multiplies by the transmission once; clouds disabled return 1
/// and enabled sample the two-layer ground map. The CSM provider consumes the common cloud
/// helper exactly once in each of its result paths.
fn assert_weather_boundary(wgsl: &str, label: &str, csm: bool) {
    for decl in ["@group(1) @binding(15)", "@group(1) @binding(16)"] {
        assert_eq!(wgsl.matches(decl).count(), 1, "{label}: {decl}");
    }
    assert_eq!(wgsl.matches("fn raa_cloud_shadow(").count(), 1, "{label}");
    assert_eq!(
        wgsl.matches("fn raa_directional_radiance(").count(),
        1,
        "{label}"
    );
    assert_eq!(
        wgsl.matches("raa_directional_radiance(raa_frame.sun_color.rgb)")
            .count(),
        1,
        "{label}: one directional boundary call in FurShell"
    );
    // Physical disabled: pass raw radiance through. Enabled: apply transmission once.
    assert!(
        wgsl.contains("if (raa_weather_lighting.flags.x == 0u) {\n        return raw;\n    }"),
        "{label}"
    );
    assert!(
        wgsl.contains("return raw * raa_weather_lighting.atmosphericTransmission.rgb;"),
        "{label}"
    );
    // Clouds disabled: neutral transmittance. Enabled: sample the two-layer ground map.
    assert!(
        wgsl.contains("if raa_weather_lighting.map.w < 0.5 {\n        return 1.0;\n    }"),
        "{label}"
    );
    assert!(
        wgsl.contains("textureLoad(raa_cloud_transmittance, vec2<i32>(0, 0), 1, 0)"),
        "{label}"
    );
    let calls = wgsl.matches("raa_cloud_shadow(world_pos)").count();
    assert_eq!(
        calls,
        if csm { 2 } else { 1 },
        "{label}: shadow result paths"
    );
    if csm {
        assert!(
            wgsl.contains("return contact * raa_cloud_shadow(world_pos);"),
            "{label}"
        );
        assert!(
            wgsl.contains("return vis * contact * raa_cloud_shadow(world_pos);"),
            "{label}"
        );
    }
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

/// S10 FurShell conservative fragment budget: sampled textures, samplers, uniform buffers
/// and fragment storage buffers, including the declared shared weather bindings.
#[cfg(not(target_arch = "wasm32"))]
const FURSHELL_FRAGMENT_BUDGET: Counts = Counts {
    textures: 10,
    samplers: 3,
    uniforms: 7,
    storages: 5,
    storage_textures: 0,
};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counts {
    textures: u32,
    samplers: u32,
    uniforms: u32,
    storages: u32,
    storage_textures: u32,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
enum ResourceKind {
    Texture,
    Sampler,
    Uniform,
    Storage,
    StorageTexture,
}

#[cfg(not(target_arch = "wasm32"))]
impl Counts {
    fn add(&mut self, kind: ResourceKind) {
        match kind {
            ResourceKind::Texture => self.textures += 1,
            ResourceKind::Sampler => self.samplers += 1,
            ResourceKind::Uniform => self.uniforms += 1,
            ResourceKind::Storage => self.storages += 1,
            ResourceKind::StorageTexture => self.storage_textures += 1,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn resource_kind(
    module: &wgpu::naga::Module,
    global: &wgpu::naga::GlobalVariable,
) -> Option<ResourceKind> {
    use wgpu::naga::{AddressSpace, ImageClass, TypeInner};
    match global.space {
        AddressSpace::Uniform => Some(ResourceKind::Uniform),
        AddressSpace::Storage { .. } => Some(ResourceKind::Storage),
        AddressSpace::Handle => match &module.types[global.ty].inner {
            TypeInner::Image {
                class: ImageClass::Storage { .. },
                ..
            } => Some(ResourceKind::StorageTexture),
            TypeInner::Image { .. } => Some(ResourceKind::Texture),
            TypeInner::Sampler { .. } => Some(ResourceKind::Sampler),
            _ => None,
        },
        _ => None,
    }
}

// Reflect every entry point, including posed/custom variants. Count declared fragment
// storage conservatively (even unused bindings), excluding only vertex-only resources.
#[cfg(not(target_arch = "wasm32"))]
fn check_resources(source: &str) {
    use wgpu::naga::{
        AddressSpace, ImageClass, ImageDimension, ScalarKind, ShaderStage, TypeInner, front::wgsl,
        valid,
    };
    let module = wgsl::parse_str(source).expect("parse resources");
    let info = valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
        .validate(&module)
        .expect("validate resources");
    let limits = wgpu::Limits::default();
    let globals: Vec<_> = module.global_variables.iter().collect();
    let mut fragment = Counts::default();
    let mut vertex_storage = 0u32;
    let mut group2 = 0u32;
    let mut history = None;
    let mut palette = None;
    let mut cloud = None;
    let mut weather = None;
    for (handle, global) in &globals {
        let Some(binding) = &global.binding else {
            continue;
        };
        assert!(
            binding.group < limits.max_bind_groups,
            "at most four bind groups"
        );
        match (binding.group, binding.binding) {
            (0, 8) => {
                history = Some(*handle);
                assert_eq!(global.name.as_deref(), Some("raa_previous_instances"));
            }
            (1, 14) => panic!("binding 14 stays reserved"),
            (1, 15) => {
                cloud = Some(*handle);
                assert_eq!(global.name.as_deref(), Some("raa_cloud_transmittance"));
                assert_eq!(global.space, AddressSpace::Handle);
                assert_eq!(
                    module.types[global.ty].inner,
                    TypeInner::Image {
                        dim: ImageDimension::D2,
                        arrayed: true,
                        class: ImageClass::Sampled {
                            kind: ScalarKind::Float,
                            multi: false,
                        },
                    }
                );
            }
            (1, 16) => {
                weather = Some(*handle);
                assert_eq!(global.name.as_deref(), Some("raa_weather_lighting"));
                assert_eq!(global.space, AddressSpace::Uniform);
                let TypeInner::Struct { span, .. } = &module.types[global.ty].inner else {
                    panic!("weather record")
                };
                assert_eq!(*span as usize, 64, "WeatherLightingGpu is 64 B");
            }
            (2, _) => {
                group2 += 1;
                assert_eq!(binding.binding, 0, "one group-2 fur shell uniform");
                assert_eq!(global.space, AddressSpace::Uniform);
                let TypeInner::Struct { span, .. } = &module.types[global.ty].inner else {
                    panic!("fur shell record")
                };
                assert_eq!(*span as usize, 16, "RaaFurShellGpu is 16 B");
            }
            (3, _) => {
                assert_eq!(binding.binding, 0, "one palette binding");
                assert!(palette.is_none(), "one palette binding");
                palette = Some(*handle);
                assert_eq!(global.name.as_deref(), Some("raa_skin_palette"));
                assert!(matches!(global.space, AddressSpace::Storage { .. }));
            }
            _ => {}
        }
        let vertex_only = (binding.group, binding.binding) == (0, 8) || binding.group == 3;
        if binding.group == 0 && matches!(global.space, AddressSpace::Storage { .. }) {
            vertex_storage += 1;
        }
        if vertex_only {
            continue;
        }
        if let Some(kind) = resource_kind(&module, global) {
            fragment.add(kind);
        }
        for (index, entry) in module.entry_points.iter().enumerate() {
            if entry.stage == ShaderStage::Fragment && vertex_only {
                assert!(
                    info.get_entry_point(index)[*handle].is_empty(),
                    "vertex-only resource"
                );
            }
            if entry.stage == ShaderStage::Fragment
                && binding.group == 1
                && [12, 13].contains(&binding.binding)
            {
                assert!(
                    !info.get_entry_point(index)[*handle].is_empty(),
                    "lights reached by fragment"
                );
            }
        }
    }
    // History is declared for the vertex stage only (the fur pass needs no motion input),
    // and the posed palette adds exactly the seventh vertex storage buffer to the six
    // group-0 storage buffers; the fragment budget itself never grows.
    assert!(history.is_some(), "history binding declared");
    assert_eq!(group2, 1, "the fur shell uniform");
    assert_eq!(vertex_storage, 6, "six group-0 vertex storage buffers");
    let total_vertex_storage = vertex_storage + u32::from(palette.is_some());
    assert_eq!(
        total_vertex_storage,
        if palette.is_some() { 7 } else { 6 },
        "posed adds only the palette vertex storage buffer"
    );
    assert!(total_vertex_storage <= limits.max_storage_buffers_per_shader_stage);
    assert_eq!(
        fragment, FURSHELL_FRAGMENT_BUDGET,
        "S10 FurShell fragment budget"
    );
    assert!(fragment.textures <= limits.max_sampled_textures_per_shader_stage);
    assert!(fragment.samplers <= limits.max_samplers_per_shader_stage);
    assert!(fragment.uniforms <= limits.max_uniform_buffers_per_shader_stage);
    assert!(fragment.storages <= limits.max_storage_buffers_per_shader_stage);
    assert!(fragment.storage_textures <= limits.max_storage_textures_per_shader_stage);

    // Actual per-stage reachability, reflected from every entry point's global uses.
    let mut used: [Vec<bool>; 2] = [vec![false; globals.len()], vec![false; globals.len()]];
    for (index, entry) in module.entry_points.iter().enumerate() {
        let stage = match entry.stage {
            ShaderStage::Vertex => 0,
            ShaderStage::Fragment => 1,
            _ => continue,
        };
        let point = info.get_entry_point(index);
        for (handle, _) in &globals {
            if !point[*handle].is_empty() {
                used[stage][handle.index()] = true;
            }
        }
        if entry.stage == ShaderStage::Fragment {
            for handle in [history, palette].into_iter().flatten() {
                assert!(
                    !used[1][handle.index()],
                    "history/palette are vertex-only resources"
                );
            }
            for handle in [cloud, weather].into_iter().flatten() {
                assert!(
                    used[1][handle.index()],
                    "the fragment stage reaches the shared weather bindings"
                );
            }
        } else if let Some(handle) = palette {
            assert!(
                used[0][handle.index()],
                "the posed vertex stage reaches the palette"
            );
        }
    }
    let mut vertex_used = Counts::default();
    let mut fragment_used = Counts::default();
    for (handle, global) in &globals {
        let Some(_) = &global.binding else {
            continue;
        };
        if let Some(kind) = resource_kind(&module, global) {
            if used[0][handle.index()] {
                vertex_used.add(kind);
            }
            if used[1][handle.index()] {
                fragment_used.add(kind);
            }
        }
    }
    assert!(fragment_used.textures <= limits.max_sampled_textures_per_shader_stage);
    assert!(fragment_used.samplers <= limits.max_samplers_per_shader_stage);
    assert!(fragment_used.uniforms <= limits.max_uniform_buffers_per_shader_stage);
    assert!(fragment_used.storages <= limits.max_storage_buffers_per_shader_stage);
    assert!(fragment_used.textures <= fragment.textures);
    assert!(fragment_used.samplers <= fragment.samplers);
    assert!(fragment_used.uniforms <= fragment.uniforms);
    assert!(fragment_used.storages <= fragment.storages);
    assert!(vertex_used.textures <= limits.max_sampled_textures_per_shader_stage);
    assert!(vertex_used.samplers <= limits.max_samplers_per_shader_stage);
    assert!(vertex_used.uniforms <= limits.max_uniform_buffers_per_shader_stage);
    assert!(vertex_used.storages <= limits.max_storage_buffers_per_shader_stage);

    // The shared cloud helper consumes the declared weather bindings and the existing
    // linear clamp sampler; the neutral record defaults to physical/cloud disabled.
    let (function, _) = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("raa_cloud_shadow"))
        .expect("cloud helper");
    let uses = &info[function];
    for (handle, global) in &globals {
        let Some(binding) = &global.binding else {
            continue;
        };
        if binding.group == 1 && [6, 15, 16].contains(&binding.binding) {
            assert!(
                !uses[*handle].is_empty(),
                "raa_cloud_shadow consumes binding {}",
                binding.binding
            );
        }
    }
    let (_, neutral) = module
        .constants
        .iter()
        .find(|(_, c)| c.name.as_deref() == Some("RAA_WEATHER_LIGHTING_DEFAULT"))
        .expect("neutral weather record");
    let wgpu::naga::Expression::Compose { components, .. } =
        &module.global_expressions[neutral.init]
    else {
        panic!("neutral record constructor")
    };
    let mut lanes = Vec::new();
    for lane in components {
        let wgpu::naga::Expression::Compose { components, .. } = &module.global_expressions[*lane]
        else {
            panic!("neutral lane constructor")
        };
        let mut values = Vec::new();
        for value in components {
            values.push(match module.global_expressions[*value] {
                wgpu::naga::Expression::Literal(wgpu::naga::Literal::F32(v)) => v,
                wgpu::naga::Expression::Literal(wgpu::naga::Literal::U32(v)) => v as f32,
                _ => panic!("neutral literal"),
            });
        }
        lanes.push(values);
    }
    assert_eq!(
        lanes,
        [
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0],
        ],
        "physical/cloud disabled defaults, transmission one"
    );
}
