//! Q11/S10: the OIT clear and resolve shaders and the Transparent pass variant of the
//! standard material function pass naga validation.
//!
//! The composed neutral (physical/cloud disabled) and physical/cloud variants over the
//! default, ibl, CSM cloud-shadow and height-fog providers keep the S10 Transparent stage
//! budget (12 sampled textures / 3 samplers / 8 uniform buffers / 8 fragment storage
//! buffers), the four-group maximum, the vertex-only history/palette resources, the
//! unchanged 16-byte `OitNode` and exactly three OIT fragment storage allocations.

// `default_slot_providers` is crate-private to raa-lit (used by `LitRenderer::new`); the test
// compiles the same source file, as raa-lit's own variants test does. The ibl, csm and fog
// providers are crate-private too; the test compiles their source files (all crates are only
// read) and exercises their fixture interfaces directly below, so they need no lint
// suppression.
#[path = "../../raa-fog/src/shaders.rs"]
mod fog_shaders;
#[path = "../../raa-ibl/src/provider.rs"]
mod ibl_provider;
#[path = "../../raa-shadow/src/provider.rs"]
mod shadow_provider;
#[allow(dead_code)]
#[path = "../../raa-lit/src/slots.rs"]
mod slots;

use raa_oit::{CLEAR_ENTRY, CLEAR_WGSL, RESOLVE_FS_ENTRY, RESOLVE_VS_ENTRY, RESOLVE_WGSL};
use raa_shader::{
    Composer, MaterialFnId, MaterialFnRegistry, PassKind, SlotId, SlotProvider, VariantKey,
};

fn build(extra: impl IntoIterator<Item = SlotProvider>) -> Composer {
    let mut composer = Composer::new();
    for p in slots::default_slot_providers().into_iter().chain(extra) {
        if let Err(e) = composer.add_slot(p) {
            panic!("{e}");
        }
    }
    composer
}

fn default_composer() -> Composer {
    build([])
}

/// The weather composition: default + ibl + the CSM cloud-shadow boundary + the height fog.
/// Physical and cloud enablement are runtime uniform states of the same neutral S10
/// declarations, so this composition carries both enabled and disabled paths.
fn weather_composer() -> Composer {
    build([
        ibl_provider::ambient_provider(),
        shadow_provider::csm_provider(),
        fog_shaders::slot_provider(),
    ])
}

/// The compiled fixture interfaces are the consumed S11 contract: provider names,
/// priorities and WGSL hooks the composed Transparent variants rely on.
#[test]
fn weather_provider_fixtures_match_the_consumed_contract() {
    assert_eq!(fog_shaders::HEIGHT_PROVIDER, "height");
    assert_eq!(fog_shaders::HEIGHT_PRIORITY, 10);
    assert!(fog_shaders::FOG_SLOT_WGSL.contains("fn raa_slot_fog("));
    assert!(fog_shaders::FOG_PASS_WGSL.contains("fn fs_main("));
    let fog_pass = fog_shaders::fog_pass_source();
    assert!(fog_pass.contains(fog_shaders::FOG_SLOT_WGSL));
    assert!(fog_pass.contains(fog_shaders::FOG_PASS_WGSL));
    assert_eq!(ibl_provider::IBL_PROVIDER, "ibl");
    assert_eq!(ibl_provider::IBL_PRIORITY, 10);
    assert!(ibl_provider::IBL_LIB_SOURCE.contains("fn raa_ibl_irradiance("));
    assert!(ibl_provider::AMBIENT_WGSL.contains("fn raa_slot_ambient("));
    assert_eq!(shadow_provider::CSM_PROVIDER, "csm");
    assert_eq!(shadow_provider::CSM_PRIORITY, 10);
    let slot = shadow_provider::csm_provider();
    assert_eq!(slot.wgsl, shadow_provider::CSM_WGSL);
    assert!(shadow_provider::CSM_WGSL.contains("fn raa_slot_shadow("));
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
fn weather_transparent_variants_stay_within_budgets() {
    let mut reg = MaterialFnRegistry::new();
    let glass = reg
        .register_builtin(
            "raa.glass",
            include_str!("../../raa-transmission/src/wgsl/glass.wgsl"),
        )
        .expect("glass material");
    let custom = reg
        .register_user(
            "test.transparent.weather",
            "fn material_main(in: MaterialInput) -> MaterialOutput { return raa_default_output(in); }",
            &weather_composer(),
        )
        .expect("custom material");
    let mut count = 0;
    for (name, composer) in [
        ("neutral (physical/cloud disabled)", default_composer()),
        ("physical/cloud (csm + fog)", weather_composer()),
    ] {
        let csm = composer.winner(SlotId::Shadow).expect("winner").provider == "csm";
        let fog = composer.winner(SlotId::Fog).expect("winner").provider;
        assert_eq!(fog, if csm { "height" } else { "raa.default" });
        for material_fn in [MaterialFnId::STANDARD, glass, custom] {
            for flags in [0, 8, 64] {
                for msaa in [1, 4] {
                    let key = VariantKey {
                        material_fn,
                        pass: PassKind::Transparent,
                        msaa,
                        flags,
                    };
                    let wgsl = composer
                        .compose(&reg, &key)
                        .unwrap_or_else(|e| panic!("{name} {key:?}: {e}"));
                    assert_weather_boundary(&wgsl, name, csm);
                    assert!(wgsl.contains("raa_oit_heads"), "{name} {key:?}");
                    if let Err(e) = Composer::validate(&wgsl) {
                        panic!("{name} {key:?}: {e}");
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    check_resources(&wgsl);
                    let posed = raa_lit::skin_draw::posed_module(&wgsl).expect("posed transparent");
                    Composer::validate(&posed).expect("posed validation");
                    #[cfg(not(target_arch = "wasm32"))]
                    check_resources(&posed);
                    count += 1;
                }
            }
        }
        println!("Transparent OIT keeps the S10 budgets with {name} slots");
    }
    assert_eq!(count, 36);
}

/// The S10 weather declarations are neutral for both states: physical disabled returns the
/// raw radiance and enabled multiplies by the transmission once; clouds disabled return 1
/// and enabled sample the two-layer ground map. The CSM provider consumes the common cloud
/// helper exactly once in each of its result paths.
fn assert_weather_boundary(wgsl: &str, label: &str, csm: bool) {
    for decl in [
        "@group(1) @binding(14)",
        "@group(1) @binding(15)",
        "@group(1) @binding(16)",
        "@group(1) @binding(17)",
        "@group(1) @binding(18)",
    ] {
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
        "{label}: one directional boundary call in Transparent"
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

/// S10 Transparent conservative fragment budget: sampled textures, samplers, uniform buffers
/// and fragment storage buffers, including the declared shared weather bindings.
#[cfg(not(target_arch = "wasm32"))]
const TRANSPARENT_FRAGMENT_BUDGET: Counts = Counts {
    textures: 12,
    samplers: 3,
    uniforms: 8,
    storages: 8,
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
    let mut oit_storage = 0u32;
    let mut history = None;
    let mut palette = None;
    let mut cloud = None;
    let mut weather = None;
    let mut oit = Vec::new();
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
            (1, 14) => {
                assert_eq!(global.name.as_deref(), Some("raa_gi_screen"));
                assert_eq!(global.space, AddressSpace::Handle);
                assert_eq!(
                    module.types[global.ty].inner,
                    TypeInner::Image {
                        dim: ImageDimension::D2,
                        arrayed: false,
                        class: ImageClass::Sampled {
                            kind: ScalarKind::Float,
                            multi: false,
                        },
                    }
                );
            }
            (1, 17) => {
                assert_eq!(global.name.as_deref(), Some("raa_gi_volume"));
                assert_eq!(global.space, AddressSpace::Handle);
                assert_eq!(
                    module.types[global.ty].inner,
                    TypeInner::Image {
                        dim: ImageDimension::D3,
                        arrayed: false,
                        class: ImageClass::Sampled {
                            kind: ScalarKind::Float,
                            multi: false,
                        },
                    }
                );
            }
            (1, 18) => {
                assert_eq!(global.name.as_deref(), Some("raa_gi_volume_params"));
                assert_eq!(global.space, AddressSpace::Uniform);
                let TypeInner::Struct { span, .. } = &module.types[global.ty].inner else {
                    panic!("gi volume params record")
                };
                assert_eq!(*span as usize, 96, "RaaGiVolumeParams is 96 B");
            }
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
            (2, 0..=2) => {
                let name = match binding.binding {
                    0 => "raa_oit_heads",
                    1 => "raa_oit_nodes",
                    _ => "raa_oit_counter",
                };
                assert_eq!(global.name.as_deref(), Some(name));
                assert!(matches!(global.space, AddressSpace::Storage { .. }));
                oit_storage += 1;
                oit.push((*handle, name));
            }
            (2, 3) => {
                assert_eq!(global.name.as_deref(), Some("raa_oit_params"));
                assert_eq!(global.space, AddressSpace::Uniform);
                let TypeInner::Struct { members, span } = &module.types[global.ty].inner else {
                    panic!("OitParams")
                };
                assert_eq!(*span as usize, 16, "OitParams is 16 B");
                for (member, offset) in members.iter().zip([0, 4, 8, 12]) {
                    assert_eq!(member.offset, offset);
                }
            }
            (2, _) => panic!("unexpected OIT binding {}", binding.binding),
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
    // History is declared for the vertex stage only and the posed palette adds exactly the
    // seventh vertex storage buffer to the six group-0 storage buffers.
    assert!(history.is_some(), "history binding declared");
    assert_eq!(vertex_storage, 6, "six group-0 vertex storage buffers");
    let total_vertex_storage = vertex_storage + u32::from(palette.is_some());
    assert_eq!(
        total_vertex_storage,
        if palette.is_some() { 7 } else { 6 },
        "posed adds only the palette vertex storage buffer"
    );
    assert!(total_vertex_storage <= limits.max_storage_buffers_per_shader_stage);
    assert_eq!(
        fragment, TRANSPARENT_FRAGMENT_BUDGET,
        "S10 Transparent budget"
    );
    assert!(fragment.textures <= limits.max_sampled_textures_per_shader_stage);
    assert!(fragment.samplers <= limits.max_samplers_per_shader_stage);
    assert!(fragment.uniforms <= limits.max_uniform_buffers_per_shader_stage);
    assert!(fragment.storages <= limits.max_storage_buffers_per_shader_stage);

    // Exactly the three OIT lists, and the unchanged 16-byte list node.
    assert_eq!(oit_storage, 3, "three OIT fragment storage allocations");
    let node = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("OitNode"))
        .unwrap()
        .1;
    let TypeInner::Struct { members, span } = &node.inner else {
        panic!("OitNode")
    };
    assert_eq!(*span as usize, 16, "OitNode stays 16 B");
    assert_eq!(members.len(), 4);
    for (member, offset) in members.iter().zip([0, 4, 8, 12]) {
        assert_eq!(member.offset, offset);
    }

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
            for (handle, name) in &oit {
                assert!(used[1][handle.index()], "fragment appends to {name}");
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
