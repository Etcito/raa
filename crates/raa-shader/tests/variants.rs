//! Q11: the standard material function (and a function with `material_vertex`) composed for
//! every pass kind at MSAA 1 and 4 with the S11 default slot providers passes naga
//! validation, with the entry points, varyings and group-2 bindings each pass declares.

use naga::{Binding, Interpolation, Sampling, ShaderStage};
use raa_shader::{Composer, MaterialFnId, MaterialFnRegistry, PassKind, VariantKey};

/// One default provider per slot (priority 0, the S11 defaults), as raa-lit installs them.
mod default_slots {
    use std::borrow::Cow;

    use raa_shader::{SlotId, SlotProvider};

    pub const PROVIDER: &str = "default";

    /// Identity deform.
    pub const DEFORM: &str = "fn raa_slot_deform(p: vec3<f32>, n: vec3<f32>, x: InstanceExtraGpu, t: f32) -> VertexMaterialOutput {
    return VertexMaterialOutput(p, n);
}
";

    /// Sun fully visible.
    pub const SHADOW: &str =
        "fn raa_slot_shadow(world_pos: vec3<f32>, n: vec3<f32>, frag: vec4<f32>) -> f32 {
    return 1.0;
}
";

    /// No screen AO.
    pub const AO: &str = "fn raa_slot_ao(frag: vec4<f32>) -> f32 {
    return 1.0;
}
";

    /// S1 PbrLite ambient term.
    pub const AMBIENT: &str = "fn raa_slot_ambient(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    return raa_frame.ambient.rgb * s.base_color * (1.0 - 0.5 * s.metallic);
}
";

    /// S1 Lambert + GGX specular (Schlick Fresnel from c.f0, Smith-Schlick G).
    pub const DIRECT: &str = "fn raa_slot_direct(s: MaterialOutput, c: ShadeCtx, l: vec3<f32>, radiance: vec3<f32>) -> vec3<f32> {
    let n = s.normal;
    let v = c.view_dir;
    let h = normalize(l + v);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = c.n_dot_v;
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);
    let alpha = s.roughness * s.roughness;
    let alpha2 = alpha * alpha;
    let d_den = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    let d = alpha2 / max(3.14159265 * d_den * d_den, 1e-6);
    let fresnel = c.f0 + (vec3<f32>(1.0) - c.f0) * pow(1.0 - v_dot_h, 5.0);
    let k = (s.roughness + 1.0) * (s.roughness + 1.0) / 8.0;
    let g_v = n_dot_v / max(n_dot_v * (1.0 - k) + k, 1e-6);
    let g_l = n_dot_l / max(n_dot_l * (1.0 - k) + k, 1e-6);
    let specular = d * g_v * g_l * fresnel / max(4.0 * n_dot_v * n_dot_l, 1e-6);
    let diffuse = s.base_color / 3.14159265 * (1.0 - s.metallic);
    return (diffuse + specular) * radiance * n_dot_l;
}
";

    /// Nothing transmitted.
    pub const TRANSMISSION: &str =
        "fn raa_slot_transmission(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    return vec3<f32>(0.0);
}
";

    /// No fog.
    pub const FOG: &str =
        "fn raa_slot_fog(color: vec3<f32>, world_pos: vec3<f32>, c: ShadeCtx) -> vec3<f32> {
    return color;
}
";

    /// The default provider of `slot`.
    pub fn provider(slot: SlotId) -> SlotProvider {
        let wgsl = match slot {
            SlotId::Deform => DEFORM,
            SlotId::Shadow => SHADOW,
            SlotId::Ao => AO,
            SlotId::Ambient => AMBIENT,
            SlotId::Direct => DIRECT,
            SlotId::Transmission => TRANSMISSION,
            SlotId::Fog => FOG,
        };
        SlotProvider {
            slot,
            priority: 0,
            provider: PROVIDER,
            wgsl: Cow::Borrowed(wgsl),
        }
    }

    /// One default provider per slot, in [`SlotId::ALL`] order.
    pub fn all() -> Vec<SlotProvider> {
        SlotId::ALL.iter().map(|&slot| provider(slot)).collect()
    }
}

/// A material function that displaces along the normal in `material_vertex`.
const WOBBLE: &str = "fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    o.base_color = o.base_color * in.color.rgb;
    return o;
}

fn material_vertex(in: VertexMaterialInput) -> VertexMaterialOutput {
    let s = sin(in.time + in.local_position.y * 4.0) * in.instance_params[0].x;
    return VertexMaterialOutput(in.local_position + in.normal * s, in.normal);
}
";

fn default_composer() -> Composer {
    let mut composer = Composer::new();
    for p in default_slots::all() {
        if let Err(e) = composer.add_slot(p) {
            panic!("{e}");
        }
    }
    composer
}

/// Composes and validates `material_fn` for every pass kind at MSAA 1 and 4; returns the
/// parsed modules with their pass.
fn compose_all(
    reg: &MaterialFnRegistry,
    material_fn: MaterialFnId,
) -> Vec<(PassKind, naga::Module)> {
    let composer = default_composer();
    let mut modules = Vec::new();
    for pass in PassKind::ALL {
        for msaa in [1, 4] {
            let key = VariantKey {
                material_fn,
                pass,
                msaa,
                flags: 0,
            };
            let wgsl = match composer.compose(reg, &key) {
                Ok(wgsl) => wgsl,
                Err(e) => panic!("{key:?}: {e}"),
            };
            if let Err(e) = Composer::validate(&wgsl) {
                panic!("{key:?}: {e}");
            }
            assert!(
                wgsl.contains(&format!("const RAA_MSAA: u32 = {msaa}u;")),
                "{key:?}"
            );
            modules.push((pass, naga::front::wgsl::parse_str(&wgsl).expect("parse")));
        }
    }
    modules
}

fn entry<'m>(module: &'m naga::Module, name: &str) -> Option<&'m naga::EntryPoint> {
    module.entry_points.iter().find(|e| e.name == name)
}

/// Location bindings of the struct members `ty` returns (or of `ty` itself).
fn result_locations(
    module: &naga::Module,
    result: &naga::FunctionResult,
) -> Vec<(u32, naga::Handle<naga::Type>, Binding)> {
    let mut out = Vec::new();
    match &module.types[result.ty].inner {
        naga::TypeInner::Struct { members, .. } => {
            for m in members {
                if let Some(b @ Binding::Location { location, .. }) = &m.binding {
                    out.push((*location, m.ty, b.clone()));
                }
            }
        }
        _ => {
            if let Some(b @ Binding::Location { location, .. }) = &result.binding {
                out.push((*location, result.ty, b.clone()));
            }
        }
    }
    out
}

fn group_bindings(module: &naga::Module, group: u32) -> Vec<u32> {
    let mut out: Vec<u32> = module
        .global_variables
        .iter()
        .filter_map(|(_, g)| g.binding.as_ref())
        .filter(|b| b.group == group)
        .map(|b| b.binding)
        .collect();
    out.sort_unstable();
    out
}

fn check_entry_points(pass: PassKind, module: &naga::Module) {
    let vs = entry(module, "vs_main").unwrap_or_else(|| panic!("{pass:?}: no vs_main"));
    assert_eq!(vs.stage, ShaderStage::Vertex, "{pass:?}");
    match (pass, entry(module, "fs_main")) {
        (PassKind::Shadow, fs) => {
            assert!(fs.is_none(), "Shadow has no fragment stage");
            assert_eq!(module.entry_points.len(), 1, "{pass:?}");
        }
        (_, Some(fs)) => {
            assert_eq!(fs.stage, ShaderStage::Fragment, "{pass:?}");
            assert_eq!(module.entry_points.len(), 2, "{pass:?}");
        }
        (_, None) => panic!("{pass:?}: no fs_main"),
    }
}

#[test]
fn default_providers_cover_every_slot_once() {
    let providers = default_slots::all();
    assert_eq!(providers.len(), raa_shader::SlotId::ALL.len());
    for (p, slot) in providers.iter().zip(raa_shader::SlotId::ALL) {
        assert_eq!(p.slot, slot);
        assert_eq!(p.priority, 0);
        assert_eq!(p.provider, default_slots::PROVIDER);
        assert!(
            p.wgsl.contains(&format!("fn {}(", slot.fn_name())),
            "{slot:?}"
        );
    }
    let composer = default_composer();
    for slot in raa_shader::SlotId::ALL {
        let w = composer.winner(slot).expect("winner");
        assert_eq!((w.slot, w.provider), (slot, default_slots::PROVIDER));
    }
}

#[test]
fn standard_fn_every_pass_and_msaa_validates() {
    let reg = MaterialFnRegistry::new();
    let modules = compose_all(&reg, MaterialFnId::STANDARD);
    assert_eq!(modules.len(), PassKind::ALL.len() * 2);
    for (pass, module) in &modules {
        check_entry_points(*pass, module);
    }
}

#[test]
fn vertex_fn_every_pass_and_msaa_validates() {
    let mut reg = MaterialFnRegistry::new();
    let id = reg
        .register_builtin("raa.test_wobble", WOBBLE)
        .expect("register");
    assert!(reg.get(id).expect("entry").has_vertex);
    for (pass, module) in compose_all(&reg, id) {
        check_entry_points(pass, &module);
    }
}

#[test]
fn flagged_variants_validate() {
    let reg = MaterialFnRegistry::new();
    let composer = default_composer();
    for pass in PassKind::ALL {
        for flags in [8, 64, 8 | 64 | 32] {
            let key = VariantKey {
                material_fn: MaterialFnId::STANDARD,
                pass,
                msaa: 4,
                flags,
            };
            let wgsl = composer
                .compose(&reg, &key)
                .unwrap_or_else(|e| panic!("{key:?}: {e}"));
            assert!(wgsl.contains(&format!("const RAA_VARIANT_FLAGS: u32 = {flags}u;")));
            if let Err(e) = Composer::validate(&wgsl) {
                panic!("{key:?}: {e}");
            }
        }
    }
}

#[test]
fn integer_varyings_are_flat_either() {
    let reg = MaterialFnRegistry::new();
    for (pass, module) in compose_all(&reg, MaterialFnId::STANDARD) {
        let vs = entry(&module, "vs_main").expect("vs_main");
        let result = vs.function.result.as_ref().expect("vs result");
        let mut integers = 0;
        for (location, ty, binding) in result_locations(&module, result) {
            let is_integer = matches!(
                module.types[ty].inner,
                naga::TypeInner::Scalar(naga::Scalar {
                    kind: naga::ScalarKind::Uint | naga::ScalarKind::Sint,
                    ..
                })
            );
            if is_integer {
                integers += 1;
                assert!(
                    matches!(
                        binding,
                        Binding::Location {
                            interpolation: Some(Interpolation::Flat),
                            sampling: Some(Sampling::Either),
                            ..
                        }
                    ),
                    "{pass:?} location {location}: {binding:?}"
                );
            }
        }
        // material and instance indices; the shadow pass has no varyings.
        let expected = if pass == PassKind::Shadow { 0 } else { 2 };
        assert_eq!(integers, expected, "{pass:?}");
    }
}

#[test]
fn fragment_targets_follow_the_pass() {
    let reg = MaterialFnRegistry::new();
    let vec4_f32 = naga::TypeInner::Vector {
        size: naga::VectorSize::Quad,
        scalar: naga::Scalar::F32,
    };
    for (pass, module) in compose_all(&reg, MaterialFnId::STANDARD) {
        let Some(fs) = entry(&module, "fs_main") else {
            assert_eq!(pass, PassKind::Shadow);
            continue;
        };
        let targets: Vec<u32> = match &fs.function.result {
            Some(result) => result_locations(&module, result)
                .into_iter()
                .map(|(location, ty, _)| {
                    assert_eq!(module.types[ty].inner, vec4_f32, "{pass:?}");
                    location
                })
                .collect(),
            None => Vec::new(),
        };
        let expected: &[u32] = match pass {
            // S12 normal and material targets.
            PassKind::Prepass => &[0, 1],
            // HDR colour (alpha = coverage for the fur shells).
            PassKind::Forward | PassKind::FurShell => &[0],
            // Writes only the OIT storage buffers.
            PassKind::Transparent => &[],
            PassKind::Shadow => unreachable!(),
        };
        assert_eq!(targets, expected, "{pass:?}");
    }
}

#[test]
fn group_two_bindings_are_private_to_their_pass() {
    let reg = MaterialFnRegistry::new();
    for (pass, module) in compose_all(&reg, MaterialFnId::STANDARD) {
        assert_eq!(group_bindings(&module, 0), [0, 1, 2], "{pass:?}");
        assert_eq!(
            group_bindings(&module, 1),
            (0..12).collect::<Vec<u32>>(),
            "{pass:?}"
        );
        let expected: &[u32] = match pass {
            // heads, nodes, counter, OitParams.
            PassKind::Transparent => &[0, 1, 2, 3],
            // Caster view-projection / shell index.
            PassKind::Shadow | PassKind::FurShell => &[0],
            PassKind::Prepass | PassKind::Forward => &[],
        };
        assert_eq!(group_bindings(&module, 2), expected, "{pass:?}");
        assert!(group_bindings(&module, 3).is_empty(), "{pass:?}");
    }
}
