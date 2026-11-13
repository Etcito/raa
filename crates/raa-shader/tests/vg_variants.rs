//! S32: both virtualized-geometry pass kinds compose, validate and stay within the
//! fragment-stage budgets of the standard function, a built-in-style function with helper
//! functions and a user function, all at MSAA 1.

use naga::valid::GlobalUse;
use naga::{Binding, BuiltIn, ShaderStage};
use raa_shader::{
    ComposeOptions, Composer, MaterialFnId, MaterialFnRegistry, PassKind, TintMode, VariantKey,
};

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

    /// Local lights fully visible (no local shadows yet).
    pub const LOCAL_VISIBILITY: &str = "fn raa_slot_local_visibility(light_index: u32, world_position: vec3<f32>, surface_normal: vec3<f32>, direction_to_light: vec3<f32>, light_distance: f32) -> f32 {
    return 1.0;
}
";

    /// Additional indirect illumination is black.
    pub const INDIRECT: &str = "fn raa_slot_indirect(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    return vec3<f32>(0.0);
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
            SlotId::LocalVisibility => LOCAL_VISIBILITY,
            SlotId::Indirect => INDIRECT,
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

/// Built-in-style function: helper functions and a `material_vertex` (like WOBBLE).
const PATTERN: &str = "fn vg_pattern_wave(p: vec3<f32>, t: f32) -> f32 {
    return 0.5 + 0.5 * sin(p.y * 6.0 + t);
}

fn vg_pattern_ring(p: vec3<f32>, r: f32) -> f32 {
    return smoothstep(r, r + 0.05, length(p.xz));
}

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let w = vg_pattern_wave(in.local_position, in.time) * 0.25;
    o.base_color = o.base_color * (0.6 + 0.4 * vg_pattern_ring(in.local_position, w));
    return o;
}

fn material_vertex(in: VertexMaterialInput) -> VertexMaterialOutput {
    return VertexMaterialOutput(in.local_position, in.normal);
}
";

/// User function: no `raa_` module-scope identifier and the exact `material_main` signature.
const USER_FN: &str = "fn vg_user_stripe(p: vec3<f32>) -> f32 {
    return step(0.5, fract(p.y * 4.0));
}

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    o.base_color = mix(o.base_color, vec3<f32>(0.9, 0.4, 0.2), vg_user_stripe(in.local_position));
    return o;
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

/// The standard function, then the built-in-style and the user function, in one registry.
fn functions(composer: &Composer) -> (MaterialFnRegistry, [MaterialFnId; 3]) {
    let mut reg = MaterialFnRegistry::new();
    let pattern = reg
        .register_builtin("raa.vg_pattern", PATTERN)
        .expect("builtin");
    let user = reg
        .register_user("vg.user_pattern", USER_FN, composer)
        .expect("user");
    (reg, [MaterialFnId::STANDARD, pattern, user])
}

fn key(material_fn: MaterialFnId, pass: PassKind) -> VariantKey {
    VariantKey {
        material_fn,
        pass,
        msaa: 1,
        flags: 0,
    }
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

fn writes_frag_depth(module: &naga::Module, fragment: &naga::EntryPoint) -> bool {
    let Some(result) = &fragment.function.result else {
        return false;
    };
    match &module.types[result.ty].inner {
        naga::TypeInner::Struct { members, .. } => members
            .iter()
            .any(|m| matches!(m.binding, Some(Binding::BuiltIn(BuiltIn::FragDepth)))),
        _ => matches!(result.binding, Some(Binding::BuiltIn(BuiltIn::FragDepth))),
    }
}

#[test]
fn pass_kind_lists_split_the_mesh_and_vg_kinds() {
    assert_eq!(PassKind::ALL.len(), 5);
    assert_eq!(PassKind::VG, [PassKind::VgGbuffer, PassKind::VgShade]);
}

#[test]
fn both_vg_kinds_compose_validate_and_stay_within_budget() {
    let composer = default_composer();
    let (reg, ids) = functions(&composer);
    let vec4_f32 = naga::TypeInner::Vector {
        size: naga::VectorSize::Quad,
        scalar: naga::Scalar::F32,
    };
    let vec2_f32 = naga::TypeInner::Vector {
        size: naga::VectorSize::Bi,
        scalar: naga::Scalar::F32,
    };
    for material_fn in ids {
        for pass in PassKind::VG {
            let k = key(material_fn, pass);
            let wgsl = composer
                .compose(&reg, &k)
                .unwrap_or_else(|e| panic!("{k:?}: {e}"));
            if let Err(e) = Composer::validate(&wgsl) {
                panic!("{k:?}: {e}");
            }
            let module = naga::front::wgsl::parse_str(&wgsl).expect("parse");
            let names: Vec<&str> = module
                .entry_points
                .iter()
                .map(|entry| entry.name.as_str())
                .collect();
            assert_eq!(names, ["vs_main", "fs_main"], "{k:?}");

            let vs = entry(&module, "vs_main").expect("vs_main");
            assert_eq!(vs.stage, ShaderStage::Vertex, "{k:?}");
            for arg in &vs.function.arguments {
                assert!(
                    !matches!(arg.binding, Some(Binding::Location { .. })),
                    "{k:?}: the fullscreen vs takes no vertex input"
                );
            }

            let fs = entry(&module, "fs_main").expect("fs_main");
            assert_eq!(fs.stage, ShaderStage::Fragment, "{k:?}");
            assert!(
                writes_frag_depth(&module, fs),
                "{k:?}: fs writes frag_depth"
            );
            let targets: Vec<(u32, naga::Handle<naga::Type>, Binding)> =
                result_locations(&module, fs.function.result.as_ref().expect("fs result"));
            let locations: Vec<u32> = targets.iter().map(|(at, _, _)| *at).collect();
            match pass {
                PassKind::VgGbuffer => {
                    // S12 normal, material and motion targets.
                    assert_eq!(locations, [0, 1, 2], "{k:?}");
                    assert_eq!(module.types[targets[0].1].inner, vec4_f32, "{k:?}");
                    assert_eq!(module.types[targets[1].1].inner, vec4_f32, "{k:?}");
                    assert_eq!(module.types[targets[2].1].inner, vec2_f32, "{k:?}");
                }
                PassKind::VgShade => {
                    assert_eq!(locations, [0], "{k:?}");
                    assert_eq!(module.types[targets[0].1].inner, vec4_f32, "{k:?}");
                }
                _ => unreachable!(),
            }
            assert_eq!(group_bindings(&module, 2), [0, 1, 2, 3, 4], "{k:?}");

            let info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .expect("validate");
            let fragment = module
                .entry_points
                .iter()
                .position(|e| e.name == "fs_main")
                .expect("fs_main");
            let usage = info.get_entry_point(fragment);
            let (mut textures, mut buffers) = (0, 0);
            for (handle, var) in module.global_variables.iter() {
                if !usage[handle].contains(GlobalUse::READ) {
                    continue;
                }
                if matches!(
                    module.types[var.ty].inner,
                    naga::TypeInner::Image {
                        class: naga::ImageClass::Sampled { .. },
                        ..
                    }
                ) {
                    textures += 1;
                }
                if matches!(var.space, naga::AddressSpace::Storage { .. }) {
                    buffers += 1;
                }
            }
            assert!(textures <= 16, "{k:?}: {textures} sampled textures");
            assert!(buffers <= 8, "{k:?}: {buffers} storage buffers");

            assert_eq!(
                wgsl.matches(raa_shader::DEPTH_WGSL).count(),
                1,
                "{k:?}: the depth helper appears once"
            );
            assert_eq!(
                wgsl.matches(raa_shader::VG_SURFACE_WGSL).count(),
                1,
                "{k:?}: the vg surface appears once"
            );
        }
    }
}

#[test]
fn graph_tint_replaces_the_tint_call_in_both_vg_bodies() {
    let composer = default_composer();
    let reg = MaterialFnRegistry::new();
    let multiply = "return raa_apply_tint(material_main(in), x);";
    let graph = "return raa_apply_tint_opacity(material_main(in), x);";
    for pass in PassKind::VG {
        let k = key(MaterialFnId::STANDARD, pass);
        let today = composer.compose(&reg, &k).expect("compose");
        let rewritten = composer
            .compose_with(
                &reg,
                &k,
                &ComposeOptions {
                    tint_mode: TintMode::Graph,
                },
            )
            .expect("graph");
        assert_eq!(today.matches(multiply).count(), 1, "{pass:?}");
        assert_eq!(rewritten.matches(multiply).count(), 0, "{pass:?}");
        assert_eq!(rewritten.matches(graph).count(), 1, "{pass:?}");
    }
}
