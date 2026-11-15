//! Contracts `area-shadow-settings`, `area-shadow-lanes` and `shadow-pcss-kernel` in
//! `LOCAL_LIGHTING_WGSL`: the area branch blends the hero PCSS map, the GI cone visibility and
//! the GI-only far fade into V, while U and the point/spot/directional paths stay unchanged.

use raa_shader::{
    Composer, LOCAL_LIGHTING_WGSL, MaterialFnId, MaterialFnRegistry, PRELUDE_WGSL, PassKind,
    VariantKey,
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

fn default_composer() -> Composer {
    let mut composer = Composer::new();
    for p in default_slots::all() {
        if let Err(e) = composer.add_slot(p) {
            panic!("{e}");
        }
    }
    composer
}

/// Composes STANDARD for the three passes the area branch runs in, at MSAA 1 and 4, and
/// naga-validates each composed module.
#[test]
fn standard_area_visibility_tiers_compose_and_validate() {
    let reg = MaterialFnRegistry::new();
    let composer = default_composer();
    let mut composed = 0;
    for pass in [PassKind::Forward, PassKind::Transparent, PassKind::FurShell] {
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
            if let Err(e) = Composer::validate(&wgsl) {
                panic!("{key:?}: {e}");
            }
            assert!(
                wgsl.contains(&format!("const RAA_MSAA: u32 = {msaa}u;")),
                "{key:?}"
            );
            naga::front::wgsl::parse_str(&wgsl).unwrap_or_else(|e| panic!("{key:?}: {e}"));
            composed += 1;
        }
    }
    assert_eq!(composed, 6);
}

fn wgsl_fn_body<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source
        .find(&format!("fn {name}("))
        .unwrap_or_else(|| panic!("{name} missing"));
    let rest = &source[start..];
    let end = rest.find("\n}\n").expect("function end");
    &rest[..end]
}

#[test]
fn area_visibility_tier_pins_are_present() {
    for pin in [
        "const RAA_GI_ACTIVE_FEATURE: u32 = 29u;",
        "light.color_inv_range_sq.w",
        "light.direction_cos_outer.w",
        "light.cos_inner",
        "raa_shadow.params.w > 0.5",
        "textureNumLayers(raa_shadow_map)",
        "fn raa_area_hero_visibility(",
        "2.7474774",
    ] {
        assert!(LOCAL_LIGHTING_WGSL.contains(pin), "missing `{pin}`");
    }
    let hero = wgsl_fn_body(LOCAL_LIGHTING_WGSL, "raa_area_hero_visibility");
    for call in [
        "raa_shadow_blockers(",
        "raa_shadow_pcf(",
        "raa_csm_rotation(frag.xy)",
    ] {
        assert!(hero.contains(call), "hero visibility missing `{call}`");
    }
}

/// S44 contract `area-stochastic-gpu`: bindings 19 and 20 declare the 80-B mid-tier slot
/// table and the two-layer mask once, the area branch reads the mid-tier V through them,
/// and the composed Forward module reflects the table layout and the arrayed float image.
#[test]
fn stochastic_mask_tier_pins_and_bindings() {
    for declaration in [
        "struct RaaAreaStochasticGpu { lights: array<vec4<u32>, 2>, weights: array<vec4<f32>, 2>, params: vec4<u32> }",
        "@group(1) @binding(19) var<uniform> raa_area_stochastic: RaaAreaStochasticGpu;",
        "@group(1) @binding(20) var raa_area_shadow_mask: texture_2d_array<f32>;",
    ] {
        assert_eq!(
            PRELUDE_WGSL.matches(declaration).count(),
            1,
            "`{declaration}` is not declared exactly once"
        );
    }
    assert!(LOCAL_LIGHTING_WGSL.contains("fn raa_area_stochastic_visibility("));
    let stochastic = wgsl_fn_body(LOCAL_LIGHTING_WGSL, "raa_area_stochastic_visibility");
    for pin in [
        "raa_area_stochastic.params.x",
        "textureDimensions(raa_area_shadow_mask)",
        "textureLoad(raa_area_shadow_mask",
    ] {
        assert!(
            stochastic.contains(pin),
            "stochastic visibility missing `{pin}`"
        );
    }
    let contribution = wgsl_fn_body(LOCAL_LIGHTING_WGSL, "raa_local_contribution");
    assert!(contribution.contains("mix(v_cone, stochastic.x, stochastic.y)"));

    let reg = MaterialFnRegistry::new();
    let composer = default_composer();
    let key = VariantKey {
        material_fn: MaterialFnId::STANDARD,
        pass: PassKind::Forward,
        msaa: 4,
        flags: 0,
    };
    let wgsl = composer
        .compose(&reg, &key)
        .unwrap_or_else(|e| panic!("{key:?}: {e}"));
    let module = naga::front::wgsl::parse_str(&wgsl).unwrap_or_else(|e| panic!("{key:?}: {e}"));
    let global = |group: u32, binding: u32| {
        module
            .global_variables
            .iter()
            .find(|(_, global)| global.binding == Some(naga::ResourceBinding { group, binding }))
            .map(|(_, global)| global)
            .unwrap_or_else(|| panic!("group {group} binding {binding} missing"))
    };

    let table = global(1, 19);
    assert_eq!(table.name.as_deref(), Some("raa_area_stochastic"));
    assert_eq!(table.space, naga::AddressSpace::Uniform);
    let naga::TypeInner::Struct { members, span } = &module.types[table.ty].inner else {
        panic!("raa_area_stochastic is not a struct");
    };
    assert_eq!(*span, 80, "RaaAreaStochasticGpu span");
    assert_eq!(
        members
            .iter()
            .map(|member| member.offset)
            .collect::<Vec<_>>(),
        [0, 32, 64],
        "RaaAreaStochasticGpu member offsets"
    );

    let mask = global(1, 20);
    assert_eq!(mask.name.as_deref(), Some("raa_area_shadow_mask"));
    assert!(
        matches!(
            module.types[mask.ty].inner,
            naga::TypeInner::Image {
                dim: naga::ImageDimension::D2,
                arrayed: true,
                class: naga::ImageClass::Sampled {
                    kind: naga::ScalarKind::Float,
                    multi: false,
                },
            }
        ),
        "raa_area_shadow_mask is not a sampled D2 array float image"
    );
}
