//! Contracts `area-shadow-settings`, `area-shadow-lanes` and `shadow-pcss-kernel` in
//! `LOCAL_LIGHTING_WGSL`: the area branch blends the hero PCSS map, the GI cone visibility and
//! the GI-only far fade into V, while U and the point/spot/directional paths stay unchanged.

use raa_shader::{
    Composer, LOCAL_LIGHTING_WGSL, MaterialFnId, MaterialFnRegistry, PassKind, VariantKey,
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
