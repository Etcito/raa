//! S48 prelude texture helpers: `raa_texture_grad` and `RAA_ATLAS_MIP_LEVELS` are declared,
//! a user material taking `dpdx`/`dpdy` at the top and sampling `raa_texture_grad` in a
//! branch composes and validates for the Forward and Prepass passes.

use raa_shader::{Composer, MaterialFnRegistry, PassKind, VariantKey};

/// One default provider per slot (priority 0, the S11 defaults), as raa-lit installs them.
/// Kept here so this test does not depend on tests/variants.rs.
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

/// Derivatives at the top of `material_main`, `raa_texture_grad` only inside a branch.
const GRAD_IN_BRANCH: &str = "fn material_main(in: MaterialInput) -> MaterialOutput {
    let duv_dx = dpdx(in.uv * 64.0);
    let duv_dy = dpdy(in.uv * 64.0);
    var o = raa_default_output(in);
    let base = raa_material_data_base(in.record);
    if (raa_material_word(base + 1u) != 0u) {
        let c = raa_texture_grad(raa_material_word(base), fract(in.uv * 64.0), duv_dx, duv_dy);
        o.base_color = c.rgb;
    }
    return o;
}
";

#[test]
fn prelude_declares_texture_helpers() {
    let prelude = raa_shader::PRELUDE_WGSL;
    for decl in [
        "fn raa_texture_grad(id: u32, uv: vec2<f32>, duv_dx: vec2<f32>, duv_dy: vec2<f32>) -> vec4<f32>",
        "const RAA_ATLAS_MIP_LEVELS: f32 = 5.0;",
        "fn raa_texture(id: u32, uv: vec2<f32>) -> vec4<f32>",
    ] {
        assert!(prelude.contains(decl), "missing `{decl}`");
    }
}

#[test]
fn grad_in_branch_composes_and_validates() {
    let composer = default_composer();
    let mut reg = MaterialFnRegistry::new();
    let id = reg
        .register_user("user.grad", GRAD_IN_BRANCH, &composer)
        .unwrap_or_else(|e| panic!("{e}"));
    for pass in [PassKind::Forward, PassKind::Prepass] {
        let key = VariantKey {
            material_fn: id,
            pass,
            msaa: 1,
            flags: 0,
        };
        let wgsl = composer
            .compose(&reg, &key)
            .unwrap_or_else(|e| panic!("{key:?}: {e}"));
        if let Err(e) = Composer::validate(&wgsl) {
            panic!("{key:?}: {e}");
        }
    }
}
