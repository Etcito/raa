//! WGSL composition and validation of the lit renderer (S11). A composed module is the
//! prelude, the helper library, one material function, the winning provider of each slot
//! and the entry points of one pass kind. [`MaterialFnRegistry`] holds the material
//! functions, [`Composer`] the slot providers ([`SlotProvider`]); [`Composer::compose`] builds
//! the module of a [`VariantKey`] and [`Composer::validate`] runs the naga WGSL front-end and
//! validator so a broken shader fails a CPU test without a GPU.

mod composer;
mod error;
mod ids;
mod registry;

pub use composer::{Composer, SlotProvider};
pub use error::ShaderError;
pub use ids::{MaterialFnId, PassKind, SlotId, VariantKey};
pub use registry::{MaterialFnEntry, MaterialFnRegistry};

/// S10 structs and bindings of groups 0-1, S11 material structs, flag constants,
/// `raa_default_output` and the tint helper.
pub const PRELUDE_WGSL: &str = include_str!("wgsl/prelude.wgsl");
/// S11 helper library (`raa_hash3`, `raa_noise3`, `raa_fbm3`, `raa_voronoi3`,
/// `raa_triplanar_weights`, `raa_perturb_normal`, `raa_luma`, `raa_srgb_to_linear`).
pub const LIB_WGSL: &str = include_str!("wgsl/lib.wgsl");

/// Prepass entry points: `vs_main` and `fs_main` writing the S12 `normal` (location 0:
/// view-space normal xyz, roughness) and `material` (location 1: F0 rgb sRGB-encoded,
/// metallic) targets.
pub const PASS_PREPASS_WGSL: &str = include_str!("wgsl/pass_prepass.wgsl");
/// Forward entry points: `vs_main` and `fs_main` writing the S11 final colour to the HDR
/// target (location 0).
pub const PASS_FORWARD_WGSL: &str = include_str!("wgsl/pass_forward.wgsl");
/// Shadow caster entry point `vs_main` (no fragment stage); the light view-projection is a
/// group-2 uniform.
pub const PASS_SHADOW_WGSL: &str = include_str!("wgsl/pass_shadow.wgsl");
/// Transparent entry points: `vs_main` and `fs_main` appending the shaded fragment to the
/// group-2 per-pixel linked list (OIT).
pub const PASS_TRANSPARENT_WGSL: &str = include_str!("wgsl/pass_transparent.wgsl");
/// Fur shell entry points: `vs_main` extruding the shell named by the group-2 uniform and
/// `fs_main` writing colour and strand coverage as alpha (alpha-to-coverage).
pub const PASS_FURSHELL_WGSL: &str = include_str!("wgsl/pass_furshell.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    const MATERIAL_AND_ENTRY: &str = "
fn material_main(in: MaterialInput) -> MaterialOutput {
    return raa_default_output(in);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>, @builtin(front_facing) ff: bool) -> @location(0) vec4<f32> {
    let px = vec2<i32>(frag.xy);
    let uv = frag.xy * raa_frame.viewport.zw;
    var in: MaterialInput;
    in.position = raa_frame.camera_pos.xyz;
    in.local_position = vec3<f32>(0.0);
    in.normal = vec3<f32>(0.0, 1.0, 0.0);
    in.tangent = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    in.uv = uv;
    in.color = vec4<f32>(1.0);
    in.view_dir = vec3<f32>(0.0, 0.0, 1.0);
    in.time = raa_frame.camera_pos.w;
    in.front_facing = ff;
    in.frag_coord = frag;
    in.instance_params = raa_extras[0].params;
    in.record = raa_materials[0];
    let o = raa_apply_tint(material_main(in), raa_extras[0]);
    let sh = textureSampleCompareLevel(raa_shadow_map, raa_shadow_cmp, uv, 0, 0.5);
    let sp = (raa_shadow.cascade_view_proj[0] * vec4<f32>(o.normal, 1.0)).x + raa_shadow.splits.x;
    let env = textureSampleLevel(raa_env_specular, raa_linear_clamp, o.normal, raa_env_sh.info.z).rgb
        + raa_env_sh.sh[0].rgb;
    let lut = textureSampleLevel(raa_brdf_lut, raa_linear_clamp, uv, 0.0).rg;
    let ao = textureLoad(raa_ao_tex, px, 0).r * textureLoad(raa_contact_tex, px, 0).r;
    let opaque = textureLoad(raa_opaque_color, px, 0).rgb;
    let depth = textureLoad(raa_opaque_depth, px, 0).r;
    let n = raa_perturb_normal(o.normal, in.tangent, vec3<f32>(0.0, 0.0, 1.0));
    let lib = raa_fbm3(in.position, 4) + raa_voronoi3(in.position).x
        + raa_luma(raa_srgb_to_linear(raa_triplanar_weights(n)));
    let c = o.base_color * (sh + sp + lut.x + ao + depth + lib) + env + opaque
        + raa_env.background.rgb + o.emissive;
    return vec4<f32>(c, o.opacity);
}
";

    fn module() -> String {
        format!("{PRELUDE_WGSL}\n{LIB_WGSL}\n{MATERIAL_AND_ENTRY}")
    }

    #[test]
    fn prelude_lib_and_default_material_validate() {
        if let Err(e) = Composer::validate(&module()) {
            panic!("{}: {}", e.code, e.message);
        }
    }

    #[test]
    fn s10_struct_sizes_match() {
        let m = naga::front::wgsl::parse_str(&module()).expect("parse");
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(m.to_ctx()).expect("layout");
        for (name, size) in [
            ("FrameGpu", 512),
            ("SurfaceMaterialGpu", 256),
            ("InstanceExtraGpu", 128),
            ("ShadowGpu", 320),
            ("EnvSh", 160),
            ("EnvGpu", 256),
        ] {
            let (h, _) = m
                .types
                .iter()
                .find(|(_, t)| t.name.as_deref() == Some(name))
                .expect(name);
            assert_eq!(layouter[h].size, size, "{name}");
        }
    }
}
