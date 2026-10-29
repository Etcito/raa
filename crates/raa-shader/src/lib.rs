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

/// Post-material instance tint of a composed module (S23). `InstanceExtraGpu` keeps its
/// 128-byte layout: `mat_instance_tint_set` reads the spare lane `reserved.x` (1 after
/// setInstanceTint, 0 after clearInstanceTint and by default), written from raa-lit
/// `InstanceExtra::tint_set`. A draw selects the mode with [`VARIANT_FLAG_TINT_GRAPH`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum TintMode {
    /// Today's shader: the tint rgb multiplies the base colour, a multiplies the opacity.
    #[default]
    Multiply,
    /// The material graph reads `mat_instance_tint` itself: only the opacity multiply.
    Graph,
}

impl TintMode {
    /// Refusal code of [`TintMode::from_name`].
    pub const MODE_UNKNOWN: &'static str = "tint.mode_unknown";

    /// `multiply` or `graph` (the m20 `TintMode` names).
    pub fn name(self) -> &'static str {
        match self {
            TintMode::Multiply => "multiply",
            TintMode::Graph => "graph",
        }
    }

    /// The mode named `name`; refuses `tint.mode_unknown` for any other name.
    pub fn from_name(name: &str) -> Result<TintMode, ShaderError> {
        match name {
            "multiply" => Ok(TintMode::Multiply),
            "graph" => Ok(TintMode::Graph),
            _ => Err(ShaderError::new(
                TintMode::MODE_UNKNOWN,
                format!("tint mode {name:?} is not \"multiply\" or \"graph\""),
            )),
        }
    }
}

/// [`VariantKey::flags`] bit of a draw whose material is in tint mode `graph` (S23). Material
/// flags use the low bits and raa-lit's posed-skin bit is bit 31, so the two variants of a
/// material hash, cache and compose as different pipelines. A key without the bit composes
/// exactly the module of [`Composer::compose`].
pub const VARIANT_FLAG_TINT_GRAPH: u32 = 1 << 30;

impl TintMode {
    /// The [`VariantKey::flags`] bit of this mode: [`VARIANT_FLAG_TINT_GRAPH`] or 0.
    pub fn variant_flag(self) -> u32 {
        match self {
            TintMode::Multiply => 0,
            TintMode::Graph => VARIANT_FLAG_TINT_GRAPH,
        }
    }

    /// The mode selected by the variant `flags`.
    pub fn from_variant_flags(flags: u32) -> TintMode {
        if flags & VARIANT_FLAG_TINT_GRAPH != 0 {
            TintMode::Graph
        } else {
            TintMode::Multiply
        }
    }
}

/// Options of [`Composer::compose_with`]. They sit beside [`VariantKey`] rather than in it,
/// so the variant keys built across the workspace keep their fields.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ComposeOptions {
    pub tint_mode: TintMode,
}

/// The post-material tint call of the prepass, forward, transparent and fur shell passes.
const PASS_TINT_CALL: &str = "return raa_apply_tint(material_main(in), x);";
/// Its tint mode `graph` replacement.
const PASS_TINT_CALL_GRAPH: &str = "return raa_apply_tint_opacity(material_main(in), x);";

impl Composer {
    /// [`Composer::compose`] with `opts`. [`TintMode::Multiply`] returns exactly the
    /// module of `compose`; [`TintMode::Graph`] replaces the pass's post-material
    /// `raa_apply_tint` by `raa_apply_tint_opacity` (the shadow pass has none and is
    /// unchanged).
    pub fn compose_with(
        &self,
        reg: &MaterialFnRegistry,
        key: &VariantKey,
        opts: &ComposeOptions,
    ) -> Result<String, ShaderError> {
        let mut wgsl = self.compose(reg, key)?;
        if opts.tint_mode == TintMode::Graph {
            // The pass is the last section; recursion is not WGSL, so the material
            // function cannot hold this call.
            if let Some(at) = wgsl.rfind(PASS_TINT_CALL) {
                wgsl.replace_range(at..at + PASS_TINT_CALL.len(), PASS_TINT_CALL_GRAPH);
            }
        }
        Ok(wgsl)
    }
}

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

    fn stub_composer() -> Composer {
        let mut composer = Composer::new();
        for slot in SlotId::ALL {
            let wgsl = match slot {
                SlotId::Deform => "fn raa_slot_deform(p: vec3<f32>, n: vec3<f32>, x: InstanceExtraGpu, t: f32) -> VertexMaterialOutput {
    return VertexMaterialOutput(p, n);
}
",
                SlotId::Shadow => "fn raa_slot_shadow(world_pos: vec3<f32>, n: vec3<f32>, frag: vec4<f32>) -> f32 {
    return 1.0;
}
",
                SlotId::Ao => "fn raa_slot_ao(frag: vec4<f32>) -> f32 {
    return 1.0;
}
",
                SlotId::Ambient => "fn raa_slot_ambient(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    return s.base_color;
}
",
                SlotId::Direct => "fn raa_slot_direct(s: MaterialOutput, c: ShadeCtx, l: vec3<f32>, radiance: vec3<f32>) -> vec3<f32> {
    return s.base_color * radiance;
}
",
                SlotId::Transmission => "fn raa_slot_transmission(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    return vec3<f32>(0.0);
}
",
                SlotId::Fog => "fn raa_slot_fog(color: vec3<f32>, world_pos: vec3<f32>, c: ShadeCtx) -> vec3<f32> {
    return color;
}
",
            };
            let p = SlotProvider {
                slot,
                priority: 0,
                provider: "stub",
                wgsl: std::borrow::Cow::Borrowed(wgsl),
            };
            composer.add_slot(p).expect("add");
        }
        composer
    }

    /// The S23 `input.player_color` emit inside a user material function.
    const PLAYER_COLOR_FN: &str = "fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let c = select(vec3<f32>(0.5), mat_instance_tint.rgb, mat_instance_tint_set > 0.5);
    o.base_color = c;
    return o;
}
";

    fn key(material_fn: MaterialFnId, pass: PassKind) -> VariantKey {
        VariantKey {
            material_fn,
            pass,
            msaa: 4,
            flags: 0,
        }
    }

    #[test]
    fn both_tint_modes_compose_valid_wgsl_for_every_pass() {
        let composer = stub_composer();
        let mut reg = MaterialFnRegistry::new();
        let player = reg
            .register_user("player_color", PLAYER_COLOR_FN, &composer)
            .expect("register");
        for material_fn in [MaterialFnId::STANDARD, player] {
            for pass in PassKind::ALL {
                for tint_mode in [TintMode::Multiply, TintMode::Graph] {
                    let k = key(material_fn, pass);
                    let wgsl = composer
                        .compose_with(&reg, &k, &ComposeOptions { tint_mode })
                        .expect("compose");
                    if let Err(e) = Composer::validate(&wgsl) {
                        panic!("{material_fn:?} {pass:?} {tint_mode:?}: {e}");
                    }
                }
            }
        }
    }

    #[test]
    fn multiply_is_todays_module_and_graph_drops_only_the_colour_multiply() {
        let composer = stub_composer();
        let reg = MaterialFnRegistry::new();
        assert_eq!(ComposeOptions::default().tint_mode, TintMode::Multiply);
        for pass in PassKind::ALL {
            let k = key(MaterialFnId::STANDARD, pass);
            let today = composer.compose(&reg, &k).expect("compose");
            let multiply = composer
                .compose_with(&reg, &k, &ComposeOptions::default())
                .expect("multiply");
            let graph = composer
                .compose_with(
                    &reg,
                    &k,
                    &ComposeOptions {
                        tint_mode: TintMode::Graph,
                    },
                )
                .expect("graph");
            assert_eq!(multiply, today, "{pass:?}");
            if pass == PassKind::Shadow {
                assert_eq!(graph, today);
                continue;
            }
            assert_eq!(multiply.matches(PASS_TINT_CALL).count(), 1, "{pass:?}");
            assert!(multiply.contains("raa_set_instance_tint(x);"), "{pass:?}");
            assert!(!graph.contains(PASS_TINT_CALL), "{pass:?}");
            assert_eq!(graph.matches(PASS_TINT_CALL_GRAPH).count(), 1, "{pass:?}");
            assert_eq!(graph.replace(PASS_TINT_CALL_GRAPH, PASS_TINT_CALL), today);
        }
    }

    #[test]
    fn graph_flag_selects_a_distinct_variant() {
        assert_eq!(TintMode::Multiply.variant_flag(), 0);
        assert_eq!(TintMode::Graph.variant_flag(), VARIANT_FLAG_TINT_GRAPH);
        for mode in [TintMode::Multiply, TintMode::Graph] {
            assert_eq!(TintMode::from_variant_flags(mode.variant_flag() | 8), mode);
        }
        let k = key(MaterialFnId::STANDARD, PassKind::Forward);
        let graph = VariantKey {
            flags: k.flags | VARIANT_FLAG_TINT_GRAPH,
            ..k
        };
        assert_ne!(k, graph);
        let composer = stub_composer();
        let reg = MaterialFnRegistry::new();
        let a = composer.compose(&reg, &k).expect("compose");
        let b = composer.compose(&reg, &graph).expect("compose");
        assert_ne!(a, b, "the flag reaches RAA_VARIANT_FLAGS");
    }

    #[test]
    fn tint_mode_names() {
        for mode in [TintMode::Multiply, TintMode::Graph] {
            assert_eq!(TintMode::from_name(mode.name()), Ok(mode));
        }
        let e = TintMode::from_name("add").unwrap_err();
        assert_eq!(e.code(), "tint.mode_unknown");
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
