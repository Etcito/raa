//! WGSL composition and validation of the lit renderer (S11). A composed module is the
//! prelude, the helper library, one material function, the winning provider of each slot
//! and the entry points of one pass kind. [`MaterialFnRegistry`] holds the material
//! functions, [`Composer`] the slot providers ([`SlotProvider`]); [`Composer::compose`] builds
//! the module of a [`VariantKey`] and [`Composer::validate`] runs the naga WGSL front-end and
//! validator so a broken shader fails a CPU test without a GPU.

mod composer;
mod error;
mod gpu_constants;
mod ids;
mod registry;
mod vg;

pub use composer::{Composer, SlotProvider};
pub use error::ShaderError;
pub use gpu_constants::*;
pub use ids::{MaterialFnId, PassKind, SlotId, VariantKey};
pub use registry::{MaterialFnEntry, MaterialFnRegistry};
pub use vg::VG_SURFACE_WGSL;

/// S10 structs and bindings of groups 0-1, S11 material structs, flag constants,
/// `raa_default_output` and the tint helper. Self-contained: no slot definitions or calls.
pub const PRELUDE_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/prelude.wgsl"));
/// Standalone S10 weather declarations, neutral defaults and directional radiance boundary.
/// Already included once in [`PRELUDE_WGSL`]. Cloud shadow composition consumes these
/// bindings; standalone directional consumers can include this source without the prelude.
pub const WEATHER_LIGHTING_WGSL: &str = include_str!("wgsl/weather_lighting.wgsl");
/// S25 ground cloud shadow sampler `raa_cloud_shadow(world_pos: vec3<f32>) -> f32`, taking
/// world position in metres and returning 1 when clouds are off. It consumes the group 1
/// weather declarations and binding 6 sampler of [`PRELUDE_WGSL`] without redeclaring them;
/// [`Composer::compose`] attaches it once before the slot providers, and standalone lighting
/// consumers include it after the prelude (or after [`WEATHER_LIGHTING_WGSL`] plus their own
/// sampler declaration).
pub const CLOUD_SHADOW_WGSL: &str = include_str!("wgsl/cloud_shadow.wgsl");
/// S11 helper library (`raa_hash3`, `raa_noise3`, `raa_fbm3`, `raa_voronoi3`,
/// `raa_triplanar_weights`, `raa_perturb_normal`, `raa_luma`, `raa_srgb_to_linear`).
pub const LIB_WGSL: &str = include_str!("wgsl/lib.wgsl");

/// Shared tiled local-light evaluation, composed alongside Direct and LocalVisibility.
/// Handoff: exports `raa_local_lighting(s: MaterialOutput, c: ShadeCtx) -> vec3<f32>`;
/// consumes frame light counts, packed light indices and the binding-13 tile texture.
pub const LOCAL_LIGHTING_WGSL: &str = include_str!("wgsl/local_lighting.wgsl");

/// S32 standard-Z depth helper: `RAA_DEPTH_FAR`, `raa_depth_nearer`, `raa_depth_nearest`,
/// `raa_depth_farthest`, `raa_depth_is_far`, `raa_depth_key` and `raa_depth_from_key`.
/// RAA is standard-Z (clip depth 0..1, clear 1, `Less`) while solstice is reversed-Z, so
/// every adapted comparison, reduction and atomic depth goes through these helpers. The
/// module declares no binding, no entry point and depends on nothing else (not even
/// [`PRELUDE_WGSL`]), so a pass-private module can include it alone.
pub const DEPTH_WGSL: &str = include_str!("wgsl/depth.wgsl");

/// S37 GI volume tracing: `raa_gi_level_at`, `raa_gi_sample`, `RaaGiHit`, `raa_gi_trace`,
/// `raa_gi_visibility`, `raa_gi_edge_fade` and `raa_gi_finest_voxel`. It declares no
/// binding and samples only with `textureSampleLevel`; it must compose after
/// [`PRELUDE_WGSL`], or after a module declaring `RaaGiVolumeParams` and group 1 bindings
/// 6 (`raa_linear_clamp`), 17 (`raa_gi_volume`) and 18 (`raa_gi_volume_params`).
pub const GI_VOLUME_WGSL: &str = include_str!("wgsl/gi_volume.wgsl");

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
/// Virtualized-geometry G-buffer entry points: a fullscreen `vs_main` with no vertex input
/// and `fs_main` writing the S12 `normal`, `material` and `motion` targets plus
/// `@builtin(frag_depth)` of the resolved surface.
pub const PASS_VG_GBUFFER_WGSL: &str = include_str!("wgsl/pass_vg_gbuffer.wgsl");
/// Virtualized-geometry shade entry points: a fullscreen `vs_main` with no vertex input and
/// `fs_main` writing the S11 opaque final colour and `@builtin(frag_depth)`.
pub const PASS_VG_SHADE_WGSL: &str = include_str!("wgsl/pass_vg_shade.wgsl");

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

    #[test]
    fn prelude_validates_without_slot_providers() {
        assert!(!PRELUDE_WGSL.contains("raa_slot_"));
        assert_eq!(Composer::validate(PRELUDE_WGSL), Ok(()));
    }

    #[test]
    fn weather_lighting_standalone_layout_and_defaults() {
        assert_eq!(PRELUDE_WGSL.matches(WEATHER_LIGHTING_WGSL).count(), 1);
        let source = format!(
            "{WEATHER_LIGHTING_WGSL}\n\
@fragment fn weather_probe() -> @location(0) vec4<f32> {{
    let cloud = textureLoad(raa_cloud_transmittance, vec2<i32>(0), 0, 0);
    return vec4<f32>(raa_directional_radiance(cloud.rgb), 1.0);
}}"
        );
        assert_eq!(Composer::validate(WEATHER_LIGHTING_WGSL), Ok(()));
        assert_eq!(Composer::validate(&source), Ok(()));
        let m = naga::front::wgsl::parse_str(&source).unwrap();
        let (handle, weather) = m
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("WeatherLightingGpu"))
            .unwrap();
        let naga::TypeInner::Struct { members, span } = &weather.inner else {
            panic!("WeatherLightingGpu is a struct");
        };
        assert_eq!(*span as usize, WEATHER_LIGHTING_GPU_SIZE);
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(m.to_ctx()).unwrap();
        assert_eq!(layouter[handle].alignment, naga::proc::Alignment::SIXTEEN);
        assert_eq!(members.len(), 4);
        for (member, (name, offset, scalar)) in members.iter().zip([
            ("map", 0, naga::Scalar::F32),
            ("light", 16, naga::Scalar::F32),
            ("atmosphericTransmission", 32, naga::Scalar::F32),
            ("flags", 48, naga::Scalar::U32),
        ]) {
            assert_eq!(member.name.as_deref(), Some(name));
            assert_eq!(member.offset, offset);
            assert_eq!(
                m.types[member.ty].inner,
                naga::TypeInner::Vector {
                    size: naga::VectorSize::Quad,
                    scalar,
                }
            );
        }
        assert_eq!(m.global_variables.len(), 2, "no fragment storage added");
        for (_, global) in m.global_variables.iter() {
            let binding = global.binding.as_ref().unwrap();
            assert_eq!(binding.group, 1);
            match binding.binding {
                15 => {
                    assert_eq!(global.name.as_deref(), Some("raa_cloud_transmittance"));
                    assert_eq!(global.space, naga::AddressSpace::Handle);
                    assert_eq!(
                        m.types[global.ty].inner,
                        naga::TypeInner::Image {
                            dim: naga::ImageDimension::D2,
                            arrayed: true,
                            class: naga::ImageClass::Sampled {
                                kind: naga::ScalarKind::Float,
                                multi: false,
                            },
                        }
                    );
                }
                16 => {
                    assert_eq!(global.name.as_deref(), Some("raa_weather_lighting"));
                    assert_eq!(global.space, naga::AddressSpace::Uniform);
                    assert_eq!(global.ty, handle);
                }
                _ => panic!("unexpected weather binding"),
            }
        }
        let (_, default) = m
            .constants
            .iter()
            .find(|(_, c)| c.name.as_deref() == Some("RAA_WEATHER_LIGHTING_DEFAULT"))
            .unwrap();
        assert_eq!(default.ty, handle);
        let naga::Expression::Compose { components, .. } = &m.global_expressions[default.init]
        else {
            panic!("neutral record constructor");
        };
        assert_eq!(components.len(), 4);
        for (lane, expected) in components.iter().zip([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]) {
            let naga::Expression::Compose { components, .. } = &m.global_expressions[*lane] else {
                panic!("neutral lane constructor");
            };
            assert_eq!(components.len(), 4);
            for (value, expected) in components.iter().zip(expected) {
                let actual = match m.global_expressions[*value] {
                    naga::Expression::Literal(naga::Literal::F32(v)) => v,
                    naga::Expression::Literal(naga::Literal::U32(v)) => v as f32,
                    _ => panic!("neutral literal"),
                };
                assert_eq!(actual, expected);
            }
        }
        // Pin the directional boundary itself, without reproducing its math on the CPU.
        assert!(
            WEATHER_LIGHTING_WGSL
                .contains("if (raa_weather_lighting.flags.x == 0u) {\n        return raw;\n    }")
        );
        assert!(
            WEATHER_LIGHTING_WGSL
                .contains("return raw * raa_weather_lighting.atmosphericTransmission.rgb;")
        );
    }

    #[test]
    fn cloud_shadow_consumes_the_weather_declarations_once() {
        // The sampler lives beside the prelude, not inside it: the prelude stays the
        // standalone declaration source and compose attaches the helper once.
        assert_eq!(CLOUD_SHADOW_WGSL.matches("fn raa_cloud_shadow(").count(), 1);
        assert!(!PRELUDE_WGSL.contains("fn raa_cloud_shadow("));
        // It consumes the landed declarations instead of repeating them.
        assert!(!CLOUD_SHADOW_WGSL.contains("struct WeatherLightingGpu"));
        assert!(!CLOUD_SHADOW_WGSL.contains("@group(1) @binding("));
        let source = format!("{PRELUDE_WGSL}\n{CLOUD_SHADOW_WGSL}");
        assert_eq!(Composer::validate(&source), Ok(()));
        // Disabled clouds return 1, so CSM/contact visibility is untouched.
        assert!(
            CLOUD_SHADOW_WGSL
                .contains("if raa_weather_lighting.map.w < 0.5 {\n        return 1.0;\n    }")
        );
        // Receiver projection and the edge-to-mean fade of the adapted source.
        assert!(
            CLOUD_SHADOW_WGSL
                .contains("world_pos.xz - to_light.xz * (world_pos.y / max(to_light.y, 0.08))")
        );
        assert!(CLOUD_SHADOW_WGSL.contains("smoothstep(0.75, 1.0, edge)"));
        assert!(
            CLOUD_SHADOW_WGSL
                .contains("textureLoad(raa_cloud_transmittance, vec2<i32>(0, 0), 1, 0)")
        );
        assert!(CLOUD_SHADOW_WGSL.contains("raa_linear_clamp"));
    }

    #[test]
    fn depth_helper_is_standalone_and_declares_the_seven_items() {
        assert_eq!(Composer::validate(DEPTH_WGSL), Ok(()));
        for name in [
            "RAA_DEPTH_FAR",
            "raa_depth_nearer",
            "raa_depth_nearest",
            "raa_depth_farthest",
            "raa_depth_is_far",
            "raa_depth_key",
            "raa_depth_from_key",
        ] {
            assert!(DEPTH_WGSL.contains(name), "depth helper declares {name}");
        }
        assert!(!DEPTH_WGSL.contains("@group"));
        assert!(!DEPTH_WGSL.contains("@binding"));
    }

    #[test]
    fn depth_helper_probe_validates() {
        let probe = format!(
            "{DEPTH_WGSL}\n\
@group(0) @binding(0) var<storage, read_write> probe_out: array<u32>;

@compute @workgroup_size(1)
fn depth_probe() {{
    probe_out[0] = raa_depth_key(0.25);
    probe_out[1] = bitcast<u32>(raa_depth_from_key(probe_out[0]));
    probe_out[2] = select(0u, 1u, raa_depth_nearer(0.25, 0.5));
    probe_out[3] = bitcast<u32>(raa_depth_farthest(0.25, 0.5));
}}"
        );
        if let Err(e) = Composer::validate(&probe) {
            panic!("{}: {}", e.code, e.message);
        }
    }

    #[test]
    fn depth_key_orders_standard_z() {
        let depths = [0.0f32, 0.25, 0.999, 1.0];
        let keys: Vec<u32> = depths.iter().map(|d| (1.0 - d).to_bits()).collect();
        for pair in keys.windows(2) {
            assert!(pair[0] > pair[1], "nearer depth has a larger key");
        }
        assert_eq!(keys[3], 0, "depth 1.0 is far or empty");
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
                SlotId::LocalVisibility => "fn raa_slot_local_visibility(light_index: u32, world_position: vec3<f32>, surface_normal: vec3<f32>, direction_to_light: vec3<f32>, light_distance: f32) -> f32 {
    return 1.0;
}
",
                SlotId::Indirect => "fn raa_slot_indirect(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    return vec3<f32>(0.0);
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

    // Layout reflection adapted from solstice@7e465b5, crates/solstice-render/src/frame.rs:250.
    // MIT License
    //
    // Copyright (c) 2026 SamG-Coder
    //
    // Permission is hereby granted, free of charge, to any person obtaining a copy
    // of this software and associated documentation files (the "Software"), to deal
    // in the Software without restriction, including without limitation the rights
    // to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    // copies of the Software, and to permit persons to whom the Software is
    // furnished to do so, subject to the following conditions:
    //
    // The above copyright notice and this permission notice shall be included in all
    // copies or substantial portions of the Software.
    //
    // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    // SOFTWARE.
    #[test]
    fn s10_struct_sizes_match() {
        let source = module()
            + "
struct LightLayoutProbe { records: array<LightGpu, MAX_LOCAL_LIGHTS>, }
@vertex fn history_probe(@location(2) id: u32) -> @builtin(position) vec4<f32> {
    let r = raa_previous_instances[id];
    return raa_previous_model(r) * vec4<f32>(0.0, 0.0, 0.0, 1.0);
}";
        let m = naga::front::wgsl::parse_str(&source).expect("parse");
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&m)
        .expect("validate every entry point");
        let (history, _) = m
            .global_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("raa_previous_instances"))
            .unwrap();
        for (i, entry) in m.entry_points.iter().enumerate() {
            assert_eq!(
                !info.get_entry_point(i)[history].is_empty(),
                entry.stage == naga::ShaderStage::Vertex
            );
        }
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(m.to_ctx()).expect("layout");
        for (name, size) in [
            ("FrameGpu", FRAME_GPU_SIZE as u32),
            ("SurfaceMaterialGpu", 256),
            ("InstanceExtraGpu", 128),
            ("RaaInstanceGpu", 128),
            ("RaaPreviousInstanceGpu", 64),
            ("ShadowGpu", 320),
            ("EnvSh", 160),
            ("EnvGpu", 256),
            ("WeatherLightingGpu", WEATHER_LIGHTING_GPU_SIZE as u32),
        ] {
            let (h, _) = m
                .types
                .iter()
                .find(|(_, t)| t.name.as_deref() == Some(name))
                .expect(name);
            assert_eq!(layouter[h].size, size, "{name}");
        }
        let (_, probe) = m
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("LightLayoutProbe"))
            .unwrap();
        let naga::TypeInner::Struct {
            members: probe_members,
            span: probe_span,
        } = &probe.inner
        else {
            panic!("light array probe is a struct");
        };
        assert_eq!(*probe_span as usize, MAX_LOCAL_LIGHTS * LIGHT_GPU_SIZE);
        let naga::TypeInner::Array { stride, size, .. } = m.types[probe_members[0].ty].inner else {
            panic!("light records are an array");
        };
        assert_eq!(stride as usize, LIGHT_GPU_SIZE);
        assert_eq!(
            size,
            naga::ArraySize::Constant(std::num::NonZeroU32::new(MAX_LOCAL_LIGHTS as u32).unwrap())
        );
        // Reflect member names, offsets and scalar/vector types as well as total size.
        let (light_handle, light) = m
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("LightGpu"))
            .expect("LightGpu");
        let naga::TypeInner::Struct { members, span } = &light.inner else {
            panic!("LightGpu is a struct");
        };
        assert_eq!(*span as usize, LIGHT_GPU_SIZE);
        assert_eq!(layouter[light_handle].size, *span);
        assert_eq!(
            layouter[light_handle].alignment,
            naga::proc::Alignment::SIXTEEN
        );
        let vector = naga::TypeInner::Vector {
            size: naga::VectorSize::Quad,
            scalar: naga::Scalar::F32,
        };
        let float = naga::TypeInner::Scalar(naga::Scalar::F32);
        let expected = [
            ("position_range", 0, vector.clone()),
            ("color_inv_range_sq", 16, vector.clone()),
            ("direction_cos_outer", 32, vector),
            ("cos_inner", 48, float.clone()),
            ("kind", 52, naga::TypeInner::Scalar(naga::Scalar::U32)),
            ("width", 56, float.clone()),
            ("height", 60, float),
        ];
        assert_eq!(members.len(), expected.len());
        for (name, offset, ty) in expected {
            let member = members
                .iter()
                .find(|m| m.name.as_deref() == Some(name))
                .unwrap_or_else(|| panic!("{name} missing from LightGpu"));
            assert_eq!(member.offset, offset, "{name}");
            assert_eq!(m.types[member.ty].inner, ty, "{name}");
        }
        let (_, previous) = m
            .global_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("raa_previous_instances"))
            .unwrap();
        assert_eq!(
            previous.binding,
            Some(naga::ResourceBinding {
                group: 0,
                binding: 8
            })
        );
        let naga::TypeInner::Array { base, stride, .. } = m.types[previous.ty].inner else {
            panic!("previous records are an array");
        };
        assert_eq!(stride, 64);
        let naga::TypeInner::Struct { ref members, span } = m.types[base].inner else {
            panic!("previous record is a struct");
        };
        assert_eq!(span, 64);
        assert_eq!(
            members.iter().map(|m| m.offset).collect::<Vec<_>>(),
            [0, 16, 32, 48, 52, 56, 60]
        );
        for member in &members[3..] {
            assert_eq!(
                m.types[member.ty].inner,
                naga::TypeInner::Scalar(naga::Scalar::U32)
            );
        }
        // Evaluate the actual reconstruction expression on translated, rotated and
        // non-uniformly scaled affine rows, without a GPU or a second shader formula.
        let (_, f) = m
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("raa_previous_model"))
            .unwrap();
        let naga::Statement::Return { value: Some(value) } = f.body.last().unwrap() else {
            panic!("matrix return");
        };
        let naga::Expression::Compose { ref components, .. } = f.expressions[*value] else {
            panic!("matrix columns");
        };
        let rows = [
            [0.0, -3.0, 0.0, 7.0],
            [2.0, 0.0, 0.0, -8.0],
            [0.0, 0.0, 4.0, 9.0],
        ];
        let mut columns = Vec::new();
        for column in components {
            let naga::Expression::Compose { ref components, .. } = f.expressions[*column] else {
                panic!("column components");
            };
            for component in components {
                columns.push(match f.expressions[*component] {
                    naga::Expression::Literal(naga::Literal::F32(v)) => v,
                    naga::Expression::AccessIndex { base, index } => {
                        let naga::Expression::AccessIndex { base, index: row } =
                            f.expressions[base]
                        else {
                            panic!("row access");
                        };
                        assert!(matches!(
                            f.expressions[base],
                            naga::Expression::FunctionArgument(0)
                        ));
                        rows[row as usize][index as usize]
                    }
                    _ => panic!("affine component"),
                });
            }
        }
        assert_eq!(
            columns,
            [
                0.0, 2.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 7.0, -8.0, 9.0, 1.0
            ]
        );
        let (frame, _) = m
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("FrameGpu"))
            .expect("FrameGpu");
        let naga::TypeInner::Struct { members, span } = &m.types[frame].inner else {
            panic!("FrameGpu is a struct");
        };
        assert_eq!(*span as usize, FRAME_GPU_SIZE);
        let member_kind = |ty: naga::Handle<naga::Type>| match &m.types[ty].inner {
            naga::TypeInner::Matrix {
                columns: naga::VectorSize::Quad,
                rows: naga::VectorSize::Quad,
                scalar,
            } if scalar.width == 4 && scalar.kind == naga::ScalarKind::Float => "mat4x4<f32>",
            naga::TypeInner::Vector {
                size: naga::VectorSize::Quad,
                scalar,
            } if scalar.width == 4 && scalar.kind == naga::ScalarKind::Float => "vec4<f32>",
            naga::TypeInner::Vector {
                size: naga::VectorSize::Quad,
                scalar,
            } if scalar.width == 4 && scalar.kind == naga::ScalarKind::Uint => "vec4<u32>",
            naga::TypeInner::Array {
                base, stride: 16, ..
            } if m.types[*base].inner
                == (naga::TypeInner::Vector {
                    size: naga::VectorSize::Quad,
                    scalar: naga::Scalar::F32,
                }) =>
            {
                "array<vec4>"
            }
            _ => "other",
        };
        for (name, offset, kind) in [
            ("view_proj", 0, "mat4x4<f32>"),
            ("view", 64, "mat4x4<f32>"),
            ("proj", 128, "mat4x4<f32>"),
            ("inv_proj", 192, "mat4x4<f32>"),
            ("inv_view", 256, "mat4x4<f32>"),
            ("camera_pos", 320, "vec4<f32>"),
            ("sun_dir", 336, "vec4<f32>"),
            ("sun_color", 352, "vec4<f32>"),
            ("ambient", 368, "vec4<f32>"),
            ("viewport", 384, "vec4<f32>"),
            ("clip", 400, "vec4<f32>"),
            ("flags", 416, "vec4<u32>"),
            ("exposure", 432, "vec4<f32>"),
            ("reserved", 448, "array<vec4>"),
            ("previous_view_proj", 512, "mat4x4<f32>"),
            ("unjittered_view_proj", 576, "mat4x4<f32>"),
            ("camera_jitter", 640, "vec4<f32>"),
            ("light_info", 656, "vec4<u32>"),
            ("temporal", 672, "vec4<u32>"),
            ("reserved2", 688, "array<vec4>"),
        ] {
            let member = members
                .iter()
                .find(|x| x.name.as_deref() == Some(name))
                .unwrap_or_else(|| panic!("FrameGpu.{name}"));
            assert_eq!(member.offset, offset, "FrameGpu.{name} offset");
            assert_eq!(member_kind(member.ty), kind, "FrameGpu.{name} type");
        }
    }
}
