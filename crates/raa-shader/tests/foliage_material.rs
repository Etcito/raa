//! M7 relightable foliage composition (contracts m7-material-wgsl and m7-material-data):
//! naga validation of every mode and pass, the mode-specific sampling fragments, the opaque
//! draw class, and CPU twins of the header/mask reader, the mask selection and the backlit
//! translucency term, pinned to the WGSL text. No GPU.

use std::borrow::Cow;

use raa_shader::{
    Composer, MaterialFnId, MaterialFnRegistry, PassKind, SlotId, SlotProvider, VariantKey,
};

const WGSL: &str = include_str!("../src/wgsl/foliage_material.wgsl");
const VERTEX_MATERIAL: &str = "fn material_main(in: MaterialInput) -> MaterialOutput { return raa_default_output(in); }
fn material_vertex(in: VertexMaterialInput) -> VertexMaterialOutput { return VertexMaterialOutput(in.local_position, in.normal); }
";

fn composer() -> Composer {
    let sources = [
        "fn raa_slot_deform(p: vec3<f32>, n: vec3<f32>, x: InstanceExtraGpu, t: f32) -> VertexMaterialOutput { return VertexMaterialOutput(p, n); }",
        "fn raa_slot_shadow(p: vec3<f32>, n: vec3<f32>, f: vec4<f32>) -> f32 { return 1.0; }",
        "fn raa_slot_ao(f: vec4<f32>) -> f32 { return 1.0; }",
        "fn raa_slot_ambient(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> { return s.base_color * c.ao; }",
        "fn raa_slot_direct(s: MaterialOutput, c: ShadeCtx, l: vec3<f32>, r: vec3<f32>) -> vec3<f32> { return r * max(dot(s.normal, l), 0.0); }",
        "fn raa_slot_transmission(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> { return vec3<f32>(0.0); }",
        "fn raa_slot_fog(color: vec3<f32>, p: vec3<f32>, c: ShadeCtx) -> vec3<f32> { return color; }",
        "fn raa_slot_local_visibility(i: u32, p: vec3<f32>, n: vec3<f32>, d: vec3<f32>, distance: f32) -> f32 { return 1.0; }",
        "fn raa_slot_indirect(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> { return vec3<f32>(0.0); }",
    ];
    let mut composer = Composer::new();
    for (slot, source) in SlotId::ALL.into_iter().zip(sources) {
        composer
            .add_slot(SlotProvider {
                slot,
                priority: 0,
                provider: "foliage-test",
                wgsl: Cow::Borrowed(source),
            })
            .unwrap();
    }
    composer
}

fn key(material_fn: MaterialFnId, pass: PassKind, flags: u32) -> VariantKey {
    VariantKey {
        material_fn,
        pass,
        msaa: 4,
        flags,
    }
}

fn compose(mode: u32, pass: PassKind) -> String {
    composer()
        .compose_foliage(
            &MaterialFnRegistry::new(),
            &key(MaterialFnId::STANDARD, pass, 0),
            mode,
        )
        .unwrap()
}

#[test]
fn every_mode_and_pass_validates_with_shared_fragments_once() {
    let composer = composer();
    let mut registry = MaterialFnRegistry::new();
    let vertex = registry
        .register_builtin("foliage.vertex", VERTEX_MATERIAL)
        .unwrap();
    for mode in 0..6u32 {
        for pass in [PassKind::Prepass, PassKind::Forward] {
            for material in [MaterialFnId::STANDARD, vertex] {
                let source = composer
                    .compose_foliage(&registry, &key(material, pass, 0), mode)
                    .unwrap();
                if let Err(e) = Composer::validate(&source) {
                    panic!("mode {mode} {pass:?}: {e}");
                }
                for once in [
                    "struct FoliageHit {",
                    "struct RaaFoliageParams {",
                    "@group(2) @binding(1) var raa_foliage_volume: texture_3d<f32>;",
                    "fn raa_foliage_settings()",
                    "fn raa_foliage_backlight(",
                    "@vertex",
                    "@fragment",
                ] {
                    assert_eq!(
                        source.matches(once).count(),
                        1,
                        "mode {mode} {pass:?}: {once}"
                    );
                }
                assert!(source.contains(&format!("const RAA_FOLIAGE_MODE: u32 = {mode}u;")));
                let module = naga::front::wgsl::parse_str(&source).unwrap();
                let mut group2: Vec<u32> = module
                    .global_variables
                    .iter()
                    .filter_map(|(_, v)| v.binding.as_ref())
                    .filter(|b| b.group == 2)
                    .map(|b| b.binding)
                    .collect();
                group2.sort_unstable();
                assert_eq!(group2, [0, 1, 2, 3, 4, 5], "mode {mode}");
                let names: Vec<&str> = module
                    .entry_points
                    .iter()
                    .map(|e| e.name.as_str())
                    .collect();
                assert_eq!(names, ["vs_main", "fs_main"], "mode {mode} {pass:?}");
            }
        }
    }
}

#[test]
fn modes_select_their_sampling_fragment() {
    for mode in 0..6u32 {
        let source = compose(mode, PassKind::Forward);
        let parallax = source.contains("fn foliage_parallax_hit(");
        let offset = source.contains("fn foliage_offset_hit(");
        assert_eq!(
            (parallax, offset),
            (mode != 2 && mode != 3, mode == 2 || mode == 3),
            "mode {mode}"
        );
        let call = if offset {
            "foliage_offset_hit(position, direction, RAA_FOLIAGE_MODE)"
        } else {
            "foliage_parallax_hit(position, direction, RAA_FOLIAGE_MODE)"
        };
        assert!(source.contains(call), "mode {mode}");
        let settings = if offset {
            "let p = raa_foliage;"
        } else {
            "let p = raa_foliage_params;"
        };
        assert!(source.contains(settings), "mode {mode}");
    }
    // Distilled modes take albedo/occlusion from the native D3 volume and the flat triangle
    // normal; the volume coordinate is the clamped normalized source-root position.
    for line in [
        "let coord = clamp((local_position - settings.min) / (settings.max - settings.min), vec3<f32>(0.0), vec3<f32>(1.0));",
        "return textureSampleLevel(raa_foliage_volume, raa_foliage_sampler, coord, 0.0);",
        "hit.albedo = select(hit.albedo, volume.rgb, distilled);",
        "let normal = normalize(select(normal_matrix * hit.normal, v.normal, distilled));",
        "@location(2) @interpolate(flat, either) normal: vec3<f32>,",
        "const RAA_FOLIAGE_MODE_DISTILLED: u32 = 4u;",
    ] {
        assert!(WGSL.contains(line), "missing {line}");
    }
    // No vertex colour stream is read.
    assert!(!WGSL.contains("v1.color") && !WGSL.contains("v.color"));
}

#[test]
fn passes_differ_and_forward_lights_once_while_opaque() {
    let prepass = compose(4, PassKind::Prepass);
    let forward = compose(1, PassKind::Forward);
    assert!(prepass.contains("struct RaaFoliagePrepassTargets {"));
    assert!(
        !prepass.contains("RaaFoliageForwardTargets")
            && !prepass.contains("+ raa_foliage_backlight(")
    );
    assert!(!forward.contains("RaaFoliagePrepassTargets"));
    for source in [&prepass, &forward] {
        assert!(source.contains("out.depth = f.depth;"));
        assert!(source.contains("let f = raa_foliage_fragment(v, front_facing);"));
    }
    assert!(forward.contains("c.ao = min(s.ao, ao);"));
    assert_eq!(forward.matches("raa_slot_ambient(s, c)").count(), 1);
    for term in [
        "raa_slot_direct(s, c, to_light, radiance * shadow)",
        "raa_slot_indirect(s, c)",
        "raa_local_lighting(s, c)",
        "raa_directional_radiance(raa_frame.sun_color.rgb)",
    ] {
        assert!(forward.contains(term), "{term}");
    }
    assert!(
        forward
            .contains("out.color = vec4<f32>(select(lit, s.base_color + s.emissive, unlit), 1.0);")
    );
    assert!(
        forward.contains(
            "c.material_flags = f.in.record.header.z & ~RAA_FOLIAGE_TRANSLUCENT_CLASSES;"
        )
    );
    assert!(forward.contains("const RAA_FOLIAGE_TRANSLUCENT_CLASSES: u32 = 3u;"));
    // Transparent (1) and transmissive (2) draw-class bits never reach a foliage variant.
    let source = composer()
        .compose_foliage(
            &MaterialFnRegistry::new(),
            &key(MaterialFnId::STANDARD, PassKind::Forward, 0b1011),
            4,
        )
        .unwrap();
    assert!(source.contains("const RAA_VARIANT_FLAGS: u32 = 8u;"));
}

#[test]
fn invalid_modes_and_passes_refuse_and_existing_outputs_stay() {
    let composer = composer();
    let registry = MaterialFnRegistry::new();
    for (mode, pass) in [
        (6, PassKind::Forward),
        (u32::MAX, PassKind::Prepass),
        (0, PassKind::Shadow),
        (4, PassKind::Transparent),
    ] {
        let e = composer
            .compose_foliage(&registry, &key(MaterialFnId::STANDARD, pass, 0), mode)
            .unwrap_err();
        assert_eq!(e.code(), "shader.validate", "{mode} {pass:?}");
    }
    let forward = composer
        .compose(
            &registry,
            &key(MaterialFnId::STANDARD, PassKind::Forward, 0),
        )
        .unwrap();
    assert!(forward.contains(raa_shader::PASS_FORWARD_WGSL));
    let reference = composer
        .compose_foliage_reference(
            &registry,
            &key(MaterialFnId::STANDARD, PassKind::Prepass, 0),
        )
        .unwrap();
    for old in [&forward, &reference] {
        assert!(!old.contains("RAA_FOLIAGE_MODE") && !old.contains("raa foliage part"));
    }
    assert!(!raa_shader::PASS_FORWARD_WGSL.contains("foliage"));
}

// ---- CPU twins of the foliage_material.wgsl helpers (same names without the prefix) ----

type V3 = [f32; 3];
const MAGIC: u32 = 0x4d37_464c;
const BASE_COLOR: u32 = 1;
const NORMAL: u32 = 2;
const AO: u32 = 4;
const OPACITY: u32 = 8;

fn header_mask(words: [u32; 4]) -> u32 {
    let [magic, version, mask, _count] = words;
    if magic != MAGIC || version != 1 || mask & !15 != 0 {
        0
    } else {
        mask
    }
}

fn backlight(
    albedo: V3,
    translucency: f32,
    normal: V3,
    to_light: V3,
    radiance: V3,
    visibility: f32,
) -> V3 {
    let facing =
        (-(normal[0] * to_light[0] + normal[1] * to_light[1] + normal[2] * to_light[2])).max(0.0);
    [0, 1, 2].map(|i| albedo[i] * translucency * facing * radiance[i] * visibility)
}

#[derive(Clone, Copy)]
struct Hit {
    albedo: V3,
    normal: V3,
    coverage: f32,
}
#[derive(Clone, Copy)]
struct Authored {
    base_color: V3,
    normal: V3,
    ao: f32,
    opacity: f32,
}
#[derive(Debug, PartialEq)]
struct Surface {
    albedo: V3,
    normal: V3,
    ao: f32,
    coverage: f32,
}

fn surface(hit: Hit, s: Authored, mask: u32, occlusion: f32, distilled: bool) -> Surface {
    let canopy = 1.0 - occlusion.clamp(0.0, 1.0);
    let coverage = if mask & OPACITY != 0 {
        hit.coverage * s.opacity
    } else {
        hit.coverage
    };
    Surface {
        albedo: if mask & BASE_COLOR != 0 {
            [0, 1, 2].map(|i| hit.albedo[i] * s.base_color[i])
        } else {
            hit.albedo
        },
        normal: if mask & NORMAL != 0 {
            s.normal
        } else {
            hit.normal
        },
        ao: if mask & AO != 0 {
            canopy * s.ao
        } else {
            canopy
        },
        coverage: if distilled { 1.0 } else { coverage },
    }
}

fn cut(coverage: f32, mask: u32, cutoff: f32) -> bool {
    coverage <= 0.0 || (mask & OPACITY != 0 && coverage < cutoff)
}

#[test]
fn twins_match_the_wgsl_text() {
    for line in [
        "const RAA_FOLIAGE_DATA_MAGIC: u32 = 0x4d37464cu;",
        "const RAA_FOLIAGE_DATA_VERSION: u32 = 1u;",
        "const RAA_FOLIAGE_MASK_BASE_COLOR: u32 = 1u;",
        "const RAA_FOLIAGE_MASK_NORMAL: u32 = 2u;",
        "const RAA_FOLIAGE_MASK_AO: u32 = 4u;",
        "const RAA_FOLIAGE_MASK_OPACITY: u32 = 8u;",
        "const RAA_FOLIAGE_MASK_ALL: u32 = 15u;",
        "if (magic != RAA_FOLIAGE_DATA_MAGIC || version != RAA_FOLIAGE_DATA_VERSION) {",
        "return select(0u, mask, (mask & ~RAA_FOLIAGE_MASK_ALL) == 0u);",
        "let base = raa_material_data_base(record);",
        "return raa_foliage_header_mask(raa_material_word(base), raa_material_word(base + 1u), raa_material_word(base + 2u));",
        "return albedo * translucency * max(dot(-normal, to_light), 0.0) * radiance * visibility;",
        "let canopy = 1.0 - clamp(occlusion, 0.0, 1.0);",
        "out.albedo = select(hit.albedo, hit.albedo * s.base_color, (mask & RAA_FOLIAGE_MASK_BASE_COLOR) != 0u);",
        "out.normal = select(hit.normal, s.normal, (mask & RAA_FOLIAGE_MASK_NORMAL) != 0u);",
        "out.ao = select(canopy, canopy * s.ao, (mask & RAA_FOLIAGE_MASK_AO) != 0u);",
        "let coverage = select(hit.coverage, hit.coverage * s.opacity, (mask & RAA_FOLIAGE_MASK_OPACITY) != 0u);",
        "out.coverage = select(coverage, 1.0, distilled);",
        "return coverage <= 0.0 || ((mask & RAA_FOLIAGE_MASK_OPACITY) != 0u && coverage < cutoff);",
        "+ raa_foliage_backlight(s.base_color, clamp(f.settings.translucency, 0.0, 1.0), s.normal, to_light, radiance, shadow);",
        "s.opacity = 1.0;",
        "s.transmission = 0.0;",
    ] {
        assert!(WGSL.contains(line), "missing {line}");
    }
}

#[test]
fn backlight_is_zero_without_back_light_and_full_when_lit_from_behind() {
    let albedo = [0.2, 0.5, 0.1];
    let normal = [0.0, 1.0, 0.0];
    let radiance = [3.0, 2.0, 1.0];
    let zero = [0.0; 3];
    // Light in front of or tangent to the surface, no translucency, or a shadowed light.
    assert_eq!(
        backlight(albedo, 1.0, normal, [0.0, 1.0, 0.0], radiance, 1.0),
        zero
    );
    assert_eq!(
        backlight(albedo, 1.0, normal, [1.0, 0.0, 0.0], radiance, 1.0),
        zero
    );
    assert_eq!(
        backlight(albedo, 0.0, normal, [0.0, -1.0, 0.0], radiance, 1.0),
        zero
    );
    assert_eq!(
        backlight(albedo, 1.0, normal, [0.0, -1.0, 0.0], radiance, 0.0),
        zero
    );
    // Full backlight: albedo x radiance exactly.
    assert_eq!(
        backlight(albedo, 1.0, normal, [0.0, -1.0, 0.0], radiance, 1.0),
        [0.6, 1.0, 0.1]
    );
    // Half translucency at 60 degrees behind: a quarter.
    let l = [0.0, -0.5, 0.75f32.sqrt()];
    let q = backlight(albedo, 0.5, normal, l, radiance, 1.0);
    for (got, want) in q.iter().zip([0.15, 0.25, 0.025]) {
        assert!((got - want).abs() < 1e-6, "{q:?}");
    }
}

#[test]
fn header_words_select_each_mask_bit_and_default_to_baked() {
    let hit = Hit {
        albedo: [0.2, 0.4, 0.1],
        normal: [0.0, 0.0, 1.0],
        coverage: 0.8,
    };
    let s = Authored {
        base_color: [0.5, 0.5, 2.0],
        normal: [0.0, 1.0, 0.0],
        ao: 0.5,
        opacity: 0.25,
    };
    let occlusion = 0.6;
    let baked = surface(hit, s, 0, occlusion, false);
    assert_eq!(
        baked,
        Surface {
            albedo: hit.albedo,
            normal: hit.normal,
            ao: 1.0 - 0.6,
            coverage: 0.8
        }
    );
    // Words of the section 97 envelope: magic, version 1, mask, graph word count.
    for (words, mask) in [
        ([0x4d37_464c, 1, 0, 7], 0),
        ([0x4d37_464c, 1, 1, 7], BASE_COLOR),
        ([0x4d37_464c, 1, 2, 0], NORMAL),
        ([0x4d37_464c, 1, 4, 3], AO),
        ([0x4d37_464c, 1, 8, 3], OPACITY),
        ([0x4d37_464c, 1, 15, 3], 15),
        ([0x4d37_464d, 1, 15, 3], 0),
        ([0x4d37_464c, 2, 15, 3], 0),
        ([0x4d37_464c, 1, 0x11, 3], 0),
        ([0, 0, 0, 0], 0),
    ] {
        assert_eq!(header_mask(words), mask, "{words:x?}");
    }
    let albedo = surface(hit, s, BASE_COLOR, occlusion, false);
    assert_eq!(albedo.albedo, [0.1, 0.2, 0.2]);
    assert_eq!(
        (albedo.normal, albedo.ao, albedo.coverage),
        (hit.normal, baked.ao, 0.8)
    );
    let normal = surface(hit, s, NORMAL, occlusion, false);
    assert_eq!(
        (normal.albedo, normal.normal, normal.ao),
        (hit.albedo, s.normal, baked.ao)
    );
    let ao = surface(hit, s, AO, occlusion, false);
    assert!((ao.ao - 0.4 * 0.5).abs() < 1e-6);
    assert_eq!(
        (ao.albedo, ao.normal, ao.coverage),
        (hit.albedo, hit.normal, 0.8)
    );
    let opacity = surface(hit, s, OPACITY, occlusion, false);
    assert!((opacity.coverage - 0.2).abs() < 1e-6);
    assert!(cut(opacity.coverage, OPACITY, 0.5) && !cut(opacity.coverage, OPACITY, 0.1));
    assert!(!cut(baked.coverage, 0, 0.9) && cut(0.0, 0, 0.0));
}

#[test]
fn distilled_triangles_stay_fully_opaque() {
    let hit = Hit {
        albedo: [0.3, 0.3, 0.3],
        normal: [1.0, 0.0, 0.0],
        coverage: 1.0,
    };
    let s = Authored {
        base_color: [1.0; 3],
        normal: [0.0, 1.0, 0.0],
        ao: 1.0,
        opacity: 0.0,
    };
    for mask in [0, OPACITY, 15] {
        let out = surface(hit, s, mask, 1.0, true);
        assert_eq!(out.coverage, 1.0, "mask {mask}");
        assert!(!cut(out.coverage, mask, 0.99));
        // Full canopy occlusion darkens only the ambient factor.
        assert_eq!(out.ao, 0.0);
    }
}
