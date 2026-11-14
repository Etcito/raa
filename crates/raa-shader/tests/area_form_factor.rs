//! Contract `area-form-factor-wgsl`: the area-light maths and the BRDF/LTC atlas uv helpers
//! live once in `PRELUDE_WGSL`, keep their contract signatures and stay pure (no slot call,
//! no texture sampling, no bitwise operator), and `LOCAL_LIGHTING_WGSL` no longer defines the
//! rectangle basis that moved here.

use raa_shader::{Composer, LOCAL_LIGHTING_WGSL, PRELUDE_WGSL};

/// Bounds of the appended section; the purity scan runs between these two markers.
const SECTION: &str = "// S11 area lights: shared rectangle form-factor maths";
const SECTION_END: &str = "// End of the S11 area form-factor section.";

/// Every contract function with its exact signature text.
const SIGNATURES: [&str; 10] = [
    "fn raa_area_basis(travel: vec3<f32>) -> mat3x3<f32>",
    "fn raa_area_corners(light: LightGpu, origin: vec3<f32>) -> array<vec3<f32>, 4>",
    "fn raa_area_edge(a: vec3<f32>, b: vec3<f32>) -> vec3<f32>",
    "fn raa_area_vector_form_factor(corners: array<vec3<f32>, 4>, m: mat3x3<f32>) -> vec3<f32>",
    "fn raa_area_clipped(length_f: f32, z: f32) -> f32",
    "fn raa_area_form_factor(light: LightGpu, origin: vec3<f32>, normal: vec3<f32>) -> f32",
    "fn raa_area_radiance(light: LightGpu) -> vec3<f32>",
    "fn raa_area_window(distance: f32, range: f32) -> f32",
    "fn raa_brdf_lut_uv(n_dot_v: f32, roughness: f32) -> vec2<f32>",
    "fn raa_ltc_uv(table: u32, roughness: f32, n_dot_v: f32) -> vec2<f32>",
];

/// A fragment entry that calls every function of the new section.
const FRAGMENT_CALLER: &str = "
@fragment
fn area_probe(@location(0) p: vec3<f32>, @location(1) n: vec3<f32>) -> @location(0) vec4<f32> {
    var light: LightGpu;
    light.position_range = vec4<f32>(0.0, 1.0, 0.0, 5.0);
    light.color_inv_range_sq = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    light.direction_cos_outer = vec4<f32>(0.0, -1.0, 0.0, 0.0);
    light.cos_inner = 0.0;
    light.kind = 3u;
    light.width = 1.0;
    light.height = 1.0;
    let basis = raa_area_basis(light.direction_cos_outer.xyz);
    let corners = raa_area_corners(light, p);
    let edge = raa_area_edge(corners[0], corners[1]);
    let vff = raa_area_vector_form_factor(corners, basis);
    let clipped = raa_area_clipped(length(vff), dot(vff, n));
    let form_factor = raa_area_form_factor(light, p, n);
    let radiance = raa_area_radiance(light);
    let window = raa_area_window(2.0, 5.0);
    let brdf_uv = raa_brdf_lut_uv(0.5, 0.5);
    let ltc_uv = raa_ltc_uv(0u, 0.5, 0.5);
    let value = basis[0] + edge + vff + radiance
        + vec3<f32>(clipped + form_factor + window + brdf_uv.x + ltc_uv.x);
    return vec4<f32>(value, 1.0);
}
";

#[test]
fn contract_functions_are_defined_exactly_once_with_exact_signatures() {
    for signature in SIGNATURES {
        assert_eq!(
            PRELUDE_WGSL.matches(signature).count(),
            1,
            "`{signature}` is not defined exactly once in PRELUDE_WGSL"
        );
        assert!(
            !LOCAL_LIGHTING_WGSL.contains(signature),
            "`{signature}` must not be defined in LOCAL_LIGHTING_WGSL"
        );
    }
}

#[test]
fn rational_polynomials_and_clipped_sphere_formula_are_present() {
    for constant in [
        "0.8543985",
        "0.4965155",
        "0.0145206",
        "3.4175940",
        "4.1616724",
    ] {
        assert!(
            PRELUDE_WGSL.contains(constant),
            "missing rational constant {constant}"
        );
    }
    assert!(PRELUDE_WGSL.contains("(length_f * length_f + z) / (length_f + 1.0)"));
    assert!(PRELUDE_WGSL.contains("inverseSqrt(max(1.0 - x * x, 1e-7))"));
    assert!(
        PRELUDE_WGSL.contains("const RAA_BRDF_ATLAS_SIZE: vec2<f32> = vec2<f32>(128.0, 192.0);")
    );
}

#[test]
fn local_lighting_no_longer_defines_the_rectangle_basis() {
    assert!(!LOCAL_LIGHTING_WGSL.contains("fn raa_area_basis("));
    assert!(LOCAL_LIGHTING_WGSL.contains("raa_area_basis(light.direction_cos_outer.xyz)"));
}

#[test]
fn new_section_is_pure() {
    let section = PRELUDE_WGSL
        .split(SECTION)
        .nth(1)
        .expect("area form-factor section")
        .split(SECTION_END)
        .next()
        .expect("section end");
    assert!(!section.contains("raa_slot_"));
    assert!(!section.contains("textureSample"));
    assert!(!section.contains("@binding"));
    assert!(!section.contains(" ^ "));
    assert!(!section.contains(" | "));
    assert!(!section.contains(" & "));
    assert!(section.contains("Copyright (c) 2010-2025 three.js authors"));
}

#[test]
fn prelude_with_fragment_caller_validates() {
    let wgsl = format!("{PRELUDE_WGSL}\n{FRAGMENT_CALLER}");
    if let Err(e) = Composer::validate(&wgsl) {
        panic!("prelude + area form-factor fragment caller: {e}");
    }
}

/// Body of WGSL function `name` in `source` (from `fn name(` to the closing `\n}\n`).
fn fn_body<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source
        .find(&format!("fn {name}("))
        .unwrap_or_else(|| panic!("{name} missing"));
    let rest = &source[start..];
    &rest[..rest.find("\n}\n").expect("function end")]
}

/// Regression for the diffuse-only (formFactor, feature 39) mode across every visibility tier:
/// the area branch has exactly one shading call per mode, the LTC (the only specular term) sits
/// in the `else` of the feature-39 test, the diffuse function carries no specular, Fresnel or
/// LTC term, and the tiers (hero PCSS map, GI cone / stratified LocalVisibility slot) only
/// produce scalar visibilities that multiply the selected shading. A future tier that adds its
/// own radiance or reflection term to the area branch fails here.
#[test]
fn form_factor_mode_has_no_specular_in_any_tier() {
    let contribution = fn_body(LOCAL_LIGHTING_WGSL, "raa_local_contribution");
    let area = contribution
        .split("} else if light.kind == 3u {")
        .nth(1)
        .expect("area branch")
        .split("} else if light.kind != 2u {")
        .next()
        .expect("area branch end");
    // Code only: the branch's comments name what it must not call.
    let area: String = area
        .lines()
        .map(|line| line.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n");
    let area = area.as_str();
    // One shading call per mode in the whole module (definition + this call each).
    assert_eq!(LOCAL_LIGHTING_WGSL.matches("raa_area_ltc(").count(), 2);
    assert_eq!(LOCAL_LIGHTING_WGSL.matches("raa_area_diffuse(").count(), 2);
    assert_eq!(area.matches("raa_area_ltc(light, s, c)").count(), 1);
    assert_eq!(area.matches("raa_area_diffuse(light, s, c)").count(), 1);
    let gate = area
        .find("if raa_feature_enabled(RAA_AREA_FORM_FACTOR_FEATURE) {")
        .expect("feature 39 gate");
    let diffuse = area.find("raa_area_diffuse(light, s, c)").unwrap();
    let otherwise = area[gate..].find("} else {").expect("else of the gate") + gate;
    let ltc = area.find("raa_area_ltc(light, s, c)").unwrap();
    assert!(gate < diffuse && diffuse < otherwise && otherwise < ltc);
    // The only returned colour is the selected shading times scalar weights.
    assert_eq!(
        area.matches("return ").count(),
        4,
        "area branch returns changed"
    );
    assert!(area.contains("return shading * (window * area_visibility * fade);"));
    for forbidden in [
        "raa_slot_direct",
        "raa_area_radiance",
        "fresnel",
        "specular",
        "raa_ltc_uv",
    ] {
        assert!(!area.contains(forbidden), "area branch adds `{forbidden}`");
    }
    // The diffuse function: Lambert x form factor, no specular, Fresnel or LTC term.
    let ff = fn_body(LOCAL_LIGHTING_WGSL, "raa_area_diffuse");
    for forbidden in [
        "specular",
        "fresnel",
        "raa_ltc_uv",
        "raa_area_vector_form_factor",
        "raa_area_clipped",
        "texture",
        "raa_area_ltc",
    ] {
        assert!(
            !ff.contains(forbidden),
            "raa_area_diffuse holds `{forbidden}`"
        );
    }
    assert!(ff.contains("raa_area_form_factor(light, c.position, s.normal)"));
    // Tiers are scalar visibilities: hero PCSS returns (visibility, map weight), the
    // LocalVisibility slot (GI cone trace / stratified rays, or the default) returns f32.
    let hero = LOCAL_LIGHTING_WGSL
        .split("fn raa_area_hero_visibility(")
        .nth(1)
        .expect("hero tier");
    let hero = &hero[..hero.find("\n}\n").expect("hero end")];
    assert!(hero.contains(") -> vec2<f32> {"));
    for forbidden in ["raa_area_radiance", "raa_area_ltc", "fresnel", "specular"] {
        assert!(!hero.contains(forbidden), "hero tier holds `{forbidden}`");
    }
    for slot in [
        include_str!("../../raa-gi/src/wgsl/slots/slot_local_visibility.wgsl"),
        include_str!("../../raa-lit/src/wgsl/slots_default.wgsl"),
    ] {
        let body = fn_body(slot, "raa_slot_local_visibility");
        assert!(
            body.contains(") -> f32 {"),
            "LocalVisibility must stay scalar"
        );
        assert!(!body.contains("raa_area_ltc") && !body.contains("specular"));
    }
}
