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
