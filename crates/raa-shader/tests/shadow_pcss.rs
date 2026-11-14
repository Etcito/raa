//! Contract `shadow-pcss-kernel`: the shared PCSS kernel (`RAA_CSM_POISSON`,
//! `raa_csm_rotation`, `raa_csm_rotate`, `raa_shadow_blockers`, `raa_shadow_pcf`) is defined
//! once in `PRELUDE_WGSL`, after the S11 area form-factor section, and a fragment caller
//! validates against it.

use raa_shader::{Composer, PRELUDE_WGSL};

/// A fragment entry that calls the blocker search, the PCF and the rotation helper.
const FRAGMENT_CALLER: &str = "
@fragment
fn fs_main(@location(0) pos: vec3<f32>) -> @location(0) vec4<f32> {
    let uv = pos.xy;
    let b = raa_shadow_blockers(4, uv, 0.5, 8.0, 2048.0);
    let pcf = raa_shadow_pcf(4, uv, 0.5, 3.0, 2048.0, raa_csm_rotation(pos.xy));
    return vec4<f32>(b.x, b.y, pcf, 1.0);
}
";

/// The five contract names with their exact declaration text.
const SIGNATURES: [&str; 5] = [
    "const RAA_CSM_POISSON: array<vec2<f32>, 32>",
    "fn raa_csm_rotation(p: vec2<f32>) -> f32",
    "fn raa_csm_rotate(v: vec2<f32>, a: f32) -> vec2<f32>",
    "fn raa_shadow_blockers(layer: i32, uv: vec2<f32>, depth: f32, search_texels: f32, res: f32) -> vec2<f32>",
    "fn raa_shadow_pcf(layer: i32, uv: vec2<f32>, depth: f32, radius_texels: f32, res: f32, rotation: f32) -> f32",
];

#[test]
fn prelude_with_fragment_caller_validates() {
    let wgsl = format!("{PRELUDE_WGSL}\n{FRAGMENT_CALLER}");
    if let Err(e) = Composer::validate(&wgsl) {
        panic!("prelude + PCSS fragment caller: {e}");
    }
}

#[test]
fn kernel_is_defined_exactly_once_after_the_area_section() {
    for signature in SIGNATURES {
        assert_eq!(
            PRELUDE_WGSL.matches(signature).count(),
            1,
            "`{signature}` is not defined exactly once in PRELUDE_WGSL"
        );
    }
    let header =
        "// Shadow-map PCSS kernel shared by the csm Shadow provider and the hero area-light maps.";
    let start = PRELUDE_WGSL.find(header).expect("PCSS section header");
    let area_end = PRELUDE_WGSL
        .find("// End of the S11 area form-factor section.")
        .expect("area section end");
    assert!(
        area_end < start,
        "the PCSS section follows the area section"
    );
}
