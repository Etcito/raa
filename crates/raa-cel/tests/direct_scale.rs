//! Cel toon direct term brightness: the lit ramp row (an albedo) times radiance / pi, the
//! Lambert term of raa-brdf at n.l = 1, so a mid-grey surface under a sun of intensity 3
//! stays below display white instead of clipping to (1, 1, 1).

const DIRECT_WGSL: &str = include_str!("../src/wgsl/direct.wgsl");

/// The f32 value of `const <name>: f32 = <value>;` in the cel direct WGSL.
fn wgsl_const(name: &str) -> f32 {
    let head = format!("const {name}: f32 = ");
    let start = DIRECT_WGSL.find(&head).expect("constant declared") + head.len();
    let end = start + DIRECT_WGSL[start..].find(';').expect("constant ends");
    DIRECT_WGSL[start..end].trim().parse().expect("f32 literal")
}

#[test]
fn toon_direct_divides_the_ramp_colour_by_pi() {
    let k = wgsl_const("RAA_CEL_FRAC_1_PI");
    assert!((k - std::f32::consts::FRAC_1_PI).abs() < 1e-7, "got {k}");
    assert!(
        DIRECT_WGSL.contains("return raa_srgb_to_linear(color)*RAA_CEL_FRAC_1_PI*radiance;"),
        "the toon direct term must scale the ramp colour by 1/pi"
    );
}

#[test]
fn lit_band_of_mid_grey_under_sun_three_matches_lambert_and_stays_below_white() {
    // Band 3 of raa_cel_palette is the anchor itself (anchor * 1.0 + 0): in linear light the
    // direct term is albedo * k * radiance.
    let k = wgsl_const("RAA_CEL_FRAC_1_PI");
    let (albedo, radiance) = (0.5_f32, 3.0_f32);
    let cel = albedo * k * radiance;
    // Independently: Lambert diffuse albedo / pi * radiance * n.l at n.l = 1.
    let lambert = albedo / std::f32::consts::PI * radiance;
    assert!((cel - lambert).abs() < 1e-6, "cel {cel} lambert {lambert}");
    assert!((cel - 0.477_464_8).abs() < 1e-5, "cel {cel}");
    assert!(cel < 1.0);
}
