//! Q11: SKY_WGSL composes after the S11 prelude and helper library and passes naga validation,
//! every function it defines is the contract function or a `raa_sky_` helper, the `sky.env` and
//! `sky.background` modules validate, and the env cube faces follow the raa-ibl convention.

use raa_shader::{Composer, LIB_WGSL, PRELUDE_WGSL};
use raa_sky::SKY_WGSL;

/// Minimal fragment entry evaluating the sky along a direction derived from the pixel, with
/// the frame's sun and the environment uniform, as the background contributor does.
const ENTRY: &str = "
@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag.xy * raa_frame.viewport.zw * 2.0 - vec2<f32>(1.0);
    let dir = normalize(vec3<f32>(uv.x, uv.y, -1.0));
    let c = raa_sky_radiance(dir, raa_env, raa_frame.sun_dir.xyz, raa_frame.sun_color.rgb, raa_frame.camera_pos.w);
    return vec4<f32>(c, 1.0);
}
";

#[test]
fn sky_wgsl_validates() {
    let wgsl = format!("{PRELUDE_WGSL}\n{LIB_WGSL}\n{SKY_WGSL}\n{ENTRY}");
    if let Err(e) = Composer::validate(&wgsl) {
        panic!("sky shader refused: {e}");
    }
}

#[test]
fn sky_helpers_are_prefixed() {
    let mut names = Vec::new();
    for line in SKY_WGSL.lines() {
        let line = line.trim_start();
        if let Some(rest) = line.strip_prefix("fn ") {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            names.push(name);
        }
    }
    assert!(
        names.iter().any(|n| n == "raa_sky_radiance"),
        "raa_sky_radiance missing: {names:?}"
    );
    for name in &names {
        assert!(
            name == "raa_sky_radiance" || name.starts_with("raa_sky_"),
            "helper {name} is not prefixed raa_sky_"
        );
    }
}

#[path = "../src/shaders.rs"]
mod shaders;

#[test]
fn env_and_background_modules_validate() {
    for (name, wgsl) in [
        ("sky.env", shaders::env_module_wgsl()),
        ("sky.background", shaders::background_module_wgsl()),
    ] {
        assert!(wgsl.contains(SKY_WGSL), "{name} does not embed SKY_WGSL");
        assert!(
            wgsl.starts_with(PRELUDE_WGSL),
            "{name} does not start with the prelude"
        );
        if let Err(e) = Composer::validate(&wgsl) {
            panic!("{name} shader refused: {e}");
        }
    }
}

#[test]
fn env_face_directions_match_ibl_convention() {
    let centres = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    for (face, want) in centres.iter().enumerate() {
        let got = shaders::cube_face_direction(face as u32, 0.0, 0.0);
        for c in 0..3 {
            assert!(
                (got[c] - want[c]).abs() < 1e-6,
                "face {face} centre {got:?}, expected {want:?}"
            );
        }
    }
    // Off-centre orientation of the wgpu cube convention (raa-ibl sh.rs): on +X, u grows
    // toward -Z and v (rows, downwards) toward -Y; on +Y, v grows toward +Z.
    let d = shaders::cube_face_direction(0, 1.0, 1.0);
    let s = 1.0 / 3.0f32.sqrt();
    for (c, want) in [s, -s, -s].iter().enumerate() {
        assert!((d[c] - want).abs() < 1e-6, "+X corner {d:?}");
    }
    let d = shaders::cube_face_direction(2, 0.0, 1.0);
    assert!(d[2] > 0.7 && d[1] > 0.7, "+Y bottom edge {d:?}");
    // Unit length everywhere on every face.
    for face in 0..6 {
        for (u, v) in [(-1.0, -1.0), (0.3, -0.7), (1.0, 0.5)] {
            let d = shaders::cube_face_direction(face, u, v);
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-6,
                "face {face} ({u}, {v}) length {len}"
            );
        }
    }
}

/// The adapted physical-atmosphere helper source (S23); no bindings, functions prefixed
/// `raa_atmosphere_`, and the S10 `PhysicalSkyGpu` record.
const ATMOSPHERE_WGSL: &str = include_str!("../src/wgsl/atmosphere.wgsl");

/// Minimal fragment entry binding the S10 group-2 record and exercising every helper.
const ATMOSPHERE_ENTRY: &str = "
@group(2) @binding(0) var<uniform> raa_sky: PhysicalSkyGpu;

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    let direction = raa_atmosphere_sky_view_direction(vec2<f32>(0.5, 0.5));
    let medium = raa_atmosphere_sample_medium(1.0, raa_sky);
    let transmittance = raa_atmosphere_transmittance_uv(raa_sky.viewer.w, 1.0, raa_sky);
    let params = raa_atmosphere_transmittance_params(transmittance, raa_sky);
    let hit = raa_atmosphere_ray_sphere(
        vec3<f32>(0.0, raa_sky.viewer.w, 0.0), direction, raa_sky.planet.y,
    );
    let phase = raa_atmosphere_rayleigh_phase(0.5) + raa_atmosphere_mie_phase(0.5, raa_sky.mie.w);
    let uv = raa_atmosphere_sky_view_uv(direction);
    return vec4<f32>(
        medium.extinction.x + medium.mie + transmittance.x + params.y + hit + phase + uv.x + uv.y,
        0.0, 0.0, 1.0,
    );
}
";

#[test]
fn atmosphere_wgsl_validates_and_helpers_are_prefixed() {
    let wgsl = format!("{PRELUDE_WGSL}\n{LIB_WGSL}\n{ATMOSPHERE_WGSL}\n{ATMOSPHERE_ENTRY}");
    if let Err(e) = Composer::validate(&wgsl) {
        panic!("atmosphere shader refused: {e}");
    }
    for name in [
        "sample_medium",
        "rayleigh_phase",
        "mie_phase",
        "ray_sphere",
        "transmittance_uv",
        "transmittance_params",
        "sky_view_uv",
        "sky_view_direction",
    ] {
        assert!(
            ATMOSPHERE_WGSL.contains(&format!("fn raa_atmosphere_{name}(")),
            "raa_atmosphere_{name} missing"
        );
    }
    let mut names = Vec::new();
    for line in ATMOSPHERE_WGSL.lines() {
        let line = line.trim_start();
        if let Some(rest) = line.strip_prefix("fn ") {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            names.push(name);
        }
    }
    assert!(!names.is_empty(), "atmosphere helpers are empty");
    for name in names {
        assert!(
            name.starts_with("raa_atmosphere_"),
            "helper {name} is not prefixed raa_atmosphere_"
        );
    }
    // The source constants stay in the helper file, parameterised by the S10 record.
    for constant in [
        "raa_atmosphere_ground_radius_km: f32 = 6360.0",
        "raa_atmosphere_top_radius_km: f32 = 6460.0",
        "vec3<f32>(5.802e-3, 13.558e-3, 33.1e-3)",
        "raa_atmosphere_mie_scale_height_km: f32 = 1.6",
        "vec3<f32>(0.650e-3, 1.881e-3, 0.085e-3)",
    ] {
        assert!(
            ATMOSPHERE_WGSL.contains(constant),
            "source constant missing: {constant}"
        );
    }
}

#[test]
fn legacy_sky_remains_standalone() {
    // The physical helper is a separate source: the legacy constant composition must
    // neither embed the S10 record nor the atmosphere helpers.
    assert!(!SKY_WGSL.contains("PhysicalSkyGpu"));
    assert!(!SKY_WGSL.contains("raa_atmosphere_"));
    let wgsl = format!("{PRELUDE_WGSL}\n{LIB_WGSL}\n{SKY_WGSL}\n{ENTRY}");
    if let Err(e) = Composer::validate(&wgsl) {
        panic!("legacy sky shader refused: {e}");
    }
}
