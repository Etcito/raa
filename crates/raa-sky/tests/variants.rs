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
