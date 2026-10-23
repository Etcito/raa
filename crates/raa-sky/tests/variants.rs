//! Q11: SKY_WGSL composes after the S11 prelude and helper library and passes naga validation,
//! and every function it defines is the contract function or a `raa_sky_` helper.

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
