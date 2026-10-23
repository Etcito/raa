//! Full WGSL modules of the `sky.env` pass and the `sky.background` contributor.
//!
//! Each module is `PRELUDE_WGSL + LIB_WGSL + SKY_WGSL +` its entry file, so groups 0 (frame)
//! and 1 (lighting) are declared exactly as in the forward pass. Self-contained (no `crate::`
//! paths) so `tests/variants.rs` can include it with `#[path]`.

use raa_shader::{LIB_WGSL, PRELUDE_WGSL};

/// The same file as `SKY_WGSL`.
const SKY_SRC: &str = include_str!("wgsl/sky.wgsl");
/// Entry points of the env cube pass (one face per instance).
const SKY_ENV_SRC: &str = include_str!("wgsl/sky_env.wgsl");
/// Entry points of the background contributor.
const SKY_BACKGROUND_SRC: &str = include_str!("wgsl/sky_background.wgsl");

fn assemble(entry: &str) -> String {
    let mut s = String::with_capacity(
        PRELUDE_WGSL.len() + LIB_WGSL.len() + SKY_SRC.len() + entry.len() + 3,
    );
    for part in [PRELUDE_WGSL, LIB_WGSL, SKY_SRC, entry] {
        s.push_str(part);
        s.push('\n');
    }
    s
}

/// The `sky.env` module: `vs_main` draws a fullscreen triangle per cube face
/// (`instance_index` = face), `fs_main` writes the sky radiance to an HDR_FORMAT target.
#[allow(dead_code)] // consumed by SkyFeature (sky.env, sky.background) and tests/variants.rs
pub(crate) fn env_module_wgsl() -> String {
    assemble(SKY_ENV_SRC)
}

/// The `sky.background` module: fullscreen triangle at clip z = 1, per-pixel sky radiance
/// along the reconstructed world view ray.
#[allow(dead_code)] // consumed by SkyFeature (sky.env, sky.background) and tests/variants.rs
pub(crate) fn background_module_wgsl() -> String {
    assemble(SKY_BACKGROUND_SRC)
}

/// CPU port of `raa_sky_env_face_dir`: unit direction of face coordinates (u, v) in [-1, 1]
/// (u grows with the texel column, v with the row) on cube face `face` in wgpu order
/// +X, -X, +Y, -Y, +Z, -Z. Same mapping as raa-ibl's SH projection.
#[allow(dead_code)] // consumed by SkyFeature (sky.env, sky.background) and tests/variants.rs
pub(crate) fn cube_face_direction(face: u32, u: f32, v: f32) -> [f32; 3] {
    let d = match face {
        0 => [1.0, -v, -u],
        1 => [-1.0, -v, u],
        2 => [u, 1.0, v],
        3 => [u, -1.0, -v],
        4 => [u, -v, 1.0],
        _ => [-u, -v, -1.0],
    };
    // One component is +-1, so the length is >= 1 for finite (u, v).
    let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    [d[0] / l, d[1] / l, d[2] / l]
}
