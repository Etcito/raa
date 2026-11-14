//! CPU twins of the milestone-2 screen-probe kernels (S39, contract `gi-screen-cpu`), ported from
//! solstice@7e465b5: `crates/solstice-render/shaders/common/math.wgsl` (`pcg` and `hash_to_unit`
//! at :30-39, `orthonormal_basis` at :61-70), `shaders/gi/probe_common.wgsl` (`probe_grid_size` at
//! :18-20, `probe_pixel` at :24-28, `hemisphere_direction` at :30-36, `probe_ray_direction` at
//! :38-50), `shaders/gi/probe_project.wgsl` (`sh_basis` at :13-25 and the mirrored projection at
//! :48-77) and `shaders/gi/probe_integrate.wgsl` (`sh_irradiance` at :16-28). MIT licence,
//! Copyright (c) 2026 SamG-Coder.
//!
//! The seeds and constants are those of contract `gi-screen-cpu`, which the WGSL twins share. The
//! temporal alpha is the S39 one; solstice's `frame_rate_blend` (`probe_integrate.wgsl:127`) is
//! not ported. Every function is pure, nothing panics, and u32 arithmetic wraps as in WGSL.

use std::f32::consts::{PI, TAU};

/// Spacing of the screen probes in pixels (`RAA_GI_PROBE_SPACING`, S39).
pub const PROBE_SPACING: u32 = 16;

/// PCG hash of `math.wgsl:30-39`: `state = v * 747796405 + 2891336453`,
/// `word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737`, then `(word >> 22) ^ word`.
pub fn pcg(v: u32) -> u32 {
    let state = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// The PCG hash mapped to `[0, 1)` by its top 24 bits (`hash_to_unit`, `math.wgsl:37-39`).
pub fn hash_to_unit(v: u32) -> f32 {
    (pcg(v) >> 8) as f32 / 16_777_216.0
}

/// An orthonormal basis around `n` (Duff et al. 2017, `math.wgsl:61-70`), as three columns:
/// tangent, bitangent and `n` itself. `s` is 1 when `n.z >= 0` and -1 otherwise.
pub fn orthonormal_basis(n: [f32; 3]) -> [[f32; 3]; 3] {
    let [x, y, z] = n;
    let s = if z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (s + z);
    let b = x * y * a;
    [
        [1.0 + s * x * x * a, s * b, -s * x],
        [b, s + y * y * a, -y],
        n,
    ]
}

/// The nine real spherical-harmonic basis values of a unit direction `d`, in the order of
/// `probe_project.wgsl:13-25` (`sh_basis`).
pub fn sh9_basis(d: [f32; 3]) -> [f32; 9] {
    let [x, y, z] = d;
    [
        0.282095,
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ]
}

/// Handoff: card 'raa-gi screen: twin tests and hand-derived fixtures' calls `probe_grid`,
/// `probe_pixel`, `ray_uv`, `hemisphere_direction`, `sh9_project`, `sh9_irradiance` and
/// `temporal_alpha` of this module (`raa_gi::parts::screen::cpu`, reached through `pub mod cpu` in
/// `crates/raa-gi/src/parts/screen.rs`). Today they are pure functions: the state passed is none,
/// and nothing is stored between calls.
///
/// Grid size of the screen probes, `[ceil(w / 16), ceil(h / 16)]` with zero staying zero
/// (the WGSL `probe_grid_size`, `probe_common.wgsl:18-20`). The sum wraps as the WGSL sum does.
pub fn probe_grid(viewport: [u32; 2]) -> [u32; 2] {
    viewport.map(|v| v.wrapping_add(PROBE_SPACING - 1) / PROBE_SPACING)
}

/// The pixel a probe sits on this frame (`probe_pixel`, `probe_common.wgsl:24-28`): the probe's
/// tile origin plus a PCG jitter in `[0, 16)` per axis, clamped to the last viewport pixel.
/// The seed is `tile.x * 7919 + tile.y * 104729 + (frame % 16) * 31337`.
pub fn probe_pixel(tile: [u32; 2], viewport: [u32; 2], frame: u32) -> [u32; 2] {
    let seed = tile[0]
        .wrapping_mul(7919)
        .wrapping_add(tile[1].wrapping_mul(104_729))
        .wrapping_add((frame % 16).wrapping_mul(31_337));
    let jitter = [
        pcg(seed) % PROBE_SPACING,
        pcg(seed.wrapping_add(1)) % PROBE_SPACING,
    ];
    std::array::from_fn(|axis| {
        let pixel = tile[axis]
            .wrapping_mul(PROBE_SPACING)
            .wrapping_add(jitter[axis]);
        pixel.min(viewport[axis].saturating_sub(1))
    })
}

/// The stratified, per-frame rotated uv of one probe ray (`probe_ray_direction`,
/// `probe_common.wgsl:38-50`, before its hemisphere mapping). Each axis of `ray + jitter` is
/// divided by `rays`, so `uv.x` lies in `[ray.x / rays, (ray.x + 1) / rays)` and `uv.y` in
/// `[0, 1)` after `fract(uv.y + rotation)`. `rays` is the ray count per side (6 or 8); zero gives
/// non-finite values but no panic.
pub fn ray_uv(tile: [u32; 2], ray: [u32; 2], frame: u32, rays: u32) -> [f32; 2] {
    let seed = tile[0]
        .wrapping_mul(2957)
        .wrapping_add(tile[1].wrapping_mul(6113))
        .wrapping_add(frame.wrapping_mul(409));
    let jitter = [
        hash_to_unit(
            seed.wrapping_add(ray[0].wrapping_mul(17))
                .wrapping_add(ray[1].wrapping_mul(257)),
        ),
        hash_to_unit(
            seed.wrapping_add(ray[0].wrapping_mul(131))
                .wrapping_add(ray[1].wrapping_mul(7))
                .wrapping_add(1),
        ),
    ];
    let rotation = hash_to_unit(seed.wrapping_add(3));
    let scale = rays as f32;
    let u = (ray[0] as f32 + jitter[0]) / scale;
    let v = (ray[1] as f32 + jitter[1]) / scale + rotation;
    [u, v - v.floor()]
}

/// Maps a uv of `[0, 1]^2` onto the unit hemisphere around `normal` with even area
/// (`hemisphere_direction`, `probe_common.wgsl:30-36`): `cos = 1 - uv.x`, `phi = 2 pi uv.y`, and
/// the result is `orthonormal_basis(normal)` times `(sin cos phi, sin sin phi, cos)`.
pub fn hemisphere_direction(uv: [f32; 2], normal: [f32; 3]) -> [f32; 3] {
    let cos_t = 1.0 - uv[0];
    let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
    let phi = TAU * uv[1];
    let local = [sin_t * phi.cos(), sin_t * phi.sin(), cos_t];
    let [col0, col1, col2] = orthonormal_basis(normal);
    std::array::from_fn(|i| col0[i] * local[0] + col1[i] * local[1] + col2[i] * local[2])
}

/// Projects `(unit direction, linear rgb radiance)` samples of the hemisphere around +Z into nine
/// rgb SH coefficients (`probe_project.wgsl:46-77`). With `N` samples, `w = 2 pi / N`, and `mean`
/// the average radiance, coefficient `i` accumulates `(radiance * basis(d)[i] + mean *
/// basis(mirrored)[i]) * w`, where `mirrored = (d.x, d.y, -d.z)`. An empty slice gives zeros.
pub fn sh9_project(samples: &[([f32; 3], [f32; 3])]) -> [[f32; 3]; 9] {
    let mut sh = [[0.0; 3]; 9];
    if samples.is_empty() {
        return sh;
    }
    let count = samples.len() as f32;
    let weight = TAU / count;
    let mut mean = [0.0f32; 3];
    for (_, radiance) in samples {
        for (m, c) in mean.iter_mut().zip(radiance) {
            *m += c;
        }
    }
    let mean = mean.map(|m| m / count);
    for (direction, radiance) in samples {
        let up = sh9_basis(*direction);
        let down = sh9_basis([direction[0], direction[1], -direction[2]]);
        for (i, coeff) in sh.iter_mut().enumerate() {
            for (ch, value) in coeff.iter_mut().enumerate() {
                *value += (radiance[ch] * up[i] + mean[ch] * down[i]) * weight;
            }
        }
    }
    sh
}

/// The cosine-convolved irradiance towards `normal` from nine rgb SH coefficients
/// (`sh_irradiance`, `probe_integrate.wgsl:16-28`): band 0 times pi, band 1 times 2 pi / 3 and
/// band 2 times pi / 4, each channel clamped to be non-negative. A constant radiance `c` over a
/// symmetric direction set gives `pi c`.
pub fn sh9_irradiance(sh: &[[f32; 3]; 9], normal: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = normal;
    let a0 = PI;
    let a1 = TAU / 3.0;
    let a2 = PI * 0.25;
    let band2 = [
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ];
    let mut out = [0.0; 3];
    for (ch, value) in out.iter_mut().enumerate() {
        let e0 = sh[0][ch] * 0.282095 * a0;
        let e1 = (sh[1][ch] * y + sh[2][ch] * z + sh[3][ch] * x) * 0.488603 * a1;
        let e2 = sh[4..]
            .iter()
            .zip(band2)
            .map(|(c, b)| c[ch] * b)
            .sum::<f32>()
            * a2;
        *value = (e0 + e1 + e2).max(0.0);
    }
    out
}

/// The temporal blend weight of a screen probe pixel (S39): 1 when the history is invalid, else
/// `min(0.08 + 0.03 * moved_pixels, 0.5)`. `moved_pixels` is the screen-space motion in pixels.
pub fn temporal_alpha(moved_pixels: f32, history_valid: bool) -> f32 {
    if history_valid {
        (0.08 + 0.03 * moved_pixels).min(0.5)
    } else {
        1.0
    }
}
