//! CPU twins of the M6 area-shadow kernels (S44, contract `area-stochastic-cpu`): the stratum and
//! R2 jitter of the stratified shadow rays, the rectangle point, the sample weight, the temporal
//! and depth-consistency tests, the bilateral weight, the ratio estimator and the slot tables of
//! the mask and history textures. Solstice has no area-shadow kernels, so these are RAA's own
//! maths; the hash and the Gaussian come from `screen/cpu.rs` (`pcg`, `hash_to_unit`) and
//! `crates/raa-gtao/src/wgsl/denoise.wgsl:25-27` (`BILATERAL_GAUSS`). The WGSL twins of later
//! cards use the same names with an `raa_area_` prefix. These twins are tested on the CPU against
//! hand-derived fixtures and are never run against the GPU. Every function is pure and nothing
//! panics; u32 arithmetic wraps as in WGSL.

use super::super::screen::cpu::{hash_to_unit, pcg};

/// Blend weight of a usable history sample (`temporal_blend`).
pub const TEMPORAL_ALPHA: f32 = 0.1;

/// Relative depth tolerance of the reprojected history test (`depth_consistent`).
pub const DEPTH_TOLERANCE: f32 = 0.05;

/// Per-axis R2 sequence constants of the frame jitter (`r2_jitter`).
pub const R2_ALPHA: [f32; 2] = [0.754_877_7, 0.569_840_3];

/// One-dimensional Gaussian of the separable bilateral filter, seven taps centred on the pixel.
pub const BILATERAL_GAUSS: [f32; 7] = [
    0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125,
];

/// Distance in pixels between the taps of the spatial filter.
pub const SPATIAL_STRIDE: i32 = 2;

/// Exponent of the normal term of the bilateral weight.
pub const NORMAL_POWER: i32 = 8;

/// Below this shadow-ray count the ratio estimator reports full visibility (`ratio_visibility`).
pub const RATIO_EPSILON: f32 = 1e-6;

/// `fract(x) = x - floor(x)`, the WGSL definition (not Rust's `f32::fract`, which truncates).
fn fract(x: f32) -> f32 {
    x - x.floor()
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

/// The side of the stratum grid: 3 for nine cells, 2 otherwise (four cells).
pub fn cell_side(cells: u32) -> u32 {
    if cells == 9 { 3 } else { 2 }
}

/// The stratum a pixel's shadow ray uses this frame: the frame counter plus a per-pixel PCG
/// offset, modulo the cell count, so that `side * side` consecutive frames visit every cell once.
pub fn stratum_cell(cells: u32, frame: u32, pixel: [u32; 2]) -> u32 {
    let side = cell_side(cells);
    let offset = pcg(pixel[0]
        .wrapping_mul(7919)
        .wrapping_add(pixel[1].wrapping_mul(104_729)));
    frame.wrapping_add(offset) % (side * side)
}

/// The per-frame jitter inside a stratum, in `[0, 1)` per component: an R2 sequence over the frame
/// (wrapped every 4096 frames) plus a per-pixel rotation from the PCG hash.
pub fn r2_jitter(frame: u32, pixel: [u32; 2]) -> [f32; 2] {
    let k = (frame % 4096) as f32;
    let seed = pixel[0]
        .wrapping_mul(2957)
        .wrapping_add(pixel[1].wrapping_mul(6113));
    let rotation = [hash_to_unit(seed), hash_to_unit(seed.wrapping_add(1))];
    [
        fract(0.5 + k * R2_ALPHA[0] + rotation[0]),
        fract(0.5 + k * R2_ALPHA[1] + rotation[1]),
    ]
}

/// The unit-square position of a shadow-ray target: the jittered point of one stratum cell of a
/// `side x side` grid over the rectangle.
pub fn stratum_uv(cells: u32, cell: u32, jitter: [f32; 2]) -> [f32; 2] {
    let side = cell_side(cells);
    [
        ((cell % side) as f32 + jitter[0]) / side as f32,
        ((cell / side) as f32 + jitter[1]) / side as f32,
    ]
}

/// The world point of a rectangle light at unit-square coordinates `uv`: its centre plus the
/// tangent and bitangent scaled by the offset from the centre and the size.
pub fn rect_point(
    position: [f32; 3],
    tangent: [f32; 3],
    bitangent: [f32; 3],
    size: [f32; 2],
    uv: [f32; 2],
) -> [f32; 3] {
    let u = (uv[0] - 0.5) * size[0];
    let v = (uv[1] - 0.5) * size[1];
    std::array::from_fn(|i| position[i] + tangent[i] * u + bitangent[i] * v)
}

/// The geometric weight of one shadow sample: the cosine at the receiver and at the light, over
/// the squared distance (clamped at 1e-4). Zero at coincidence, where the direction is undefined.
pub fn sample_weight(
    receiver: [f32; 3],
    normal: [f32; 3],
    point: [f32; 3],
    travel: [f32; 3],
) -> f32 {
    let d: [f32; 3] = std::array::from_fn(|i| point[i] - receiver[i]);
    let r = length(d);
    if r == 0.0 {
        return 0.0;
    }
    let l = d.map(|c| c / r);
    let to_receiver = l.map(|c| -c);
    let cos_surface = dot(normal, l).max(0.0);
    let cos_light = dot(travel, to_receiver).max(0.0);
    cos_surface * cos_light / (r * r).max(1e-4)
}

/// Blends a reprojected history value towards the current one, or takes the current value when
/// the history is not usable.
pub fn temporal_blend(history: f32, current: f32, usable: bool) -> f32 {
    if usable {
        history + (current - history) * TEMPORAL_ALPHA
    } else {
        current
    }
}

/// Whether a reprojected linear depth agrees with the current one: the previous depth must be
/// positive and within `DEPTH_TOLERANCE` of the current depth, relative to it.
pub fn depth_consistent(previous: f32, current: f32) -> bool {
    previous > 0.0 && (previous - current).abs() <= DEPTH_TOLERANCE * current
}

/// The weight of one bilateral tap: the Gaussian times a depth term and a normal term
/// (`max(dot, 0)^NORMAL_POWER`). Taps past the seventh have no weight.
pub fn bilateral_weight(
    tap: usize,
    center_depth: f32,
    tap_depth: f32,
    center_normal: [f32; 3],
    tap_normal: [f32; 3],
) -> f32 {
    let Some(&gauss) = BILATERAL_GAUSS.get(tap) else {
        return 0.0;
    };
    let depth_scale = 0.02 * center_depth.max(1e-4);
    let depth_term = (-(tap_depth - center_depth).abs() / depth_scale).exp();
    let normal_term = dot(center_normal, tap_normal).max(0.0).powi(NORMAL_POWER);
    gauss * depth_term * normal_term
}

/// The ratio estimator (Heitz, Hill, McGuire, I3D 2018): visibility `s / u` clamped to [0, 1],
/// where `s` is the weighted visible sum and `u` the weight sum. With no weight, full visibility.
pub fn ratio_visibility(s: f32, u: f32) -> f32 {
    if u <= RATIO_EPSILON {
        1.0
    } else {
        (s / u).clamp(0.0, 1.0)
    }
}

/// The mask texel of a mid-tier slot: layer `slot / 4`, channel `slot % 4`.
pub fn mask_slot(slot: u32) -> (u32, u32) {
    (slot / 4, slot % 4)
}

/// The history texel of a mid-tier slot: group `slot / 2`, and the first of its two channels
/// (`S` at that channel, `U` at the next), `(slot % 2) * 2`.
pub fn history_slot(slot: u32) -> (u32, u32) {
    (slot / 2, (slot % 2) * 2)
}

/// The number of history groups for a slot limit: two slots per group, at most eight slots.
pub fn groups_for(limit: u32) -> u32 {
    limit.min(8).div_ceil(2)
}

/// The number of raw-visibility layers for a slot limit: the groups, and never fewer than two.
pub fn raw_layers(limit: u32) -> u32 {
    groups_for(limit).max(2)
}
