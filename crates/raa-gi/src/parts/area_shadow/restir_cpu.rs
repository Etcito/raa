//! CPU twins of the M7 ReSTIR kernels (S12 order 43, contract `area-restir-cpu`), share 1: the
//! constants, the binary16 pair codec, the octahedral normal codec and the packed reservoir
//! record. These are RAA's own maths from Bitterli et al. 2020 (ReSTIR), Wyman and Panteleev HPG
//! 2021, Lin et al. 2022 (GRIS) and Urena et al. EGSR 2013; Bevy Solari is not supplied and
//! nothing is copied from it or from NVIDIA RTXDI. The WGSL twins (`restir_common.wgsl`) use the
//! same names with an `raa_restir_` prefix. These twins are tested on the CPU against hand-derived
//! fixtures and are never run against the GPU. Every function is pure and nothing panics; u32
//! arithmetic wraps as in WGSL.
//!
//! Share 2 adds the reservoir reuse maths: the weighted reservoir sampling acceptance, the
//! contribution weight, the confidence clamp, the generalized balance heuristic, the slot remap,
//! the boiling filter, the depth test and the hashed spatial offsets.

use std::f32::consts::PI;

use super::super::screen::cpu::hash_to_unit;

/// Initial candidates per pixel of the ReSTIR pass.
pub const RESTIR_CANDIDATES: u32 = 8;

/// Spatial-reuse neighbours sampled per pixel.
pub const RESTIR_SPATIAL_SAMPLES: u32 = 4;

/// Spatial-reuse radius in pixels.
pub const RESTIR_SPATIAL_RADIUS: f32 = 30.0;

/// Per-pixel cap of the temporal and spatial candidate count M.
pub const RESTIR_M_CAP: f32 = 20.0;

/// Boiling-filter threshold as a multiple of the mean reservoir weight.
pub const RESTIR_BOILING_FACTOR: f32 = 8.0;

/// Cosine threshold of the reservoir normal rejection.
pub const RESTIR_NORMAL_COS: f32 = 0.9;

/// Relative depth tolerance of the reservoir depth rejection.
pub const RESTIR_DEPTH_TOLERANCE: f32 = 0.1;

/// The slot value of an empty reservoir.
pub const RESERVOIR_EMPTY_SLOT: u32 = 0xFF;

/// The remap value of a slot whose occupant generation did not survive (`remap_table`).
pub const REMAP_INVALID: u32 = 0xFFFF_FFFF;

/// The largest finite binary16 magnitude.
pub const HALF_MAX: f32 = 65504.0;

/// Luminance weights of the RGB channels, as used by `target_luminance`.
pub const LUMINANCE: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Floor of the GGX roughness before it is squared (`target_luminance`).
pub const ROUGHNESS_MIN: f32 = 0.0525;

/// `sign(x)` with `sign(0) = +1`, the octahedral convention (WGSL's `sign` would give 0).
fn sign_one(x: f32) -> f32 {
    if x < 0.0 { -1.0 } else { 1.0 }
}

/// Right-shifts `value` by `shift` bits, rounding to nearest even.
fn round_shift(value: u32, shift: u32) -> u32 {
    if shift == 0 {
        return value;
    }
    let half = 1u32 << (shift - 1);
    let mask = (1u32 << shift) - 1;
    let rem = value & mask;
    let base = value >> shift;
    if rem > half || (rem == half && base & 1 == 1) {
        base + 1
    } else {
        base
    }
}

/// Packs one f32 into IEEE binary16 bits, round to nearest even, magnitudes above `HALF_MAX`
/// saturating to it and NaN packing 0.
fn pack_half(x: f32) -> u16 {
    if x.is_nan() {
        return 0;
    }
    let sign = if x.is_sign_negative() { 0x8000u16 } else { 0 };
    let a = x.abs();
    if a == 0.0 {
        return sign;
    }
    if a > HALF_MAX {
        return sign | 0x7BFF;
    }
    let bits = a.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mantissa = bits & 0x7F_FFFF;
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        let m = mantissa | 0x80_0000;
        let rounded = round_shift(m, (14 - exponent) as u32);
        return sign | rounded as u16;
    }
    let sig = round_shift(mantissa | 0x80_0000, 13);
    let h = ((exponent as u32) << 10) + (sig - 0x400);
    sign | h as u16
}

/// Unpacks IEEE binary16 bits into one f32 (subnormals and infinities included).
fn unpack_half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = (bits & 0x3FF) as f32;
    match exponent {
        0 => sign * mantissa * 2.0f32.powi(-24),
        0x1F => {
            if mantissa == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1.0 + mantissa / 1024.0) * 2.0f32.powi(exponent as i32 - 15),
    }
}

/// Packs two f32 into binary16 pairs: `v[0]` in bits 0-15 and `v[1]` in bits 16-31, round to
/// nearest even. Magnitudes above `HALF_MAX` saturate to it and NaN packs 0.
pub fn pack2x16float(v: [f32; 2]) -> u32 {
    u32::from(pack_half(v[0])) | (u32::from(pack_half(v[1])) << 16)
}

/// Unpacks the two binary16 halves of `bits`: `[low, high]`.
pub fn unpack2x16float(bits: u32) -> [f32; 2] {
    [unpack_half(bits as u16), unpack_half((bits >> 16) as u16)]
}

/// Octahedral encoding of a unit vector: `p = n / (|x| + |y| + |z|)`, and for `z < 0` the folded
/// `(1 - |p.yx|) * sign(p.xy)` with `sign(0) = +1`. A zero vector encodes `[0, 0]`.
pub fn oct_encode(n: [f32; 3]) -> [f32; 2] {
    let l1 = n[0].abs() + n[1].abs() + n[2].abs();
    if l1 <= 0.0 {
        return [0.0, 0.0];
    }
    let p = [n[0] / l1, n[1] / l1];
    let z = n[2] / l1;
    if z < 0.0 {
        [
            (1.0 - p[1].abs()) * sign_one(p[0]),
            (1.0 - p[0].abs()) * sign_one(p[1]),
        ]
    } else {
        p
    }
}

/// The unit vector of an octahedral coordinate: the inverse of [`oct_encode`], normalised.
pub fn oct_decode(e: [f32; 2]) -> [f32; 3] {
    let z = 1.0 - e[0].abs() - e[1].abs();
    let (x, y) = if z < 0.0 {
        (
            (1.0 - e[1].abs()) * sign_one(e[0]),
            (1.0 - e[0].abs()) * sign_one(e[1]),
        )
    } else {
        (e[0], e[1])
    };
    let len = (x * x + y * y + z * z).sqrt();
    [x / len, y / len, z / len]
}

/// One ReSTIR candidate, packed into four u32 words per pixel.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Reservoir {
    /// The stable area-stochastic slot of the sampled light.
    pub slot: u32,
    /// The unit-square point of the rectangle light.
    pub uv: [f32; 2],
    /// The reservoir weight (the unbiased contribution weight).
    pub weight: f32,
    /// The candidate count M of the reservoir.
    pub confidence: f32,
    /// The octahedral normal used for the reuse rejection.
    pub normal_oct: [f32; 2],
}

impl Reservoir {
    /// The empty reservoir: slot `RESERVOIR_EMPTY_SLOT` and every other field zero.
    pub const EMPTY: Reservoir = Reservoir {
        slot: RESERVOIR_EMPTY_SLOT,
        uv: [0.0, 0.0],
        weight: 0.0,
        confidence: 0.0,
        normal_oct: [0.0, 0.0],
    };
}

/// Clamps to `[0, HALF_MAX]`, mapping NaN and negatives to zero (the pack halves' domain).
fn half_range(x: f32) -> f32 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(0.0, HALF_MAX)
    }
}

/// Packs a reservoir into its four WGSL words: `x = slot & 0xFF`,
/// `y = pack2x16float(uv clamped to [0, 1])`,
/// `z = pack2x16float([weight, confidence] clamped to [0, HALF_MAX], NaN -> 0)` and
/// `w = pack2x16float(normal_oct)`.
///
/// Handoff: card 'raa-gi: ReSTIR CPU twins share 2 - reservoir reuse maths and the slot remap'
/// extends `restir_cpu.rs` and `area_shadow_twins.rs` and calls nothing new here. Today
/// `pack_reservoir` packs a reservoir into the four words [`unpack_reservoir`] reads back, and
/// this file holds share 1 only: the constants and the half-float, octahedral and reservoir
/// codecs.
pub fn pack_reservoir(r: &Reservoir) -> [u32; 4] {
    let uv =
        r.uv.map(|c| if c.is_nan() { 0.0 } else { c.clamp(0.0, 1.0) });
    [
        r.slot & 0xFF,
        pack2x16float(uv),
        pack2x16float([half_range(r.weight), half_range(r.confidence)]),
        pack2x16float(r.normal_oct),
    ]
}

/// Unpacks the four WGSL words of [`pack_reservoir`]; the slot keeps its low eight bits.
pub fn unpack_reservoir(words: [u32; 4]) -> Reservoir {
    let uv = unpack2x16float(words[1]);
    let weight_confidence = unpack2x16float(words[2]);
    Reservoir {
        slot: words[0] & 0xFF,
        uv,
        weight: weight_confidence[0],
        confidence: weight_confidence[1],
        normal_oct: unpack2x16float(words[3]),
    }
}

/// Weighted reservoir sampling (Bitterli et al. 2020): accepts a candidate of weight `w` when
/// `w_sum` (which already includes it) is positive and `xi * w_sum < w`, `xi` in `[0, 1)`.
pub fn wrs_accept(w_sum: f32, w: f32, xi: f32) -> bool {
    w_sum > 0.0 && xi * w_sum < w
}

/// The unbiased contribution weight `w_sum / target`; 0 when `target <= 0` or the quotient is
/// not finite.
pub fn contribution_weight(w_sum: f32, target: f32) -> f32 {
    if target <= 0.0 {
        return 0.0;
    }
    let weight = w_sum / target;
    if weight.is_finite() { weight } else { 0.0 }
}

/// Clamps the previous candidate count against `cap * current`, never below 0:
/// `max(min(previous, cap * current), 0)`.
pub fn clamp_confidence(previous: f32, current: f32, cap: f32) -> f32 {
    previous.min(cap * current).max(0.0)
}

/// The generalized balance heuristic (Lin et al. 2022) of term `i`: terms are
/// `(M_j, target_j)`, and the result is `M_i * target_i / sum_j M_j * target_j`. 0 when the sum
/// is not positive or `i` is out of range.
pub fn balance_weight(terms: &[(f32, f32)], i: usize) -> f32 {
    if i >= terms.len() {
        return 0.0;
    }
    let sum: f32 = terms.iter().map(|&(m, target)| m * target).sum();
    if sum <= 0.0 {
        return 0.0;
    }
    let (m, target) = terms[i];
    m * target / sum
}

/// Maps a reservoir's stable area-stochastic slot through a [`remap_table`] row: `None` for
/// `slot >= 8` or a lane equal to [`REMAP_INVALID`].
pub fn remap_slot(remap: &[u32; 8], slot: u32) -> Option<u32> {
    if slot >= 8 {
        return None;
    }
    let lane = remap[slot as usize];
    if lane == REMAP_INVALID {
        None
    } else {
        Some(lane)
    }
}

/// Slot remap from the per-slot occupant generations of `AreaStochasticFrame`: lane `s` is `s`
/// when `history_valid` and `previous[s] == current[s]`, else [`REMAP_INVALID`]. RAA repacks
/// packed light indices every frame, so a reservoir stores the stable slot and reads this
/// frame's packed index from the slot table.
///
/// Handoff: card 'raa-gi: gi.area-shadow part runs the ReSTIR ultra tier on Restir frames'
/// calls `remap_table` with the previous and current `AreaStochasticFrame.generations` and the
/// reservoir history validity. Today `remap_table` compares the two generations lane by lane
/// and returns this frame's slot or [`REMAP_INVALID`].
pub fn remap_table(previous: &[u32; 8], current: &[u32; 8], history_valid: bool) -> [u32; 8] {
    std::array::from_fn(|s| {
        if history_valid && previous[s] == current[s] {
            s as u32
        } else {
            REMAP_INVALID
        }
    })
}

/// The boiling filter: keeps a reservoir when `mean` is not positive or `weight <= factor *
/// mean`, so a reservoir far above the running mean is dropped instead of boiling.
pub fn boiling_keep(weight: f32, mean: f32, factor: f32) -> bool {
    mean <= 0.0 || weight <= factor * mean
}

/// The temporal depth rejection: both depths are positive and the neighbour is within
/// [`RESTIR_DEPTH_TOLERANCE`] (relative) of the centre.
pub fn depth_similar(neighbour: f32, centre: f32) -> bool {
    centre > 0.0 && neighbour > 0.0 && (neighbour - centre).abs() <= RESTIR_DEPTH_TOLERANCE * centre
}

/// The hashed disk offset of spatial-reuse sample `k`: the seed is
/// `pixel.x * 7919 + pixel.y * 104729 + frame * 31337 + k * 977` (wrapping), radius
/// `r = radius * sqrt(hash_to_unit(seed))` and angle `a = 2 pi hash_to_unit(seed + 1)`,
/// returning `[round(r cos a), round(r sin a)]`.
pub fn spatial_offset(pixel: [u32; 2], frame: u32, k: u32, radius: f32) -> [i32; 2] {
    let seed = pixel[0]
        .wrapping_mul(7919)
        .wrapping_add(pixel[1].wrapping_mul(104_729))
        .wrapping_add(frame.wrapping_mul(31_337))
        .wrapping_add(k.wrapping_mul(977));
    let r = radius * hash_to_unit(seed).sqrt();
    let a = 2.0 * PI * hash_to_unit(seed.wrapping_add(1));
    [(r * a.cos()).round() as i32, (r * a.sin()).round() as i32]
}
