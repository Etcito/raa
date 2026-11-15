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
//! the boiling filter, the depth test and the hashed spatial offsets. Share 3 adds the
//! spherical-rectangle solid angle and sampling (Urena et al. EGSR 2013), the area PDF, the sRGB
//! decode, luminance and the Lambert + GGX target.

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

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn len3(a: [f32; 3]) -> f32 {
    dot3(a, a).sqrt()
}

/// Normalises `a`, or returns the zero vector when it has no finite length (no NaN escapes).
fn normalized(a: [f32; 3]) -> [f32; 3] {
    let len = len3(a);
    if len > 0.0 && len.is_finite() {
        [a[0] / len, a[1] / len, a[2] / len]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// `num / den`, or 0 when `den` is not positive and finite.
fn ratio(num: f32, den: f32) -> f32 {
    if den > 0.0 && den.is_finite() {
        num / den
    } else {
        0.0
    }
}

/// The local frame of a spherical rectangle (Urena et al. EGSR 2013): the edge lengths and the
/// corner coordinates in the rectangle's `(x, y, z)` frame, the two `z` components of the edge
/// normals and the angles that give the solid angle.
struct RectFrame {
    exl: f32,
    eyl: f32,
    z0: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    b0: f32,
    b1: f32,
    k: f32,
    solid_angle: f32,
}

/// Builds the rectangle frame; `None` for a degenerate rectangle, a non-finite input or an origin
/// in the rectangle's plane (`z0 == 0`).
fn rect_frame(
    origin: [f32; 3],
    corner: [f32; 3],
    edge_x: [f32; 3],
    edge_y: [f32; 3],
) -> Option<RectFrame> {
    let exl = len3(edge_x);
    let eyl = len3(edge_y);
    if !exl.is_finite() || !eyl.is_finite() || exl <= 0.0 || eyl <= 0.0 {
        return None;
    }
    let x = [edge_x[0] / exl, edge_x[1] / exl, edge_x[2] / exl];
    let y = [edge_y[0] / eyl, edge_y[1] / eyl, edge_y[2] / eyl];
    let z = cross3(x, y);
    let d = sub(corner, origin);
    let mut z0 = dot3(d, z);
    if !z0.is_finite() {
        return None;
    }
    if z0 > 0.0 {
        z0 = -z0;
    }
    if z0 == 0.0 {
        return None;
    }
    let x0 = dot3(d, x);
    let y0 = dot3(d, y);
    let x1 = x0 + exl;
    let y1 = y0 + eyl;
    let n0 = normalized(cross3([x0, y0, z0], [x1, y0, z0]));
    let n1 = normalized(cross3([x1, y0, z0], [x1, y1, z0]));
    let n2 = normalized(cross3([x1, y1, z0], [x0, y1, z0]));
    let n3 = normalized(cross3([x0, y1, z0], [x0, y0, z0]));
    let angle = |a: [f32; 3], b: [f32; 3]| (-dot3(a, b)).clamp(-1.0, 1.0).acos();
    let g0 = angle(n0, n1);
    let g1 = angle(n1, n2);
    let g2 = angle(n2, n3);
    let g3 = angle(n3, n0);
    let k = 2.0 * PI - g2 - g3;
    let solid_angle = g0 + g1 - k;
    if !solid_angle.is_finite() || solid_angle <= 0.0 {
        return None;
    }
    Some(RectFrame {
        exl,
        eyl,
        z0,
        x0,
        y0,
        x1,
        y1,
        b0: n0[2],
        b1: n2[2],
        k,
        solid_angle,
    })
}

/// The solid angle of the rectangle `corner + [0, 1] edge_x + [0, 1] edge_y` seen from `origin`,
/// in steradians. 0 for a degenerate rectangle or an origin in its plane.
pub fn rect_solid_angle(
    origin: [f32; 3],
    corner: [f32; 3],
    edge_x: [f32; 3],
    edge_y: [f32; 3],
) -> f32 {
    rect_frame(origin, corner, edge_x, edge_y).map_or(0.0, |frame| frame.solid_angle)
}

/// The area-preserving spherical-rectangle sample of `xi` (Urena et al. EGSR 2013): the unit-square
/// `uv` from `corner` along `edge_x` and `edge_y`, and the rectangle's solid angle. A degenerate
/// rectangle or an origin in its plane returns `([0.5, 0.5], 0)`.
pub fn sample_rect(
    origin: [f32; 3],
    corner: [f32; 3],
    edge_x: [f32; 3],
    edge_y: [f32; 3],
    xi: [f32; 2],
) -> ([f32; 2], f32) {
    let Some(frame) = rect_frame(origin, corner, edge_x, edge_y) else {
        return ([0.5, 0.5], 0.0);
    };
    let u = xi[0].clamp(0.0, 1.0);
    let v = xi[1].clamp(0.0, 1.0);
    let au = u * frame.solid_angle + frame.k;
    let mut sin_au = au.sin();
    if sin_au.abs() < 1e-9 {
        sin_au = if sin_au < 0.0 { -1e-9 } else { 1e-9 };
    }
    let fu = (au.cos() * frame.b0 - frame.b1) / sin_au;
    let cu = (sign_one(fu) / (fu * fu + frame.b0 * frame.b0).sqrt()).clamp(-1.0, 1.0);
    let root = (1.0 - cu * cu).max(0.0).sqrt();
    let xu = if root > 0.0 {
        -(cu * frame.z0) / root
    } else if cu >= 0.0 {
        frame.x1
    } else {
        frame.x0
    };
    let xu = xu.clamp(frame.x0, frame.x1);
    let dd = (xu * xu + frame.z0 * frame.z0).sqrt();
    let h0 = ratio(frame.y0, (dd * dd + frame.y0 * frame.y0).sqrt());
    let h1 = ratio(frame.y1, (dd * dd + frame.y1 * frame.y1).sqrt());
    let hv = h0 + v * (h1 - h0);
    let yv = if hv * hv < 1.0 - 1e-6 {
        hv * dd / (1.0 - hv * hv).sqrt()
    } else {
        frame.y1
    };
    let uv = [
        ((xu - frame.x0) / frame.exl).clamp(0.0, 1.0),
        ((yv - frame.y0) / frame.eyl).clamp(0.0, 1.0),
    ];
    (uv, frame.solid_angle)
}

/// The area PDF of a point on a light: `cos_light / (distance^2 * solid_angle)`. 0 when any
/// input is not positive.
pub fn area_pdf(solid_angle: f32, cos_light: f32, distance: f32) -> f32 {
    if solid_angle <= 0.0 || cos_light <= 0.0 || distance <= 0.0 {
        return 0.0;
    }
    let pdf = cos_light / (distance * distance * solid_angle);
    if pdf.is_finite() { pdf.max(0.0) } else { 0.0 }
}

/// The sRGB transfer function's inverse (encoded to linear), with the exact 0.04045 knee.
pub fn srgb_decode(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// The linear luminance of an RGB colour: the dot product with [`LUMINANCE`].
pub fn luminance(c: [f32; 3]) -> f32 {
    dot3(c, LUMINANCE)
}

/// The shading inputs of one receiver, as the prepass publishes them. `view` is the unit vector
/// to the camera.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Surface {
    /// The receiver's position in metres.
    pub position: [f32; 3],
    /// The receiver's unit normal.
    pub normal: [f32; 3],
    /// The unit vector from the receiver to the camera.
    pub view: [f32; 3],
    /// The material roughness.
    pub roughness: f32,
    /// The material's normal-incidence reflectance per channel.
    pub f0: [f32; 3],
    /// The material metallicness.
    pub metallic: f32,
}

/// The ReSTIR target luminance of a light sample: Lambert diffuse plus a GGX specular lobe (white
/// Lambert, because the prepass stores no albedo and the visibility ratio cancels it), times the
/// light cosine, the receiver cosine, the inverse-square distance and `window`. 0 for a
/// coincident point or a light facing away.
pub fn target_luminance(
    surface: &Surface,
    point: [f32; 3],
    travel: [f32; 3],
    radiance: [f32; 3],
    window: f32,
) -> f32 {
    let d = sub(point, surface.position);
    let r = len3(d);
    if r == 0.0 || !r.is_finite() {
        return 0.0;
    }
    let l = [d[0] / r, d[1] / r, d[2] / r];
    let nl = dot3(surface.normal, l);
    let cos_y = dot3(travel, [-l[0], -l[1], -l[2]]);
    if nl <= 0.0 || cos_y <= 0.0 {
        return 0.0;
    }
    let nv = dot3(surface.normal, surface.view).max(1e-4);
    let h = normalized(add(l, surface.view));
    let nh = dot3(surface.normal, h).max(0.0);
    let vh = dot3(surface.view, h).max(0.0);
    let alpha = surface.roughness.max(ROUGHNESS_MIN).powi(2);
    let alpha2 = alpha * alpha;
    let spec_den = PI * (nh * nh * (alpha2 - 1.0) + 1.0).powi(2);
    let distribution = ratio(alpha2, spec_den);
    let vis_den = nl * (nv * nv * (1.0 - alpha2) + alpha2).sqrt()
        + nv * (nl * nl * (1.0 - alpha2) + alpha2).sqrt();
    let visibility = ratio(0.5, vis_den);
    let diffuse = (1.0 - surface.metallic) / PI;
    let fresnel = (1.0 - vh).powi(5);
    let scaled = std::array::from_fn(|i| {
        let f0 = surface.f0[i];
        let spec = distribution * visibility * (f0 + (1.0 - f0) * fresnel);
        radiance[i] * (diffuse + spec)
    });
    let value = luminance(scaled) * nl * cos_y / (r * r) * window;
    if value.is_finite() { value } else { 0.0 }
}
