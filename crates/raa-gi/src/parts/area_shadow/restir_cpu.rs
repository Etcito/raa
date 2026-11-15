//! CPU twins of the M7 ReSTIR kernels (S12 order 43, contract `area-restir-cpu`), share 1: the
//! constants, the binary16 pair codec, the octahedral normal codec and the packed reservoir
//! record. These are RAA's own maths from Bitterli et al. 2020 (ReSTIR), Wyman and Panteleev HPG
//! 2021, Lin et al. 2022 (GRIS) and Urena et al. EGSR 2013; Bevy Solari is not supplied and
//! nothing is copied from it or from NVIDIA RTXDI. The WGSL twins (`restir_common.wgsl`) use the
//! same names with an `raa_restir_` prefix. These twins are tested on the CPU against hand-derived
//! fixtures and are never run against the GPU. Every function is pure and nothing panics; u32
//! arithmetic wraps as in WGSL.

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
