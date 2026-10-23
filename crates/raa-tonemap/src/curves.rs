//! CPU ports of the tone mapping curves of the `tonemap` shader. Inputs are
//! scene-referred linear rgb (exposure already applied); outputs are display-referred
//! linear rgb in [0, 1], encoded by [`srgb_oetf`] when the canvas is not *Srgb.

use crate::{Look, ToneMapper, ToneMapping};

type Mat3 = [[f32; 3]; 3];

/// Row-major 3x3 times column vector.
fn mul(m: &Mat3, v: [f32; 3]) -> [f32; 3] {
    let row = |r: &[f32; 3]| r[0] * v[0] + r[1] * v[1] + r[2] * v[2];
    [row(&m[0]), row(&m[1]), row(&m[2])]
}

fn map3(v: [f32; 3], f: impl Fn(f32) -> f32) -> [f32; 3] {
    [f(v[0]), f(v[1]), f(v[2])]
}

/// Rec. 709 luminance of linear rgb.
pub fn luminance(v: [f32; 3]) -> f32 {
    0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]
}

/// AgX inset matrix (Sobotka's AgX base; rows of the column-major reference).
const AGX_INSET: Mat3 = [
    [0.842_479_06, 0.078_433_6, 0.079_223_745],
    [0.042_328_242, 0.878_468_6, 0.079_166_13],
    [0.042_375_655, 0.078_433_6, 0.879_143],
];

/// AgX outset matrix (inverse of the inset).
const AGX_OUTSET: Mat3 = [
    [1.196_879, -0.098_020_88, -0.099_029_74],
    [-0.052_896_85, 1.151_903_1, -0.098_961_18],
    [-0.052_971_635, -0.098_043_45, 1.151_073_7],
];

/// Lower bound of the AgX log2 encoding, in EV relative to 0.18 scaled to 1.
pub const AGX_MIN_EV: f32 = -12.473_93;
/// Upper bound of the AgX log2 encoding.
pub const AGX_MAX_EV: f32 = 4.026_069;

/// The 6th-order polynomial fit of the AgX default contrast sigmoid on [0, 1].
pub fn agx_sigmoid(x: f32) -> f32 {
    let x2 = x * x;
    let x4 = x2 * x2;
    15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x
        - 0.00232
}

/// AgX look preset as (slope, power, saturation).
fn agx_look(look: Look) -> Option<([f32; 3], [f32; 3], f32)> {
    match look {
        Look::None => None,
        Look::Punchy => Some(([1.0; 3], [1.35; 3], 1.4)),
        Look::Golden => Some(([1.0, 0.9, 0.5], [0.8; 3], 0.8)),
    }
}

/// AgX: inset, log2 encoding over [AGX_MIN_EV, AGX_MAX_EV], sigmoid, look (applied in
/// the encoded space), outset, then the 2.2 display decode back to linear.
pub fn agx(rgb: [f32; 3], look: Look) -> [f32; 3] {
    let inset = mul(&AGX_INSET, rgb);
    let encoded = map3(inset, |c| {
        let ev = c.max(1e-10).log2().clamp(AGX_MIN_EV, AGX_MAX_EV);
        agx_sigmoid((ev - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV)).clamp(0.0, 1.0)
    });
    let looked = match agx_look(look) {
        None => encoded,
        Some((slope, power, saturation)) => {
            let luma = luminance(encoded);
            let mut v = encoded;
            for i in 0..3 {
                let graded = (v[i] * slope[i]).max(0.0).powf(power[i]);
                v[i] = luma + saturation * (graded - luma);
            }
            v
        }
    };
    map3(mul(&AGX_OUTSET, looked), |c| c.clamp(0.0, 1.0).powf(2.2))
}

/// sRGB -> ACES AP1 (RRT_SAT) input matrix of Hill's fit.
const ACES_INPUT: Mat3 = [
    [0.597_19, 0.354_58, 0.048_23],
    [0.076_00, 0.908_34, 0.015_66],
    [0.028_40, 0.133_83, 0.837_77],
];

/// ODT_SAT -> sRGB output matrix of Hill's fit.
const ACES_OUTPUT: Mat3 = [
    [1.604_75, -0.531_08, -0.073_67],
    [-0.102_08, 1.108_13, -0.006_05],
    [-0.003_27, -0.072_76, 1.076_02],
];

/// Scale applied before the fit so mid grey lands where AgX puts it (the common
/// `exposure / 0.6` convention of Hill's fit).
pub const ACES_PRE_SCALE: f32 = 1.0 / 0.6;

/// Hill's rational fit of the ACES RRT + sRGB ODT.
pub fn rrt_and_odt_fit(v: f32) -> f32 {
    let a = v * (v + 0.024_578_6) - 0.000_090_537;
    let b = v * (0.983_729 * v + 0.432_951) + 0.238_081;
    a / b
}

/// ACES (Hill's fitted RRT+ODT): pre-scale, input matrix, RRTAndODTFit, output
/// matrix, clamp to [0, 1].
pub fn aces(rgb: [f32; 3]) -> [f32; 3] {
    let v = mul(&ACES_INPUT, map3(rgb, |c| c.max(0.0) * ACES_PRE_SCALE));
    let v = map3(v, rrt_and_odt_fit);
    map3(mul(&ACES_OUTPUT, v), |c| c.clamp(0.0, 1.0))
}

/// Linear: clamp to [0, 1].
pub fn linear(rgb: [f32; 3]) -> [f32; 3] {
    map3(rgb, |c| c.clamp(0.0, 1.0))
}

/// Applies one mapper to exposed linear rgb; `look` is used by AgX only.
pub fn map(mapper: ToneMapper, look: Look, rgb: [f32; 3]) -> [f32; 3] {
    match mapper {
        ToneMapper::Agx => agx(rgb, look),
        ToneMapper::Aces => aces(rgb),
        ToneMapper::Linear => linear(rgb),
    }
}

/// Full display transform without dither: exposure 2^ev, then the mapper (and look).
/// Returns display-referred linear rgb.
pub fn tonemap(settings: &ToneMapping, hdr: [f32; 3]) -> [f32; 3] {
    let scale = settings.exposure_scale();
    map(settings.mapper, settings.look, map3(hdr, |c| c * scale))
}

/// The sRGB OETF (IEC 61966-2-1) on a display-linear value in [0, 1].
pub fn srgb_oetf(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}
