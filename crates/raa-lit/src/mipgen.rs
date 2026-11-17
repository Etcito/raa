//! Box downsample of the material atlas mip chain (S48, P5-ATLAS-MIPS).
//!
//! Level `k` is made from level `k - 1`, never from level 0. A side `s` halves to `(s + 1) / 2`,
//! written out inline here (the level helper of `textures` is not called). Output texel `(i, j)`
//! averages the source texels `(2i, 2j)`, `(2i+1, 2j)`, `(2i, 2j+1)` and `(2i+1, 2j+1)`, with x
//! clamped to `width - 1` and y to `height - 1`, so an odd side repeats its last column or row.
//! The linear rule per channel is `(a + b + c + d + 2) / 4` in u32. With sRGB, r, g and b are
//! decoded to linear light, averaged in f32, re-encoded and quantized to 8 bits; alpha always
//! uses the linear rule.

use crate::textures::quantize;

/// The rgba bytes of texel `(x, y)` of a `width`-texel-wide image; `[0; 4]` out of range.
fn texel(src: &[u8], width: usize, x: usize, y: usize) -> [u8; 4] {
    let offset = (y * width + x) * 4;
    src.get(offset..offset + 4)
        .and_then(|t| <[u8; 4]>::try_from(t).ok())
        .unwrap_or([0; 4])
}

/// Decodes one 8-bit sRGB channel to linear light (`x = byte / 255`).
fn srgb_to_linear(byte: u8) -> f32 {
    let x = f32::from(byte) / 255.0;
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

/// Re-encodes a linear light value in [0, 1] as sRGB, still in [0, 1] (quantized by the caller).
fn linear_to_srgb(l: f32) -> f32 {
    if l <= 0.0031308 {
        12.92 * l
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

/// One output texel from its four taps: the linear rule for every channel, or the sRGB rule
/// for r, g and b when `srgb` is set (alpha stays linear).
fn average(taps: [[u8; 4]; 4], srgb: bool) -> [u8; 4] {
    let mut out = [0; 4];
    for (channel, value) in out.iter_mut().enumerate() {
        *value = if srgb && channel < 3 {
            let sum: f32 = taps.iter().map(|t| srgb_to_linear(t[channel])).sum();
            quantize(linear_to_srgb(sum / 4.0))
        } else {
            let sum: u32 = taps.iter().map(|t| u32::from(t[channel])).sum();
            ((sum + 2) / 4) as u8
        };
    }
    out
}

/// Handoff: `downsample_rgba8` is called once per mip level by the card "raa-lit: 5-level atlas
/// upload, mipmap sampler and LitRenderer::add_texture_with", which feeds each output back in as
/// the next call's `src`. Today it returns the next level: `(width + 1) / 2` by `(height + 1) / 2`
/// texels of rgba8 rows, top first, with `srgb` selecting the sRGB rule for r, g and b; a zero
/// side or a `src` shorter than `width * height * 4` returns an empty Vec with that size.
pub fn downsample_rgba8(src: &[u8], width: u32, height: u32, srgb: bool) -> (Vec<u8>, u32, u32) {
    // `div_ceil(2)` is `(s + 1) / 2` without overflow at u32::MAX.
    let (out_width, out_height) = (width.div_ceil(2), height.div_ceil(2));
    let texel_bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4));
    let Some(len) = texel_bytes else {
        return (Vec::new(), out_width, out_height);
    };
    if width == 0 || height == 0 || src.len() < len {
        return (Vec::new(), out_width, out_height);
    }
    let (w, h) = (width as usize, height as usize);
    let mut out = Vec::with_capacity(out_width as usize * out_height as usize * 4);
    for j in 0..out_height as usize {
        let y0 = (2 * j).min(h - 1);
        let y1 = (2 * j + 1).min(h - 1);
        for i in 0..out_width as usize {
            let x0 = (2 * i).min(w - 1);
            let x1 = (2 * i + 1).min(w - 1);
            let taps = [
                texel(src, w, x0, y0),
                texel(src, w, x1, y0),
                texel(src, w, x0, y1),
                texel(src, w, x1, y1),
            ];
            out.extend_from_slice(&average(taps, srgb));
        }
    }
    (out, out_width, out_height)
}
