//! CPU twin of the sharpening kernel (S47). The WGSL of the pass computes the same.
//!
//! Adapted from solstice@7e465b5 crates/solstice-render/shaders/post/tonemap.wgsl, MIT licence,
//! Copyright (c) 2026 SamG-Coder. The solstice lines 95-103 are:
//!
//! ```text
//! let low = min(c, min(n, min(s, min(w, e))));
//! let high = max(c, max(n, max(s, max(w, e))));
//! let amount = sqrt(clamp(min(low, vec3f(1.0) - high) / max(high, vec3f(1.0e-4)), vec3f(0.0), vec3f(1.0)));
//! let weight = -amount * params.sharpen * 0.2;
//! let toned_color = clamp((c + (n + s + w + e) * weight) / (1.0 + 4.0 * weight), low, high);
//! ```
//!
//! Solstice sharpens display values; the RAA chain is HDR, so every tap goes through
//! [`compress`] first and the result through [`expand`].

use crate::{COMPRESS_LIMIT, SHARPEN_WEIGHT};

fn max3(c: [f32; 3]) -> f32 {
    c[0].max(c[1]).max(c[2])
}

/// Reversible range compression: `c' = max(c, 0)`, then `c' / (1 + max(c'.r, c'.g, c'.b))`.
pub fn compress(c: [f32; 3]) -> [f32; 3] {
    let c = c.map(|v| v.max(0.0));
    let d = 1.0 + max3(c);
    c.map(|v| v / d)
}

/// Inverse of [`compress`]: `d / (1 - min(max(d.r, d.g, d.b), 0.999))`.
pub fn expand(d: [f32; 3]) -> [f32; 3] {
    let k = 1.0 - max3(d).min(COMPRESS_LIMIT);
    d.map(|v| v / k)
}

/// Sharpens one texel from its centre (rgba) and its four neighbours (rgb), per channel in f32.
/// The result stays inside the range of the five compressed taps; its alpha is the centre's.
pub fn sharpen_texel(
    center: [f32; 4],
    n: [f32; 3],
    s: [f32; 3],
    w: [f32; 3],
    e: [f32; 3],
    sharpness: f32,
) -> [f32; 4] {
    let c = compress([center[0], center[1], center[2]]);
    let (n, s, w, e) = (compress(n), compress(s), compress(w), compress(e));
    let mut out = [0.0; 3];
    for i in 0..3 {
        let low = c[i].min(n[i]).min(s[i]).min(w[i]).min(e[i]);
        let high = c[i].max(n[i]).max(s[i]).max(w[i]).max(e[i]);
        let amount = (low.min(1.0 - high) / high.max(1.0e-4))
            .clamp(0.0, 1.0)
            .sqrt();
        let weight = -amount * sharpness * SHARPEN_WEIGHT;
        let sharp = (c[i] + (n[i] + s[i] + w[i] + e[i]) * weight) / (1.0 + 4.0 * weight);
        out[i] = sharp.clamp(low, high);
    }
    let [r, g, b] = expand(out);
    [r, g, b, center[3]]
}
