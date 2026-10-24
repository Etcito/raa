//! The 16 B OIT list node, as written by the Transparent pass shader.

/// Size of one node in bytes.
pub const NODE_SIZE: usize = 16;
/// End-of-list marker for `next` and for cleared heads.
pub const END: u32 = 0xffff_ffff;

/// One list node: straight (not premultiplied) linear colour and alpha.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct OitNode {
    /// pack2x16float(r, g)
    pub color_rg: u32,
    /// pack2x16float(b, alpha)
    pub color_ba: u32,
    /// Fragment depth bits (f32).
    pub depth_bits: u32,
    pub next: u32,
}

impl OitNode {
    pub fn pack(color: [f32; 3], alpha: f32, depth: f32, next: u32) -> Self {
        OitNode {
            color_rg: pack2x16float(color[0], color[1]),
            color_ba: pack2x16float(color[2], alpha),
            depth_bits: depth.to_bits(),
            next,
        }
    }

    /// Returns (straight colour, alpha).
    pub fn color_alpha(&self) -> ([f32; 3], f32) {
        let (r, g) = unpack2x16float(self.color_rg);
        let (b, a) = unpack2x16float(self.color_ba);
        ([r, g, b], a)
    }

    pub fn depth(&self) -> f32 {
        f32::from_bits(self.depth_bits)
    }

    pub fn to_le_bytes(&self) -> [u8; NODE_SIZE] {
        let mut out = [0u8; NODE_SIZE];
        out[0..4].copy_from_slice(&self.color_rg.to_le_bytes());
        out[4..8].copy_from_slice(&self.color_ba.to_le_bytes());
        out[8..12].copy_from_slice(&self.depth_bits.to_le_bytes());
        out[12..16].copy_from_slice(&self.next.to_le_bytes());
        out
    }

    pub fn from_le_bytes(b: [u8; NODE_SIZE]) -> Self {
        let w = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        OitNode {
            color_rg: w(0),
            color_ba: w(4),
            depth_bits: w(8),
            next: w(12),
        }
    }
}

/// Nodes in the pool: min(w x h x factor, limit_bytes / 16), at least 1.
pub fn pool_nodes(width: u32, height: u32, factor: u32, limit_bytes: u64) -> u32 {
    let wanted = (u64::from(width) * u64::from(height)).saturating_mul(u64::from(factor));
    let cap = limit_bytes / NODE_SIZE as u64;
    let n = wanted.min(cap).max(1);
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// f32 to IEEE half bits, round to nearest even; overflow gives infinity.
pub fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let mut h = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        if rem > half || (rem == half && (h & 1) == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1; // a carry into the exponent is the correct rounding
    }
    sign | h as u16
}

/// IEEE half bits to f32.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x03ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x03ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// WGSL pack2x16float: x in the low 16 bits.
pub fn pack2x16float(x: f32, y: f32) -> u32 {
    u32::from(f32_to_f16(x)) | (u32::from(f32_to_f16(y)) << 16)
}

pub fn unpack2x16float(p: u32) -> (f32, f32) {
    (
        f16_to_f32((p & 0xffff) as u16),
        f16_to_f32((p >> 16) as u16),
    )
}
