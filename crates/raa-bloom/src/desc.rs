use raa_core::RaaError;

/// Maximum number of bloom mip levels.
pub const MAX_MIPS: usize = 8;
/// Mip levels are kept while both sides are at least this many pixels.
pub const MIN_MIP_SIZE: u32 = 8;

/// Bloom parameters: composite `mix(hdr, bloom, strength)`, tent scaled by `radius`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BloomDesc {
    /// Mix factor in `[0, 1]`.
    pub strength: f32,
    /// Upsample tent radius in `[0, 1]`.
    pub radius: f32,
}

impl Default for BloomDesc {
    fn default() -> Self {
        BloomDesc {
            strength: 0.04,
            radius: 0.85,
        }
    }
}

impl BloomDesc {
    /// Refuses with `bloom.invalid`, naming the offending field; non-finite values are refused.
    pub fn validate(&self) -> Result<(), RaaError> {
        unit("strength", self.strength)?;
        unit("radius", self.radius)
    }
}

fn unit(field: &str, v: f32) -> Result<(), RaaError> {
    let message = if !v.is_finite() {
        format!("`{field}` must be finite, got {v}")
    } else if !(0.0..=1.0).contains(&v) {
        format!("`{field}` must be in [0, 1], got {v}")
    } else {
        return Ok(());
    };
    Err(RaaError::FrameGraph {
        code: "bloom.invalid",
        message,
    })
}

/// Mip sizes for an HDR target; returns the array and the number of valid levels.
pub fn mip_sizes(width: u32, height: u32) -> ([(u32, u32); MAX_MIPS], usize) {
    let mut out = [(0, 0); MAX_MIPS];
    let (mut w, mut h) = (width, height);
    let mut n = 0;
    while n < MAX_MIPS {
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        if w < MIN_MIP_SIZE || h < MIN_MIP_SIZE {
            break;
        }
        out[n] = (w, h);
        n += 1;
    }
    (out, n)
}
