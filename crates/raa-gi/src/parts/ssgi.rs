//! The GI part `ssgi` (milestone M8): screen-space indirect diffuse from a visibility bitmask. For
//! each pixel, every slice of the view plane (`SsgiQuality::slices`) marches the depth buffer's
//! near field, `cpu::RADIUS_METRES` (2 m), `SsgiQuality::steps` times per side. Each sample sets
//! the 32 sectors it occludes in the slice's u32 mask (`cpu::sample_bits`), after Therrien,
//! Levesque and Gilet, "Screen Space Indirect Lighting with Visibility Bitmask" (2023), in its
//! cosine-weighted sector form. The slice and step structure follows raa's GTAO
//! (`crates/raa-gtao/src/wgsl/gtao.wgsl` lines 126-158). The method is original to this
//! repository: solstice has no SSGI to port.
//!
//! No double counting: `cpu::accumulate` lights only the sectors a sample opens for the first time,
//! and `cpu::compose` sets the composite `E_ss + V x E_probe` with alpha `min(V, probe a)`. Screen
//! space replaces the probe and volume irradiance in the sectors it found occluded, so that
//! irradiance is never added a second time.
//!
//! This root holds the pass name, its order and the byte budget. The CPU twins live in `cpu`.

use super::PartsCtx;
use raa_core::RaaError;

#[path = "ssgi/cpu.rs"]
pub mod cpu;
#[path = "ssgi/passes.rs"]
pub mod passes;
#[path = "ssgi/resources.rs"]
mod resources;

/// The name of the compute pass `gi.ssgi` (S12 order 44).
pub const SSGI_PASS: &str = "gi.ssgi";

/// The order of `gi.ssgi`: after `gi.area-shadow` (43) and before `shadow.contact` (45).
pub const SSGI_ORDER: u32 = 44;

/// Bytes the part allocates for a viewport of `(W, H)` pixels, each axis clamped to at least 1:
/// `32 x W x H + 48`. That is 8 B per pixel for each of the raw texture, the two history textures
/// and the output, plus the 32-B params and two 1x1 Rgba16Float dummies of 8 B each. The sum
/// saturates at `u64::MAX` for a viewport too large for a u64.
pub fn ssgi_bytes(viewport: (u32, u32)) -> u64 {
    let width = u64::from(viewport.0.max(1));
    let height = u64::from(viewport.1.max(1));
    32u64
        .saturating_mul(width)
        .saturating_mul(height)
        .saturating_add(48)
}

/// Handoff: `install` is the body that card
/// 'raa-gi ssgi: SsgiPart, the gi.ssgi pass (order 44) and the binding-14 pick' replaces. Today
/// `install` adds nothing to `ctx` and returns `Ok(())`; the generated `install_parts` of
/// `parts.rs` calls it once per GI install, in file-name order.
pub(crate) fn install(ctx: &mut PartsCtx<'_>) -> Result<(), RaaError> {
    let _ = ctx;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssgi_constants_match_the_s12_row() {
        assert_eq!(SSGI_PASS, "gi.ssgi");
        assert_eq!(SSGI_ORDER, 44);
    }

    #[test]
    fn ssgi_bytes_matches_the_hand_values() {
        assert_eq!(ssgi_bytes((1280, 720)), 29_491_248);
        assert_eq!(ssgi_bytes((320, 180)), 1_843_248);
        assert_eq!(ssgi_bytes((0, 0)), 80);
    }
}
