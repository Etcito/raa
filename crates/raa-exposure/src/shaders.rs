//! WGSL of the exposure compute pass (S46) and the CPU side of its layouts.

use crate::{AutoExposureDesc, PARAMS_SIZE, RESULT_SIZE};

/// The histogram and resolve compute module.
pub const EXPOSURE_COMPUTE_WGSL: &str = include_str!("wgsl/exposure_compute.wgsl");
/// Compute entry building the luminance histogram.
pub const BUILD_HISTOGRAM: &str = "build_histogram";
/// Compute entry resolving the band average and the adapted exposure.
pub const RESOLVE: &str = "resolve";

/// Writes the S46 `ExposureParams` uniform, little-endian, into `out`.
///
/// Handoff: raa-exposure 6/6: exposure.auto pass, install and handle. This is the entry point
/// `pack_params(out: &mut [u8; PARAMS_SIZE], desc: &AutoExposureDesc, dt_s: f32, first: bool,
/// frame: u64)`, exported by `crates/raa-exposure/src/shaders.rs` and re-exported by
/// `crates/raa-exposure/src/lib.rs`. Today it fills the 48 B record (dt_s @0, first @4,
/// 2^compensation_ev @8, low/high percent @12/@16, speed up/down @20/@24, 2^min_ev @28,
/// 2^max_ev @32, low 32 bits of frame @36, zero @40-48) and keeps no state; the pass packs into
/// its own scratch, then calls `queue.write_buffer`. The desc it reads is checked by
/// `AutoExposureDesc::validate` (`crates/raa-exposure/src/desc.rs`, whose Handoff covers it).
pub fn pack_params(
    out: &mut [u8; PARAMS_SIZE],
    desc: &AutoExposureDesc,
    dt_s: f32,
    first: bool,
    frame: u64,
) {
    let words: [[u8; 4]; 12] = [
        dt_s.to_le_bytes(),
        u32::from(first).to_le_bytes(),
        desc.compensation_ev.exp2().to_le_bytes(),
        desc.low_percent.to_le_bytes(),
        desc.high_percent.to_le_bytes(),
        desc.speed_up.to_le_bytes(),
        desc.speed_down.to_le_bytes(),
        desc.min_ev.exp2().to_le_bytes(),
        desc.max_ev.exp2().to_le_bytes(),
        (frame as u32).to_le_bytes(),
        [0; 4],
        [0; 4],
    ];
    for (chunk, word) in out.chunks_exact_mut(4).zip(words) {
        chunk.copy_from_slice(&word);
    }
}

/// The decoded `ExposureState` record.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ExposureResult {
    /// Adapted average luminance.
    pub average_luminance: f32,
    /// Linear exposure multiplier.
    pub exposure: f32,
    /// True once resolve ran.
    pub valid: bool,
    /// Low 32 bits of the frame that wrote it.
    pub frame: u32,
}

/// Decodes the 16 B `ExposureState` record.
pub fn read_result(bytes: &[u8; RESULT_SIZE]) -> ExposureResult {
    let mut w = [[0u8; 4]; 4];
    for (word, chunk) in w.iter_mut().zip(bytes.chunks_exact(4)) {
        word.copy_from_slice(chunk);
    }
    ExposureResult {
        average_luminance: f32::from_le_bytes(w[0]),
        exposure: f32::from_le_bytes(w[1]),
        valid: u32::from_le_bytes(w[2]) != 0,
        frame: u32::from_le_bytes(w[3]),
    }
}
