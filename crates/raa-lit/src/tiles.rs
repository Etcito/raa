//! Compute source registration; GPU allocation/recording is the order-35 pass share.

/// Tile coverage and conservative candidate source, appended after PRELUDE_WGSL.
pub const LIGHT_TILES_WGSL: &str = include_str!("wgsl/light_tiles.wgsl");

/// Compose the registered compute source with the authoritative frame/light ABI/constants.
///
/// Handoff: `pub fn light_tiles_source() -> String` in `crates/raa-lit/src/tiles.rs`
/// is re-exported by `lib.rs`; the recording share compiles entry point `light_tiles`.
/// It binds FrameGpu at group 0/0 and single-sample depth, the handle-ordered
/// `LocalLights::write_gpu` uniform records and tile-word storage texture at group 2/0..2.
/// Selection attaches after the final barrier to candidate_eligible/scores/indices.
/// Handoff: coverage follows `pub(crate) fn select_tile(candidates: &[(u64, f32)], out: &mut [u64; 127]) -> (usize, u32)`
/// in `crates/raa-lit/src/tile_selection.rs`, attached by the test module in `lib.rs`.
/// That landed CPU oracle specifies qualified `(LightHandle, score)` pairs from
/// `LocalLights`; the compute workgroup uses their dense handle-ordered GPU indices.
/// No selection, overflow readback or pass installation is performed by this source share.
pub fn light_tiles_source() -> String {
    format!("{}\n{LIGHT_TILES_WGSL}", raa_shader::PRELUDE_WGSL)
}
