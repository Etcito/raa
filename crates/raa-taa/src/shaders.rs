//! WGSL shared by every temporal pass of the TAA milestone (contract `raa-temporal-wgsl`):
//! the S10 `FrameGpu` twin `RaaTemporalFrame` and the reprojection helpers.

/// The shared reprojection source: `struct RaaTemporalFrame` (the 768-byte S10 `FrameGpu`
/// twin) and the functions `raa_temporal_on_screen`, `raa_temporal_surface_uv`,
/// `raa_temporal_sky_uv` and `raa_temporal_history_usable`. The text declares no bindings and
/// no entry points, so a pass concatenates it before its own source.
///
/// Handoff: card 'raa-taa: TAA resolve compute shader ported from solstice' takes over
/// `REPROJECT_WGSL`, which today holds the `RaaTemporalFrame` twin and those four functions;
/// the resolve concatenates it before its own source and calls `raa_temporal_surface_uv`,
/// `raa_temporal_sky_uv` and `raa_temporal_history_usable`.
pub const REPROJECT_WGSL: &str = include_str!("wgsl/reproject.wgsl");

/// Bytes of one `RaaTemporalFrame`: the S10 `FrameGpu` size, 768.
pub const TEMPORAL_FRAME_SIZE: u64 = 768;

/// The TAA resolve compute source (contract `taa-bindings`): group 0 entries 0-8 and the
/// `cs_taa` entry.
pub const TAA_WGSL: &str = include_str!("wgsl/taa.wgsl");

/// The resolve compute entry point.
pub const TAA_ENTRY: &str = "cs_taa";

/// The entry's workgroup width: one invocation per output pixel at `8 x 8`.
pub const TAA_WORKGROUP: u32 = 8;

/// Bytes of one `RaaTaaParams`: `history_usable` then three reserved words.
pub const TAA_PARAMS_SIZE: u64 = 16;

/// `taa_module_source` returns the TAA resolve compute module of contract `taa-bindings`:
/// `REPROJECT_WGSL`, a newline, then `TAA_WGSL`, whose group 0 entries 0-8 feed `TAA_ENTRY`
/// (`cs_taa`) at a `TAA_WORKGROUP`-wide workgroup (8).
///
/// Handoff: card 'raa-taa: TAA bind group layout, pipeline and per-size bind groups' takes
/// over `taa_module_source`, `TAA_ENTRY` and `TAA_WORKGROUP`.
pub fn taa_module_source() -> String {
    format!("{REPROJECT_WGSL}\n{TAA_WGSL}")
}
