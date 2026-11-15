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
