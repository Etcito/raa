//! WGSL source of the SSR input passes (Hi-Z pyramid and HDR mip chain), their entry points
//! and the size of the Hi-Z uniform.
//!
//! Standalone (no `crate::` paths) so tests/variants.rs can compile it with `#[path]`.
#![allow(dead_code)]

/// Module (no PRELUDE), group 0 only: 0 `HizParams` uniform, 1 `depth` (texture_depth_2d,
/// cs_hiz_copy), 2 Hi-Z level n (cs_hiz_reduce), 3 the r32float write-only storage level
/// written (both compute entries), 4 previous HDR mip (fs_downsample), 5 linear clamp
/// sampler (fs_downsample). Each pipeline's layout holds only the bindings its entry uses.
pub(crate) const SSR_HIZ_WGSL: &str = include_str!("wgsl/ssr_hiz.wgsl");

/// Copies `depth` into Hi-Z mip 0; dispatch `ceil(w / 8) x ceil(h / 8)`.
pub(crate) const CS_HIZ_COPY: &str = "cs_hiz_copy";
/// Writes Hi-Z level n + 1 from level n by the rule of `hiz::reduce_min`; dispatch over the
/// destination size in 8x8 workgroups.
pub(crate) const CS_HIZ_REDUCE: &str = "cs_hiz_reduce";
/// Compute workgroup size along x and y.
pub(crate) const HIZ_WORKGROUP: u32 = 8;
/// Fullscreen triangle vertex entry, varying `uv` (target coordinates, (0, 0) top-left).
pub(crate) const VS_FULLSCREEN: &str = "vs_fullscreen";
/// 13-tap filtered downsample of the previous HDR mip into the next one.
pub(crate) const FS_DOWNSAMPLE: &str = "fs_downsample";

/// Byte size of `HizParams`: `size: vec4<u32>` = (src w, src h, dst w, dst h).
pub(crate) const HIZ_PARAMS_SIZE: u64 = 16;
