//! WGSL source of the `bloom` pass, its entry points and the size of its uniform.
//!
//! Standalone (no `crate::` paths) so tests/variants.rs can compile it with `#[path]`.

/// Render module (no PRELUDE), group 0 only: 0 `BloomParams` uniform, 1 source level
/// (filterable float), 2 linear clamp sampler, 3 bloom level 0 (composite only). Every
/// fragment entry is drawn with the fullscreen triangle of [`VS_MAIN`] (3 vertices).
pub(crate) const BLOOM_WGSL: &str = include_str!("wgsl/bloom.wgsl");

/// Fullscreen triangle vertex entry, varying `uv` (target coordinates, (0, 0) top-left).
pub(crate) const VS_MAIN: &str = "vs_main";
/// First downsample, from hdr.src(): 13-tap filter with Karis luma-weighted box averages.
pub(crate) const FS_DOWN_FIRST: &str = "fs_down_first";
/// Later downsamples: plain 13-tap filter.
pub(crate) const FS_DOWN: &str = "fs_down";
/// Upsample: 3x3 tent scaled by `radius`, blended (One, One) into the next larger level.
pub(crate) const FS_UP: &str = "fs_up";
/// Composite: `mix(hdr, bloom, strength)` into hdr.dst().
pub(crate) const FS_COMPOSITE: &str = "fs_composite";

/// Byte size of `BloomParams`: texel_size @0 (1 / source size), radius @8, strength @12.
pub(crate) const PARAMS_SIZE: u64 = 16;
