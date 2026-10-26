//! WGSL source of the `grading` pass, its entry points and the size of its uniform.
//!
//! Standalone (no `crate::` paths) so tests/variants.rs can compile it with `#[path]`.

/// Render module (no PRELUDE), group 0 only: 0 `GradingParams` uniform, 1 hdr.src()
/// (filterable float), 2 linear clamp sampler. The fragment entry is drawn with the
/// fullscreen triangle of [`VS_MAIN`] (3 vertices) into hdr.dst().
pub(crate) const GRADING_WGSL: &str = include_str!("wgsl/grading.wgsl");

/// Fullscreen triangle vertex entry, varying `uv` (target coordinates, (0, 0) top-left).
pub(crate) const VS_MAIN: &str = "vs_main";
/// Grading: white balance, exposure, log contrast, CDL, split toning, saturation and
/// vibrance, vignette; HDR out, negative or NaN clamped to 0.
pub(crate) const FS_GRADING: &str = "fs_grading";

/// Byte size of `GradingParams`: wb columns @0/@16/@32, exposure_contrast @48 (2^ev,
/// contrast, saturation, vibrance), lift @64, gamma @80, gain @96, shadows_tint @112,
/// highlights_tint @128, misc @144 (split_balance, vignette, aspect, 0).
pub(crate) const PARAMS_SIZE: u64 = 160;
