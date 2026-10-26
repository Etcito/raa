//! WGSL source of the `sunshafts` pass, its entry points and the size of its uniform.
//!
//! Standalone (no `crate::` paths) so tests/variants.rs can compile it with `#[path]`.

/// Render module (no PRELUDE). Group 0 (private): 0 `ShaftsParams` uniform, 1 scene depth
/// (`texture_depth_2d`), 2 half-res march result, 3 hdr.src(), 4 linear clamp sampler;
/// `fs_march` reads only 0-1, so its bind group may put dummies at 2-4. Group 1: the
/// lighting bindings 0 (shadow map), 1 (comparison sampler), 2 (`ShadowGpu`) and 11
/// (`EnvGpu`) of `LightingResources::layout()`. Every fragment entry is drawn with the
/// fullscreen triangle of [`VS_MAIN`] (3 vertices).
pub(crate) const SHAFTS_WGSL: &str = include_str!("wgsl/shafts.wgsl");

/// Fullscreen triangle vertex entry, varying `uv` (target coordinates, (0, 0) top-left).
pub(crate) const VS_MAIN: &str = "vs_main";
/// Half-res 32-step march into an Rgba16Float target: rgb in-scatter, a view depth; 0 when
/// shadows are disabled.
pub(crate) const FS_MARCH: &str = "fs_march";
/// Full res: bilateral 4-tap upsample of the march added to hdr.src(), into hdr.dst().
pub(crate) const FS_COMPOSITE: &str = "fs_composite";

/// Byte size of `ShaftsParams`: inv_view_proj @0, view @64, camera_pos @128, sun_dir @144
/// (travel direction, w intensity), sun_color @160, viewport @176 (half w, h, full w, h),
/// params @192 (density, max_distance_m, frame index, 0).
pub(crate) const PARAMS_SIZE: u64 = 208;
