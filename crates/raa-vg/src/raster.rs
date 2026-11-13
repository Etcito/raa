//! The S32 raster modules of RAA's virtualized geometry: the hardware rasteriser's WGSL
//! constants and the module builder. The software rasteriser, the visibility buffers and the
//! pipelines land with their own cards. Adapted from solstice (MIT), Copyright (c) 2026
//! SamG-Coder: `crates/solstice-render/shaders/vg/raster_common.wgsl`, `raster_hw.wgsl`,
//! `raster_hw_fragment.wgsl` and `vis_write.wgsl`, revision 7e465b5. The reversed-Z visibility
//! write becomes the standard-Z `raa_depth_key` integer compare.

use crate::culling::PASS_VIEW_WGSL;

/// S32 `raster_common.wgsl`: the group-2 scene bindings (0 `PassView`, 1 instances, 2 clusters,
/// 3 vertices, 4 triangles, 5 visible, 6 `CullStateRead`) and the `encode_id` and
/// `local_vertex` helpers shared by both rasterisers. The builder concatenates it after
/// `raa_shader::DEPTH_WGSL` and [`PASS_VIEW_WGSL`].
pub const RASTER_COMMON_WGSL: &str = include_str!("wgsl/raster_common.wgsl");

/// S32 `vis_write.wgsl`: group-2 bindings 7 (`vis_depth`) and 8 (`vis_id`), both
/// `array<atomic<u32>>`, and `write_visibility`, which keeps the larger `raa_depth_key` of
/// the fragment depth with the integer compare-exchange loop of the source, then stores the id.
pub const VIS_WRITE_WGSL: &str = include_str!("wgsl/vis_write.wgsl");

const RASTER_HW_WGSL: &str = include_str!("wgsl/raster_hw.wgsl");

const RASTER_HW_FRAGMENT_WGSL: &str = include_str!("wgsl/raster_hw_fragment.wgsl");

/// The hardware raster module: `raa_shader::DEPTH_WGSL` + [`PASS_VIEW_WGSL`] +
/// [`RASTER_COMMON_WGSL`] + [`VIS_WRITE_WGSL`] + `raster_hw.wgsl` + `raster_hw_fragment.wgsl`,
/// with entry points `vs_visibility`, `vs_depth` (vertex) and `fs_visibility` (fragment) at
/// group 2 bindings 0-8.
pub fn raster_hw_module_wgsl() -> String {
    [
        raa_shader::DEPTH_WGSL,
        PASS_VIEW_WGSL,
        RASTER_COMMON_WGSL,
        VIS_WRITE_WGSL,
        RASTER_HW_WGSL,
        RASTER_HW_FRAGMENT_WGSL,
    ]
    .concat()
}
