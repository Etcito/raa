//! Trace kernel of the `gi.area-shadow` pass (S44, contract `area-shadow-trace-wgsl`): the
//! composed WGSL module `PRELUDE_WGSL` + `CLOUD_SHADOW_WGSL` + `GI_VOLUME_WGSL` + `DEPTH_WGSL` +
//! probe_common.wgsl + screen_trace.wgsl + wgsl/area_shadow/trace.wgsl, its entry name and the
//! eight group-2 bindings the kernel uses. `AreaShadowParamsGpu` and the group-2 storage outputs
//! live in trace.wgsl; the slot table is the forward pass's `RaaAreaStochasticGpu`.
//!
//! The temporal and spatial kernels, the pipelines, the resources and the part are later shares
//! of this module, so nothing in this crate builds a pipeline from these builders yet.

use raa_pipeline::AREA_STOCHASTIC_GPU_SIZE;
use raa_shader::{CLOUD_SHADOW_WGSL, DEPTH_WGSL, GI_VOLUME_WGSL, PRELUDE_WGSL};
use wgpu::{
    BindGroupLayoutEntry, BindingType, StorageTextureAccess, TextureFormat, TextureViewDimension,
};

use super::super::screen::passes::{
    depth_entry, entry, float_entry, storage_texture_entry, uniform_entry,
};

const PROBE_COMMON_WGSL: &str = include_str!("../../wgsl/screen/probe_common.wgsl");
const SCREEN_TRACE_WGSL: &str = include_str!("../../wgsl/screen/screen_trace.wgsl");
const AREA_TRACE_WGSL: &str = include_str!("../../wgsl/area_shadow/trace.wgsl");

/// `GiFrameGpu` bytes (S38): five matrices, then the seven 16-B lanes.
const FRAME_SIZE: u64 = 432;

/// Entry of the area-shadow trace pipeline.
pub const AREA_TRACE_ENTRY: &str = "raa_gi_area_trace";

/// Side of the trace compute workgroup (8x8 threads).
pub const WORKGROUP: u32 = 8;

/// Size of `AreaShadowParamsGpu`: `frame_index` and three pad lanes.
pub const AREA_PARAMS_SIZE: u64 = 16;

/// A write-only `D2Array` storage texture; the screen helper only builds `D2` views.
fn storage_texture_array_entry(binding: u32, format: TextureFormat) -> BindGroupLayoutEntry {
    entry(
        binding,
        BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format,
            view_dimension: TextureViewDimension::D2Array,
        },
    )
}

/// `trace_module_wgsl` builds the trace module of contract 'area-shadow-trace-wgsl':
/// `PRELUDE_WGSL` + `CLOUD_SHADOW_WGSL` + `GI_VOLUME_WGSL` + `DEPTH_WGSL` + probe_common.wgsl +
/// screen_trace.wgsl + wgsl/area_shadow/trace.wgsl. The entry `AREA_TRACE_ENTRY` of
/// wgsl/area_shadow/trace.wgsl traces one stratified shadow ray per pixel and mid-tier slot.
///
/// Handoff: card 'raa-gi: AreaShadowPipelines and the raa-gi.area-shadow compute pass record'
/// takes over `trace_module_wgsl` together with `trace_group2_entries` and `AREA_TRACE_ENTRY`
/// and builds the trace pipeline from them; this file builds the composed module text and the
/// eight group-2 entries and builds no pipeline.
pub fn trace_module_wgsl() -> String {
    format!(
        "{PRELUDE_WGSL}{CLOUD_SHADOW_WGSL}{GI_VOLUME_WGSL}{DEPTH_WGSL}\
         {PROBE_COMMON_WGSL}{SCREEN_TRACE_WGSL}{AREA_TRACE_WGSL}"
    )
}

/// Trace group 2, ascending: 0 `GiFrameGpu` (432 B), 1 the S12 depth, 2 the S12 normal, 4 the
/// depth pyramid (all mips), 5 the Rgba16Float raw history array, 6 the R32Float depth output,
/// 8 the `RaaAreaStochasticGpu` slot table (80 B) and 9 `AreaShadowParamsGpu` (16 B).
pub fn trace_group2_entries() -> [BindGroupLayoutEntry; 8] {
    [
        uniform_entry(0, FRAME_SIZE),
        depth_entry(1),
        float_entry(2, false),
        float_entry(4, false),
        storage_texture_array_entry(5, TextureFormat::Rgba16Float),
        storage_texture_entry(6, TextureFormat::R32Float),
        uniform_entry(8, AREA_STOCHASTIC_GPU_SIZE as u64),
        uniform_entry(9, AREA_PARAMS_SIZE),
    ]
}
