//! Kernels of the `gi.area-shadow` pass (S44, contracts `area-shadow-trace-wgsl` and
//! `area-shadow-denoise-wgsl`): the trace module (`PRELUDE_WGSL` + `CLOUD_SHADOW_WGSL` +
//! `GI_VOLUME_WGSL` + `DEPTH_WGSL` + probe_common.wgsl + screen_trace.wgsl +
//! wgsl/area_shadow/trace.wgsl) with its entry and eight group-2 bindings, the temporal module
//! (`REPROJECT_WGSL` + temporal.wgsl) with its nine group-0 bindings, and the spatial module
//! (spatial.wgsl) with the six group-0 bindings its two separable entries share.
//! `AreaShadowParamsGpu` and the group-2 storage outputs live in trace.wgsl;
//! `AreaDispatchGpu` and the denoiser bindings live in temporal.wgsl and spatial.wgsl.
//!
//! The pipelines, the resources and the part are later shares of this module, so nothing in
//! this crate builds a pipeline from these builders yet.

use raa_pipeline::AREA_STOCHASTIC_GPU_SIZE;
use raa_shader::{CLOUD_SHADOW_WGSL, DEPTH_WGSL, GI_VOLUME_WGSL, PRELUDE_WGSL};
use raa_taa::{REPROJECT_WGSL, TEMPORAL_FRAME_SIZE};
use wgpu::{
    BindGroupLayoutEntry, BindingType, SamplerBindingType, StorageTextureAccess, TextureFormat,
    TextureViewDimension,
};

use super::super::screen::passes::{
    depth_entry, entry, float_entry, storage_texture_entry, uniform_entry,
};

const PROBE_COMMON_WGSL: &str = include_str!("../../wgsl/screen/probe_common.wgsl");
const SCREEN_TRACE_WGSL: &str = include_str!("../../wgsl/screen/screen_trace.wgsl");
const AREA_TRACE_WGSL: &str = include_str!("../../wgsl/area_shadow/trace.wgsl");
const AREA_TEMPORAL_WGSL: &str = include_str!("../../wgsl/area_shadow/temporal.wgsl");
const AREA_SPATIAL_WGSL: &str = include_str!("../../wgsl/area_shadow/spatial.wgsl");

/// `GiFrameGpu` bytes (S38): five matrices, then the seven 16-B lanes.
const FRAME_SIZE: u64 = 432;

/// Entry of the area-shadow trace pipeline.
pub const AREA_TRACE_ENTRY: &str = "raa_gi_area_trace";

/// Side of the trace compute workgroup (8x8 threads).
pub const WORKGROUP: u32 = 8;

/// Size of `AreaShadowParamsGpu`: `frame_index` and three pad lanes.
pub const AREA_PARAMS_SIZE: u64 = 16;

/// Size of `AreaDispatchGpu`: `index`, `usable`, `pairs` and `limit`.
pub const AREA_DISPATCH_SIZE: u64 = 16;

/// Entry of the area-shadow temporal (history reprojection and blend) pipeline.
pub const AREA_TEMPORAL_ENTRY: &str = "raa_gi_area_temporal";

/// Entry of the area-shadow horizontal spatial (bilateral) pipeline.
pub const AREA_SPATIAL_H_ENTRY: &str = "raa_gi_area_spatial_h";

/// Entry of the area-shadow vertical spatial (bilateral and ratio) pipeline.
pub const AREA_SPATIAL_V_ENTRY: &str = "raa_gi_area_spatial_v";

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

/// A filtering sampler (the temporal history's bilinear read); the screen helpers build no
/// sampler.
fn sampler_entry(binding: u32) -> BindGroupLayoutEntry {
    entry(binding, BindingType::Sampler(SamplerBindingType::Filtering))
}

/// `temporal_module_wgsl` builds the temporal module of contract 'area-shadow-denoise-wgsl':
/// `raa_taa::REPROJECT_WGSL` (the `RaaTemporalFrame` twin and `raa_temporal_surface_uv`) then
/// wgsl/area_shadow/temporal.wgsl, whose entry `AREA_TEMPORAL_ENTRY` reprojects the history pair
/// through the S12 motion target and blends it into the raw trace.
///
/// Handoff: card 'raa-gi: AreaShadowPipelines and the raa-gi.area-shadow compute pass record'
/// builds the temporal and the spatial pipelines from `temporal_module_wgsl`,
/// `temporal_group0_entries`, `AREA_TEMPORAL_ENTRY`, `spatial_module_wgsl`,
/// `spatial_group0_entries`, `AREA_SPATIAL_H_ENTRY` and `AREA_SPATIAL_V_ENTRY`; this file builds
/// the module texts and the group-0 entry arrays and builds no pipeline. It extends the trace
/// share's Handoff at crates/raa-gi/src/parts/area_shadow/passes.rs:76 `trace_module_wgsl` and
/// does not restate it.
pub fn temporal_module_wgsl() -> String {
    format!("{REPROJECT_WGSL}\n{AREA_TEMPORAL_WGSL}")
}

/// Temporal group 0, ascending: 0 `raa_area_frame` (`RaaTemporalFrame`, 768 B), 1 the depth the
/// trace wrote this frame, 2 the S12 motion, 3 the raw history layer, 4 the filterable history
/// layer read, 5 the previous depth, 6 the Rgba16Float history write, 7 `AreaDispatchGpu` (16 B)
/// and 8 the filtering sampler.
pub fn temporal_group0_entries() -> [BindGroupLayoutEntry; 9] {
    [
        uniform_entry(0, TEMPORAL_FRAME_SIZE),
        float_entry(1, false),
        float_entry(2, false),
        float_entry(3, false),
        float_entry(4, true),
        float_entry(5, false),
        storage_texture_entry(6, TextureFormat::Rgba16Float),
        uniform_entry(7, AREA_DISPATCH_SIZE),
        sampler_entry(8),
    ]
}

/// `spatial_module_wgsl` builds the spatial module of contract 'area-shadow-denoise-wgsl':
/// wgsl/area_shadow/spatial.wgsl alone, whose entries `AREA_SPATIAL_H_ENTRY` and
/// `AREA_SPATIAL_V_ENTRY` filter the reprojected history along x then y and write the ratio
/// estimator's visibility into the mask. Both entries share `spatial_group0_entries`.
pub fn spatial_module_wgsl() -> String {
    AREA_SPATIAL_WGSL.to_string()
}

/// Spatial group 0, ascending: 1 the depth the trace wrote this frame, 2 the S12 view normal,
/// 3 the source layer, 4 the second source layer (unused by the horizontal entry), 5 the
/// Rgba16Float destination and 7 `AreaDispatchGpu` (16 B).
pub fn spatial_group0_entries() -> [BindGroupLayoutEntry; 6] {
    [
        float_entry(1, false),
        float_entry(2, false),
        float_entry(3, false),
        float_entry(4, false),
        storage_texture_entry(5, TextureFormat::Rgba16Float),
        uniform_entry(7, AREA_DISPATCH_SIZE),
    ]
}
