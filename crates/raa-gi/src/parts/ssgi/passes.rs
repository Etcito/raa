//! Kernels of the `gi.ssgi` pass (contract 'ssgi-trace-wgsl'): the trace module
//! (`PRELUDE_WGSL`, `CLOUD_SHADOW_WGSL`, `GI_VOLUME_WGSL`, `DEPTH_WGSL`, probe_common.wgsl,
//! screen_trace.wgsl and wgsl/ssgi/trace.wgsl, in that order), its entry name and its six
//! group-2 bindings. `SsgiParamsGpu` and the raw storage output live in trace.wgsl. The pipeline
//! layout is [empty group 0, `GiShared::compute_group1_layout`, `ssgi_trace_group2_entries`];
//! the entry statically uses group 1 binding 6 only (`raa_linear_clamp`, through
//! `raa_gi_screen_hit_radiance`).
#![allow(dead_code)]

use raa_shader::{CLOUD_SHADOW_WGSL, DEPTH_WGSL, GI_VOLUME_WGSL, PRELUDE_WGSL};
use wgpu::{BindGroupLayoutEntry, TextureFormat};

use super::super::screen::passes::{
    depth_entry, float_entry, storage_texture_entry, uniform_entry,
};

const PROBE_COMMON_WGSL: &str = include_str!("../../wgsl/screen/probe_common.wgsl");
const SCREEN_TRACE_WGSL: &str = include_str!("../../wgsl/screen/screen_trace.wgsl");
const SSGI_TRACE_WGSL: &str = include_str!("../../wgsl/ssgi/trace.wgsl");

/// `GiFrameGpu` bytes (S38): five matrices, then the seven 16-B lanes.
const FRAME_SIZE: u64 = 432;

/// Entry of the SSGI trace pipeline.
pub const SSGI_TRACE_ENTRY: &str = "raa_gi_ssgi_trace";

/// Side of the trace compute workgroup (`@workgroup_size(8, 8)`).
pub const WORKGROUP: u32 = 8;

/// Size of `SsgiParamsGpu`: slices, steps, history_valid, frame (u32), radius_m, thickness_m
/// (f32), temporal_valid and pad (u32).
pub const SSGI_PARAMS_SIZE: u64 = 32;

/// `ssgi_trace_module_wgsl` builds the trace module of contract 'ssgi-trace-wgsl', composed as
/// crates/raa-gi/src/parts/area_shadow/passes.rs `trace_module_wgsl` is: `PRELUDE_WGSL` +
/// `CLOUD_SHADOW_WGSL` + `GI_VOLUME_WGSL` + `DEPTH_WGSL` + probe_common.wgsl +
/// screen_trace.wgsl + wgsl/ssgi/trace.wgsl. Its entry `SSGI_TRACE_ENTRY` writes the raw
/// texel (E_ss rgb, V) of the visibility-bitmask trace per pixel.
///
/// Handoff: card 'raa-gi ssgi: temporal resolve kernel, SsgiPipelines and the compute record'
/// builds the trace pipeline from `ssgi_trace_module_wgsl`, `ssgi_trace_group2_entries` and
/// `SSGI_TRACE_ENTRY`; today this file builds the composed module text and the six group-2
/// entries and builds no pipeline.
pub fn ssgi_trace_module_wgsl() -> String {
    format!(
        "{PRELUDE_WGSL}{CLOUD_SHADOW_WGSL}{GI_VOLUME_WGSL}{DEPTH_WGSL}\
         {PROBE_COMMON_WGSL}{SCREEN_TRACE_WGSL}{SSGI_TRACE_WGSL}"
    )
}

/// Trace group 2, ascending: 0 `GiFrameGpu` (432 B), 1 the S12 depth, 2 the S12 view-space
/// normal, 3 the previous lit HDR (`GiShared::history_view`, or a 1x1 dummy, filterable), 5 the
/// Rgba16Float raw output `raa_gi_ssgi_raw` and 8 `raa_gi_ssgi_params` (`SSGI_PARAMS_SIZE`).
pub fn ssgi_trace_group2_entries() -> [BindGroupLayoutEntry; 6] {
    [
        uniform_entry(0, FRAME_SIZE),
        depth_entry(1),
        float_entry(2, false),
        float_entry(3, true),
        storage_texture_entry(5, TextureFormat::Rgba16Float),
        uniform_entry(8, SSGI_PARAMS_SIZE),
    ]
}
