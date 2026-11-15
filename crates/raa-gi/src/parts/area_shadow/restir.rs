//! The M7 ultra tier of the area-light shadows (S12 order 43, contract `area-restir-wgsl`):
//! ReSTIR DI (Bitterli et al. 2020) organised as in Wyman and Panteleev, HPG 2021, with GRIS
//! weights (Lin et al. 2022). This module holds the Rust builders of the three compute stages
//! and the group-2 binding registry they share. Share 1 (this file today) builds the stage-less
//! common module and the 17 group-2 entries; the stage builders, the pipelines and the resources
//! are later shares of this module; the initial-candidates share adds `restir_initial_module_wgsl`
//! and `RESTIR_INITIAL_ENTRY`. The WGSL is RAA's own code; Bevy Solari is not supplied and
//! nothing is copied from it or from NVIDIA RTXDI.
#![allow(dead_code)]

use raa_pipeline::AREA_STOCHASTIC_GPU_SIZE;
use raa_shader::{DEPTH_WGSL, PRELUDE_WGSL};
use raa_taa::{REPROJECT_WGSL, TEMPORAL_FRAME_SIZE};
use wgpu::{
    BindGroupLayoutEntry, BindingType, StorageTextureAccess, TextureFormat, TextureSampleType,
    TextureViewDimension,
};

use super::passes::trace_module_wgsl;
use crate::parts::screen::passes::{
    depth_entry, entry, float_entry, storage_entry, storage_texture_entry, uniform_entry,
};

const RESTIR_COMMON_WGSL: &str = include_str!("../../wgsl/area_shadow/restir_common.wgsl");
const PROBE_COMMON_WGSL: &str = include_str!("../../wgsl/screen/probe_common.wgsl");
const RESTIR_INITIAL_WGSL: &str = include_str!("../../wgsl/area_shadow/restir_initial.wgsl");

/// Entry of the ReSTIR initial-candidates pipeline (`@compute @workgroup_size(8, 8)`), one
/// invocation per viewport pixel.
pub const RESTIR_INITIAL_ENTRY: &str = "raa_gi_area_restir_initial";

/// `GiFrameGpu` bytes (S38): five matrices, then the seven 16-B lanes. The stage files declare
/// the record and its group-2 binding 0; `probe_common.wgsl` declares both in the stage-less
/// validation module built by [`restir_common_module_wgsl`].
const FRAME_SIZE: u64 = 432;

/// The one group-2 binding the stage-less common module uses besides `probe_common.wgsl`'s:
/// 8, the mid-tier `RaaAreaStochasticGpu` slot table share 2 reads through `raa_area_table`.
const RESTIR_VALIDATION_BINDINGS_WGSL: &str =
    "@group(2) @binding(8) var<uniform> raa_area_table: RaaAreaStochasticGpu;\n\n";

/// Size of `RestirParamsGpu`: four u32, the two `remap` lanes, then four f32.
pub const RESTIR_PARAMS_SIZE: u64 = 64;

/// Size of one packed reservoir record: `pack2x16float` pairs of uv, (W, M) and the normal.
pub const RESERVOIR_BYTES: u64 = 16;

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

/// The raa-sdf-trace SDF field: an unfilterable sampled `texture_3d<f32>` (R8Snorm, five mips,
/// or a 1x1x1 dummy); the screen helpers build only `D2` entries.
fn sdf_field_entry(binding: u32) -> BindGroupLayoutEntry {
    entry(
        binding,
        BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: false },
            view_dimension: TextureViewDimension::D3,
            multisampled: false,
        },
    )
}

/// `restir_common_module_wgsl` builds the stage-less common module of contract
/// 'area-restir-wgsl': `PRELUDE_WGSL` + `DEPTH_WGSL` + `probe_common.wgsl` (the `GiFrameGpu`
/// record, the group-2 bindings 0-2 and the world-position, world-normal and view-position
/// helpers share 2 reads) + `raa_taa::REPROJECT_WGSL` (the `RaaTemporalFrame` twin) + a
/// `raa_area_table` group-2 binding 8 + wgsl/area_shadow/restir_common.wgsl. It has no entry
/// point: today it holds the group-2 bindings 12-19, `RestirParamsGpu`, `RaaRestirReservoir`
/// and the share-1 codec and reuse helpers. The stage files declare their own `GiFrameGpu` and
/// group-2 bindings 0-11, so a stage builder composes `PRELUDE_WGSL` + `DEPTH_WGSL` +
/// `probe_common.wgsl`'s helpers (or its own) + `REPROJECT_WGSL` + the stage file + the
/// `RESTIR_COMMON_WGSL` include of this file instead of calling this validation builder.
///
/// Handoff: card 'raa-gi: ReSTIR initial candidates kernel with visibility reuse' takes over
/// `restir_common_module_wgsl` and the `RESTIR_COMMON_WGSL` include: today the builder returns
/// the stage-less common module text and this share declares no entry point.
pub fn restir_common_module_wgsl() -> String {
    format!(
        "{PRELUDE_WGSL}{DEPTH_WGSL}{PROBE_COMMON_WGSL}{REPROJECT_WGSL}\
         {RESTIR_VALIDATION_BINDINGS_WGSL}{RESTIR_COMMON_WGSL}"
    )
}

/// `restir_initial_module_wgsl` builds the initial-candidates module of contract
/// 'area-restir-wgsl': the area-shadow trace module of `passes::trace_module_wgsl`
/// (`PRELUDE_WGSL` + `CLOUD_SHADOW_WGSL` + `GI_VOLUME_WGSL` + `DEPTH_WGSL` + probe_common.wgsl +
/// screen_trace.wgsl + trace.wgsl, which declare the `GiFrameGpu` record, the group-2 bindings
/// 0-11 and `raa_area_ray_visibility`) + `raa_taa::REPROJECT_WGSL` + restir_common.wgsl +
/// wgsl/area_shadow/restir_initial.wgsl. Its compute entries are `AREA_TRACE_ENTRY` and
/// `RESTIR_INITIAL_ENTRY`; the latter draws the RIS candidates of a pixel, traces one
/// visibility ray towards the kept sample and writes `raa_restir_current` and
/// `raa_area_depth_out`, using only group-2 bindings of `restir_group2_entries`.
pub fn restir_initial_module_wgsl() -> String {
    format!(
        "{}{REPROJECT_WGSL}{RESTIR_COMMON_WGSL}{RESTIR_INITIAL_WGSL}",
        trace_module_wgsl()
    )
}

/// Group 2 of every ReSTIR stage, ascending: 0 `GiFrameGpu` (432 B), 1 the S12 depth, 2 the S12
/// normal, 4 the depth pyramid (or its 1x1 dummy), 5 the Rgba16Float raw history array, 6 the
/// R32Float depth-pair write view, 8 `RaaAreaStochasticGpu` (80 B), 10 the raa-sdf-trace SDF
/// field, 11 its `RaaAreaSdfVolume` uniform (64 B), 12 the S12 motion, 13 the lit
/// `RaaTemporalFrame` (768 B), 14 the depth pair's read view, 15 `RestirParamsGpu` (64 B), 16
/// the read_write current reservoirs, 17 the read_write history reservoirs, 18 the filterable
/// LTC atlas and 19 the S12 material. Binding 9 is not in this layout.
///
/// Handoff: card 'raa-gi: RestirResources (reservoir buffers, params, LTC atlas, bind group)'
/// takes over `restir_group2_entries`: it builds the ReSTIR group-2 layout and bind group from
/// these 17 ascending entries today, and this share returns the entries and builds no layout.
pub fn restir_group2_entries() -> [BindGroupLayoutEntry; 17] {
    [
        uniform_entry(0, FRAME_SIZE),
        depth_entry(1),
        float_entry(2, false),
        float_entry(4, false),
        storage_texture_array_entry(5, TextureFormat::Rgba16Float),
        storage_texture_entry(6, TextureFormat::R32Float),
        uniform_entry(8, AREA_STOCHASTIC_GPU_SIZE as u64),
        sdf_field_entry(10),
        uniform_entry(11, 64),
        float_entry(12, false),
        uniform_entry(13, TEMPORAL_FRAME_SIZE),
        float_entry(14, false),
        uniform_entry(15, RESTIR_PARAMS_SIZE),
        storage_entry(16, false),
        storage_entry(17, false),
        float_entry(18, true),
        float_entry(19, false),
    ]
}
