//! The M7 ultra tier of the area-light shadows (S12 order 43, contract `area-restir-wgsl`):
//! ReSTIR DI (Bitterli et al. 2020) organised as in Wyman and Panteleev, HPG 2021, with GRIS
//! weights (Lin et al. 2022). This module holds the Rust builders of the three compute stages
//! and the group-2 binding registry they share. Share 1 (this file today) builds the stage-less
//! common module and the 17 group-2 entries; the stage builders, the pipelines and the resources
//! are later shares of this module; the initial-candidates share adds `restir_initial_module_wgsl`
//! and `RESTIR_INITIAL_ENTRY`, the temporal-reuse share `restir_temporal_module_wgsl` and
//! `RESTIR_TEMPORAL_ENTRY`, the spatial-reuse share `restir_spatial_module_wgsl` and
//! `RESTIR_SPATIAL_ENTRY`. The WGSL is RAA's own code; Bevy Solari is not supplied and
//! nothing is copied from it or from NVIDIA RTXDI.
#![allow(dead_code)]

use raa_core::GpuContext;
use raa_pipeline::AREA_STOCHASTIC_GPU_SIZE;
use raa_shader::{DEPTH_WGSL, PRELUDE_WGSL};
use raa_taa::{REPROJECT_WGSL, TEMPORAL_FRAME_SIZE};
use wgpu::{
    BindGroupLayoutEntry, BindingType, StorageTextureAccess, TextureFormat, TextureSampleType,
    TextureViewDimension,
};

use crate::tunables::GiTunables;

use super::passes::{AreaShadowPipelines, AreaShadowWork, trace_module_wgsl, tuned_pipeline};
use crate::parts::screen::passes::{
    depth_entry, dispatch, entry, float_entry, layout, module, storage_entry,
    storage_texture_entry, uniform_entry,
};

/// The binding-free ReSTIR core (codec, random numbers, WRS, GRIS weight, boiling test), shared
/// with the indirect-diffuse ReSTIR GI of the screen part, then the DI bindings and helpers.
const RESTIR_COMMON_WGSL: &str = concat!(
    include_str!("../../wgsl/area_shadow/restir_core.wgsl"),
    include_str!("../../wgsl/area_shadow/restir_common.wgsl")
);
const PROBE_COMMON_WGSL: &str = include_str!("../../wgsl/screen/probe_common.wgsl");
const RESTIR_INITIAL_WGSL: &str = include_str!("../../wgsl/area_shadow/restir_initial.wgsl");
const RESTIR_TEMPORAL_WGSL: &str = include_str!("../../wgsl/area_shadow/restir_temporal.wgsl");
const RESTIR_SPATIAL_WGSL: &str = include_str!("../../wgsl/area_shadow/restir_spatial.wgsl");

/// The (override, GI tunable key) pair of restir_core.wgsl, which every stage module composes.
/// The ReSTIR GI modules of the screen part compose restir_core.wgsl too and keep its default.
pub(crate) const AREA_RESTIR_OVERRIDES: &[(&str, &str)] =
    &[("RAA_RESTIR_DEPTH_TOLERANCE", "restirDepthTolerance")];

/// Entry of the ReSTIR initial-candidates pipeline (`@compute @workgroup_size(8, 8)`), one
/// invocation per viewport pixel.
pub const RESTIR_INITIAL_ENTRY: &str = "raa_gi_area_restir_initial";

/// Entry of the ReSTIR temporal-reuse pipeline (`@compute @workgroup_size(8, 8)`), one
/// invocation per viewport pixel.
pub const RESTIR_TEMPORAL_ENTRY: &str = "raa_gi_area_restir_temporal";

/// Entry of the ReSTIR spatial-reuse and resolve pipeline (`@compute @workgroup_size(8, 8)`),
/// one invocation per viewport pixel.
pub const RESTIR_SPATIAL_ENTRY: &str = "raa_gi_area_restir_spatial";

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

/// `restir_temporal_module_wgsl` builds the temporal-reuse module of contract
/// 'area-restir-wgsl': the same trace module + `raa_taa::REPROJECT_WGSL` + restir_common.wgsl
/// as `restir_initial_module_wgsl`, then wgsl/area_shadow/restir_temporal.wgsl. Its compute
/// entries are `AREA_TRACE_ENTRY` and `RESTIR_TEMPORAL_ENTRY`; the latter reprojects the
/// previous frame's reservoir of `raa_restir_history` through the S12 motion target, rejects
/// it on depth, normal or slot remap, clamps its confidence, combines it with the pixel's
/// `raa_restir_current` reservoir by GRIS, applies the workgroup boiling filter and writes
/// `raa_restir_current` in place, using only group-2 bindings of `restir_group2_entries`.
pub fn restir_temporal_module_wgsl() -> String {
    format!(
        "{}{REPROJECT_WGSL}{RESTIR_COMMON_WGSL}{RESTIR_TEMPORAL_WGSL}",
        trace_module_wgsl()
    )
}

/// `restir_spatial_module_wgsl` builds the spatial-reuse and resolve module of contract
/// 'area-restir-wgsl': the same trace module + `raa_taa::REPROJECT_WGSL` + restir_common.wgsl
/// as `restir_initial_module_wgsl`, then wgsl/area_shadow/restir_spatial.wgsl. Its compute
/// entries are `AREA_TRACE_ENTRY` and `RESTIR_SPATIAL_ENTRY`; the latter combines the pixel's
/// `raa_restir_current` reservoir with up to four depth- and normal-compatible neighbours by
/// GRIS with the generalized balance heuristic, writes the merged reservoir into
/// `raa_restir_history` (the next frame's temporal history), traces the pixel's second
/// visibility ray towards the merged sample and writes (S, U~, S, U~) into raw layer 0 of
/// `raa_area_raw_out`, U~ the slot-table sum of `raa_restir_ltc_luminance`, using only group-2
/// bindings of `restir_group2_entries`.
pub fn restir_spatial_module_wgsl() -> String {
    format!(
        "{}{REPROJECT_WGSL}{RESTIR_COMMON_WGSL}{RESTIR_SPATIAL_WGSL}",
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

/// The three ReSTIR compute pipelines (initial, temporal, spatial) over [empty group 0, the GI
/// compute group 1, `layout`], with `layout` built from `restir_group2_entries`.
///
/// Handoff: card 'raa-gi: RestirResources (reservoir buffers, params, LTC atlas, bind group)'
/// builds its group-2 bind group over `RestirPipelines.layout`; card 'raa-gi: gi.area-shadow
/// part runs the ReSTIR ultra tier on Restir frames' calls `RestirPipelines::new` and
/// `RestirPipelines::record`. LIFECYCLE: RestirPipelines is created lazily by
/// GiAreaShadowPass::prepare on the first Restir frame and kept for the part's lifetime; record
/// runs in GiAreaShadowPass::record after gi.screen (order 42) published the pyramid. `record`
/// calls `AreaShadowPipelines::record_denoise` (Handoff at passes.rs:336 `record_denoise`).
pub(crate) struct RestirPipelines {
    pub(crate) layout: wgpu::BindGroupLayout,
    initial: wgpu::ComputePipeline,
    temporal: wgpu::ComputePipeline,
    spatial: wgpu::ComputePipeline,
    empty: wgpu::BindGroupLayout,
    group1: wgpu::BindGroupLayout,
    modules: [wgpu::ShaderModule; 3],
    constants: Vec<(&'static str, f64)>,
}

/// Label and entry of the initial, temporal and spatial stages.
const STAGES: [(&str, &str); 3] = [
    ("raa-gi.area-restir.initial", RESTIR_INITIAL_ENTRY),
    ("raa-gi.area-restir.temporal", RESTIR_TEMPORAL_ENTRY),
    ("raa-gi.area-restir.spatial", RESTIR_SPATIAL_ENTRY),
];

impl RestirPipelines {
    /// Builds one shader module per builder and the three pipelines over [empty group 0,
    /// `group1`, `layout`], at the GI tunable defaults of `AREA_RESTIR_OVERRIDES`.
    pub(crate) fn new(gpu: &GpuContext, group1: &wgpu::BindGroupLayout) -> RestirPipelines {
        RestirPipelines::tuned(gpu, group1, &GiTunables::new())
    }

    /// [`RestirPipelines::new`] with the override constants of `tunables`.
    pub(crate) fn tuned(
        gpu: &GpuContext,
        group1: &wgpu::BindGroupLayout,
        tunables: &GiTunables,
    ) -> RestirPipelines {
        let device = gpu.device();
        let empty = layout(device, "raa-gi.area-restir.empty", &[]);
        let group2 = layout(device, "raa-gi.area-restir", &restir_group2_entries());
        let sources = [
            restir_initial_module_wgsl(),
            restir_temporal_module_wgsl(),
            restir_spatial_module_wgsl(),
        ];
        let mut stage = 0;
        let modules = sources.map(|source| {
            stage += 1;
            module(device, STAGES[stage - 1].0, &source)
        });
        let constants = tunables.constants(AREA_RESTIR_OVERRIDES);
        let [initial, temporal, spatial] =
            Self::build(device, &modules, [&empty, group1, &group2], &constants);
        RestirPipelines {
            layout: group2,
            initial,
            temporal,
            spatial,
            empty,
            group1: group1.clone(),
            modules,
            constants,
        }
    }

    /// The three stage pipelines of `modules` over `groups` with `constants`.
    fn build(
        device: &wgpu::Device,
        modules: &[wgpu::ShaderModule; 3],
        groups: [&wgpu::BindGroupLayout; 3],
        constants: &[(&'static str, f64)],
    ) -> [wgpu::ComputePipeline; 3] {
        let layouts = groups.map(Some);
        let mut index = 0;
        STAGES.map(|(label, entry_point)| {
            index += 1;
            tuned_pipeline(
                device,
                label,
                &modules[index - 1],
                entry_point,
                &layouts,
                constants,
            )
        })
    }

    /// Rebuilds the three pipelines when the constants of `tunables` for
    /// `AREA_RESTIR_OVERRIDES` differ from the ones they were built with; the layouts stay.
    pub(crate) fn refresh(&mut self, gpu: &GpuContext, tunables: &GiTunables) {
        let constants = tunables.constants(AREA_RESTIR_OVERRIDES);
        if constants == self.constants {
            return;
        }
        let groups = [&self.empty, &self.group1, &self.layout];
        [self.initial, self.temporal, self.spatial] =
            Self::build(gpu.device(), &self.modules, groups, &constants);
        self.constants = constants;
    }

    /// The initial-candidates pipeline (`RESTIR_INITIAL_ENTRY`), recorded alone by `area.restir.initial`.
    pub(crate) fn initial(&self) -> &wgpu::ComputePipeline {
        &self.initial
    }

    /// The temporal-reuse pipeline (`RESTIR_TEMPORAL_ENTRY`), recorded alone by `area.restir.temporal`.
    pub(crate) fn temporal(&self) -> &wgpu::ComputePipeline {
        &self.temporal
    }

    /// The spatial-reuse pipeline (`RESTIR_SPATIAL_ENTRY`), recorded alone by `area.restir.spatial`.
    pub(crate) fn spatial(&self) -> &wgpu::ComputePipeline {
        &self.spatial
    }

    /// Records one `raa-gi.area-shadow.restir` compute pass: group 1 and group 2 set once, the
    /// initial, temporal and spatial dispatches over `work.viewport`
    /// (`groups_for(width) x groups_for(height) x 1`), then `denoise.record_denoise`. Returns
    /// false and records nothing when `work.temporal` is empty or a viewport axis is 0.
    pub(crate) fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        group1: &wgpu::BindGroup,
        group2: &wgpu::BindGroup,
        denoise: &AreaShadowPipelines,
        work: &AreaShadowWork<'_>,
    ) -> bool {
        let (width, height) = work.viewport;
        if work.temporal.is_empty() || width == 0 || height == 0 {
            return false;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raa-gi.area-shadow.restir"),
            timestamp_writes: None,
        });
        pass.set_bind_group(1, group1, &[]);
        pass.set_bind_group(2, group2, &[]);
        for pipeline in [&self.initial, &self.temporal, &self.spatial] {
            pass.set_pipeline(pipeline);
            dispatch(&mut pass, width, height);
        }
        denoise.record_denoise(&mut pass, work);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parts::GiShared;
    use raa_core::testing::{block_on, skip};

    #[test]
    fn restir_pipelines_build_in_a_validation_scope_and_record_nothing_without_history() {
        let test =
            "restir_pipelines_build_in_a_validation_scope_and_record_nothing_without_history";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let shared = GiShared::new(&gpu);
        let group1_layout = shared.compute_group1_layout();
        let scope = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let mut tunables = shared.tunables();
        let mut restir = RestirPipelines::tuned(&gpu, &group1_layout, &tunables);
        let denoise = AreaShadowPipelines::tuned(&gpu, &group1_layout, &tunables);
        assert!(matches!(
            tunables.set("restirDepthTolerance", 0.2),
            Ok(true)
        ));
        restir.refresh(&gpu, &tunables);
        assert!(
            block_on(scope.pop()).is_none(),
            "the three ReSTIR pipelines and the denoiser create without a validation error"
        );

        let empty = gpu
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("raa-gi.area-restir.test.empty"),
                entries: &[],
            });
        let group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-gi.area-restir.test.empty"),
            layout: &empty,
            entries: &[],
        });
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("raa-gi.area-restir.test"),
            });
        let work = AreaShadowWork {
            trace: &group,
            temporal: &[],
            spatial_h: &[],
            spatial_v: &[],
            viewport: (64, 64),
        };
        assert!(
            !restir.record(&mut encoder, &group, &group, &denoise, &work),
            "an empty temporal slice records nothing"
        );
    }
}
