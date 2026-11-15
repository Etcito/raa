//! Kernels of the `gi.area-shadow` pass (S44, contracts `area-shadow-trace-wgsl` and
//! `area-shadow-denoise-wgsl`): the trace module (`PRELUDE_WGSL` + `CLOUD_SHADOW_WGSL` +
//! `GI_VOLUME_WGSL` + `DEPTH_WGSL` + probe_common.wgsl + screen_trace.wgsl +
//! wgsl/area_shadow/trace.wgsl) with its entry and eight group-2 bindings, the temporal module
//! (`REPROJECT_WGSL` + temporal.wgsl) with its nine group-0 bindings, and the spatial module
//! (spatial.wgsl) with the six group-0 bindings its two separable entries share.
//! `AreaShadowParamsGpu` and the group-2 storage outputs live in trace.wgsl;
//! `AreaDispatchGpu` and the denoiser bindings live in temporal.wgsl and spatial.wgsl.
//!
//! `AreaShadowPipelines` builds the four compute pipelines from these builders and records the
//! pass's one compute pass; the resources and the part are later shares of this module.
#![allow(dead_code)]

use raa_core::GpuContext;
use raa_pipeline::AREA_STOCHASTIC_GPU_SIZE;
use raa_shader::{CLOUD_SHADOW_WGSL, DEPTH_WGSL, GI_VOLUME_WGSL, PRELUDE_WGSL};
use raa_taa::{REPROJECT_WGSL, TEMPORAL_FRAME_SIZE};
use wgpu::{
    BindGroupLayoutEntry, BindingType, SamplerBindingType, StorageTextureAccess, TextureFormat,
    TextureViewDimension,
};

use super::super::screen::passes::{
    compute_pipeline, depth_entry, dispatch, entry, float_entry, layout, module,
    storage_texture_entry, uniform_entry,
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

/// Size of `RaaAreaSdfVolume`: the frozen `TraceVolume` ABI of raa-sdf-trace (`origin`,
/// `shape`, `encoding` and `build` at 16-B offsets).
pub const SDF_VOLUME_SIZE: u64 = 64;

/// The raa-sdf-trace SDF field: an unfilterable sampled `texture_3d<f32>` (R8Snorm, five
/// mips, or a 1x1x1 dummy); the screen helpers build only `D2` entries.
fn sdf_field_entry(binding: u32) -> BindGroupLayoutEntry {
    entry(
        binding,
        BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: TextureViewDimension::D3,
            multisampled: false,
        },
    )
}

/// Trace group 2, ascending: 0 `GiFrameGpu` (432 B), 1 the S12 depth, 2 the S12 normal, 4 the
/// depth pyramid (all mips), 5 the Rgba16Float raw history array, 6 the R32Float depth output,
/// 8 the `RaaAreaStochasticGpu` slot table (80 B), 9 `AreaShadowParamsGpu` (16 B), 10 the
/// raa-sdf-trace SDF field (`TraceVolume.view`) and 11 its `RaaAreaSdfVolume` uniform (64 B,
/// `TraceVolume.uniform`).
pub fn trace_group2_entries() -> [BindGroupLayoutEntry; 10] {
    [
        uniform_entry(0, FRAME_SIZE),
        depth_entry(1),
        float_entry(2, false),
        float_entry(4, false),
        storage_texture_array_entry(5, TextureFormat::Rgba16Float),
        storage_texture_entry(6, TextureFormat::R32Float),
        uniform_entry(8, AREA_STOCHASTIC_GPU_SIZE as u64),
        uniform_entry(9, AREA_PARAMS_SIZE),
        sdf_field_entry(10),
        uniform_entry(11, SDF_VOLUME_SIZE),
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

/// `AreaShadowPipelines` holds the four compute pipelines of the `raa-gi.area-shadow` pass and
/// the private group layouts of their three stages: the trace over [empty group 0, the GI
/// compute group 1, `trace_layout`], and the temporal and the two spatial entries over their
/// group-0 layouts, one shader module per builder.
///
/// Handoff: card 'raa-gi: AreaShadowResources (raw, histories, mask, params, dummies, bind
/// groups)' takes over `AreaShadowPipelines`: it calls `AreaShadowPipelines::new`, which builds
/// the four pipelines over `AreaShadowPipelines.trace_layout`,
/// `AreaShadowPipelines.temporal_layout` and `AreaShadowPipelines.spatial_layout` today, reads
/// those layouts to build the trace, per-history-group temporal, horizontal spatial and
/// per-mask-layer vertical spatial bind groups of `AreaShadowWork`, and the part card records
/// them through `AreaShadowPipelines::record` in one `raa-gi.area-shadow` pass. It extends the
/// Handoff at crates/raa-gi/src/parts/area_shadow/passes.rs:79 `trace_module_wgsl` and
/// crates/raa-gi/src/parts/area_shadow/passes.rs:141 `temporal_module_wgsl` and does not
/// restate them.
pub(crate) struct AreaShadowPipelines {
    pub(crate) trace_layout: wgpu::BindGroupLayout,
    pub(crate) temporal_layout: wgpu::BindGroupLayout,
    pub(crate) spatial_layout: wgpu::BindGroupLayout,
    trace: wgpu::ComputePipeline,
    temporal: wgpu::ComputePipeline,
    spatial_h: wgpu::ComputePipeline,
    spatial_v: wgpu::ComputePipeline,
}

/// The per-frame bind groups and viewport `AreaShadowPipelines::record` dispatches over: the
/// trace group 2, one temporal and one horizontal spatial group per history group
/// (`cpu::groups_for(limit)`), one vertical spatial group per mask layer in use
/// (`limit.min(8).div_ceil(4)`), and the viewport the trace and every filter cover.
pub(crate) struct AreaShadowWork<'a> {
    pub(crate) trace: &'a wgpu::BindGroup,
    pub(crate) temporal: &'a [wgpu::BindGroup],
    pub(crate) spatial_h: &'a [wgpu::BindGroup],
    pub(crate) spatial_v: &'a [wgpu::BindGroup],
    pub(crate) viewport: (u32, u32),
}

impl AreaShadowPipelines {
    /// Builds one shader module per builder and the four pipelines: the trace over [empty group
    /// 0, `group1`, its trace group-2 layout], the temporal over its group-0 layout and the two
    /// spatial entries over their shared group-0 layout.
    pub(crate) fn new(gpu: &GpuContext, group1: &wgpu::BindGroupLayout) -> AreaShadowPipelines {
        let device = gpu.device();
        let empty = layout(device, "raa-gi.area-shadow.empty", &[]);
        let trace_layout = layout(device, "raa-gi.area-shadow.trace", &trace_group2_entries());
        let temporal_layout = layout(
            device,
            "raa-gi.area-shadow.temporal",
            &temporal_group0_entries(),
        );
        let spatial_layout = layout(
            device,
            "raa-gi.area-shadow.spatial",
            &spatial_group0_entries(),
        );

        let trace_module = module(device, "raa-gi.area-shadow.trace", &trace_module_wgsl());
        let temporal_module = module(
            device,
            "raa-gi.area-shadow.temporal",
            &temporal_module_wgsl(),
        );
        let spatial_module = module(device, "raa-gi.area-shadow.spatial", &spatial_module_wgsl());

        let trace = compute_pipeline(
            device,
            "raa-gi.area-shadow.trace",
            &trace_module,
            AREA_TRACE_ENTRY,
            &[Some(&empty), Some(group1), Some(&trace_layout)],
            None,
        );
        let temporal = compute_pipeline(
            device,
            "raa-gi.area-shadow.temporal",
            &temporal_module,
            AREA_TEMPORAL_ENTRY,
            &[Some(&temporal_layout)],
            None,
        );
        let spatial_h = compute_pipeline(
            device,
            "raa-gi.area-shadow.spatial-h",
            &spatial_module,
            AREA_SPATIAL_H_ENTRY,
            &[Some(&spatial_layout)],
            None,
        );
        let spatial_v = compute_pipeline(
            device,
            "raa-gi.area-shadow.spatial-v",
            &spatial_module,
            AREA_SPATIAL_V_ENTRY,
            &[Some(&spatial_layout)],
            None,
        );

        AreaShadowPipelines {
            trace_layout,
            temporal_layout,
            spatial_layout,
            trace,
            temporal,
            spatial_h,
            spatial_v,
        }
    }

    /// Records one `raa-gi.area-shadow` compute pass: group 1 set once, then the trace over its
    /// group 2, then every history group's temporal and horizontal spatial pass over group 0,
    /// then every mask layer's vertical spatial pass over group 0. Every dispatch is
    /// `groups_for(width) x groups_for(height) x 1`. Returns false and records nothing when
    /// `work.temporal` is empty or the viewport has a zero axis; otherwise it returns true.
    pub(crate) fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        group1: &wgpu::BindGroup,
        work: &AreaShadowWork<'_>,
    ) -> bool {
        let (width, height) = work.viewport;
        if work.temporal.is_empty() || width == 0 || height == 0 {
            return false;
        }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raa-gi.area-shadow"),
            timestamp_writes: None,
        });
        pass.set_bind_group(1, group1, &[]);

        pass.set_pipeline(&self.trace);
        pass.set_bind_group(2, work.trace, &[]);
        dispatch(&mut pass, width, height);

        self.record_denoise(&mut pass, work);
        true
    }

    /// Records the denoise of `work` into an open pass: the temporal pipeline over every
    /// `work.temporal` group, then spatial_h over every `work.spatial_h` group, then spatial_v
    /// over every `work.spatial_v` group, each bound at group 0 and dispatched
    /// `groups_for(width) x groups_for(height) x 1` of `work.viewport`. It neither begins a pass
    /// nor sets group 1 or 2.
    ///
    /// Handoff: card 'raa-gi: RestirPipelines and the ReSTIR compute pass record' takes over
    /// `AreaShadowPipelines::record_denoise`: it calls it inside its own compute pass after the
    /// ReSTIR resolve wrote (S, U~, S, U~) into raw layer 0 and the linear depth into the depth
    /// pair, with the `AreaShadowWork` of `AreaShadowResources::work_restir`; today
    /// `AreaShadowPipelines::record` calls it after the trace dispatch. It cites the Handoff at
    /// crates/raa-gi/src/parts/area_shadow/passes.rs:200 `AreaShadowPipelines` and does not
    /// restate it.
    pub(crate) fn record_denoise(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        work: &AreaShadowWork<'_>,
    ) {
        let (width, height) = work.viewport;
        pass.set_pipeline(&self.temporal);
        for group in work.temporal {
            pass.set_bind_group(0, group, &[]);
            dispatch(pass, width, height);
        }

        pass.set_pipeline(&self.spatial_h);
        for group in work.spatial_h {
            pass.set_bind_group(0, group, &[]);
            dispatch(pass, width, height);
        }

        pass.set_pipeline(&self.spatial_v);
        for group in work.spatial_v {
            pass.set_bind_group(0, group, &[]);
            dispatch(pass, width, height);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parts::GiShared;
    use raa_core::testing::{block_on, skip, software_gpu};

    #[test]
    fn pipelines_build_in_a_validation_scope_and_record_nothing_without_history() {
        let test = "pipelines_build_in_a_validation_scope_and_record_nothing_without_history";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let shared = GiShared::new(&gpu);
        let scope = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let pipelines = AreaShadowPipelines::new(&gpu, &shared.compute_group1_layout());
        assert!(
            block_on(scope.pop()).is_none(),
            "the four pipelines create without a validation error"
        );

        let layout = gpu
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("raa-gi.area-shadow.test.empty"),
                entries: &[],
            });
        let group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-gi.area-shadow.test.empty"),
            layout: &layout,
            entries: &[],
        });
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("raa-gi.area-shadow.test"),
            });
        let work = AreaShadowWork {
            trace: &group,
            temporal: &[],
            spatial_h: &[],
            spatial_v: &[],
            viewport: (64, 64),
        };
        assert!(
            !pipelines.record(&mut encoder, &group, &work),
            "an empty temporal slice records nothing"
        );
    }
}
