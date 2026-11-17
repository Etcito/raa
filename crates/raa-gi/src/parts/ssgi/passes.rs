//! Kernels of the `gi.ssgi` pass (contract 'ssgi-trace-wgsl'): the trace module
//! (`PRELUDE_WGSL`, `CLOUD_SHADOW_WGSL`, `GI_VOLUME_WGSL`, `DEPTH_WGSL`, probe_common.wgsl,
//! screen_trace.wgsl and wgsl/ssgi/trace.wgsl, in that order), its entry name and its six
//! group-2 bindings. `SsgiParamsGpu` and the raw storage output live in trace.wgsl. The pipeline
//! layout is [empty group 0, `GiShared::compute_group1_layout`, `ssgi_trace_group2_entries`];
//! the entry statically uses group 1 binding 6 only (`raa_linear_clamp`, through
//! `raa_gi_screen_hit_radiance`).
// `SsgiPipelines` and `SsgiWork` have no caller until the SsgiResources and SsgiPart shares land.
#![allow(dead_code)]

use crate::parts::GiShared;
use crate::tunables::GiTunables;
use raa_core::GpuContext;
use raa_shader::{CLOUD_SHADOW_WGSL, DEPTH_WGSL, GI_VOLUME_WGSL, PRELUDE_WGSL};
use raa_taa::{REPROJECT_WGSL, TEMPORAL_FRAME_SIZE};
use wgpu::{BindGroupLayoutEntry, TextureFormat};

use super::super::area_shadow::passes::tuned_pipeline;
use super::super::screen::passes::{
    compute_pipeline, depth_entry, dispatch, float_entry, layout, module, storage_texture_entry,
    uniform_entry,
};

const PROBE_COMMON_WGSL: &str = include_str!("../../wgsl/screen/probe_common.wgsl");
const SCREEN_TRACE_WGSL: &str = include_str!("../../wgsl/screen/screen_trace.wgsl");
const SSGI_TRACE_WGSL: &str = include_str!("../../wgsl/ssgi/trace.wgsl");
const SSGI_RESOLVE_WGSL: &str = include_str!("../../wgsl/ssgi/resolve.wgsl");

/// The (override name, tunable key) pairs of resolve.wgsl, passed through `GiTunables::constants`.
pub(crate) const SSGI_RESOLVE_OVERRIDES: &[(&str, &str)] =
    &[("RAA_SSGI_CHANGE_SIGMA", "changeSigma")];

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

/// Entry of the SSGI temporal resolve pipeline (`@workgroup_size(8, 8)`).
pub const SSGI_RESOLVE_ENTRY: &str = "raa_gi_ssgi_resolve";

/// `ssgi_resolve_module_wgsl` builds the resolve module of contract 'ssgi-resolve-wgsl':
/// `raa_taa::REPROJECT_WGSL` then wgsl/ssgi/resolve.wgsl, whose entry `SSGI_RESOLVE_ENTRY`
/// filters the raw trace, accumulates it in the history pair and writes the probe composite.
pub fn ssgi_resolve_module_wgsl() -> String {
    format!("{REPROJECT_WGSL}\n{SSGI_RESOLVE_WGSL}")
}

/// Resolve group 0, ascending: 0 `RaaTemporalFrame` (768 B, `LitHandle::frame_buffer`), 1 the
/// S12 depth, 2 the S12 motion, 3 the raw trace, 4 the history read, 5 the probe output (or a
/// 1x1 (0, 0, 0, 1) dummy), 6 the Rgba16Float history write, 7 the Rgba16Float SSGI output and
/// 8 `SsgiParamsGpu` (`SSGI_PARAMS_SIZE`).
pub fn ssgi_resolve_group0_entries() -> [BindGroupLayoutEntry; 9] {
    [
        uniform_entry(0, TEMPORAL_FRAME_SIZE),
        depth_entry(1),
        float_entry(2, false),
        float_entry(3, false),
        float_entry(4, false),
        float_entry(5, false),
        storage_texture_entry(6, TextureFormat::Rgba16Float),
        storage_texture_entry(7, TextureFormat::Rgba16Float),
        uniform_entry(8, SSGI_PARAMS_SIZE),
    ]
}

/// The per-frame bind groups `SsgiPipelines::record` dispatches over: the trace group 2, the
/// resolve group 0, and the viewport both cover.
pub(crate) struct SsgiWork<'a> {
    pub(crate) trace: &'a wgpu::BindGroup,
    pub(crate) resolve: &'a wgpu::BindGroup,
    pub(crate) viewport: (u32, u32),
}

/// `SsgiPipelines` holds the two compute pipelines of `gi.ssgi` and their private layouts: the
/// trace over [empty group 0, the GI compute group 1, `trace_layout`] and the resolve over
/// [`resolve_layout`]. It builds the trace pipeline from the entry points of the Handoff at
/// crates/raa-gi/src/parts/ssgi/passes.rs:49 `ssgi_trace_module_wgsl` and does not restate it.
///
/// Handoff: card 'raa-gi ssgi: SsgiResources (raw, history pair, output, params, bind groups)'
/// calls `SsgiPipelines::new` and reads `SsgiPipelines.trace_layout` and
/// `SsgiPipelines.resolve_layout` to build the bind groups of `SsgiWork`; card 'raa-gi ssgi:
/// SsgiPart, the gi.ssgi pass (order 44) and the binding-14 pick' calls `SsgiPipelines::record`,
/// which today records the trace then the resolve in one "raa-gi.ssgi" compute pass.
pub(crate) struct SsgiPipelines {
    pub(crate) trace_layout: wgpu::BindGroupLayout,
    pub(crate) resolve_layout: wgpu::BindGroupLayout,
    trace: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    resolve_module: wgpu::ShaderModule,
    /// The GI tunables the resolve pipeline was built with, refreshed by `refresh`.
    pub(crate) tunables: GiTunables,
}

impl SsgiPipelines {
    /// Builds both modules and pipelines: the trace over [empty, `group1`, `trace_layout`] and
    /// the resolve over [`resolve_layout`].
    pub(crate) fn new(gpu: &GpuContext, group1: &wgpu::BindGroupLayout) -> SsgiPipelines {
        let device = gpu.device();
        let empty = layout(device, "raa-gi.ssgi.empty", &[]);
        let trace_layout = layout(device, "raa-gi.ssgi.trace", &ssgi_trace_group2_entries());
        let resolve_layout = layout(
            device,
            "raa-gi.ssgi.resolve",
            &ssgi_resolve_group0_entries(),
        );
        let trace_module = module(device, "raa-gi.ssgi.trace", &ssgi_trace_module_wgsl());
        let resolve_module = module(device, "raa-gi.ssgi.resolve", &ssgi_resolve_module_wgsl());
        let trace = compute_pipeline(
            device,
            "raa-gi.ssgi.trace",
            &trace_module,
            SSGI_TRACE_ENTRY,
            &[Some(&empty), Some(group1), Some(&trace_layout)],
            None,
        );
        let tunables = GiTunables::new();
        let resolve = resolve_pipeline(device, &resolve_module, &resolve_layout, &tunables);
        SsgiPipelines {
            trace_layout,
            resolve_layout,
            trace,
            resolve,
            resolve_module,
            tunables,
        }
    }

    /// When `GiShared::tunables_generation` differs from the cached copy's generation, clones
    /// `GiShared::tunables` once and rebuilds the resolve pipeline with its constants. Otherwise
    /// does nothing and allocates nothing.
    pub(crate) fn refresh(&mut self, gpu: &GpuContext, shared: &GiShared) {
        if shared.tunables_generation() == self.tunables.generation() {
            return;
        }
        self.tunables = shared.tunables();
        self.resolve = resolve_pipeline(
            gpu.device(),
            &self.resolve_module,
            &self.resolve_layout,
            &self.tunables,
        );
    }

    /// Records one "raa-gi.ssgi" compute pass: group 1 once, the trace over group 2, then the
    /// resolve over group 0, each `groups_for(w) x groups_for(h) x 1`. Returns false and records
    /// nothing when the viewport has a zero axis.
    pub(crate) fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        group1: &wgpu::BindGroup,
        work: &SsgiWork<'_>,
    ) -> bool {
        let (width, height) = work.viewport;
        if width == 0 || height == 0 {
            return false;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raa-gi.ssgi"),
            timestamp_writes: None,
        });
        pass.set_bind_group(1, group1, &[]);
        pass.set_pipeline(&self.trace);
        pass.set_bind_group(2, work.trace, &[]);
        dispatch(&mut pass, width, height);
        pass.set_pipeline(&self.resolve);
        pass.set_bind_group(0, work.resolve, &[]);
        dispatch(&mut pass, width, height);
        true
    }
}

/// The resolve pipeline over [`resolve_layout`] with the `SSGI_RESOLVE_OVERRIDES` constants.
fn resolve_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    resolve_layout: &wgpu::BindGroupLayout,
    tunables: &GiTunables,
) -> wgpu::ComputePipeline {
    tuned_pipeline(
        device,
        "raa-gi.ssgi.resolve",
        module,
        SSGI_RESOLVE_ENTRY,
        &[Some(resolve_layout)],
        &tunables.constants(SSGI_RESOLVE_OVERRIDES),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parts::GiShared;
    use raa_core::testing::{block_on, skip};

    #[test]
    fn pipelines_build_in_a_validation_scope_and_record_nothing_for_an_empty_viewport() {
        let test = "pipelines_build_in_a_validation_scope_and_record_nothing_for_an_empty_viewport";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        assert_eq!(SSGI_TRACE_ENTRY, "raa_gi_ssgi_trace");
        assert_eq!(SSGI_RESOLVE_ENTRY, "raa_gi_ssgi_resolve");
        let shared = GiShared::new(&gpu);
        let scope = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let pipelines = SsgiPipelines::new(&gpu, &shared.compute_group1_layout());
        assert!(
            block_on(scope.pop()).is_none(),
            "the trace and resolve pipelines create without a validation error"
        );

        let empty = gpu
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("raa-gi.ssgi.test.empty"),
                entries: &[],
            });
        let group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-gi.ssgi.test.empty"),
            layout: &empty,
            entries: &[],
        });
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("raa-gi.ssgi.test"),
            });
        let work = SsgiWork {
            trace: &group,
            resolve: &group,
            viewport: (0, 64),
        };
        assert!(
            !pipelines.record(&mut encoder, &group, &work),
            "a zero-width viewport records nothing"
        );
    }
}
