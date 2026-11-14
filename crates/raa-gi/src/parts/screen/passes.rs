// Adapted from solstice crates/solstice-render/src/gi/screen.rs at 7e465b5: the module
// builders, group-2 layouts and compute pipelines of the screen-probe passes (`:63-145`) and
// their record order (`:261-315`).
//
// MIT License
//
// Copyright (c) 2026 SamG-Coder
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! Screen probes of the gi.screen pass (S39, contracts `gi-screen-passes` and `gi-screen-wgsl`):
//! the four composed WGSL modules, the depth-pyramid sizes, the six group-2 binding registries
//! and the compute pipelines that `ScreenPipelines` records in one compute pass. Every module
//! keeps its WGSL file's MIT notice and solstice source path.
//!
//! The pipelines and `ScreenWork` are built and recorded by later shares of the part
//! (`ScreenResources` and the `gi.screen` pass), so nothing in this crate calls them yet.
#![allow(dead_code)]

use raa_core::GpuContext;
use raa_shader::{CLOUD_SHADOW_WGSL, DEPTH_WGSL, GI_VOLUME_WGSL, PRELUDE_WGSL};
use std::num::NonZeroU64;

use wgpu::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, ComputePipeline, ShaderModule,
    ShaderStages, StorageTextureAccess, TextureFormat, TextureSampleType, TextureViewDimension,
};

const DEPTH_PYRAMID_WGSL: &str = include_str!("../../wgsl/screen/depth_pyramid.wgsl");
const PROBE_COMMON_WGSL: &str = include_str!("../../wgsl/screen/probe_common.wgsl");
const SCREEN_TRACE_WGSL: &str = include_str!("../../wgsl/screen/screen_trace.wgsl");
const PROBE_TRACE_WGSL: &str = include_str!("../../wgsl/screen/probe_trace.wgsl");
const PROBE_PROJECT_WGSL: &str = include_str!("../../wgsl/screen/probe_project.wgsl");
const PROBE_INTEGRATE_WGSL: &str = include_str!("../../wgsl/screen/probe_integrate.wgsl");

/// `GiFrameGpu` bytes (S38): five matrices, then the seven 16-B lanes.
const FRAME_SIZE: u64 = 432;

/// The rays a probe traces on a side. Each option has its own trace and project pipeline, since
/// `PROBE_RAYS` is an override constant that the WGSL loops are unrolled over.
pub const PROBE_RAY_OPTIONS: [u32; 2] = [6, 8];

/// Size of `Probe` in `probe_common.wgsl`: nine SH vectors, position and normal.
pub const PROBE_BYTES: u64 = 176;

/// Size of `GiScreenGpu`: `history_valid`, `screen_traces`, `parity` and one pad lane.
pub const SCREEN_PARAMS_SIZE: u64 = 16;

/// Side of every screen compute workgroup (8x8 threads).
pub const WORKGROUP: u32 = 8;

/// Entry of the depth-pyramid copy pipeline.
pub const PYRAMID_COPY_ENTRY: &str = "raa_gi_pyramid_copy";

/// Entry of the depth-pyramid reduce pipeline.
pub const PYRAMID_REDUCE_ENTRY: &str = "raa_gi_pyramid_reduce";

/// Entry of the probe-trace pipeline.
pub const PROBE_TRACE_ENTRY: &str = "raa_gi_probe_trace";

/// Entry of the probe-project pipeline.
pub const PROBE_PROJECT_ENTRY: &str = "raa_gi_probe_project";

/// Entry of the probe spatial-filter pipeline.
pub const PROBE_FILTER_ENTRY: &str = "raa_gi_probe_filter";

/// Entry of the probe-integrate pipeline.
pub const PROBE_INTEGRATE_ENTRY: &str = "raa_gi_probe_integrate";

/// `depth_pyramid_module_wgsl` builds the depth-pyramid module of contract 'gi-screen-wgsl':
/// `DEPTH_WGSL` + depth_pyramid.wgsl. The upstream card's report at
/// crates/raa-gi/src/wgsl/screen/depth_pyramid.wgsl:42 `raa_gi_pyramid_copy` states what the
/// copy and reduce entries do today.
pub fn depth_pyramid_module_wgsl() -> String {
    format!("{DEPTH_WGSL}{DEPTH_PYRAMID_WGSL}")
}

/// `probe_trace_module_wgsl` builds the trace module of contract 'gi-screen-wgsl':
/// `PRELUDE_WGSL` + `CLOUD_SHADOW_WGSL` + `GI_VOLUME_WGSL` + `DEPTH_WGSL` + probe_common.wgsl +
/// screen_trace.wgsl + probe_trace.wgsl. The upstream card's report at
/// crates/raa-gi/src/wgsl/screen/probe_trace.wgsl:47 `raa_gi_probe_trace` states what the entry
/// does today.
///
/// Handoff: card 'raa-gi screen: screen_variants.rs (modules, layouts, gi compositions)' calls
/// `probe_trace_module_wgsl` today together with `depth_pyramid_module_wgsl`,
/// `probe_project_module_wgsl`, `probe_integrate_module_wgsl`,
/// `pyramid_copy_group2_entries`, `pyramid_reduce_group2_entries`, `trace_group2_entries`,
/// `project_group2_entries`, `filter_group2_entries` and `integrate_group2_entries` through
/// `raa_gi::parts::screen::passes`; they return the four composed module texts and the six
/// group-2 entry arrays, paired here with the entry names `PYRAMID_COPY_ENTRY`,
/// `PYRAMID_REDUCE_ENTRY`, `PROBE_TRACE_ENTRY`, `PROBE_PROJECT_ENTRY`, `PROBE_FILTER_ENTRY` and
/// `PROBE_INTEGRATE_ENTRY`.
pub fn probe_trace_module_wgsl() -> String {
    format!(
        "{PRELUDE_WGSL}{CLOUD_SHADOW_WGSL}{GI_VOLUME_WGSL}{DEPTH_WGSL}\
         {PROBE_COMMON_WGSL}{SCREEN_TRACE_WGSL}{PROBE_TRACE_WGSL}"
    )
}

/// `probe_project_module_wgsl` builds the project module of contract 'gi-screen-wgsl':
/// `PRELUDE_WGSL` + `CLOUD_SHADOW_WGSL` + `GI_VOLUME_WGSL` + `DEPTH_WGSL` + probe_common.wgsl +
/// probe_project.wgsl. The file header at
/// crates/raa-gi/src/wgsl/screen/probe_project.wgsl:6 states the group-2 registry its two
/// entries use today.
pub fn probe_project_module_wgsl() -> String {
    format!(
        "{PRELUDE_WGSL}{CLOUD_SHADOW_WGSL}{GI_VOLUME_WGSL}{DEPTH_WGSL}{PROBE_COMMON_WGSL}{PROBE_PROJECT_WGSL}"
    )
}

/// `probe_integrate_module_wgsl` builds the integrate module of contract 'gi-screen-wgsl':
/// `PRELUDE_WGSL` + `CLOUD_SHADOW_WGSL` + `GI_VOLUME_WGSL` + `DEPTH_WGSL` + probe_common.wgsl +
/// probe_integrate.wgsl. The upstream card's report at
/// crates/raa-gi/src/wgsl/screen/probe_integrate.wgsl:103 `raa_gi_probe_integrate` states what
/// the entry does today.
pub fn probe_integrate_module_wgsl() -> String {
    format!(
        "{PRELUDE_WGSL}{CLOUD_SHADOW_WGSL}{GI_VOLUME_WGSL}{DEPTH_WGSL}{PROBE_COMMON_WGSL}{PROBE_INTEGRATE_WGSL}"
    )
}

/// The depth-pyramid levels of `viewport`, mip 0 first: `viewport` with each axis at least 1,
/// then `(max(1, w / 2), max(1, h / 2))` down to 1x1. The count is
/// `floor(log2(max(w, h))) + 1`, the rule of crates/raa-ssr/src/hiz.rs:12-20.
pub fn pyramid_sizes(viewport: (u32, u32)) -> Vec<(u32, u32)> {
    let capacity = viewport.0.max(viewport.1).max(1).ilog2() as usize + 1;
    let mut sizes = Vec::with_capacity(capacity);
    let mut size = (viewport.0.max(1), viewport.1.max(1));
    sizes.push(size);
    while size != (1, 1) {
        size = ((size.0 / 2).max(1), (size.1 / 2).max(1));
        sizes.push(size);
    }
    sizes
}

fn entry(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

fn texture_entry(binding: u32, sample_type: TextureSampleType) -> BindGroupLayoutEntry {
    entry(
        binding,
        BindingType::Texture {
            sample_type,
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
    )
}

fn depth_entry(binding: u32) -> BindGroupLayoutEntry {
    texture_entry(binding, TextureSampleType::Depth)
}

fn float_entry(binding: u32, filterable: bool) -> BindGroupLayoutEntry {
    texture_entry(binding, TextureSampleType::Float { filterable })
}

fn storage_texture_entry(binding: u32, format: TextureFormat) -> BindGroupLayoutEntry {
    entry(
        binding,
        BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format,
            view_dimension: TextureViewDimension::D2,
        },
    )
}

fn uniform_entry(binding: u32, size: u64) -> BindGroupLayoutEntry {
    entry(
        binding,
        BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(size),
        },
    )
}

fn storage_entry(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
    entry(
        binding,
        BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    )
}

/// Copy group 2: 0 the S12 depth, 2 the R32Float destination mip.
pub fn pyramid_copy_group2_entries() -> [BindGroupLayoutEntry; 2] {
    [
        depth_entry(0),
        storage_texture_entry(2, TextureFormat::R32Float),
    ]
}

/// Reduce group 2: 1 the one-mip source (`filterable: false`), 2 the R32Float destination mip.
pub fn pyramid_reduce_group2_entries() -> [BindGroupLayoutEntry; 2] {
    [
        float_entry(1, false),
        storage_texture_entry(2, TextureFormat::R32Float),
    ]
}

/// Trace group 2: 0 `GiFrameGpu` (432 B), 1 the S12 depth, 2 the S12 normal, 3 the filterable
/// history, 4 the unfilterable pyramid, 5 the Rgba16Float ray atlas and 7 `GiScreenGpu` (16 B).
pub fn trace_group2_entries() -> [BindGroupLayoutEntry; 7] {
    [
        uniform_entry(0, FRAME_SIZE),
        depth_entry(1),
        float_entry(2, false),
        float_entry(3, true),
        float_entry(4, false),
        storage_texture_entry(5, TextureFormat::Rgba16Float),
        uniform_entry(7, SCREEN_PARAMS_SIZE),
    ]
}

/// Project group 2: 0 `GiFrameGpu` (432 B), 1 depth, 2 normal, 3 the unfilterable ray atlas and
/// 5 the probes written as a read_write storage buffer.
pub fn project_group2_entries() -> [BindGroupLayoutEntry; 5] {
    [
        uniform_entry(0, FRAME_SIZE),
        depth_entry(1),
        float_entry(2, false),
        float_entry(3, false),
        storage_entry(5, false),
    ]
}

/// Filter group 2: 0 `GiFrameGpu` (432 B), 4 the probes read and 5 the probes read_write.
pub fn filter_group2_entries() -> [BindGroupLayoutEntry; 3] {
    [
        uniform_entry(0, FRAME_SIZE),
        storage_entry(4, true),
        storage_entry(5, false),
    ]
}

/// Integrate group 2: 0 `GiFrameGpu` (432 B), 1 depth, 2 normal, 3 the S12 motion, 4 the other
/// parity output, 5 the probes read, 6 the Rgba16Float output and 7 `GiScreenGpu` (16 B).
pub fn integrate_group2_entries() -> [BindGroupLayoutEntry; 8] {
    [
        uniform_entry(0, FRAME_SIZE),
        depth_entry(1),
        float_entry(2, false),
        float_entry(3, false),
        float_entry(4, false),
        storage_entry(5, true),
        storage_texture_entry(6, TextureFormat::Rgba16Float),
        uniform_entry(7, SCREEN_PARAMS_SIZE),
    ]
}

/// `ScreenPipelines` holds the screen-probe compute pipelines and their private group-2
/// layouts: the copy and reduce pipelines build the depth pyramid mip chain, trace and project
/// hold one pipeline per `PROBE_RAY_OPTIONS` entry, and filter and integrate are single
/// pipelines. Every pipeline layout is [empty group 0, the GI compute group 1, its group-2
/// layout].
///
/// Handoff: card 'raa-gi screen: ScreenResources in screen.rs (textures, buffers, bind groups)'
/// takes over `ScreenPipelines`: it calls `ScreenPipelines::new`, which builds the nine
/// pipelines over the six group-2 layouts today, and reads the `ScreenPipelines.copy_layout`,
/// `ScreenPipelines.reduce_layout`, `ScreenPipelines.trace_layout`,
/// `ScreenPipelines.project_layout`, `ScreenPipelines.filter_layout` and
/// `ScreenPipelines.integrate_layout` fields to build the per-mip and per-entry group-2 bind
/// groups that `ScreenPipelines::record` binds in one `raa-gi.screen` pass. The part-root
/// Handoff it extends stays crates/raa-gi/src/parts/screen.rs:15 `install`.
pub(crate) struct ScreenPipelines {
    pub(crate) copy_layout: wgpu::BindGroupLayout,
    pub(crate) reduce_layout: wgpu::BindGroupLayout,
    pub(crate) trace_layout: wgpu::BindGroupLayout,
    pub(crate) project_layout: wgpu::BindGroupLayout,
    pub(crate) filter_layout: wgpu::BindGroupLayout,
    pub(crate) integrate_layout: wgpu::BindGroupLayout,
    copy: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    trace: [wgpu::ComputePipeline; 2],
    project: [wgpu::ComputePipeline; 2],
    filter: wgpu::ComputePipeline,
    integrate: wgpu::ComputePipeline,
}

/// The per-frame bind groups and sizes `ScreenPipelines::record` dispatches over: one pyramid
/// group per mip with its size, the trace, project, filter and integrate groups, the probe-ray
/// option, the probe grid and the viewport.
pub(crate) struct ScreenWork<'a> {
    pub(crate) pyramid: &'a [wgpu::BindGroup],
    pub(crate) pyramid_sizes: &'a [(u32, u32)],
    pub(crate) trace: &'a wgpu::BindGroup,
    pub(crate) project: &'a wgpu::BindGroup,
    pub(crate) filter: &'a wgpu::BindGroup,
    pub(crate) integrate: &'a wgpu::BindGroup,
    pub(crate) probe_rays: u32,
    pub(crate) grid: (u32, u32),
    pub(crate) viewport: (u32, u32),
}

fn layout(
    device: &wgpu::Device,
    label: &str,
    entries: &[BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    })
}

fn module(device: &wgpu::Device, label: &str, source: &str) -> ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

/// A compute pipeline of `module`'s `entry` over the bind group layouts `groups`; when
/// `probe_rays` is given, `PROBE_RAYS` is set to it through `compilation_options.constants`.
fn compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &ShaderModule,
    entry: &str,
    groups: &[Option<&wgpu::BindGroupLayout>],
    probe_rays: Option<u32>,
) -> ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: groups,
        immediate_size: 0,
    });
    let constants = [("PROBE_RAYS", probe_rays.map_or(0.0, f64::from))];
    let compilation_options = match probe_rays {
        Some(_) => wgpu::PipelineCompilationOptions {
            constants: &constants,
            ..Default::default()
        },
        None => Default::default(),
    };
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some(entry),
        compilation_options,
        cache: None,
    })
}

fn groups_for(n: u32) -> u32 {
    n.div_ceil(WORKGROUP)
}

fn dispatch(pass: &mut wgpu::ComputePass<'_>, width: u32, height: u32) {
    pass.dispatch_workgroups(groups_for(width), groups_for(height), 1);
}

impl ScreenPipelines {
    /// Builds one shader module per builder and the nine pipelines over the layout [empty group
    /// 0, `group1`, their group-2 layout]; trace and project are built once per
    /// `PROBE_RAY_OPTIONS` entry with `PROBE_RAYS` set to that value.
    pub(crate) fn new(gpu: &GpuContext, group1: &wgpu::BindGroupLayout) -> ScreenPipelines {
        let device = gpu.device();
        let empty = layout(device, "raa-gi.screen.empty", &[]);
        let copy_layout = layout(device, "raa-gi.screen.copy", &pyramid_copy_group2_entries());
        let reduce_layout = layout(
            device,
            "raa-gi.screen.reduce",
            &pyramid_reduce_group2_entries(),
        );
        let trace_layout = layout(device, "raa-gi.screen.trace", &trace_group2_entries());
        let project_layout = layout(device, "raa-gi.screen.project", &project_group2_entries());
        let filter_layout = layout(device, "raa-gi.screen.filter", &filter_group2_entries());
        let integrate_layout = layout(
            device,
            "raa-gi.screen.integrate",
            &integrate_group2_entries(),
        );

        let pyramid_module = module(
            device,
            "raa-gi.screen.pyramid",
            &depth_pyramid_module_wgsl(),
        );
        let trace_module = module(device, "raa-gi.screen.trace", &probe_trace_module_wgsl());
        let project_module = module(
            device,
            "raa-gi.screen.project",
            &probe_project_module_wgsl(),
        );
        let integrate_module = module(
            device,
            "raa-gi.screen.integrate",
            &probe_integrate_module_wgsl(),
        );

        let copy = compute_pipeline(
            device,
            "raa-gi.screen.copy",
            &pyramid_module,
            PYRAMID_COPY_ENTRY,
            &[Some(&empty), Some(group1), Some(&copy_layout)],
            None,
        );
        let reduce = compute_pipeline(
            device,
            "raa-gi.screen.reduce",
            &pyramid_module,
            PYRAMID_REDUCE_ENTRY,
            &[Some(&empty), Some(group1), Some(&reduce_layout)],
            None,
        );
        let trace = PROBE_RAY_OPTIONS.map(|rays| {
            compute_pipeline(
                device,
                "raa-gi.screen.trace",
                &trace_module,
                PROBE_TRACE_ENTRY,
                &[Some(&empty), Some(group1), Some(&trace_layout)],
                Some(rays),
            )
        });
        let project = PROBE_RAY_OPTIONS.map(|rays| {
            compute_pipeline(
                device,
                "raa-gi.screen.project",
                &project_module,
                PROBE_PROJECT_ENTRY,
                &[Some(&empty), Some(group1), Some(&project_layout)],
                Some(rays),
            )
        });
        let filter = compute_pipeline(
            device,
            "raa-gi.screen.filter",
            &project_module,
            PROBE_FILTER_ENTRY,
            &[Some(&empty), Some(group1), Some(&filter_layout)],
            None,
        );
        let integrate = compute_pipeline(
            device,
            "raa-gi.screen.integrate",
            &integrate_module,
            PROBE_INTEGRATE_ENTRY,
            &[Some(&empty), Some(group1), Some(&integrate_layout)],
            None,
        );

        ScreenPipelines {
            copy_layout,
            reduce_layout,
            trace_layout,
            project_layout,
            filter_layout,
            integrate_layout,
            copy,
            reduce,
            trace,
            project,
            filter,
            integrate,
        }
    }

    /// Records one `raa-gi.screen` compute pass: the pyramid copy of `work.pyramid[0]` over mip
    /// 0, each later mip's reduce, then trace, project, filter and integrate with the counts of
    /// contract 'gi-screen-wgsl'. Group 1 is set once and group 0 is never set. Returns false
    /// and records nothing when `work.probe_rays` is not in `PROBE_RAY_OPTIONS` or the pyramid
    /// is empty; otherwise it returns true.
    pub(crate) fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        group1: &wgpu::BindGroup,
        work: &ScreenWork<'_>,
    ) -> bool {
        let Some(variant) = PROBE_RAY_OPTIONS
            .iter()
            .position(|&rays| rays == work.probe_rays)
        else {
            return false;
        };
        let Some((first, rest)) = work.pyramid.split_first() else {
            return false;
        };
        let (width, height) = work.pyramid_sizes.first().copied().unwrap_or(work.viewport);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raa-gi.screen"),
            timestamp_writes: None,
        });
        pass.set_bind_group(1, group1, &[]);

        pass.set_pipeline(&self.copy);
        pass.set_bind_group(2, first, &[]);
        dispatch(&mut pass, width, height);

        for (index, bind) in rest.iter().enumerate() {
            let Some(&(width, height)) = work.pyramid_sizes.get(index + 1) else {
                break;
            };
            pass.set_pipeline(&self.reduce);
            pass.set_bind_group(2, bind, &[]);
            dispatch(&mut pass, width, height);
        }

        pass.set_pipeline(&self.trace[variant]);
        pass.set_bind_group(2, work.trace, &[]);
        pass.dispatch_workgroups(
            groups_for(work.grid.0 * work.probe_rays),
            groups_for(work.grid.1 * work.probe_rays),
            1,
        );

        pass.set_pipeline(&self.project[variant]);
        pass.set_bind_group(2, work.project, &[]);
        dispatch(&mut pass, work.grid.0, work.grid.1);

        pass.set_pipeline(&self.filter);
        pass.set_bind_group(2, work.filter, &[]);
        dispatch(&mut pass, work.grid.0, work.grid.1);

        pass.set_pipeline(&self.integrate);
        pass.set_bind_group(2, work.integrate, &[]);
        dispatch(&mut pass, work.viewport.0, work.viewport.1);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyramid_sizes_reduce_to_one() {
        let sizes = pyramid_sizes((1280, 720));
        assert_eq!(sizes.len(), 11);
        assert_eq!(sizes[5], (40, 22));
        assert_eq!(sizes[9], (2, 1));
        assert_eq!(sizes[10], (1, 1));
        assert_eq!(pyramid_sizes((1, 1)), [(1, 1)]);
    }

    #[test]
    fn builders_hold_their_entry_names() {
        let pyramid = depth_pyramid_module_wgsl();
        assert!(pyramid.contains(PYRAMID_COPY_ENTRY));
        assert!(pyramid.contains(PYRAMID_REDUCE_ENTRY));
        let trace = probe_trace_module_wgsl();
        assert!(trace.contains(PROBE_TRACE_ENTRY));
        assert!(trace.contains("raa_gi_probe_ray_direction"));
        let project = probe_project_module_wgsl();
        assert!(project.contains(PROBE_PROJECT_ENTRY));
        assert!(project.contains(PROBE_FILTER_ENTRY));
        let integrate = probe_integrate_module_wgsl();
        assert!(integrate.contains(PROBE_INTEGRATE_ENTRY));
    }
}
