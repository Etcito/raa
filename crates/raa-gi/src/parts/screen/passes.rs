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

use raa_core::{GpuContext, Lazy};
use raa_shader::{CLOUD_SHADOW_WGSL, DEPTH_WGSL, GI_VOLUME_WGSL, PRELUDE_WGSL};
use std::num::NonZeroU64;

use crate::emissive::{SCREEN_OVERRIDE, override_value};
use crate::tunables::GiTunables;

use wgpu::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, ComputePipeline, ShaderModule,
    ShaderStages, StorageTextureAccess, TextureFormat, TextureSampleType, TextureViewDimension,
};

const DEPTH_PYRAMID_WGSL: &str = include_str!("../../wgsl/screen/depth_pyramid.wgsl");
const PROBE_COMMON_WGSL: &str = include_str!("../../wgsl/screen/probe_common.wgsl");
const SCREEN_TRACE_WGSL: &str = include_str!("../../wgsl/screen/screen_trace.wgsl");
const SCREEN_EMISSION_WGSL: &str = include_str!("../../wgsl/screen/screen_emission.wgsl");
const PROBE_TRACE_WGSL: &str = include_str!("../../wgsl/screen/probe_trace.wgsl");
const PROBE_PROJECT_WGSL: &str = include_str!("../../wgsl/screen/probe_project.wgsl");
const PROBE_INTEGRATE_WGSL: &str = include_str!("../../wgsl/screen/probe_integrate.wgsl");
const ACCUMULATE_WGSL: &str = include_str!("../../wgsl/screen/accumulate.wgsl");
const ATROUS_WGSL: &str = include_str!("../../wgsl/screen/atrous.wgsl");

/// The (override, tunable key) pairs of the trace module: probe_trace.wgsl and screen_trace.wgsl.
const TRACE_OVERRIDES: &[(&str, &str)] = &[
    ("RAA_GI_PROBE_SCREEN_DISTANCE", "probeScreenDistance"),
    ("RAA_GI_PROBE_SCREEN_STEPS", "probeScreenSteps"),
    ("RAA_GI_PROBE_SCREEN_THICKNESS", "probeScreenThickness"),
    ("RAA_GI_PROBE_VOLUME_DISTANCE", "probeVolumeDistance"),
    ("RAA_GI_PROBE_VOLUME_STEPS", "probeVolumeSteps"),
    ("RAA_GI_PROBE_VOLUME_HIT_VOXELS", "probeVolumeHitVoxels"),
    ("RAA_GI_SCREEN_EDGE_FADE", "screenEdgeFade"),
];
/// The (override, tunable key) pairs of the integrate module: accumulate.wgsl.
const INTEGRATE_OVERRIDES: &[(&str, &str)] = &[
    ("RAA_GI_ACC_DEPTH_TOLERANCE", "accDepthTolerance"),
    ("RAA_GI_ACC_NORMAL_COS", "accNormalCos"),
    ("RAA_GI_ACC_MIN_WEIGHT", "accMinWeight"),
    ("RAA_GI_ACC_CHANGE_SLACK", "accChangeSlack"),
    ("RAA_GI_ACC_CHANGE_MIN", "accChangeMin"),
    ("RAA_GI_OCCLUSION_DIRECTIONS", "occlusionDirections"),
    ("RAA_GI_OCCLUSION_STEPS", "occlusionSteps"),
    ("RAA_GI_OCCLUSION_RADIUS", "occlusionRadius"),
];
/// The (override, tunable key) pairs of the a-trous module: atrous.wgsl.
const ATROUS_OVERRIDES: &[(&str, &str)] = &[
    ("RAA_GI_ATROUS_SIGMA_LUMINANCE", "atrousSigmaLuminance"),
    ("RAA_GI_ATROUS_NORMAL_POWER", "atrousNormalPower"),
    ("RAA_GI_ATROUS_PLANE", "atrousPlane"),
];

/// `GiFrameGpu` bytes (S38): five matrices, then the seven 16-B lanes.
const FRAME_SIZE: u64 = 432;

/// The rays a probe traces on a side. Each option has its own trace and project pipeline, since
/// `PROBE_RAYS` is an override constant that the WGSL loops are unrolled over.
pub const PROBE_RAY_OPTIONS: [u32; 2] = [6, 8];

/// Size of `Probe` in `probe_common.wgsl`: nine SH vectors, position and normal.
pub const PROBE_BYTES: u64 = 176;

/// Size of `GiScreenGpu`: `history_valid`, `screen_traces`, `parity`, `debug_view`,
/// `history_max`, `clamp_gamma`, `m_cap` and one pad lane.
pub const SCREEN_PARAMS_SIZE: u64 = 32;

/// The most a-trous passes (`spatialPasses`), steps 1, 2, 4, 8 and 16 pixels.
pub const ATROUS_MAX_PASSES: u32 = 5;

/// Bytes between two a-trous pass records in their uniform buffer (the uniform offset
/// alignment).
pub const ATROUS_SLOT_BYTES: u64 = 256;

/// Size of one `RaaGiAtrousGpu` record: step, last flag and two pad lanes.
pub const ATROUS_PARAMS_SIZE: u64 = 16;

/// Entry of the a-trous pipeline.
pub const ATROUS_ENTRY: &str = "raa_gi_atrous_pass";

/// The 16 B of the a-trous record of pass `index` (step 2^index), `last` when it writes the
/// screen output.
pub fn atrous_record(index: u32, last: bool) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&(1i32 << index.min(ATROUS_MAX_PASSES - 1)).to_le_bytes());
    out[4..8].copy_from_slice(&u32::from(last).to_le_bytes());
    out
}

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
         {PROBE_COMMON_WGSL}{SCREEN_TRACE_WGSL}{SCREEN_EMISSION_WGSL}{PROBE_TRACE_WGSL}"
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
/// accumulate.wgsl (the shared temporal accumulation) + probe_integrate.wgsl.
pub fn probe_integrate_module_wgsl() -> String {
    format!(
        "{PRELUDE_WGSL}{CLOUD_SHADOW_WGSL}{GI_VOLUME_WGSL}{DEPTH_WGSL}{PROBE_COMMON_WGSL}\
         {ACCUMULATE_WGSL}{PROBE_INTEGRATE_WGSL}"
    )
}

/// The temporal-accumulation text the ReSTIR GI denoise module composes after probe_common.
pub(crate) fn accumulate_wgsl() -> &'static str {
    ACCUMULATE_WGSL
}

/// `atrous_module_wgsl` builds the a-trous module: `PRELUDE_WGSL` + `CLOUD_SHADOW_WGSL` +
/// `GI_VOLUME_WGSL` + `DEPTH_WGSL` + probe_common.wgsl + atrous.wgsl.
pub fn atrous_module_wgsl() -> String {
    format!(
        "{PRELUDE_WGSL}{CLOUD_SHADOW_WGSL}{GI_VOLUME_WGSL}{DEPTH_WGSL}{PROBE_COMMON_WGSL}{ATROUS_WGSL}"
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

pub(crate) fn entry(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

pub(crate) fn texture_entry(binding: u32, sample_type: TextureSampleType) -> BindGroupLayoutEntry {
    entry(
        binding,
        BindingType::Texture {
            sample_type,
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
    )
}

pub(crate) fn depth_entry(binding: u32) -> BindGroupLayoutEntry {
    texture_entry(binding, TextureSampleType::Depth)
}

pub(crate) fn float_entry(binding: u32, filterable: bool) -> BindGroupLayoutEntry {
    texture_entry(binding, TextureSampleType::Float { filterable })
}

pub(crate) fn uint_entry(binding: u32) -> BindGroupLayoutEntry {
    texture_entry(binding, TextureSampleType::Uint)
}

pub(crate) fn storage_texture_entry(binding: u32, format: TextureFormat) -> BindGroupLayoutEntry {
    entry(
        binding,
        BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format,
            view_dimension: TextureViewDimension::D2,
        },
    )
}

pub(crate) fn uniform_entry(binding: u32, size: u64) -> BindGroupLayoutEntry {
    entry(
        binding,
        BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(size),
        },
    )
}

pub(crate) fn storage_entry(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
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
/// history, 4 the unfilterable pyramid, 5 the Rgba16Float ray atlas, 6 the GI voxel words (read)
/// and 7 `GiScreenGpu` (16 B).
pub fn trace_group2_entries() -> [BindGroupLayoutEntry; 8] {
    [
        uniform_entry(0, FRAME_SIZE),
        depth_entry(1),
        float_entry(2, false),
        float_entry(3, true),
        float_entry(4, false),
        storage_texture_entry(5, TextureFormat::Rgba16Float),
        storage_entry(6, true),
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

/// The accumulation bindings shared by the integrate and the ReSTIR GI denoise (accumulate.wgsl):
/// 3 the S12 motion, 4 the other parity history, 6 the Rgba16Float history, 7 `GiScreenGpu`,
/// 8 the other parity moments, 9 the Rgba32Uint moments, 10 the Rgba16Float a-trous input,
/// 11 the Rgba32Uint debug record and 12 the statistics counters.
pub fn accumulate_group2_entries() -> [BindGroupLayoutEntry; 9] {
    [
        float_entry(3, false),
        float_entry(4, false),
        storage_texture_entry(6, TextureFormat::Rgba16Float),
        uniform_entry(7, SCREEN_PARAMS_SIZE),
        uint_entry(8),
        storage_texture_entry(9, TextureFormat::Rgba32Uint),
        storage_texture_entry(10, TextureFormat::Rgba16Float),
        storage_texture_entry(11, TextureFormat::Rgba32Uint),
        storage_entry(12, false),
    ]
}

/// Integrate group 2: 0 `GiFrameGpu` (432 B), 1 depth, 2 normal, 5 the probes read and the
/// accumulation bindings of [`accumulate_group2_entries`].
pub fn integrate_group2_entries() -> [BindGroupLayoutEntry; 13] {
    let shared = accumulate_group2_entries();
    [
        uniform_entry(0, FRAME_SIZE),
        depth_entry(1),
        float_entry(2, false),
        shared[0],
        shared[1],
        storage_entry(5, true),
        shared[2],
        shared[3],
        shared[4],
        shared[5],
        shared[6],
        shared[7],
        shared[8],
    ]
}

/// A-trous group 2: 0 `GiFrameGpu`, 1 depth, 2 normal, 15 the input (rgb, variance), 16 the
/// Rgba16Float output, 17 the history (occlusion) and 18 the 16-B pass record.
pub fn atrous_group2_entries() -> [BindGroupLayoutEntry; 7] {
    [
        uniform_entry(0, FRAME_SIZE),
        depth_entry(1),
        float_entry(2, false),
        float_entry(15, false),
        storage_texture_entry(16, TextureFormat::Rgba16Float),
        float_entry(17, false),
        uniform_entry(18, ATROUS_PARAMS_SIZE),
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
    pub(crate) atrous_layout: wgpu::BindGroupLayout,
    /// The pyramid and a-trous pipelines build on their first record too: a graph without a screen GI pass never
    /// records them.
    copy: Lazy<wgpu::ComputePipeline>,
    reduce: Lazy<wgpu::ComputePipeline>,
    trace: [Lazy<wgpu::ComputePipeline>; 2],
    project: [Lazy<wgpu::ComputePipeline>; 2],
    filter: Lazy<wgpu::ComputePipeline>,
    integrate: Lazy<wgpu::ComputePipeline>,
    atrous: Lazy<wgpu::ComputePipeline>,
    /// What the trace, integrate and a-trous pipelines are rebuilt from when a tunable changes.
    tuned: TunedSources,
    /// The tunables generation the trace, integrate and a-trous pipelines were built with.
    tunables_generation: u64,
    /// The emissive generation the trace pipelines were built with.
    emissive_generation: u64,
}

/// The layouts and modules of the pipelines whose modules declare gi.screen overrides.
struct TunedSources {
    empty: wgpu::BindGroupLayout,
    group1: wgpu::BindGroupLayout,
    trace_module: ShaderModule,
    integrate_module: ShaderModule,
    atrous_module: ShaderModule,
}

/// The trace (one per `PROBE_RAY_OPTIONS` entry), integrate and a-trous pipelines with the
/// overrides of their modules set from `tunables` (contract 'gi-tunables-store').
fn tuned_pipelines(
    device: &wgpu::Device,
    sources: &TunedSources,
    trace_layout: &wgpu::BindGroupLayout,
    integrate_layout: &wgpu::BindGroupLayout,
    atrous_layout: &wgpu::BindGroupLayout,
    tunables: &GiTunables,
    screen_emissive: bool,
) -> (
    [Lazy<ComputePipeline>; 2],
    Lazy<ComputePipeline>,
    Lazy<ComputePipeline>,
) {
    let (empty, group1) = (&sources.empty, &sources.group1);
    let mut trace_constants = tunables.constants(TRACE_OVERRIDES);
    trace_constants.push((SCREEN_OVERRIDE, override_value(screen_emissive)));
    // Trace and integrate are recorded only by the probe mode (`record`); ReSTIR GI and SSGI
    // record the pyramid and the a-trous passes only, so these build on the first probe frame.
    let trace = PROBE_RAY_OPTIONS.map(|rays| {
        lazy_compute_pipeline(
            device,
            "raa-gi.screen.trace",
            &sources.trace_module,
            PROBE_TRACE_ENTRY,
            [empty, group1, trace_layout],
            Some(rays),
            trace_constants.clone(),
        )
    });
    let integrate = lazy_compute_pipeline(
        device,
        "raa-gi.screen.integrate",
        &sources.integrate_module,
        PROBE_INTEGRATE_ENTRY,
        [empty, group1, integrate_layout],
        None,
        tunables.constants(INTEGRATE_OVERRIDES),
    );
    let atrous = lazy_compute_pipeline(
        device,
        "raa-gi.screen.atrous",
        &sources.atrous_module,
        ATROUS_ENTRY,
        [empty, group1, atrous_layout],
        None,
        tunables.constants(ATROUS_OVERRIDES),
    );
    (trace, integrate, atrous)
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
    /// The a-trous groups of this parity: index `2 i + 1` for pass i writing the screen output,
    /// `2 i` for pass i writing the other filter texture.
    pub(crate) atrous: &'a [wgpu::BindGroup],
    /// The a-trous passes to record (`spatialPasses`).
    pub(crate) atrous_passes: u32,
    pub(crate) probe_rays: u32,
    pub(crate) grid: (u32, u32),
    pub(crate) viewport: (u32, u32),
}

pub(crate) fn layout(
    device: &wgpu::Device,
    label: &str,
    entries: &[BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    })
}

pub(crate) fn module(device: &wgpu::Device, label: &str, source: &str) -> ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

/// A compute pipeline of `module`'s `entry` over the bind group layouts `groups`; when
/// `probe_rays` is given, `PROBE_RAYS` is set to it through `compilation_options.constants`.
pub(crate) fn compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &ShaderModule,
    entry: &str,
    groups: &[Option<&wgpu::BindGroupLayout>],
    probe_rays: Option<u32>,
) -> ComputePipeline {
    tuned_compute_pipeline(device, label, module, entry, groups, probe_rays, &[])
}

/// [`compute_pipeline`] with the override constants `tuned` passed before `PROBE_RAYS`.
fn tuned_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &ShaderModule,
    entry: &str,
    groups: &[Option<&wgpu::BindGroupLayout>],
    probe_rays: Option<u32>,
    tuned: &[(&'static str, f64)],
) -> ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: groups,
        immediate_size: 0,
    });
    let mut constants = tuned.to_vec();
    if let Some(rays) = probe_rays {
        constants.push(("PROBE_RAYS", f64::from(rays)));
    }
    let compilation_options = wgpu::PipelineCompilationOptions {
        constants: &constants,
        ..Default::default()
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

/// [`tuned_compute_pipeline`] over the layout `groups`, built on its first use ([`Lazy`]).
fn lazy_compute_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    module: &ShaderModule,
    entry: &'static str,
    groups: [&wgpu::BindGroupLayout; 3],
    probe_rays: Option<u32>,
    tuned: Vec<(&'static str, f64)>,
) -> Lazy<ComputePipeline> {
    let (device, module) = (device.clone(), module.clone());
    let groups = groups.map(Clone::clone);
    Lazy::new(move || {
        let [a, b, c] = &groups;
        tuned_compute_pipeline(
            &device,
            label,
            &module,
            entry,
            &[Some(a), Some(b), Some(c)],
            probe_rays,
            &tuned,
        )
    })
}

pub(crate) fn groups_for(n: u32) -> u32 {
    n.div_ceil(WORKGROUP)
}

pub(crate) fn dispatch(pass: &mut wgpu::ComputePass<'_>, width: u32, height: u32) {
    pass.dispatch_workgroups(groups_for(width), groups_for(height), 1);
}

impl ScreenPipelines {
    /// Builds one shader module per builder and the nine pipelines over the layout [empty group
    /// 0, `group1`, their group-2 layout]; trace and project are built once per
    /// `PROBE_RAY_OPTIONS` entry with `PROBE_RAYS` set to that value.
    /// The trace, integrate and a-trous pipelines take the gi.screen overrides from `tunables`;
    /// the trace pipelines also take `RAA_GI_SCREEN_EMISSIVE` from `emissive` (the screen flag
    /// and the emissive generation of `GiShared`).
    pub(crate) fn new(
        gpu: &GpuContext,
        group1: &wgpu::BindGroupLayout,
        tunables: &GiTunables,
        emissive: (bool, u64),
    ) -> ScreenPipelines {
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
        let atrous_layout = layout(device, "raa-gi.screen.atrous", &atrous_group2_entries());

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
        let atrous_module = module(device, "raa-gi.screen.atrous", &atrous_module_wgsl());

        let copy = lazy_compute_pipeline(
            device,
            "raa-gi.screen.copy",
            &pyramid_module,
            PYRAMID_COPY_ENTRY,
            [&empty, group1, &copy_layout],
            None,
            Vec::new(),
        );
        let reduce = lazy_compute_pipeline(
            device,
            "raa-gi.screen.reduce",
            &pyramid_module,
            PYRAMID_REDUCE_ENTRY,
            [&empty, group1, &reduce_layout],
            None,
            Vec::new(),
        );
        let project = PROBE_RAY_OPTIONS.map(|rays| {
            lazy_compute_pipeline(
                device,
                "raa-gi.screen.project",
                &project_module,
                PROBE_PROJECT_ENTRY,
                [&empty, group1, &project_layout],
                Some(rays),
                Vec::new(),
            )
        });
        let filter = lazy_compute_pipeline(
            device,
            "raa-gi.screen.filter",
            &project_module,
            PROBE_FILTER_ENTRY,
            [&empty, group1, &filter_layout],
            None,
            Vec::new(),
        );
        let tuned = TunedSources {
            empty,
            group1: group1.clone(),
            trace_module,
            integrate_module,
            atrous_module,
        };
        let (trace, integrate, atrous) = tuned_pipelines(
            device,
            &tuned,
            &trace_layout,
            &integrate_layout,
            &atrous_layout,
            tunables,
            emissive.0,
        );

        ScreenPipelines {
            copy_layout,
            reduce_layout,
            trace_layout,
            project_layout,
            filter_layout,
            integrate_layout,
            atrous_layout,
            copy,
            reduce,
            trace,
            project,
            filter,
            integrate,
            atrous,
            tuned,
            tunables_generation: tunables.generation(),
            emissive_generation: emissive.1,
        }
    }

    /// The tunables generation the trace, integrate and a-trous pipelines were built with.
    pub(crate) fn tunables_generation(&self) -> u64 {
        self.tunables_generation
    }

    /// The emissive generation the trace pipelines were built with.
    pub(crate) fn emissive_generation(&self) -> u64 {
        self.emissive_generation
    }

    /// Rebuilds the trace, integrate and a-trous pipelines with the overrides of `tunables`. The
    /// bind group layouts are kept, so the bind groups built over them stay valid.
    pub(crate) fn rebuild_tuned(
        &mut self,
        gpu: &GpuContext,
        tunables: &GiTunables,
        emissive: (bool, u64),
    ) {
        let (trace, integrate, atrous) = tuned_pipelines(
            gpu.device(),
            &self.tuned,
            &self.trace_layout,
            &self.integrate_layout,
            &self.atrous_layout,
            tunables,
            emissive.0,
        );
        self.trace = trace;
        self.integrate = integrate;
        self.atrous = atrous;
        self.tunables_generation = tunables.generation();
        self.emissive_generation = emissive.1;
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
        if work.pyramid.is_empty() {
            return false;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raa-gi.screen"),
            timestamp_writes: None,
        });
        pass.set_bind_group(1, group1, &[]);
        self.record_pyramid(&mut pass, work);

        pass.set_pipeline(self.trace[variant].get());
        pass.set_bind_group(2, work.trace, &[]);
        pass.dispatch_workgroups(
            groups_for(work.grid.0 * work.probe_rays),
            groups_for(work.grid.1 * work.probe_rays),
            1,
        );

        pass.set_pipeline(self.project[variant].get());
        pass.set_bind_group(2, work.project, &[]);
        dispatch(&mut pass, work.grid.0, work.grid.1);

        pass.set_pipeline(self.filter.get());
        pass.set_bind_group(2, work.filter, &[]);
        dispatch(&mut pass, work.grid.0, work.grid.1);

        pass.set_pipeline(self.integrate.get());
        pass.set_bind_group(2, work.integrate, &[]);
        dispatch(&mut pass, work.viewport.0, work.viewport.1);
        self.record_atrous(&mut pass, work);
        true
    }

    /// Records the `work.atrous_passes` a-trous passes (group 1 already set): pass i reads one
    /// filter texture and writes the other, the last one writes the screen output.
    pub(crate) fn record_atrous(&self, pass: &mut wgpu::ComputePass<'_>, work: &ScreenWork<'_>) {
        let passes = work.atrous_passes.min(ATROUS_MAX_PASSES);
        for index in 0..passes {
            let last = index + 1 == passes;
            let Some(group) = work.atrous.get((index * 2 + u32::from(last)) as usize) else {
                return;
            };
            pass.set_pipeline(self.atrous.get());
            pass.set_bind_group(2, group, &[]);
            dispatch(pass, work.viewport.0, work.viewport.1);
        }
    }

    /// Records the depth-pyramid copy of mip 0 and the reduce of every later mip into `pass`
    /// (group 1 already set); false when the pyramid is empty. The probe record and the ReSTIR GI
    /// record both start with it.
    pub(crate) fn record_pyramid(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        work: &ScreenWork<'_>,
    ) -> bool {
        let Some((first, rest)) = work.pyramid.split_first() else {
            return false;
        };
        let (width, height) = work.pyramid_sizes.first().copied().unwrap_or(work.viewport);
        pass.set_pipeline(self.copy.get());
        pass.set_bind_group(2, first, &[]);
        dispatch(pass, width, height);
        for (index, bind) in rest.iter().enumerate() {
            let Some(&(width, height)) = work.pyramid_sizes.get(index + 1) else {
                break;
            };
            pass.set_pipeline(self.reduce.get());
            pass.set_bind_group(2, bind, &[]);
            dispatch(pass, width, height);
        }
        true
    }

    /// Records the pyramid copy over `groups[0]` at `sizes[0]` and the reduce over each later
    /// group at its size into `pass` (group 1 already set); false and nothing recorded when
    /// `groups` or `sizes` is empty. The `gi.pyramid` stage records with it.
    pub(crate) fn record_pyramid_groups(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        groups: &[wgpu::BindGroup],
        sizes: &[(u32, u32)],
    ) -> bool {
        let (Some(first), Some(&(width, height))) = (groups.first(), sizes.first()) else {
            return false;
        };
        pass.set_pipeline(self.copy.get());
        pass.set_bind_group(2, first, &[]);
        dispatch(pass, width, height);
        for (bind, &(width, height)) in groups.iter().zip(sizes).skip(1) {
            pass.set_pipeline(self.reduce.get());
            pass.set_bind_group(2, bind, &[]);
            dispatch(pass, width, height);
        }
        true
    }

    /// Records the trace, project and filter dispatches of `record` into `pass` (group 1 already
    /// set); false and nothing recorded when `probe_rays` is not in `PROBE_RAY_OPTIONS`. The
    /// `gi.probes` stage records with it.
    pub(crate) fn record_probe_chain(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        trace: &wgpu::BindGroup,
        project: &wgpu::BindGroup,
        filter: &wgpu::BindGroup,
        probe_rays: u32,
        grid: (u32, u32),
    ) -> bool {
        let Some(variant) = PROBE_RAY_OPTIONS
            .iter()
            .position(|&rays| rays == probe_rays)
        else {
            return false;
        };
        pass.set_pipeline(self.trace[variant].get());
        pass.set_bind_group(2, trace, &[]);
        pass.dispatch_workgroups(
            groups_for(grid.0 * probe_rays),
            groups_for(grid.1 * probe_rays),
            1,
        );

        pass.set_pipeline(self.project[variant].get());
        pass.set_bind_group(2, project, &[]);
        dispatch(pass, grid.0, grid.1);

        pass.set_pipeline(self.filter.get());
        pass.set_bind_group(2, filter, &[]);
        dispatch(pass, grid.0, grid.1);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_core::testing::skip;

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
        assert!(atrous_module_wgsl().contains(ATROUS_ENTRY));
    }

    #[test]
    fn modules_validate() {
        for (label, source) in [
            ("integrate", probe_integrate_module_wgsl()),
            ("atrous", atrous_module_wgsl()),
            ("trace", probe_trace_module_wgsl()),
            ("project", probe_project_module_wgsl()),
        ] {
            if let Err(e) = raa_shader::Composer::validate(&source) {
                panic!("{label}: {}: {}", e.code, e.message);
            }
        }
    }

    #[test]
    fn atrous_records_double_the_step() {
        // Pass i spreads its taps 2^i pixels: 1, 2, 4, 8, 16.
        for (index, step) in [(0u32, 1i32), (1, 2), (2, 4), (3, 8), (4, 16)] {
            let bytes = atrous_record(index, index == 4);
            assert_eq!(i32::from_le_bytes(bytes[0..4].try_into().unwrap()), step);
            assert_eq!(
                u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
                u32::from(index == 4)
            );
        }
    }

    #[test]
    fn probe_chain_and_pyramid_groups_refuse_before_recording() {
        let test = "probe_chain_and_pyramid_groups_refuse_before_recording";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let device = gpu.device();
        let group1 = crate::parts::GiShared::new(&gpu).compute_group1_layout();
        let pipelines = ScreenPipelines::new(&gpu, &group1, &GiTunables::new(), (false, 0));
        // An empty layout and group: both methods refuse before they bind anything.
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[],
        });
        let empty = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[],
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            let chain = pipelines.record_probe_chain(&mut pass, &empty, &empty, &empty, 7, (1, 1));
            assert!(!chain, "probe_rays 7 is not an option");
            assert!(!pipelines.record_pyramid_groups(&mut pass, &[], &[]));
        }
        drop(encoder.finish());
    }
}
