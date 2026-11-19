//! ReSTIR GI for the indirect diffuse of the screen part (GI mode `restir`, `IndirectMode`),
//! after Ouyang et al., "ReSTIR GI: Path Resampling for Real-Time Path Tracing", HPG 2021: the
//! three composed WGSL modules, their group-2 layouts and pipelines, the reservoir resources and
//! the record. The gi.screen pass builds the depth pyramid, then runs stage 1 (one secondary ray
//! per pixel + temporal reuse of last frame's final reservoirs, whose light and visibility are
//! re-read where their points are today), stage 2 (spatial reuse with the reconnection Jacobian
//! and a voxel visibility test, resolved to irradiance, its merged reservoir written back as next
//! frame's temporal input) and stage 3 (bilateral + the screen part's temporal accumulation)
//! into the same parity history the probes write; the a-trous passes follow, so
//! `raa_slot_indirect` and the ssgi composite read it unchanged. Stage 1 writes a scratch buffer
//! that stage 2 reads, so stage 2 can write the final reservoirs without a race. The reservoir codec, WRS, GRIS weight, confidence clamp and boiling test
//! come from the ReSTIR core shared with the area-light ReSTIR DI (`restir_core.wgsl`). RAA's own
//! code; CPU twins in `restir_gi_cpu`.

use raa_core::GpuContext;
use raa_shader::{CLOUD_SHADOW_WGSL, DEPTH_WGSL, GI_VOLUME_WGSL, PRELUDE_WGSL};
use wgpu::TextureFormat;

use super::AccumulateViews;
use super::passes::{
    accumulate_group2_entries, accumulate_wgsl, depth_entry, dispatch, float_entry, layout, module,
    storage_entry, storage_texture_entry, uniform_entry,
};
use super::restir_gi_cpu::{
    RESTIR_GI_BOILING_FACTOR, RESTIR_GI_NORMAL_COS, RESTIR_GI_SPATIAL_RADIUS,
    RESTIR_GI_SPATIAL_SAMPLES,
};
use crate::emissive::{SCREEN_OVERRIDE, override_value};
use crate::parts::GBufferViews;
use crate::parts::GiShared;
use crate::tunables::GiTunables;

const PROBE_COMMON_WGSL: &str = include_str!("../../wgsl/screen/probe_common.wgsl");
const SCREEN_TRACE_WGSL: &str = include_str!("../../wgsl/screen/screen_trace.wgsl");
const SCREEN_EMISSION_WGSL: &str = include_str!("../../wgsl/screen/screen_emission.wgsl");
const RESTIR_CORE_WGSL: &str = include_str!("../../wgsl/area_shadow/restir_core.wgsl");
const RESTIR_GI_COMMON_WGSL: &str = include_str!("../../wgsl/screen/restir_gi_common.wgsl");
const RESTIR_GI_MAIN_WGSL: &str = include_str!("../../wgsl/screen/restir_gi_main.wgsl");
const RESTIR_GI_SPATIAL_WGSL: &str = include_str!("../../wgsl/screen/restir_gi_spatial.wgsl");
const RESTIR_GI_DENOISE_WGSL: &str = include_str!("../../wgsl/screen/restir_gi_denoise.wgsl");
const RESTIR_GI_STAGES_WGSL: &str = include_str!("../../wgsl/screen/restir_gi_stages.wgsl");

/// The (override, tunable key) pairs of restir_gi_common.wgsl, composed into stages 1 and 2.
const COMMON_OVERRIDES: [(&str, &str); 3] = [
    ("RAA_RGI_JACOBIAN_MAX", "restirJacobianMax"),
    ("RAA_RGI_VISIBILITY_STEPS", "restirVisibilitySteps"),
    ("RAA_RGI_VISIBILITY_DISTANCE", "restirVisibilityDistance"),
];
/// The (override, tunable key) pairs of stage 1: restir_gi_common.wgsl and restir_gi_main.wgsl.
const MAIN_OVERRIDES: [(&str, &str); 9] = [
    COMMON_OVERRIDES[0],
    COMMON_OVERRIDES[1],
    COMMON_OVERRIDES[2],
    ("RAA_RGI_SCREEN_DISTANCE", "restirScreenDistance"),
    ("RAA_RGI_SCREEN_STEPS", "restirScreenSteps"),
    ("RAA_RGI_SCREEN_THICKNESS", "restirScreenThickness"),
    ("RAA_RGI_VOLUME_DISTANCE", "restirVolumeDistance"),
    ("RAA_RGI_VOLUME_STEPS", "restirVolumeSteps"),
    ("RAA_RGI_VOLUME_HIT_VOXELS", "restirVolumeHitVoxels"),
];
/// The (override, tunable key) pairs of stage 3: the accumulate.wgsl overrides it composes.
const DENOISE_OVERRIDES: [(&str, &str); 8] = [
    ("RAA_GI_ACC_DEPTH_TOLERANCE", "accDepthTolerance"),
    ("RAA_GI_ACC_NORMAL_COS", "accNormalCos"),
    ("RAA_GI_ACC_MIN_WEIGHT", "accMinWeight"),
    ("RAA_GI_ACC_CHANGE_SLACK", "accChangeSlack"),
    ("RAA_GI_ACC_CHANGE_MIN", "accChangeMin"),
    ("RAA_GI_OCCLUSION_DIRECTIONS", "occlusionDirections"),
    ("RAA_GI_OCCLUSION_STEPS", "occlusionSteps"),
    ("RAA_GI_OCCLUSION_RADIUS", "occlusionRadius"),
];

/// Entry of stage 1 (initial sample + temporal reuse).
pub const RESTIR_GI_MAIN_ENTRY: &str = "raa_gi_restir_gi_main";
/// Entry of the `gi.restir.initial` stage (the one-candidate initial reservoir).
pub const RESTIR_GI_INITIAL_ENTRY: &str = "raa_gi_restir_gi_initial";
/// Entry of the `gi.restir.temporal` stage (initial reservoir + temporal merge).
pub const RESTIR_GI_TEMPORAL_ENTRY: &str = "raa_gi_restir_gi_temporal";
/// Entry of stage 2 (spatial reuse + resolve).
pub const RESTIR_GI_SPATIAL_ENTRY: &str = "raa_gi_restir_gi_spatial";
/// Entry of stage 3 (denoise).
pub const RESTIR_GI_DENOISE_ENTRY: &str = "raa_gi_restir_gi_denoise";

/// `GiFrameGpu` bytes (S38).
const FRAME_SIZE: u64 = 432;
/// Size of `RestirGiParamsGpu`: four u32 then four f32.
pub const RESTIR_GI_PARAMS_SIZE: u64 = 32;
/// Bytes of one packed reservoir: two vec4<u32>.
pub const RESTIR_GI_RESERVOIR_BYTES: u64 = 32;

/// Stage 1: PRELUDE + CLOUD_SHADOW + GI_VOLUME + DEPTH + probe_common + screen_trace +
/// screen_emission + restir_core + restir_gi_common + restir_gi_main.
pub fn restir_gi_main_module_wgsl() -> String {
    format!(
        "{PRELUDE_WGSL}{CLOUD_SHADOW_WGSL}{GI_VOLUME_WGSL}{DEPTH_WGSL}{PROBE_COMMON_WGSL}\
         {SCREEN_TRACE_WGSL}{SCREEN_EMISSION_WGSL}{RESTIR_CORE_WGSL}{RESTIR_GI_COMMON_WGSL}{RESTIR_GI_MAIN_WGSL}"
    )
}

/// Stages 1 split (S58): the stage-1 prefix of [`restir_gi_main_module_wgsl`] plus
/// restir_gi_stages.wgsl, whose `raa_gi_restir_gi_initial` and `raa_gi_restir_gi_temporal` entry
/// points reuse the main file's functions.
pub fn restir_gi_stages_module_wgsl() -> String {
    format!("{}{RESTIR_GI_STAGES_WGSL}", restir_gi_main_module_wgsl())
}

/// Stage 2: PRELUDE + CLOUD_SHADOW + GI_VOLUME + DEPTH + probe_common + restir_core +
/// restir_gi_common + restir_gi_spatial.
pub fn restir_gi_spatial_module_wgsl() -> String {
    format!(
        "{PRELUDE_WGSL}{CLOUD_SHADOW_WGSL}{GI_VOLUME_WGSL}{DEPTH_WGSL}{PROBE_COMMON_WGSL}\
         {RESTIR_CORE_WGSL}{RESTIR_GI_COMMON_WGSL}{RESTIR_GI_SPATIAL_WGSL}"
    )
}

/// Stage 3: PRELUDE + CLOUD_SHADOW + GI_VOLUME + DEPTH + probe_common + accumulate (the shared
/// temporal accumulation and the screen occlusion) + restir_gi_denoise.
pub fn restir_gi_denoise_module_wgsl() -> String {
    let accumulate = accumulate_wgsl();
    format!(
        "{PRELUDE_WGSL}{CLOUD_SHADOW_WGSL}{GI_VOLUME_WGSL}{DEPTH_WGSL}{PROBE_COMMON_WGSL}\
         {accumulate}{RESTIR_GI_DENOISE_WGSL}"
    )
}

/// Stage 1 group 2: 0 frame, 1 depth, 2 normal, 3 filterable lit history, 4 pyramid, 6 the GI
/// voxel words (read), 7 screen params, 8 motion, 10 previous final reservoirs (read), 11 the scratch reservoirs
/// (read_write), 12 params.
pub fn main_group2_entries() -> [wgpu::BindGroupLayoutEntry; 11] {
    [
        uniform_entry(0, FRAME_SIZE),
        depth_entry(1),
        float_entry(2, false),
        float_entry(3, true),
        float_entry(4, false),
        storage_entry(6, true),
        uniform_entry(7, super::passes::SCREEN_PARAMS_SIZE),
        float_entry(8, false),
        storage_entry(10, true),
        storage_entry(11, false),
        uniform_entry(12, RESTIR_GI_PARAMS_SIZE),
    ]
}

/// Initial stage group 2: the bindings of stage 1 (0 frame, 1 depth, 2 normal, 3 history, 4
/// pyramid, 6 voxels, 7 screen params, 8 motion, 10 previous reservoirs, 11 the stage-1 scratch,
/// 12 params) plus 16 the initial reservoirs (read_write, the stage's output).
pub fn initial_group2_entries() -> [wgpu::BindGroupLayoutEntry; 12] {
    let [a, b, c, d, e, f, g, h, i, j, k] = main_group2_entries();
    [a, b, c, d, e, f, g, h, i, j, k, storage_entry(16, false)]
}

/// Temporal stage group 2: the initial stage's bindings plus 17 the Rgba16Float raw irradiance
/// (written).
pub fn temporal_group2_entries() -> [wgpu::BindGroupLayoutEntry; 13] {
    let [a, b, c, d, e, f, g, h, i, j, k, initial] = initial_group2_entries();
    let raw = storage_texture_entry(17, TextureFormat::Rgba16Float);
    [a, b, c, d, e, f, g, h, i, j, k, initial, raw]
}

/// Stage 2 group 2: 0 frame, 1 depth, 2 normal, 11 the scratch reservoirs, 12 params, 13 the
/// Rgba16Float raw irradiance, 15 the final reservoirs (written).
pub fn spatial_group2_entries() -> [wgpu::BindGroupLayoutEntry; 7] {
    [
        uniform_entry(0, FRAME_SIZE),
        depth_entry(1),
        float_entry(2, false),
        storage_entry(11, false),
        uniform_entry(12, RESTIR_GI_PARAMS_SIZE),
        storage_texture_entry(13, TextureFormat::Rgba16Float),
        storage_entry(15, false),
    ]
}

/// Stage 3 group 2: 0 frame, 1 depth, 2 normal, the accumulation bindings 3 4 6 7 8 9 10 11 12
/// of `accumulate_group2_entries` and 14 the raw irradiance.
pub fn denoise_group2_entries() -> [wgpu::BindGroupLayoutEntry; 13] {
    let shared = accumulate_group2_entries();
    [
        uniform_entry(0, FRAME_SIZE),
        depth_entry(1),
        float_entry(2, false),
        shared[0],
        shared[1],
        shared[2],
        shared[3],
        shared[4],
        shared[5],
        shared[6],
        shared[7],
        shared[8],
        float_entry(14, false),
    ]
}

/// The three ReSTIR GI pipelines and their group-2 layouts, over [empty, GI compute group 1,
/// group 2]. The pipelines take the overrides of the GI tunables (`GiTunables::constants`, see
/// the Handoff of `GiTunables` in crates/raa-gi/src/tunables.rs) and are rebuilt by
/// [`RestirGiPipelines::refresh`] when the published generation changes; the layouts stay.
pub(crate) struct RestirGiPipelines {
    empty: wgpu::BindGroupLayout,
    group1: wgpu::BindGroupLayout,
    main_layout: wgpu::BindGroupLayout,
    spatial_layout: wgpu::BindGroupLayout,
    denoise_layout: wgpu::BindGroupLayout,
    modules: [wgpu::ShaderModule; 3],
    main: wgpu::ComputePipeline,
    spatial: wgpu::ComputePipeline,
    denoise: wgpu::ComputePipeline,
    generation: u64,
    /// The emissive generation the main pipeline was built with.
    emissive_generation: u64,
}

/// A compute pipeline of `module` over [empty, `group1`, `group2`] with the override `constants`.
fn tuned_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    entry: &str,
    groups: [&wgpu::BindGroupLayout; 3],
    constants: &[(&'static str, f64)],
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(groups[0]), Some(groups[1]), Some(groups[2])],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants,
            ..Default::default()
        },
        cache: None,
    })
}

impl RestirGiPipelines {
    /// Builds the layouts, the modules and the pipelines with the tunables `shared` publishes.
    pub(crate) fn new(
        gpu: &GpuContext,
        group1: &wgpu::BindGroupLayout,
        shared: &GiShared,
    ) -> RestirGiPipelines {
        let device = gpu.device();
        let empty = layout(device, "raa-gi.restir-gi.empty", &[]);
        let main_layout = layout(device, "raa-gi.restir-gi.main", &main_group2_entries());
        let spatial_layout = layout(
            device,
            "raa-gi.restir-gi.spatial",
            &spatial_group2_entries(),
        );
        let denoise_layout = layout(
            device,
            "raa-gi.restir-gi.denoise",
            &denoise_group2_entries(),
        );
        let modules = [
            module(
                device,
                "raa-gi.restir-gi.main",
                &restir_gi_main_module_wgsl(),
            ),
            module(
                device,
                "raa-gi.restir-gi.spatial",
                &restir_gi_spatial_module_wgsl(),
            ),
            module(
                device,
                "raa-gi.restir-gi.denoise",
                &restir_gi_denoise_module_wgsl(),
            ),
        ];
        let tunables = shared.tunables();
        let [main, spatial, denoise] = Self::build(
            device,
            &modules,
            [&empty, group1],
            [&main_layout, &spatial_layout, &denoise_layout],
            &tunables,
            shared.emissive_sources().screen,
        );
        RestirGiPipelines {
            empty,
            group1: group1.clone(),
            main_layout,
            spatial_layout,
            denoise_layout,
            modules,
            main,
            spatial,
            denoise,
            generation: tunables.generation(),
            emissive_generation: shared.emissive_generation(),
        }
    }

    /// The three pipelines with the overrides of `tunables`.
    fn build(
        device: &wgpu::Device,
        modules: &[wgpu::ShaderModule; 3],
        shared: [&wgpu::BindGroupLayout; 2],
        group2: [&wgpu::BindGroupLayout; 3],
        tunables: &GiTunables,
        screen_emissive: bool,
    ) -> [wgpu::ComputePipeline; 3] {
        let stages = [
            (
                "raa-gi.restir-gi.main",
                RESTIR_GI_MAIN_ENTRY,
                &MAIN_OVERRIDES[..],
            ),
            (
                "raa-gi.restir-gi.spatial",
                RESTIR_GI_SPATIAL_ENTRY,
                &COMMON_OVERRIDES[..],
            ),
            (
                "raa-gi.restir-gi.denoise",
                RESTIR_GI_DENOISE_ENTRY,
                &DENOISE_OVERRIDES[..],
            ),
        ];
        std::array::from_fn(|i| {
            let (label, entry, overrides) = stages[i];
            let mut constants = tunables.constants(overrides);
            if i == 0 {
                constants.push((SCREEN_OVERRIDE, override_value(screen_emissive)));
            }
            tuned_pipeline(
                device,
                label,
                &modules[i],
                entry,
                [shared[0], shared[1], group2[i]],
                &constants,
            )
        })
    }

    /// Rebuilds the three pipelines when the generation `shared` publishes differs from the one
    /// they were built with; the read is allocation-free when it does not.
    pub(crate) fn refresh(&mut self, gpu: &GpuContext, shared: &GiShared) {
        if shared.tunables_generation() == self.generation
            && shared.emissive_generation() == self.emissive_generation
        {
            return;
        }
        let tunables = shared.tunables();
        let [main, spatial, denoise] = Self::build(
            gpu.device(),
            &self.modules,
            [&self.empty, &self.group1],
            [
                &self.main_layout,
                &self.spatial_layout,
                &self.denoise_layout,
            ],
            &tunables,
            shared.emissive_sources().screen,
        );
        self.main = main;
        self.spatial = spatial;
        self.denoise = denoise;
        self.generation = tunables.generation();
        self.emissive_generation = shared.emissive_generation();
    }

    /// Records the three stages in `pass` (group 1 already set) with the groups of `work`.
    pub(crate) fn record(&self, pass: &mut wgpu::ComputePass<'_>, work: &RestirGiWork<'_>) {
        let (width, height) = work.viewport;
        pass.set_pipeline(&self.main);
        pass.set_bind_group(2, work.main, &[]);
        dispatch(pass, width, height);
        pass.set_pipeline(&self.spatial);
        pass.set_bind_group(2, work.spatial, &[]);
        dispatch(pass, width, height);
        pass.set_pipeline(&self.denoise);
        pass.set_bind_group(2, work.denoise, &[]);
        dispatch(pass, width, height);
    }
}

/// The four stage pipelines of the S58 stage graph and their group-2 layouts, over [empty, the GI
/// compute group 1, group 2]: `initial` and `temporal` are the two entries of
/// [`restir_gi_stages_module_wgsl`], `spatial` and `denoise` the existing stage-2 and stage-3
/// builders.
///
/// Handoff: raa-gi stages 6/11: gi.restir.initial and gi.restir.temporal stages.
/// `RestirStagePipelines` is built by `RestirStagePipelines::new(gpu: &GpuContext, group1:
/// &wgpu::BindGroupLayout, tunables: &GiTunables, emissive: bool) -> RestirStagePipelines` from
/// the tunable overrides (initial and temporal: `MAIN_OVERRIDES`; spatial: `COMMON_OVERRIDES`;
/// denoise: `DENOISE_OVERRIDES`) and the screen emission gate. The related items are
/// `initial_group2_entries() -> [wgpu::BindGroupLayoutEntry; 12]`, `temporal_group2_entries() ->
/// [wgpu::BindGroupLayoutEntry; 13]`, `RESTIR_GI_INITIAL_ENTRY` and `RESTIR_GI_TEMPORAL_ENTRY`.
/// The state it passes is the pub(crate) pipeline fields `initial`, `temporal`, `spatial` and
/// `denoise` with their group-2 layouts `initial_layout`, `temporal_layout`, `spatial_layout` and
/// `denoise_layout`; the stage builds the pipelines once and binds its groups over those layouts.
pub(crate) struct RestirStagePipelines {
    pub(crate) initial: wgpu::ComputePipeline,
    pub(crate) temporal: wgpu::ComputePipeline,
    pub(crate) spatial: wgpu::ComputePipeline,
    pub(crate) denoise: wgpu::ComputePipeline,
    pub(crate) initial_layout: wgpu::BindGroupLayout,
    pub(crate) temporal_layout: wgpu::BindGroupLayout,
    pub(crate) spatial_layout: wgpu::BindGroupLayout,
    pub(crate) denoise_layout: wgpu::BindGroupLayout,
}

impl RestirStagePipelines {
    /// Builds the two stage entry points of `restir_gi_stages_module_wgsl` and the spatial and
    /// denoise pipelines over [empty, `group1`, their group-2 layout]; `emissive` is the screen
    /// emission gate `RestirGiPipelines::new` passes.
    pub(crate) fn new(
        gpu: &GpuContext,
        group1: &wgpu::BindGroupLayout,
        tunables: &GiTunables,
        emissive: bool,
    ) -> RestirStagePipelines {
        let device = gpu.device();
        let group2 =
            |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| layout(device, label, entries);
        let empty = group2("raa-gi.restir-stages.empty", &[]);
        let initial_layout = group2("raa-gi.restir-stages.initial", &initial_group2_entries());
        let temporal_layout = group2("raa-gi.restir-stages.temporal", &temporal_group2_entries());
        let spatial_layout = group2("raa-gi.restir-stages.spatial", &spatial_group2_entries());
        let denoise_layout = group2("raa-gi.restir-stages.denoise", &denoise_group2_entries());
        let stages = module(
            device,
            "raa-gi.restir-stages.stages",
            &restir_gi_stages_module_wgsl(),
        );
        let spatial_module = module(
            device,
            "raa-gi.restir-stages.spatial",
            &restir_gi_spatial_module_wgsl(),
        );
        let denoise_module = module(
            device,
            "raa-gi.restir-stages.denoise",
            &restir_gi_denoise_module_wgsl(),
        );
        let mut initial_constants = tunables.constants(&MAIN_OVERRIDES);
        initial_constants.push((SCREEN_OVERRIDE, override_value(emissive)));
        let initial = tuned_pipeline(
            device,
            "raa-gi.restir-stages.initial",
            &stages,
            RESTIR_GI_INITIAL_ENTRY,
            [&empty, group1, &initial_layout],
            &initial_constants,
        );
        let temporal = tuned_pipeline(
            device,
            "raa-gi.restir-stages.temporal",
            &stages,
            RESTIR_GI_TEMPORAL_ENTRY,
            [&empty, group1, &temporal_layout],
            &tunables.constants(&MAIN_OVERRIDES),
        );
        let spatial = tuned_pipeline(
            device,
            "raa-gi.restir-stages.spatial",
            &spatial_module,
            RESTIR_GI_SPATIAL_ENTRY,
            [&empty, group1, &spatial_layout],
            &tunables.constants(&COMMON_OVERRIDES),
        );
        let denoise = tuned_pipeline(
            device,
            "raa-gi.restir-stages.denoise",
            &denoise_module,
            RESTIR_GI_DENOISE_ENTRY,
            [&empty, group1, &denoise_layout],
            &tunables.constants(&DENOISE_OVERRIDES),
        );
        RestirStagePipelines {
            initial,
            temporal,
            spatial,
            denoise,
            initial_layout,
            temporal_layout,
            spatial_layout,
            denoise_layout,
        }
    }
}

/// The bind groups of one frame.
pub(crate) struct RestirGiWork<'a> {
    pub(crate) main: &'a wgpu::BindGroup,
    pub(crate) spatial: &'a wgpu::BindGroup,
    pub(crate) denoise: &'a wgpu::BindGroup,
    pub(crate) viewport: (u32, u32),
}

/// Views of the screen part the ReSTIR groups bind: the frame record, the lit history (or the
/// dummy), the pyramid, the screen params and the accumulation targets.
pub(crate) struct RestirGiInputs<'a> {
    pub(crate) frame: &'a wgpu::Buffer,
    pub(crate) history: &'a wgpu::TextureView,
    /// The GI voxel words, or the screen part's zeroed dummy while GI has none.
    pub(crate) voxels: &'a wgpu::Buffer,
    pub(crate) pyramid: &'a wgpu::TextureView,
    pub(crate) screen_params: &'a wgpu::Buffer,
    pub(crate) accumulate: AccumulateViews,
}

struct RestirGiBindings {
    frame: wgpu::Buffer,
    history: wgpu::TextureView,
    views: GBufferViews,
    voxels: wgpu::Buffer,
    outputs: wgpu::TextureView,
    main: [wgpu::BindGroup; 2],
    spatial: [wgpu::BindGroup; 2],
    denoise: [wgpu::BindGroup; 2],
}

/// The viewport-sized ReSTIR GI resources: two final reservoir buffers (ping-pong by parity), the
/// stage-1 scratch reservoirs, the raw irradiance texture and the params uniform; bind groups
/// cached until the frame buffer, the history or the screen resources change or
/// `invalidate_bindings` runs.
pub(crate) struct RestirGiResources {
    viewport: (u32, u32),
    reservoirs: [wgpu::Buffer; 2],
    scratch: wgpu::Buffer,
    raw_view: wgpu::TextureView,
    params: wgpu::Buffer,
    bindings: Option<RestirGiBindings>,
}

/// GPU bytes of the resources of `viewport`: three reservoir buffers (two final, one scratch) of
/// 32 B per pixel, the raw Rgba16Float texture (8 B per pixel) and the 32-B params.
pub(crate) fn restir_gi_bytes(viewport: (u32, u32)) -> u64 {
    let pixels = u64::from(viewport.0.max(1)) * u64::from(viewport.1.max(1));
    3 * RESTIR_GI_RESERVOIR_BYTES * pixels + 8 * pixels + RESTIR_GI_PARAMS_SIZE
}

/// The params-uniform tunables of `RestirGiParamsGpu`: spatial samples, spatial radius, boiling
/// factor and normal cosine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RestirGiTuned {
    pub(crate) spatial_samples: u32,
    pub(crate) spatial_radius: f32,
    pub(crate) boiling_factor: f32,
    pub(crate) normal_cos: f32,
}

impl RestirGiTuned {
    /// The defaults: the constants of `restir_gi_cpu`.
    pub(crate) const DEFAULT: RestirGiTuned = RestirGiTuned {
        spatial_samples: RESTIR_GI_SPATIAL_SAMPLES,
        spatial_radius: RESTIR_GI_SPATIAL_RADIUS,
        boiling_factor: RESTIR_GI_BOILING_FACTOR,
        normal_cos: RESTIR_GI_NORMAL_COS,
    };

    /// The current values `shared` publishes (allocation-free), a default for an absent key.
    pub(crate) fn read(shared: &GiShared) -> RestirGiTuned {
        let d = RestirGiTuned::DEFAULT;
        let f = |key: &str, default: f32| shared.tunable_value(key).map_or(default, |v| v as f32);
        RestirGiTuned {
            spatial_samples: shared
                .tunable_value("restirSpatialSamples")
                .map_or(d.spatial_samples, |v| {
                    v.clamp(0.0, f64::from(u32::MAX)) as u32
                }),
            spatial_radius: f("restirSpatialRadius", d.spatial_radius),
            boiling_factor: f("restirBoilingFactor", d.boiling_factor),
            normal_cos: f("restirNormalCos", d.normal_cos),
        }
    }
}

/// The 32 B of `RestirGiParamsGpu`, little-endian; `m_cap` is the `restirMCap` setting and
/// `tuned` the params-uniform tunables.
pub(crate) fn params_bytes(
    frame_index: u32,
    history_usable: bool,
    screen_traces: bool,
    m_cap: f32,
    tuned: RestirGiTuned,
) -> [u8; 32] {
    let words: [u32; 8] = [
        frame_index,
        u32::from(history_usable),
        tuned.spatial_samples,
        u32::from(screen_traces),
        tuned.spatial_radius.to_bits(),
        m_cap.to_bits(),
        tuned.boiling_factor.to_bits(),
        tuned.normal_cos.to_bits(),
    ];
    let mut out = [0u8; 32];
    for (i, w) in words.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

impl RestirGiResources {
    pub(crate) fn new(gpu: &GpuContext, viewport: (u32, u32)) -> RestirGiResources {
        let viewport = (viewport.0.max(1), viewport.1.max(1));
        let device = gpu.device();
        let size = RESTIR_GI_RESERVOIR_BYTES * u64::from(viewport.0) * u64::from(viewport.1);
        let reservoir = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let reservoirs = [
            reservoir("raa-gi.restir-gi.reservoirs-a"),
            reservoir("raa-gi.restir-gi.reservoirs-b"),
        ];
        let scratch = reservoir("raa-gi.restir-gi.reservoirs-stage1");
        let raw = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("raa-gi.restir-gi.raw"),
            size: wgpu::Extent3d {
                width: viewport.0,
                height: viewport.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let raw_view = raw.create_view(&wgpu::TextureViewDescriptor::default());
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-gi.restir-gi.params"),
            size: RESTIR_GI_PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        RestirGiResources {
            viewport,
            reservoirs,
            scratch,
            raw_view,
            params,
            bindings: None,
        }
    }

    pub(crate) fn viewport(&self) -> (u32, u32) {
        self.viewport
    }

    pub(crate) fn allocated_bytes(&self) -> u64 {
        restir_gi_bytes(self.viewport)
    }

    pub(crate) fn write_params(
        &self,
        gpu: &GpuContext,
        frame_index: u32,
        history_usable: bool,
        screen_traces: bool,
        m_cap: f32,
        shared: &GiShared,
    ) {
        let tuned = RestirGiTuned::read(shared);
        gpu.queue().write_buffer(
            &self.params,
            0,
            &params_bytes(frame_index, history_usable, screen_traces, m_cap, tuned),
        );
    }

    pub(crate) fn invalidate_bindings(&mut self) {
        self.bindings = None;
    }

    /// The groups of `parity`: stage 1 reads reservoirs[1 - parity] and writes the scratch, stage
    /// 2 reads the scratch and writes reservoirs[parity], stage 3 accumulates into the screen
    /// history of `parity`.
    pub(crate) fn work<'a>(
        &'a mut self,
        gpu: &GpuContext,
        pipelines: &RestirGiPipelines,
        views: &GBufferViews,
        inputs: &RestirGiInputs<'_>,
        parity: usize,
    ) -> RestirGiWork<'a> {
        let stale = self.bindings.as_ref().is_none_or(|b| {
            b.frame != *inputs.frame
                || b.history != *inputs.history
                || b.views != *views
                || b.voxels != *inputs.voxels
                || b.outputs != inputs.accumulate.outputs[0]
        });
        if stale {
            self.bindings = Some(self.build(gpu, pipelines, views, inputs));
        }
        let b = self.bindings.as_ref().expect("built above");
        RestirGiWork {
            main: &b.main[parity & 1],
            spatial: &b.spatial[parity & 1],
            denoise: &b.denoise[parity & 1],
            viewport: self.viewport,
        }
    }

    fn build(
        &self,
        gpu: &GpuContext,
        pipelines: &RestirGiPipelines,
        views: &GBufferViews,
        inputs: &RestirGiInputs<'_>,
    ) -> RestirGiBindings {
        let device = gpu.device();
        let tex = wgpu::BindingResource::TextureView;
        let group = |label: &str,
                     layout: &wgpu::BindGroupLayout,
                     entries: &[(u32, wgpu::BindingResource<'_>)]| {
            let entries: Vec<wgpu::BindGroupEntry<'_>> = entries
                .iter()
                .map(|(binding, resource)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: resource.clone(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &entries,
            })
        };
        let main = std::array::from_fn(|parity| {
            group(
                "raa-gi.restir-gi.main",
                &pipelines.main_layout,
                &[
                    (0, inputs.frame.as_entire_binding()),
                    (1, tex(&views.depth)),
                    (2, tex(&views.normal)),
                    (3, tex(inputs.history)),
                    (4, tex(inputs.pyramid)),
                    (6, inputs.voxels.as_entire_binding()),
                    (7, inputs.screen_params.as_entire_binding()),
                    (8, tex(&views.motion)),
                    (10, self.reservoirs[1 - parity].as_entire_binding()),
                    (11, self.scratch.as_entire_binding()),
                    (12, self.params.as_entire_binding()),
                ],
            )
        });
        let spatial = std::array::from_fn(|parity| {
            group(
                "raa-gi.restir-gi.spatial",
                &pipelines.spatial_layout,
                &[
                    (0, inputs.frame.as_entire_binding()),
                    (1, tex(&views.depth)),
                    (2, tex(&views.normal)),
                    (11, self.scratch.as_entire_binding()),
                    (12, self.params.as_entire_binding()),
                    (13, tex(&self.raw_view)),
                    (15, self.reservoirs[parity].as_entire_binding()),
                ],
            )
        });
        let acc = &inputs.accumulate;
        let denoise = std::array::from_fn(|parity: usize| {
            group(
                "raa-gi.restir-gi.denoise",
                &pipelines.denoise_layout,
                &[
                    (0, inputs.frame.as_entire_binding()),
                    (1, tex(&views.depth)),
                    (2, tex(&views.normal)),
                    (3, tex(&views.motion)),
                    (4, tex(&acc.outputs[1 - parity])),
                    (6, tex(&acc.outputs[parity])),
                    (7, inputs.screen_params.as_entire_binding()),
                    (8, tex(&acc.metas[1 - parity])),
                    (9, tex(&acc.metas[parity])),
                    (10, tex(&acc.filter)),
                    (11, tex(&acc.debug)),
                    (12, acc.counters.as_entire_binding()),
                    (14, tex(&self.raw_view)),
                ],
            )
        });
        RestirGiBindings {
            frame: inputs.frame.clone(),
            history: inputs.history.clone(),
            views: views.clone(),
            voxels: inputs.voxels.clone(),
            outputs: acc.outputs[0].clone(),
            main,
            spatial,
            denoise,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modules_validate_and_hold_their_entries() {
        for (source, entry) in [
            (restir_gi_main_module_wgsl(), RESTIR_GI_MAIN_ENTRY),
            (restir_gi_spatial_module_wgsl(), RESTIR_GI_SPATIAL_ENTRY),
            (restir_gi_denoise_module_wgsl(), RESTIR_GI_DENOISE_ENTRY),
        ] {
            assert!(source.contains(entry));
            if let Err(e) = raa_shader::Composer::validate(&source) {
                panic!("{entry}: {}: {}", e.code, e.message);
            }
        }
    }

    #[test]
    fn stages_module_validates_and_holds_its_entries() {
        let source = restir_gi_stages_module_wgsl();
        for entry in [RESTIR_GI_INITIAL_ENTRY, RESTIR_GI_TEMPORAL_ENTRY] {
            assert!(source.contains(entry));
        }
        if let Err(e) = raa_shader::Composer::validate(&source) {
            panic!("restir_gi_stages: {}: {}", e.code, e.message);
        }
    }

    #[test]
    fn stage_group2_entries_are_unique_and_hold_the_new_bindings() {
        let initial = initial_group2_entries();
        let temporal = temporal_group2_entries();
        for entries in [&initial[..], &temporal[..]] {
            let mut numbers: Vec<u32> = entries.iter().map(|entry| entry.binding).collect();
            numbers.sort_unstable();
            let len = numbers.len();
            numbers.dedup();
            assert_eq!(numbers.len(), len, "duplicate group-2 binding");
        }
        assert!(initial.iter().any(|entry| entry.binding == 16));
        assert!(temporal.iter().any(|entry| entry.binding == 16));
        assert!(temporal.iter().any(|entry| entry.binding == 17));
    }

    #[test]
    fn main_module_stays_unchanged() {
        assert!(!restir_gi_main_module_wgsl().contains(RESTIR_GI_INITIAL_ENTRY));
    }

    #[test]
    fn params_and_bytes_follow_the_layout() {
        let bytes = params_bytes(7, true, false, 30.0, RestirGiTuned::DEFAULT);
        assert_eq!(&bytes[0..4], &7u32.to_le_bytes());
        assert_eq!(&bytes[4..8], &1u32.to_le_bytes());
        assert_eq!(&bytes[8..12], &4u32.to_le_bytes());
        assert_eq!(&bytes[12..16], &0u32.to_le_bytes());
        assert_eq!(&bytes[16..20], &24.0f32.to_le_bytes());
        assert_eq!(&bytes[20..24], &30.0f32.to_le_bytes());
        assert_eq!(&bytes[24..28], &20.0f32.to_le_bytes());
        assert_eq!(&bytes[28..32], &0.9f32.to_le_bytes());
        let tuned = RestirGiTuned {
            spatial_samples: 6,
            spatial_radius: 12.0,
            boiling_factor: 5.0,
            normal_cos: 0.5,
        };
        let bytes = params_bytes(7, true, false, 30.0, tuned);
        assert_eq!(&bytes[8..12], &6u32.to_le_bytes());
        assert_eq!(&bytes[16..20], &12.0f32.to_le_bytes());
        assert_eq!(&bytes[24..28], &5.0f32.to_le_bytes());
        assert_eq!(&bytes[28..32], &0.5f32.to_le_bytes());
        // 1280 x 720 = 921,600 px: 3 x 32 + 8 = 104 B per pixel, plus 32.
        assert_eq!(restir_gi_bytes((1280, 720)), 921_600 * 104 + 32);
    }
}
