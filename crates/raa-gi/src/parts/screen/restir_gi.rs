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
use raa_pipeline::Targets;
use raa_shader::{CLOUD_SHADOW_WGSL, DEPTH_WGSL, GI_VOLUME_WGSL, PRELUDE_WGSL};
use wgpu::TextureFormat;

use super::AccumulateViews;
use super::passes::{
    accumulate_group2_entries, accumulate_wgsl, compute_pipeline, depth_entry, dispatch,
    float_entry, layout, module, storage_entry, storage_texture_entry, uniform_entry,
};
use super::restir_gi_cpu::{
    RESTIR_GI_BOILING_FACTOR, RESTIR_GI_NORMAL_COS, RESTIR_GI_SPATIAL_RADIUS,
    RESTIR_GI_SPATIAL_SAMPLES,
};

const PROBE_COMMON_WGSL: &str = include_str!("../../wgsl/screen/probe_common.wgsl");
const SCREEN_TRACE_WGSL: &str = include_str!("../../wgsl/screen/screen_trace.wgsl");
const RESTIR_CORE_WGSL: &str = include_str!("../../wgsl/area_shadow/restir_core.wgsl");
const RESTIR_GI_COMMON_WGSL: &str = include_str!("../../wgsl/screen/restir_gi_common.wgsl");
const RESTIR_GI_MAIN_WGSL: &str = include_str!("../../wgsl/screen/restir_gi_main.wgsl");
const RESTIR_GI_SPATIAL_WGSL: &str = include_str!("../../wgsl/screen/restir_gi_spatial.wgsl");
const RESTIR_GI_DENOISE_WGSL: &str = include_str!("../../wgsl/screen/restir_gi_denoise.wgsl");

/// Entry of stage 1 (initial sample + temporal reuse).
pub const RESTIR_GI_MAIN_ENTRY: &str = "raa_gi_restir_gi_main";
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
/// restir_core + restir_gi_common + restir_gi_main.
pub fn restir_gi_main_module_wgsl() -> String {
    format!(
        "{PRELUDE_WGSL}{CLOUD_SHADOW_WGSL}{GI_VOLUME_WGSL}{DEPTH_WGSL}{PROBE_COMMON_WGSL}\
         {SCREEN_TRACE_WGSL}{RESTIR_CORE_WGSL}{RESTIR_GI_COMMON_WGSL}{RESTIR_GI_MAIN_WGSL}"
    )
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

/// Stage 1 group 2: 0 frame, 1 depth, 2 normal, 3 filterable lit history, 4 pyramid, 7 screen
/// params, 8 motion, 10 previous final reservoirs (read), 11 the scratch reservoirs
/// (read_write), 12 params.
pub fn main_group2_entries() -> [wgpu::BindGroupLayoutEntry; 10] {
    [
        uniform_entry(0, FRAME_SIZE),
        depth_entry(1),
        float_entry(2, false),
        float_entry(3, true),
        float_entry(4, false),
        uniform_entry(7, super::passes::SCREEN_PARAMS_SIZE),
        float_entry(8, false),
        storage_entry(10, true),
        storage_entry(11, false),
        uniform_entry(12, RESTIR_GI_PARAMS_SIZE),
    ]
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
/// group 2].
pub(crate) struct RestirGiPipelines {
    main_layout: wgpu::BindGroupLayout,
    spatial_layout: wgpu::BindGroupLayout,
    denoise_layout: wgpu::BindGroupLayout,
    main: wgpu::ComputePipeline,
    spatial: wgpu::ComputePipeline,
    denoise: wgpu::ComputePipeline,
}

impl RestirGiPipelines {
    pub(crate) fn new(gpu: &GpuContext, group1: &wgpu::BindGroupLayout) -> RestirGiPipelines {
        let device = gpu.device();
        let empty = layout(device, "raa-gi.restir-gi.empty", &[]);
        let main_layout = layout(device, "raa-gi.restir-gi.main", &main_group2_entries());
        let spatial_layout = layout(device, "raa-gi.restir-gi.spatial", &spatial_group2_entries());
        let denoise_layout = layout(device, "raa-gi.restir-gi.denoise", &denoise_group2_entries());
        let main_module = module(device, "raa-gi.restir-gi.main", &restir_gi_main_module_wgsl());
        let spatial_module = module(
            device,
            "raa-gi.restir-gi.spatial",
            &restir_gi_spatial_module_wgsl(),
        );
        let denoise_module = module(
            device,
            "raa-gi.restir-gi.denoise",
            &restir_gi_denoise_module_wgsl(),
        );
        let main = compute_pipeline(
            device,
            "raa-gi.restir-gi.main",
            &main_module,
            RESTIR_GI_MAIN_ENTRY,
            &[Some(&empty), Some(group1), Some(&main_layout)],
            None,
        );
        let spatial = compute_pipeline(
            device,
            "raa-gi.restir-gi.spatial",
            &spatial_module,
            RESTIR_GI_SPATIAL_ENTRY,
            &[Some(&empty), Some(group1), Some(&spatial_layout)],
            None,
        );
        let denoise = compute_pipeline(
            device,
            "raa-gi.restir-gi.denoise",
            &denoise_module,
            RESTIR_GI_DENOISE_ENTRY,
            &[Some(&empty), Some(group1), Some(&denoise_layout)],
            None,
        );
        RestirGiPipelines {
            main_layout,
            spatial_layout,
            denoise_layout,
            main,
            spatial,
            denoise,
        }
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
    pub(crate) pyramid: &'a wgpu::TextureView,
    pub(crate) screen_params: &'a wgpu::Buffer,
    pub(crate) accumulate: AccumulateViews,
}

struct RestirGiBindings {
    frame: wgpu::Buffer,
    history: wgpu::TextureView,
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

/// The 32 B of `RestirGiParamsGpu`, little-endian; `m_cap` is the `restirMCap` setting.
pub(crate) fn params_bytes(
    frame_index: u32,
    history_usable: bool,
    screen_traces: bool,
    m_cap: f32,
) -> [u8; 32] {
    let words: [u32; 8] = [
        frame_index,
        u32::from(history_usable),
        RESTIR_GI_SPATIAL_SAMPLES,
        u32::from(screen_traces),
        RESTIR_GI_SPATIAL_RADIUS.to_bits(),
        m_cap.to_bits(),
        RESTIR_GI_BOILING_FACTOR.to_bits(),
        RESTIR_GI_NORMAL_COS.to_bits(),
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
    ) {
        gpu.queue().write_buffer(
            &self.params,
            0,
            &params_bytes(frame_index, history_usable, screen_traces, m_cap),
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
        targets: &Targets,
        inputs: &RestirGiInputs<'_>,
        parity: usize,
    ) -> RestirGiWork<'a> {
        let stale = self.bindings.as_ref().is_none_or(|b| {
            b.frame != *inputs.frame
                || b.history != *inputs.history
                || b.outputs != inputs.accumulate.outputs[0]
        });
        if stale {
            self.bindings = Some(self.build(gpu, pipelines, targets, inputs));
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
        targets: &Targets,
        inputs: &RestirGiInputs<'_>,
    ) -> RestirGiBindings {
        let device = gpu.device();
        let tex = wgpu::BindingResource::TextureView;
        let group = |label: &str, layout: &wgpu::BindGroupLayout, entries: &[(u32, wgpu::BindingResource<'_>)]| {
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
                    (1, tex(targets.depth())),
                    (2, tex(targets.normal())),
                    (3, tex(inputs.history)),
                    (4, tex(inputs.pyramid)),
                    (7, inputs.screen_params.as_entire_binding()),
                    (8, tex(targets.motion())),
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
                    (1, tex(targets.depth())),
                    (2, tex(targets.normal())),
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
                    (1, tex(targets.depth())),
                    (2, tex(targets.normal())),
                    (3, tex(targets.motion())),
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
    fn params_and_bytes_follow_the_layout() {
        let bytes = params_bytes(7, true, false, 30.0);
        assert_eq!(&bytes[0..4], &7u32.to_le_bytes());
        assert_eq!(&bytes[4..8], &1u32.to_le_bytes());
        assert_eq!(&bytes[12..16], &0u32.to_le_bytes());
        assert_eq!(&bytes[20..24], &30.0f32.to_le_bytes());
        // 1280 x 720 = 921,600 px: 3 x 32 + 8 = 104 B per pixel, plus 32.
        assert_eq!(restir_gi_bytes((1280, 720)), 921_600 * 104 + 32);
    }
}
