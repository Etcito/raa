//! The stage `gi.accumulate` (contract 'rg-gi-stage-types'): a [`StageImpl`] that
//! `crate::stages::stage_impl` hands to the stage pass. It publishes the accumulated irradiance
//! (`output`, rgb with the occlusion in a), the moments (`meta`), the a-trous seed (`filter`) and
//! the debug record (`debug`). Its input decides the mode: the probes mode records the integrate
//! dispatch over the `probes` buffer, the irradiance mode records the ReSTIR denoise over the bound
//! `irradiance` image (see [`AccumulateMode`]). The previous frame comes from the bound `history`
//! and `historyMeta` inputs, which the render-graph executor owns; the stage keeps no parity
//! alternation of its own.

use raa_core::{GpuContext, RaaError};
use raa_pipeline::{ImageDesc, ImageDim, PrepareCtx, RecordCtx, ResourceTable, validate_tunable};

use super::passes::{
    PROBE_INTEGRATE_ENTRY, SCREEN_PARAMS_SIZE, ScreenPipelines, dispatch, layout, module,
    probe_integrate_module_wgsl,
};
use super::restir_gi::RestirStagePipelines;
use super::stage_restir::stage_pipelines;
use super::{buffer, full_view, screen_params_bytes, texture, texture_resource};
use crate::parts::GBufferViews;
use crate::settings::GiSettings;
use crate::stages::{StageCx, StageImpl};
use crate::{GiTunables, tunables};

/// The keys of `gi.accumulate`: `historyMax`, `historyMin`, `clampGamma`, then the eight
/// accumulation and occlusion tunables.
const ACCUMULATE_KEYS: &[&str] = &[
    "historyMax",
    "historyMin",
    "clampGamma",
    "accDepthTolerance",
    "accNormalCos",
    "accMinWeight",
    "accChangeSlack",
    "accChangeMin",
    "occlusionDirections",
    "occlusionSteps",
    "occlusionRadius",
];

/// The (override, tunable key) pairs of the integrate module (accumulate.wgsl). `passes.rs` keeps
/// its own copy private, so the stage builds its integrate pipeline with the same names.
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

/// The integrate pipeline over `screen.integrate_layout` with the overrides of `tunables`: the
/// same module and entry as the whole pass's integrate pipeline, with group 0 empty.
fn integrate_pipeline(
    gpu: &GpuContext,
    screen: &ScreenPipelines,
    group1: &wgpu::BindGroupLayout,
    tunables: &GiTunables,
) -> wgpu::ComputePipeline {
    let device = gpu.device();
    let empty = layout(device, "raa-gi.stage.accumulate.empty", &[]);
    let shader = module(
        device,
        "raa-gi.stage.accumulate",
        &probe_integrate_module_wgsl(),
    );
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("raa-gi.stage.accumulate"),
        bind_group_layouts: &[Some(&empty), Some(group1), Some(&screen.integrate_layout)],
        immediate_size: 0,
    });
    let constants = tunables.constants(INTEGRATE_OVERRIDES);
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("raa-gi.stage.accumulate"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some(PROBE_INTEGRATE_ENTRY),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &constants,
            ..Default::default()
        },
        cache: None,
    })
}

/// The pipelines of one values generation: the layouts of `ScreenPipelines` and the integrate
/// pipeline, with the stamp (values generation, emissive generation) they were built for.
struct Built {
    screen: ScreenPipelines,
    integrate: wgpu::ComputePipeline,
    stamp: (u64, u64),
}

/// The four outputs of one viewport, with the stage's uniform and counters.
struct Outputs {
    viewport: (u32, u32),
    output: wgpu::TextureView,
    meta: wgpu::TextureView,
    filter: wgpu::TextureView,
    debug: wgpu::TextureView,
    params: wgpu::Buffer,
    counters: wgpu::Buffer,
}

impl Outputs {
    /// Allocates the outputs for `viewport`: TEXTURE_BINDING | STORAGE_BINDING | COPY_SRC, the
    /// 32-B `GiScreenGpu` uniform and the 16-B counters.
    fn new(gpu: &GpuContext, viewport: (u32, u32)) -> Outputs {
        let usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        let view = |label: &str, format: wgpu::TextureFormat| {
            full_view(&texture(gpu, label, viewport, 1, format, usage))
        };
        Outputs {
            viewport,
            output: view(
                "raa-gi.stage.accumulate.output",
                wgpu::TextureFormat::Rgba16Float,
            ),
            meta: view(
                "raa-gi.stage.accumulate.meta",
                wgpu::TextureFormat::Rgba32Uint,
            ),
            filter: view(
                "raa-gi.stage.accumulate.filter",
                wgpu::TextureFormat::Rgba16Float,
            ),
            debug: view(
                "raa-gi.stage.accumulate.debug",
                wgpu::TextureFormat::Rgba32Uint,
            ),
            params: buffer(
                gpu,
                "raa-gi.stage.accumulate.params",
                SCREEN_PARAMS_SIZE,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            counters: buffer(
                gpu,
                "raa-gi.stage.accumulate.counters",
                16,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
        }
    }
}

/// The 1x1 history (alpha 1, as `ScreenResources::new` writes it) and the 1x1 zero moments the
/// integrate group binds while `history` or `historyMeta` is unbound.
struct Dummies {
    history: wgpu::TextureView,
    meta: wgpu::TextureView,
}

impl Dummies {
    fn new(gpu: &GpuContext) -> Dummies {
        let history = texture(
            gpu,
            "raa-gi.stage.accumulate.history-dummy",
            (1, 1),
            1,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        gpu.queue().write_texture(
            history.as_image_copy(),
            &[0, 0, 0, 0, 0, 0, 0x00, 0x3C],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let meta = texture(
            gpu,
            "raa-gi.stage.accumulate.meta-dummy",
            (1, 1),
            1,
            wgpu::TextureFormat::Rgba32Uint,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        Dummies {
            history: full_view(&history),
            meta: full_view(&meta),
        }
    }
}

/// The per-frame resources the integrate group binds besides the stage's own outputs; the group
/// is rebuilt when any of them changes.
#[derive(PartialEq)]
struct GroupKey {
    frame: wgpu::Buffer,
    views: GBufferViews,
    probes: wgpu::Buffer,
    history: Option<wgpu::TextureView>,
    meta: Option<wgpu::TextureView>,
}

/// The input the stage accumulates. A `probes` source wins over an `irradiance` input; the
/// render-graph executor refuses a graph that links both, before the stage runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum AccumulateMode {
    /// Neither input is set: the stage records nothing.
    #[default]
    None,
    /// The probes mode: the integrate dispatch over the `probes` buffer.
    Probes,
    /// The irradiance mode: the ReSTIR denoise over the bound `irradiance` image.
    Irradiance,
}

/// The mode of the inputs this frame: `Probes` when a `probes` source is set, else `Irradiance`
/// when the `irradiance` input is bound, else `None`.
fn mode_for(cx: &StageCx<'_>, resources: &ResourceTable) -> AccumulateMode {
    if cx
        .buffer_sources
        .iter()
        .any(|(port, _, _)| *port == "probes")
    {
        AccumulateMode::Probes
    } else if resources.input(cx.entry, "irradiance").is_some() {
        AccumulateMode::Irradiance
    } else {
        AccumulateMode::None
    }
}

/// The `gi.accumulate` stage: the probe irradiance (probes mode) or the raw irradiance (irradiance
/// mode) accumulated over the bound history, with the stage's own outputs and `GiScreenGpu`
/// uniform.
#[derive(Default)]
pub(crate) struct AccumulateStage {
    built: Option<Built>,
    outputs: Option<Outputs>,
    dummies: Option<Dummies>,
    groups: Option<(GroupKey, wgpu::BindGroup)>,
    /// The mode the latest `prepare` chose; `record` dispatches by it.
    mode: AccumulateMode,
    /// The ReSTIR denoise pipelines of the irradiance mode, built by `prepare` in that mode.
    restir: Option<(RestirStagePipelines, u64)>,
    /// The 32 uniform bytes `prepare` last wrote; the tests read the history word of them.
    #[cfg(test)]
    written: [u8; SCREEN_PARAMS_SIZE as usize],
}

/// One bind group entry, the binding of `binding_entry`'s resource.
fn binding_entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

impl AccumulateStage {
    /// Handoff: raa-gi stages 7/11: gi.restir.spatial stage and gi.accumulate irradiance mode.
    /// Today this returns the accumulation bindings 3, 4, 6, 7, 8, 9, 10, 11 and 12, in that
    /// order, over this stage's own outputs (`output`, `meta`, `filter`, `debug`), its uniform and
    /// its counters; `motion`, `history` and `history_meta` are the caller's views. It is empty
    /// before `prepare` allocates the outputs. The probes mode builds its integrate group with it,
    /// and card 7/11 adds binding 14 to the group it builds with it.
    pub(crate) fn accumulation_entries<'a>(
        &'a self,
        motion: &'a wgpu::TextureView,
        history: &'a wgpu::TextureView,
        history_meta: &'a wgpu::TextureView,
    ) -> Vec<wgpu::BindGroupEntry<'a>> {
        let Some(outputs) = self.outputs.as_ref() else {
            return Vec::new();
        };
        vec![
            binding_entry(3, texture_resource(motion)),
            binding_entry(4, texture_resource(history)),
            binding_entry(6, texture_resource(&outputs.output)),
            binding_entry(7, outputs.params.as_entire_binding()),
            binding_entry(8, texture_resource(history_meta)),
            binding_entry(9, texture_resource(&outputs.meta)),
            binding_entry(10, texture_resource(&outputs.filter)),
            binding_entry(11, texture_resource(&outputs.debug)),
            binding_entry(12, outputs.counters.as_entire_binding()),
        ]
    }
}

impl AccumulateStage {
    /// The mode the latest `prepare` chose.
    pub(crate) fn mode(&self) -> AccumulateMode {
        self.mode
    }

    /// Records the ReSTIR denoise over the viewport with the accumulation bindings, the frame
    /// buffer, the depth and normal, and the bound `irradiance` at binding 14. False (nothing
    /// recorded) without the `irradiance` input, group 1, the frame buffer or the built pipelines.
    fn record_irradiance(
        &self,
        cx: &StageCx<'_>,
        ctx: &mut RecordCtx<'_>,
    ) -> Result<bool, RaaError> {
        let Some(irradiance) = ctx.resources.input(cx.entry, "irradiance") else {
            return Ok(false);
        };
        let Some(group1) = cx.shared.compute_group1(ctx.lighting) else {
            return Ok(false);
        };
        let Some(frame) = cx.shared.frame_buffer() else {
            return Ok(false);
        };
        let (Some((pipelines, _)), Some(dummies), Some(outputs)) = (
            self.restir.as_ref(),
            self.dummies.as_ref(),
            self.outputs.as_ref(),
        ) else {
            return Ok(false);
        };
        let views = GBufferViews::resolve(ctx.targets, ctx.resources, cx.entry);
        let history = ctx
            .resources
            .input(cx.entry, "history")
            .map(|image| image.view);
        let meta = ctx
            .resources
            .input(cx.entry, "historyMeta")
            .map(|image| image.view);
        let history = history.as_ref().unwrap_or(&dummies.history);
        let meta = meta.as_ref().unwrap_or(&dummies.meta);
        let mut entries = self.accumulation_entries(&views.motion, history, meta);
        entries.extend([
            binding_entry(0, frame.as_entire_binding()),
            binding_entry(1, texture_resource(&views.depth)),
            binding_entry(2, texture_resource(&views.normal)),
            binding_entry(14, texture_resource(&irradiance.view)),
        ]);
        let bind_group = ctx
            .gpu
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raa-gi.stage.accumulate.denoise"),
                layout: &pipelines.denoise_layout,
                entries: &entries,
            });
        ctx.encoder.clear_buffer(&outputs.counters, 0, None);
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa-gi.stage.accumulate"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&pipelines.denoise);
        pass.set_bind_group(1, &group1, &[]);
        pass.set_bind_group(2, &bind_group, &[]);
        dispatch(&mut pass, outputs.viewport.0, outputs.viewport.1);
        Ok(true)
    }
}

impl StageImpl for AccumulateStage {
    fn param_keys(&self) -> &'static [&'static str] {
        ACCUMULATE_KEYS
    }

    /// `historyMax` and `historyMin` take 1 to 256, `clampGamma` 0.5 to 10; a tunable takes its
    /// `validate_tunable` range.
    fn check_value(&self, key: &str, value: f64) -> Result<(), String> {
        let range = match key {
            "historyMax" | "historyMin" => 1.0..=256.0,
            "clampGamma" => 0.5..=10.0,
            _ => return validate_tunable(tunables(), key, value).map(|_| ()),
        };
        if range.contains(&value) {
            Ok(())
        } else {
            Err(format!(
                "`{key}` is outside [{}, {}]",
                range.start(),
                range.end()
            ))
        }
    }

    /// Builds the pipelines for the values and emissive generation, allocates the outputs for the
    /// viewport, writes the uniform (history valid when both history inputs are bound and the
    /// camera is continuous; the other settings from `GiSettings::DEFAULT` with the stage values),
    /// and publishes `output`, `meta`, `filter` and `debug`.
    fn prepare(
        &mut self,
        cx: &StageCx<'_>,
        ctx: &mut PrepareCtx<'_>,
        published: &mut Vec<(&'static str, ImageDesc)>,
    ) -> Result<(), RaaError> {
        self.mode = mode_for(cx, ctx.resources);
        if self.mode == AccumulateMode::Irradiance {
            stage_pipelines(&mut self.restir, ctx.gpu, cx);
        }
        let emissive = (
            cx.shared.emissive_sources().screen,
            cx.shared.emissive_generation(),
        );
        let stamp = (cx.values_generation, emissive.1);
        if self.built.as_ref().is_none_or(|built| built.stamp != stamp) {
            let tunables = cx.tunables();
            let group1 = cx.shared.compute_group1_layout();
            let screen = ScreenPipelines::new(ctx.gpu, &group1, &tunables, emissive);
            let integrate = integrate_pipeline(ctx.gpu, &screen, &group1, &tunables);
            self.built = Some(Built {
                screen,
                integrate,
                stamp,
            });
        }
        let viewport = (ctx.size.0.max(1), ctx.size.1.max(1));
        if self
            .outputs
            .as_ref()
            .is_none_or(|outputs| outputs.viewport != viewport)
        {
            self.outputs = Some(Outputs::new(ctx.gpu, viewport));
            self.groups = None;
        }
        self.dummies.get_or_insert_with(|| Dummies::new(ctx.gpu));
        let bound = ctx.resources.input(cx.entry, "history").is_some()
            && ctx.resources.input(cx.entry, "historyMeta").is_some();
        let history_valid = bound && cx.shared.camera_valid();
        let settings = GiSettings {
            history_max: cx.value("historyMax", f64::from(GiSettings::DEFAULT.history_max)) as u32,
            history_min: cx.value("historyMin", f64::from(GiSettings::DEFAULT.history_min)) as u32,
            clamp_gamma: cx.value("clampGamma", f64::from(GiSettings::DEFAULT.clamp_gamma)) as f32,
            ..GiSettings::DEFAULT
        };
        // Parity 0: `ScreenState.parity` selects which whole-pass history half a frame writes. The
        // stage binds its history and moments explicitly, and accumulate.wgsl never reads the
        // parity word, so the stage writes parity 0 and the executor owns the alternation.
        let bytes = screen_params_bytes(history_valid, false, 0, &settings);
        let Some(outputs) = self.outputs.as_ref() else {
            return Ok(());
        };
        ctx.gpu.queue().write_buffer(&outputs.params, 0, &bytes);
        #[cfg(test)]
        {
            self.written = bytes;
        }
        for (port, view, format) in [
            ("output", &outputs.output, wgpu::TextureFormat::Rgba16Float),
            ("meta", &outputs.meta, wgpu::TextureFormat::Rgba32Uint),
            ("filter", &outputs.filter, wgpu::TextureFormat::Rgba16Float),
            ("debug", &outputs.debug, wgpu::TextureFormat::Rgba32Uint),
        ] {
            let desc = ImageDesc::new(ImageDim::D2, format, viewport.0, viewport.1, 1, 1);
            ctx.resources.publish(cx.entry, port, view, desc);
            published.push((port, desc));
        }
        Ok(())
    }

    /// In the irradiance mode records [`Self::record_irradiance`]. Otherwise clears the counters
    /// and records the integrate dispatch over the viewport in one compute pass with group 1 set;
    /// false (nothing recorded) without the `probes` buffer, group 1 or the frame buffer.
    fn record(&mut self, cx: &StageCx<'_>, ctx: &mut RecordCtx<'_>) -> Result<bool, RaaError> {
        if self.mode() == AccumulateMode::Irradiance {
            return self.record_irradiance(cx, ctx);
        }
        let Some((probes, _)) = cx.buffer(ctx.resources, "probes") else {
            return Ok(false);
        };
        let Some(group1) = cx.shared.compute_group1(ctx.lighting) else {
            return Ok(false);
        };
        let Some(frame) = cx.shared.frame_buffer() else {
            return Ok(false);
        };
        let (Some(built), Some(dummies), Some(outputs)) = (
            self.built.as_ref(),
            self.dummies.as_ref(),
            self.outputs.as_ref(),
        ) else {
            return Ok(false);
        };
        let key = GroupKey {
            frame,
            views: GBufferViews::resolve(ctx.targets, ctx.resources, cx.entry),
            probes,
            history: ctx
                .resources
                .input(cx.entry, "history")
                .map(|image| image.view),
            meta: ctx
                .resources
                .input(cx.entry, "historyMeta")
                .map(|image| image.view),
        };
        if self
            .groups
            .as_ref()
            .is_none_or(|(cached, _)| *cached != key)
        {
            let history = key.history.as_ref().unwrap_or(&dummies.history);
            let meta = key.meta.as_ref().unwrap_or(&dummies.meta);
            let bind_group = {
                let mut entries = self.accumulation_entries(&key.views.motion, history, meta);
                entries.extend([
                    binding_entry(0, key.frame.as_entire_binding()),
                    binding_entry(1, texture_resource(&key.views.depth)),
                    binding_entry(2, texture_resource(&key.views.normal)),
                    binding_entry(5, key.probes.as_entire_binding()),
                ]);
                ctx.gpu
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("raa-gi.stage.accumulate.integrate"),
                        layout: &built.screen.integrate_layout,
                        entries: &entries,
                    })
            };
            self.groups = Some((key, bind_group));
        }
        let Some((_, bind_group)) = self.groups.as_ref() else {
            return Ok(false);
        };
        ctx.encoder.clear_buffer(&outputs.counters, 0, None);
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa-gi.stage.accumulate"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&built.integrate);
        pass.set_bind_group(1, &group1, &[]);
        pass.set_bind_group(2, bind_group, &[]);
        dispatch(&mut pass, outputs.viewport.0, outputs.viewport.1);
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use raa_core::testing::skip;
    use raa_lit::{LitOptions, LitRenderer};
    use raa_pipeline::{LightingResources, Pipeline, PublishedImage, ResourceTable, Targets};

    use super::*;
    use crate::error::GI_INVALID;
    use crate::parts::GiShared;
    use crate::stages::{GiStageKind, install_stage};

    /// A stage's shared state with a 432-B frame buffer.
    fn shared(gpu: &GpuContext) -> GiShared {
        let shared = GiShared::new(gpu);
        let frame = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("accumulate.test.frame"),
            size: 432,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        shared.set_frame(Some(frame));
        shared
    }

    /// An 8x8 image of `format` for the stage's history inputs.
    fn bound(gpu: &GpuContext, format: wgpu::TextureFormat) -> PublishedImage {
        let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("accumulate.test.input"),
            size: wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        PublishedImage {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            desc: ImageDesc::new(ImageDim::D2, format, 8, 8, 1, 1),
        }
    }

    /// Runs `prepare` of `stage` on an 8x8 viewport with `table` as its resources and returns the
    /// published list. The lit handle is a fresh renderer's; no other pass ran.
    fn prepare(
        gpu: &GpuContext,
        stage: &mut AccumulateStage,
        shared: &GiShared,
        table: &ResourceTable,
    ) -> Vec<(&'static str, ImageDesc)> {
        prepare_with(gpu, stage, shared, table, &[])
    }

    /// `prepare` with the buffer `sources` of the stage, as `set_buffer_source` stores them.
    fn prepare_with(
        gpu: &GpuContext,
        stage: &mut AccumulateStage,
        shared: &GiShared,
        table: &ResourceTable,
        sources: &[(&'static str, String, String)],
    ) -> Vec<(&'static str, ImageDesc)> {
        let pipeline = Rc::new(RefCell::new(Pipeline::new(
            gpu,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        )));
        let lit = LitRenderer::new(gpu, &pipeline, LitOptions::default()).expect("lit renderer");
        let lit = lit.handle();
        let cx = StageCx {
            entry: "gi.accumulate#t",
            shared,
            lit: &lit,
            values: &[],
            values_generation: 0,
            history_valid: false,
            buffer_sources: sources,
        };
        let targets = Targets::new(gpu, (8, 8));
        let mut lighting = LightingResources::new(gpu);
        let mut ctx = PrepareCtx {
            gpu,
            size: (8, 8),
            frame: 0,
            time_s: 0.0,
            dt_s: 0.0,
            targets: &targets,
            lighting: &mut lighting,
            resources: table,
        };
        let mut published = Vec::new();
        stage
            .prepare(&cx, &mut ctx, &mut published)
            .expect("prepare");
        published
    }

    /// The `historyValid` word of the last uniform `prepare` wrote.
    fn history_word(stage: &AccumulateStage) -> u32 {
        let b = stage.written;
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    #[test]
    fn accumulate_publishes_the_four_outputs_at_8x8() {
        let test = "accumulate_publishes_the_four_outputs_at_8x8";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let table = ResourceTable::new();
        let published = prepare(&gpu, &mut AccumulateStage::default(), &shared(&gpu), &table);
        let desc = |format| ImageDesc::new(ImageDim::D2, format, 8, 8, 1, 1);
        let rgba16 = wgpu::TextureFormat::Rgba16Float;
        let rgba32 = wgpu::TextureFormat::Rgba32Uint;
        assert_eq!(
            published,
            vec![
                ("output", desc(rgba16)),
                ("meta", desc(rgba32)),
                ("filter", desc(rgba16)),
                ("debug", desc(rgba32)),
            ]
        );
        assert_eq!(table.entries().len(), 4);
        let found = table.get("gi.accumulate#t", "meta").map(|image| image.desc);
        assert_eq!(found, Some(desc(rgba32)));
    }

    #[test]
    fn accumulate_history_valid_needs_both_inputs_and_a_continuous_camera() {
        let test = "accumulate_history_valid_needs_both_inputs_and_a_continuous_camera";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let shared = shared(&gpu);
        let table = ResourceTable::new();
        let mut stage = AccumulateStage::default();
        prepare(&gpu, &mut stage, &shared, &table);
        assert_eq!(history_word(&stage), 0, "no history input is bound");
        table.bind_input(
            "gi.accumulate#t",
            "history",
            bound(&gpu, wgpu::TextureFormat::Rgba16Float),
        );
        table.bind_input(
            "gi.accumulate#t",
            "historyMeta",
            bound(&gpu, wgpu::TextureFormat::Rgba32Uint),
        );
        shared.set_frame_validity(true, true);
        prepare(&gpu, &mut stage, &shared, &table);
        assert_eq!(history_word(&stage), 1, "both inputs bound, camera valid");
    }

    /// An irradiance input alone is the irradiance mode, a probes source wins over it, and neither
    /// is no mode.
    #[test]
    fn accumulate_mode_follows_the_bound_inputs() {
        let test = "accumulate_mode_follows_the_bound_inputs";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let shared = shared(&gpu);
        let table = ResourceTable::new();
        let irradiance = bound(&gpu, wgpu::TextureFormat::Rgba16Float);
        table.bind_input("gi.accumulate#t", "irradiance", irradiance);
        let mut stage = AccumulateStage::default();
        prepare(&gpu, &mut stage, &shared, &table);
        assert_eq!(stage.mode(), AccumulateMode::Irradiance);
        let sources = [("probes", "gi.probes#t".to_string(), "probesRaw".to_string())];
        prepare_with(&gpu, &mut stage, &shared, &table, &sources);
        assert_eq!(stage.mode(), AccumulateMode::Probes);
        prepare(&gpu, &mut stage, &shared, &ResourceTable::new());
        assert_eq!(stage.mode(), AccumulateMode::None);
    }

    #[test]
    fn accumulate_refuses_a_zero_history_max() {
        let test = "accumulate_refuses_a_zero_history_max";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let pipeline = Rc::new(RefCell::new(Pipeline::new(
            &gpu,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        )));
        let handle = install_stage(&gpu, &pipeline, GiStageKind::Accumulate, "t").expect("install");
        let refused = handle.set_values(&[("historyMax".to_string(), 0.0)]);
        assert!(matches!(&refused, Err(error) if error.code() == GI_INVALID));
    }

    #[test]
    fn accumulate_records_the_integrate_dispatch_over_the_probes_buffer() {
        let test = "accumulate_records_the_integrate_dispatch_over_the_probes_buffer";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let shared = shared(&gpu);
        // The 3D volume atlas group 1 binds; the volume pass publishes it in a real frame.
        let volume = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("accumulate.test.volume"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        shared.set_atlas(Some(volume.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        })));
        let mut stage = AccumulateStage::default();
        let table = ResourceTable::new();
        prepare(&gpu, &mut stage, &shared, &table);
        // The probes come from the `probesRaw` buffer of `gi.probes#t`, bound as buffer source.
        let bytes = crate::parts::screen::passes::PROBE_BYTES;
        let probes = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("accumulate.test.probes"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        table.publish_buffer("gi.probes#t", "probesRaw", &probes, bytes);
        let sources = [("probes", "gi.probes#t".to_string(), "probesRaw".to_string())];
        let pipeline = Rc::new(RefCell::new(Pipeline::new(
            &gpu,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        )));
        let lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit renderer");
        let lit = lit.handle();
        let cx = StageCx {
            entry: "gi.accumulate#t",
            shared: &shared,
            lit: &lit,
            values: &[],
            values_generation: 0,
            history_valid: false,
            buffer_sources: &sources,
        };
        let targets = Targets::new(&gpu, (8, 8));
        let lighting = LightingResources::new(&gpu);
        let backbuffer = texture(
            &gpu,
            "accumulate.test.backbuffer",
            (8, 8),
            1,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let backbuffer = full_view(&backbuffer);
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut ctx = RecordCtx {
            gpu: &gpu,
            encoder: &mut encoder,
            size: (8, 8),
            frame: 0,
            time_s: 0.0,
            targets: &targets,
            lighting: &lighting,
            resources: &table,
            backbuffer: &backbuffer,
            backbuffer_format: wgpu::TextureFormat::Rgba8Unorm,
            surface_is_srgb: false,
        };
        let recorded = stage.record(&cx, &mut ctx).expect("record");
        gpu.queue().submit(Some(encoder.finish()));
        assert!(recorded, "the integrate dispatch was recorded");
    }
}
