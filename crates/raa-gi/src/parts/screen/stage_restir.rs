//! The stages `gi.restir.initial`, `gi.restir.temporal` and `gi.restir.spatial` (contract
//! 'rg-gi-stage-api'; types and ports of 'rg-gi-stage-types'). The initial stage writes one
//! candidate reservoir per pixel; the temporal stage merges it with its own previous reservoirs,
//! which it keeps internally in two buffers written alternately, and publishes the merged
//! reservoirs and their raw irradiance; the spatial stage reuses the reservoirs of its input and
//! publishes the final reservoirs and the raw irradiance. All three build their pipelines through
//! [`stage_pipelines`], which wraps the `RestirStagePipelines::new`
//! of `restir_gi.rs` (`RestirStagePipelines` at restir_gi.rs:404, with its Handoff doc comment at
//! restir_gi.rs:394), and record its entry points over the group-2 layouts of that file.

use raa_core::{GpuContext, RaaError};
use raa_pipeline::{ImageDesc, ImageDim, PrepareCtx, RecordCtx, validate_tunable};

use super::passes::{SCREEN_PARAMS_SIZE, dispatch};
use super::restir_gi::{
    RESTIR_GI_PARAMS_SIZE, RESTIR_GI_RESERVOIR_BYTES, RestirGiTuned, RestirStagePipelines,
    params_bytes,
};
use super::{buffer, full_view, group, screen_params_bytes, texture, texture_resource};
use crate::parts::GBufferViews;
use crate::settings::GiSettings;
use crate::stages::{StageCx, StageImpl};
use crate::tunables;

/// The format of the raw irradiance the temporal stage publishes.
const IRRADIANCE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The keys of `gi.restir.initial`: `screenTraces`, then the nine ReSTIR tunables of the stage.
const INITIAL_KEYS: &[&str] = &[
    "screenTraces",
    "restirJacobianMax",
    "restirVisibilitySteps",
    "restirVisibilityDistance",
    "restirScreenDistance",
    "restirScreenSteps",
    "restirScreenThickness",
    "restirVolumeDistance",
    "restirVolumeSteps",
    "restirVolumeHitVoxels",
];

/// The keys of `gi.restir.temporal`: `restirMCap`, then the five ReSTIR tunables of the stage.
const TEMPORAL_KEYS: &[&str] = &[
    "restirMCap",
    "restirBoilingFactor",
    "restirNormalCos",
    "restirJacobianMax",
    "restirVisibilitySteps",
    "restirVisibilityDistance",
];

/// The keys of `gi.restir.spatial`: the three params-uniform tunables, then the three overrides of
/// the spatial pipeline.
const SPATIAL_KEYS: &[&str] = &[
    "restirSpatialSamples",
    "restirSpatialRadius",
    "restirNormalCos",
    "restirJacobianMax",
    "restirVisibilitySteps",
    "restirVisibilityDistance",
];

/// `screenTraces` takes 0 or 1, `restirMCap` takes [1, 64], and a tunable takes its
/// `validate_tunable` range.
fn check(key: &str, value: f64) -> Result<(), String> {
    match key {
        "screenTraces" if value == 0.0 || value == 1.0 => Ok(()),
        "screenTraces" => Err("screenTraces takes 0 or 1".to_string()),
        "restirMCap" if (1.0..=64.0).contains(&value) => Ok(()),
        "restirMCap" => Err(format!("restirMCap {value} is outside [1, 64]")),
        _ => validate_tunable(tunables(), key, value).map(|_| ()),
    }
}

/// The bytes of one viewport of reservoirs.
fn reservoir_bytes(size: (u32, u32)) -> u64 {
    RESTIR_GI_RESERVOIR_BYTES * u64::from(size.0) * u64::from(size.1)
}

/// Builds the stage pipelines on first use and rebuilds them when the key stored beside them
/// changes. The key packs `cx.values_generation` above `GiShared::emissive_generation`; both are
/// counters far below 2^32, so the packing does not collide.
///
/// Handoff: raa-gi stages 7/11: gi.restir.spatial stage and gi.accumulate irradiance mode.
/// `stage_pipelines` (crates/raa-gi/src/parts/screen/stage_restir.rs:77) is the entry point, which
/// wraps `RestirStagePipelines::new` (restir_gi.rs:419). Card 7/11 calls it from its two stages
/// (the spatial stage and the irradiance mode of gi.accumulate). The state passed is the cached `RestirStagePipelines` and its key (the
/// values generation and the emissive generation), held in the caller's slot; today it builds
/// `RestirStagePipelines::new` with `cx.tunables()` and `cx.shared.emissive_sources().screen`.
pub(crate) fn stage_pipelines<'a>(
    slot: &'a mut Option<(RestirStagePipelines, u64)>,
    gpu: &GpuContext,
    cx: &StageCx<'_>,
) -> &'a RestirStagePipelines {
    let key = (cx.values_generation << 32) ^ cx.shared.emissive_generation();
    if slot.as_ref().is_some_and(|(_, stored)| *stored != key) {
        *slot = None;
    }
    let (pipelines, _) = slot.get_or_insert_with(|| {
        let layout = cx.shared.compute_group1_layout();
        let emissive = cx.shared.emissive_sources().screen;
        let built = RestirStagePipelines::new(gpu, &layout, &cx.tunables(), emissive);
        (built, key)
    });
    pipelines
}

/// The uniforms and dummy inputs both stages bind. `frame` counts the prepares of the stage.
struct Common {
    params: wgpu::Buffer,
    screen: wgpu::Buffer,
    /// A zeroed 32-B buffer for the previous reservoirs of the initial stage, which never reads them.
    zero: wgpu::Buffer,
    /// A 32-B buffer for the scratch reservoirs of the initial stage, which never writes them.
    scratch: wgpu::Buffer,
    voxel: wgpu::Buffer,
    history: wgpu::TextureView,
    pyramid: wgpu::TextureView,
    frame: u32,
}

impl Common {
    fn new(gpu: &GpuContext) -> Common {
        let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let storage = wgpu::BufferUsages::STORAGE;
        let dummy = |label: &str, format: wgpu::TextureFormat| {
            let usage = wgpu::TextureUsages::TEXTURE_BINDING;
            full_view(&texture(gpu, label, (1, 1), 1, format, usage))
        };
        Common {
            params: buffer(
                gpu,
                "raa-gi.stage.restir.params",
                RESTIR_GI_PARAMS_SIZE,
                uniform,
            ),
            screen: buffer(
                gpu,
                "raa-gi.stage.restir.screen",
                SCREEN_PARAMS_SIZE,
                uniform,
            ),
            zero: buffer(
                gpu,
                "raa-gi.stage.restir.zero",
                RESTIR_GI_RESERVOIR_BYTES,
                storage,
            ),
            scratch: buffer(
                gpu,
                "raa-gi.stage.restir.scratch",
                RESTIR_GI_RESERVOIR_BYTES,
                storage,
            ),
            voxel: buffer(gpu, "raa-gi.stage.restir.voxel-dummy", 16, storage),
            history: dummy(
                "raa-gi.stage.restir.history-dummy",
                wgpu::TextureFormat::Rgba16Float,
            ),
            pyramid: dummy(
                "raa-gi.stage.restir.pyramid-dummy",
                wgpu::TextureFormat::R32Float,
            ),
            frame: 0,
        }
    }

    /// Writes the 32-B `RestirGiParamsGpu` and the 32-B `GiScreenGpu` of this prepare, with the
    /// frame counter, then advances the counter. The screen traces and the m_cap come from the
    /// stage values; the other fields of `GiScreenGpu` are `GiSettings::DEFAULT`.
    fn write(&mut self, gpu: &GpuContext, cx: &StageCx<'_>, history_usable: bool) {
        let default = GiSettings::DEFAULT;
        let screen_traces = cx.value("screenTraces", f64::from(u8::from(default.screen_traces)));
        let screen_traces = screen_traces != 0.0;
        let m_cap = cx.value("restirMCap", f64::from(default.restir_m_cap)) as f32;
        let tuned = RestirGiTuned::read(cx.shared);
        let params = params_bytes(self.frame, history_usable, screen_traces, m_cap, tuned);
        gpu.queue().write_buffer(&self.params, 0, &params);
        let screen = screen_params_bytes(history_usable, screen_traces, 0, &default);
        gpu.queue().write_buffer(&self.screen, 0, &screen);
        self.frame = self.frame.wrapping_add(1);
    }
}

/// A reservoir buffer of one viewport.
fn reservoir(gpu: &GpuContext, label: &str, size: (u32, u32)) -> wgpu::Buffer {
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    buffer(gpu, label, reservoir_bytes(size), usage)
}

/// The two temporal reservoir buffers written alternately, the zeroed previous buffer of a fresh
/// history, and the raw irradiance of one viewport.
struct TemporalOutputs {
    size: (u32, u32),
    reservoirs: [wgpu::Buffer; 2],
    zero: wgpu::Buffer,
    irradiance: wgpu::TextureView,
}

impl TemporalOutputs {
    fn new(gpu: &GpuContext, size: (u32, u32)) -> TemporalOutputs {
        let usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        let irradiance = texture(
            gpu,
            "raa-gi.stage.restir-temporal.irradiance",
            size,
            1,
            IRRADIANCE_FORMAT,
            usage,
        );
        TemporalOutputs {
            size,
            reservoirs: [
                reservoir(gpu, "raa-gi.stage.restir-temporal.reservoirs-a", size),
                reservoir(gpu, "raa-gi.stage.restir-temporal.reservoirs-b", size),
            ],
            // wgpu zero-initialises a new buffer, so this is the previous reservoirs of a history
            // that is not valid.
            zero: buffer(
                gpu,
                "raa-gi.stage.restir-temporal.zero",
                reservoir_bytes(size),
                wgpu::BufferUsages::STORAGE,
            ),
            irradiance: full_view(&irradiance),
        }
    }
}

/// The `gi.restir.initial` stage: one candidate reservoir per pixel, published as `reservoirs`
/// (no image is published, so `published` stays empty).
#[derive(Default)]
pub(crate) struct RestirInitialStage {
    pipelines: Option<(RestirStagePipelines, u64)>,
    common: Option<Common>,
    output: Option<((u32, u32), wgpu::Buffer)>,
}

impl StageImpl for RestirInitialStage {
    fn param_keys(&self) -> &'static [&'static str] {
        INITIAL_KEYS
    }

    fn check_value(&self, key: &str, value: f64) -> Result<(), String> {
        check(key, value)
    }

    /// Builds the pipelines on first use, writes the uniforms (history from the shared state),
    /// and allocates and publishes the reservoirs of the viewport.
    fn prepare(
        &mut self,
        cx: &StageCx<'_>,
        ctx: &mut PrepareCtx<'_>,
        _published: &mut Vec<(&'static str, ImageDesc)>,
    ) -> Result<(), RaaError> {
        stage_pipelines(&mut self.pipelines, ctx.gpu, cx);
        let common = self.common.get_or_insert_with(|| Common::new(ctx.gpu));
        common.write(ctx.gpu, cx, cx.shared.history_valid());
        if self
            .output
            .as_ref()
            .is_none_or(|(size, _)| *size != ctx.size)
        {
            let output = reservoir(ctx.gpu, "raa-gi.stage.restir-initial.reservoirs", ctx.size);
            self.output = Some((ctx.size, output));
        }
        if let Some((size, output)) = self.output.as_ref() {
            ctx.resources
                .publish_buffer(cx.entry, "reservoirs", output, reservoir_bytes(*size));
        }
        Ok(())
    }

    /// Records the initial entry over the viewport. False (nothing recorded) without the
    /// `pyramid` input, the frame buffer, group 1, or the built pipelines and outputs.
    fn record(&mut self, cx: &StageCx<'_>, ctx: &mut RecordCtx<'_>) -> Result<bool, RaaError> {
        let Some(pyramid) = ctx.resources.input(cx.entry, "pyramid") else {
            return Ok(false);
        };
        let Some(frame) = cx.shared.frame_buffer() else {
            return Ok(false);
        };
        let Some(group1) = cx.shared.compute_group1(ctx.lighting) else {
            return Ok(false);
        };
        let (Some((pipelines, _)), Some(common), Some((size, output))) = (
            self.pipelines.as_ref(),
            self.common.as_ref(),
            self.output.as_ref(),
        ) else {
            return Ok(false);
        };
        let views = GBufferViews::resolve(ctx.targets, ctx.resources, cx.entry);
        let history_owned = cx.shared.history_view();
        let history = history_owned.as_ref().unwrap_or(&common.history);
        let voxels_owned = cx.shared.voxel_buffer().map(|(voxels, _)| voxels);
        let voxels = voxels_owned.as_ref().unwrap_or(&common.voxel);
        let bindings = group(
            ctx.gpu.device(),
            "raa-gi.stage.restir-initial",
            &pipelines.initial_layout,
            &[
                (0, frame.as_entire_binding()),
                (1, texture_resource(&views.depth)),
                (2, texture_resource(&views.normal)),
                (3, texture_resource(history)),
                (4, texture_resource(&pyramid.view)),
                (6, voxels.as_entire_binding()),
                (7, common.screen.as_entire_binding()),
                (8, texture_resource(&views.motion)),
                (10, common.zero.as_entire_binding()),
                (11, common.scratch.as_entire_binding()),
                (12, common.params.as_entire_binding()),
                (16, output.as_entire_binding()),
            ],
        );
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa-gi.stage.restir-initial"),
                timestamp_writes: None,
            });
        pass.set_bind_group(1, &group1, &[]);
        pass.set_bind_group(2, &bindings, &[]);
        pass.set_pipeline(&pipelines.initial);
        dispatch(&mut pass, size.0, size.1);
        Ok(true)
    }
}

/// The `gi.restir.temporal` stage: merges the initial reservoirs with its previous reservoirs
/// (kept in two buffers written alternately) and publishes `reservoirs` and `irradiance`.
#[derive(Default)]
pub(crate) struct RestirTemporalStage {
    pipelines: Option<(RestirStagePipelines, u64)>,
    common: Option<Common>,
    outputs: Option<TemporalOutputs>,
    /// The index of the reservoir buffer this frame writes; the other one is the previous.
    write: usize,
    /// True when the previous buffer holds the history: the lit history and the camera are
    /// continuous and the buffers were not reallocated.
    previous_valid: bool,
}

impl StageImpl for RestirTemporalStage {
    fn param_keys(&self) -> &'static [&'static str] {
        TEMPORAL_KEYS
    }

    fn check_value(&self, key: &str, value: f64) -> Result<(), String> {
        check(key, value)
    }

    /// Builds the pipelines on first use, reallocates the outputs when the viewport changed,
    /// chooses this frame's buffer by the frame counter, and publishes `reservoirs` and
    /// `irradiance`. The previous buffer counts as valid only with `cx.history_valid` and camera
    /// continuity; otherwise `record` binds the zeroed buffer.
    fn prepare(
        &mut self,
        cx: &StageCx<'_>,
        ctx: &mut PrepareCtx<'_>,
        published: &mut Vec<(&'static str, ImageDesc)>,
    ) -> Result<(), RaaError> {
        stage_pipelines(&mut self.pipelines, ctx.gpu, cx);
        let common = self.common.get_or_insert_with(|| Common::new(ctx.gpu));
        let fresh = self.outputs.as_ref().is_none_or(|out| out.size != ctx.size);
        if fresh {
            self.outputs = Some(TemporalOutputs::new(ctx.gpu, ctx.size));
        }
        self.previous_valid = !fresh && cx.history_valid && cx.shared.camera_valid();
        self.write = (common.frame & 1) as usize;
        common.write(ctx.gpu, cx, self.previous_valid);
        if let Some(outputs) = self.outputs.as_ref() {
            let size = reservoir_bytes(outputs.size);
            ctx.resources.publish_buffer(
                cx.entry,
                "reservoirs",
                &outputs.reservoirs[self.write],
                size,
            );
            let (w, h) = outputs.size;
            let desc = ImageDesc::new(ImageDim::D2, IRRADIANCE_FORMAT, w, h, 1, 1);
            ctx.resources
                .publish(cx.entry, "irradiance", &outputs.irradiance, desc);
            published.push(("irradiance", desc));
        }
        Ok(())
    }

    /// Records the temporal entry over the viewport, reading the `initial` buffer. False
    /// (nothing recorded) without that buffer, the frame buffer, group 1, or the built pipelines
    /// and outputs.
    fn record(&mut self, cx: &StageCx<'_>, ctx: &mut RecordCtx<'_>) -> Result<bool, RaaError> {
        let Some((initial, _)) = cx.buffer(ctx.resources, "initial") else {
            return Ok(false);
        };
        let Some(frame) = cx.shared.frame_buffer() else {
            return Ok(false);
        };
        let Some(group1) = cx.shared.compute_group1(ctx.lighting) else {
            return Ok(false);
        };
        let (Some((pipelines, _)), Some(common), Some(outputs)) = (
            self.pipelines.as_ref(),
            self.common.as_ref(),
            self.outputs.as_ref(),
        ) else {
            return Ok(false);
        };
        let views = GBufferViews::resolve(ctx.targets, ctx.resources, cx.entry);
        let history_owned = cx.shared.history_view();
        let history = history_owned.as_ref().unwrap_or(&common.history);
        let voxels_owned = cx.shared.voxel_buffer().map(|(voxels, _)| voxels);
        let voxels = voxels_owned.as_ref().unwrap_or(&common.voxel);
        let previous = if self.previous_valid {
            &outputs.reservoirs[self.write ^ 1]
        } else {
            &outputs.zero
        };
        let bindings = group(
            ctx.gpu.device(),
            "raa-gi.stage.restir-temporal",
            &pipelines.temporal_layout,
            &[
                (0, frame.as_entire_binding()),
                (1, texture_resource(&views.depth)),
                (2, texture_resource(&views.normal)),
                (3, texture_resource(history)),
                (4, texture_resource(&common.pyramid)),
                (6, voxels.as_entire_binding()),
                (7, common.screen.as_entire_binding()),
                (8, texture_resource(&views.motion)),
                (10, previous.as_entire_binding()),
                (11, outputs.reservoirs[self.write].as_entire_binding()),
                (12, common.params.as_entire_binding()),
                (16, initial.as_entire_binding()),
                (17, texture_resource(&outputs.irradiance)),
            ],
        );
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa-gi.stage.restir-temporal"),
                timestamp_writes: None,
            });
        pass.set_bind_group(1, &group1, &[]);
        pass.set_bind_group(2, &bindings, &[]);
        pass.set_pipeline(&pipelines.temporal);
        dispatch(&mut pass, outputs.size.0, outputs.size.1);
        Ok(true)
    }
}

/// The outputs of `gi.restir.spatial` for one viewport: the final reservoirs, the raw irradiance
/// and the params uniform.
struct SpatialOutputs {
    size: (u32, u32),
    reservoirs: wgpu::Buffer,
    irradiance: wgpu::TextureView,
    params: wgpu::Buffer,
}

impl SpatialOutputs {
    fn new(gpu: &GpuContext, size: (u32, u32)) -> SpatialOutputs {
        let usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        let irradiance = texture(
            gpu,
            "raa-gi.stage.restir-spatial.irradiance",
            size,
            1,
            IRRADIANCE_FORMAT,
            usage,
        );
        SpatialOutputs {
            size,
            reservoirs: reservoir(gpu, "raa-gi.stage.restir-spatial.reservoirs", size),
            irradiance: full_view(&irradiance),
            params: buffer(
                gpu,
                "raa-gi.stage.restir-spatial.params",
                RESTIR_GI_PARAMS_SIZE,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
        }
    }
}

/// The `gi.restir.spatial` stage: the spatial reuse of the reservoirs of its `reservoirs` input,
/// publishing the final `reservoirs` and the raw `irradiance`.
#[derive(Default)]
pub(crate) struct RestirSpatialStage {
    pipelines: Option<(RestirStagePipelines, u64)>,
    outputs: Option<SpatialOutputs>,
    /// The frame counter the spatial sampling pattern reads; the stage counts its prepares.
    frame: u32,
}

impl StageImpl for RestirSpatialStage {
    fn param_keys(&self) -> &'static [&'static str] {
        SPATIAL_KEYS
    }

    fn check_value(&self, key: &str, value: f64) -> Result<(), String> {
        check(key, value)
    }

    /// Builds the pipelines on first use, reallocates the outputs when the viewport changed,
    /// writes the params uniform from the stage values (the samples, radius and normal cosine;
    /// the m_cap is `GiSettings::DEFAULT`), and publishes `reservoirs` and `irradiance`.
    fn prepare(
        &mut self,
        cx: &StageCx<'_>,
        ctx: &mut PrepareCtx<'_>,
        published: &mut Vec<(&'static str, ImageDesc)>,
    ) -> Result<(), RaaError> {
        stage_pipelines(&mut self.pipelines, ctx.gpu, cx);
        if self.outputs.as_ref().is_none_or(|out| out.size != ctx.size) {
            self.outputs = Some(SpatialOutputs::new(ctx.gpu, ctx.size));
        }
        let default = RestirGiTuned::DEFAULT;
        let tuned = RestirGiTuned {
            spatial_samples: cx.value("restirSpatialSamples", f64::from(default.spatial_samples))
                as u32,
            spatial_radius: cx.value("restirSpatialRadius", f64::from(default.spatial_radius))
                as f32,
            normal_cos: cx.value("restirNormalCos", f64::from(default.normal_cos)) as f32,
            ..default
        };
        let m_cap = GiSettings::DEFAULT.restir_m_cap;
        let bytes = params_bytes(self.frame, false, false, m_cap, tuned);
        if let Some(outputs) = self.outputs.as_ref() {
            ctx.gpu.queue().write_buffer(&outputs.params, 0, &bytes);
            let (w, h) = outputs.size;
            let desc = ImageDesc::new(ImageDim::D2, IRRADIANCE_FORMAT, w, h, 1, 1);
            ctx.resources.publish_buffer(
                cx.entry,
                "reservoirs",
                &outputs.reservoirs,
                reservoir_bytes(outputs.size),
            );
            ctx.resources
                .publish(cx.entry, "irradiance", &outputs.irradiance, desc);
            published.push(("irradiance", desc));
        }
        self.frame = self.frame.wrapping_add(1);
        Ok(())
    }

    /// Records the spatial entry over the viewport, reading the `reservoirs` input buffer at
    /// binding 11. False (nothing recorded) without that buffer, the frame buffer, group 1, or the
    /// built pipelines and outputs.
    fn record(&mut self, cx: &StageCx<'_>, ctx: &mut RecordCtx<'_>) -> Result<bool, RaaError> {
        let Some((reservoirs, _)) = cx.buffer(ctx.resources, "reservoirs") else {
            return Ok(false);
        };
        let Some(frame) = cx.shared.frame_buffer() else {
            return Ok(false);
        };
        let Some(group1) = cx.shared.compute_group1(ctx.lighting) else {
            return Ok(false);
        };
        let (Some((pipelines, _)), Some(outputs)) =
            (self.pipelines.as_ref(), self.outputs.as_ref())
        else {
            return Ok(false);
        };
        let views = GBufferViews::resolve(ctx.targets, ctx.resources, cx.entry);
        let bindings = group(
            ctx.gpu.device(),
            "raa-gi.stage.restir-spatial",
            &pipelines.spatial_layout,
            &[
                (0, frame.as_entire_binding()),
                (1, texture_resource(&views.depth)),
                (2, texture_resource(&views.normal)),
                (11, reservoirs.as_entire_binding()),
                (12, outputs.params.as_entire_binding()),
                (13, texture_resource(&outputs.irradiance)),
                (15, outputs.reservoirs.as_entire_binding()),
            ],
        );
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa-gi.stage.restir-spatial"),
                timestamp_writes: None,
            });
        pass.set_bind_group(1, &group1, &[]);
        pass.set_bind_group(2, &bindings, &[]);
        pass.set_pipeline(&pipelines.spatial);
        dispatch(&mut pass, outputs.size.0, outputs.size.1);
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use raa_core::testing::skip;
    use raa_lit::{LitOptions, LitRenderer};
    use raa_pipeline::{FeaturePass, LightingResources, Pipeline, ResourceTable, Targets};

    use super::*;
    use crate::error::GI_INVALID;
    use crate::parts::GiShared;
    use crate::stages::{GiStageHandle, GiStageKind, install_stage};

    /// Installs `kind` as `t`, attaches a shared state with a 432-B frame buffer and a lit handle,
    /// and takes the pass out of the pipeline. No input is bound and no other pass ran.
    fn attached(
        gpu: &GpuContext,
        kind: GiStageKind,
    ) -> (GiStageHandle, Box<dyn FeaturePass>, ResourceTable) {
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, format, false)));
        let handle = install_stage(gpu, &pipeline, kind, "t").expect("install");
        let shared = GiShared::new(gpu);
        let frame = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("restir.test.frame"),
            size: 432,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        shared.set_frame(Some(frame));
        let lit = LitRenderer::new(gpu, &pipeline, LitOptions::default()).expect("lit renderer");
        handle.attach_shared(shared, lit.handle());
        let pass = pipeline
            .borrow_mut()
            .remove(&handle.entry())
            .expect("the stage pass");
        (handle, pass, ResourceTable::new())
    }

    /// One prepare of `pass` on an 8x8 viewport, with `frame` as the pipeline's frame index.
    fn prepare_once(
        gpu: &GpuContext,
        pass: &mut dyn FeaturePass,
        table: &ResourceTable,
        frame: u64,
    ) {
        let targets = Targets::new(gpu, (8, 8));
        let mut lighting = LightingResources::new(gpu);
        let mut ctx = PrepareCtx {
            gpu,
            size: (8, 8),
            frame,
            time_s: 0.0,
            dt_s: 0.0,
            targets: &targets,
            lighting: &mut lighting,
            resources: table,
        };
        pass.resize(gpu, &targets);
        pass.prepare(&mut ctx).expect("prepare");
    }

    #[test]
    fn initial_publishes_the_reservoirs_of_the_viewport() {
        let test = "initial_publishes_the_reservoirs_of_the_viewport";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let (handle, mut pass, table) = attached(&gpu, GiStageKind::RestirInitial);
        prepare_once(&gpu, pass.as_mut(), &table, 0);
        let bytes = RESTIR_GI_RESERVOIR_BYTES * 8 * 8;
        let found = table.buffer("gi.restir.initial#t", "reservoirs");
        assert_eq!(found.map(|(_, size)| size), Some(bytes));
        assert!(handle.published().is_empty(), "buffers are not listed");
    }

    #[test]
    fn temporal_publishes_reservoirs_and_irradiance() {
        let test = "temporal_publishes_reservoirs_and_irradiance";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let (handle, mut pass, table) = attached(&gpu, GiStageKind::RestirTemporal);
        prepare_once(&gpu, pass.as_mut(), &table, 0);
        let bytes = RESTIR_GI_RESERVOIR_BYTES * 8 * 8;
        let found = table.buffer("gi.restir.temporal#t", "reservoirs");
        assert_eq!(found.map(|(_, size)| size), Some(bytes));
        let desc = ImageDesc::new(ImageDim::D2, IRRADIANCE_FORMAT, 8, 8, 1, 1);
        assert_eq!(handle.published(), vec![("irradiance", desc)]);
        let found = table.get("gi.restir.temporal#t", "irradiance");
        assert_eq!(found.map(|image| image.desc), Some(desc));
    }

    #[test]
    fn temporal_alternates_its_reservoir_buffers() {
        let test = "temporal_alternates_its_reservoir_buffers";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let (_handle, mut pass, table) = attached(&gpu, GiStageKind::RestirTemporal);
        let key = "gi.restir.temporal#t";
        prepare_once(&gpu, pass.as_mut(), &table, 0);
        let first = table.buffer(key, "reservoirs").map(|(buffer, _)| buffer);
        prepare_once(&gpu, pass.as_mut(), &table, 1);
        let second = table.buffer(key, "reservoirs").map(|(buffer, _)| buffer);
        assert!(first.is_some(), "the first prepare publishes a buffer");
        assert_ne!(
            first, second,
            "consecutive prepares write different buffers"
        );
    }

    /// No input is bound: prepare publishes the 8x8 reservoirs and the Rgba16Float irradiance.
    #[test]
    fn spatial_publishes_reservoirs_and_irradiance() {
        let test = "spatial_publishes_reservoirs_and_irradiance";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let (handle, mut pass, table) = attached(&gpu, GiStageKind::RestirSpatial);
        prepare_once(&gpu, pass.as_mut(), &table, 0);
        let bytes = RESTIR_GI_RESERVOIR_BYTES * 8 * 8;
        let found = table.buffer("gi.restir.spatial#t", "reservoirs");
        assert_eq!(found.map(|(_, size)| size), Some(bytes));
        let desc = ImageDesc::new(ImageDim::D2, IRRADIANCE_FORMAT, 8, 8, 1, 1);
        assert_eq!(handle.published(), vec![("irradiance", desc)]);
        let found = table.get("gi.restir.spatial#t", "irradiance");
        assert_eq!(found.map(|image| image.desc), Some(desc));
    }

    #[test]
    fn restir_m_cap_zero_is_refused() {
        let test = "restir_m_cap_zero_is_refused";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let (handle, _pass, _table) = attached(&gpu, GiStageKind::RestirTemporal);
        let refused = handle.set_values(&[("restirMCap".to_string(), 0.0)]);
        assert!(matches!(&refused, Err(error) if error.code() == GI_INVALID));
        let accepted = handle.set_values(&[("restirMCap".to_string(), 20.0)]);
        assert!(accepted.is_ok());
    }
}
