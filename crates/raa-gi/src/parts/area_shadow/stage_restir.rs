//! The area-shadow ReSTIR stages `area.restir.initial`, `area.restir.temporal` and
//! `area.restir.spatial` (contract 'rg-gi-stage-api', card 10/11). Each stage records one entry
//! of `RestirPipelines` (`RESTIR_INITIAL_ENTRY`, `RESTIR_TEMPORAL_ENTRY`, `RESTIR_SPATIAL_ENTRY`)
//! over the group-2 layout of `restir_group2_entries`. They run only on a ReSTIR frame with a
//! slot limit above 0 (`StochasticMode::Restir`) and publish nothing otherwise.
//!
//! Bindings. Binding 1, the current depth, takes the G-buffer depth, as
//! `RestirResources::build_group` does; no layout changes. Binding 6 is the linear depth the
//! initial entry writes (`raa_area_depth_out`), published by `area.restir.initial` as
//! `linearDepth`. Binding 5 is `raw`: only the spatial entry writes it, layer 0
//! (`restir_spatial.wgsl`), so `area.restir.spatial` publishes `raw`. Binding 14, the previous
//! depth, is the temporal stage's own zero texture: no input carries the previous linear depth,
//! so temporal reuse is rejected on depth until one is wired.
//!
//! Reservoirs. `initial` writes its reservoirs (binding 16) into `reservoirs`. `temporal` copies
//! the `initial` input into the buffer it writes, alternating two buffers, reads the buffer it
//! wrote last frame at 17 when the history is valid, and updates its output in place. `spatial`
//! reads the `reservoirs` input at 16 and writes its merged reservoirs at 17, published as its
//! own `reservoirs`.

use raa_core::{GpuContext, RaaError};
use raa_lit::StochasticMode;
use raa_pipeline::{
    ImageDesc, ImageDim, MAX_STOCHASTIC_LIGHTS, PrepareCtx, RecordCtx, create_brdf_atlas,
    validate_tunable,
};

use super::passes::SDF_VOLUME_SIZE;
use super::restir::{RESERVOIR_BYTES, RESTIR_PARAMS_SIZE, RestirPipelines};
use super::restir_cpu::{
    RESTIR_BOILING_FACTOR, RESTIR_CANDIDATES, RESTIR_M_CAP, RESTIR_NORMAL_COS,
    RESTIR_SPATIAL_RADIUS, RESTIR_SPATIAL_SAMPLES, remap_table,
};
use super::stages::{
    AreaFrameState, area_frame, area_layers, bind_group, sampled, texture, uniform,
};
use crate::parts::GBufferViews;
use crate::parts::screen::passes::dispatch;
use crate::stages::{StageCx, StageImpl};
use crate::tunables;

/// The format of the linear depth of the initial entry and of the previous depth (`R32Float`).
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
/// The format of the raw visibility array (`Rgba16Float`).
const RAW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The frame state of one ReSTIR stage: the area frame counter, the light generations of the
/// last prepare and, in tests, a forced limit (`force_running_frame_for_test`).
#[derive(Default)]
struct Gate {
    area: AreaFrameState,
    generations: [u32; MAX_STOCHASTIC_LIGHTS],
    #[cfg(test)]
    forced: Option<u32>,
}

/// One ReSTIR prepare: the slot limit, the slot remap of the light generations (`remap_table`),
/// whether the history is usable (history valid, camera valid and generations unchanged), and
/// the frame counter.
struct RestirFrame {
    limit: u32,
    remap: [u32; MAX_STOCHASTIC_LIGHTS],
    valid: bool,
    index: u32,
}

/// The ReSTIR frame of one prepare, or None unless the frame is a ReSTIR frame with a limit above
/// 0. The remap compares this frame's generations with the last prepare's.
fn restir_frame(cx: &StageCx<'_>, gate: &mut Gate) -> Option<RestirFrame> {
    #[cfg(test)]
    if let Some(limit) = gate.forced {
        return Some(RestirFrame {
            limit,
            remap: [0; MAX_STOCHASTIC_LIGHTS],
            valid: false,
            index: 1,
        });
    }
    let frame = area_frame(cx, &mut gate.area)?;
    if frame.mode != StochasticMode::Restir {
        return None;
    }
    let valid = cx.history_valid && cx.shared.camera_valid() && !frame.generations_changed;
    let remap = remap_table(&gate.generations, &frame.generations, valid);
    gate.generations = frame.generations;
    Some(RestirFrame {
        limit: frame.limit,
        remap,
        valid,
        index: frame.frame_index,
    })
}

/// The `raa_gi::tunables()` key of a parameter of these stages: the six ReSTIR rows of
/// pass.gi.area-shadow carry an `area` prefix in the store (`tunables_gi.rs` `store_key`).
fn binding_key(key: &str) -> &str {
    match key {
        "restirCandidates" => "areaRestirCandidates",
        "restirSpatialSamples" => "areaRestirSpatialSamples",
        "restirSpatialRadius" => "areaRestirSpatialRadius",
        "restirMCap" => "areaRestirMCap",
        "restirBoilingFactor" => "areaRestirBoilingFactor",
        "restirNormalCos" => "areaRestirNormalCos",
        _ => key,
    }
}

/// Checks a value of these stages against the tunable spec of its binding key.
fn check(key: &str, value: f64) -> Result<(), String> {
    validate_tunable(tunables(), binding_key(key), value).map(|_| ())
}

/// Writes the 64-B `RestirParamsGpu` as `RestirResources::write_params` does: the frame counter,
/// the history validity, the candidate and spatial sample counts, the eight remap lanes, then
/// the spatial radius, M cap, boiling factor and normal cosine. The values are this stage's
/// parameters (`cx.value`), the `restir_cpu` defaults otherwise.
fn write_params(gpu: &GpuContext, buffer: &wgpu::Buffer, cx: &StageCx<'_>, frame: &RestirFrame) {
    let int = |key: &str, default: u32| cx.value(key, f64::from(default)) as u32;
    let float = |key: &str, default: f32| cx.value(key, f64::from(default)) as f32;
    let words = [
        frame.index,
        u32::from(frame.valid),
        int("restirCandidates", RESTIR_CANDIDATES),
        int("restirSpatialSamples", RESTIR_SPATIAL_SAMPLES),
    ];
    let floats = [
        float("restirSpatialRadius", RESTIR_SPATIAL_RADIUS),
        float("restirMCap", RESTIR_M_CAP),
        float("restirBoilingFactor", RESTIR_BOILING_FACTOR),
        float("restirNormalCos", RESTIR_NORMAL_COS),
    ];
    let mut bytes = [0u8; RESTIR_PARAMS_SIZE as usize];
    for (lane, value) in words.iter().chain(frame.remap.iter()).enumerate() {
        bytes[lane * 4..lane * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    for (lane, value) in floats.iter().enumerate() {
        bytes[48 + lane * 4..52 + lane * 4].copy_from_slice(&value.to_le_bytes());
    }
    gpu.queue().write_buffer(buffer, 0, &bytes);
}

/// The ReSTIR pipelines of one stage: built over the GI compute group 1 layout at the first
/// prepare of a ReSTIR frame, and refreshed with `cx.tunables()` when the values generation moves.
fn restir_pipelines<'a>(
    slot: &'a mut Option<(RestirPipelines, u64)>,
    gpu: &GpuContext,
    cx: &StageCx<'_>,
) -> &'a RestirPipelines {
    let (pipelines, generation) = slot.get_or_insert_with(|| {
        let layout = cx.shared.compute_group1_layout();
        (
            RestirPipelines::tuned(gpu, &layout, &cx.tunables()),
            cx.values_generation,
        )
    });
    if *generation != cx.values_generation {
        pipelines.refresh(gpu, &cx.tunables());
        *generation = cx.values_generation;
    }
    pipelines
}

/// The attached SDF volume of the trace layout (bindings 10 and 11), when one is attached.
fn sdf_volume(cx: &StageCx<'_>, gpu: &GpuContext) -> Option<(wgpu::TextureView, wgpu::Buffer)> {
    let handle = cx.shared.trace()?;
    let mut scene = handle.try_borrow_mut().ok()?;
    let volume = scene.volume(gpu)?;
    Some((volume.view.clone(), volume.uniform.clone()))
}

/// The bytes of one reservoir buffer of `size`: `RESERVOIR_BYTES` per pixel.
fn reservoir_bytes(size: (u32, u32)) -> u64 {
    u64::from(size.0) * u64::from(size.1) * RESERVOIR_BYTES
}

/// A storage buffer that is copied from and to, zero until written.
fn reservoir_buffer(gpu: &GpuContext, label: &str, size: u64) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The per-viewport resources of one ReSTIR stage, keyed by the viewport and the raw layers: two
/// reservoir buffers (the temporal stage alternates them), a zero reservoir buffer for an unusable
/// history, the params, the linear depth, the previous depth, the raw array (one layer unless
/// spatial), a 1x1 pyramid at depth 1 and the dummies and LTC atlas of the trace layout.
struct Outputs {
    key: ((u32, u32), u32),
    reservoirs: [wgpu::Buffer; 2],
    spare: wgpu::Buffer,
    params: wgpu::Buffer,
    linear: wgpu::TextureView,
    previous: wgpu::TextureView,
    raw: wgpu::TextureView,
    scratch: wgpu::TextureView,
    pyramid: wgpu::TextureView,
    ltc: wgpu::TextureView,
    sdf_field: wgpu::TextureView,
    sdf_uniform: wgpu::Buffer,
}

impl Outputs {
    fn new(gpu: &GpuContext, size: (u32, u32), layers: u32) -> Outputs {
        let bytes = reservoir_bytes(size);
        let d2 = wgpu::TextureDimension::D2;
        let view =
            |texture: &wgpu::Texture| texture.create_view(&wgpu::TextureViewDescriptor::default());
        let copy = wgpu::TextureUsages::COPY_SRC;
        let linear = texture(
            gpu,
            "raa-gi.area-restir.linear-depth",
            size,
            1,
            d2,
            DEPTH_FORMAT,
            sampled() | copy,
        );
        let previous = texture(
            gpu,
            "raa-gi.area-restir.previous-depth",
            size,
            1,
            d2,
            DEPTH_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let raw = texture(
            gpu,
            "raa-gi.area-restir.raw",
            size,
            layers,
            d2,
            RAW_FORMAT,
            sampled() | copy,
        );
        let scratch = texture(
            gpu,
            "raa-gi.area-restir.scratch",
            (1, 1),
            1,
            d2,
            DEPTH_FORMAT,
            sampled(),
        );
        let pyramid = texture(
            gpu,
            "raa-gi.area-restir.pyramid-dummy",
            (1, 1),
            1,
            d2,
            DEPTH_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        gpu.queue().write_texture(
            pyramid.as_image_copy(),
            &1.0f32.to_le_bytes(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let sdf = texture(
            gpu,
            "raa-gi.area-restir.sdf-dummy",
            (1, 1),
            1,
            wgpu::TextureDimension::D3,
            wgpu::TextureFormat::R8Snorm,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let ltc = create_brdf_atlas(gpu, "raa-gi.area-restir.ltc", [1.0, 0.0]);
        Outputs {
            key: (size, layers),
            reservoirs: [
                reservoir_buffer(gpu, "raa-gi.area-restir.reservoirs-a", bytes),
                reservoir_buffer(gpu, "raa-gi.area-restir.reservoirs-b", bytes),
            ],
            spare: reservoir_buffer(gpu, "raa-gi.area-restir.spare", bytes),
            params: uniform(gpu, "raa-gi.area-restir.params", RESTIR_PARAMS_SIZE),
            linear: view(&linear),
            previous: view(&previous),
            raw: raw.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            }),
            scratch: view(&scratch),
            pyramid: view(&pyramid),
            ltc: view(&ltc),
            sdf_field: view(&sdf),
            sdf_uniform: uniform(gpu, "raa-gi.area-restir.sdf-uniform", SDF_VOLUME_SIZE),
        }
    }
}

/// The outputs of a stage for `size` and `layers`, reallocated when either changed.
fn outputs_for<'a>(
    slot: &'a mut Option<Outputs>,
    gpu: &GpuContext,
    size: (u32, u32),
    layers: u32,
) -> &'a Outputs {
    if slot.as_ref().is_none_or(|out| out.key != (size, layers)) {
        *slot = Some(Outputs::new(gpu, size, layers));
    }
    slot.get_or_insert_with(|| Outputs::new(gpu, size, layers))
}

/// The bindings one ReSTIR stage supplies itself: the depth it writes at 6, the reservoirs it
/// reads at 16 and writes at 17, and its params at 15.
struct Slots<'a> {
    depth_write: &'a wgpu::TextureView,
    current: &'a wgpu::Buffer,
    history: &'a wgpu::Buffer,
    params: &'a wgpu::Buffer,
}

/// The group-2 bind group of a ReSTIR entry on `layout` (`restir_group2_entries`), or None without
/// the GI frame buffer. The G-buffer, the GI frame, the stochastic table, the SDF volume, the lit
/// temporal frame and the LTC atlas are shared by the three stages.
fn group2(
    ctx: &RecordCtx<'_>,
    cx: &StageCx<'_>,
    layout: &wgpu::BindGroupLayout,
    out: &Outputs,
    slots: Slots<'_>,
) -> Option<wgpu::BindGroup> {
    let frame = cx.shared.frame_buffer()?;
    let lit = cx.lit.frame_buffer();
    let views = GBufferViews::resolve(ctx.targets, ctx.resources, cx.entry);
    let sdf = sdf_volume(cx, ctx.gpu);
    let (sdf_field, sdf_uniform) = sdf
        .as_ref()
        .map_or((&out.sdf_field, &out.sdf_uniform), |(field, buffer)| {
            (field, buffer)
        });
    Some(bind_group(
        ctx.gpu.device(),
        layout,
        &[
            (0, frame.as_entire_binding()),
            (1, wgpu::BindingResource::TextureView(&views.depth)),
            (2, wgpu::BindingResource::TextureView(&views.normal)),
            (4, wgpu::BindingResource::TextureView(&out.pyramid)),
            (5, wgpu::BindingResource::TextureView(&out.raw)),
            (6, wgpu::BindingResource::TextureView(slots.depth_write)),
            (8, ctx.lighting.area_stochastic_buffer().as_entire_binding()),
            (10, wgpu::BindingResource::TextureView(sdf_field)),
            (11, sdf_uniform.as_entire_binding()),
            (12, wgpu::BindingResource::TextureView(&views.motion)),
            (13, lit.as_entire_binding()),
            (14, wgpu::BindingResource::TextureView(&out.previous)),
            (15, slots.params.as_entire_binding()),
            (16, slots.current.as_entire_binding()),
            (17, slots.history.as_entire_binding()),
            (18, wgpu::BindingResource::TextureView(&out.ltc)),
            (19, wgpu::BindingResource::TextureView(&views.material)),
        ],
    ))
}

/// Records one ReSTIR entry: group 1 and group 2 set, then `pipeline` dispatched over `size`.
fn dispatch_stage(
    ctx: &mut RecordCtx<'_>,
    label: &str,
    groups: (&wgpu::BindGroup, &wgpu::BindGroup),
    pipeline: &wgpu::ComputePipeline,
    size: (u32, u32),
) {
    let mut pass = ctx
        .encoder
        .begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
    pass.set_bind_group(1, groups.0, &[]);
    pass.set_bind_group(2, groups.1, &[]);
    pass.set_pipeline(pipeline);
    dispatch(&mut pass, size.0, size.1);
}

/// `area.restir.initial`: the RIS candidates of each pixel, one visibility ray towards the kept
/// sample and the linear depth the trace writes. Publishes `reservoirs` and `linearDepth`.
#[derive(Default)]
pub(crate) struct AreaRestirInitialStage {
    gate: Gate,
    pipelines: Option<(RestirPipelines, u64)>,
    outputs: Option<Outputs>,
    ready: bool,
}

impl StageImpl for AreaRestirInitialStage {
    fn param_keys(&self) -> &'static [&'static str] {
        &["restirCandidates", "restirDepthTolerance"]
    }

    fn check_value(&self, key: &str, value: f64) -> Result<(), String> {
        check(key, value)
    }

    #[cfg(test)]
    fn force_running_frame_for_test(&mut self, limit: u32) {
        self.gate.forced = Some(limit);
    }

    /// Allocates the outputs of the viewport, writes the params and publishes `reservoirs` and
    /// `linearDepth`. Publishes nothing unless the frame is a ReSTIR frame with a limit above 0.
    fn prepare(
        &mut self,
        cx: &StageCx<'_>,
        ctx: &mut PrepareCtx<'_>,
        published: &mut Vec<(&'static str, ImageDesc)>,
    ) -> Result<(), RaaError> {
        self.ready = false;
        let Some(frame) = restir_frame(cx, &mut self.gate) else {
            return Ok(());
        };
        restir_pipelines(&mut self.pipelines, ctx.gpu, cx);
        let out = outputs_for(&mut self.outputs, ctx.gpu, ctx.size, 1);
        write_params(ctx.gpu, &out.params, cx, &frame);
        let (width, height) = ctx.size;
        let linear = ImageDesc::new(ImageDim::D2, DEPTH_FORMAT, width, height, 1, 1);
        ctx.resources.publish_buffer(
            cx.entry,
            "reservoirs",
            &out.reservoirs[0],
            reservoir_bytes(ctx.size),
        );
        ctx.resources
            .publish(cx.entry, "linearDepth", &out.linear, linear);
        published.push(("linearDepth", linear));
        self.ready = true;
        Ok(())
    }

    /// Records the initial entry over the viewport. False, with nothing recorded, unless the
    /// latest prepare published.
    fn record(&mut self, cx: &StageCx<'_>, ctx: &mut RecordCtx<'_>) -> Result<bool, RaaError> {
        let (Some(out), Some((pipelines, _))) = (self.outputs.as_ref(), self.pipelines.as_ref())
        else {
            return Ok(false);
        };
        if !self.ready {
            return Ok(false);
        }
        let Some(group1) = cx.shared.compute_group1(ctx.lighting) else {
            return Ok(false);
        };
        let slots = Slots {
            depth_write: &out.linear,
            current: &out.reservoirs[0],
            history: &out.spare,
            params: &out.params,
        };
        let Some(group2) = group2(ctx, cx, &pipelines.layout, out, slots) else {
            return Ok(false);
        };
        let groups = (&group1, &group2);
        dispatch_stage(
            ctx,
            "raa-gi.area-restir.initial",
            groups,
            pipelines.initial(),
            out.key.0,
        );
        Ok(true)
    }
}

/// `area.restir.temporal`: reuses last frame's reservoirs of its own pair of buffers when the
/// history is valid, and updates the reservoirs of the `initial` input in place.
#[derive(Default)]
pub(crate) struct AreaRestirTemporalStage {
    gate: Gate,
    pipelines: Option<(RestirPipelines, u64)>,
    outputs: Option<Outputs>,
    parity: usize,
    valid: bool,
    ready: bool,
}

impl StageImpl for AreaRestirTemporalStage {
    fn param_keys(&self) -> &'static [&'static str] {
        &["restirMCap", "restirBoilingFactor", "restirNormalCos"]
    }

    fn check_value(&self, key: &str, value: f64) -> Result<(), String> {
        check(key, value)
    }

    /// Publishes the buffer this frame writes (`reservoirs`), with the params and the history
    /// validity of the frame. Publishes nothing unless the frame is a ReSTIR frame. It publishes
    /// no image, so `published` stays empty.
    fn prepare(
        &mut self,
        cx: &StageCx<'_>,
        ctx: &mut PrepareCtx<'_>,
        _published: &mut Vec<(&'static str, ImageDesc)>,
    ) -> Result<(), RaaError> {
        self.ready = false;
        let Some(frame) = restir_frame(cx, &mut self.gate) else {
            return Ok(());
        };
        restir_pipelines(&mut self.pipelines, ctx.gpu, cx);
        let out = outputs_for(&mut self.outputs, ctx.gpu, ctx.size, 1);
        write_params(ctx.gpu, &out.params, cx, &frame);
        ctx.resources.publish_buffer(
            cx.entry,
            "reservoirs",
            &out.reservoirs[self.parity],
            reservoir_bytes(ctx.size),
        );
        self.valid = frame.valid;
        self.ready = true;
        Ok(())
    }

    /// Copies the `initial` input into the buffer this frame writes, binds last frame's buffer
    /// at 17 (or the zero buffer when the history is not valid), records the temporal entry and
    /// flips the pair. False, with nothing recorded, unless the latest prepare published and the
    /// `initial` input is bound.
    fn record(&mut self, cx: &StageCx<'_>, ctx: &mut RecordCtx<'_>) -> Result<bool, RaaError> {
        let (Some(out), Some((pipelines, _))) = (self.outputs.as_ref(), self.pipelines.as_ref())
        else {
            return Ok(false);
        };
        if !self.ready {
            return Ok(false);
        }
        let Some((initial, bytes)) = cx.buffer(ctx.resources, "initial") else {
            return Ok(false);
        };
        let Some(group1) = cx.shared.compute_group1(ctx.lighting) else {
            return Ok(false);
        };
        let (write, read) = (self.parity, 1 - self.parity);
        let history = if self.valid {
            &out.reservoirs[read]
        } else {
            &out.spare
        };
        let slots = Slots {
            depth_write: &out.scratch,
            current: &out.reservoirs[write],
            history,
            params: &out.params,
        };
        let Some(group2) = group2(ctx, cx, &pipelines.layout, out, slots) else {
            return Ok(false);
        };
        let size = bytes.min(reservoir_bytes(out.key.0));
        ctx.encoder
            .copy_buffer_to_buffer(&initial, 0, &out.reservoirs[write], 0, size);
        let groups = (&group1, &group2);
        dispatch_stage(
            ctx,
            "raa-gi.area-restir.temporal",
            groups,
            pipelines.temporal(),
            out.key.0,
        );
        self.parity = read;
        Ok(true)
    }
}

/// `area.restir.spatial`: merges each pixel's reservoir with its compatible neighbours, traces
/// the merged sample and writes layer 0 of `raw`. Publishes `raw` and `reservoirs`.
#[derive(Default)]
pub(crate) struct AreaRestirSpatialStage {
    gate: Gate,
    pipelines: Option<(RestirPipelines, u64)>,
    outputs: Option<Outputs>,
    ready: bool,
}

impl StageImpl for AreaRestirSpatialStage {
    fn param_keys(&self) -> &'static [&'static str] {
        &[
            "restirSpatialSamples",
            "restirSpatialRadius",
            "restirNormalCos",
        ]
    }

    fn check_value(&self, key: &str, value: f64) -> Result<(), String> {
        check(key, value)
    }

    /// Allocates the outputs for the raw layer count of the limit and publishes `raw`
    /// (D2Array Rgba16Float) and `reservoirs`. Publishes nothing unless the frame is a ReSTIR frame.
    fn prepare(
        &mut self,
        cx: &StageCx<'_>,
        ctx: &mut PrepareCtx<'_>,
        published: &mut Vec<(&'static str, ImageDesc)>,
    ) -> Result<(), RaaError> {
        self.ready = false;
        let Some(frame) = restir_frame(cx, &mut self.gate) else {
            return Ok(());
        };
        restir_pipelines(&mut self.pipelines, ctx.gpu, cx);
        let layers = area_layers(frame.limit);
        let out = outputs_for(&mut self.outputs, ctx.gpu, ctx.size, layers);
        write_params(ctx.gpu, &out.params, cx, &frame);
        let (width, height) = ctx.size;
        let raw = ImageDesc::new(ImageDim::D2Array, RAW_FORMAT, width, height, layers, 1);
        ctx.resources.publish(cx.entry, "raw", &out.raw, raw);
        ctx.resources.publish_buffer(
            cx.entry,
            "reservoirs",
            &out.reservoirs[0],
            reservoir_bytes(ctx.size),
        );
        published.push(("raw", raw));
        self.ready = true;
        Ok(())
    }

    /// Records the spatial entry over the viewport, reading the `reservoirs` input at 16. False,
    /// with nothing recorded, unless the latest prepare published and the input is bound.
    fn record(&mut self, cx: &StageCx<'_>, ctx: &mut RecordCtx<'_>) -> Result<bool, RaaError> {
        let (Some(out), Some((pipelines, _))) = (self.outputs.as_ref(), self.pipelines.as_ref())
        else {
            return Ok(false);
        };
        if !self.ready {
            return Ok(false);
        }
        let Some((input, _)) = cx.buffer(ctx.resources, "reservoirs") else {
            return Ok(false);
        };
        let Some(group1) = cx.shared.compute_group1(ctx.lighting) else {
            return Ok(false);
        };
        let slots = Slots {
            depth_write: &out.scratch,
            current: &input,
            history: &out.reservoirs[0],
            params: &out.params,
        };
        let Some(group2) = group2(ctx, cx, &pipelines.layout, out, slots) else {
            return Ok(false);
        };
        let groups = (&group1, &group2);
        dispatch_stage(
            ctx,
            "raa-gi.area-restir.spatial",
            groups,
            pipelines.spatial(),
            out.key.0,
        );
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use crate::parts::GiShared;
    use crate::stages::{GiStageHandle, GiStageKind, install_stage};
    use raa_core::testing::skip;
    use raa_lit::{LitOptions, LitRenderer, table_limit};
    use raa_pipeline::{FeaturePass, LightingResources, Pipeline, ResourceTable, Targets};

    /// Prepares one pass at 8x8 with no inputs bound.
    fn prepare(
        gpu: &GpuContext,
        pass: &mut dyn FeaturePass,
        table: &ResourceTable,
        frame: u64,
    ) -> Result<(), RaaError> {
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
        pass.prepare(&mut ctx)
    }

    /// Installs `kinds` on one pipeline, attaches them to a shared GI frame and a lit renderer
    /// (no area light, no SDF volume), and returns the handles with their removed passes.
    fn installed(
        gpu: &GpuContext,
        kinds: &[GiStageKind],
    ) -> (Vec<GiStageHandle>, Vec<Box<dyn FeaturePass>>) {
        let pipeline = Rc::new(RefCell::new(Pipeline::new(
            gpu,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        )));
        let handles: Vec<GiStageHandle> = kinds
            .iter()
            .map(|kind| install_stage(gpu, &pipeline, *kind, "t").expect("install"))
            .collect();
        let shared = GiShared::new(gpu);
        let frame = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-gi.area-restir.test.frame"),
            size: 432,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        shared.set_frame(Some(frame));
        let lit = LitRenderer::new(gpu, &pipeline, LitOptions::default())
            .expect("lit renderer")
            .handle();
        assert_eq!(table_limit(&lit.area_stochastic().table), 0);
        let passes = {
            let mut pipeline = pipeline.borrow_mut();
            handles
                .iter()
                .map(|handle| pipeline.remove(&handle.entry()).expect("stage pass"))
                .collect()
        };
        for handle in &handles {
            handle.attach_shared(shared.clone(), lit.clone());
        }
        (handles, passes)
    }

    #[test]
    fn restir_stages_publish_nothing_without_an_area_light() {
        let test = "restir_stages_publish_nothing_without_an_area_light";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let kinds = [
            GiStageKind::AreaRestirInitial,
            GiStageKind::AreaRestirTemporal,
            GiStageKind::AreaRestirSpatial,
        ];
        let (handles, mut passes) = installed(&gpu, &kinds);
        let table = ResourceTable::new();
        for pass in &mut passes {
            prepare(&gpu, pass.as_mut(), &table, 0).expect("prepare without area light");
        }
        for handle in &handles {
            assert!(handle.published().is_empty());
            assert!(table.buffer(&handle.entry(), "reservoirs").is_none());
        }
        assert!(table.buffer_entries().is_empty());
    }

    #[test]
    fn restir_initial_publishes_reservoirs_of_the_viewport_on_a_forced_frame() {
        let test = "restir_initial_publishes_reservoirs_of_the_viewport_on_a_forced_frame";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let (handles, mut passes) = installed(&gpu, &[GiStageKind::AreaRestirInitial]);
        handles[0].force_running_frame_for_test(1);
        let table = ResourceTable::new();
        prepare(&gpu, passes[0].as_mut(), &table, 1).expect("forced ReSTIR prepare");
        let (_, size) = table
            .buffer(&handles[0].entry(), "reservoirs")
            .expect("reservoirs buffer");
        // 8 x 8 pixels at RESERVOIR_BYTES each: the reservoir part of restir_resource_bytes.
        assert_eq!(size, 64 * RESERVOIR_BYTES);
        assert_eq!(handles[0].published().len(), 1);
        assert_eq!(handles[0].published()[0].0, "linearDepth");
    }
}
