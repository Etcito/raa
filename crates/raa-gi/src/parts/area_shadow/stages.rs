//! The area-shadow stages `area.trace` and `area.temporal` (contract 'rg-gi-stage-api', card 8/11).
//! `AreaTraceStage` traces one stratified shadow ray per pixel and slot into its `raw` array and
//! publishes this frame's linear depth as `linearDepth`. `AreaTemporalStage` reprojects the
//! `history` pair into `filtered` and copies the bound `linearDepth` into `depth`. Both read the
//! stochastic frame through [`area_frame`], run only in the Stratified mode with a slot limit
//! above 0, and publish nothing otherwise. The GI pyramid is the bound `pyramid` input of the trace.
// The types of `area_frame` are read by the cards 9/11 and 10/11, not by these two stages.
#![allow(dead_code)]

use raa_core::{GpuContext, RaaError};
use raa_lit::{StochasticMode, table_limit};
use raa_pipeline::{
    ImageDesc, ImageDim, MAX_STOCHASTIC_LIGHTS, PrepareCtx, RecordCtx, validate_tunable,
};

use super::cpu;
use super::passes::{
    AREA_ANTI_LAG, AREA_DISPATCH_SIZE, AREA_PARAMS_SIZE, AREA_TRACE_ENTRY, AreaShadowPipelines,
    AreaShadowWork, SDF_VOLUME_SIZE, trace_module_wgsl, tuned_pipeline,
};
use crate::parts::GBufferViews;
use crate::parts::screen::passes::{dispatch, layout, module};
use crate::stages::{StageCx, StageImpl};
use crate::tunables;

/// The format of the raw visibility array and of the filtered array (`Rgba16Float`).
const RAW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The format of the linear depth of the trace and of the depth of the temporal stage.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// The rows of `raw` and `filtered` for a slot limit: `cpu::raw_layers`.
pub(crate) fn area_layers(limit: u32) -> u32 {
    cpu::raw_layers(limit)
}

/// The per-stage frame state of [`area_frame`]: the light generations seen by the last prepare
/// and the frame counter the stage writes into its params.
#[derive(Default)]
pub(crate) struct AreaFrameState {
    generations: [u32; MAX_STOCHASTIC_LIGHTS],
    frame_index: u32,
}

/// The stochastic frame of one prepare: the slot limit (at most 8) and its history groups, the
/// mode, the light generations, whether they differ from the last prepare, and the frame counter.
pub(crate) struct AreaFrame {
    pub(super) limit: u32,
    pub(super) groups: u32,
    pub(super) mode: StochasticMode,
    pub(super) generations: [u32; MAX_STOCHASTIC_LIGHTS],
    pub(super) generations_changed: bool,
    pub(super) frame_index: u32,
}

/// Handoff: raa-gi stages 9/11: area.spatial and area.mask stages. `area_frame(cx, state)` (this
/// file) is the entry point: each stage of 9/11 keeps an `AreaFrameState` and calls it at the
/// start of `prepare`. Today it reads
/// `cx.lit.area_stochastic()`, takes `limit = table_limit(..).min(8)` and `groups =
/// cpu::groups_for(limit)`, advances `state.frame_index`, compares the light generations with
/// `state.generations` and stores the new ones. It returns None when the limit is 0. It does not
/// filter by mode.
/// Handoff: raa-gi stages 10/11: area ReSTIR stages. Each ReSTIR stage keeps an `AreaFrameState`
/// and calls `area_frame(cx, state)` at the start of prepare, with the same behaviour and the same
/// `AreaFrame` result. The ReSTIR mode check is the caller's: `AreaFrame.mode` is the mode of the
/// frame.
pub(crate) fn area_frame(cx: &StageCx<'_>, state: &mut AreaFrameState) -> Option<AreaFrame> {
    let frame = cx.lit.area_stochastic();
    let limit = table_limit(&frame.table).min(8);
    let generations_changed = frame.generations != state.generations;
    state.generations = frame.generations;
    state.frame_index = state.frame_index.wrapping_add(1);
    if limit == 0 {
        return None;
    }
    Some(AreaFrame {
        limit,
        groups: cpu::groups_for(limit),
        mode: frame.mode,
        generations: frame.generations,
        generations_changed,
        frame_index: state.frame_index,
    })
}

/// The area pipelines of one stage: built over the GI compute group 1 layout at the first
/// prepare and refreshed with `cx.tunables()` when the values generation moves.
fn area_pipelines<'a>(
    slot: &'a mut Option<(AreaShadowPipelines, u64)>,
    gpu: &GpuContext,
    cx: &StageCx<'_>,
) -> &'a mut AreaShadowPipelines {
    let (pipelines, generation) = slot.get_or_insert_with(|| {
        let layout = cx.shared.compute_group1_layout();
        (
            AreaShadowPipelines::tuned(gpu, &layout, &cx.tunables()),
            cx.values_generation,
        )
    });
    if *generation != cx.values_generation {
        pipelines.refresh(gpu, &cx.tunables());
        *generation = cx.values_generation;
    }
    pipelines
}

/// The `raa_gi_area_trace` pipeline over the empty group 0, the GI group 1 and the trace group-2
/// layout `trace_layout` (the module and entry of `AreaShadowPipelines`).
fn trace_pipeline(
    gpu: &GpuContext,
    group1: &wgpu::BindGroupLayout,
    trace_layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let device = gpu.device();
    let empty = layout(device, "raa-gi.area-trace.empty", &[]);
    let shader = module(device, "raa-gi.area-trace", &trace_module_wgsl());
    tuned_pipeline(
        device,
        "raa-gi.area-trace",
        &shader,
        AREA_TRACE_ENTRY,
        &[Some(&empty), Some(group1), Some(trace_layout)],
        &[],
    )
}

/// A texture of `layers` layers and one mip and sample.
pub(super) fn texture(
    gpu: &GpuContext,
    label: &str,
    size: (u32, u32),
    layers: u32,
    dimension: wgpu::TextureDimension,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension,
        format,
        usage,
        view_formats: &[],
    })
}

/// The storage and sampled usages of the stage textures.
pub(super) fn sampled() -> wgpu::TextureUsages {
    wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING
}

/// A `UNIFORM | COPY_DST` buffer of `size` bytes, zero until written.
pub(super) fn uniform(gpu: &GpuContext, label: &str, size: u64) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The D2 view of one array layer of a texture (`layer_view` of resources.rs).
fn layer_view(texture: &wgpu::Texture, layer: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        base_array_layer: layer,
        array_layer_count: Some(1),
        dimension: Some(wgpu::TextureViewDimension::D2),
        ..Default::default()
    })
}

/// A bind group of `layout` from (binding, resource) pairs.
pub(super) fn bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    bindings: &[(u32, wgpu::BindingResource<'_>)],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry<'_>> = bindings
        .iter()
        .map(|(binding, resource)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: resource.clone(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raa-gi.area-stage"),
        layout,
        entries: &entries,
    })
}

/// The 16-B `AreaDispatchGpu` of one history group: (index, usable, pairs, limit), little-endian.
fn dispatch_bytes(lanes: [u32; 4]) -> [u8; 16] {
    let mut bytes = [0u8; AREA_DISPATCH_SIZE as usize];
    for (lane, value) in lanes.into_iter().enumerate() {
        bytes[lane * 4..lane * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// The history pairs of group `index` of `groups`: two per group, the last one possibly alone.
fn pairs(groups: u32, index: u32) -> u32 {
    groups.saturating_sub(2 * index).min(2)
}

/// The trace outputs of one (viewport, raw layers) key: the raw array, the linear depth the
/// trace writes at binding 6, the 16-B params, and the dummies of the optional inputs (a 1x1
/// pyramid at depth 1 and a zero 1x1x1 SDF field with its 64-B uniform).
struct TraceOutputs {
    key: ((u32, u32), u32),
    raw: wgpu::TextureView,
    depth: wgpu::TextureView,
    params: wgpu::Buffer,
    dummy_pyramid: wgpu::TextureView,
    dummy_sdf_field: wgpu::TextureView,
    dummy_sdf_uniform: wgpu::Buffer,
}

impl TraceOutputs {
    fn new(gpu: &GpuContext, key: ((u32, u32), u32)) -> TraceOutputs {
        let ((width, height), layers) = key;
        let copy_out = wgpu::TextureUsages::COPY_SRC;
        let raw = texture(
            gpu,
            "raa-gi.area-trace.raw",
            (width, height),
            layers,
            wgpu::TextureDimension::D2,
            RAW_FORMAT,
            sampled() | copy_out,
        );
        let depth = texture(
            gpu,
            "raa-gi.area-trace.depth",
            (width, height),
            1,
            wgpu::TextureDimension::D2,
            DEPTH_FORMAT,
            sampled() | copy_out,
        );
        let pyramid = texture(
            gpu,
            "raa-gi.area-trace.pyramid-dummy",
            (1, 1),
            1,
            wgpu::TextureDimension::D2,
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
            "raa-gi.area-trace.sdf-dummy",
            (1, 1),
            1,
            wgpu::TextureDimension::D3,
            wgpu::TextureFormat::R8Snorm,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let view =
            |texture: &wgpu::Texture| texture.create_view(&wgpu::TextureViewDescriptor::default());
        TraceOutputs {
            key,
            raw: view(&raw),
            depth: view(&depth),
            params: uniform(gpu, "raa-gi.area-trace.params", AREA_PARAMS_SIZE),
            dummy_pyramid: view(&pyramid),
            dummy_sdf_field: view(&sdf),
            dummy_sdf_uniform: uniform(gpu, "raa-gi.area-trace.sdf-uniform", SDF_VOLUME_SIZE),
        }
    }
}

/// `area.trace`: one stratified shadow ray per pixel and slot. It publishes `raw` (the raw
/// visibility array of `area_layers(limit)` layers) and `linearDepth` (this frame's linear depth,
/// R32Float), and records only the trace dispatch; the denoise is `area.temporal`.
#[derive(Default)]
pub(crate) struct AreaTraceStage {
    area: AreaFrameState,
    pipelines: Option<(AreaShadowPipelines, u64)>,
    trace: Option<wgpu::ComputePipeline>,
    outputs: Option<TraceOutputs>,
    sdf: Option<(wgpu::TextureView, wgpu::Buffer)>,
    ready: bool,
}

impl StageImpl for AreaTraceStage {
    fn param_keys(&self) -> &'static [&'static str] {
        &[]
    }

    fn check_value(&self, key: &str, _value: f64) -> Result<(), String> {
        Err(format!("`{key}` is not a parameter of area.trace"))
    }

    /// Reallocates the outputs when the viewport or the layer count changed, writes the frame
    /// counter into the params, fetches the attached SDF volume and publishes `raw` and
    /// `linearDepth` with their real layer count. Publishes nothing unless the frame is
    /// Stratified with a limit above 0.
    fn prepare(
        &mut self,
        cx: &StageCx<'_>,
        ctx: &mut PrepareCtx<'_>,
        published: &mut Vec<(&'static str, ImageDesc)>,
    ) -> Result<(), RaaError> {
        self.ready = false;
        let Some(frame) = area_frame(cx, &mut self.area) else {
            return Ok(());
        };
        if frame.mode != StochasticMode::Stratified {
            return Ok(());
        }
        let pipelines = area_pipelines(&mut self.pipelines, ctx.gpu, cx);
        if self.trace.is_none() {
            let group1 = cx.shared.compute_group1_layout();
            self.trace = Some(trace_pipeline(ctx.gpu, &group1, &pipelines.trace_layout));
        }
        let layers = area_layers(frame.limit);
        let key = (ctx.size, layers);
        if self.outputs.as_ref().is_none_or(|out| out.key != key) {
            self.outputs = Some(TraceOutputs::new(ctx.gpu, key));
        }
        let Some(outputs) = self.outputs.as_ref() else {
            return Ok(());
        };
        ctx.gpu
            .queue()
            .write_buffer(&outputs.params, 0, &frame.frame_index.to_le_bytes());
        self.sdf = cx.shared.trace().and_then(|handle| {
            let mut scene = handle.try_borrow_mut().ok()?;
            let volume = scene.volume(ctx.gpu)?;
            Some((volume.view.clone(), volume.uniform.clone()))
        });
        let (width, height) = ctx.size;
        let raw = ImageDesc::new(ImageDim::D2Array, RAW_FORMAT, width, height, layers, 1);
        let linear = ImageDesc::new(ImageDim::D2, DEPTH_FORMAT, width, height, 1, 1);
        ctx.resources.publish(cx.entry, "raw", &outputs.raw, raw);
        ctx.resources
            .publish(cx.entry, "linearDepth", &outputs.depth, linear);
        published.push(("raw", raw));
        published.push(("linearDepth", linear));
        self.ready = true;
        Ok(())
    }

    /// Records the trace over the viewport: group 1 set, the trace pipeline, then the trace group
    /// (`GBufferViews::resolve`, the bound `pyramid` or its dummy, the SDF inputs or their
    /// dummies, the slot table and the params). False, with nothing recorded, unless the latest
    /// prepare published.
    fn record(&mut self, cx: &StageCx<'_>, ctx: &mut RecordCtx<'_>) -> Result<bool, RaaError> {
        if !self.ready {
            return Ok(false);
        }
        let (Some(outputs), Some((pipelines, _)), Some(trace)) = (
            self.outputs.as_ref(),
            self.pipelines.as_ref(),
            self.trace.as_ref(),
        ) else {
            return Ok(false);
        };
        let (Some(frame), Some(group1)) = (
            cx.shared.frame_buffer(),
            cx.shared.compute_group1(ctx.lighting),
        ) else {
            return Ok(false);
        };
        let views = GBufferViews::resolve(ctx.targets, ctx.resources, cx.entry);
        let pyramid = ctx.resources.input(cx.entry, "pyramid");
        let pyramid = pyramid
            .as_ref()
            .map_or(&outputs.dummy_pyramid, |image| &image.view);
        let (sdf_field, sdf_uniform) = match self.sdf.as_ref() {
            Some((view, buffer)) => (view, buffer),
            None => (&outputs.dummy_sdf_field, &outputs.dummy_sdf_uniform),
        };
        let trace_group = bind_group(
            ctx.gpu.device(),
            &pipelines.trace_layout,
            &[
                (0, frame.as_entire_binding()),
                (1, wgpu::BindingResource::TextureView(&views.depth)),
                (2, wgpu::BindingResource::TextureView(&views.normal)),
                (4, wgpu::BindingResource::TextureView(pyramid)),
                (5, wgpu::BindingResource::TextureView(&outputs.raw)),
                (6, wgpu::BindingResource::TextureView(&outputs.depth)),
                (8, ctx.lighting.area_stochastic_buffer().as_entire_binding()),
                (9, outputs.params.as_entire_binding()),
                (10, wgpu::BindingResource::TextureView(sdf_field)),
                (11, sdf_uniform.as_entire_binding()),
            ],
        );
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa-gi.area-trace"),
                timestamp_writes: None,
            });
        pass.set_bind_group(1, &group1, &[]);
        pass.set_pipeline(trace);
        pass.set_bind_group(2, &trace_group, &[]);
        let (width, height) = outputs.key.0;
        dispatch(&mut pass, width, height);
        Ok(true)
    }
}

/// The temporal outputs of one (viewport, raw layers) key: the filtered array, the R32Float
/// depth copied from `linearDepth`, the four 16-B dispatch records (one per history group), and
/// the 1x1 dummies bound in place of an invalid history pair.
struct TemporalOutputs {
    key: ((u32, u32), u32),
    filtered: wgpu::TextureView,
    depth: wgpu::TextureView,
    dispatch: [wgpu::Buffer; 4],
    dummy_history: wgpu::TextureView,
    dummy_depth: wgpu::TextureView,
}

impl TemporalOutputs {
    fn new(gpu: &GpuContext, key: ((u32, u32), u32)) -> TemporalOutputs {
        let ((width, height), layers) = key;
        let copy = wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST;
        let filtered = texture(
            gpu,
            "raa-gi.area-temporal.filtered",
            (width, height),
            layers,
            wgpu::TextureDimension::D2,
            RAW_FORMAT,
            sampled() | wgpu::TextureUsages::COPY_SRC,
        );
        let depth = texture(
            gpu,
            "raa-gi.area-temporal.depth",
            (width, height),
            1,
            wgpu::TextureDimension::D2,
            DEPTH_FORMAT,
            sampled() | copy,
        );
        let dummy_history = texture(
            gpu,
            "raa-gi.area-temporal.history-dummy",
            (1, 1),
            1,
            wgpu::TextureDimension::D2,
            RAW_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let dummy_depth = texture(
            gpu,
            "raa-gi.area-temporal.depth-dummy",
            (1, 1),
            1,
            wgpu::TextureDimension::D2,
            DEPTH_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let view =
            |texture: &wgpu::Texture| texture.create_view(&wgpu::TextureViewDescriptor::default());
        TemporalOutputs {
            key,
            filtered: view(&filtered),
            depth: view(&depth),
            dispatch: std::array::from_fn(|_| {
                uniform(gpu, "raa-gi.area-temporal.dispatch", AREA_DISPATCH_SIZE)
            }),
            dummy_history: view(&dummy_history),
            dummy_depth: view(&dummy_depth),
        }
    }
}

/// What the cached temporal bind groups were built from: the viewport, the group count, the
/// usable-history bits, and every bound view and buffer.
#[derive(PartialEq)]
struct TemporalKey {
    size: (u32, u32),
    groups: u32,
    usable: u32,
    motion: wgpu::TextureView,
    raw: wgpu::TextureView,
    linear: wgpu::TextureView,
    history: Option<wgpu::TextureView>,
    depth_history: Option<wgpu::TextureView>,
    frame: wgpu::Buffer,
}

/// The frame of the latest temporal prepare that published: the slot limit, its history groups,
/// and the slot generations for the samples recorded this frame.
#[derive(Clone, Copy)]
struct TemporalFrame {
    limit: u32,
    groups: u32,
    generations: [u32; MAX_STOCHASTIC_LIGHTS],
}

/// `area.temporal`: reprojects the history pair and blends the raw visibility into `filtered`.
/// `prepare` allocates and publishes `filtered` and `depth`. `record` binds one layer of `raw`,
/// `history` and `filtered` per history group, writes each group's dispatch record (usable only
/// when both histories are bound, the history has the layer, the camera is valid and the light
/// generation of that slot did not change), records the temporal dispatches of `record_denoise`, then copies
/// the bound `linearDepth` into `depth`.
#[derive(Default)]
pub(crate) struct AreaTemporalStage {
    area: AreaFrameState,
    frame: Option<TemporalFrame>,
    submitted_generations: Option<[u32; MAX_STOCHASTIC_LIGHTS]>,
    pipelines: Option<(AreaShadowPipelines, u64)>,
    outputs: Option<TemporalOutputs>,
    binds: Option<(TemporalKey, Vec<wgpu::BindGroup>)>,
}

impl StageImpl for AreaTemporalStage {
    fn param_keys(&self) -> &'static [&'static str] {
        &["temporalAlpha", "depthTolerance", "antiLag"]
    }

    /// Validate the staged temporal controls against the GI tunable catalog.
    fn check_value(&self, key: &str, value: f64) -> Result<(), String> {
        let specs = if key == AREA_ANTI_LAG.key {
            std::slice::from_ref(&AREA_ANTI_LAG)
        } else {
            tunables()
        };
        validate_tunable(specs, key, value).map(|_| ())
    }

    /// Allocates `filtered` and `depth` for the viewport and the layer count, and publishes them.
    /// Publishes nothing unless the frame is Stratified with a limit above 0.
    fn prepare(
        &mut self,
        cx: &StageCx<'_>,
        ctx: &mut PrepareCtx<'_>,
        published: &mut Vec<(&'static str, ImageDesc)>,
    ) -> Result<(), RaaError> {
        self.frame = None;
        let Some(frame) = area_frame(cx, &mut self.area) else {
            return Ok(());
        };
        if frame.mode != StochasticMode::Stratified {
            return Ok(());
        }
        area_pipelines(&mut self.pipelines, ctx.gpu, cx);
        let layers = area_layers(frame.limit);
        let key = (ctx.size, layers);
        if self.outputs.as_ref().is_none_or(|out| out.key != key) {
            self.outputs = Some(TemporalOutputs::new(ctx.gpu, key));
            self.binds = None;
        }
        let Some(outputs) = self.outputs.as_ref() else {
            return Ok(());
        };
        let (width, height) = ctx.size;
        let filtered = ImageDesc::new(ImageDim::D2Array, RAW_FORMAT, width, height, layers, 1);
        let depth = ImageDesc::new(ImageDim::D2, DEPTH_FORMAT, width, height, 1, 1);
        ctx.resources
            .publish(cx.entry, "filtered", &outputs.filtered, filtered);
        ctx.resources
            .publish(cx.entry, "depth", &outputs.depth, depth);
        published.push(("filtered", filtered));
        published.push(("depth", depth));
        self.frame = Some(TemporalFrame {
            limit: frame.limit,
            groups: frame.groups,
            generations: frame.generations,
        });
        Ok(())
    }

    /// Writes the dispatch records, rebuilds the per-group bind groups when a bound view changed,
    /// records the temporal dispatches and the `linearDepth` copy. False, with nothing recorded,
    /// unless the latest prepare published and `raw` has at least as many layers as groups.
    fn record(&mut self, cx: &StageCx<'_>, ctx: &mut RecordCtx<'_>) -> Result<bool, RaaError> {
        let (Some(frame), Some((pipelines, _)), Some(outputs)) =
            (self.frame, self.pipelines.as_ref(), self.outputs.as_ref())
        else {
            return Ok(false);
        };
        let (Some(raw), Some(linear)) = (
            ctx.resources.input(cx.entry, "raw"),
            ctx.resources.input(cx.entry, "linearDepth"),
        ) else {
            return Ok(false);
        };
        if raw.desc.layers < frame.groups {
            return Ok(false);
        }
        let history = ctx.resources.input(cx.entry, "history");
        let depth_history = ctx.resources.input(cx.entry, "depthHistory");
        let previous = self.submitted_generations;
        let camera = cx.shared.camera_valid() && previous.is_some();
        let usable = |group: u32| {
            camera
                && history
                    .as_ref()
                    .is_some_and(|image| image.desc.layers > group)
                && depth_history.is_some()
        };
        let mask = (0..frame.groups)
            .filter(|group| usable(*group))
            .fold(0, |mask, group| mask | (1 << group));
        let (width, height) = outputs.key.0;
        for (group, buffer) in outputs
            .dispatch
            .iter()
            .enumerate()
            .take(frame.groups as usize)
        {
            let group = group as u32;
            let mut flags = u32::from(usable(group));
            if let Some(generations) = previous {
                flags |= cpu::slot_rejection_bits(&generations, &frame.generations, group as usize);
            }
            if cx.value(AREA_ANTI_LAG.key, AREA_ANTI_LAG.default) != 0.0 {
                flags |= 8;
            }
            let lanes = [group, flags, pairs(frame.groups, group), frame.limit];
            ctx.gpu
                .queue()
                .write_buffer(buffer, 0, &dispatch_bytes(lanes));
        }

        let views = GBufferViews::resolve(ctx.targets, ctx.resources, cx.entry);
        let frame_buffer = cx.lit.frame_buffer();
        let key = TemporalKey {
            size: outputs.key.0,
            groups: frame.groups,
            usable: mask,
            motion: views.motion.clone(),
            raw: raw.view.clone(),
            linear: linear.view.clone(),
            history: history.as_ref().map(|image| image.view.clone()),
            depth_history: depth_history.as_ref().map(|image| image.view.clone()),
            frame: frame_buffer.clone(),
        };
        if self.binds.as_ref().is_none_or(|(stored, _)| *stored != key) {
            let groups = (0..frame.groups)
                .map(|group| {
                    let (history_view, depth_view) =
                        match (history.as_ref(), depth_history.as_ref()) {
                            (Some(past), Some(depth)) if usable(group) => {
                                (layer_view(past.view.texture(), group), depth.view.clone())
                            }
                            _ => (outputs.dummy_history.clone(), outputs.dummy_depth.clone()),
                        };
                    let raw_layer = layer_view(raw.view.texture(), group);
                    let filtered_layer = layer_view(outputs.filtered.texture(), group);
                    bind_group(
                        ctx.gpu.device(),
                        &pipelines.temporal_layout,
                        &[
                            (0, frame_buffer.as_entire_binding()),
                            (1, wgpu::BindingResource::TextureView(&linear.view)),
                            (2, wgpu::BindingResource::TextureView(&views.motion)),
                            (3, wgpu::BindingResource::TextureView(&raw_layer)),
                            (4, wgpu::BindingResource::TextureView(&history_view)),
                            (5, wgpu::BindingResource::TextureView(&depth_view)),
                            (6, wgpu::BindingResource::TextureView(&filtered_layer)),
                            (7, outputs.dispatch[group as usize].as_entire_binding()),
                            (
                                8,
                                wgpu::BindingResource::Sampler(ctx.lighting.linear_clamp()),
                            ),
                        ],
                    )
                })
                .collect();
            self.binds = Some((key, groups));
        }
        let Some((_, groups)) = self.binds.as_ref() else {
            return Ok(false);
        };
        let Some(first) = groups.first() else {
            return Ok(false);
        };
        // `record_denoise` reads only `temporal`; `trace` is not used and takes the first group.
        let work = AreaShadowWork {
            trace: first,
            temporal: groups,
            spatial_h: &[],
            spatial_v: &[],
            viewport: (width, height),
        };
        {
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("raa-gi.area-temporal"),
                    timestamp_writes: None,
                });
            pipelines.record_denoise(&mut pass, &work);
        }
        ctx.encoder.copy_texture_to_texture(
            linear.view.texture().as_image_copy(),
            outputs.depth.texture().as_image_copy(),
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        Ok(true)
    }

    fn set_history_valid(&mut self, valid: bool) {
        self.submitted_generations = self.frame.filter(|_| valid).map(|frame| frame.generations);
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
    use raa_lit::{LitOptions, LitRenderer};
    use raa_pipeline::{FeaturePass, LightingResources, Pipeline, ResourceTable, Targets};

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
            label: Some("area.stages.test.frame"),
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

    /// One prepare of `pass` on an 8x8 viewport.
    fn prepare_once(gpu: &GpuContext, pass: &mut dyn FeaturePass, table: &ResourceTable) {
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
        pass.resize(gpu, &targets);
        pass.prepare(&mut ctx).expect("prepare");
    }

    #[test]
    fn area_layers_are_the_raw_layers_of_the_contract() {
        for (limit, layers) in [(1, 2), (2, 2), (3, 2), (8, 4)] {
            assert_eq!(area_layers(limit), cpu::raw_layers(limit), "limit {limit}");
            assert_eq!(area_layers(limit), layers, "limit {limit}");
        }
    }

    #[test]
    fn area_trace_and_temporal_publish_nothing_without_an_area_light() {
        let test = "area_trace_and_temporal_publish_nothing_without_an_area_light";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        for kind in [GiStageKind::AreaTrace, GiStageKind::AreaTemporal] {
            let (handle, mut pass, table) = attached(&gpu, kind);
            prepare_once(&gpu, pass.as_mut(), &table);
            assert!(handle.published().is_empty(), "{}", kind.type_id());
            assert!(!handle.ran(), "{}", kind.type_id());
        }
    }

    #[test]
    fn pairs_and_dispatch_lanes_follow_the_group_layout() {
        assert_eq!(
            [pairs(4, 0), pairs(4, 1), pairs(4, 2), pairs(4, 3)],
            [2, 2, 0, 0]
        );
        assert_eq!([pairs(3, 0), pairs(3, 1)], [2, 1]);
        let bytes = dispatch_bytes([1, 0, 2, 8]);
        assert_eq!(bytes[0..4], 1u32.to_le_bytes());
        assert_eq!(bytes[12..16], 8u32.to_le_bytes());
    }
}
