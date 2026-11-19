//! The `area.spatial` and `area.mask` stages. Both attach through `area_frame` from
//! `area_shadow/stages.rs:66`; spatial runs the separable denoiser and mask copies its result
//! while installing the current stochastic slot table for forward shading. A mask linked straight
//! to `area.temporal` or `area.trace` (stage value `srcPairs` 1) converts their (S, U) pairs to
//! visibility S / U itself instead of copying.
#![allow(dead_code)]

use raa_core::{GpuContext, RaaError};
use raa_lit::{StochasticMode, table_limit};
use raa_pipeline::{ImageDesc, ImageDim, PrepareCtx, RecordCtx, validate_tunable};

use super::cpu;
use super::passes::{AREA_DISPATCH_SIZE, AreaShadowPipelines, AreaShadowWork};
use super::stages::{AreaFrameState, area_frame};
use crate::parts::GBufferViews;
use crate::stages::{StageCx, StageImpl};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const SPATIAL_KEYS: &[&str] = &["spatialStride", "normalPower"];

fn texture(
    gpu: &GpuContext,
    label: &str,
    size: (u32, u32),
    layers: u32,
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
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage,
        view_formats: &[],
    })
}

fn layer_view(texture: &wgpu::Texture, layer: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        base_array_layer: layer,
        array_layer_count: Some(1),
        dimension: Some(wgpu::TextureViewDimension::D2),
        ..Default::default()
    })
}

fn bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    entries: &[(u32, wgpu::BindingResource<'_>)],
) -> wgpu::BindGroup {
    let entries: Vec<_> = entries
        .iter()
        .map(|(binding, resource)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: resource.clone(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raa-gi.area-stage.spatial"),
        layout,
        entries: &entries,
    })
}

fn dispatch_bytes(lanes: [u32; 4]) -> [u8; AREA_DISPATCH_SIZE as usize] {
    let mut bytes = [0; AREA_DISPATCH_SIZE as usize];
    for (index, value) in lanes.into_iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

struct SpatialOutputs {
    size: (u32, u32),
    groups: u32,
    mask: wgpu::Texture,
    mask_view: wgpu::TextureView,
    intermediate: wgpu::Texture,
    dispatch_h: [wgpu::Buffer; 4],
    dispatch_v: [wgpu::Buffer; 2],
}

impl SpatialOutputs {
    fn new(gpu: &GpuContext, size: (u32, u32), groups: u32) -> Self {
        let mask = texture(
            gpu,
            "raa-gi.area-spatial.mask",
            size,
            2,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        );
        let intermediate = texture(
            gpu,
            "raa-gi.area-spatial.intermediate",
            size,
            groups,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let make_dispatch = || {
            gpu.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("raa-gi.area-spatial.dispatch"),
                size: AREA_DISPATCH_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Self {
            size,
            groups,
            mask_view: mask.create_view(&Default::default()),
            mask,
            intermediate,
            dispatch_h: std::array::from_fn(|_| make_dispatch()),
            dispatch_v: std::array::from_fn(|_| make_dispatch()),
        }
    }

    fn publish(
        &self,
        cx: &StageCx<'_>,
        ctx: &PrepareCtx<'_>,
        published: &mut Vec<(&'static str, ImageDesc)>,
        port: &'static str,
    ) {
        let desc = ImageDesc::new(ImageDim::D2Array, FORMAT, self.size.0, self.size.1, 2, 1);
        ctx.resources.publish(cx.entry, port, &self.mask_view, desc);
        published.push((port, desc));
    }
}

#[derive(PartialEq)]
struct SpatialKey {
    size: (u32, u32),
    groups: u32,
    limit: u32,
    src: wgpu::TextureView,
    linear_depth: wgpu::TextureView,
    normal: wgpu::TextureView,
}

#[derive(Clone, Copy)]
struct SpatialFrame {
    groups: u32,
    limit: u32,
}

#[derive(Default)]
pub(crate) struct AreaSpatialStage {
    area: AreaFrameState,
    frame: Option<SpatialFrame>,
    pipelines: Option<(AreaShadowPipelines, u64)>,
    outputs: Option<SpatialOutputs>,
    binds: Option<(SpatialKey, Vec<wgpu::BindGroup>, Vec<wgpu::BindGroup>)>,
}

fn area_pipelines<'a>(
    slot: &'a mut Option<(AreaShadowPipelines, u64)>,
    gpu: &GpuContext,
    cx: &StageCx<'_>,
) -> &'a AreaShadowPipelines {
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

impl StageImpl for AreaSpatialStage {
    fn param_keys(&self) -> &'static [&'static str] {
        SPATIAL_KEYS
    }

    fn check_value(&self, key: &str, value: f64) -> Result<(), String> {
        validate_tunable(crate::tunables(), key, value).map(|_| ())
    }

    fn prepare(
        &mut self,
        cx: &StageCx<'_>,
        ctx: &mut PrepareCtx<'_>,
        published: &mut Vec<(&'static str, ImageDesc)>,
    ) -> Result<(), RaaError> {
        self.frame = None;
        let Some(_) = area_frame(cx, &mut self.area) else {
            return Ok(());
        };
        let stochastic = cx.lit.area_stochastic();
        if stochastic.mode != StochasticMode::Stratified {
            return Ok(());
        }
        let limit = table_limit(&stochastic.table).min(8);
        let groups = cpu::groups_for(limit);
        area_pipelines(&mut self.pipelines, ctx.gpu, cx);
        let changed = self
            .outputs
            .as_ref()
            .is_none_or(|out| out.size != ctx.size || out.groups != groups);
        if changed {
            self.outputs = Some(SpatialOutputs::new(ctx.gpu, ctx.size, groups));
            self.binds = None;
        }
        let Some(outputs) = &self.outputs else {
            return Ok(());
        };
        outputs.publish(cx, ctx, published, "mask");
        self.frame = Some(SpatialFrame { groups, limit });
        Ok(())
    }

    fn record(&mut self, cx: &StageCx<'_>, ctx: &mut RecordCtx<'_>) -> Result<bool, RaaError> {
        let (Some(frame), Some(outputs), Some((pipelines, _))) =
            (self.frame, self.outputs.as_ref(), self.pipelines.as_ref())
        else {
            return Ok(false);
        };
        let (Some(src), Some(depth)) = (
            ctx.resources.input(cx.entry, "src"),
            ctx.resources.input(cx.entry, "linearDepth"),
        ) else {
            return Ok(false);
        };
        if src.desc.dim != ImageDim::D2Array
            || src.desc.format != FORMAT
            || src.desc.width != outputs.size.0
            || src.desc.height != outputs.size.1
            || src.desc.layers < frame.groups
            || depth.desc.dim != ImageDim::D2
            || depth.desc.format != DEPTH_FORMAT
            || depth.desc.width != outputs.size.0
            || depth.desc.height != outputs.size.1
        {
            return Ok(false);
        }
        let views = GBufferViews::resolve(ctx.targets, ctx.resources, cx.entry);
        let key = SpatialKey {
            size: outputs.size,
            groups: frame.groups,
            limit: frame.limit,
            src: src.view.clone(),
            linear_depth: depth.view.clone(),
            normal: views.normal,
        };
        if self
            .binds
            .as_ref()
            .is_none_or(|(stored, _, _)| *stored != key)
        {
            let horizontal = (0..frame.groups)
                .map(|group| {
                    let source = layer_view(src.view.texture(), group);
                    let destination = layer_view(&outputs.intermediate, group);
                    bind_group(
                        ctx.gpu.device(),
                        &pipelines.spatial_layout,
                        &[
                            (1, wgpu::BindingResource::TextureView(&depth.view)),
                            (2, wgpu::BindingResource::TextureView(&key.normal)),
                            (3, wgpu::BindingResource::TextureView(&source)),
                            (4, wgpu::BindingResource::TextureView(&source)),
                            (5, wgpu::BindingResource::TextureView(&destination)),
                            (7, outputs.dispatch_h[group as usize].as_entire_binding()),
                        ],
                    )
                })
                .collect();
            let vertical = (0..2)
                .map(|layer| {
                    let first = (layer * 2).min(frame.groups - 1);
                    let second = (first + 1).min(frame.groups - 1);
                    let source_a = layer_view(&outputs.intermediate, first);
                    let source_b = layer_view(&outputs.intermediate, second);
                    let destination = layer_view(&outputs.mask, layer);
                    bind_group(
                        ctx.gpu.device(),
                        &pipelines.spatial_layout,
                        &[
                            (1, wgpu::BindingResource::TextureView(&depth.view)),
                            (2, wgpu::BindingResource::TextureView(&key.normal)),
                            (3, wgpu::BindingResource::TextureView(&source_a)),
                            (4, wgpu::BindingResource::TextureView(&source_b)),
                            (5, wgpu::BindingResource::TextureView(&destination)),
                            (7, outputs.dispatch_v[layer as usize].as_entire_binding()),
                        ],
                    )
                })
                .collect();
            self.binds = Some((key, horizontal, vertical));
        }
        let Some((_, horizontal, vertical)) = &self.binds else {
            return Ok(false);
        };
        for group in 0..frame.groups {
            let lanes = [group, 1, 0, frame.limit];
            ctx.gpu.queue().write_buffer(
                &outputs.dispatch_h[group as usize],
                0,
                &dispatch_bytes(lanes),
            );
        }
        for layer in 0..2 {
            let lanes = [
                layer,
                0,
                frame.groups.saturating_sub(layer * 2).min(2),
                frame.limit,
            ];
            ctx.gpu.queue().write_buffer(
                &outputs.dispatch_v[layer as usize],
                0,
                &dispatch_bytes(lanes),
            );
        }
        let work = AreaShadowWork {
            trace: &horizontal[0],
            temporal: &[],
            spatial_h: horizontal,
            spatial_v: vertical,
            viewport: outputs.size,
        };
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa-gi.area-spatial"),
                timestamp_writes: None,
            });
        pipelines.record_denoise(&mut pass, &work);
        Ok(true)
    }
}

struct MaskOutput {
    size: (u32, u32),
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl MaskOutput {
    fn new(gpu: &GpuContext, size: (u32, u32)) -> Self {
        let texture = texture(
            gpu,
            "raa-gi.area-mask.output",
            size,
            2,
            wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING,
        );
        let view = texture.create_view(&Default::default());
        Self {
            size,
            texture,
            view,
        }
    }

    fn publish(
        &self,
        cx: &StageCx<'_>,
        ctx: &PrepareCtx<'_>,
        published: &mut Vec<(&'static str, ImageDesc)>,
    ) {
        let desc = ImageDesc::new(ImageDim::D2Array, FORMAT, self.size.0, self.size.1, 2, 1);
        ctx.resources
            .publish(cx.entry, "areaShadow", &self.view, desc);
        published.push(("areaShadow", desc));
    }
}

/// The key of the `area.mask` stage value that tells it its `src` holds the ratio estimator's
/// pairs (S, U, S, U) of `area.trace.raw` or `area.temporal.filtered` instead of visibility. The
/// graph executor writes 1 for a mask linked straight to one of them and 0 otherwise.
pub(crate) const MASK_PAIRS_KEY: &str = "srcPairs";

/// Converts the (S, U) pairs of two history layers into visibility V = S / U for the four slots
/// of one mask layer: the vertical half of `spatial.wgsl` without any filtering. Slots at or past
/// the table limit and texels without weight (`U` zero, sky) write 1.
const RATIO_MASK_WGSL: &str = "
struct RatioParams { slot: u32, limit: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var raa_mask_first: texture_2d<f32>;
@group(0) @binding(1) var raa_mask_second: texture_2d<f32>;
@group(0) @binding(2) var raa_mask_dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> raa_mask_params: RatioParams;

fn raa_mask_visibility(s: f32, u: f32, slot: u32) -> f32 {
    if slot >= raa_mask_params.limit || u <= 1e-6 {
        return 1.0;
    }
    return clamp(s / u, 0.0, 1.0);
}

@compute @workgroup_size(8, 8)
fn raa_gi_area_mask_ratio(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(raa_mask_dst)) {
        return;
    }
    let coord = vec2<i32>(id.xy);
    let first = textureLoad(raa_mask_first, coord, 0);
    let second = textureLoad(raa_mask_second, coord, 0);
    let slot = raa_mask_params.slot;
    textureStore(raa_mask_dst, coord, vec4<f32>(
        raa_mask_visibility(first.x, first.y, slot),
        raa_mask_visibility(first.z, first.w, slot + 1u),
        raa_mask_visibility(second.x, second.y, slot + 2u),
        raa_mask_visibility(second.z, second.w, slot + 3u),
    ));
}
";

/// The compute pipeline of the ratio conversion, its two per-layer uniform buffers and the bind
/// groups of the bound `src`, rebuilt only when the size, the layer count or the view changes.
struct RatioMask {
    pipeline: wgpu::ComputePipeline,
    params: [wgpu::Buffer; 2],
    binds: Option<RatioBinds>,
}

/// The bind groups of one bound `src` and what they were built for.
struct RatioBinds {
    size: (u32, u32),
    layers: u32,
    src: wgpu::TextureView,
    groups: [wgpu::BindGroup; 2],
}

impl RatioMask {
    fn new(gpu: &GpuContext) -> Self {
        let device = gpu.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-gi.area-mask.ratio"),
            source: wgpu::ShaderSource::Wgsl(RATIO_MASK_WGSL.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raa-gi.area-mask.ratio"),
            layout: None,
            module: &module,
            entry_point: Some("raa_gi_area_mask_ratio"),
            compilation_options: Default::default(),
            cache: None,
        });
        let make = || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("raa-gi.area-mask.ratio.params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Self {
            pipeline,
            params: [make(), make()],
            binds: None,
        }
    }

    /// Records one dispatch per mask layer; `src` has `layers` pair layers.
    fn record(
        &mut self,
        ctx: &mut RecordCtx<'_>,
        output: &MaskOutput,
        src: &raa_pipeline::PublishedImage,
        limit: u32,
    ) {
        let layers = src.desc.layers;
        let stale = self.binds.as_ref().is_none_or(|binds| {
            binds.size != output.size || binds.layers != layers || binds.src != src.view
        });
        if stale {
            let layout = self.pipeline.get_bind_group_layout(0);
            let groups: [wgpu::BindGroup; 2] = std::array::from_fn(|layer| {
                let first = (layer as u32 * 2).min(layers - 1);
                let second = (first + 1).min(layers - 1);
                let a = layer_view(src.view.texture(), first);
                let b = layer_view(src.view.texture(), second);
                let dst = layer_view(&output.texture, layer as u32);
                ctx.gpu
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("raa-gi.area-mask.ratio"),
                        layout: &layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&a),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&b),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&dst),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: self.params[layer].as_entire_binding(),
                            },
                        ],
                    })
            });
            self.binds = Some(RatioBinds {
                size: output.size,
                layers,
                src: src.view.clone(),
                groups,
            });
        }
        let Some(RatioBinds { groups, .. }) = &self.binds else {
            return;
        };
        for layer in 0..2u32 {
            let lanes = [layer * 4, limit, 0, 0];
            ctx.gpu
                .queue()
                .write_buffer(&self.params[layer as usize], 0, &dispatch_bytes(lanes));
        }
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa-gi.area-mask.ratio"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&self.pipeline);
        for group in groups {
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(output.size.0.div_ceil(8), output.size.1.div_ceil(8), 1);
        }
    }
}

#[derive(Default)]
pub(crate) struct AreaMaskStage {
    area: AreaFrameState,
    output: Option<MaskOutput>,
    ratio: Option<RatioMask>,
    ready: bool,
    #[cfg(test)]
    forced_limit: Option<u32>,
}

impl StageImpl for AreaMaskStage {
    fn param_keys(&self) -> &'static [&'static str] {
        &[MASK_PAIRS_KEY]
    }

    fn check_value(&self, key: &str, value: f64) -> Result<(), String> {
        if value == 0.0 || value == 1.0 {
            Ok(())
        } else {
            Err(format!("`{key}` is 0 or 1"))
        }
    }

    fn prepare(
        &mut self,
        cx: &StageCx<'_>,
        ctx: &mut PrepareCtx<'_>,
        published: &mut Vec<(&'static str, ImageDesc)>,
    ) -> Result<(), RaaError> {
        self.ready = false;
        let frame = area_frame(cx, &mut self.area);
        let running =
            frame.is_some() && cx.lit.area_stochastic().mode == StochasticMode::Stratified;
        #[cfg(test)]
        let running = running || self.forced_limit.is_some_and(|limit| limit > 0);
        if !running {
            return Ok(());
        }
        if self
            .output
            .as_ref()
            .is_none_or(|output| output.size != ctx.size)
        {
            self.output = Some(MaskOutput::new(ctx.gpu, ctx.size));
        }
        let Some(output) = &self.output else {
            return Ok(());
        };
        let stochastic = cx.lit.area_stochastic();
        ctx.lighting
            .set_area_shadow(output.view.clone(), &stochastic.table)?;
        output.publish(cx, ctx, published);
        self.ready = true;
        Ok(())
    }

    fn record(&mut self, cx: &StageCx<'_>, ctx: &mut RecordCtx<'_>) -> Result<bool, RaaError> {
        if !self.ready {
            return Ok(false);
        }
        let Some(output) = &self.output else {
            return Ok(false);
        };
        let Some(src) = ctx.resources.input(cx.entry, "src") else {
            return Ok(false);
        };
        if cx.value(MASK_PAIRS_KEY, 0.0) > 0.5 {
            if src.desc.dim != ImageDim::D2Array
                || src.desc.format != FORMAT
                || src.desc.width != output.size.0
                || src.desc.height != output.size.1
                || src.desc.layers == 0
            {
                return Ok(false);
            }
            let limit = table_limit(&cx.lit.area_stochastic().table).min(8);
            self.ratio
                .get_or_insert_with(|| RatioMask::new(ctx.gpu))
                .record(ctx, output, &src, limit);
            return Ok(true);
        }
        if src.desc.dim != ImageDim::D2Array
            || src.desc.format != FORMAT
            || src.desc.width != output.size.0
            || src.desc.height != output.size.1
            || src.desc.layers != 2
        {
            return Ok(false);
        }
        ctx.encoder.copy_texture_to_texture(
            src.view.texture().as_image_copy(),
            output.texture.as_image_copy(),
            wgpu::Extent3d {
                width: output.size.0,
                height: output.size.1,
                depth_or_array_layers: 2,
            },
        );
        Ok(true)
    }

    #[cfg(test)]
    fn force_running_frame_for_test(&mut self, limit: u32) {
        self.forced_limit = Some(limit);
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use crate::parts::GiShared;
    use crate::stages::{GiStageKind, install_stage};
    use raa_core::testing::skip;
    use raa_lit::{LitOptions, LitRenderer};
    use raa_pipeline::{FeaturePass, LightingResources, Pipeline, ResourceTable, Targets};

    #[test]
    fn spatial_and_mask_gate_empty_tables_and_mask_publishes_on_forced_frame() {
        let test = "spatial_and_mask_gate_empty_tables_and_mask_publishes_on_forced_frame";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let pipeline = Rc::new(RefCell::new(Pipeline::new(
            &gpu,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        )));
        let spatial =
            install_stage(&gpu, &pipeline, GiStageKind::AreaSpatial, "t").expect("install spatial");
        let mask =
            install_stage(&gpu, &pipeline, GiStageKind::AreaMask, "t").expect("install mask");
        let shared = GiShared::new(&gpu);
        let frame = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-gi.area-mask.test.frame"),
            size: 432,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        shared.set_frame(Some(frame));
        let lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default())
            .expect("lit renderer")
            .handle();
        assert_eq!(table_limit(&lit.area_stochastic().table), 0);
        spatial.attach_shared(shared.clone(), lit.clone());
        mask.attach_shared(shared, lit);
        let mut passes: Vec<Box<dyn FeaturePass>> = {
            let mut pipeline = pipeline.borrow_mut();
            vec![
                pipeline.remove(&spatial.entry()).expect("spatial pass"),
                pipeline.remove(&mask.entry()).expect("mask pass"),
            ]
        };
        let table = ResourceTable::new();
        let targets = Targets::new(&gpu, (8, 8));
        let mut lighting = LightingResources::new(&gpu);
        for pass in &mut passes {
            let mut ctx = PrepareCtx {
                gpu: &gpu,
                size: (8, 8),
                frame: 0,
                time_s: 0.0,
                dt_s: 0.0,
                targets: &targets,
                lighting: &mut lighting,
                resources: &table,
            };
            pass.resize(&gpu, &targets);
            pass.prepare(&mut ctx).expect("prepare without area light");
        }
        assert!(spatial.published().is_empty());
        assert!(mask.published().is_empty());
        mask.force_running_frame_for_test(2);
        let mut ctx = PrepareCtx {
            gpu: &gpu,
            size: (8, 8),
            frame: 1,
            time_s: 0.0,
            dt_s: 0.0,
            targets: &targets,
            lighting: &mut lighting,
            resources: &table,
        };
        passes[1].prepare(&mut ctx).expect("forced mask prepare");
        assert_eq!(mask.published().len(), 1);
        let (port, desc) = mask.published()[0];
        assert_eq!(port, "areaShadow");
        assert_eq!(desc.dim, ImageDim::D2Array);
        assert_eq!(desc.format, FORMAT);
        assert_eq!((desc.width, desc.height, desc.layers), (8, 8, 2));
        assert!(table.input(&mask.entry(), "src").is_none());
        assert!(!spatial.ran() && !mask.ran());
    }
}
